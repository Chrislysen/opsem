"""
lme_interaction.py — Cross-corpus replication of the interaction-function
finding on LongMemEval-S, on field-standard retrieval metrics.

Same controlled comparison as tune13 (LoCoMo): the retrieval unit is held
fixed at *session*; only the query<->session scoring operator varies over
the SAME turn vectors:
    early  = cos(q, L2norm(mean_t turn_t))      (mean-pool / one session vec)
    late   = max_t cos(q, turn_t)               (max-sim)
    top3   = mean of top-3 cos
plus BM25 alone and BM25 (+) late-interaction fusion (CV alpha).

Reports Recall@1/3/5, MRR, NDCG@5 (the metrics the LongMemEval paper uses
for its retrieval-quality analysis), so the numbers are directly
comparable to published baselines. Bootstrap over questions.

MEMORY NOTE: LongMemEval-S haystacks are huge (~45 sessions, ~115k tokens).
We therefore compute per-session scores INLINE and discard turn embeddings
immediately (an earlier version that cached all embeddings in RAM OOM-ed).
Only the tiny per-session score vectors are cached, to
results/lme_signals__<slug>__<subset>.npz, so re-running metrics/bootstrap
is instant; embedding still happens once per encoder.

Eval set: challenge rows only (haystack has more sessions than the answer
set) = the 500 canonical LongMemEval-S retrieval questions with full
haystacks. The HF mirror also stores a degenerate "oracle" row per
question (haystack == gold); --all includes those and is NOT comparable to
the published retrieval numbers.

Usage:
  python lme_interaction.py --model BAAI/bge-base-en-v1.5
  python lme_interaction.py --model intfloat/e5-large-v2 --qpref "query: " --ppref "passage: "

Output: results/lme-interaction-<ts>/{lme.json, lme.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
RES = _HERE / "results"

# reuse the exact BM25 + tokenizer from lme_maxsim for continuity
from lme_maxsim import BM25, _sess_text, _turn_texts, _z, _rrf  # noqa: E402

METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]


def metrics_from_scores(score, gold, ks=(1, 3, 5)):
    out = {}
    if score.size == 0 or not gold:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out["mrr"] = 0.0
        out["ndcg@5"] = 0.0
        return out
    order = np.argsort(-score, kind="stable")
    g = len(gold)
    for k in ks:
        out[f"recall@{k}"] = len([i for i in order[:k] if int(i) in gold]) / g
    rr = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(idx) in gold:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    dcg = sum(1.0 / np.log2(r + 1.0)
              for r, idx in enumerate(order[:5], start=1) if int(idx) in gold)
    ideal = sum(1.0 / np.log2(r + 1.0) for r in range(1, min(5, g) + 1))
    out["ndcg@5"] = float(dcg / ideal) if ideal > 0 else 0.0
    return out


def _slug(model_id):
    return model_id.replace("/", "_").replace(".", "-")


def compute_signals(exs, model_id, qpref, ppref, sig_path):
    """Per example, compute BM25 + per-session early/late/top3 dense scores
    INLINE (turn embeddings discarded immediately). Cache the small score
    vectors (object arrays) to disk. Returns dict of lists."""
    if sig_path.exists():
        print(f"  loading cached signals {sig_path.name}", flush=True)
        z = np.load(sig_path, allow_pickle=True)
        return {"bm": list(z["bm"]), "mean": list(z["mean"]),
                "max": list(z["max"]), "top3": list(z["top3"]),
                "gold": [set(g.tolist()) for g in z["gold"]],
                "qtype": list(z["qtype"]),
                "goldturns": list(z["goldturns"]) if "goldturns" in z.files
                else [0] * len(z["bm"])}

    from sentence_transformers import SentenceTransformer
    print(f"  encoding with {model_id} (qpref={qpref!r} ppref={ppref!r})",
          flush=True)
    model = SentenceTransformer(model_id)
    q_emb = model.encode([qpref + e["question"] for e in exs],
                         convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=64, show_progress_bar=True).astype(np.float32)

    bm_l, mean_l, max_l, top3_l, gold_l, qtype_l = [], [], [], [], [], []
    goldturns_l = []
    n = len(exs)
    for i, e in enumerate(exs):
        if i % 25 == 0:
            print(f"  scoring {i}/{n}", flush=True)
        sids = e["haystack_session_ids"]
        gold = set(e["answer_session_ids"])
        gold_idx = {j for j, s in enumerate(sids) if s in gold}
        gold_l.append(gold_idx)
        qtype_l.append(e.get("question_type", "?"))
        # size (turns) of the largest gold session — for the dilution test
        gt = max((len(e["haystack_sessions"][j]) for j in gold_idx), default=0)
        goldturns_l.append(gt)
        bm_l.append(BM25([_sess_text(s) for s in e["haystack_sessions"]])
                    .score(e["question"]).astype(np.float32))
        qv = q_emb[i]
        S = len(e["haystack_sessions"])
        mean_s = np.full(S, -9.0, np.float32)
        max_s = np.full(S, -9.0, np.float32)
        top3_s = np.full(S, -9.0, np.float32)
        # batch-encode all turns of THIS example, then discard
        all_turns, spans = [], []
        for sess in e["haystack_sessions"]:
            tt = [ppref + t for t in _turn_texts(sess)]
            spans.append((len(all_turns), len(all_turns) + len(tt)))
            all_turns.extend(tt)
        if all_turns:
            temb = model.encode(all_turns, convert_to_numpy=True,
                                normalize_embeddings=True, batch_size=128,
                                show_progress_bar=False).astype(np.float32)
            sims_all = temb @ qv
            for j, (a, b) in enumerate(spans):
                if b > a:
                    sims = sims_all[a:b]
                    max_s[j] = float(sims.max())
                    k = min(3, b - a)
                    top3_s[j] = float(np.sort(sims)[-k:].mean())
                    mv = temb[a:b].mean(0)
                    nr = float(np.linalg.norm(mv))
                    mean_s[j] = float(mv @ qv / nr) if nr > 1e-9 else -9.0
            del temb, sims_all  # free immediately
        mean_l.append(mean_s)
        max_l.append(max_s)
        top3_l.append(top3_s)

    np.savez_compressed(
        sig_path,
        bm=np.array(bm_l, dtype=object), mean=np.array(mean_l, dtype=object),
        max=np.array(max_l, dtype=object), top3=np.array(top3_l, dtype=object),
        gold=np.array([np.array(sorted(g)) for g in gold_l], dtype=object),
        qtype=np.array(qtype_l), goldturns=np.array(goldturns_l))
    print(f"  cached signals -> {sig_path.name}", flush=True)
    return {"bm": bm_l, "mean": mean_l, "max": max_l, "top3": top3_l,
            "gold": gold_l, "qtype": qtype_l, "goldturns": goldturns_l}


def run(out_dir, challenge_only, n_boot, model_id, qpref, ppref,
        sample=0, seed=0):
    from datasets import load_dataset
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"lme-interaction-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading LongMemEval-S...", flush=True)
    ds = load_dataset("LIXINYI33/longmemeval-s", split="train")
    exs = []
    for ex in ds:
        if challenge_only and not (len(ex["haystack_session_ids"])
                                   > len(ex["answer_session_ids"])):
            continue
        if not ex["haystack_sessions"] or not ex["answer_session_ids"]:
            continue
        exs.append(ex)
    subset = "chal" if challenge_only else "full"
    if sample and sample < len(exs):
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(exs), size=sample, replace=False).tolist())
        exs = [exs[k] for k in pick]
        subset += f"_s{sample}seed{seed}"
    print(f"  {len(exs)} examples (challenge_only={challenge_only}, "
          f"sample={sample})", flush=True)
    sig_path = RES / f"lme_signals__{_slug(model_id)}__{subset}.npz"
    sig = compute_signals(exs, model_id, qpref, ppref, sig_path)
    bm_scores = sig["bm"]
    golds = sig["gold"]
    dense = {"mean(early)": sig["mean"], "max-sim(late)": sig["max"],
             "top3(late)": sig["top3"]}
    n = len(exs)

    def eval_list(score_list):
        agg = {m: 0.0 for m in METRIC_KEYS}
        r1 = np.zeros(n, np.int8)
        for i in range(n):
            mm = metrics_from_scores(np.asarray(score_list[i], float), golds[i])
            for k in METRIC_KEYS:
                agg[k] += mm[k]
            r1[i] = int(mm["recall@1"] > 0)
        for k in METRIC_KEYS:
            agg[k] /= max(n, 1)
        return agg, r1

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    def cv_fuse(dense_list, seed=0):
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 5, size=n)
        cache = {a: [a * _z(bm_scores[i]) + (1 - a) * _z(dense_list[i])
                     for i in range(n)] for a in alphas}
        ah = {a: np.array([int(int(np.argmax(cache[a][i])) in golds[i])
                           for i in range(n)], np.int8) for a in alphas}
        out = [None] * n
        chosen = []
        for f in range(5):
            te = fold == f
            tr = ~te
            ba = max(alphas, key=lambda a: ah[a][tr].mean())
            chosen.append(float(ba))
            for i in np.where(te)[0]:
                out[i] = cache[ba][i]
        return out, chosen

    results, r1_arrays = {}, {}
    m, r1 = eval_list(bm_scores)
    results["BM25"], r1_arrays["BM25"] = m, r1
    for p in dense:
        m, r1 = eval_list(dense[p])
        results[f"dense:{p}"], r1_arrays[f"dense:{p}"] = m, r1
    fl, ch_late = cv_fuse(dense["max-sim(late)"])
    fe, _ = cv_fuse(dense["mean(early)"])
    m, r1 = eval_list(fl)
    results["fusion:late"], r1_arrays["fusion:late"] = m, r1
    m, r1 = eval_list(fe)
    results["fusion:early"], r1_arrays["fusion:early"] = m, r1
    rrf = [_rrf(bm_scores[i]) + _rrf(dense["max-sim(late)"][i]) for i in range(n)]
    m, r1 = eval_list(rrf)
    results["RRF(BM25,late)"], r1_arrays["RRF(BM25,late)"] = m, r1

    def boot(a, b, seed=1):
        rng = np.random.default_rng(seed)
        d = np.empty(n_boot)
        for bi in range(n_boot):
            idx = rng.integers(0, n, n)
            d[bi] = a[idx].mean() - b[idx].mean()
        return {"diff": float(a.mean() - b.mean()),
                "lo": float(np.quantile(d, 0.025)),
                "hi": float(np.quantile(d, 0.975)),
                "p_one_sided": float(np.mean(d <= 0))}

    sig_tests = {
        "dense late-vs-early": boot(r1_arrays["dense:max-sim(late)"],
                                    r1_arrays["dense:mean(early)"], 11),
        "fusion late-vs-early": boot(r1_arrays["fusion:late"],
                                     r1_arrays["fusion:early"], 12),
        "fusion-late vs BM25": boot(r1_arrays["fusion:late"],
                                    r1_arrays["BM25"], 13),
        "fusion-late vs RRF": boot(r1_arrays["fusion:late"],
                                   r1_arrays["RRF(BM25,late)"], 14),
    }

    # dilution-vs-interaction test: late-early R@1 gap by gold-session size
    gt = np.array(sig["goldturns"])
    de = r1_arrays["dense:mean(early)"]
    dl = r1_arrays["dense:max-sim(late)"]
    length_buckets = []
    for lo, hi, lab in [(1, 1, '1'), (2, 3, '2-3'), (4, 7, '4-7'),
                        (8, 15, '8-15'), (16, 10**9, '16+')]:
        msk = (gt >= lo) & (gt <= hi)
        if msk.sum() == 0:
            continue
        length_buckets.append({"bucket": lab, "n": int(msk.sum()),
                               "early": float(de[msk].mean()),
                               "late": float(dl[msk].mean()),
                               "gap_pp": float((dl[msk].mean()
                                                - de[msk].mean()) * 100)})

    print("\n  method                     R@1     R@3     R@5     MRR    NDCG@5",
          flush=True)
    for k, m in results.items():
        print(f"  {k:24s} " + " ".join(f"{m[mk]:.4f}" for mk in METRIC_KEYS),
              flush=True)
    print("", flush=True)
    for k, s in sig_tests.items():
        print(f"  {k:26s} d={s['diff']*100:+.2f}pp "
              f"CI=[{s['lo']*100:+.2f},{s['hi']*100:+.2f}] "
              f"p={s['p_one_sided']:.4f}", flush=True)

    print("\n  late-early gap by gold-session #turns (dilution test):",
          flush=True)
    for b in length_buckets:
        print(f"    {b['bucket']:>6} n={b['n']:>4} early={b['early']:.3f} "
              f"late={b['late']:.3f} gap={b['gap_pp']:+.2f}pp", flush=True)

    summary = {"ts": ts, "model": model_id, "n_eval": n,
               "challenge_only": challenge_only, "n_boot": n_boot,
               "fusion_late_alpha_folds": ch_late,
               "metric_keys": METRIC_KEYS, "results": results,
               "significance": sig_tests, "length_buckets": length_buckets}
    (run_dir / "lme.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    md = [f"## LongMemEval-S: interaction-function control ({model_id})", "",
          f"n_eval = {n} (challenge_only={challenge_only}). Unit fixed at "
          f"session; only the scoring operator varies. Bootstrap over "
          f"questions (n_boot={n_boot}).", "",
          "| method | R@1 | R@3 | R@5 | MRR | NDCG@5 |", "|---|---|---|---|---|---|"]
    for k, m in results.items():
        md.append(f"| {k} | " + " | ".join(f"{m[mk]:.4f}" for mk in METRIC_KEYS)
                  + " |")
    md += ["", "### Significance (R@1, bootstrap over questions)", "",
           "| comparison | Δ pp | 95% CI | p |", "|---|---|---|---|"]
    for k, s in sig_tests.items():
        md.append(f"| {k} | {s['diff']*100:+.2f} | "
                  f"[{s['lo']*100:+.2f}, {s['hi']*100:+.2f}] | "
                  f"{s['p_one_sided']:.4f} |")
    md += ["", "### Dilution test: late−early R@1 gap by gold-session #turns",
           "", "| gold-session turns | n | early | late | gap pp |",
           "|---|---|---|---|---|"]
    for b in length_buckets:
        md.append(f"| {b['bucket']} | {b['n']} | {b['early']:.3f} | "
                  f"{b['late']:.3f} | {b['gap_pp']:+.2f} |")
    md += ["", "Raw: `lme.json`.", ""]
    (run_dir / "lme.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir / 'lme.md'}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES))
    ap.add_argument("--all", action="store_true", help="full set, not just challenge")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--qpref", default="")
    ap.add_argument("--ppref", default="")
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample this many challenge questions (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), challenge_only=not args.all, n_boot=args.n_boot,
        model_id=args.model, qpref=args.qpref, ppref=args.ppref,
        sample=args.sample, seed=args.seed)

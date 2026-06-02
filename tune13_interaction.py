"""
tune13_interaction.py — The lever is the INTERACTION FUNCTION, not the unit.

Motivation
----------
The LoCoMo/LongMemEval literature conflates two axes under the single word
"granularity":

  Axis A — retrieval UNIT: what text span is the atomic retrievable item
           (turn vs session vs segment). Direction is *disputed*: SeCom
           reports turn > session on LoCoMo; SGMem / RMM / LongMemEval
           report session >= turn. This axis is prior art either way.

  Axis B — query<->session INTERACTION function: how a session's score is
           computed from its turn vectors.
             * early interaction = pool turns into ONE session vector,
               score = cos(q, mean_t turn_t)            [the prior "session-mean"]
             * late interaction  = score = max_t cos(q, turn_t)   [max-sim]
             * smooth late        = (1/beta) logsumexp(beta * cos)  [soft max]
             * top-k late         = mean of top-k cos

This script ISOLATES Axis B. It holds the retrieval unit fixed at *session*
(we always return a whole session id, exactly like session-level RAG) and
varies ONLY how the session is scored from the SAME cached turn vectors.
Identical embeddings; the only thing that changes is the pooling operator.

If max-sim (late) crushes mean-pool (early) under this control, then the
"granularity" story in the literature is mis-attributed: the lever is not
the unit you retrieve, it is whether you let the query interact with the
single answer-bearing turn (late) or dilute it into a session average
(early). That reconciles the contradiction: methods that feed whole
session *text* to an LLM reader never pay the pooling tax, so "session is
fine"; methods that *embed* a pooled session do, so "session dense is
catastrophic". Both are true; the hidden variable is the interaction
function.

Metrics are the field-standard retrieval set (Recall@1/3/5, MRR, NDCG@5),
not just Hit@1, so the result is comparable to the benchmark papers.

CPU only, runs entirely from cached embeddings (no model load):
  results/locomo_turn_cache.npz + locomo_bge_cache.npz  (bge-base)
  results/turns__<slug>.npz + q__<slug>.npy             (5 more encoders)

Output: results/tune13-interaction-<ts>/{tune13.json, tune13.md}
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
import locomo  # noqa: E402
from tune7_bm25 import ConvStats, score_classic, _tokenize_text  # noqa: E402

LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
RES = _HERE / "results"

ENCODERS = [
    ("bge-base", None, None),  # special-cased below
    ("e5-base-v2", "intfloat_e5-base-v2", 768),
    ("gte-base", "thenlper_gte-base", 768),
    ("bge-large", "BAAI_bge-large-en-v1-5", 1024),
    ("mxbai-large", "mixedbread-ai_mxbai-embed-large-v1", 1024),
    ("e5-large-v2", "intfloat_e5-large-v2", 1024),
]


# ----------------------------- poolers ------------------------------------
def pool_mean(sims, turn_mat, qv):
    """Early interaction: cos(q, L2norm(mean of turn vectors))."""
    m = turn_mat.mean(axis=0)
    nrm = np.linalg.norm(m)
    return float(m @ qv / nrm) if nrm > 1e-9 else -9.0


def pool_max(sims, turn_mat, qv):
    """Late interaction (max-sim)."""
    return float(sims.max())


def pool_topk(sims, turn_mat, qv, k=3):
    k = min(k, sims.shape[0])
    return float(np.sort(sims)[-k:].mean())


def pool_lse(sims, turn_mat, qv, beta=10.0):
    """Smooth late interaction: (1/beta) logsumexp(beta*cos). beta->inf == max."""
    s = beta * sims
    m = s.max()
    return float((m + np.log(np.exp(s - m).sum())) / beta)


POOLERS = {
    "mean(early)": pool_mean,
    "max-sim(late)": pool_max,
    "top3(late)": pool_topk,
    "lse10(late)": pool_lse,
}


# ----------------------------- metrics ------------------------------------
def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def metrics_from_scores(score, gold, ks=(1, 3, 5)):
    """Return dict of hit@1, recall@k, mrr, ndcg@5 for one example.

    gold is a set of gold session indices. Relevance is binary.
    """
    out = {}
    if score.size == 0 or not gold:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out["hit@1"] = 0.0
        out["mrr"] = 0.0
        out["ndcg@5"] = 0.0
        return out
    order = np.argsort(-score, kind="stable")
    out["hit@1"] = float(int(order[0]) in gold)
    g = len(gold)
    for k in ks:
        topk = order[:k]
        out[f"recall@{k}"] = len([i for i in topk if int(i) in gold]) / g
    # MRR (first relevant)
    rr = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(idx) in gold:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    # NDCG@5 binary
    k = 5
    dcg = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        if int(idx) in gold:
            dcg += 1.0 / np.log2(rank + 1.0)
    ideal = sum(1.0 / np.log2(r + 1.0) for r in range(1, min(k, g) + 1))
    out["ndcg@5"] = float(dcg / ideal) if ideal > 0 else 0.0
    return out


METRIC_KEYS = ["hit@1", "recall@3", "recall@5", "mrr", "ndcg@5"]


# --------------------------- data assembly --------------------------------
def load_turns(slug):
    return np.load(RES / f"turns__{slug}.npz", allow_pickle=True)


def build_signals():
    """For every (example, encoder, pooler) compute the per-session score
    vector. Also build BM25 once. Returns rows + structures for eval."""
    exs = [e for e in locomo.iter_examples(LOCOMO) if e["gold_session_ids"]]
    n = len(exs)
    conv_of = np.array([e["conv_idx"] for e in exs])
    convs = sorted(set(conv_of.tolist()))

    # BM25 (shared) + gold idx
    cs = {}
    bm_scores, golds, cats = [], [], []
    for e in exs:
        ci = e["conv_idx"]
        if ci not in cs:
            cs[ci] = ConvStats([_tokenize_text(s) for s in e["haystack_sessions"]])
        bm_scores.append(score_classic(cs[ci],
                         set(_tokenize_text(e["question"])), 1.5, 0.75))
        sid2 = {s: j for j, s in enumerate(e["session_ids"])}
        golds.append({sid2[g] for g in e["gold_session_ids"] if g in sid2})
        cats.append(e["category"])

    # per-encoder turn caches + query embs
    enc_data = {}
    for name, slug, _dim in ENCODERS:
        if name == "bge-base":
            tc = np.load(RES / "locomo_turn_cache.npz", allow_pickle=True)
            q = np.load(RES / "locomo_bge_cache.npz", allow_pickle=True)["q_emb"]
        else:
            tnpz = RES / f"turns__{slug}.npz"
            qnpy = RES / f"q__{slug}.npy"
            if not (tnpz.exists() and qnpy.exists()):
                continue
            tc = np.load(tnpz, allow_pickle=True)
            q = np.load(qnpy)
        enc_data[name] = (tc, q)

    # For each encoder + pooler, build score vector per example.
    # scores[enc][pooler] = list of np arrays (len = n_sessions of that ex)
    scores = {enc: {p: [None] * n for p in POOLERS} for enc in enc_data}
    for i, e in enumerate(exs):
        ci = e["conv_idx"]
        sids = e["session_ids"]
        S = len(sids)
        for enc, (tc, q) in enc_data.items():
            qv = q[i]
            # gather per-session sims arrays
            per_pool = {p: np.full(S, -9.0) for p in POOLERS}
            for j, sid in enumerate(sids):
                key = f"turns_{ci}_{int(sid[1:])}"
                if key not in tc.files:
                    continue
                tm = tc[key]
                sims = tm @ qv
                for pname, fn in POOLERS.items():
                    per_pool[pname][j] = fn(sims, tm, qv)
            for pname in POOLERS:
                scores[enc][pname][i] = per_pool[pname]

    return {"n": n, "conv_of": conv_of, "convs": convs, "bm": bm_scores,
            "golds": golds, "cats": cats, "encoders": list(enc_data),
            "scores": scores}


# ------------------------------- eval -------------------------------------
def eval_method(score_list, golds):
    """Mean metrics over all examples for a list of per-example score vecs.
    Also returns the per-example hit@1 array for bootstrapping."""
    n = len(golds)
    agg = {m: 0.0 for m in METRIC_KEYS}
    hit1 = np.zeros(n, np.int8)
    for i in range(n):
        m = metrics_from_scores(np.asarray(score_list[i], float), golds[i])
        for k in METRIC_KEYS:
            agg[k] += m[k]
        hit1[i] = int(m["hit@1"])
    for k in METRIC_KEYS:
        agg[k] /= max(n, 1)
    return agg, hit1


def fuse_loco(bm, dense, golds, conv_of, convs, alphas):
    """LOCO-CV alpha selection (maximize hit@1 on train folds), return
    per-example fused score vectors + chosen alphas."""
    n = len(golds)
    # precompute hit@1 per alpha for fold selection
    ah_hit = {}
    fused_cache = {}
    for a in alphas:
        fl = [a * _z(bm[i]) + (1 - a) * _z(dense[i]) for i in range(n)]
        fused_cache[a] = fl
        ah_hit[a] = np.array([int(int(np.argmax(fl[i])) in golds[i])
                              for i in range(n)], np.int8)
    chosen = {}
    out = [None] * n
    for c in convs:
        te = conv_of == c
        tr = ~te
        ba = max(alphas, key=lambda a: ah_hit[a][tr].mean())
        chosen[c] = float(ba)
        for i in np.where(te)[0]:
            out[i] = fused_cache[ba][i]
    return out, chosen


def cboot_hit(a, b, conv_of, convs, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx_by_conv = {c: np.where(conv_of == c)[0] for c in convs}
    cl = np.array(convs)
    d = np.empty(n_boot)
    for bi in range(n_boot):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ia = np.concatenate([idx_by_conv[c] for c in pick])
        d[bi] = a[ia].mean() - b[ia].mean()
    return {"diff": float(a.mean() - b.mean()),
            "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)),
            "p_one_sided": float(np.mean(d <= 0))}


def run(out_dir, n_boot=4000):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune13-interaction-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("building signals from cache (no model load)...", flush=True)
    D = build_signals()
    n, golds, conv_of, convs = D["n"], D["golds"], D["conv_of"], D["convs"]
    bm = D["bm"]
    print(f"  {n} examples, {len(convs)} conversations, "
          f"encoders={D['encoders']}", flush=True)

    bm_metrics, bm_hit1 = eval_method(bm, golds)
    print(f"  BM25  " + "  ".join(f"{k}={bm_metrics[k]:.4f}"
                                   for k in METRIC_KEYS), flush=True)

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    # ---- core finding: dense-alone, every pooler x encoder ----
    dense_table = []   # rows for dense-alone
    fusion_table = []  # rows for BM25-fused
    contrasts = {}     # max-sim vs mean, per encoder (dense + fusion)

    for enc in D["encoders"]:
        for pname in POOLERS:
            dense = D["scores"][enc][pname]
            dm, dhit1 = eval_method(dense, golds)
            dense_table.append({"encoder": enc, "pooler": pname, **dm})

            fused, chosen = fuse_loco(bm, dense, golds, conv_of, convs, alphas)
            fm, fhit1 = eval_method(fused, golds)
            fusion_table.append({"encoder": enc, "pooler": pname,
                                 "loco_alpha_mode": float(np.median(
                                     list(chosen.values()))),
                                 **fm})
            # stash hit1 arrays for the key contrast
            if pname in ("max-sim(late)", "mean(early)"):
                contrasts.setdefault(enc, {})[f"dense_{pname}"] = dhit1
                contrasts.setdefault(enc, {})[f"fusion_{pname}"] = fhit1

    # significance: late vs early (dense) and fusion-late vs fusion-early
    sig = {}
    for enc in D["encoders"]:
        c = contrasts[enc]
        sig[enc] = {
            "dense_late_vs_early": cboot_hit(
                c["dense_max-sim(late)"], c["dense_mean(early)"],
                conv_of, convs, n_boot, seed=abs(hash("d" + enc)) % 99999),
            "fusion_late_vs_early": cboot_hit(
                c["fusion_max-sim(late)"], c["fusion_mean(early)"],
                conv_of, convs, n_boot, seed=abs(hash("f" + enc)) % 99999),
            "fusion_late_vs_bm25": cboot_hit(
                c["fusion_max-sim(late)"], bm_hit1,
                conv_of, convs, n_boot, seed=abs(hash("b" + enc)) % 99999),
        }
        s = sig[enc]
        print(f"  [{enc}] dense late-early "
              f"d={s['dense_late_vs_early']['diff']*100:+.2f}pp "
              f"p={s['dense_late_vs_early']['p_one_sided']:.4f} | "
              f"fusion late-early "
              f"d={s['fusion_late_vs_early']['diff']*100:+.2f}pp "
              f"p={s['fusion_late_vs_early']['p_one_sided']:.4f}", flush=True)

    summary = {
        "ts": ts, "n_eval": n, "n_conversations": len(convs),
        "n_boot": n_boot, "encoders": D["encoders"],
        "metric_keys": METRIC_KEYS,
        "bm25": bm_metrics,
        "dense_table": dense_table,
        "fusion_table": fusion_table,
        "significance": sig,
    }
    (run_dir / "tune13.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- markdown report ----
    md = ["## tune13: the lever is the INTERACTION FUNCTION, not the unit (LoCoMo)",
          "",
          f"n_eval = {n} across {len(convs)} conversations. Retrieval unit is "
          f"held fixed at *session*; only the query<->session scoring operator "
          f"varies, over the SAME cached turn vectors. Field-standard metrics; "
          f"conversation-cluster bootstrap (n_boot={n_boot}). CPU, no training.",
          "",
          f"BM25 baseline: " + ", ".join(f"{k} {bm_metrics[k]:.4f}"
                                         for k in METRIC_KEYS),
          "",
          "### Dense-alone: early (mean-pool) vs late (max-sim) interaction",
          "",
          "| encoder | pooler | Hit@1 | R@3 | R@5 | MRR | NDCG@5 |",
          "|---|---|---|---|---|---|---|"]
    for r in dense_table:
        md.append(f"| {r['encoder']} | {r['pooler']} | {r['hit@1']:.4f} | "
                  f"{r['recall@3']:.4f} | {r['recall@5']:.4f} | "
                  f"{r['mrr']:.4f} | {r['ndcg@5']:.4f} |")
    md += ["",
           "### BM25-fused (LOCO-CV alpha): same pooler sweep",
           "",
           "| encoder | pooler | Hit@1 | R@3 | R@5 | MRR | NDCG@5 |",
           "|---|---|---|---|---|---|---|"]
    for r in fusion_table:
        md.append(f"| {r['encoder']} | {r['pooler']} | {r['hit@1']:.4f} | "
                  f"{r['recall@3']:.4f} | {r['recall@5']:.4f} | "
                  f"{r['mrr']:.4f} | {r['ndcg@5']:.4f} |")
    md += ["",
           "### The control contrast: late minus early interaction",
           "(identical turn vectors; only the pooling operator differs)",
           "",
           "| encoder | dense Δ(late-early) pp | p | fusion Δ(late-early) pp | p | fusion-late vs BM25 pp | p |",
           "|---|---|---|---|---|---|---|"]
    for enc in D["encoders"]:
        s = sig[enc]
        md.append(
            f"| {enc} | {s['dense_late_vs_early']['diff']*100:+.2f} | "
            f"{s['dense_late_vs_early']['p_one_sided']:.4f} | "
            f"{s['fusion_late_vs_early']['diff']*100:+.2f} | "
            f"{s['fusion_late_vs_early']['p_one_sided']:.4f} | "
            f"{s['fusion_late_vs_bm25']['diff']*100:+.2f} | "
            f"{s['fusion_late_vs_bm25']['p_one_sided']:.4f} |")
    md += ["", "Raw: `tune13.json`.", ""]
    (run_dir / "tune13.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir / 'tune13.md'}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    run(Path(args.out), n_boot=args.n_boot)

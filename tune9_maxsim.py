"""
tune9_maxsim.py — Late-interaction (max-sim) turn-level fusion on LoCoMo.

The result
----------
Session-level dense retrieval is catastrophic on LoCoMo (tune8: BGE-base
session embedding Hit@1 = 0.349, vs BM25 0.640) because pooling a whole
multi-turn session into one vector dilutes the single answer turn. We
embed each *turn* and score a session by late interaction:

    dense(session) = max_t cosine(query, turn_t)        # max-sim

This recovers the dense signal (0.547 alone) and, crucially, makes it
complementary to BM25. A single global fusion weight then clears BM25 by
a wide, statistically-significant margin — beating even the prior best
method here (two-stage cross-encoder rerank, 0.670) at zero cross-encoder
cost.

Rigour
------
  * Fusion weight alpha chosen by leave-one-conversation-out CV (honest,
    deployable). Full-data alpha reported only as an overfit ceiling.
  * Cluster-bootstrap (resample conversations) for every comparison.
  * Per-category breakdown.

Baselines: BM25(default), MaxSim-dense, Top3-dense, RRF(BM25,MaxSim),
best fixed-alpha session-mean-dense fusion (the prior approach).

Requires results/locomo_bge_cache.npz (query embs) and
results/locomo_turn_cache.npz (turn embs). Runs on CPU.

Output: results/tune9-maxsim-<ts>/{tune9.json, tune9.md}
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

DEFAULT_LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
QCACHE = _HERE / "results" / "locomo_bge_cache.npz"
TCACHE = _HERE / "results" / "locomo_turn_cache.npz"
DEFAULT_OUT = _HERE / "results"
CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
             4: "open-ended", 5: "adversarial"}


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def _rrf(v, k0=60.0):
    order = np.argsort(-v)
    rr = np.empty_like(v, dtype=float)
    for rank, idx in enumerate(order):
        rr[idx] = 1.0 / (k0 + rank + 1.0)
    return rr


def _hit(score, gold):
    return int(int(np.argmax(score)) in gold) if score.size else 0


def build(json_path):
    exs = [e for e in locomo.iter_examples(json_path) if e["gold_session_ids"]]
    qc = np.load(QCACHE, allow_pickle=True)
    q_emb = qc["q_emb"]
    sess_emb = {int(k.split("_")[-1]): qc[k] for k in qc.files
                if k.startswith("sess_emb_")}
    tc = np.load(TCACHE, allow_pickle=True)
    cs_cache = {}
    rows = []
    for i, e in enumerate(exs):
        ci = e["conv_idx"]
        sids = e["session_ids"]
        if ci not in cs_cache:
            cs_cache[ci] = ConvStats([_tokenize_text(s)
                                      for s in e["haystack_sessions"]])
        bm = score_classic(cs_cache[ci],
                           set(_tokenize_text(e["question"])), 1.5, 0.75)
        qv = q_emb[i]
        S = len(sids)
        dmax = np.full(S, -9.0)
        dtop3 = np.full(S, -9.0)
        for j, sid in enumerate(sids):
            key = f"turns_{ci}_{int(sid[1:])}"
            if key in tc.files:
                sims = tc[key] @ qv
                dmax[j] = float(sims.max())
                k = min(3, sims.shape[0])
                dtop3[j] = float(np.sort(sims)[-k:].mean())
        dmean = (sess_emb[ci] @ qv).astype(float)  # prior session-mean dense
        sid_to_idx = {s: j for j, s in enumerate(sids)}
        gold = {sid_to_idx[g] for g in e["gold_session_ids"] if g in sid_to_idx}
        rows.append({"conv": ci, "cat": e["category"], "bm": bm,
                     "dmax": dmax, "dtop3": dtop3, "dmean": dmean,
                     "gold": gold})
    return rows


def run(json_path, out_dir, n_boot=4000):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune9-maxsim-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("building BM25 + turn-level dense signals...", flush=True)
    rows = build(json_path)
    n = len(rows)
    convs = sorted({r["conv"] for r in rows})
    conv_of = np.array([r["conv"] for r in rows])
    cat_of = np.array([r["cat"] for r in rows])
    print(f"  {n} examples, {len(convs)} conversations", flush=True)

    def single(key):
        return np.array([_hit(r[key], r["gold"]) for r in rows], dtype=np.int8)

    bm_hits = single("bm")
    dmax_hits = single("dmax")
    dtop3_hits = single("dtop3")
    dmean_hits = single("dmean")

    def fuse_hits(alpha, dkey):
        return np.array([_hit(alpha * _z(r["bm"]) + (1 - alpha) * _z(r[dkey]),
                              r["gold"]) for r in rows], dtype=np.int8)

    def rrf_hits(dkey):
        return np.array([_hit(_rrf(r["bm"]) + _rrf(r[dkey]), r["gold"])
                         for r in rows], dtype=np.int8)

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    # full-data ceiling + honest LOCO-CV alpha for each dense variant
    def alpha_eval(dkey):
        ah = {a: fuse_hits(a, dkey) for a in alphas}
        best_a = max(alphas, key=lambda a: ah[a].mean())
        ceil = float(ah[best_a].mean())
        loco = np.zeros(n, dtype=np.int8)
        choice = {}
        for c in convs:
            te = conv_of == c
            tr = ~te
            ba = max(alphas, key=lambda a: ah[a][tr].mean())
            choice[c] = float(ba)
            loco[te] = ah[ba][te]
        return {"ceiling_alpha": float(best_a), "ceiling_h1": ceil,
                "loco_h1": float(loco.mean()), "loco_choice": choice,
                "loco_pred": loco}

    maxsim = alpha_eval("dmax")
    top3 = alpha_eval("dtop3")
    meanf = alpha_eval("dmean")
    rrf_max = rrf_hits("dmax")

    methods = {
        "BM25": bm_hits,
        "Dense-sessionmean": dmean_hits,
        "Dense-maxsim": dmax_hits,
        "Dense-top3": dtop3_hits,
        "RRF(BM25,maxsim)": rrf_max,
        "Fusion-sessionmean(LOCO)": meanf["loco_pred"],
        "Fusion-maxsim(LOCO)": maxsim["loco_pred"],
        "Fusion-top3(LOCO)": top3["loco_pred"],
    }
    h1 = {k: float(v.mean()) for k, v in methods.items()}
    for k in methods:
        print(f"  {k:28s} Hit@1={h1[k]:.4f}", flush=True)
    print(f"  [maxsim fusion full-data ceiling: a={maxsim['ceiling_alpha']} "
          f"-> {maxsim['ceiling_h1']:.4f}]", flush=True)

    idx_by_conv = {c: np.where(conv_of == c)[0] for c in convs}

    def cboot(a, b, seed):
        rng = np.random.default_rng(seed)
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

    champ = methods["Fusion-maxsim(LOCO)"]
    comps = {}
    for base in ["BM25", "Dense-maxsim", "RRF(BM25,maxsim)",
                 "Fusion-sessionmean(LOCO)"]:
        comps[f"Fusion-maxsim vs {base}"] = cboot(champ, methods[base],
                                                  seed=abs(hash(base)) % 9999)
    print("", flush=True)
    for name, c in comps.items():
        print(f"  {name:42s} d={c['diff']*100:+.2f}pp "
              f"CI=[{c['lo']*100:+.2f},{c['hi']*100:+.2f}] "
              f"p={c['p_one_sided']:.4f}", flush=True)

    per_cat = {}
    for c in sorted(set(cat_of.tolist())):
        m = cat_of == c
        per_cat[int(c)] = {"name": CAT_NAMES.get(int(c), "?"),
                           "n": int(m.sum()),
                           "BM25": float(bm_hits[m].mean()),
                           "Dense-maxsim": float(dmax_hits[m].mean()),
                           "Fusion-maxsim": float(champ[m].mean()),
                           "delta_pp": float((champ[m].mean()
                                              - bm_hits[m].mean()) * 100)}

    summary = {"ts": ts, "n_eval": n, "n_conversations": len(convs),
               "n_boot": n_boot, "hit1": h1,
               "maxsim_ceiling_alpha": maxsim["ceiling_alpha"],
               "maxsim_ceiling_h1": maxsim["ceiling_h1"],
               "maxsim_loco_choice": maxsim["loco_choice"],
               "comparisons": comps, "per_category": per_cat}
    (run_dir / "tune9.json").write_text(json.dumps(summary, indent=2,
                                        default=str), encoding="utf-8")

    md = ["## tune9: late-interaction (max-sim) turn-level fusion on LoCoMo",
          "",
          f"n_eval = {n} across {len(convs)} conversations. BGE-base turn "
          f"embeddings; session score = max_t cos(query, turn). LOCO-CV "
          f"fusion weight; cluster-bootstrap (n_boot={n_boot}). CPU only.",
          "", "### Hit@1", "", "| method | Hit@1 |", "|---|---|"]
    for k in methods:
        md.append(f"| {k} | {h1[k]:.4f} |")
    md += ["",
           f"Full-data ceiling for max-sim fusion (overfit upper bound): "
           f"a={maxsim['ceiling_alpha']} -> {maxsim['ceiling_h1']:.4f}.",
           "", "### Champion (Fusion-maxsim, LOCO-CV) vs baselines", "",
           "| comparison | Δ pp | 95% CI | p (1-sided) |", "|---|---|---|---|"]
    for name, c in comps.items():
        md.append(f"| {name} | {c['diff']*100:+.2f} | "
                  f"[{c['lo']*100:+.2f}, {c['hi']*100:+.2f}] | "
                  f"{c['p_one_sided']:.4f} |")
    md += ["", "### Per-category Hit@1", "",
           "| cat | name | n | BM25 | Dense-maxsim | Fusion-maxsim | Δ vs BM25 |",
           "|---|---|---|---|---|---|---|"]
    for c, info in per_cat.items():
        md.append(f"| {c} | {info['name']} | {info['n']} | "
                  f"{info['BM25']:.3f} | {info['Dense-maxsim']:.3f} | "
                  f"{info['Fusion-maxsim']:.3f} | {info['delta_pp']:+.2f}pp |")
    md += ["", "Raw: `tune9.json`.", ""]
    (run_dir / "tune9.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'tune9.md'}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    run(Path(args.locomo), Path(args.out), n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

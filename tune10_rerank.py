"""
tune10_rerank.py — Cross-encoder rerank on top of max-sim fusion (stretch).

First stage: BM25 + max-sim turn dense fusion (alpha=0.6, the tune9
champion). Second stage: take the top-K sessions; for each, cross-encode
(query, best-matching-turn-text) with a MS-MARCO cross-encoder and rerank.
The best turn per session is the argmax-cosine turn already identified by
max-sim, so the CE sees the most relevant snippet rather than a truncated
2000-char session blob.

Compares CE-reranked Hit@1 to the first-stage fusion, cluster-bootstrap
over conversations. CPU; CE inference is the slow part.

Output: results/tune10-rerank-<ts>/{tune10.json, tune10.md}
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
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALPHA = 0.6


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def run(json_path, out_dir, K=10, n_boot=4000):
    from sentence_transformers import CrossEncoder

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune10-rerank-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(json_path, encoding="utf-8"))
    # turn text lookup: (ci, si) -> list[str]
    turn_text = {}
    for ci, rec in enumerate(data):
        conv = rec["conversation"]
        for k in conv:
            if k.startswith("session_") and not k.endswith("_date_time"):
                si = int(k.split("_")[1])
                turn_text[(ci, si)] = [
                    f"{t.get('speaker','?')}: {t.get('text','') or ''}"
                    for t in conv[k]]

    exs = [e for e in locomo.iter_examples(json_path) if e["gold_session_ids"]]
    qc = np.load(QCACHE, allow_pickle=True)
    q_emb = qc["q_emb"]
    tc = np.load(TCACHE, allow_pickle=True)
    cs_cache = {}

    print("first stage: max-sim fusion...", flush=True)
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
        best_turn = [0] * S
        for j, sid in enumerate(sids):
            key = f"turns_{ci}_{int(sid[1:])}"
            if key in tc.files:
                sims = tc[key] @ qv
                bt = int(np.argmax(sims))
                best_turn[j] = bt
                dmax[j] = float(sims[bt])
        fused = ALPHA * _z(bm) + (1 - ALPHA) * _z(dmax)
        sid_to_idx = {s: j for j, s in enumerate(sids)}
        gold = {sid_to_idx[g] for g in e["gold_session_ids"] if g in sid_to_idx}
        rows.append({"i": i, "conv": ci, "cat": e["category"],
                     "q": e["question"], "sids": sids, "fused": fused,
                     "best_turn": best_turn, "gold": gold})

    fusion_hit = np.array([int(int(np.argmax(r["fused"])) in r["gold"])
                           for r in rows], dtype=np.int8)
    print(f"  first-stage fusion Hit@1 = {fusion_hit.mean():.4f}", flush=True)

    print(f"loading CE {CE_MODEL}...", flush=True)
    ce = CrossEncoder(CE_MODEL, max_length=256)

    print(f"second stage: CE rerank top-{K}...", flush=True)
    ce_hit = np.zeros(len(rows), dtype=np.int8)
    for n_done, r in enumerate(rows):
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(rows)}", flush=True)
        order = np.argsort(-r["fused"])[:K]
        pairs, sess_js = [], []
        for j in order:
            ci = r["conv"]
            si = int(r["sids"][j][1:])
            txt = turn_text.get((ci, si), [""])
            bt = r["best_turn"][j]
            snippet = txt[bt] if bt < len(txt) else (txt[0] if txt else "")
            pairs.append((r["q"], snippet))
            sess_js.append(j)
        scores = ce.predict(pairs, show_progress_bar=False)
        top_j = sess_js[int(np.argmax(scores))]
        ce_hit[n_done] = int(top_j in r["gold"])
    print(f"  CE-reranked Hit@1 = {ce_hit.mean():.4f}", flush=True)

    conv_of = np.array([r["conv"] for r in rows])
    convs = sorted(set(conv_of.tolist()))
    idx_by_conv = {c: np.where(conv_of == c)[0] for c in convs}

    def cboot(a, b, seed=1):
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

    boot = cboot(ce_hit, fusion_hit)
    print(f"  CE vs fusion: d={boot['diff']*100:+.2f}pp "
          f"CI=[{boot['lo']*100:+.2f},{boot['hi']*100:+.2f}] "
          f"p={boot['p_one_sided']:.4f}", flush=True)

    summary = {"ts": ts, "K": K, "alpha": ALPHA, "ce_model": CE_MODEL,
               "n_eval": len(rows), "n_boot": n_boot,
               "fusion_hit1": float(fusion_hit.mean()),
               "ce_hit1": float(ce_hit.mean()),
               "ce_vs_fusion": boot}
    (run_dir / "tune10.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    md = ["## tune10: cross-encoder rerank on top of max-sim fusion", "",
          f"First stage: max-sim fusion (alpha={ALPHA}). Second stage: CE "
          f"`{CE_MODEL}` over (query, best-turn) of top-{K} sessions. "
          f"n_eval={len(rows)}, cluster-bootstrap n_boot={n_boot}.", "",
          "| stage | Hit@1 |", "|---|---|",
          f"| max-sim fusion (first stage) | {fusion_hit.mean():.4f} |",
          f"| + CE rerank top-{K} | {ce_hit.mean():.4f} |", "",
          f"CE vs fusion: Δ = {boot['diff']*100:+.2f} pp, 95% CI "
          f"[{boot['lo']*100:+.2f}, {boot['hi']*100:+.2f}], "
          f"p = {boot['p_one_sided']:.4f}.", ""]
    (run_dir / "tune10.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'tune10.md'}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    run(Path(args.locomo), Path(args.out), K=args.K, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

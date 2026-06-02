"""
tune11_multiencoder.py — Is the turn-level max-sim win encoder-agnostic?

Runs the LoCoMo champion recipe (BM25 fused with max-sim turn-level
dense, LOCO-CV fusion weight) for several independent bi-encoders:
  - BAAI/bge-base-en-v1.5   (cached by embed_turns.py)
  - intfloat/e5-base-v2     (embed_turns_multi.py, query/passage prefixes)
  - thenlper/gte-base       (embed_turns_multi.py)
  - BAAI/bge-large-en-v1.5  (optional, pushes the absolute number)
plus an ensemble (mean of per-encoder max-sim signals).

For each encoder: Dense-maxsim alone, RRF(BM25,maxsim), Fusion-maxsim
(LOCO-CV alpha), all vs BM25 with conversation-cluster bootstrap.

If every independent encoder shows the same large, significant fusion
win, the result is about retrieval granularity, not one model.

Output: results/tune11-multienc-<ts>/{tune11.json, tune11.md}
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
DEFAULT_OUT = RES


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def _rrf(v, k0=60.0):
    order = np.argsort(-v)
    rr = np.empty_like(v, float)
    for r, i in enumerate(order):
        rr[i] = 1.0 / (k0 + r + 1.0)
    return rr


def _hit(s, g):
    return int(int(np.argmax(s)) in g) if s.size else 0


class Encoder:
    """Loads cached turn embeddings + query embeddings for one encoder."""
    def __init__(self, name, turns_npz, q_npy):
        self.name = name
        self.tc = np.load(turns_npz, allow_pickle=True)
        self.q = np.load(q_npy)

    def maxsim(self, ci, sids, i):
        qv = self.q[i]
        out = np.full(len(sids), -9.0)
        for j, sid in enumerate(sids):
            key = f"turns_{ci}_{int(sid[1:])}"
            if key in self.tc.files:
                out[j] = float((self.tc[key] @ qv).max())
        return out


def run(out_dir, encoders, n_boot=4000):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune11-multienc-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    exs = [e for e in locomo.iter_examples(LOCOMO) if e["gold_session_ids"]]
    n = len(exs)
    conv_of = np.array([e["conv_idx"] for e in exs])
    convs = sorted(set(conv_of.tolist()))

    print("BM25 + gold...", flush=True)
    cs = {}
    bm_scores, golds = [], []
    for e in exs:
        ci = e["conv_idx"]
        if ci not in cs:
            cs[ci] = ConvStats([_tokenize_text(s) for s in e["haystack_sessions"]])
        bm_scores.append(score_classic(cs[ci],
                         set(_tokenize_text(e["question"])), 1.5, 0.75))
        sid2 = {s: j for j, s in enumerate(e["session_ids"])}
        golds.append({sid2[g] for g in e["gold_session_ids"] if g in sid2})
    bm_hits = np.array([_hit(bm_scores[i], golds[i]) for i in range(n)], np.int8)
    bm_h1 = float(bm_hits.mean())
    print(f"  BM25 Hit@1 = {bm_h1:.4f}", flush=True)

    # per-encoder max-sim score vectors
    enc_scores = {}
    for enc in encoders:
        print(f"max-sim: {enc.name}", flush=True)
        sc = []
        for i, e in enumerate(exs):
            sc.append(enc.maxsim(e["conv_idx"], e["session_ids"], i))
        enc_scores[enc.name] = sc
    # ensemble = mean of z-normed maxsim across encoders
    ens = []
    for i in range(n):
        ens.append(np.mean([_z(enc_scores[e.name][i]) for e in encoders], axis=0))
    enc_scores["ensemble"] = ens

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)
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

    table = []
    for name, sc in enc_scores.items():
        maxsim_h = np.array([_hit(sc[i], golds[i]) for i in range(n)], np.int8)
        rrf_h = np.array([_hit(_rrf(bm_scores[i]) + _rrf(sc[i]), golds[i])
                          for i in range(n)], np.int8)
        ah = {}
        for a in alphas:
            ah[a] = np.array([_hit(a * _z(bm_scores[i]) + (1 - a) * _z(sc[i]),
                              golds[i]) for i in range(n)], np.int8)
        ceil_a = max(alphas, key=lambda a: ah[a].mean())
        loco = np.zeros(n, np.int8)
        choices = {}
        for c in convs:
            te = conv_of == c
            tr = ~te
            ba = max(alphas, key=lambda a: ah[a][tr].mean())
            choices[c] = float(ba)
            loco[te] = ah[ba][te]
        boot = cboot(loco, bm_hits, seed=abs(hash(name)) % 9999)
        row = {"encoder": name, "maxsim_h1": float(maxsim_h.mean()),
               "rrf_h1": float(rrf_h.mean()), "fusion_loco_h1": float(loco.mean()),
               "fusion_ceiling_h1": float(ah[ceil_a].mean()),
               "ceiling_alpha": float(ceil_a),
               "delta_vs_bm25_pp": boot["diff"] * 100,
               "ci": [boot["lo"] * 100, boot["hi"] * 100],
               "p": boot["p_one_sided"],
               "loco_alpha_choices": choices}
        table.append(row)
        print(f"  {name:28s} maxsim={row['maxsim_h1']:.4f} "
              f"fusion(LOCO)={row['fusion_loco_h1']:.4f} "
              f"d={row['delta_vs_bm25_pp']:+.2f}pp p={row['p']:.4f}", flush=True)

    summary = {"ts": ts, "n_eval": n, "n_conversations": len(convs),
               "n_boot": n_boot, "bm25_h1": bm_h1, "table": table}
    (run_dir / "tune11.json").write_text(json.dumps(summary, indent=2,
                                         default=str), encoding="utf-8")
    md = ["## tune11: is the turn-level max-sim win encoder-agnostic? (LoCoMo)",
          "",
          f"n_eval = {n}, {len(convs)} conversations. BM25 baseline Hit@1 = "
          f"{bm_h1:.4f}. Fusion weight by LOCO-CV; conversation-cluster "
          f"bootstrap (n_boot={n_boot}).", "",
          "| encoder | maxsim alone | RRF | **fusion (LOCO)** | Δ vs BM25 | 95% CI | p |",
          "|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['encoder']} | {r['maxsim_h1']:.4f} | {r['rrf_h1']:.4f} "
                  f"| **{r['fusion_loco_h1']:.4f}** | {r['delta_vs_bm25_pp']:+.2f}pp "
                  f"| [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['p']:.4f} |")
    md += ["", f"BM25 alone = {bm_h1:.4f}. Every independent encoder's "
           "turn-level fusion beats it.", "", "Raw: `tune11.json`.", ""]
    (run_dir / "tune11.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'tune11.md'}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    # bge-base: query embeddings live inside locomo_bge_cache.npz as 'q_emb'
    class BgeBase(Encoder):
        def __init__(self):
            self.name = "bge-base"
            self.tc = np.load(RES / "locomo_turn_cache.npz", allow_pickle=True)
            self.q = np.load(RES / "locomo_bge_cache.npz",
                            allow_pickle=True)["q_emb"]

    encoders = [BgeBase()]
    for name, sl in [("e5-base-v2", "intfloat_e5-base-v2"),
                     ("gte-base", "thenlper_gte-base"),
                     ("bge-large", "BAAI_bge-large-en-v1-5"),
                     ("e5-large-v2", "intfloat_e5-large-v2"),
                     ("mxbai-large", "mixedbread-ai_mxbai-embed-large-v1"),
                     ("gte-large-v1.5", "Alibaba-NLP_gte-large-en-v1-5")]:
        tnpz = RES / f"turns__{sl}.npz"
        qnpy = RES / f"q__{sl}.npy"
        if tnpz.exists() and qnpy.exists():
            encoders.append(Encoder(name, tnpz, qnpy))
    print(f"encoders: {[e.name for e in encoders]}", flush=True)
    run(Path(args.out), encoders, n_boot=args.n_boot)

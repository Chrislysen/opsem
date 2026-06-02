"""latency.py --- CPU wall-clock per retrieval stage on LoCoMo (bge-base).

Quantifies the cost side of "fuse, don't rerank": fusion is arithmetic on
already-computed scores; a cross-encoder reranker is a neural forward pass per
candidate. Times each online stage per query over a sample and reports mean /
median ms. Unoptimized reference implementation --- the comparison is relative.

    python latency.py
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import locomo
from tune7_bm25 import ConvStats, score_classic, _tokenize_text

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOCOMO = HERE / "data" / "locomo" / "locomo10.json"
N = 200          # queries to time
K_RERANK = 10    # rerank depth


def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def main():
    exs = [e for e in locomo.iter_examples(LOCOMO) if e["gold_session_ids"]][:N]
    tc = np.load(RES / "locomo_turn_cache.npz", allow_pickle=True)
    q_emb = np.load(RES / "locomo_bge_cache.npz", allow_pickle=True)["q_emb"]

    # pre-build per-conversation BM25 stats (offline; not timed)
    cs = {}
    for e in exs:
        ci = e["conv_idx"]
        if ci not in cs:
            cs[ci] = ConvStats([_tokenize_text(s) for s in e["haystack_sessions"]])

    t_bm, t_dense, t_fuse, t_ce, t_qenc = [], [], [], [], []

    # --- load CE + query encoder once (offline; not timed per-query) ---
    from sentence_transformers import CrossEncoder, SentenceTransformer
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # warm up
    _ = ce.predict([("a", "b")])
    _ = enc.encode(["warm up"], convert_to_numpy=True, normalize_embeddings=True)

    for i, e in enumerate(exs):
        ci = e["conv_idx"]
        sids = e["session_ids"]
        qv = q_emb[i]
        q = e["question"]

        # (1) query embedding (one forward pass)
        t0 = time.perf_counter()
        _ = enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        t_qenc.append((time.perf_counter() - t0) * 1000)

        # (2) BM25 scoring over all sessions
        t0 = time.perf_counter()
        bm = score_classic(cs[ci], set(_tokenize_text(q)), 1.5, 0.75)
        t_bm.append((time.perf_counter() - t0) * 1000)

        # (3) dense max-sim over cached turn vectors
        t0 = time.perf_counter()
        S = len(sids)
        dense = np.full(S, -9.0)
        for j, sid in enumerate(sids):
            key = f"turns_{ci}_{int(sid[1:])}"
            if key in tc.files:
                dense[j] = float((tc[key] @ qv).max())
        t_dense.append((time.perf_counter() - t0) * 1000)

        # (4) fusion (z-norm + weighted)
        t0 = time.perf_counter()
        fused = 0.6 * _z(bm) + 0.4 * _z(dense)
        order = np.argsort(-fused)
        t_fuse.append((time.perf_counter() - t0) * 1000)

        # (5) cross-encoder rerank of fused top-K (one representative turn/session)
        pairs = []
        for j in order[:K_RERANK]:
            sess = e["haystack_sessions"][int(j)] if int(j) < len(e["haystack_sessions"]) else ""
            lines = [ln for ln in sess.split("\n") if ln.strip()]
            doc = max(lines, key=len) if lines else ""   # representative turn
            pairs.append((q, doc))
        t0 = time.perf_counter()
        if pairs:
            _ = ce.predict(pairs)
        t_ce.append((time.perf_counter() - t0) * 1000)

        if i % 50 == 0:
            print(f"  timed {i}/{len(exs)}", flush=True)

    def stat(xs):
        a = np.array(xs)
        return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "p95_ms": float(np.quantile(a, 0.95))}

    stages = {"query_encode": stat(t_qenc), "bm25": stat(t_bm),
              "dense_maxsim": stat(t_dense), "fusion": stat(t_fuse),
              "ce_rerank_top10": stat(t_ce)}
    retrieve_total = (stages["query_encode"]["mean_ms"] + stages["bm25"]["mean_ms"]
                      + stages["dense_maxsim"]["mean_ms"] + stages["fusion"]["mean_ms"])
    ce_mult = stages["ce_rerank_top10"]["mean_ms"] / max(retrieve_total, 1e-9)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RES / f"latency-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"ts": ts, "n_queries": len(exs), "encoder": "bge-base",
           "ce_model": "ms-marco-MiniLM-L-6-v2", "k_rerank": K_RERANK,
           "stages_ms": stages, "retrieve_total_mean_ms": retrieve_total,
           "ce_rerank_vs_retrieve_x": ce_mult}
    (out / "latency.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    md = ["## Latency per query (LoCoMo, bge-base, CPU; unoptimized reference)", "",
          f"n={len(exs)} queries. Relative comparison (implementation-dependent constants).", "",
          "| stage | mean ms | median ms | p95 ms |", "|---|---|---|---|"]
    for k, s in stages.items():
        md.append(f"| {k} | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} |")
    md += ["",
           f"Retrieval (encode+BM25+dense+fuse) total: **{retrieve_total:.2f} ms/query**.",
           f"Cross-encoder rerank of top-{K_RERANK}: **{stages['ce_rerank_top10']['mean_ms']:.2f} ms/query** "
           f"(**{ce_mult:.1f}x** the entire retrieval pipeline) --- and it *lowers* Hit@1.", ""]
    (out / "latency.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

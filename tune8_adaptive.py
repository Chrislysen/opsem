"""
tune8_adaptive.py — Deployable query-adaptive fusion on LoCoMo.

Motivation
----------
tune4/tune5 established two facts on LoCoMo session retrieval:
  * A single global fusion weight (BM25 + dense) only TIES BM25 overall;
    the per-category optimum is strongly heterogeneous (single-hop wants
    dense, open-ended wants BM25).
  * An ORACLE that applies each category's optimal alpha gains +2.43pp
    over BM25 — but it cheats, using the gold category label.
tune7 showed the lexical knob (k1, b) is exhausted: no deployable gain.

This script asks the deployable question: can a model choose the fusion
weighting PER QUERY, from query text alone (no gold label, no category),
and beat both BM25 and the best fixed-alpha fusion?

Method
------
A session-level learning-to-rank model (L2 logistic regression, numpy)
over features:
    [bm25_z, dense_z, rrf_bm25, rrf_dense]                  base signals
  + [qfeat_k * bm25_z, qfeat_k * dense_z  for each qfeat]   query-conditional
where qfeat_k are cheap, query-text-only features (length, temporal
words, entity count, ...). The interaction terms let the effective
BM25/dense weighting vary per query — a learned, deployable analog of
the oracle category gate.

Validation
----------
Leave-one-conversation-out CV: the model is fit on nine conversations
and applied frozen to the held-out one. Reported Hit@1 aggregates
held-out predictions only. Cluster-bootstrap (resample conversations)
for every comparison. Conversations are the resampling unit because
questions within a conversation share a haystack.

Baselines (all honest / parameter-free or LOCO-CV-selected):
  BM25 (default), Dense (BGE-base), RRF hybrid, best fixed-alpha (LOCO-CV).

Output: results/tune8-adaptive-<ts>/{tune8.json, tune8.md}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import locomo  # noqa: E402
from tune7_bm25 import ConvStats, score_classic, _tokenize_text  # noqa: E402

DEFAULT_LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
CACHE = _HERE / "results" / "locomo_bge_cache.npz"
DEFAULT_OUT = _HERE / "results"
CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
             4: "open-ended", 5: "adversarial"}

_TEMPORAL = {"when", "date", "year", "month", "day", "first", "last",
             "before", "after", "ago", "recent", "earlier", "later",
             "during", "time", "long"}
_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _znorm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    sd = v.std()
    if sd < 1e-9:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def _rrf_rank(v: np.ndarray, k0: float = 60.0) -> np.ndarray:
    """RRF contribution: 1/(k0 + rank) where rank is by descending score."""
    order = np.argsort(-v)
    rr = np.empty_like(v, dtype=np.float64)
    for rank, idx in enumerate(order):
        rr[idx] = 1.0 / (k0 + rank + 1.0)
    return rr


def query_features(q: str) -> dict[str, float]:
    toks = _tokenize_text(q)
    ql = q.lower()
    n = max(len(toks), 1)
    n_temporal = sum(1 for t in ql.split() if t.strip("?,.") in _TEMPORAL)
    return {
        "short": 1.0 if len(toks) <= 5 else 0.0,
        "long": 1.0 if len(toks) >= 12 else 0.0,
        "temporal": 1.0 if n_temporal > 0 else 0.0,
        "when": 1.0 if ql.strip().startswith("when") else 0.0,
        "would": 1.0 if "would" in ql or "if " in ql else 0.0,
        "n_caps": min(len(_CAP_RE.findall(q)), 4) / 4.0,
        "has_num": 1.0 if re.search(r"\d", q) else 0.0,
    }


FEAT_KEYS = ["short", "long", "temporal", "when", "would", "n_caps", "has_num"]


def build(json_path: Path):
    examples = [e for e in locomo.iter_examples(json_path)
                if e["gold_session_ids"]]
    cache = np.load(CACHE, allow_pickle=True)
    meta = json.loads(str(cache["meta"]))
    assert meta["n"] == len(examples), "cache/example mismatch"
    q_emb = cache["q_emb"]
    sess_emb = {int(k.split("_")[-1]): cache[k] for k in cache.files
                if k.startswith("sess_emb_")}

    conv_stats: dict[int, ConvStats] = {}
    rows = []
    for i, e in enumerate(examples):
        ci = e["conv_idx"]
        if ci not in conv_stats:
            conv_stats[ci] = ConvStats(
                [_tokenize_text(s) for s in e["haystack_sessions"]])
        cs = conv_stats[ci]
        qt = set(_tokenize_text(e["question"]))
        bm = score_classic(cs, qt, k1=1.5, b=0.75)
        dense = (sess_emb[ci] @ q_emb[i]).astype(np.float64)
        sid_to_idx = {sid: j for j, sid in enumerate(e["session_ids"])}
        gold = {sid_to_idx[g] for g in e["gold_session_ids"] if g in sid_to_idx}
        rows.append({
            "conv": ci, "cat": e["category"],
            "bm": bm, "dense": dense, "gold": gold,
            "qf": query_features(e["question"]),
            "S": len(e["session_ids"]),
        })
    return rows


def hit1(score: np.ndarray, gold: set[int]) -> int:
    return int(int(np.argmax(score)) in gold) if score.size else 0


# ── numpy L2 logistic regression (full-batch gradient descent) ─────────
def fit_logreg(X, y, w=None, l2=1.0, lr=0.5, iters=300):
    n, d = X.shape
    if w is None:
        w = np.ones(n)
    theta = np.zeros(d)
    wsum = w.sum()
    for _ in range(iters):
        z = X @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (w * (p - y)) / wsum + l2 * theta / n
        theta -= lr * grad
    return theta


def design_matrix(rows, idxs):
    """Stack per-session feature rows for the given example indices.
    Returns X, y, and a list mapping back (ex_local_idx, sess_idx)."""
    feats, labels, groups = [], [], []
    for li, gi in enumerate(idxs):
        r = rows[gi]
        bmz, dz = _znorm(r["bm"]), _znorm(r["dense"])
        rb, rd = _rrf_rank(r["bm"]), _rrf_rank(r["dense"])
        qf = np.array([r["qf"][k] for k in FEAT_KEYS])
        S = r["S"]
        for s in range(S):
            base = [1.0, bmz[s], dz[s], rb[s] * 60.0, rd[s] * 60.0]
            inter = list(qf * bmz[s]) + list(qf * dz[s])
            feats.append(base + inter)
            labels.append(1.0 if s in r["gold"] else 0.0)
            groups.append((li, s))
    return (np.array(feats, dtype=np.float64),
            np.array(labels, dtype=np.float64), groups)


def predict_hits(rows, idxs, theta):
    hits, cats = [], []
    for gi in idxs:
        r = rows[gi]
        bmz, dz = _znorm(r["bm"]), _znorm(r["dense"])
        rb, rd = _rrf_rank(r["bm"]), _rrf_rank(r["dense"])
        qf = np.array([r["qf"][k] for k in FEAT_KEYS])
        S = r["S"]
        X = []
        for s in range(S):
            base = [1.0, bmz[s], dz[s], rb[s] * 60.0, rd[s] * 60.0]
            inter = list(qf * bmz[s]) + list(qf * dz[s])
            X.append(base + inter)
        score = np.array(X) @ theta
        hits.append(hit1(score, r["gold"]))
        cats.append(r["cat"])
    return np.array(hits), np.array(cats)


def run(json_path: Path, out_dir: Path, n_boot=4000, l2=1.0):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune8-adaptive-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("building features (BM25 + dense + query features)...", flush=True)
    rows = build(json_path)
    n = len(rows)
    convs = sorted({r["conv"] for r in rows})
    conv_of = np.array([r["conv"] for r in rows])
    cat_of = np.array([r["cat"] for r in rows])
    print(f"  {n} examples, {len(convs)} conversations", flush=True)

    # ── fixed baselines (per-example scoring) ─────────────────────────
    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    def fused_hits(alpha):
        h = np.zeros(n, dtype=np.int8)
        for i, r in enumerate(rows):
            s = alpha * _znorm(r["bm"]) + (1 - alpha) * _znorm(r["dense"])
            h[i] = hit1(s, r["gold"])
        return h

    bm_hits = fused_hits(1.0)
    dense_hits = fused_hits(0.0)
    rrf_hits = np.zeros(n, dtype=np.int8)
    for i, r in enumerate(rows):
        s = _rrf_rank(r["bm"]) + _rrf_rank(r["dense"])
        rrf_hits[i] = hit1(s, r["gold"])

    alpha_hits = {a: fused_hits(a) for a in alphas}

    # honest LOCO-CV best-fixed-alpha
    fixed_pred = np.zeros(n, dtype=np.int8)
    fixed_choice = {}
    for c in convs:
        te = conv_of == c
        tr = ~te
        best_a = max(alphas, key=lambda a: alpha_hits[a][tr].mean())
        fixed_choice[c] = float(best_a)
        fixed_pred[te] = alpha_hits[best_a][te]

    # ── proposed: LOCO-CV learned query-adaptive fusion ───────────────
    adapt_pred = np.zeros(n, dtype=np.int8)
    for c in convs:
        tr_idx = [i for i in range(n) if conv_of[i] != c]
        te_idx = [i for i in range(n) if conv_of[i] == c]
        X, y, _ = design_matrix(rows, tr_idx)
        # standardize columns (except bias col 0)
        mu = X[:, 1:].mean(0); sd = X[:, 1:].std(0); sd[sd < 1e-9] = 1.0
        Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1.0)
        w = np.where(y > 0, pos_w, 1.0)
        theta = fit_logreg(Xs, y, w=w, l2=l2)
        # apply to held-out, with same standardization
        hh = []
        for gi in te_idx:
            r = rows[gi]
            bmz, dz = _znorm(r["bm"]), _znorm(r["dense"])
            rb, rd = _rrf_rank(r["bm"]), _rrf_rank(r["dense"])
            qf = np.array([r["qf"][k] for k in FEAT_KEYS])
            Xte = []
            for s in range(r["S"]):
                base = [1.0, bmz[s], dz[s], rb[s] * 60.0, rd[s] * 60.0]
                inter = list(qf * bmz[s]) + list(qf * dz[s])
                Xte.append(base + inter)
            Xte = np.array(Xte)
            Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
            score = Xte @ theta
            hh.append(hit1(score, r["gold"]))
        adapt_pred[np.array(te_idx)] = np.array(hh)

    results = {
        "BM25": bm_hits, "Dense": dense_hits, "RRF": rrf_hits,
        "FixedAlpha(LOCO)": fixed_pred, "Adaptive(LOCO)": adapt_pred,
    }
    h1 = {k: float(v.mean()) for k, v in results.items()}
    for k in results:
        print(f"  {k:20s} Hit@1 = {h1[k]:.4f}", flush=True)

    # ── cluster bootstrap helper ──────────────────────────────────────
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

    comps = {}
    for base in ["BM25", "Dense", "RRF", "FixedAlpha(LOCO)"]:
        comps[f"Adaptive vs {base}"] = cboot(adapt_pred, results[base],
                                             seed=hash(base) % 9999)
    for name, c in comps.items():
        print(f"  {name:28s} d={c['diff']*100:+.2f}pp "
              f"CI=[{c['lo']*100:+.2f},{c['hi']*100:+.2f}] "
              f"p={c['p_one_sided']:.4f}", flush=True)

    # per-category
    per_cat = {}
    for c in sorted(set(cat_of.tolist())):
        m = cat_of == c
        per_cat[int(c)] = {"name": CAT_NAMES.get(int(c), "?"),
                           "n": int(m.sum()),
                           **{k: float(v[m].mean()) for k, v in results.items()}}

    summary = {"ts": ts, "n_eval": n, "n_conversations": len(convs),
               "n_boot": n_boot, "l2": l2, "hit1": h1,
               "fixed_alpha_choices": fixed_choice,
               "comparisons": comps, "per_category": per_cat,
               "feat_keys": FEAT_KEYS}
    (run_dir / "tune8.json").write_text(json.dumps(summary, indent=2,
                                                   default=str),
                                        encoding="utf-8")

    md = ["## tune8: deployable query-adaptive fusion on LoCoMo", "",
          f"n_eval = {n} across {len(convs)} conversations. BGE-base dense + "
          f"BM25(default) + cheap query features. LOCO-CV; cluster-bootstrap "
          f"(n_boot={n_boot}).", "",
          "### Hit@1 (honest)", "",
          "| method | Hit@1 |", "|---|---|"]
    for k in ["BM25", "Dense", "RRF", "FixedAlpha(LOCO)", "Adaptive(LOCO)"]:
        md.append(f"| {k} | {h1[k]:.4f} |")
    md += ["", "### Adaptive vs baselines (cluster-bootstrap)", "",
           "| comparison | Δ pp | 95% CI | p (1-sided) |", "|---|---|---|---|"]
    for name, c in comps.items():
        md.append(f"| {name} | {c['diff']*100:+.2f} | "
                  f"[{c['lo']*100:+.2f}, {c['hi']*100:+.2f}] | "
                  f"{c['p_one_sided']:.4f} |")
    md += ["", "### Per-category Hit@1", "",
           "| cat | name | n | BM25 | Dense | RRF | FixedA | Adaptive |",
           "|---|---|---|---|---|---|---|---|"]
    for c, info in per_cat.items():
        md.append(f"| {c} | {info['name']} | {info['n']} | "
                  f"{info['BM25']:.3f} | {info['Dense']:.3f} | "
                  f"{info['RRF']:.3f} | {info['FixedAlpha(LOCO)']:.3f} | "
                  f"{info['Adaptive(LOCO)']:.3f} |")
    md += ["", "Raw: `tune8.json`.", ""]
    (run_dir / "tune8.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'tune8.md'}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=str(DEFAULT_LOCOMO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    run(Path(args.locomo), Path(args.out), n_boot=args.n_boot, l2=args.l2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

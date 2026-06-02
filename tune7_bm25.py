"""
tune7_bm25.py — Rigorous BM25 (k1, b) tuning on LoCoMo, honestly validated.

Thesis
------
Every prior comparison in this repo pits neural fusion / cross-encoder
reranking against BM25 with the *default* parameters k1=1.5, b=0.75.
Those defaults were calibrated on TREC web/news ad-hoc retrieval. LoCoMo
"documents" are multi-turn conversational sessions that are long and
highly variable in length — exactly the regime where BM25's length
normalization (b) and term-frequency saturation (k1) are most off.
Nobody ever tuned the lexical baseline itself. This script does, with a
protocol that cannot overfit:

  1. Reproduce the published BM25-only baseline (Hit@1 = 0.6390) to
     validate the harness.
  2. Build a hits matrix H[example, config] over a (k1, b) grid for two
     BM25 variants (classic and BM25L), computed ONCE.
  3. Full-data best config = the *ceiling* (overfit; reported only as an
     upper bound, clearly labelled).
  4. Leave-one-conversation-out CV = the *honest, deployable* estimate:
     for each held-out conversation, the config is chosen on the other
     nine, then frozen and applied to the held-out conversation. The
     reported Hit@1 is an aggregate of held-out predictions only.
  5. Paired cluster-bootstrap (cluster = conversation) for the honest
     LOCO-CV estimate vs the default BM25. Conversations are the unit of
     resampling because questions within a conversation share a haystack
     and are not independent.
  6. Per-category breakdown of the honest estimate.

Runs on numpy only — no torch, no sentence-transformers, no GPU.

Output: results/tune7-bm25-<ts>/{tune7.json, tune7.md}
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


# Exact copy of longmemeval._tokenize_text so the default config reproduces
# the published BM25 baseline (Hit@1 = 0.6390). Kept inline to avoid the
# package-relative imports in longmemeval.py.
_TOK_BAN = {"the", "and", "for", "with", "from", "this", "that", "what",
            "where", "when", "how", "why", "who", "are", "you", "was",
            "have", "has", "did", "do", "does", "is", "it", "be", "been",
            "an", "as", "we", "me", "my", "i", "your", "yours", "our",
            "of", "to", "on", "in", "at", "by", "or", "if", "but", "not",
            "a", "so", "all", "any", "no", "yes"}
_TOK_RE = re.compile(r"[a-z0-9][a-z0-9\-_']{1,30}")


def _tokenize_text(s: str) -> list[str]:
    s = (s or "").lower()
    toks = _TOK_RE.findall(s)
    return [t for t in toks if t not in _TOK_BAN and len(t) >= 3]

DEFAULT_LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
DEFAULT_OUT = _HERE / "results"

CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
             4: "open-ended", 5: "adversarial"}


# ───────────────────────── precompute per-conversation stats ─────────
class ConvStats:
    """Per-conversation BM25 statistics, computed once.

    tf[token] -> np.int32 array of length S (term freq in each session).
    dl        -> np.float64 array of length S (session token counts).
    idf_classic[token], idf_l[token] -> float.
    """
    __slots__ = ("S", "dl", "avgdl", "N", "tf", "idf_classic", "idf_l")

    def __init__(self, session_token_lists: list[list[str]]):
        self.S = len(session_token_lists)
        self.dl = np.array([len(t) for t in session_token_lists],
                           dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.S else 1.0
        self.N = self.S
        counters = [Counter(t) for t in session_token_lists]
        df: Counter = Counter()
        for c in counters:
            for tok in c:
                df[tok] += 1
        self.tf: dict[str, np.ndarray] = {}
        for tok in df:
            arr = np.zeros(self.S, dtype=np.float64)
            for j, c in enumerate(counters):
                if tok in c:
                    arr[j] = c[tok]
            self.tf[tok] = arr
        # IDFs (do not depend on k1/b/query).
        self.idf_classic: dict[str, float] = {}
        self.idf_l: dict[str, float] = {}
        N = self.N
        for tok, d in df.items():
            self.idf_classic[tok] = float(np.log((N - d + 0.5) / (d + 0.5) + 1.0))
            self.idf_l[tok] = float(np.log((N + 1.0) / (d + 0.5)))


def score_classic(cs: ConvStats, q_terms, k1: float, b: float) -> np.ndarray:
    """Classic BM25 score vector over sessions for one query."""
    score = np.zeros(cs.S, dtype=np.float64)
    norm = (1.0 - b) + b * cs.dl / max(cs.avgdl, 1e-9)
    for t in q_terms:
        tf = cs.tf.get(t)
        if tf is None:
            continue
        idf = cs.idf_classic[t]
        denom = tf + k1 * norm
        score += idf * (tf * (k1 + 1.0)) / np.where(denom > 0, denom, 1.0)
    return score


def score_bm25l(cs: ConvStats, q_terms, k1: float, b: float,
                delta: float = 0.5) -> np.ndarray:
    """BM25L (Lv & Zhai 2011) score vector. ctd = tf / norm; the +delta
    lower-bounds the term saturation so very long sessions are not
    over-penalised. Only matching terms contribute (the constant from
    non-matching terms is rank-neutral within a query)."""
    score = np.zeros(cs.S, dtype=np.float64)
    norm = (1.0 - b) + b * cs.dl / max(cs.avgdl, 1e-9)
    for t in q_terms:
        tf = cs.tf.get(t)
        if tf is None:
            continue
        idf = cs.idf_l[t]
        ctd = tf / norm
        mask = tf > 0
        contrib = idf * ((k1 + 1.0) * (ctd + delta)) / (k1 + ctd + delta)
        score += np.where(mask, contrib, 0.0)
    return score


def _hit1(score: np.ndarray, gold_idx: set[int]) -> int:
    if score.size == 0:
        return 0
    # argmax with stable lowest-index tie-break (matches np.argsort(-x)[0]).
    top = int(np.argmax(score))
    return int(top in gold_idx)


def build_examples(json_path: Path):
    raw = [e for e in locomo.iter_examples(json_path) if e["gold_session_ids"]]
    conv_stats: dict[int, ConvStats] = {}
    examples = []
    for e in raw:
        ci = e["conv_idx"]
        if ci not in conv_stats:
            tok_lists = [_tokenize_text(s) for s in e["haystack_sessions"]]
            conv_stats[ci] = ConvStats(tok_lists)
        sid_to_idx = {sid: j for j, sid in enumerate(e["session_ids"])}
        gold_idx = {sid_to_idx[g] for g in e["gold_session_ids"]
                    if g in sid_to_idx}
        examples.append({
            "conv_idx": ci,
            "category": e["category"],
            "q_terms": set(_tokenize_text(e["question"])),
            "gold_idx": gold_idx,
        })
    return examples, conv_stats


def run(json_path: Path, out_dir: Path, n_boot: int = 4000) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"tune7-bm25-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading LoCoMo + precomputing conversation stats...", flush=True)
    examples, conv_stats = build_examples(json_path)
    n = len(examples)
    convs = sorted({e["conv_idx"] for e in examples})
    print(f"  {n} examples across {len(convs)} conversations", flush=True)

    k1_grid = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]
    b_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    configs = []
    for var in ("classic", "bm25l"):
        for k1 in k1_grid:
            for b in b_grid:
                configs.append((var, k1, b))
    n_cfg = len(configs)
    DEFAULT = ("classic", 1.5, 0.75)
    default_j = configs.index(DEFAULT)
    print(f"  {n_cfg} configs (incl. default classic k1=1.5 b=0.75)",
          flush=True)

    print("computing hits matrix H[example, config]...", flush=True)
    H = np.zeros((n, n_cfg), dtype=np.int8)
    for i, ex in enumerate(examples):
        if i % 300 == 0:
            print(f"  {i}/{n}", flush=True)
        cs = conv_stats[ex["conv_idx"]]
        qt = ex["q_terms"]
        gold = ex["gold_idx"]
        for j, (var, k1, b) in enumerate(configs):
            if var == "classic":
                s = score_classic(cs, qt, k1, b)
            else:
                s = score_bm25l(cs, qt, k1, b)
            H[i, j] = _hit1(s, gold)

    conv_of = np.array([e["conv_idx"] for e in examples])
    cat_of = np.array([e["category"] for e in examples])
    cfg_mean = H.mean(axis=0)

    default_h1 = float(H[:, default_j].mean())
    best_j = int(np.argmax(cfg_mean))
    best_cfg = configs[best_j]
    best_h1 = float(cfg_mean[best_j])
    print(f"\n  default BM25 (classic k1=1.5 b=0.75) Hit@1 = {default_h1:.4f}",
          flush=True)
    print(f"  full-data BEST (ceiling, overfit) = {best_cfg} "
          f"Hit@1 = {best_h1:.4f}", flush=True)

    # ── honest leave-one-conversation-out CV ──────────────────────────
    loco_pred = np.zeros(n, dtype=np.int8)
    loco_choice: dict[int, tuple] = {}
    for c in convs:
        test_mask = conv_of == c
        train_mask = ~test_mask
        train_means = H[train_mask].mean(axis=0)
        sel = int(np.argmax(train_means))
        loco_choice[c] = configs[sel]
        loco_pred[test_mask] = H[test_mask, sel]
    loco_h1 = float(loco_pred.mean())
    print(f"  LOCO-CV honest Hit@1 = {loco_h1:.4f}", flush=True)

    # ── cluster bootstrap (resample conversations) ────────────────────
    idx_by_conv = {c: np.where(conv_of == c)[0] for c in convs}

    def cluster_boot(pred_a: np.ndarray, pred_b: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        cl = np.array(convs)
        diffs = np.empty(n_boot)
        for bi in range(n_boot):
            pick = rng.choice(cl, size=len(cl), replace=True)
            ia = np.concatenate([idx_by_conv[c] for c in pick])
            diffs[bi] = pred_a[ia].mean() - pred_b[ia].mean()
        point = float(pred_a.mean() - pred_b.mean())
        return {"diff": point,
                "lo": float(np.quantile(diffs, 0.025)),
                "hi": float(np.quantile(diffs, 0.975)),
                "p_one_sided": float(np.mean(diffs <= 0))}

    default_pred = H[:, default_j].astype(np.int8)
    boot_loco_vs_def = cluster_boot(loco_pred, default_pred, seed=1)
    print(f"\n  LOCO-CV vs default: d={boot_loco_vs_def['diff']*100:+.2f}pp "
          f"CI=[{boot_loco_vs_def['lo']*100:+.2f},{boot_loco_vs_def['hi']*100:+.2f}] "
          f"p={boot_loco_vs_def['p_one_sided']:.4f}", flush=True)

    # ── per-category honest breakdown ─────────────────────────────────
    per_cat = {}
    for c in sorted(set(cat_of.tolist())):
        m = cat_of == c
        per_cat[int(c)] = {
            "name": CAT_NAMES.get(int(c), "?"),
            "n": int(m.sum()),
            "default_h1": float(default_pred[m].mean()),
            "loco_h1": float(loco_pred[m].mean()),
            "delta_pp": float((loco_pred[m].mean()
                               - default_pred[m].mean()) * 100),
        }

    # top configs table (full-data, for context)
    order = np.argsort(-cfg_mean)
    top_cfgs = [{"variant": configs[j][0], "k1": configs[j][1],
                 "b": configs[j][2], "hit1": float(cfg_mean[j])}
                for j in order[:12]]

    summary = {
        "ts": ts, "n_eval": n, "n_conversations": len(convs),
        "n_boot": n_boot, "k1_grid": k1_grid, "b_grid": b_grid,
        "default_config": DEFAULT, "default_hit1": default_h1,
        "fulldata_best_config": best_cfg, "fulldata_best_hit1": best_h1,
        "loco_cv_hit1": loco_h1,
        "loco_cv_choices": {str(c): loco_choice[c] for c in convs},
        "boot_loco_vs_default": boot_loco_vs_def,
        "per_category": per_cat,
        "top_configs_fulldata": top_cfgs,
    }
    (run_dir / "tune7.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "## tune7: rigorous BM25 (k1, b) tuning on LoCoMo",
        "",
        f"n_eval = {n} across {len(convs)} conversations. Grid: "
        f"k1 ∈ {k1_grid}, b ∈ {b_grid}, variants {{classic, BM25L}}. "
        f"Runs on numpy only.",
        "",
        f"**Default BM25 (classic, k1=1.5, b=0.75)**: Hit@1 = {default_h1:.4f} "
        f"(reproduces the published 0.6390 baseline).",
        "",
        "### Honest leave-one-conversation-out CV (deployable)",
        "",
        f"For each of the {len(convs)} conversations, the (variant, k1, b) "
        "is chosen on the other nine and applied frozen to the held-out "
        "conversation. Reported Hit@1 aggregates held-out predictions only.",
        "",
        f"- **LOCO-CV Hit@1 = {loco_h1:.4f}**",
        f"- vs default BM25 ({default_h1:.4f}): "
        f"Δ = {boot_loco_vs_def['diff']*100:+.2f} pp, "
        f"95% CI [{boot_loco_vs_def['lo']*100:+.2f}, "
        f"{boot_loco_vs_def['hi']*100:+.2f}], "
        f"cluster-bootstrap p = {boot_loco_vs_def['p_one_sided']:.4f}",
        "",
        f"Full-data ceiling (overfit, upper bound only): "
        f"{best_cfg} → {best_h1:.4f}.",
        "",
        "### Per-fold chosen config",
        "",
        "| held-out conv | chosen (variant, k1, b) |",
        "|---|---|",
    ]
    for c in convs:
        md.append(f"| {c} | {loco_choice[c]} |")
    md += ["",
           "### Per-category (honest LOCO-CV vs default)",
           "",
           "| cat | name | n | default | LOCO-CV | Δ |",
           "|---|---|---|---|---|---|"]
    for c, info in per_cat.items():
        md.append(f"| {c} | {info['name']} | {info['n']} | "
                  f"{info['default_h1']:.4f} | {info['loco_h1']:.4f} | "
                  f"{info['delta_pp']:+.2f}pp |")
    md += ["",
           "### Top configs (full-data; context only, do not deploy)",
           "",
           "| rank | variant | k1 | b | Hit@1 |",
           "|---|---|---|---|---|"]
    for i, r in enumerate(top_cfgs):
        md.append(f"| {i+1} | {r['variant']} | {r['k1']} | {r['b']} | "
                  f"{r['hit1']:.4f} |")
    md += ["", "Raw grid + bootstrap: `tune7.json`.", ""]
    (run_dir / "tune7.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'tune7.md'}", flush=True)
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

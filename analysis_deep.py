"""analysis_deep.py --- deeper analyses for the paper, all from cached signals.

Produces real new results (no new training, CPU, from cache):
  1. Per-category LoCoMo breakdown (BM25 vs dense max-sim vs BM25(+)max-sim fusion).
  2. alpha-sensitivity curve (global alpha sweep, Hit@1) for the headline encoder.
  3. RRF vs weighted fusion on LoCoMo.
  4. Length-dilution: late-early Hit@1 gap by gold-session #turns, per encoder.

Outputs results/analysis-deep-<ts>/{analysis.json, analysis.md} and two figures.
    python analysis_deep.py
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tune13_interaction as T

RES = T.RES
HEAD = "e5-large-v2"  # headline encoder
# standard LoCoMo category id -> name (Maharana et al. 2024)
CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain",
            4: "single-hop", 5: "adversarial"}


def hit1(scores, golds, mask=None):
    n = len(golds)
    idx = range(n) if mask is None else [i for i in range(n) if mask[i]]
    if not idx:
        return float("nan"), 0
    h = [int(int(np.argmax(np.asarray(scores[i], float))) in golds[i]) for i in idx]
    return float(np.mean(h)), len(idx)


def rrf_scores(bm, dense, k=60):
    """Reciprocal-rank fusion of two score lists -> fused score per session."""
    out = []
    for b, d in zip(bm, dense):
        b = np.asarray(b, float); d = np.asarray(d, float)
        rb = (-b).argsort().argsort()  # 0 = best
        rd = (-d).argsort().argsort()
        out.append(1.0 / (k + rb + 1) + 1.0 / (k + rd + 1))
    return out


def gold_lengths(D, exs):
    """#turns in the (first) gold session per example, from cached turn mats."""
    tc = np.load(RES / "locomo_turn_cache.npz", allow_pickle=True)
    lens = []
    for e in exs:
        ci = e["conv_idx"]
        sid2 = {s: j for j, s in enumerate(e["session_ids"])}
        gs = [g for g in e["gold_session_ids"] if g in sid2]
        n = 0
        for g in gs:
            key = f"turns_{ci}_{int(g[1:])}"
            if key in tc.files:
                n = max(n, tc[key].shape[0])
        lens.append(n)
    return np.array(lens)


def main(n_boot=4000):
    D = T.build_signals()
    n, golds, conv_of, convs, bm = (D["n"], D["golds"], D["conv_of"],
                                    D["convs"], D["bm"])
    cats = np.array(D["cats"])
    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)
    exs = [e for e in T.locomo.iter_examples(T.LOCOMO) if e["gold_session_ids"]]

    dense = D["scores"][HEAD]["max-sim(late)"]
    fused, _ = T.fuse_loco(bm, dense, golds, conv_of, convs, alphas)

    # ---- 1. per-category ----
    per_cat = []
    for c in sorted(set(cats.tolist())):
        m = cats == c
        b_h, nb = hit1(bm, golds, m)
        d_h, _ = hit1(dense, golds, m)
        f_h, _ = hit1(fused, golds, m)
        per_cat.append({"category": int(c), "name": CAT_NAME.get(int(c), str(c)),
                        "n": nb, "bm25": b_h, "dense_late": d_h, "fusion": f_h,
                        "fusion_minus_bm25_pp": (f_h - b_h) * 100})

    # ---- 2. alpha sensitivity (global alpha; z-normed) ----
    def zl(v):
        v = np.asarray(v, float); sd = v.std()
        return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd
    alpha_curve = []
    for a in alphas:
        fl = [a * zl(bm[i]) + (1 - a) * zl(dense[i]) for i in range(n)]
        h, _ = hit1(fl, golds)
        alpha_curve.append({"alpha": float(a), "hit1": h})
    best_a = max(alpha_curve, key=lambda r: r["hit1"])

    # ---- 3. RRF vs weighted ----
    rrf = rrf_scores(bm, dense)
    rrf_h, _ = hit1(rrf, golds)
    wf_h, _ = hit1(fused, golds)
    bm_h, _ = hit1(bm, golds)
    d_h, _ = hit1(dense, golds)

    # ---- 4. length-dilution: late-early gap by gold-session #turns ----
    lens = gold_lengths(D, exs)
    buckets = [(2, 7), (8, 15), (16, 25), (26, 999)]
    length_rows = []
    for enc in D["encoders"]:
        e_early = D["scores"][enc]["mean(early)"]
        e_late = D["scores"][enc]["max-sim(late)"]
        row = {"encoder": enc, "buckets": []}
        for lo, hi in buckets:
            m = (lens >= lo) & (lens <= hi)
            eh, nn = hit1(e_early, golds, m)
            lh, _ = hit1(e_late, golds, m)
            row["buckets"].append({"range": f"{lo}-{hi if hi < 999 else '+'}",
                                   "n": int(m.sum()),
                                   "gap_pp": (lh - eh) * 100 if nn else None})
        length_rows.append(row)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RES / f"analysis-deep-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"ts": ts, "headline_encoder": HEAD, "n_eval": n,
           "per_category": per_cat, "alpha_curve": alpha_curve,
           "alpha_best": best_a,
           "fusion_comparators": {"bm25": bm_h, "dense_late": d_h,
                                  "rrf": rrf_h, "weighted_loco": wf_h},
           "length_dilution": length_rows}
    (out / "analysis.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # ---- markdown ----
    md = [f"## Deep analysis (LoCoMo, headline encoder {HEAD})", "",
          "### 1. Per-category Hit@1", "",
          "| category | n | BM25 | dense max-sim | BM25(+)max-sim | fusion-BM25 pp |",
          "|---|---|---|---|---|---|"]
    for r in per_cat:
        md.append(f"| {r['name']} | {r['n']} | {r['bm25']:.3f} | {r['dense_late']:.3f} "
                  f"| {r['fusion']:.3f} | {r['fusion_minus_bm25_pp']:+.2f} |")
    md += ["", "### 2. Alpha sensitivity (global alpha, Hit@1)", "",
           f"Best global alpha = {best_a['alpha']:.2f} (Hit@1 {best_a['hit1']:.4f}). "
           f"alpha=0 is pure dense, alpha=1 is pure BM25.", "",
           "| alpha | Hit@1 |", "|---|---|"]
    for r in alpha_curve:
        md.append(f"| {r['alpha']:.2f} | {r['hit1']:.4f} |")
    md += ["", "### 3. RRF vs weighted fusion (LoCoMo, Hit@1)", "",
           f"BM25 {bm_h:.4f} | dense max-sim {d_h:.4f} | RRF {rrf_h:.4f} | "
           f"weighted LOCO-CV {wf_h:.4f}", "",
           "### 4. Length-dilution: late-early Hit@1 gap by gold-session #turns", ""]
    for row in length_rows:
        cells = " | ".join(f"{b['range']}: {b['gap_pp']:+.1f}pp (n={b['n']})"
                           if b['gap_pp'] is not None else f"{b['range']}: -- (n={b['n']})"
                           for b in row["buckets"])
        md.append(f"- **{row['encoder']}**: {cells}")
    md += ["", "Raw: `analysis.json`.", ""]
    (out / "analysis.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    # alpha curve
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = [r["alpha"] for r in alpha_curve]; ys = [r["hit1"] for r in alpha_curve]
    ax.plot(xs, ys, "-o", ms=3, color="#2c6fbb")
    ax.axhline(bm_h, ls="--", color="#888", lw=1, label=f"BM25 ({bm_h:.3f})")
    ax.axvline(best_a["alpha"], ls=":", color="#c33", lw=1,
               label=f"best alpha={best_a['alpha']:.2f}")
    ax.set_xlabel(r"fusion weight $\alpha$ (0=dense, 1=BM25)")
    ax.set_ylabel("Hit@1"); ax.set_title(f"Fusion alpha-sensitivity ({HEAD})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_alpha.png", dpi=130, bbox_inches="tight")
    fig.savefig(RES / "fig_alpha.pdf", bbox_inches="tight")
    # per-category bars
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = [r["name"] for r in per_cat]
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, [r["bm25"] for r in per_cat], w, label="BM25", color="#9aa7b8")
    ax.bar(x, [r["dense_late"] for r in per_cat], w, label="dense max-sim", color="#7fa0c8")
    ax.bar(x + w, [r["fusion"] for r in per_cat], w, label="BM25 ⊕ max-sim", color="#2c6fbb")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("Hit@1"); ax.set_title(f"Per-category LoCoMo ({HEAD})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_category.png", dpi=130, bbox_inches="tight")
    fig.savefig(RES / "fig_category.pdf", bbox_inches="tight")

    print(f"wrote {out/'analysis.md'}")
    print("per-category fusion-BM25 pp:",
          {r["name"]: round(r["fusion_minus_bm25_pp"], 1) for r in per_cat})
    print(f"alpha best={best_a['alpha']:.2f} hit1={best_a['hit1']:.4f}; "
          f"RRF={rrf_h:.4f} weighted={wf_h:.4f}")


if __name__ == "__main__":
    main()

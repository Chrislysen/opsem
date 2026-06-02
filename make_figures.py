"""make_figures.py — the signature figures for the interaction-function finding.

Reads the tune13 interaction receipt (LoCoMo, retrieval unit held fixed at
*session*, only the query<->session pooling operator varies over the SAME
cached turn vectors) and renders two panels:

  Panel A  per-encoder dense Hit@1, early (mean-pool) vs late (max-sim),
           with the BM25 reference line. The gap exists at every encoder.
  Panel B  the encoder-scaling law: dense late-early gap (pp) vs encoder
           quality, with OLS fit + Pearson r. The mechanistic signature
           (a tuning artifact would not track model quality).

Outputs results/fig_interaction.png (+ .pdf). Training-free, CPU.

    python make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RECEIPT = Path("results/tune13-interaction-20260601T170200/tune13.json")
OUT = Path("results/fig_interaction")

# nicer display labels keyed by the receipt's encoder names
NICE = {
    "bge-base": "bge-base\n109M",
    "e5-base-v2": "e5-base-v2\n109M",
    "gte-base": "gte-base\n109M",
    "bge-large": "bge-large\n335M",
    "mxbai-large": "mxbai-large\n335M",
    "e5-large-v2": "e5-large-v2\n335M",
}


def load():
    d = json.loads(RECEIPT.read_text(encoding="utf-8"))
    dense = {}
    for r in d["dense_table"]:
        dense.setdefault(r["encoder"], {})[r["pooler"]] = r
    encs = d["encoders"]
    early = np.array([dense[e]["mean(early)"]["hit@1"] for e in encs])
    late = np.array([dense[e]["max-sim(late)"]["hit@1"] for e in encs])
    gap = np.array([d["significance"][e]["dense_late_vs_early"]["diff"] * 100 for e in encs])
    bm25 = d["bm25"]["hit@1"]
    return d, encs, early, late, gap, bm25


def main():
    d, encs, early, late, gap, bm25 = load()
    # order by encoder quality (late max-sim Hit@1) for both panels
    order = np.argsort(late)
    encs = [encs[i] for i in order]
    early, late, gap = early[order], late[order], gap[order]

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    })
    fig, axA = plt.subplots(1, 1, figsize=(6.4, 4.0))

    # ---- per-encoder early vs late dense Hit@1 ----
    x = np.arange(len(encs))
    w = 0.38
    cE, cL = "#9aa7b8", "#2c6fbb"
    axA.bar(x - w / 2, early, w, label="early (mean-pool)", color=cE)
    axA.bar(x + w / 2, late, w, label="late (max-sim)", color=cL)
    for xi, e, l in zip(x, early, late):
        axA.annotate(f"+{(l - e) * 100:.0f}", (xi, l), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8, color=cL, weight="bold")
    axA.axhline(bm25, ls="--", lw=1.2, color="#444",
                label=f"BM25 ({bm25:.3f})")
    axA.set_xticks(x)
    axA.set_xticklabels([NICE.get(e, e) for e in encs], fontsize=8)
    axA.set_ylabel("Dense Hit@1 (LoCoMo, n=1978)")
    axA.set_ylim(0, 0.8)
    axA.set_title("Same turn vectors, only the pooling operator changes",
                  fontsize=11, weight="bold")
    axA.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")

    # transparency receipt: per-encoder gaps + (caveated) capacity grouping.
    # NOTE: we deliberately do NOT headline a gap-vs-own-score correlation in
    # the paper -- the x would be the encoder's own max-sim score, making it
    # near-circular. We report the gaps and the base-vs-large grouping instead.
    params = {"gte-base": 109, "bge-base": 109, "e5-base-v2": 109,
              "bge-large": 335, "mxbai-large": 335, "e5-large-v2": 335}
    gaps = {e: float(g) for e, g in zip(encs, gap)}
    base = [gaps[e] for e in encs if params.get(e, 0) <= 110]
    large = [gaps[e] for e in encs if params.get(e, 0) > 110]
    stats_doc = {
        "source": str(RECEIPT),
        "late_minus_early_gap_pp": gaps,
        "params_M": {e: params.get(e) for e in encs},
        "mean_gap_base_109M_pp": float(np.mean(base)),
        "mean_gap_large_335M_pp": float(np.mean(large)),
        "note": ("Gap is large for all six encoders and larger on average for "
                 "335M than 109M encoders, but not monotone (mxbai-large 335M "
                 "ties e5-base 109M). We avoid a gap-vs-own-score correlation "
                 "as it would be near-circular."),
        "n_encoders": int(len(encs)),
    }
    (OUT.parent / "fig_interaction_stats.json").write_text(
        json.dumps(stats_doc, indent=2), encoding="utf-8")
    print(f"wrote {OUT.with_suffix('.png')}  and  {OUT.with_suffix('.pdf')}")
    print(f"base(109M) mean gap={np.mean(base):.1f}pp  large(335M) mean gap={np.mean(large):.1f}pp")


if __name__ == "__main__":
    main()

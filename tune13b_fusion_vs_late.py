"""tune13b_fusion_vs_late.py — does BM25 fusion add OVER late-interaction alone?

The key delta vs dense-only turn-max-sim retrieval (e.g. Nano-Memory's TIR):
holding the dense term fixed at late interaction (max-sim over turn vectors),
does score-level fusion with BM25 significantly improve Hit@1? Reuses
tune13's cached signals (CPU, no model load) and the same
conversation-cluster bootstrap.

    python tune13b_fusion_vs_late.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tune13_interaction as T

RES = T.RES


def main(n_boot=4000):
    D = T.build_signals()
    n, golds, conv_of, convs = D["n"], D["golds"], D["conv_of"], D["convs"]
    bm = D["bm"]
    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    _, bm_hit1 = T.eval_method(bm, golds)
    rows = []
    for enc in D["encoders"]:
        dense_late = D["scores"][enc]["max-sim(late)"]
        dm, dhit1 = T.eval_method(dense_late, golds)
        fused, chosen = T.fuse_loco(bm, dense_late, golds, conv_of, convs, alphas)
        fm, fhit1 = T.eval_method(fused, golds)
        cmp = T.cboot_hit(fhit1, dhit1, conv_of, convs, n_boot,
                          seed=abs(hash("fl" + enc)) % 99999)
        rows.append({
            "encoder": enc,
            "dense_late_hit1": dm["hit@1"],
            "fusion_late_hit1": fm["hit@1"],
            "loco_alpha_median": float(np.median(list(chosen.values()))),
            "delta_pp": cmp["diff"] * 100,
            "ci_lo_pp": cmp["lo"] * 100,
            "ci_hi_pp": cmp["hi"] * 100,
            "p_one_sided": cmp["p_one_sided"],
        })
        print(f"  {enc:14s} dense-late {dm['hit@1']:.4f} -> fusion {fm['hit@1']:.4f}"
              f"  d={cmp['diff']*100:+.2f}pp  CI[{cmp['lo']*100:+.2f},{cmp['hi']*100:+.2f}]"
              f"  p={cmp['p_one_sided']:.4f}  (alpha~{np.median(list(chosen.values())):.2f})",
              flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RES / f"tune13b-fusion-vs-late-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tune13b.json").write_text(
        json.dumps({"ts": ts, "n_eval": n, "n_conversations": len(convs),
                    "n_boot": n_boot, "comparison": "fusion(max-sim) vs dense(max-sim) alone",
                    "rows": rows}, indent=2), encoding="utf-8")
    md = ["## tune13b: does BM25 fusion add OVER late-interaction (max-sim) alone? (LoCoMo)",
          "",
          f"n_eval = {n}, {len(convs)} conversations, conversation-cluster bootstrap "
          f"(n_boot={n_boot}). Dense term fixed at max-sim; only BM25 fusion (LOCO-CV α) added.",
          "",
          "| encoder | dense max-sim Hit@1 | +BM25 fusion Hit@1 | Δ pp | 95% CI | p |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['encoder']} | {r['dense_late_hit1']:.4f} | "
                  f"{r['fusion_late_hit1']:.4f} | {r['delta_pp']:+.2f} | "
                  f"[{r['ci_lo_pp']:+.2f}, {r['ci_hi_pp']:+.2f}] | {r['p_one_sided']:.4f} |")
    md += ["", "Raw: `tune13b.json`.", ""]
    (out / "tune13b.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out / 'tune13b.md'}")


if __name__ == "__main__":
    main()

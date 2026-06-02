## tune10: cross-encoder rerank on top of max-sim fusion

First stage: max-sim fusion (alpha=0.6). Second stage: CE `cross-encoder/ms-marco-MiniLM-L-6-v2` over (query, best-turn) of top-10 sessions. n_eval=1978, cluster-bootstrap n_boot=2000.

| stage | Hit@1 |
|---|---|
| max-sim fusion (first stage) | 0.7012 |
| + CE rerank top-10 | 0.6325 |

CE vs fusion: Δ = -6.88 pp, 95% CI [-9.34, -4.34], p = 1.0000.

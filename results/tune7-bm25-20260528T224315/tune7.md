## tune7: rigorous BM25 (k1, b) tuning on LoCoMo

n_eval = 1978 across 10 conversations. Grid: k1 ∈ [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0], b ∈ [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0], variants {classic, BM25L}. Runs on numpy only.

**Default BM25 (classic, k1=1.5, b=0.75)**: Hit@1 = 0.6395 (reproduces the published 0.6390 baseline).

### Honest leave-one-conversation-out CV (deployable)

For each of the 10 conversations, the (variant, k1, b) is chosen on the other nine and applied frozen to the held-out conversation. Reported Hit@1 aggregates held-out predictions only.

- **LOCO-CV Hit@1 = 0.6395**
- vs default BM25 (0.6395): Δ = +0.00 pp, 95% CI [-1.04, +0.88], cluster-bootstrap p = 0.5002

Full-data ceiling (overfit, upper bound only): ('bm25l', 1.2, 0.2) → 0.6486.

### Per-fold chosen config

| held-out conv | chosen (variant, k1, b) |
|---|---|
| 0 | ('classic', 0.8, 0.9) |
| 1 | ('bm25l', 1.2, 0.2) |
| 2 | ('classic', 0.6, 0.4) |
| 3 | ('bm25l', 1.2, 0.2) |
| 4 | ('bm25l', 1.2, 0.2) |
| 5 | ('bm25l', 1.2, 0.2) |
| 6 | ('bm25l', 1.0, 0.6) |
| 7 | ('bm25l', 1.2, 0.2) |
| 8 | ('bm25l', 1.2, 0.1) |
| 9 | ('classic', 0.6, 0.4) |

### Per-category (honest LOCO-CV vs default)

| cat | name | n | default | LOCO-CV | Δ |
|---|---|---|---|---|---|
| 1 | single-hop | 281 | 0.4342 | 0.4342 | +0.00pp |
| 2 | multi-hop | 321 | 0.5826 | 0.6012 | +1.87pp |
| 3 | temporal | 89 | 0.3708 | 0.3596 | -1.12pp |
| 4 | open-ended | 841 | 0.7122 | 0.7075 | -0.48pp |
| 5 | adversarial | 446 | 0.7265 | 0.7242 | -0.22pp |

### Top configs (full-data; context only, do not deploy)

| rank | variant | k1 | b | Hit@1 |
|---|---|---|---|---|
| 1 | bm25l | 1.2 | 0.2 | 0.6486 |
| 2 | bm25l | 1.0 | 0.2 | 0.6481 |
| 3 | classic | 0.6 | 0.4 | 0.6476 |
| 4 | bm25l | 1.0 | 0.6 | 0.6476 |
| 5 | bm25l | 1.2 | 0.4 | 0.6476 |
| 6 | bm25l | 1.2 | 0.5 | 0.6476 |
| 7 | bm25l | 1.5 | 0.2 | 0.6476 |
| 8 | classic | 0.6 | 0.5 | 0.6471 |
| 9 | bm25l | 1.0 | 0.3 | 0.6471 |
| 10 | bm25l | 1.0 | 0.5 | 0.6471 |
| 11 | bm25l | 0.8 | 0.6 | 0.6471 |
| 12 | bm25l | 0.6 | 0.6 | 0.6471 |

Raw grid + bootstrap: `tune7.json`.

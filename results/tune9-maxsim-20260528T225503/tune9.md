## tune9: late-interaction (max-sim) turn-level fusion on LoCoMo

n_eval = 1978 across 10 conversations. BGE-base turn embeddings; session score = max_t cos(query, turn). LOCO-CV fusion weight; cluster-bootstrap (n_boot=4000). CPU only.

### Hit@1

| method | Hit@1 |
|---|---|
| BM25 | 0.6395 |
| Dense-sessionmean | 0.3493 |
| Dense-maxsim | 0.5470 |
| Dense-top3 | 0.5521 |
| RRF(BM25,maxsim) | 0.6507 |
| Fusion-sessionmean(LOCO) | 0.6527 |
| Fusion-maxsim(LOCO) | 0.6987 |
| Fusion-top3(LOCO) | 0.6967 |

Full-data ceiling for max-sim fusion (overfit upper bound): a=0.6 -> 0.7012.

### Champion (Fusion-maxsim, LOCO-CV) vs baselines

| comparison | Δ pp | 95% CI | p (1-sided) |
|---|---|---|---|
| Fusion-maxsim vs BM25 | +5.92 | [+4.61, +7.10] | 0.0000 |
| Fusion-maxsim vs Dense-maxsim | +15.17 | [+13.10, +17.18] | 0.0000 |
| Fusion-maxsim vs RRF(BM25,maxsim) | +4.80 | [+3.46, +6.16] | 0.0000 |
| Fusion-maxsim vs Fusion-sessionmean(LOCO) | +4.60 | [+3.03, +5.97] | 0.0000 |

### Per-category Hit@1

| cat | name | n | BM25 | Dense-maxsim | Fusion-maxsim | Δ vs BM25 |
|---|---|---|---|---|---|---|
| 1 | single-hop | 281 | 0.434 | 0.438 | 0.516 | +8.19pp |
| 2 | multi-hop | 321 | 0.583 | 0.561 | 0.657 | +7.48pp |
| 3 | temporal | 89 | 0.371 | 0.348 | 0.427 | +5.62pp |
| 4 | open-ended | 841 | 0.712 | 0.586 | 0.759 | +4.64pp |
| 5 | adversarial | 446 | 0.726 | 0.572 | 0.785 | +5.83pp |

Raw: `tune9.json`.

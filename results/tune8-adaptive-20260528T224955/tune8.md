## tune8: deployable query-adaptive fusion on LoCoMo

n_eval = 1978 across 10 conversations. BGE-base dense + BM25(default) + cheap query features. LOCO-CV; cluster-bootstrap (n_boot=4000).

### Hit@1 (honest)

| method | Hit@1 |
|---|---|
| BM25 | 0.6395 |
| Dense | 0.3493 |
| RRF | 0.4985 |
| FixedAlpha(LOCO) | 0.6527 |
| Adaptive(LOCO) | 0.6527 |

### Adaptive vs baselines (cluster-bootstrap)

| comparison | Δ pp | 95% CI | p (1-sided) |
|---|---|---|---|
| Adaptive vs BM25 | +1.31 | [-0.46, +3.31] | 0.0845 |
| Adaptive vs Dense | +30.33 | [+27.15, +33.82] | 0.0000 |
| Adaptive vs RRF | +15.42 | [+12.14, +18.67] | 0.0000 |
| Adaptive vs FixedAlpha(LOCO) | +0.00 | [-0.60, +0.66] | 0.5513 |

### Per-category Hit@1

| cat | name | n | BM25 | Dense | RRF | FixedA | Adaptive |
|---|---|---|---|---|---|---|---|
| 1 | single-hop | 281 | 0.434 | 0.399 | 0.491 | 0.498 | 0.495 |
| 2 | multi-hop | 321 | 0.583 | 0.355 | 0.477 | 0.579 | 0.579 |
| 3 | temporal | 89 | 0.371 | 0.236 | 0.337 | 0.393 | 0.393 |
| 4 | open-ended | 841 | 0.712 | 0.340 | 0.507 | 0.712 | 0.713 |
| 5 | adversarial | 446 | 0.726 | 0.354 | 0.536 | 0.742 | 0.742 |

Raw: `tune8.json`.

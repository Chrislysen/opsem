## LongMemEval-S: turn-level max-sim fusion transfer test

n_eval = 500 (challenge_only=True). BGE-base turn embeddings; bootstrap over questions (n_boot=4000).

| method | Hit@1 |
|---|---|
| BM25 | 0.8600 |
| Dense-sessionmean | 0.8340 |
| Dense-maxsim | 0.9100 |
| RRF(BM25,maxsim) | 0.9040 |
| Fusion-maxsim(CV) | 0.9300 |

Fusion full-data ceiling: a=0.25 -> 0.9300.

### Fusion vs baselines

| comparison | Δ pp | 95% CI | p |
|---|---|---|---|
| Fusion vs BM25 | +7.00 | [+4.60, +9.60] | 0.0000 |
| Fusion vs Dense-maxsim | +2.00 | [+0.20, +4.00] | 0.0180 |
| Fusion vs RRF(BM25,maxsim) | +2.60 | [+1.00, +4.40] | 0.0020 |

Raw: `lme.json`.

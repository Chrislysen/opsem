## LongMemEval-S: turn-level max-sim fusion transfer test

n_eval = 200 (challenge_only=True). BGE-base turn embeddings; bootstrap over questions (n_boot=4000).

| method | Hit@1 |
|---|---|
| BM25 | 0.8700 |
| Dense-sessionmean | 0.8600 |
| Dense-maxsim | 0.9350 |
| RRF(BM25,maxsim) | 0.9050 |
| Fusion-maxsim(CV) | 0.9350 |

Fusion full-data ceiling: a=0.0 -> 0.9350.

### Fusion vs baselines

| comparison | Δ pp | 95% CI | p |
|---|---|---|---|
| Fusion vs BM25 | +6.50 | [+2.50, +11.00] | 0.0010 |
| Fusion vs Dense-maxsim | +0.00 | [+0.00, +0.00] | 1.0000 |
| Fusion vs RRF(BM25,maxsim) | +3.00 | [+0.50, +6.00] | 0.0195 |

Raw: `lme.json`.

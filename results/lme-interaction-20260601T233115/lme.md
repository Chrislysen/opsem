## LongMemEval-S: interaction-function control (sentence-transformers/all-MiniLM-L6-v2)

n_eval = 150 (challenge_only=True). Unit fixed at session; only the scoring operator varies. Bootstrap over questions (n_boot=4000).

| method | R@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| BM25 | 0.5889 | 0.8878 | 0.9480 | 0.9398 | 0.9164 |
| dense:mean(early) | 0.5450 | 0.8698 | 0.9201 | 0.8944 | 0.8757 |
| dense:max-sim(late) | 0.5678 | 0.8859 | 0.9534 | 0.9185 | 0.9055 |
| dense:top3(late) | 0.5689 | 0.8939 | 0.9507 | 0.9189 | 0.9075 |
| fusion:late | 0.5944 | 0.9129 | 0.9592 | 0.9418 | 0.9297 |
| fusion:early | 0.5978 | 0.9192 | 0.9522 | 0.9454 | 0.9268 |
| RRF(BM25,late) | 0.5944 | 0.9279 | 0.9559 | 0.9450 | 0.9316 |

### Significance (R@1, bootstrap over questions)

| comparison | Δ pp | 95% CI | p |
|---|---|---|---|
| dense late-vs-early | +4.67 | [-1.33, +10.67] | 0.0710 |
| fusion late-vs-early | +0.00 | [-2.00, +2.00] | 0.6488 |
| fusion-late vs BM25 | +0.67 | [-2.67, +4.00] | 0.4323 |
| fusion-late vs RRF | +0.00 | [-2.00, +2.00] | 0.6583 |

### Dilution test: late−early R@1 gap by gold-session #turns

| gold-session turns | n | early | late | gap pp |
|---|---|---|---|---|
| 2-3 | 3 | 1.000 | 1.000 | +0.00 |
| 4-7 | 3 | 1.000 | 1.000 | +0.00 |
| 8-15 | 138 | 0.812 | 0.862 | +5.07 |
| 16+ | 6 | 1.000 | 1.000 | +0.00 |

Raw: `lme.json`.

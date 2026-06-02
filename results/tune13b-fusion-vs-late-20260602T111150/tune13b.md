## tune13b: does BM25 fusion add OVER late-interaction (max-sim) alone? (LoCoMo)

n_eval = 1978, 10 conversations, conversation-cluster bootstrap (n_boot=4000). Dense term fixed at max-sim; only BM25 fusion (LOCO-CV α) added.

| encoder | dense max-sim Hit@1 | +BM25 fusion Hit@1 | Δ pp | 95% CI | p |
|---|---|---|---|---|---|
| bge-base | 0.5470 | 0.6987 | +15.17 | [+12.98, +17.16] | 0.0000 |
| e5-base-v2 | 0.5804 | 0.7017 | +12.13 | [+9.80, +14.83] | 0.0000 |
| gte-base | 0.5187 | 0.6906 | +17.19 | [+13.65, +20.63] | 0.0000 |
| bge-large | 0.6021 | 0.7209 | +11.88 | [+9.51, +14.40] | 0.0000 |
| mxbai-large | 0.5779 | 0.7128 | +13.50 | [+10.30, +16.68] | 0.0000 |
| e5-large-v2 | 0.6638 | 0.7518 | +8.80 | [+6.62, +11.19] | 0.0000 |

Raw: `tune13b.json`.

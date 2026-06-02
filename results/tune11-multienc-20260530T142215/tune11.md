## tune11: is the turn-level max-sim win encoder-agnostic? (LoCoMo)

n_eval = 1978, 10 conversations. BM25 baseline Hit@1 = 0.6395. Fusion weight by LOCO-CV; conversation-cluster bootstrap (n_boot=4000).

| encoder | maxsim alone | RRF | **fusion (LOCO)** | Δ vs BM25 | 95% CI | p |
|---|---|---|---|---|---|---|
| bge-base | 0.5470 | 0.6507 | **0.6987** | +5.92pp | [+4.62, +7.08] | 0.0000 |
| e5-base-v2 | 0.5804 | 0.6835 | **0.7017** | +6.22pp | [+4.77, +7.71] | 0.0000 |
| gte-base | 0.5187 | 0.6299 | **0.6906** | +5.11pp | [+3.83, +6.28] | 0.0000 |
| bge-large | 0.6021 | 0.6790 | **0.7209** | +8.14pp | [+6.10, +10.35] | 0.0000 |
| e5-large-v2 | 0.6638 | 0.7179 | **0.7518** | +11.22pp | [+9.46, +13.08] | 0.0000 |
| mxbai-large | 0.5779 | 0.6658 | **0.7128** | +7.33pp | [+5.58, +8.79] | 0.0000 |
| ensemble | 0.6350 | 0.7007 | **0.7235** | +8.39pp | [+6.58, +9.87] | 0.0000 |

BM25 alone = 0.6395. Every independent encoder's turn-level fusion beats it.

Raw: `tune11.json`.

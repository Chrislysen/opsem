## tune11: is the turn-level max-sim win encoder-agnostic? (LoCoMo)

n_eval = 1978, 10 conversations. BM25 baseline Hit@1 = 0.6395. Fusion weight by LOCO-CV; conversation-cluster bootstrap (n_boot=4000).

| encoder | maxsim alone | RRF | **fusion (LOCO)** | Δ vs BM25 | 95% CI | p |
|---|---|---|---|---|---|---|
| bge-base | 0.5470 | 0.6507 | **0.6987** | +5.92pp | [+4.59, +7.14] | 0.0000 |
| e5-base-v2 | 0.5804 | 0.6835 | **0.7017** | +6.22pp | [+4.73, +7.72] | 0.0000 |
| gte-base | 0.5187 | 0.6299 | **0.6906** | +5.11pp | [+3.93, +6.28] | 0.0000 |
| ensemble | 0.5859 | 0.6744 | **0.6997** | +6.02pp | [+4.56, +7.50] | 0.0000 |

BM25 alone = 0.6395. Every independent encoder's turn-level fusion beats it.

Raw: `tune11.json`.

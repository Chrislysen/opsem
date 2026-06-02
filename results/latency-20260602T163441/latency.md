## Latency per query (LoCoMo, bge-base, CPU; unoptimized reference)

n=200 queries. Implementation-dependent CPU constants; the comparison is RELATIVE. Query embedding is a one-time per-query cost shared by every dense retriever (and is inflated here by per-call overhead), so the headline ratio excludes it.

| stage | mean ms | median ms | p95 ms |
|---|---|---|---|
| query_encode | 3494.3 | 2764.9 | 9784.6 |
| bm25 | 26.7 | 7.7 | 118.0 |
| dense_maxsim | 161.0 | 101.3 | 461.5 |
| fusion | 2.6 | 0.3 | 9.7 |
| ce_rerank_top10 | 4122.1 | 3212.7 | 8958.8 |

**Ranking all candidate sessions** (BM25 + dense max-sim + fusion arithmetic, given the query vector): **190 ms/query**.
**Cross-encoder rerank of the top-10**: **4122 ms/query** = **22x** the cost of ranking the entire haystack --- while *lowering* Hit@1 (see tune10).

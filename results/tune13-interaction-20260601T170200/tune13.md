## tune13: the lever is the INTERACTION FUNCTION, not the unit (LoCoMo)

n_eval = 1978 across 10 conversations. Retrieval unit is held fixed at *session*; only the query<->session scoring operator varies, over the SAME cached turn vectors. Field-standard metrics; conversation-cluster bootstrap (n_boot=4000). CPU, no training.

BM25 baseline: hit@1 0.6395, recall@3 0.7788, recall@5 0.8325, mrr 0.7504, ndcg@5 0.7456

### Dense-alone: early (mean-pool) vs late (max-sim) interaction

| encoder | pooler | Hit@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|---|
| bge-base | mean(early) | 0.3862 | 0.5709 | 0.6740 | 0.5423 | 0.5364 |
| bge-base | max-sim(late) | 0.5470 | 0.6990 | 0.7906 | 0.6748 | 0.6752 |
| bge-base | top3(late) | 0.5521 | 0.7006 | 0.7895 | 0.6778 | 0.6749 |
| bge-base | lse10(late) | 0.3397 | 0.5252 | 0.6466 | 0.5029 | 0.4995 |
| e5-base-v2 | mean(early) | 0.4075 | 0.5738 | 0.6872 | 0.5567 | 0.5518 |
| e5-base-v2 | max-sim(late) | 0.5804 | 0.7441 | 0.8173 | 0.7049 | 0.7081 |
| e5-base-v2 | top3(late) | 0.5763 | 0.7260 | 0.8036 | 0.6980 | 0.6968 |
| e5-base-v2 | lse10(late) | 0.1264 | 0.2737 | 0.3894 | 0.2866 | 0.2592 |
| gte-base | mean(early) | 0.3837 | 0.5521 | 0.6728 | 0.5351 | 0.5322 |
| gte-base | max-sim(late) | 0.5187 | 0.6706 | 0.7635 | 0.6479 | 0.6458 |
| gte-base | top3(late) | 0.5172 | 0.6755 | 0.7682 | 0.6494 | 0.6484 |
| gte-base | lse10(late) | 0.1264 | 0.2691 | 0.3781 | 0.2846 | 0.2543 |
| bge-large | mean(early) | 0.4034 | 0.5831 | 0.6955 | 0.5552 | 0.5546 |
| bge-large | max-sim(late) | 0.6021 | 0.7590 | 0.8231 | 0.7209 | 0.7215 |
| bge-large | top3(late) | 0.5900 | 0.7462 | 0.8252 | 0.7136 | 0.7145 |
| bge-large | lse10(late) | 0.3402 | 0.5506 | 0.6669 | 0.5114 | 0.5128 |
| mxbai-large | mean(early) | 0.4050 | 0.5889 | 0.6962 | 0.5577 | 0.5569 |
| mxbai-large | max-sim(late) | 0.5779 | 0.7324 | 0.8125 | 0.7012 | 0.7028 |
| mxbai-large | top3(late) | 0.5723 | 0.7311 | 0.8135 | 0.6977 | 0.6993 |
| mxbai-large | lse10(late) | 0.4009 | 0.5969 | 0.7044 | 0.5597 | 0.5611 |
| e5-large-v2 | mean(early) | 0.4267 | 0.6006 | 0.7125 | 0.5760 | 0.5732 |
| e5-large-v2 | max-sim(late) | 0.6638 | 0.7919 | 0.8631 | 0.7669 | 0.7694 |
| e5-large-v2 | top3(late) | 0.6365 | 0.7786 | 0.8471 | 0.7478 | 0.7483 |
| e5-large-v2 | lse10(late) | 0.1269 | 0.2781 | 0.3842 | 0.2870 | 0.2573 |

### BM25-fused (LOCO-CV alpha): same pooler sweep

| encoder | pooler | Hit@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|---|
| bge-base | mean(early) | 0.6587 | 0.7962 | 0.8509 | 0.7672 | 0.7626 |
| bge-base | max-sim(late) | 0.6987 | 0.8120 | 0.8710 | 0.7943 | 0.7908 |
| bge-base | top3(late) | 0.6967 | 0.8129 | 0.8694 | 0.7953 | 0.7900 |
| bge-base | lse10(late) | 0.6668 | 0.7897 | 0.8535 | 0.7721 | 0.7675 |
| e5-base-v2 | mean(early) | 0.6613 | 0.7977 | 0.8554 | 0.7702 | 0.7666 |
| e5-base-v2 | max-sim(late) | 0.7017 | 0.8190 | 0.8710 | 0.7982 | 0.7944 |
| e5-base-v2 | top3(late) | 0.7017 | 0.8117 | 0.8708 | 0.7981 | 0.7932 |
| e5-base-v2 | lse10(late) | 0.6476 | 0.7795 | 0.8433 | 0.7560 | 0.7529 |
| gte-base | mean(early) | 0.6673 | 0.7925 | 0.8511 | 0.7716 | 0.7660 |
| gte-base | max-sim(late) | 0.6906 | 0.8063 | 0.8621 | 0.7878 | 0.7825 |
| gte-base | top3(late) | 0.6936 | 0.8102 | 0.8658 | 0.7907 | 0.7861 |
| gte-base | lse10(late) | 0.6380 | 0.7780 | 0.8409 | 0.7506 | 0.7487 |
| bge-large | mean(early) | 0.6502 | 0.7888 | 0.8480 | 0.7610 | 0.7578 |
| bge-large | max-sim(late) | 0.7209 | 0.8326 | 0.8837 | 0.8122 | 0.8090 |
| bge-large | top3(late) | 0.6926 | 0.8220 | 0.8777 | 0.7962 | 0.7950 |
| bge-large | lse10(late) | 0.6552 | 0.7911 | 0.8558 | 0.7668 | 0.7646 |
| mxbai-large | mean(early) | 0.6557 | 0.7943 | 0.8524 | 0.7655 | 0.7621 |
| mxbai-large | max-sim(late) | 0.7128 | 0.8238 | 0.8785 | 0.8056 | 0.8019 |
| mxbai-large | top3(late) | 0.6997 | 0.8179 | 0.8733 | 0.7976 | 0.7939 |
| mxbai-large | lse10(late) | 0.6658 | 0.7982 | 0.8595 | 0.7736 | 0.7709 |
| e5-large-v2 | mean(early) | 0.6658 | 0.7967 | 0.8512 | 0.7722 | 0.7656 |
| e5-large-v2 | max-sim(late) | 0.7518 | 0.8464 | 0.8943 | 0.8347 | 0.8292 |
| e5-large-v2 | top3(late) | 0.7240 | 0.8293 | 0.8905 | 0.8148 | 0.8125 |
| e5-large-v2 | lse10(late) | 0.6461 | 0.7790 | 0.8439 | 0.7556 | 0.7528 |

### The control contrast: late minus early interaction
(identical turn vectors; only the pooling operator differs)

| encoder | dense Δ(late-early) pp | p | fusion Δ(late-early) pp | p | fusion-late vs BM25 pp | p |
|---|---|---|---|---|---|---|
| bge-base | +16.08 | 0.0000 | +3.99 | 0.0000 | +5.92 | 0.0000 |
| e5-base-v2 | +17.29 | 0.0000 | +4.04 | 0.0000 | +6.22 | 0.0000 |
| gte-base | +13.50 | 0.0000 | +2.33 | 0.0000 | +5.11 | 0.0000 |
| bge-large | +19.87 | 0.0000 | +7.08 | 0.0000 | +8.14 | 0.0000 |
| mxbai-large | +17.29 | 0.0000 | +5.71 | 0.0000 | +7.33 | 0.0000 |
| e5-large-v2 | +23.71 | 0.0000 | +8.59 | 0.0000 | +11.22 | 0.0000 |

Raw: `tune13.json`.

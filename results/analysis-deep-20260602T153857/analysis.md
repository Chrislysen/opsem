## Deep analysis (LoCoMo, headline encoder e5-large-v2)

### 1. Per-category Hit@1

| category | n | BM25 | dense max-sim | BM25(+)max-sim | fusion-BM25 pp |
|---|---|---|---|---|---|
| multi-hop | 281 | 0.434 | 0.552 | 0.609 | +17.44 |
| temporal | 321 | 0.583 | 0.704 | 0.745 | +16.20 |
| open-domain | 89 | 0.371 | 0.472 | 0.461 | +8.99 |
| single-hop | 841 | 0.712 | 0.716 | 0.810 | +9.75 |
| adversarial | 446 | 0.726 | 0.646 | 0.796 | +6.95 |

### 2. Alpha sensitivity (global alpha, Hit@1)

Best global alpha = 0.40 (Hit@1 0.7518). alpha=0 is pure dense, alpha=1 is pure BM25.

| alpha | Hit@1 |
|---|---|
| 0.00 | 0.6638 |
| 0.05 | 0.6815 |
| 0.10 | 0.6931 |
| 0.15 | 0.7098 |
| 0.20 | 0.7224 |
| 0.25 | 0.7326 |
| 0.30 | 0.7417 |
| 0.35 | 0.7477 |
| 0.40 | 0.7518 |
| 0.45 | 0.7432 |
| 0.50 | 0.7366 |
| 0.55 | 0.7305 |
| 0.60 | 0.7285 |
| 0.65 | 0.7214 |
| 0.70 | 0.7133 |
| 0.75 | 0.7022 |
| 0.80 | 0.6906 |
| 0.85 | 0.6769 |
| 0.90 | 0.6673 |
| 0.95 | 0.6532 |
| 1.00 | 0.6395 |

### 3. RRF vs weighted fusion (LoCoMo, Hit@1)

BM25 0.6395 | dense max-sim 0.6638 | RRF 0.7179 | weighted LOCO-CV 0.7518

### 4. Length-dilution: late-early Hit@1 gap by gold-session #turns

- **bge-base**: 2-7: -- (n=0) | 8-15: +14.5pp (n=186) | 16-25: +14.9pp (n=1123) | 26-+: +18.5pp (n=669)
- **e5-base-v2**: 2-7: -- (n=0) | 8-15: +9.1pp (n=186) | 16-25: +16.2pp (n=1123) | 26-+: +21.4pp (n=669)
- **gte-base**: 2-7: -- (n=0) | 8-15: +12.4pp (n=186) | 16-25: +13.1pp (n=1123) | 26-+: +14.5pp (n=669)
- **bge-large**: 2-7: -- (n=0) | 8-15: +11.8pp (n=186) | 16-25: +19.6pp (n=1123) | 26-+: +22.6pp (n=669)
- **mxbai-large**: 2-7: -- (n=0) | 8-15: +9.1pp (n=186) | 16-25: +16.9pp (n=1123) | 26-+: +20.2pp (n=669)
- **e5-large-v2**: 2-7: -- (n=0) | 8-15: +19.4pp (n=186) | 16-25: +22.4pp (n=1123) | 26-+: +27.1pp (n=669)

Raw: `analysis.json`.

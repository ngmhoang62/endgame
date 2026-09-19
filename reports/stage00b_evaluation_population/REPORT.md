# Stage 00B — Evaluation Population Forensics

**No folds are created in this stage.**

## Primary population proposal

- Raw official train queries: **7000**
- Raw official literal document IDs preserved: **yes**
- Duplicate document IDs canonicalized: **no**
- Empty-gold queries dropped: **no**

## Corpus audit

- Documents: **8532**
- Empty passages: **20**
- Exact duplicate-passage groups: **4**
- Missing gold document IDs: **0**
- Queries touching empty gold docs: **11**
- Queries touching duplicate-content gold docs: **4**

## Query duplicates

- Strict duplicate groups: **16**
- Punctuation-fold duplicate groups: **16**
- Accent-fold duplicate groups: **16**

## Near-duplicate threshold sensitivity

| Emb cosine | Edges | Components | Queries grouped | Max comp | Gold-overlap edge rate | Median lexical |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 207 | 166 | 354 | 5 | 82.13% | 0.7563 |
| 0.92 | 138 | 119 | 249 | 4 | 83.33% | 0.7729 |
| 0.94 | 100 | 87 | 181 | 3 | 85.00% | 0.7990 |
| 0.95 | 78 | 67 | 141 | 3 | 84.62% | 0.8335 |
| 0.96 | 62 | 52 | 110 | 3 | 83.87% | 0.8757 |
| 0.97 | 45 | 39 | 81 | 3 | 80.00% | 0.8940 |
| 0.98 | 31 | 29 | 59 | 3 | 77.42% | 1.0000 |
| 0.99 | 22 | 22 | 44 | 2 | 86.36% | 1.0000 |

## Next decision

Inspect `NEAR_DUPLICATE_CANDIDATES.csv` and choose the grouping rule.
Only after that will Stage 00C materialize the new ENDGAME folds.

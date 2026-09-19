# Stage 02A — Raw-Parent Acquisition Anchor

**Status: COMPLETE**

## Truncation

- Docs >512: **0 (0.00%)**
- Gold occurrences >512: **0 (0.00%)**
- Queries with all gold >512: **0 (0.00%)**

## Top-5

| Method | Recall@5 | Precision@5 |
|---|---:|---:|
| Dense E5 parent | 0.593608 | 0.125962 |
| BM25 parent | 0.477266 | 0.100844 |
| RRF60 | 0.714173 | 0.151452 |

## Candidate acquisition

| K | Dense | BM25 | Union oracle | Union+dup expansion |
|---:|---:|---:|---:|---:|
| 1 | 0.328160 | 0.234497 | 0.460132 | 0.460204 |
| 5 | 0.593608 | 0.477266 | 0.767747 | 0.767747 |
| 10 | 0.692366 | 0.582950 | 0.853667 | 0.853667 |
| 20 | 0.778875 | 0.672081 | 0.913188 | 0.913188 |
| 50 | 0.866636 | 0.779087 | 0.956547 | 0.956547 |
| 100 | 0.911448 | 0.846934 | 0.974968 | 0.974968 |

- BM25 improves dense top100 pool on **489** queries.
- Dense improves BM25 top100 pool on **969** queries.
- Union+duplicate expansion still fails full coverage on **222** queries.

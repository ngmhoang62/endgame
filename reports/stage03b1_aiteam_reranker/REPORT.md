# Stage 03B1 — AITeam Vietnamese Reranker OOF

- Rerank depth: **30**
- Reranker params: **567,755,777**
- Provisional active model ledger (AIT embedding + LAL + reranker): **1.732B / 4B**

| Method | OOF R@5 | P@5 | Single R@5 | Multi R@5 |
|---|---:|---:|---:|---:|
| ce_feature_lr | 0.942598 | 0.201287 | 0.960248 | 0.736298 |
| nested_rank_ce_blend | 0.940810 | 0.200887 | 0.958385 | 0.735390 |
| selector_top30_order | 0.935815 | 0.199714 | 0.953727 | 0.726467 |
| ce_only | 0.856105 | 0.182578 | 0.873602 | 0.651603 |

Best method is not automatically promoted until fold/stress behavior is inspected.

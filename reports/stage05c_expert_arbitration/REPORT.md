# Stage 05C — Whole-List Expert Arbitration

- Best-list oracle: **0.967790**
- Oracle single-gold: **0.980435**
- Oracle multi-gold: **0.819994**

| Method | R@5 | P@5 | Single | Multi |
|---|---:|---:|---:|---:|
| ce_lr_baseline | 0.942598 | 0.201287 | 0.960248 | 0.736298 |
| expert_router_top1 | 0.943241 | 0.201402 | 0.960870 | 0.737205 |
| expert_router_top2_rrf | 0.940524 | 0.200887 | 0.957764 | 0.739020 |
| expert_router_all_weighted_rrf | 0.939463 | 0.200544 | 0.957143 | 0.732819 |

**Decision: DO_NOT_PROMOTE** (gate 0.960; best=expert_router_top1 0.943241)

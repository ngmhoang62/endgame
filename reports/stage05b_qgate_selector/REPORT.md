# Stage 05B — Query-Conditioned Selector Tournament

| Method | OOF R@5 | P@5 | Single | Multi |
|---|---:|---:|---:|---:|
| stage03b1_ce_lr | 0.942598 | 0.201287 | 0.960248 | 0.736298 |
| lr_plus_title | 0.942753 | 0.201430 | 0.959938 | 0.741894 |
| qgate_listwise | 0.942848 | 0.201516 | 0.959783 | 0.744918 |

- Existing-arm best-arm oracle: **0.967790**
- Rank-only ∪ CE-LR top5 cap-5 oracle: **0.948915**
- All-arm top5 cap-5 oracle: **0.969065**
- CE-LR vs QGate best-arm oracle: **0.952026**
- CE-LR ∪ QGate cap-5 oracle: **0.952217**

**Decision: DO_NOT_PROMOTE** (gate R@5 >= 0.960; best=qgate_listwise 0.942848)

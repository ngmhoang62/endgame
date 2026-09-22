# Stage 06B1 — BGE LoRA DEV Gate

**CERT folds 3–4 were not evaluated.**

| Method | R@5 | P@5 | Single | Multi |
|---|---:|---:|---:|---:|
| baseline_ce_lr | 0.949309 | 0.203431 | 0.964286 | 0.775526 |
| bge_lora_standalone | 0.905349 | 0.192995 | 0.923913 | 0.689940 |
| best_blend | 0.952168 | 0.204146 | 0.966615 | 0.784535 |

- Best alpha(BGE): **0.20**
- Delta R@5: **+0.002859**
- Delta single: **+0.002329**
- Decision: **KILL_OR_REDESIGN**

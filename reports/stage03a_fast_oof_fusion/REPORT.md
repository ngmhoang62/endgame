# Stage 03A — Fast OOF Fusion

- Candidate oracle @50: **0.991484**
- Mean unique pool: **119.0**
- OOF Recall@5: **0.927912**
- OOF Precision@5: **0.198083**
- Single-gold R@5: **0.945342**
- Multi-gold R@5: **0.724198**

## Standalone controls

| Source | R@5 | P@5 |
|---|---:|---:|
| ait_atomic | 0.911224 | 0.194193 |
| ait_coarse1024 | 0.909686 | 0.193992 |
| lal_b4 | 0.857495 | 0.182749 |
| bm25 | 0.477266 | 0.100844 |

Fold-clean OOF only. No private labels or cross-encoder features.

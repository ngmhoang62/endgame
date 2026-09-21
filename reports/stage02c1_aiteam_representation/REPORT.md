# Stage 02C1 — AITeam-v1 Representation Screen

Status: **COMPLETE**

LAL and BM25 are frozen read-only complements; only AITeam document representation changes.

| Representation | Vectors | R@5 | R@20 | R@100 | AITeam+LAL+BM25 oracle@100 | Δ core |
|---|---:|---:|---:|---:|---:|---:|
| baseline_atomic_truncate_2048 | 179737 | 0.884833 | 0.945649 | 0.979662 | 0.992598 | +0.000000 |
| atomic_split_2048 | 196117 | 0.911224 | 0.966342 | 0.986969 | 0.993504 | +0.000906 |
| coarse_pack_1024 | 112899 | 0.909686 | 0.968917 | 0.988685 | 0.993861 | +0.001264 |
| coarse_pack_2048 | 56170 | 0.899375 | 0.964006 | 0.986981 | 0.993563 | +0.000966 |

No representation is auto-promoted; inspect head recall, acquisition depth, complementarity, and vector count together.

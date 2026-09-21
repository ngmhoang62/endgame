# Stage 02C1b — Representation Complementarity

Cache-only analysis; no encoding.

| Configuration | Oracle R@100 | Mean pool | RRF60 R@5 | RRF60 R@100 |
|---|---:|---:|---:|---:|
| baseline_core | 0.992598 | 217.5 | 0.838142 | 0.988993 |
| coarse1024_core | 0.993861 | 219.3 | 0.851028 | 0.990531 |
| atomic_core | 0.993504 | 217.4 | 0.847201 | 0.990161 |
| dual_aiteam_core | 0.994076 | 235.5 | 0.887169 | 0.990407 |
| dual_aiteam_no_lal | 0.991704 | 197.4 | 0.847595 | 0.989477 |
| all_three_aiteam_core | 0.994076 | 249.4 | 0.903202 | 0.990454 |

RRF60 is an untuned sanity check, not final fusion.

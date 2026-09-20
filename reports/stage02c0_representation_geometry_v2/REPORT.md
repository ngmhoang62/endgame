# Stage 02C0 v2 — Representation Geometry Audit

v1 LAL token counts are invalidated; v2 explicitly disables backend truncation.

| Model | Representation | Vectors | Ratio | Median vec/doc | P95 vec/doc | Coverage |
|---|---|---:|---:|---:|---:|---:|
| aiteamvn_v1 | current_atomic_truncate_2048 | 179737 | 1.000x | 14.0 | 62.0 | 71.4234% |
| aiteamvn_v1 | atomic_split_2048 | 195050 | 1.085x | 16.0 | 65.0 | 100.0000% |
| aiteamvn_v1 | coarse_pack_1024 | 109982 | 0.612x | 7.0 | 42.0 | 100.0000% |
| aiteamvn_v1 | coarse_pack_1792 | 62790 | 0.349x | 4.0 | 23.0 | 100.0000% |
| vnlegal_lal | current_atomic_truncate_2048 | 179737 | 1.000x | 14.0 | 62.0 | 67.9399% |
| vnlegal_lal | atomic_split_2048 | 201024 | 1.118x | 16.0 | 66.0 | 100.0000% |
| vnlegal_lal | coarse_pack_1024 | 137957 | 0.768x | 9.0 | 52.0 | 100.0000% |
| vnlegal_lal | coarse_pack_1792 | 78680 | 0.438x | 5.0 | 29.0 | 100.0000% |

# Stage 02C0 — Representation Geometry Audit

Tokenizer-only audit. No model forward pass.

| Model | Representation | Vectors | Ratio vs B4 | Median vec/doc | P95 vec/doc | Token coverage |
|---|---|---:|---:|---:|---:|---:|
| aiteamvn_v1 | current_atomic_truncate_2048 | 179737 | 1.000x | 14.0 | 62.0 | 71.4234% |
| aiteamvn_v1 | atomic_split_2048 | 195050 | 1.085x | 16.0 | 65.0 | 100.0000% |
| aiteamvn_v1 | coarse_pack_1024 | 109982 | 0.612x | 7.0 | 42.0 | 100.0000% |
| aiteamvn_v1 | coarse_pack_1792 | 62790 | 0.349x | 4.0 | 23.0 | 100.0000% |
| vnlegal_lal | current_atomic_truncate_2048 | 179737 | 1.000x | 14.0 | 62.0 | 100.0000% |
| vnlegal_lal | atomic_split_2048 | 179737 | 1.000x | 14.0 | 62.0 | 100.0000% |
| vnlegal_lal | coarse_pack_1024 | 71444 | 0.397x | 5.0 | 25.0 | 100.0000% |
| vnlegal_lal | coarse_pack_1792 | 37122 | 0.207x | 3.0 | 13.0 | 100.0000% |

Selection of a representation happens only after reviewing fragmentation,
coverage, and estimated embedding cost from this audit.

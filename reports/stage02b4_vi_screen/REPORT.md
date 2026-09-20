# Stage 02B4 — Vietnamese-first fixed-geometry encoder screen

Status: **COMPLETE**

Same 179,737 moderate regions for every model; one vector per region;
no extra splitting/overlap; exact max aggregation to parent.

| Model | Params | Doc max | R@5 | R@20 | R@100 | Full@100 |
|---|---:|---:|---:|---:|---:|---:|
| vietlegal_e5 | nanB | 512 | 0.832749 | 0.920903 | 0.967754 | 96.1379% |
| vnlegal_lal | 0.596B | 2048 | 0.857495 | 0.941770 | 0.979941 | 97.3823% |
| aiteamvn_v1 | 0.568B | 2048 | 0.884833 | 0.945649 | 0.979662 | 97.4539% |
| aiteamvn_v2 | 0.568B | 2048 | 0.876465 | 0.939153 | 0.975354 | 96.9961% |
| vietnamese_legal_embedding | 0.278B | 512 | 0.818302 | 0.918443 | 0.966793 | 95.9233% |
| vietlegal_harrier | 0.596B | 512 | 0.838078 | 0.926720 | 0.969780 | 96.3525% |

No model is promoted automatically from this report.
Next step uses standalone + pairwise rescue behavior to choose encoders
for representation/packing experiments.

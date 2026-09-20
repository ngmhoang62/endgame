# Stage 02B5 — Cached Acquisition Oracle

**No models loaded; no embeddings recomputed.**

Core: `AITeamVN-v1 + VNLegal-LAL + Harrier + BM25`.

- Macro oracle Recall@100: **0.993170**
- Full-gold query coverage: **98.9844%**
- Gold occurrence coverage: **98.9378%**
- Mean unique pool size: **234.4**
- Queries not fully covered: **71**

## Proposed-kill challenge @100

| Challenger | Δ macro recall | Query rescues | Gold rescues | Full-query rescues |
|---|---:|---:|---:|---:|
| aiteamvn_v2 | +0.000727 | 9 | 9 | 8 |
| vietlegal_e5 | +0.000167 | 3 | 3 | 3 |
| vietnamese_legal_embedding | +0.000596 | 6 | 6 | 6 |

No source is promoted automatically from this oracle analysis.

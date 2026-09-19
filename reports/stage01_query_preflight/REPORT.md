# Stage 01B Query Preflight

**Status: PASS**

| Population | N | Median words | Empty | Strict dup groups | Accent-fold dup groups |
|---|---:|---:|---:|---:|---:|
| PRIVATE | 2080 | 19.0 | 0 | 0 | 0 |
| TRAIN | 7000 | 19.0 | 0 | 16 | 16 |
| PUBLIC | 1000 | 19.0 | 0 | 0 | 0 |

## Cross-population overlap

### PRIVATE_vs_TRAIN
- QID intersection: 0
- strict: 6 (0.29%)
- punctuation: 7 (0.34%)
- accent: 7 (0.34%)

### PRIVATE_vs_PUBLIC
- QID intersection: 0
- strict: 0 (0.00%)
- punctuation: 0 (0.00%)
- accent: 0 (0.00%)

### PUBLIC_vs_TRAIN
- QID intersection: 0
- strict: 5 (0.50%)
- punctuation: 5 (0.50%)
- accent: 5 (0.50%)

## Legacy preprocessing reference

- Available: no (**non-blocking**)
- Reason: legacy preprocessing artifacts unavailable or inaccessible
- manifest: `PermissionError: [WinError 5] Access is denied: 'D:\\Study\\DSC2026\\LegalIR\\cache\\final_preprocessed_v2\\manifest.json'`
- exclusions: `PermissionError: [WinError 5] Access is denied: 'D:\\Study\\DSC2026\\LegalIR\\cache\\final_preprocessed_v2\\exclusions.json'`

## Important contract

- Model input text is **not aggressively normalized**.
- Accent/punctuation folding exists only to detect contamination/duplicates.
- No query is removed or merged by this stage.
- Legacy LegalIR preprocessing is optional historical evidence, never a Stage 01B dependency.
- PRIVATE labels used: **false**.

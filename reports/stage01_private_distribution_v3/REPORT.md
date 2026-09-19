# Stage 01B v3.1 — Private Distribution Diagnostic

**Fresh ENDGAME model only. Diagnostic, not evaluation protocol.**

- Model: `mainguyen9/vietlegal-e5`
- Hub SHA: `a814728d93e14566f9634b50a054e67699ea8818`
- Historical model/cache reused: **false**
- `fix_mistral_regex=True`: **enabled**
- Canonical vector postprocess: **explicit float32 L2 normalization**

## Nearest-neighbor summary

| Reference | Lexical median | Embedding median | Embedding >= .95 |
|---|---:|---:|---:|
| TRAIN | 0.5256 | 0.6689 | 43 |
| PUBLIC | 0.3935 | 0.5116 | 4 |
| HIST_CAL_A | 0.2590 | 0.3395 | 2 |
| HIST_CAL_B | 0.2517 | 0.3336 | 1 |
| HIST_CAL_C | 0.2576 | 0.3419 | 1 |
| HIST_CAL_D | 0.3145 | 0.4173 | 2 |

## Scientific status

- Stage 01B v2 query preflight remains valid.
- Stage 01B v2 exact/lexical diagnostics remain valid.
- Stage 01B v2 embedding similarities and clusters remain invalidated.
- This v3.1 report is the authoritative semantic-distribution diagnostic.

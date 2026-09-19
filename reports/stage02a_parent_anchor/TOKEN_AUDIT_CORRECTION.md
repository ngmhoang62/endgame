# Stage 02A token-audit correction

**Original `CORPUS_TOKEN_AUDIT.json`: INVALID for length conclusions.**

Reason: the backend tokenizer had active truncation at 512, so the old
audit measured already-clipped sequences.

The retrieval scores from Stage 02A remain valid because model inference
was intentionally performed at max length 512.

Authoritative token-length artifact:

`CORPUS_TOKEN_AUDIT_V2.json`

## Corrected headline

- Documents >512: **8506 / 8512 (99.93%)**
- Unique gold documents >512: **3098 / 3099 (99.97%)**
- Gold occurrences >512: **7624 / 7626 (99.97%)**
- Queries with any gold >512: **6989 / 6991 (99.97%)**
- Queries with all gold >512: **6989 / 6991 (99.97%)**

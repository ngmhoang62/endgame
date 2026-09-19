# DSC2026 Vietnamese Legal IR — ENDGAME

Historical repositories are read-only evidence. New runtime assets live under
ENDGAME.

All commands use **Bash / Git Bash**.

## Canonical raw snapshot

```text
data/official_v1/
  train.json
  public-official.json
  private-official.json
  DATA_SNAPSHOT_MANIFEST.json
  selected-contexts/        # local only / gitignored
```

The old root-level `private-official.json` is no longer canonical.

## Stage 01

Private-distribution diagnostics are complete. They are diagnostic only and do
not define the validation protocol.

## Stage 00B

Raw-data forensics established:

- 7,000 train queries;
- 8,532 raw documents;
- 20 empty-passage documents;
- zero missing gold document IDs;
- 11 queries touch an empty gold document;
- 4 exact duplicate-passage groups;
- 16 exact duplicate-query groups.

## Stage 00C v2 — rebuilt retrieval/evaluation universe

The old LegalIR preprocessing is treated as evidence, not copied blindly.

ENDGAME policy:

```text
RAW_CORPUS_8532
  - remove 20 empty-passage docs
  = RETRIEVAL_CORPUS_8512
```

Exact duplicate-content documents remain as separate literal document IDs.
They are **not** collapsed or remapped because official scoring is ID-literal
and retaining five extra duplicate docs has negligible compute cost.

For train labels:

```text
RAW_TRAIN_7000
  - remove empty-passage gold occurrences
  - exclude only queries with no retrievable gold left
  = PRIMARY_EVAL_6991
```

Stage 00C v2 is now **frozen** as the primary OOF contract:

- 6,991 primary evaluable queries;
- 9 non-evaluable queries;
- 8,512 retrievable documents;
- 5 duplicate-safe grouped folds;
- fold sizes 1,397–1,399;
- selected fold seed: 276;
- all population, coverage, group-isolation, and balance gates PASS.

Primary local development metric: macro Recall@5 over `PRIMARY_EVAL_6991`.
Secondary precision uses the official variable-K denominator.

The authoritative metric implementation is:

```text
src/common/evaluation.py
```

## Stage 02A — raw-parent acquisition anchor

Before rebuilding document chunking, establish a zero-training acquisition
anchor under the frozen Stage 00C v2 protocol.

Run:

```bash
python src/stage02_candidate_generation/run_parent_anchor.py
```

The stage performs:

- exact untruncated tokenizer-length audit for all 8,512 parent passages;
- fresh VietLegal-E5 query/document embeddings;
- `query: ` prefix for queries and `passage: ` prefix for passages;
- exact dense cosine search to top 100;
- deterministic parent-level BM25 to top 100;
- fixed RRF60 dense+BM25;
- dense+BM25 candidate-union oracle at K = 1/5/10/20/50/100;
- exact-duplicate-content literal-ID candidate expansion;
- per-fold and stress-slice diagnostics;
- missing-gold forensics.

Large runtime artifacts remain local:

```text
cache/stage02a_parent_anchor/
```

Version the reports:

```text
reports/stage02a_parent_anchor/
  CORPUS_TOKEN_AUDIT.json
  BASELINE_ANCHOR.json
  QUERY_RETRIEVAL_AUDIT.csv
  MISSING_GOLD_FORENSICS.json
  REPORT.md
```

This stage is an acquisition anchor, not a final submission candidate. Its
main purpose is to determine whether raw-parent truncation/candidate coverage
justifies Stage 02B structure-aware article/section chunking.

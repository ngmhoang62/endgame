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

The expected 6,991 / 9 counts are verified from raw ENDGAME data at runtime;
historical exclusion artifacts are not used to manufacture them.

Run:

```bash
python src/stage00_audit/build_endgame_evaluation_v2.py
```

Expected PASS headline:

```text
raw_queries       = 7000
primary_queries   = 6991
non_evaluable     = 9
raw_docs          = 8532
empty_docs        = 20
retrieval_docs    = 8512
```

Primary outputs:

```text
data/evaluation_v2/
  retrieval_corpus_8512.jsonl   # local-only; large
  primary_golds_6991.json
  non_evaluable_9.json
  empty_gold_affected_evaluable.json
  empty_document_ids.json
  exact_duplicate_passage_groups.json
  folds_v2.json
  qid_to_fold.json
  group_id_by_qid.json
  stress_slices.json
  EVALUATION_CONTRACT.json

reports/stage00c_evaluation_protocol_v2/
  FOLD_AUDIT.json
  FOLD_ASSIGNMENTS.csv
  SEED_SEARCH.json
  REPORT.md
```

Primary folds use a label-free leakage grouping rule:

```text
exact/accent-normalized duplicate
OR embedding cosine >= 0.95
OR (embedding cosine >= 0.94 AND lexical cosine >= 0.80)
```

Gold overlap is audit-only and never selects a grouping edge.

The authoritative metric implementation is:

```text
src/common/evaluation.py
```

Primary local development metric: macro Recall@5 over `PRIMARY_EVAL_6991`.
Secondary precision uses the official variable-K denominator.

`RAW_TRAIN_7000` remains a secondary diagnostic population rather than the
model-promotion population.

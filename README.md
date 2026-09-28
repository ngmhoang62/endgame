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

## GOLD_PROP_AIT_W75_K5 local-CV replay

This section concerns the **6,991-query local OOF evaluation**, not the
private leaderboard score or regeneration of the submission ZIP. The locked
historical result is Recall@5 `0.9505364039479332`, multi-gold Recall@5
`0.7680580762250454`, nested fold-weighted Recall@5 `0.9502503218423687`,
and robust fusion weight `w_prop=0.75`. The immutable evidence is
`reports/stage07y_final_clean_fusion/FUSION_AUDIT.json`.

The GitHub repository intentionally does not contain model weights, the
official `selected-contexts/` corpus, embeddings, OOF arrays, checkpoints, or
submissions. To rebuild them on Windows x64 with Python 3.11, a CUDA GPU,
internet access to Hugging Face, and the official competition data:

```bash
git clone https://github.com/ngmhoang62/endgame.git
cd endgame
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-gold-local-lock.txt
```

Download the **official competition** `selected-contexts/` directory and
place its 8,532 `context_*.json` files under
`data/official_v1/selected-contexts/`. The three official JSON splits are
already in Git. Do not use a processed corpus or a copy from `sota` or
`LegalIR`. The repository's `DATA_SNAPSHOT_MANIFEST.json` fixes every raw
file's size and SHA-256; the preflight rejects missing or differing inputs:

```bash
python reproduce_gold_local_cv.py check-data
python reproduce_gold_local_cv.py run
```

`run` executes Stage00C, materializes four public models at their historical
Hugging Face commits, rebuilds geometry, retrieval sources and representations,
recomputes the Stage03B1 OOF reranker, derives raw AIT parent centroids from
Stage02B4 embeddings, fits the fold-clean Stage07R/Stage07O selectors, and
recomputes the Stage07Y local fusion audit. It never scores the private split.
The model commits are E5 `a814728d93e14566f9634b50a054e67699ea8818`,
VNLegal-LAL `de759324ef931a2475ae8db97137b6a6cbb98aa0`, AIT embedding
`dea33aa1ab339f38d66ae0a40e6c40e0a9249568`, and AIT reranker
`f536976248403314225d7fdfdbc87f0e9516a54e`. The cache path named
`stage07d_teacher_distill_head` holds only the raw, teacher-free centroid
derivative in this replay. No Stage07D teacher output is used.

After a partial run, `python reproduce_gold_local_cv.py verify-cv` recomputes
the fusion metric from the generated OOF arrays and fails if any locked metric
or the selected weight differs. Reading the committed historical report is
**not** accepted as reproduction. The full run requires substantial GPU time
and disk; keep `cache/`, `models/`, and `submissions/` untracked.

**Exactness limit (2026-09-28):** The historical OOF arrays pass the exact
check, but exact replay from source has **not** been demonstrated. In a
controlled Stage07O retrain using the original local centroid and upstream
cache, the centroid-head Recall@5 was `0.9463762933295189` rather than the
historical `0.9473537405235303`; the recomputed robust fusion chose `0.80`
and reached `0.9501072807895867`. A fresh centroid build also differed at
floating-point byte level. The GPU aggregation/training path is not
bitwise-deterministic, so the current repository supplies a complete
provenance-checked replay route and a fail-closed exact metric test, **not** a
verified guarantee of the exact historical CV result on a clean clone.

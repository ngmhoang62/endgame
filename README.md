# DSC2026 Vietnamese Legal IR — ENDGAME

Private-test endgame workspace.

Historical repositories are **read-only evidence**:

- `ngmhoang62/DSC2026-LegalIR-Huy`
- `ngmhoang62/DSC2026-LegalIR`

New runtime assets belong to ENDGAME only:

```text
endgame/
  data/
  models/
  cache/
  reports/
  results/
```

Do not use a live model/index/cache path from `sota/` or `LegalIR/` in a new
ENDGAME experiment. Historical artifacts may be inspected as evidence, but new
runtime dependencies must be materialized and fingerprinted under ENDGAME.

All command examples use **Bash / Git Bash**.

## Environment

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

## Stage 01A — private-only audit

```bash
python src/stage01_query_analysis/audit_private_queries.py
```

## Stage 01B preflight

```bash
python src/stage01_query_analysis/preflight_query_data.py \
  --sota-root D:/Study/DSC2026/sota \
  --legalir-root D:/Study/DSC2026/LegalIR
```

The historical repositories are used here only as source/reference locations.

## Fresh model materialization

```bash
python src/stage00_audit/materialize_hf_model.py
```

Current authoritative VietLegal-E5 materialization:

- source: `mainguyen9/vietlegal-e5`
- immutable Hub SHA: `a814728d93e14566f9634b50a054e67699ea8818`
- local model: `models/vietlegal-e5/`
- tokenizer: `XLMRobertaTokenizerFast`
- `fix_mistral_regex=True`
- canonical embedding postprocess: explicit float32 L2 normalization

SentenceTransformers 5.7 renamed
`get_sentence_embedding_dimension()` to `get_embedding_dimension()`. New code
should use the latter. This warning does not invalidate the existing model
materialization.

## Stage 01B v3.1 — semantic distribution diagnostic

```bash
python src/stage01_query_analysis/compare_private_distribution_v3.py \
  --sota-root D:/Study/DSC2026/sota
```

Authoritative report:

```text
reports/stage01_private_distribution_v3/
```

Scientific interpretation:

- Stage 01B is diagnostic only.
- Historical CAL600 is not the ENDGAME evaluation protocol.
- Raw nearest-neighbor similarity against TRAIN/PUBLIC/CAL populations of
  different sizes must not be compared as a direct distribution-distance
  measure because nearest-neighbor maxima depend on reference-set cardinality.
- The private embedding clustering has weak separation; it is not currently a
  sufficient basis for query routing.

## Stage 00B — evaluation protocol rebuild

The historical CAL600 and historical `cv_folds.json` are reference artifacts,
not ENDGAME validation truth.

The starting hypothesis for the new primary protocol is **all 7,000 official
train queries with literal official document IDs**. We do not inherit the old
6,991-query canonicalized population without re-auditing it, because official
evaluation is literal-ID based and historical duplicate/alias collapse hurt
Recall.

### 00B.1 Snapshot official data into ENDGAME

```bash
python src/stage00_audit/snapshot_official_data.py \
  --sota-root D:/Study/DSC2026/sota
```

Creates:

```text
data/official_v1/
reports/stage00_data_snapshot/
```

After this succeeds, future ENDGAME stages should read TRAIN/PUBLIC/corpus from
`data/official_v1/`, not from the live `sota/` tree.

### 00B.2 Evaluation-population forensics

```bash
python src/stage00_audit/audit_evaluation_population.py
```

Creates:

```text
reports/stage00b_evaluation_population/
  EVALUATION_POPULATION_AUDIT.json
  NEAR_DUPLICATE_CANDIDATES.csv
  QUERY_AUDIT.csv
  REPORT.md
```

This stage deliberately does **not** create folds. It audits:

- raw official labels and answer-size distribution;
- missing/empty corpus documents;
- exact duplicate document passages;
- exact/punctuation/accent query duplicates;
- fresh VietLegal-E5 semantic near-neighbor graph;
- gold-frequency strata;
- near-duplicate graph sensitivity over cosine thresholds 0.90–0.99.

Only after inspecting these artifacts will Stage 00C freeze a duplicate-safe
full-population fold algorithm.

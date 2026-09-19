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
ENDGAME experiment. Historical artifacts may be inspected or explicitly
imported as evidence, but a new runtime dependency must be materialized and
fingerprinted under ENDGAME.

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

The old repositories are still used here only to locate raw historical
TRAIN/PUBLIC inputs and optional preprocessing evidence. Runtime model/cache
reuse is forbidden.

## Fresh model materialization

Before any new neural diagnostic or retrieval experiment, materialize the model
inside ENDGAME:

```bash
python src/stage00_audit/materialize_hf_model.py
```

This downloads a fresh, immutable Hub revision of `mainguyen9/vietlegal-e5`
to:

```text
models/vietlegal-e5/
```

and redirects Hugging Face caches under:

```text
cache/huggingface/
cache/sentence_transformers/
```

Version the small provenance files:

```text
reports/stage00_model_materialization/vietlegal-e5/
```

## Stage 01B v3 — semantic distribution diagnostic

Stage 01B v2's embedding/clustering component is invalidated because the
historical tokenizer emitted an incorrect-regex warning. Its preprocessing,
exact-overlap and lexical results remain valid.

Run the authoritative semantic diagnostic:

```bash
python src/stage01_query_analysis/compare_private_distribution_v3.py \
  --sota-root D:/Study/DSC2026/sota
```

New neural cache:

```text
cache/stage01_query_analysis/private_distribution_v3/
```

Authoritative semantic report:

```text
reports/stage01_private_distribution_v3/
```

Historical CAL600 is diagnostic only. ENDGAME will build a new full-population,
duplicate-safe OOF evaluation protocol before promotion decisions.

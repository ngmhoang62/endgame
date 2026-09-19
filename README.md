# DSC2026 Vietnamese Legal IR — ENDGAME

Private-test endgame workspace.

Historical references:

- `ngmhoang62/DSC2026-LegalIR-Huy` — D1/public-phase lineage.
- `ngmhoang62/DSC2026-LegalIR` — structural corpus / full-train research lineage.

All command examples use **Bash / Git Bash**.

## Environment

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify CUDA before any GPU stage:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

## Stage 01A — private-only audit

```bash
python src/stage01_query_analysis/audit_private_queries.py
```

## Stage 01B — distribution diagnostic

**Important:** Stage 01B is diagnostic only. Historical CAL600 is not the
ENDGAME evaluation protocol.

First run the preprocessing/data-contract gate:

```bash
python src/stage01_query_analysis/preflight_query_data.py \
  --sota-root D:/Study/DSC2026/sota \
  --legalir-root D:/Study/DSC2026/LegalIR
```

Do not continue unless the result is `PASS`.

Then run the semantic distribution comparison:

```bash
python src/stage01_query_analysis/compare_private_distribution.py \
  --sota-root D:/Study/DSC2026/sota
```

Version these small artifacts:

```text
reports/stage01_query_preflight/
reports/stage01_private_distribution/
```

Keep embeddings/local caches under `cache/`; do not commit them.

## Evaluation protocol

The old CAL600 is a historical diagnostic only. ENDGAME will build a new
duplicate-safe, full-labeled-population OOF validation protocol before model
promotion decisions are made.

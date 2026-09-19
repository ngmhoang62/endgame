# DSC2026 Vietnamese Legal IR — ENDGAME

Private-test endgame workspace. New experiments live here; the two historical
repositories are read-only scientific references:

- `ngmhoang62/DSC2026-LegalIR-Huy` — public-phase D1 and late-stage forensic experiments.
- `ngmhoang62/DSC2026-LegalIR` — older/parallel structural corpus, retrieval and ranking experiments.

## Environment

Python 3.11 is recommended.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 01A — private query audit

Keep the official private query file at:

```text
private-official.json
```

Run:

```powershell
python src/stage01_query_analysis/audit_private_queries.py
```

Generated small artifacts:

```text
reports/stage01_private_query_audit/
  PRIVATE_QUERY_AUDIT.json
  PRIVATE_QUERY_FEATURES.csv
  PRIVATE_QUERY_FEATURES.jsonl
  REPORT.md
```

Commit those report files so they can be inspected remotely. Do not commit
embeddings, model weights, FAISS/HNSW indexes or large pickle caches.

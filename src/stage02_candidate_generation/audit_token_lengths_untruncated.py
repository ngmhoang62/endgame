#!/usr/bin/env python
"""Correct Stage 02A untruncated token-length audit.

The original Stage 02A audit used the tokenizer backend attached to the loaded
SentenceTransformer. That backend had truncation configured at 512, so lengths
were silently clipped and the resulting "0 docs >512" conclusion is invalid.

This script:
- loads a fresh tokenizer from ENDGAME/models/vietlegal-e5;
- explicitly disables backend truncation;
- sets a very large model_max_length for audit only;
- calls the tokenizer with truncation=False;
- never runs the embedding model;
- writes a corrected v2 token audit.

Run:
    python src/stage02_candidate_generation/audit_token_lengths_untruncated.py
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "vietlegal-e5"
EVAL = ROOT / "data" / "evaluation_v2"
OUT = ROOT / "reports" / "stage02a_parent_anchor"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values):
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "min": int(x.min()),
        "p25": float(np.quantile(x, .25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.quantile(x, .75)),
        "p90": float(np.quantile(x, .90)),
        "p95": float(np.quantile(x, .95)),
        "p99": float(np.quantile(x, .99)),
        "max": int(x.max()),
    }


def main():
    from transformers import AutoTokenizer

    corpus = []
    with (EVAL / "retrieval_corpus_8512.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                corpus.append((str(row["id"]), str(row["passage"])))
    if len(corpus) != 8512:
        raise RuntimeError(f"Expected 8512 documents, got {len(corpus)}")

    golds = {
        str(q): [str(d) for d in ds]
        for q, ds in read_json(EVAL / "primary_golds_6991.json").items()
    }
    if len(golds) != 6991:
        raise RuntimeError(f"Expected 6991 queries, got {len(golds)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=True,
        fix_mistral_regex=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer required for this audit.")

    # The old bug: backend tokenizer inherited truncation=512.
    old_backend_truncation = getattr(tokenizer.backend_tokenizer, "truncation", None)
    tokenizer.backend_tokenizer.no_truncation()

    # Prevent Transformers from treating >512 as an error/warning threshold.
    # This affects only this tokenizer instance used for counting, never model inference.
    tokenizer.model_max_length = 10**9

    lengths = {}
    batch_size = 64
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start:start + batch_size]
        texts = ["passage: " + passage for _, passage in batch]
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = enc["length"]
        for (doc_id, _), n in zip(batch, batch_lengths):
            lengths[doc_id] = int(n)
        done = min(start + batch_size, len(corpus))
        if done % 1024 == 0 or done == len(corpus):
            print(f"[token-audit-v2] {done}/{len(corpus)}", flush=True)

    if len(lengths) != 8512:
        raise RuntimeError("Length audit coverage mismatch.")

    all_gold_unique = {d for ds in golds.values() for d in ds}
    missing = all_gold_unique - set(lengths)
    if missing:
        raise RuntimeError(f"Gold docs missing from corpus: {sorted(missing)[:20]}")

    gold_occ_lengths = [lengths[d] for ds in golds.values() for d in ds]
    unique_gold_lengths = [lengths[d] for d in all_gold_unique]
    over512 = {d for d, n in lengths.items() if n > 512}

    q_any = sum(any(d in over512 for d in ds) for ds in golds.values())
    q_all = sum(all(d in over512 for d in ds) for ds in golds.values())

    payload = {
        "schema_version": "dsc2026.endgame.stage02a.token_audit.v2",
        "status": "PASS",
        "supersedes": "CORPUS_TOKEN_AUDIT.json",
        "original_audit_status": "INVALID_TRUNCATED_COUNTER",
        "bug": (
            "Original audit counted tokens using a backend tokenizer with active "
            "truncation=512, so token lengths were clipped before counting."
        ),
        "audit_contract": {
            "model_inference_max_length": 512,
            "counting_truncation": False,
            "backend_no_truncation_called": True,
            "audit_only_model_max_length": 10**9,
            "prefix": "passage: ",
            "tokenizer_class": type(tokenizer).__name__,
            "backend_truncation_before_fix": old_backend_truncation,
        },
        "all_documents": {
            "count": len(lengths),
            "token_length": describe(lengths.values()),
            "over_512_count": sum(n > 512 for n in lengths.values()),
            "over_512_rate": sum(n > 512 for n in lengths.values()) / len(lengths),
            "over_1024_count": sum(n > 1024 for n in lengths.values()),
            "over_2048_count": sum(n > 2048 for n in lengths.values()),
            "over_4096_count": sum(n > 4096 for n in lengths.values()),
        },
        "unique_primary_gold_documents": {
            "count": len(all_gold_unique),
            "token_length": describe(unique_gold_lengths),
            "over_512_count": sum(n > 512 for n in unique_gold_lengths),
            "over_512_rate": sum(n > 512 for n in unique_gold_lengths) / len(unique_gold_lengths),
            "over_1024_count": sum(n > 1024 for n in unique_gold_lengths),
        },
        "primary_gold_occurrences": {
            "count": len(gold_occ_lengths),
            "token_length": describe(gold_occ_lengths),
            "over_512_count": sum(n > 512 for n in gold_occ_lengths),
            "over_512_rate": sum(n > 512 for n in gold_occ_lengths) / len(gold_occ_lengths),
            "over_1024_count": sum(n > 1024 for n in gold_occ_lengths),
        },
        "queries": {
            "count": len(golds),
            "any_gold_over_512_count": q_any,
            "any_gold_over_512_rate": q_any / len(golds),
            "all_golds_over_512_count": q_all,
            "all_golds_over_512_rate": q_all / len(golds),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "CORPUS_TOKEN_AUDIT_V2.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Stage 02A token-audit correction",
        "",
        "**Original `CORPUS_TOKEN_AUDIT.json`: INVALID for length conclusions.**",
        "",
        "Reason: the backend tokenizer had active truncation at 512, so the old",
        "audit measured already-clipped sequences.",
        "",
        "The retrieval scores from Stage 02A remain valid because model inference",
        "was intentionally performed at max length 512.",
        "",
        "Authoritative token-length artifact:",
        "",
        "`CORPUS_TOKEN_AUDIT_V2.json`",
        "",
        "## Corrected headline",
        "",
        f"- Documents >512: **{payload['all_documents']['over_512_count']} / 8512 ({payload['all_documents']['over_512_rate']:.2%})**",
        f"- Unique gold documents >512: **{payload['unique_primary_gold_documents']['over_512_count']} / {len(all_gold_unique)} ({payload['unique_primary_gold_documents']['over_512_rate']:.2%})**",
        f"- Gold occurrences >512: **{payload['primary_gold_occurrences']['over_512_count']} / {len(gold_occ_lengths)} ({payload['primary_gold_occurrences']['over_512_rate']:.2%})**",
        f"- Queries with any gold >512: **{q_any} / {len(golds)} ({q_any/len(golds):.2%})**",
        f"- Queries with all gold >512: **{q_all} / {len(golds)} ({q_all/len(golds):.2%})**",
        "",
    ]
    (OUT / "TOKEN_AUDIT_CORRECTION.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "documents_over_512": payload["all_documents"]["over_512_count"],
        "documents_over_512_rate": payload["all_documents"]["over_512_rate"],
        "gold_occurrences_over_512": payload["primary_gold_occurrences"]["over_512_count"],
        "gold_occurrences_over_512_rate": payload["primary_gold_occurrences"]["over_512_rate"],
        "queries_any_gold_over_512": q_any,
        "queries_all_gold_over_512": q_all,
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

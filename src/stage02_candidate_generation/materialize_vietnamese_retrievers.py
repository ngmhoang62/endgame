#!/usr/bin/env python
"""Stage 02B3 — materialize Vietnamese/Vietnamese-legal retriever candidates.

This stage is setup only. It does NOT retrieve or rank models.

Fresh candidates:
- AITeamVN/Vietnamese_Embedding
- AITeamVN/Vietnamese_Embedding_v2
- bqbbao6/vietnamese-legal-embedding
- mainguyen9/vietlegal-harrier-0.6b
- bkai-foundation-models/vietnamese-bi-encoder
- contextboxai/halong_embedding

Already-materialized anchors:
- mainguyen9/vietlegal-e5
- darklethelong/vnlegal-lal

For each fresh candidate:
- resolve main -> immutable Hub SHA;
- download under ENDGAME/models/retrievers_vi/;
- hash local snapshot;
- load strictly locally;
- count actual parameters;
- record max sequence length, embedding dimension, pooling stack and prompts;
- smoke-test encode;
- for Qwen-family Harrier, enable fix_mistral_regex=True.

Run:
    python src/stage02_candidate_generation/materialize_vietnamese_retrievers.py
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT / "modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

MODEL_ROOT = ROOT / "models" / "retrievers_vi"
OUT = ROOT / "reports" / "stage02b3_vietnamese_model_materialization"

CANDIDATES = {
    "aiteamvn_vietnamese_embedding": {
        "repo": "AITeamVN/Vietnamese_Embedding",
        "dirname": "aiteamvn-vietnamese-embedding",
        "trust_remote_code": False,
        "fix_mistral_regex": False,
        "expected_max": 2048,
        "scientific_role": "Vietnamese retrieval fine-tune",
    },
    "aiteamvn_vietnamese_embedding_v2": {
        "repo": "AITeamVN/Vietnamese_Embedding_v2",
        "dirname": "aiteamvn-vietnamese-embedding-v2",
        "trust_remote_code": False,
        "fix_mistral_regex": False,
        "expected_max": 2048,
        "scientific_role": "larger-data Vietnamese retrieval fine-tune",
    },
    "vietnamese_legal_embedding": {
        "repo": "bqbbao6/vietnamese-legal-embedding",
        "dirname": "vietnamese-legal-embedding",
        "trust_remote_code": True,
        "fix_mistral_regex": False,
        "expected_max": 512,
        "scientific_role": "Vietnamese legal retrieval fine-tune",
    },
    "vietlegal_harrier_0_6b": {
        "repo": "mainguyen9/vietlegal-harrier-0.6b",
        "dirname": "vietlegal-harrier-0.6b",
        "trust_remote_code": False,
        "fix_mistral_regex": True,
        "expected_max": 512,
        "scientific_role": "Vietnamese legal Qwen/Harrier fine-tune",
    },
    "bkai_vietnamese_bi_encoder": {
        "repo": "bkai-foundation-models/vietnamese-bi-encoder",
        "dirname": "bkai-vietnamese-bi-encoder",
        "trust_remote_code": False,
        "fix_mistral_regex": False,
        "expected_max": None,
        "scientific_role": "PhoBERT Vietnamese bi-encoder with Zalo legal training",
    },
    "halong_embedding": {
        "repo": "contextboxai/halong_embedding",
        "dirname": "halong-embedding",
        "trust_remote_code": False,
        "fix_mistral_regex": False,
        "expected_max": 512,
        "scientific_role": "Vietnamese-focused compact embedding",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path):
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".cache/") or rel == "_ENDGAME_MODEL_MANIFEST.json":
            continue
        rows.append({
            "path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return rows


def fingerprint(rows):
    raw = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_np(x):
    import numpy as np
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise RuntimeError(f"Bad embeddings: {a.shape}")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n <= 0) or not np.isfinite(n).all():
        raise RuntimeError("Bad embedding norms.")
    a /= n
    return a


def module_summary(model):
    rows = []
    try:
        for name, module in model._modules.items():
            rows.append({
                "name": str(name),
                "class": type(module).__name__,
                "repr": repr(module)[:2000],
            })
    except Exception:
        pass
    return rows


def materialize(key, spec, force=False):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer

    target = MODEL_ROOT / spec["dirname"]
    report_dir = OUT / key
    report_dir.mkdir(parents=True, exist_ok=True)

    sha = str(HfApi().model_info(spec["repo"], revision="main").sha)
    if len(sha) < 20:
        raise RuntimeError(f"Could not resolve SHA for {spec['repo']}")

    if target.exists():
        local_manifest = target / "_ENDGAME_MODEL_MANIFEST.json"
        if force:
            shutil.rmtree(target)
        elif local_manifest.exists():
            old = json.loads(local_manifest.read_text(encoding="utf-8"))
            if old.get("resolved_revision_sha") == sha:
                print(f"[{key}] immutable snapshot already materialized", flush=True)
            else:
                raise RuntimeError(
                    f"{key}: local SHA {old.get('resolved_revision_sha')} != Hub {sha}; "
                    "use --force deliberately."
                )
        else:
            raise RuntimeError(
                f"{target} exists without ENDGAME manifest; refusing to mix files."
            )

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{spec['dirname']}-", dir=str(target.parent)))
        try:
            print(f"[{key}] download {spec['repo']}@{sha}", flush=True)
            snapshot_download(repo_id=spec["repo"], revision=sha, local_dir=str(temp))
            local_cache = temp / ".cache"
            if local_cache.exists():
                shutil.rmtree(local_cache, ignore_errors=True)
            temp.replace(target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    rows = inventory(target)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    tokenizer_kwargs = {}
    if spec["fix_mistral_regex"]:
        tokenizer_kwargs["fix_mistral_regex"] = True

    print(f"[{key}] local load", flush=True)
    model = SentenceTransformer(
        str(target),
        device="cuda",
        local_files_only=True,
        trust_remote_code=spec["trust_remote_code"],
        model_kwargs={"torch_dtype": torch.float16},
        tokenizer_kwargs=tokenizer_kwargs or None,
    )

    max_seq = int(model.max_seq_length)
    expected = spec["expected_max"]
    max_check = {
        "expected": expected,
        "observed": max_seq,
        "matches_expected": expected is None or int(expected) == max_seq,
    }

    # Do not fail solely on metadata mismatch here. Record it for adapter design.
    params = int(sum(p.numel() for p in model.parameters()))
    dim = int(model.get_embedding_dimension())
    prompts = dict(getattr(model, "prompts", {}) or {})
    default_prompt_name = getattr(model, "default_prompt_name", None)

    examples = [
        "Điều kiện để người lao động được hưởng trợ cấp thất nghiệp là gì?",
        "Điều 49 quy định điều kiện hưởng trợ cấp thất nghiệp của người lao động.",
    ]
    emb = normalize_np(model.encode(
        examples,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ))
    cosine = float(emb[0] @ emb[1])

    manifest = {
        "schema_version": "dsc2026.endgame.vietnamese_retriever_materialization.v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_key": key,
        "scientific_role": spec["scientific_role"],
        "model_id": spec["repo"],
        "resolved_revision_sha": sha,
        "target_path": str(target.resolve()),
        "historical_model_reused": False,
        "parameter_count": params,
        "parameter_billions": params / 1e9,
        "competition_budget_fraction_if_used": params / 4_000_000_000,
        "embedding_dimension": dim,
        "max_seq_length": max_seq,
        "max_sequence_length_card_check": max_check,
        "fix_mistral_regex": bool(spec["fix_mistral_regex"]),
        "trust_remote_code": bool(spec["trust_remote_code"]),
        "sentence_transformer_prompts": prompts,
        "default_prompt_name": default_prompt_name,
        "module_summary": module_summary(model),
        "smoke": {
            "shape": list(emb.shape),
            "cosine": cosine,
        },
        "content_fingerprint": fingerprint(rows),
        "files": rows,
        "file_count": len(rows),
    }

    (target / "_ENDGAME_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def existing_anchor(path, report_path, key):
    if not report_path.exists():
        return {"candidate_key": key, "status": "MISSING"}
    m = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "candidate_key": key,
        "status": m.get("status"),
        "model_id": m.get("model_id"),
        "resolved_revision_sha": m.get("resolved_revision_sha"),
        "parameter_count": m.get("parameter_count"),
        "parameter_billions": m.get("parameter_billions"),
        "max_seq_length": m.get("max_seq_length") or m.get("model_contract", {}).get("max_seq_length"),
        "embedding_dimension": m.get("embedding_dimension") or m.get("model_contract", {}).get("embedding_dimension"),
        "target_path": str(path.resolve()),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    passed, failed = {}, {}

    for key, spec in CANDIDATES.items():
        try:
            m = materialize(key, spec, args.force)
            passed[key] = {
                "model_id": m["model_id"],
                "resolved_revision_sha": m["resolved_revision_sha"],
                "parameter_count": m["parameter_count"],
                "parameter_billions": m["parameter_billions"],
                "max_seq_length": m["max_seq_length"],
                "embedding_dimension": m["embedding_dimension"],
                "scientific_role": m["scientific_role"],
            }
            print(json.dumps({
                "candidate": key,
                "status": "PASS",
                "params_b": m["parameter_billions"],
                "max_seq": m["max_seq_length"],
                "dim": m["embedding_dimension"],
            }, ensure_ascii=False, indent=2))
        except Exception as exc:
            failed[key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(json.dumps({
                "candidate": key, "status": "FAIL", "error": str(exc)
            }, ensure_ascii=False, indent=2))
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    anchors = {
        "vietlegal_e5": existing_anchor(
            ROOT / "models/vietlegal-e5",
            ROOT / "reports/stage00_model_materialization/vietlegal-e5/MODEL_MANIFEST.json",
            "vietlegal_e5",
        ),
        "vnlegal_lal": existing_anchor(
            ROOT / "models/retrievers/vnlegal-lal",
            ROOT / "reports/stage02b1_model_materialization/vnlegal_lal/MODEL_MANIFEST.json",
            "vnlegal_lal",
        ),
    }

    total_fresh = sum(x["parameter_count"] for x in passed.values())
    registry = {
        "schema_version": "dsc2026.endgame.vietnamese_retriever_registry.v1",
        "status": "PASS" if not failed else "PARTIAL_FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "Vietnamese/legal-first screening registry; materialization is not promotion"
        ),
        "competition_parameter_limit": 4_000_000_000,
        "fresh_candidates": passed,
        "existing_anchors": anchors,
        "failed": failed,
        "sum_fresh_candidate_parameters_if_combined": total_fresh,
        "note": (
            "Screening models are run sequentially. Competition parameter budget "
            "applies to the final active pipeline, not to the number of models screened."
        ),
    }
    (OUT / "VIETNAMESE_RETRIEVER_REGISTRY.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": registry["status"],
        "passed": list(passed),
        "failed": list(failed),
        "anchors": anchors,
        "report": str((OUT / "VIETNAMESE_RETRIEVER_REGISTRY.json").resolve()),
    }, ensure_ascii=False, indent=2))

    raise SystemExit(0 if not failed else 2)


if __name__ == "__main__":
    main()

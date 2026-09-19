#!/usr/bin/env python
"""Stage 02B1 prerequisite — fresh candidate retriever materialization.

Downloads each candidate from Hugging Face into ENDGAME, pins an immutable Hub
commit SHA, fingerprints the local snapshot, loads it locally, counts actual
parameters, verifies native max sequence length, and runs a short query/document
embedding smoke test.

This stage does NOT retrieve and does NOT choose a winner.

Candidates:
- VNLegal-LAL: Vietnamese legal-specific, 2048 context
- Qwen3-Embedding-0.6B: multilingual/instruction-aware, 32768 context
- BGE-M3: multilingual dense/sparse/multi-vector family, 8192 context
- GTE-multilingual-base: compact long-context multilingual encoder, 8192 context

Existing fresh VietLegal-E5 remains the historical anchor and is not redownloaded.

Run all:
    python src/stage02_candidate_generation/materialize_retriever_candidates.py

Run one:
    python src/stage02_candidate_generation/materialize_retriever_candidates.py \
      --candidate qwen3_embedding_0_6b
"""
from __future__ import annotations

import argparse
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
HF_CACHE = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

OUT = ROOT / "reports" / "stage02b1_model_materialization"
MODEL_ROOT = ROOT / "models" / "retrievers"

LEGAL_INSTRUCTION = (
    "Given a Vietnamese legal question, retrieve relevant legal passages "
    "that answer the question"
)

CANDIDATES = {
    "vnlegal_lal": {
        "repo": "darklethelong/vnlegal-lal",
        "target": "vnlegal-lal",
        "expected_max_seq_length": 2048,
        "trust_remote_code": False,
        "padding_side": "left",
        "query_mode": "manual_instruction",
        "query_prefix": f"Instruct: {LEGAL_INSTRUCTION}\nQuery: ",
        "document_prefix": "",
    },
    "qwen3_embedding_0_6b": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "target": "qwen3-embedding-0.6b",
        "expected_max_seq_length": 32768,
        "trust_remote_code": False,
        "padding_side": "left",
        "query_mode": "sentence_transformers_prompt",
        "prompt_name": "query",
        "document_prefix": "",
    },
    "bge_m3": {
        "repo": "BAAI/bge-m3",
        "target": "bge-m3",
        "expected_max_seq_length": 8192,
        "trust_remote_code": False,
        "padding_side": None,
        "query_mode": "plain",
        "document_prefix": "",
    },
    "gte_multilingual_base": {
        "repo": "Alibaba-NLP/gte-multilingual-base",
        "target": "gte-multilingual-base",
        "expected_max_seq_length": 8192,
        "trust_remote_code": True,
        "padding_side": None,
        "query_mode": "plain",
        "document_prefix": "",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".cache/"):
            continue
        rows.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def fingerprint(rows: list[dict[str, Any]]) -> str:
    raw = json.dumps(
        rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_array(x):
    import numpy as np
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got {a.shape}")
    if not np.isfinite(a).all():
        raise RuntimeError("Non-finite embeddings.")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n <= 0) or not np.isfinite(n).all():
        raise RuntimeError("Invalid embedding norms.")
    a = a / n
    final = np.linalg.norm(a, axis=1)
    if float(np.max(np.abs(final - 1.0))) > 2e-6:
        raise RuntimeError("Explicit float32 L2 normalization failed.")
    return a


def smoke_encode(model, spec):
    query = "Người lao động được hưởng trợ cấp thất nghiệp trong điều kiện nào?"
    document = (
        "Điều 49. Điều kiện hưởng trợ cấp thất nghiệp. "
        "Người lao động đang đóng bảo hiểm thất nghiệp được hưởng trợ cấp "
        "khi đáp ứng các điều kiện theo quy định của pháp luật."
    )

    mode = spec["query_mode"]
    if mode == "manual_instruction":
        q = model.encode(
            [spec["query_prefix"] + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    elif mode == "sentence_transformers_prompt":
        prompt_name = spec["prompt_name"]
        prompts = getattr(model, "prompts", {}) or {}
        if prompt_name not in prompts:
            raise RuntimeError(
                f"Expected SentenceTransformer prompt {prompt_name!r}; "
                f"available={sorted(prompts)}"
            )
        q = model.encode(
            [query],
            prompt_name=prompt_name,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    elif mode == "plain":
        q = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        raise RuntimeError(mode)

    d = model.encode(
        [spec.get("document_prefix", "") + document],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    q = normalize_array(q)
    d = normalize_array(d)
    return {
        "query_embedding_shape": list(q.shape),
        "document_embedding_shape": list(d.shape),
        "cosine": float((q @ d.T)[0, 0]),
        "query_norm": float((q * q).sum() ** 0.5),
        "document_norm": float((d * d).sum() ** 0.5),
    }


def materialize_one(key: str, force: bool) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch
    import sentence_transformers
    import transformers

    spec = CANDIDATES[key]
    target = (MODEL_ROOT / spec["target"]).resolve()
    report_dir = (OUT / key).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(spec["repo"], revision="main")
    resolved_sha = str(info.sha)
    if len(resolved_sha) < 20:
        raise RuntimeError(f"Could not resolve immutable SHA: {resolved_sha!r}")

    if target.exists():
        if not force:
            # Existing target is allowed only if its own local manifest proves
            # it is exactly the requested immutable snapshot.
            local_manifest = target / "_ENDGAME_MODEL_MANIFEST.json"
            if not local_manifest.exists():
                raise FileExistsError(
                    f"{target} exists without ENDGAME manifest. "
                    "Refusing to mix files; use --force deliberately."
                )
            old = json.loads(local_manifest.read_text(encoding="utf-8"))
            if old.get("resolved_revision_sha") != resolved_sha:
                raise FileExistsError(
                    f"{target} is pinned to {old.get('resolved_revision_sha')}, "
                    f"Hub main now resolves to {resolved_sha}. "
                    "Use --force only if you intentionally want the new revision."
                )
            print(f"[{key}] existing immutable snapshot accepted", flush=True)
        else:
            shutil.rmtree(target)

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(
            tempfile.mkdtemp(prefix=f".{spec['target']}-", dir=str(target.parent))
        )
        try:
            print(
                f"[{key}] download {spec['repo']} @ {resolved_sha}",
                flush=True,
            )
            snapshot_download(
                repo_id=spec["repo"],
                revision=resolved_sha,
                local_dir=str(temp),
            )
            local_hf_cache = temp / ".cache"
            if local_hf_cache.exists():
                shutil.rmtree(local_hf_cache, ignore_errors=True)
            temp.replace(target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    # Fingerprint BEFORE adding/updating our own local provenance manifest.
    rows = [
        r for r in inventory(target)
        if r["path"] != "_ENDGAME_MODEL_MANIFEST.json"
    ]
    content_fp = fingerprint(rows)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for model smoke test.")

    tokenizer_kwargs = {}
    if spec.get("padding_side"):
        tokenizer_kwargs["padding_side"] = spec["padding_side"]

    print(f"[{key}] local load + smoke test", flush=True)
    model = SentenceTransformer(
        str(target),
        device="cuda",
        local_files_only=True,
        trust_remote_code=bool(spec["trust_remote_code"]),
        model_kwargs={"torch_dtype": torch.float16},
        tokenizer_kwargs=tokenizer_kwargs or None,
    )

    max_seq = int(model.max_seq_length)
    expected = int(spec["expected_max_seq_length"])
    if max_seq != expected:
        raise RuntimeError(
            f"{key}: max_seq_length={max_seq}, expected={expected}. "
            "Do not silently cap/extend native context."
        )

    param_count = int(sum(p.numel() for p in model.parameters()))
    trainable_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    embedding_dim = int(model.get_embedding_dimension())
    smoke = smoke_encode(model, spec)

    manifest = {
        "schema_version": "dsc2026.endgame.retriever_materialization.v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_key": key,
        "model_id": spec["repo"],
        "resolved_revision_sha": resolved_sha,
        "target_path": str(target),
        "historical_model_reused": False,
        "content_fingerprint": content_fp,
        "parameter_count": param_count,
        "parameter_billions": param_count / 1e9,
        "trainable_parameter_count_at_load": trainable_count,
        "embedding_dimension": embedding_dim,
        "max_seq_length": max_seq,
        "competition_budget_fraction_if_used": param_count / 4_000_000_000,
        "query_contract": {
            "mode": spec["query_mode"],
            "prefix": spec.get("query_prefix"),
            "prompt_name": spec.get("prompt_name"),
        },
        "document_contract": {
            "prefix": spec.get("document_prefix", ""),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "smoke": smoke,
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidate",
        choices=["all", *CANDIDATES.keys()],
        default="all",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    keys = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    OUT.mkdir(parents=True, exist_ok=True)

    passed = {}
    failed = {}
    for key in keys:
        try:
            passed[key] = materialize_one(key, args.force)
            print(
                json.dumps({
                    "candidate": key,
                    "status": "PASS",
                    "params_b": passed[key]["parameter_billions"],
                    "max_seq_length": passed[key]["max_seq_length"],
                    "embedding_dim": passed[key]["embedding_dimension"],
                }, ensure_ascii=False, indent=2),
                flush=True,
            )
        except Exception as exc:
            failed[key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(
                json.dumps({
                    "candidate": key,
                    "status": "FAIL",
                    "error": str(exc),
                }, ensure_ascii=False, indent=2),
                flush=True,
            )
            # Release any partially loaded GPU state before trying next.
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    summary = {
        "schema_version": "dsc2026.endgame.retriever_candidate_registry.v1",
        "status": "PASS" if not failed else "PARTIAL_FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_pipeline_parameter_limit": 4_000_000_000,
        "note": (
            "Materializing/screening a model does not mean it belongs in the final "
            "pipeline. Final active-parameter accounting sums only promoted pipeline "
            "components, and must remain <=4B."
        ),
        "passed": {
            k: {
                "model_id": v["model_id"],
                "resolved_revision_sha": v["resolved_revision_sha"],
                "parameter_count": v["parameter_count"],
                "parameter_billions": v["parameter_billions"],
                "max_seq_length": v["max_seq_length"],
                "embedding_dimension": v["embedding_dimension"],
                "budget_fraction": v["competition_budget_fraction_if_used"],
            }
            for k, v in passed.items()
        },
        "failed": failed,
        "sum_parameters_if_all_passed_candidates_were_combined": sum(
            v["parameter_count"] for v in passed.values()
        ),
        "sum_billions_if_all_passed_candidates_were_combined": sum(
            v["parameter_count"] for v in passed.values()
        ) / 1e9,
    }
    (OUT / "CANDIDATE_REGISTRY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": summary["status"],
        "passed": list(passed),
        "failed": list(failed),
        "sum_candidate_params_b": summary[
            "sum_billions_if_all_passed_candidates_were_combined"
        ],
        "report": str((OUT / "CANDIDATE_REGISTRY.json").resolve()),
    }, ensure_ascii=False, indent=2))

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

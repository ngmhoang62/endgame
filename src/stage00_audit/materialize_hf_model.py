#!/usr/bin/env python
"""Materialize a fresh, revision-pinned Hugging Face model inside ENDGAME.

v1.1 numerical contract:
- SentenceTransformer may return unit-normalized FP16 vectors whose measured
  float32 norm differs from 1 by a few 1e-4 due to half precision.
- We audit that raw drift is <= 1e-3.
- We then re-normalize in float32 and require <= 2e-6 norm error.
- Downstream ENDGAME code uses the same explicit float32 post-normalization.

Run:
    python src/stage00_audit/materialize_hf_model.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

HF_CACHE_ROOT = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

RAW_FP16_NORM_MAX_ERROR = 1e-3
FINAL_FP32_NORM_MAX_ERROR = 2e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_inventory(root: Path) -> list[dict[str, Any]]:
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


def normalize_f32(vectors):
    import numpy as np
    x = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise RuntimeError("Non-finite embedding norm.")
    if np.any(norms <= 0):
        raise RuntimeError("Zero embedding norm.")
    return x / norms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="mainguyen9/vietlegal-e5")
    p.add_argument("--revision", default="main")
    p.add_argument("--target", type=Path, default=ROOT / "models" / "vietlegal-e5")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports" / "stage00_model_materialization" / "vietlegal-e5",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    import numpy as np
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    target = args.target.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(args.model_id, revision=args.revision)
    resolved_sha = str(info.sha)
    if not resolved_sha or len(resolved_sha) < 20:
        raise RuntimeError(f"Could not resolve immutable revision: {resolved_sha!r}")

    if target.exists():
        if not args.force:
            raise FileExistsError(
                f"Target already exists: {target}\n"
                "Refusing to mix/overwrite model files. Remove it explicitly or pass --force."
            )
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.materializing-", dir=str(target.parent))
    )

    try:
        print(f"Resolved {args.model_id}@{args.revision} -> {resolved_sha}", flush=True)
        print(f"Downloading fresh snapshot into {temp_dir}", flush=True)

        snapshot_download(
            repo_id=args.model_id,
            revision=resolved_sha,
            local_dir=str(temp_dir),
        )

        if not file_inventory(temp_dir):
            raise RuntimeError("Downloaded model snapshot is empty.")

        model = SentenceTransformer(
            str(temp_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else None,
            processor_kwargs={"fix_mistral_regex": True},
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable during model materialization smoke test.")

        tokenizer = model.tokenizer
        probes = [
            "query: Thủ tục đăng ký kinh doanh gồm những bước nào?",
            "query: Điều 27. Trình tự, thủ tục đăng ký doanh nghiệp.",
            "query: Nghị định 33/2023/NĐ-CP có hiệu lực khi nào?",
            "query: Người lao động có được hưởng trợ cấp không?",
        ]

        token_probe = []
        for text in probes:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
            ids = [int(x) for x in encoded["input_ids"]]
            if not ids:
                raise RuntimeError(f"Tokenizer returned zero tokens for probe: {text}")
            token_probe.append({
                "text": text,
                "token_count": len(ids),
                "input_ids": ids,
            })

        raw_embeddings = model.encode(
            probes,
            batch_size=4,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] != len(probes):
            raise RuntimeError(f"Unexpected embedding shape: {raw_embeddings.shape}")
        if not np.isfinite(raw_embeddings).all():
            raise RuntimeError("Non-finite embedding values.")

        raw_norms = np.linalg.norm(raw_embeddings, axis=1)
        raw_max_error = float(np.max(np.abs(raw_norms - 1.0)))
        if raw_max_error > RAW_FP16_NORM_MAX_ERROR:
            raise RuntimeError(
                "Raw SentenceTransformer normalization drift is too large: "
                f"max_error={raw_max_error:.8g}, norms={raw_norms.tolist()}"
            )

        # Canonical ENDGAME representation: explicit float32 L2 normalization.
        embeddings = normalize_f32(raw_embeddings)
        final_norms = np.linalg.norm(embeddings, axis=1)
        final_max_error = float(np.max(np.abs(final_norms - 1.0)))
        if final_max_error > FINAL_FP32_NORM_MAX_ERROR:
            raise RuntimeError(
                "Explicit float32 normalization failed: "
                f"max_error={final_max_error:.8g}, norms={final_norms.tolist()}"
            )

        dim = int(model.get_sentence_embedding_dimension())
        max_seq = int(model.max_seq_length)
        if embeddings.shape[1] != dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: output={embeddings.shape[1]} model={dim}"
            )

        local_cache = temp_dir / ".cache"
        if local_cache.exists():
            shutil.rmtree(local_cache, ignore_errors=True)

        inventory = file_inventory(temp_dir)
        content_fingerprint = canonical_hash(inventory)

        manifest = {
            "schema_version": "dsc2026.endgame.hf_model_materialization.v1.1",
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision_sha": resolved_sha,
            "source": "Hugging Face Hub fresh snapshot",
            "historical_model_reused": False,
            "target_path": str(target),
            "hf_cache_root": str(HF_CACHE_ROOT.resolve()),
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0),
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
            },
            "load_contract": {
                "library": "sentence_transformers.SentenceTransformer",
                "local_files_only": True,
                "trust_remote_code": False,
                "processor_kwargs": {"fix_mistral_regex": True},
                "dtype": "float16",
            },
            "embedding_contract": {
                "sentence_transformer_normalize_embeddings": True,
                "canonical_postprocess": "astype_float32_then_explicit_l2_normalize",
                "raw_fp16_norm_max_allowed_error": RAW_FP16_NORM_MAX_ERROR,
                "final_fp32_norm_max_allowed_error": FINAL_FP32_NORM_MAX_ERROR,
            },
            "model_contract": {
                "max_seq_length": max_seq,
                "embedding_dimension": dim,
            },
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "fix_mistral_regex": True,
                "probe": token_probe,
            },
            "embedding_probe": {
                "shape": list(embeddings.shape),
                "raw_norms": [float(x) for x in raw_norms],
                "raw_max_abs_norm_error": raw_max_error,
                "final_norms": [float(x) for x in final_norms],
                "final_max_abs_norm_error": final_max_error,
                "pairwise_cosine_after_f32_normalization":
                    (embeddings @ embeddings.T).round(8).tolist(),
            },
            "files": inventory,
            "file_count": len(inventory),
            "content_fingerprint": content_fingerprint,
        }

        (temp_dir / "_ENDGAME_MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_inventory = file_inventory(temp_dir)
        manifest["materialized_file_count_including_local_manifest"] = len(final_inventory)

        temp_dir.replace(target)

        report_path = report_dir / "MODEL_MANIFEST.json"
        report_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (report_dir / "REPORT.md").write_text(
            "\n".join([
                "# Fresh model materialization — VietLegal-E5",
                "",
                "- Status: **PASS**",
                f"- Model: `{args.model_id}`",
                f"- Hub commit: `{resolved_sha}`",
                f"- Local target: `{target}`",
                f"- Embedding dimension: **{dim}**",
                f"- Max sequence length: **{max_seq}**",
                f"- Tokenizer: `{type(tokenizer).__name__}`",
                "- `fix_mistral_regex=True`: **enabled**",
                "- Historical model reused: **false**",
                f"- Raw FP16 max norm error: **{raw_max_error:.8g}**",
                f"- Final FP32 max norm error: **{final_max_error:.8g}**",
                "- Canonical vector postprocess: **explicit float32 L2 normalization**",
                "",
            ]),
            encoding="utf-8",
        )

        print(json.dumps({
            "status": "PASS",
            "model_id": args.model_id,
            "resolved_revision_sha": resolved_sha,
            "target": str(target),
            "embedding_dimension": dim,
            "max_seq_length": max_seq,
            "raw_max_abs_norm_error": raw_max_error,
            "final_max_abs_norm_error": final_max_error,
            "content_fingerprint": content_fingerprint,
            "report": str(report_path),
        }, ensure_ascii=False, indent=2))
        return 0

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

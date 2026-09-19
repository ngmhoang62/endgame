#!/usr/bin/env python
"""Stage 02B2 — practical context-capacity probe on the local GPU.

Purpose:
Native context length != practical context length.

Probe each freshly materialized challenger with batch size 1 at increasing
sequence lengths, recording:
- whether the forward/encode succeeds;
- actual token count fed after truncation;
- elapsed seconds;
- CUDA peak allocated/reserved memory;
- OOM vs other failure.

No retrieval and no model selection occurs here.

Run:
    python src/stage02_candidate_generation/probe_retriever_context_capacity.py
"""
from __future__ import annotations

import gc
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT / "modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

MODEL_ROOT = ROOT / "models" / "retrievers"
DEP_ROOT = ROOT / "models" / "retriever_dependencies"
REPORT_ROOT = ROOT / "reports" / "stage02b1_model_materialization"
OUT = ROOT / "reports" / "stage02b2_context_capacity"

TARGETS = {
    "vnlegal_lal": [512, 1024, 1536, 2048],
    "qwen3_embedding_0_6b": [512, 1024, 2048, 4096, 8192, 16384, 32768],
    "bge_m3": [512, 1024, 2048, 4096, 8192],
    "gte_multilingual_base": [512, 1024, 2048, 4096, 8192],
}

PATHS = {
    "vnlegal_lal": MODEL_ROOT / "vnlegal-lal",
    "qwen3_embedding_0_6b": MODEL_ROOT / "qwen3-embedding-0.6b",
    "bge_m3": MODEL_ROOT / "bge-m3",
    "gte_multilingual_base": MODEL_ROOT / "gte-multilingual-base",
}

LONG_UNIT = (
    "Điều 49 quy định quyền, nghĩa vụ, điều kiện, thủ tục, thời hạn và trách nhiệm "
    "của cơ quan, tổ chức, cá nhân theo quy định của pháp luật Việt Nam. "
)


def rj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clear_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_long_text(tokenizer, required_tokens: int) -> str:
    # Grow geometrically until untruncated token count is safely above target.
    repeats = max(32, required_tokens // 18)
    while True:
        text = LONG_UNIT * repeats
        ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if len(ids) >= required_tokens + 64:
            return text
        repeats = int(repeats * 1.5) + 1


def cuda_stats():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_total_bytes": int(total),
        "peak_allocated_fraction": float(torch.cuda.max_memory_allocated() / total),
        "peak_reserved_fraction": float(torch.cuda.max_memory_reserved() / total),
    }


def probe_sentence_transformer(key, path, targets, trust_remote_code=False, code_revision=None):
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    cfg_kwargs = {}
    if code_revision:
        kwargs["code_revision"] = code_revision
        cfg_kwargs["code_revision"] = code_revision

    model = SentenceTransformer(
        str(path),
        device="cuda",
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch.float16, **kwargs},
        config_kwargs=cfg_kwargs or None,
    )
    tokenizer = model.tokenizer
    native = int(model.max_seq_length)
    results = []

    longest = max(targets)
    text = build_long_text(tokenizer, longest)

    for target in targets:
        if target > native:
            continue
        clear_cuda()
        model.max_seq_length = target
        actual = len(tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=target,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"])
        row = {"target_length": target, "actual_token_count": actual}
        t0 = time.perf_counter()
        try:
            emb = model.encode(
                [text],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            torch.cuda.synchronize()
            row.update({
                "status": "PASS",
                "elapsed_seconds": time.perf_counter() - t0,
                "embedding_shape": list(emb.shape),
                **cuda_stats(),
            })
        except torch.cuda.OutOfMemoryError as exc:
            row.update({
                "status": "OOM",
                "elapsed_seconds": time.perf_counter() - t0,
                "error": str(exc),
                **cuda_stats(),
            })
        except Exception as exc:
            row.update({
                "status": "FAIL",
                "elapsed_seconds": time.perf_counter() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                **cuda_stats(),
            })
        results.append(row)
        print(json.dumps({"model": key, **row}, ensure_ascii=False), flush=True)
        clear_cuda()

    model.max_seq_length = native
    del model
    clear_cuda()
    return native, results


def probe_lal(path, targets):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(path), local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()

    native = 2048  # frozen ENDGAME operational contract from Stage02B1 repair
    text = build_long_text(tokenizer, max(targets))
    results = []

    for target in targets:
        clear_cuda()
        batch = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=target,
            return_tensors="pt",
        ).to("cuda")
        actual = int(batch["input_ids"].shape[1])
        row = {"target_length": target, "actual_token_count": actual}
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model(**batch)
                last_idx = batch["attention_mask"].sum(dim=1) - 1
                emb = out.last_hidden_state[
                    torch.arange(1, device="cuda"), last_idx
                ]
                emb = F.normalize(emb.float(), p=2, dim=1)
            torch.cuda.synchronize()
            row.update({
                "status": "PASS",
                "elapsed_seconds": time.perf_counter() - t0,
                "embedding_shape": list(emb.shape),
                **cuda_stats(),
            })
        except torch.cuda.OutOfMemoryError as exc:
            row.update({
                "status": "OOM",
                "elapsed_seconds": time.perf_counter() - t0,
                "error": str(exc),
                **cuda_stats(),
            })
        except Exception as exc:
            row.update({
                "status": "FAIL",
                "elapsed_seconds": time.perf_counter() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                **cuda_stats(),
            })
        results.append(row)
        print(json.dumps({"model": "vnlegal_lal", **row}, ensure_ascii=False), flush=True)
        del batch
        clear_cuda()

    del model
    clear_cuda()
    return native, results


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    registry = rj(REPORT_ROOT / "CANDIDATE_REGISTRY.json")
    if registry.get("status") != "PASS":
        raise RuntimeError("Stage02B1 candidate registry is not PASS.")

    OUT.mkdir(parents=True, exist_ok=True)
    gpu = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }

    all_results = {}

    for key in ("vnlegal_lal", "qwen3_embedding_0_6b", "bge_m3", "gte_multilingual_base"):
        print(f"=== {key} ===", flush=True)
        clear_cuda()
        if key == "vnlegal_lal":
            native, rows = probe_lal(PATHS[key], TARGETS[key])
        elif key == "gte_multilingual_base":
            manifest = rj(REPORT_ROOT / key / "MODEL_MANIFEST.json")
            dep_sha = manifest["adapter_contract"]["dynamic_code_dependency"]["resolved_revision_sha"]
            native, rows = probe_sentence_transformer(
                key, PATHS[key], TARGETS[key],
                trust_remote_code=True,
                code_revision=dep_sha,
            )
        else:
            native, rows = probe_sentence_transformer(
                key, PATHS[key], TARGETS[key],
                trust_remote_code=False,
            )

        passed = [r for r in rows if r["status"] == "PASS"]
        all_results[key] = {
            "parameter_count": registry["passed"][key]["parameter_count"],
            "native_or_operational_max_length": native,
            "probe_rows": rows,
            "largest_passed_length": max(
                (r["target_length"] for r in passed), default=None
            ),
            "largest_passed_under_90pct_reserved": max(
                (
                    r["target_length"] for r in passed
                    if r["peak_reserved_fraction"] <= 0.90
                ),
                default=None,
            ),
        }

    report = {
        "schema_version": "dsc2026.endgame.stage02b2_context_capacity.v1",
        "status": "COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "operational GPU-capacity probe only; no retrieval quality conclusion"
        ),
        "gpu": gpu,
        "models": all_results,
        "decision_rule_note": (
            "Do not automatically use the largest passing length. Geometry selection "
            "must consider retrieval quality, vector count, throughput, and memory headroom."
        ),
    }
    (OUT / "CONTEXT_CAPACITY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 02B2 — Practical Context Capacity",
        "",
        f"GPU: **{gpu['name']}**",
        "",
        "| Model | Native/operational max | Largest PASS | Largest PASS <=90% reserved |",
        "|---|---:|---:|---:|",
    ]
    for key, row in all_results.items():
        lines.append(
            f"| {key} | {row['native_or_operational_max_length']} | "
            f"{row['largest_passed_length']} | "
            f"{row['largest_passed_under_90pct_reserved']} |"
        )
    lines += [
        "",
        "This report is a capacity probe only. It does not promote an encoder or context length.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "COMPLETE",
        "summary": {
            k: {
                "largest_passed": v["largest_passed_length"],
                "largest_passed_under_90pct_reserved": v[
                    "largest_passed_under_90pct_reserved"
                ],
            }
            for k, v in all_results.items()
        },
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations
import gc, hashlib, json, os, shutil, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT / "modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

MODEL_ROOT = ROOT / "models" / "retrievers"
DEP_ROOT = ROOT / "models" / "retriever_dependencies"
OUT = ROOT / "reports" / "stage02b1_model_materialization"
REGISTRY = OUT / "CANDIDATE_REGISTRY.json"

LEGAL_INSTRUCTION = "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question"
VN_PREFIX = f"Instruct: {LEGAL_INSTRUCTION}\nQuery: "

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def inventory(root: Path, exclude_manifest=True):
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".cache/"):
            continue
        if exclude_manifest and rel == "_ENDGAME_MODEL_MANIFEST.json":
            continue
        rows.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return rows

def fingerprint(rows) -> str:
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def normalize_np(x):
    import numpy as np
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise RuntimeError(f"Invalid embedding array: {a.shape}")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n <= 0) or not np.isfinite(n).all():
        raise RuntimeError("Invalid embedding norms.")
    a /= n
    err = float(np.max(np.abs(np.linalg.norm(a, axis=1) - 1.0)))
    if err > 2e-6:
        raise RuntimeError(f"Float32 normalization failed: {err}")
    return a

def resolve_repo_sha(repo: str) -> str:
    from huggingface_hub import HfApi
    sha = str(HfApi().model_info(repo, revision="main").sha)
    if len(sha) < 20:
        raise RuntimeError(f"Could not resolve SHA for {repo}: {sha!r}")
    return sha

def common_manifest(*, key, repo, target, resolved_sha, rows, params, dim, max_seq, adapter, smoke, extra=None):
    import torch, transformers, sentence_transformers
    out = {
        "schema_version": "dsc2026.endgame.retriever_materialization.v1.1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_key": key,
        "model_id": repo,
        "resolved_revision_sha": resolved_sha,
        "target_path": str(target.resolve()),
        "historical_model_reused": False,
        "content_fingerprint": fingerprint(rows),
        "parameter_count": int(params),
        "parameter_billions": int(params) / 1e9,
        "embedding_dimension": int(dim),
        "max_seq_length": int(max_seq),
        "competition_budget_fraction_if_used": int(params) / 4_000_000_000,
        "adapter_contract": adapter,
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
    if extra:
        out.update(extra)
    return out

def repair_vnlegal():
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    key = "vnlegal_lal"
    repo = "darklethelong/vnlegal-lal"
    target = MODEL_ROOT / "vnlegal-lal"
    if not target.exists():
        raise FileNotFoundError(target)

    resolved_sha = resolve_repo_sha(repo)
    rows = inventory(target)

    tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(str(target), local_files_only=True)
    backbone_limit = int(getattr(config, "max_position_embeddings", 0) or 0)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if backbone_limit < 2048:
        raise RuntimeError(f"Backbone max_position_embeddings={backbone_limit} < 2048")

    model = AutoModel.from_pretrained(
        str(target), local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()

    params = int(sum(p.numel() for p in model.parameters()))
    dim = int(getattr(config, "hidden_size", 0))
    if dim != 1024:
        raise RuntimeError(f"Unexpected VNLegal hidden size: {dim}")

    def encode(texts, is_query=False, max_length=2048):
        batch = [VN_PREFIX + t for t in texts] if is_query else texts
        inp = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            out = model(**inp)
        last_idx = inp["attention_mask"].sum(dim=1) - 1
        emb = out.last_hidden_state[
            torch.arange(len(batch), device="cuda"), last_idx
        ]
        emb = F.normalize(emb.float(), p=2, dim=1)
        return emb.cpu().numpy(), inp

    q, _ = encode(["Điều kiện hưởng trợ cấp thất nghiệp là gì?"], True, 2048)
    d, _ = encode(
        ["Điều 49. Người lao động được hưởng trợ cấp thất nghiệp khi đáp ứng các điều kiện theo quy định."],
        False, 2048
    )
    q = normalize_np(q); d = normalize_np(d)

    long_doc = ("Người lao động có quyền và nghĩa vụ theo quy định của pháp luật. " * 180)
    raw_long_len = len(tokenizer(long_doc, add_special_tokens=True, truncation=False)["input_ids"])
    _, long_inputs = encode([long_doc], False, 1024)
    forwarded_len = int(long_inputs["input_ids"].shape[1])
    if raw_long_len <= 512 or forwarded_len <= 512:
        raise RuntimeError(
            f"VNLegal >512 smoke failed: raw={raw_long_len}, forwarded={forwarded_len}"
        )

    manifest = common_manifest(
        key=key, repo=repo, target=target, resolved_sha=resolved_sha, rows=rows,
        params=params, dim=dim, max_seq=2048,
        adapter={
            "loader": "transformers.AutoModel+AutoTokenizer",
            "pooling": "last_non_padding_token",
            "normalization": "L2",
            "query_prefix": VN_PREFIX,
            "document_prefix": "",
            "operational_max_length": 2048,
            "tokenizer_metadata_model_max_length": tokenizer_limit,
            "backbone_max_position_embeddings": backbone_limit,
            "reason_for_override": (
                "Official VNLegal-LAL usage specifies max_len=2048; tokenizer metadata "
                "reports 512 while the Qwen3 backbone supports longer context."
            ),
        },
        smoke={
            "short_query_document_cosine": float((q @ d.T)[0, 0]),
            "raw_long_document_token_count": raw_long_len,
            "long_smoke_forwarded_token_count": forwarded_len,
            "proved_forward_above_512": True,
        },
    )

    (target / "_ENDGAME_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rd = OUT / key; rd.mkdir(parents=True, exist_ok=True)
    (rd / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest

def repair_gte():
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    key = "gte_multilingual_base"
    repo = "Alibaba-NLP/gte-multilingual-base"
    dependency_repo = "Alibaba-NLP/new-impl"
    target = MODEL_ROOT / "gte-multilingual-base"
    if not target.exists():
        raise FileNotFoundError(target)

    resolved_sha = resolve_repo_sha(repo)
    dep_sha = resolve_repo_sha(dependency_repo)
    rows = inventory(target)

    DEP_ROOT.mkdir(parents=True, exist_ok=True)
    dep_target = DEP_ROOT / "alibaba-new-impl"
    dep_manifest_path = dep_target / "_ENDGAME_DEPENDENCY_MANIFEST.json"

    rebuild_dep = True
    if dep_target.exists() and dep_manifest_path.exists():
        old = read_json(dep_manifest_path)
        rebuild_dep = old.get("resolved_revision_sha") != dep_sha
    if rebuild_dep:
        if dep_target.exists():
            shutil.rmtree(dep_target)
        print(f"[gte] download dependency {dependency_repo}@{dep_sha}", flush=True)
        snapshot_download(repo_id=dependency_repo, revision=dep_sha, local_dir=str(dep_target))
        local_cache = dep_target / ".cache"
        if local_cache.exists():
            shutil.rmtree(local_cache, ignore_errors=True)
        dep_rows = inventory(dep_target, exclude_manifest=False)
        dep_manifest = {
            "schema_version": "dsc2026.endgame.hf_dynamic_code_dependency.v1",
            "repo": dependency_repo,
            "resolved_revision_sha": dep_sha,
            "content_fingerprint": fingerprint(dep_rows),
            "files": dep_rows,
        }
        dep_manifest_path.write_text(
            json.dumps(dep_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        dep_manifest = read_json(dep_manifest_path)

    model_kwargs = {"torch_dtype": torch.float16, "code_revision": dep_sha}
    config_kwargs = {"code_revision": dep_sha}

    print("[gte] pinned dynamic-code warm load", flush=True)
    model = SentenceTransformer(
        str(target),
        device="cuda",
        local_files_only=False,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        config_kwargs=config_kwargs,
    )
    if int(model.max_seq_length) != 8192:
        raise RuntimeError(f"GTE max_seq_length={model.max_seq_length}, expected=8192")

    params = int(sum(p.numel() for p in model.parameters()))
    dim = int(model.get_embedding_dimension())
    probe = normalize_np(model.encode(
        [
            "Điều kiện hưởng trợ cấp thất nghiệp",
            "Điều 49 quy định điều kiện hưởng trợ cấp thất nghiệp",
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ))
    cosine = float(probe[0] @ probe[1])

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("[gte] offline/local-only reload", flush=True)
    model2 = SentenceTransformer(
        str(target),
        device="cuda",
        local_files_only=True,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        config_kwargs=config_kwargs,
    )
    if int(model2.max_seq_length) != 8192:
        raise RuntimeError("GTE offline reload max_seq_length drift")
    params2 = int(sum(p.numel() for p in model2.parameters()))
    if params2 != params:
        raise RuntimeError(f"GTE parameter count drift: {params} -> {params2}")

    probe2 = normalize_np(model2.encode(
        ["Quy định về hợp đồng lao động"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ))

    manifest = common_manifest(
        key=key, repo=repo, target=target, resolved_sha=resolved_sha, rows=rows,
        params=params, dim=dim, max_seq=8192,
        adapter={
            "loader": "sentence_transformers.SentenceTransformer",
            "trust_remote_code": True,
            "pooling": "checkpoint SentenceTransformer contract",
            "normalization": "checkpoint Normalize + explicit float32 L2",
            "query_prefix": "",
            "document_prefix": "",
            "max_seq_length": 8192,
            "dynamic_code_dependency": {
                "repo": dependency_repo,
                "resolved_revision_sha": dep_sha,
                "materialized_path": str(dep_target.resolve()),
                "hf_modules_cache": os.environ["HF_MODULES_CACHE"],
            },
        },
        smoke={
            "warm_load_cosine": cosine,
            "offline_local_only_reload": True,
            "offline_probe_shape": list(probe2.shape),
        },
        extra={"dynamic_code_dependency_manifest": dep_manifest},
    )

    (target / "_ENDGAME_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rd = OUT / key; rd.mkdir(parents=True, exist_ok=True)
    (rd / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    del model2
    gc.collect()
    torch.cuda.empty_cache()
    return manifest

def registry_entry(m):
    return {
        "model_id": m["model_id"],
        "resolved_revision_sha": m["resolved_revision_sha"],
        "parameter_count": m["parameter_count"],
        "parameter_billions": m["parameter_billions"],
        "max_seq_length": m["max_seq_length"],
        "embedding_dimension": m["embedding_dimension"],
        "budget_fraction": m["competition_budget_fraction_if_used"],
        "adapter": m["adapter_contract"]["loader"],
    }

def main():
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    reg = read_json(REGISTRY)
    passed = dict(reg.get("passed", {}))
    failed = dict(reg.get("failed", {}))

    for key, fn in [("vnlegal_lal", repair_vnlegal), ("gte_multilingual_base", repair_gte)]:
        try:
            print(f"=== repair {key} ===", flush=True)
            m = fn()
            passed[key] = registry_entry(m)
            failed.pop(key, None)
            print(json.dumps({
                "candidate": key, "status": "PASS",
                "params_b": m["parameter_billions"],
                "max_seq_length": m["max_seq_length"],
                "adapter": m["adapter_contract"]["loader"],
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

    total = sum(int(v["parameter_count"]) for v in passed.values())
    updated = {
        "schema_version": "dsc2026.endgame.retriever_candidate_registry.v1.1",
        "status": "PASS" if not failed else "PARTIAL_FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_pipeline_parameter_limit": 4_000_000_000,
        "scientific_status": "materialization only; no retriever promoted",
        "passed": passed,
        "failed": failed,
        "sum_parameters_if_all_passed_candidates_were_combined": total,
        "sum_billions_if_all_passed_candidates_were_combined": total / 1e9,
    }
    REGISTRY.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": updated["status"],
        "passed": sorted(passed),
        "failed": sorted(failed),
        "sum_params_b": total / 1e9,
        "registry": str(REGISTRY.resolve()),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failed else 2)

if __name__ == "__main__":
    main()

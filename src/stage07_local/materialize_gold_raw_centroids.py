#!/usr/bin/env python
"""Rebuild the raw AIT parent centroids used by GOLD, without teacher data.

The historical cache path is kept for compatibility with Stage07O. Only the
Stage02B4 frozen region embeddings and Stage02B0 geometry are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage02_candidate_generation import screen_vietnamese_retrievers as geometry
from src.stage03_rerank import run_aiteam_reranker_oof as b1

REGION = ROOT / "cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy"
OUT = ROOT / "cache/stage07d_teacher_distill_head"
EXPECTED_REGION_SHA256 = "8a5ed2a7ee034f4148d0d2c8fbfe6f40da68a08aa0c2c995463eba7b93ee5a5d"
HISTORICAL_CENTROID_SHA256 = "660827c01e605e79c1b2b3dcdb94f58162364361439ffec791660d6116b37b53"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    centroid = out / "aiteam_parent_centroids.f32.npy"
    meta = out / "aiteam_parent_centroids.json"

    region_sha = sha256(REGION)
    print(f"[centroid] historical_region_bytes={region_sha == EXPECTED_REGION_SHA256}", flush=True)
    _, _, _, _, _, docs, _ = b1.load_world()
    geom = geometry.load_geometry()
    if list(map(str, geom["doc_ids"])) != docs:
        raise RuntimeError("Stage02 geometry document order drift")
    parent_index = np.asarray(geom["parent_index"], np.int64)
    region = np.load(REGION, mmap_mode="r")
    if region.shape != (len(parent_index), 1024):
        raise RuntimeError(f"AIT region embedding shape drift: {region.shape}")

    if centroid.is_file() and meta.is_file():
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old.get("source") != "cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy":
            raise RuntimeError("Existing centroid provenance drift")
        if old.get("source_sha256", EXPECTED_REGION_SHA256) != region_sha:
            raise RuntimeError("Existing centroid source hash drift")
        values = np.load(centroid, mmap_mode="r")
        if values.shape != (len(docs), 1024) or not np.isfinite(values).all():
            raise RuntimeError("Existing centroid shape/value drift")
        print(f"GOLD_RAW_CENTROID_PASS (existing; historical_bytes={sha256(centroid) == HISTORICAL_CENTROID_SHA256})", flush=True)
        return
    if centroid.exists() or meta.exists():
        raise RuntimeError("Incomplete centroid cache; refusing to overwrite")

    import torch

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sums = torch.zeros((len(docs), 1024), dtype=torch.float32, device=dev)
    counts = torch.zeros((len(docs), 1), dtype=torch.float32, device=dev)
    chunk = 4096 if dev.type == "cuda" else 2048
    for start in range(0, len(region), chunk):
        end = min(start + chunk, len(region))
        vectors = torch.from_numpy(np.asarray(region[start:end], np.float32).copy()).to(dev)
        indices = torch.from_numpy(parent_index[start:end].copy()).to(dev)
        sums.index_add_(0, indices, vectors)
        counts.index_add_(0, indices, torch.ones((end - start, 1), device=dev))
        if end == len(region) or start % (chunk * 10) == 0:
            print(f"[centroid] {end}/{len(region)} regions", flush=True)
    result = torch.nn.functional.normalize(sums / counts.clamp_min(1), dim=1)
    values = result.cpu().numpy().astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(centroid, values)
    meta.write_text(json.dumps({
        "schema": "stage07d.aiteam_parent_centroid.v1",
        "docs": len(docs), "dim": 1024,
        "source": "cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy",
        "aggregation": "mean(region embeddings) then L2 normalize",
        "source_sha256": region_sha,
        "historical_centroid_sha256": HISTORICAL_CENTROID_SHA256,
        "centroid_sha256": sha256(centroid),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"GOLD_RAW_CENTROID_PASS (rebuilt; historical_bytes={sha256(centroid) == HISTORICAL_CENTROID_SHA256})", flush=True)


if __name__ == "__main__":
    main()

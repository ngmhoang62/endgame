#!/usr/bin/env python
"""Resumable zero-shot Qwen0.6B pilot on clean gold fold3 evidence.

Uses the registered, original Qwen/Qwen3-Reranker-0.6B. No student checkpoint,
teacher logits, teacher targets, or distillation artifacts are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "artifacts/stage07b_colab_payload"
OUT = ROOT / "cache/stage07x_qwen06b_base_local"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
SCRIPT = ROOT / "artifacts/stage07b/run_qwen06b_gold_supervised_l4.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("stage07n_gold_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metric(score, y, count):
    top = np.argsort(-score, axis=1, kind="stable")[:, :5]
    rec = np.take_along_axis(y, top, axis=1).sum(1) / count
    return float(rec.mean()), float(rec[count == 1].mean()), float(rec[count > 1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-pairs", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--sample-seed", type=int, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    n = load_contract()
    folds = np.load(PAYLOAD / "fold_id.i1.npy")
    fold_indices = np.where(folds == 3)[0].astype(np.int32)
    if args.sample_seed is None:
        idx = fold_indices[:args.limit]
        tag = f"fold3_top{len(idx)}"
    else:
        idx = np.sort(np.random.default_rng(args.sample_seed).choice(
            fold_indices, size=args.limit, replace=False)).astype(np.int32)
        tag = f"fold3_random{args.sample_seed}_n{len(idx)}"
    y = np.load(PAYLOAD / "labels_top30.u1.npy")[idx]
    gold_count = np.load(PAYLOAD / "gold_count.i2.npy")[idx]
    X45 = np.load(PAYLOAD / "X45.f32.npy", mmap_mode="r")
    labels_all = np.load(PAYLOAD / "labels_top30.u1.npy")
    train_idx = np.where(np.isin(folds, (0, 1, 2)))[0].astype(np.int32)
    prior = n.held_prior(X45, labels_all, train_idx, idx)
    zprior = n.zrows(prior)
    print(f"[pilot] queries={len(idx)} baseline={metric(zprior, y, gold_count)[0]:.9f}", flush=True)
    print(f"[pilot] loading clean evidence {PAYLOAD / 'evidence_top20.pkl'}", flush=True)
    with (PAYLOAD / "evidence_top20.pkl").open("rb") as f:
        evidence = pickle.load(f)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", use_fast=True, local_files_only=True)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=amp, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    model.eval()
    yes_id, no_id = n.yes_no_ids(tok)
    OUT.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / f"{tag}_scores.npy"
    done_path = OUT / f"{tag}_done.npy"
    if scores_path.exists() and done_path.exists():
        scores = np.load(scores_path)
        done = np.load(done_path)
    else:
        scores = np.full((len(idx), 20), np.nan, np.float32)
        done = np.zeros(len(idx), np.uint8)
    t0 = time.monotonic()
    with torch.inference_mode():
        for li, qi in enumerate(idx):
            if done[li]:
                continue
            pairs = evidence[int(qi)]
            row = []
            for start in range(0, 20, args.batch_pairs):
                feats = n.qwen_features(tok, pairs[start:start + args.batch_pairs], args.maxlen)
                batch = tok.pad(feats, padding=True, pad_to_multiple_of=8, return_tensors="pt")
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=amp):
                    s = n.last_score(model, batch, yes_id, no_id)
                row.extend(s.float().cpu().numpy().tolist())
            scores[li] = np.asarray(row, np.float32)
            done[li] = 1
            if (li + 1) % 10 == 0 or li + 1 == len(idx):
                np.save(scores_path, scores)
                np.save(done_path, done)
                rate = (li + 1) / max(time.monotonic() - t0, 1e-6)
                print(f"[pilot] {li+1}/{len(idx)} rate={rate:.3f} q/s", flush=True)
    qonly = np.full((len(idx), 30), -1e9, np.float32)
    qonly[:, :20] = scores
    fused = zprior.copy()
    fused[:, :20] += 0.25 * (scores - scores.mean(1, keepdims=True))
    report = {"schema": "stage07x.qwen06b_base_pilot.v1", "model": MODEL_ID,
              "original_model_only": True, "teacher_used": False, "distillation_used": False,
              "fold": 3, "queries": len(idx), "maxlen": args.maxlen, "batch_pairs": args.batch_pairs,
              "baseline": metric(zprior, y, gold_count),
              "qwen_only": metric(qonly, y, gold_count),
              "fixed_fusion": metric(fused, y, gold_count),
              "elapsed_seconds": time.monotonic() - t0,
              "evidence_sha256": sha(PAYLOAD / "MANIFEST.json")}
    (OUT / f"{tag}_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Stage07X-A — Qwen0.6B persistence probe on A100.

Purpose:
  Reproduce the suspicious Stage07N discrepancy:
    original in-memory fold3 ~0.95386
    reloaded checkpoint fold3 ~0.94611

This script trains a FRESH folds(0,1,2) model from the original
Qwen/Qwen3-Reranker-0.6B using gold labels only, then compares:
  1) in-memory model immediately after training
  2) the exact saved checkpoint reloaded from Drive

No teacher, no distillation.

Required:
  /content/stage07b/run_qwen06b_gold_supervised_l4.py
  /content/stage07b/payload/*

Run:
  python /content/run_qwen_persistence_probe_a100.py
"""
from __future__ import annotations

import argparse, gc, json, math, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/content")
sys.path.insert(0, str(BASE))
from stage07b import run_qwen06b_gold_supervised_l4 as n

OUT = BASE / "drive/MyDrive/DSC2026/stage07x_qwen_persistence_probe"
DEPTH = 20
FULL = 30

def batch_infer(model, tok, evidence, idx, maxlen, batch_queries=4):
    yes_id, no_id = n.yes_no_ids(tok)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = np.empty((len(idx), DEPTH), np.float32)
    model.eval()
    with torch.inference_mode():
        for st in range(0, len(idx), batch_queries):
            ids = idx[st:st+batch_queries]
            pairs = []
            for qi in ids:
                pairs.extend(evidence[int(qi)][:DEPTH])
            feats = n.qwen_features(tok, pairs, maxlen)
            batch = tok.pad(feats, padding=True, pad_to_multiple_of=8, return_tensors="pt")
            batch = {k:v.to("cuda", non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                s = n.last_score(model, batch, yes_id, no_id).float()
            out[st:st+len(ids)] = s.reshape(len(ids), DEPTH).cpu().numpy()
            if (st//batch_queries+1) % 25 == 0 or st+len(ids) == len(idx):
                print(f"[infer] {min(st+len(ids),len(idx))}/{len(idx)}", flush=True)
    return out

def metrics(prior, raw, alpha, y, gold_count, idx):
    final = n.zrows(prior)
    final[:, :DEPTH] += float(alpha) * raw
    m, q, _ = n.evaluate(final, y, gold_count, idx)
    return m, q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-queries", type=int, default=4)
    ap.add_argument("--sample", type=int, default=128)
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A100 required")
    print("[device]", torch.cuda.get_device_name(0), flush=True)

    n.OUT = OUT
    n.TRAIN_FOLDS = (0,1,2)

    if args.force_retrain and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    qids, short, X45, y, gold_count, fold_id, evidence = n.load_payload()
    train_idx = np.where(np.isin(fold_id, (0,1,2)))[0].astype(np.int32)
    held = np.where(fold_id == 3)[0].astype(np.int32)

    cfg = argparse.Namespace(
        epochs=1, grad_accum=4, maxlen=1024,
        lr=8e-6, alpha_lr=8e-4, weight_decay=.01,
        hard_negs=4, max_negs=4,
        listwise_weight=.55, pair_weight=.25, bce_weight=.20,
        resid_l2=5e-4, gold_temp=1.0, margin=.4,
        hard_query_bonus=1.25, multi_bonus=.15,
        dev_min_delta=.002, seed=276,
    )

    print("[1/6] FRESH/REUSED gold-only train012", flush=True)
    tok_mem, model_mem, alpha, got_train = n.train_model(
        qids, X45, y, fold_id, evidence, cfg
    )
    if not np.array_equal(got_train, train_idx):
        raise RuntimeError("train population drift")
    prior = n.held_prior(X45, y, train_idx, held)
    bm, _, _ = n.evaluate(prior, y, gold_count, held)
    print(f"[baseline] fold3={bm['recall_at_5']:.9f}", flush=True)

    # Small paired test first.
    sample = held[:min(args.sample, len(held))]
    print(f"[2/6] in-memory sample n={len(sample)}", flush=True)
    raw_mem_s = batch_infer(model_mem, tok_mem, evidence, sample, cfg.maxlen, args.batch_queries)

    print("[3/6] reload exact persisted checkpoint", flush=True)
    # A100 has ample VRAM for two 0.6B FP32 models. Keep the live model resident:
    # this avoids a second ~30 minute retrain and makes the comparison truly paired.
    tok_disk, model_disk = n.load_saved_model()
    raw_disk_s = batch_infer(model_disk, tok_disk, evidence, sample, cfg.maxlen, args.batch_queries)

    diff = np.abs(raw_mem_s - raw_disk_s)
    corr = float(np.corrcoef(raw_mem_s.ravel(), raw_disk_s.ravel())[0,1])
    sample_diag = {
        "sample_queries": int(len(sample)),
        "max_abs_raw_diff": float(diff.max()),
        "mean_abs_raw_diff": float(diff.mean()),
        "corr": corr,
    }
    print("[sample-diff]", json.dumps(sample_diag), flush=True)

    # Even if logits are numerically close, ranking can be sensitive near the K=5
    # boundary. Score the full fold once with each state; no retraining required.
    print("[4/6] full fold3 with LIVE in-memory model", flush=True)
    raw_mem = batch_infer(model_mem, tok_mem, evidence, held, cfg.maxlen, args.batch_queries)
    mm, mq = metrics(prior, raw_mem, alpha, y, gold_count, held)
    np.save(OUT/"fold3_raw_inmemory.f32.npy", raw_mem)
    print(f"[in-memory] R={mm['recall_at_5']:.9f}", flush=True)

    print("[5/6] full fold3 with RELOADED model", flush=True)
    raw_disk = batch_infer(model_disk, tok_disk, evidence, held, cfg.maxlen, args.batch_queries)
    dm, dq = metrics(prior, raw_disk, alpha, y, gold_count, held)
    np.save(OUT/"fold3_raw_reloaded.f32.npy", raw_disk)
    print(f"[reloaded] R={dm['recall_at_5']:.9f}", flush=True)

    report = {
        "schema":"stage07x.qwen_persistence_probe.v1",
        "status":"COMPLETE",
        "teacher_used":False, "distillation_used":False,
        "baseline":bm,
        "alpha":float(alpha),
        "sample_diag":sample_diag,
        "inmemory_fold3":mm,
        "reloaded_fold3":dm,
        "delta_inmemory_vs_baseline":float(mm["recall_at_5"]-bm["recall_at_5"]),
        "delta_reloaded_vs_baseline":float(dm["recall_at_5"]-bm["recall_at_5"]),
        "recommendation":(
            "PROMOTE_INMEMORY_FULLTRAIN_PRIVATE"
            if mm["recall_at_5"] >= .951 and mm["recall_at_5"] > dm["recall_at_5"] + .002
            else "KILL_PERSISTENCE_HYPOTHESIS"
        ),
    }
    (OUT/"PROBE_REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print("[6/6]", report["recommendation"], flush=True)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()

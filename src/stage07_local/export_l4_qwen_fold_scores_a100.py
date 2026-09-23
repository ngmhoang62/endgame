#!/usr/bin/env python
"""Export raw held-fold scores from the clean L4 gold Qwen checkpoint.

Fold 3 is DEV for testing one frozen calibration hypothesis. Fold 4 is only
requested after the hypothesis is selected. No teacher/distillation inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path("/content")
sys.path.insert(0, str(BASE))
from stage07b import run_qwen06b_gold_supervised_l4 as n

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=(3, 4), required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device_name = torch.cuda.get_device_name(0)
    export_suffix = "held_score_exports_l4" if "L4" in device_name else "held_score_exports_a100_v2"
    out = BASE / "drive/MyDrive/DSC2026/stage07n_t4_gold_qwen" / export_suffix
    meta_path = n.OUT / "TRAIN_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("status") != "PASS" or meta.get("training_folds") != [0, 1, 2]
            or meta.get("teacher_used") is not False or meta.get("distillation_used") is not False):
        raise RuntimeError("L4 checkpoint provenance drift")
    qids, short, X45, y, gold_count, fold_id, evidence = n.load_payload()
    idx = np.where(fold_id == args.fold)[0].astype(np.int32)
    train_idx = np.where(np.isin(fold_id, (0, 1, 2)))[0].astype(np.int32)
    prior = n.held_prior(X45, y, train_idx, idx)
    tok, model = n.load_saved_model()
    model.eval()
    infer_args = argparse.Namespace(maxlen=int(meta["config"]["maxlen"]))
    raw = n.infer_scores(model, tok, evidence, idx, infer_args)
    if raw.shape != (len(idx), 20) or not np.isfinite(raw).all():
        raise RuntimeError("held raw score incomplete")
    final = n.zrows(prior)
    final[:, :20] += float(meta["alpha"]) * raw
    bm, _ = n.evaluate(prior, y, gold_count, idx)
    qm, _ = n.evaluate(final, y, gold_count, idx)
    parity = (args.fold != 3 or
              (abs(bm["recall_at_5"] - .947186457) <= 3e-6
               and abs(qm["recall_at_5"] - .953862661) <= 3e-6))
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, arr in (("raw", raw), ("prior", prior), ("indices", idx)):
        local = BASE / f"stage07n_fold{args.fold}_{name}.npy"
        dst = out / local.name
        np.save(local, arr)
        shutil.copy2(local, dst)
        paths[name] = {"file": dst.name, "sha256": sha(dst), "shape": list(arr.shape)}
    report = {"schema": "stage07n.l4_held_scores.v1", "fold": args.fold,
              "device": device_name, "explicit_model_eval": True,
              "teacher_used": False, "distillation_used": False,
              "checkpoint_train_folds": [0, 1, 2],
              "checkpoint_meta_sha256": sha(meta_path), "alpha": meta["alpha"],
              "baseline": bm, "gold_qwen": qm, "l4_dev_parity": parity,
              "files": paths}
    (out / f"FOLD{args.fold}_SCORE_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FOLD{args.fold} READY: baseline={bm['recall_at_5']:.9f} "
          f"qwen={qm['recall_at_5']:.9f} rows={len(idx)} parity={parity}", flush=True)


if __name__ == "__main__":
    main()

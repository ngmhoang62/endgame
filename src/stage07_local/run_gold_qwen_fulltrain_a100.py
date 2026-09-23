#!/usr/bin/env python
"""Gold-only fulltrain Qwen0.6B on A100; independent of the L4 DEV/CERT run.

This wrapper reuses the exact Stage07N L4 implementation and changes only the
training fold population and output folder. Submission remains gated by the
independent L4 fold3/fold4 report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path("/content")
sys.path.insert(0, str(BASE))
from stage07b import run_qwen06b_gold_supervised_l4 as n

OUT = BASE / "drive/MyDrive/DSC2026/stage07n_full_gold_qwen_a100"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not n.torch.cuda.is_available() or "A100" not in n.torch.cuda.get_device_name(0):
        raise RuntimeError("A100 runtime required")
    n.OUT = OUT
    n.TRAIN_FOLDS = (0, 1, 2, 3, 4)
    args = argparse.Namespace(
        epochs=1, grad_accum=4, maxlen=1024,
        lr=8e-6, alpha_lr=8e-4, weight_decay=.01,
        hard_negs=4, max_negs=4,
        listwise_weight=.55, pair_weight=.25, bce_weight=.20,
        resid_l2=5e-4, gold_temp=1.0, margin=.4,
        hard_query_bonus=1.25, multi_bonus=.15,
        dev_min_delta=.002, seed=276,
    )
    qids, short, X45, y, gold_count, fold_id, evidence = n.load_payload()
    if len(qids) != 6991 or set(map(int, fold_id)) != set(range(5)):
        raise RuntimeError("fulltrain population or fold drift")
    if len(evidence) != len(qids) or any(len(row) != 20 for row in evidence):
        raise RuntimeError("fulltrain evidence drift")
    print("[eligibility] teacher_used=False distillation_used=False gold_labels_only=True", flush=True)
    print("[fulltrain] A100 training all 6991 query labels from original Qwen0.6B", flush=True)
    tok, model, alpha, train_idx = n.train_model(qids, X45, y, fold_id, evidence, args)
    meta = json.loads((OUT / "TRAIN_META.json").read_text(encoding="utf-8"))
    if meta.get("status") != "PASS" or meta.get("training_folds") != list(range(5)) or len(train_idx) != len(qids):
        raise RuntimeError("fulltrain checkpoint provenance failed")
    report = {"schema": "stage07n.a100.full_gold_train.v1", "status": "TRAINED_AWAITING_L4_GATE",
              "teacher_used": False, "distillation_used": False,
              "base_model": n.MODEL_ID, "gold_queries": len(qids),
              "training_folds": list(range(5)), "alpha": alpha,
              "payload_manifest_sha256": sha(n.PAYLOAD / "MANIFEST.json"),
              "train_meta_sha256": sha(OUT / "TRAIN_META.json"),
              "checkpoint_dir": str(OUT / "train012_model")}
    (OUT / "FULLTRAIN_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("READY FULLTRAIN:", OUT / "train012_model", flush=True)


if __name__ == "__main__":
    main()

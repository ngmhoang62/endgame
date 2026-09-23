#!/usr/bin/env python
"""Precompute A100 fulltrain private Qwen scores while L4 DEV/CERT is running.

This cannot materialize a submission. The normal private runner must still
verify L4 PROMOTE_TO_FULLTRAIN before using these cached scores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE = Path("/content")
TRAIN = BASE / "drive/MyDrive/DSC2026/stage07n_full_gold_qwen_a100"
os.environ["STAGE07N_CHECKPOINT_ROOT"] = str(TRAIN)

import run_gold_qwen_private_colab as p


def main():
    manifest, qids, docs, short, prior, evidence = p.load_payload()
    meta_path = TRAIN / "TRAIN_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("status") != "PASS" or meta.get("training_folds") != [0, 1, 2, 3, 4]
            or meta.get("teacher_used") is not False or meta.get("distillation_used") is not False
            or not (TRAIN / "train012_model/model.safetensors").is_file()):
        raise RuntimeError("A100 fulltrain checkpoint provenance failed")
    tok, model = p.load_saved_model()
    contract = {"schema": "stage07n.private_score_cache.v1",
                "private_sha256": manifest["private_sha256"],
                "evidence_sha256": manifest["files"]["evidence_top20.pkl"]["sha256"],
                "train_meta_sha256": p.sha(meta_path),
                "maxlen": int(meta["config"]["maxlen"])}
    score = p.scores(model, tok, evidence, contract["maxlen"], contract)
    print(f"SCORE_ONLY_READY {len(qids)} queries, {score.shape[1]} candidates each", flush=True)
    print("No submission materialized. Run normal private notebook cells only after L4 gate passes.", flush=True)


if __name__ == "__main__":
    main()

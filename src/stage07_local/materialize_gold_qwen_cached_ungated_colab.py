#!/usr/bin/env python
"""Materialize the already-scored A100 fulltrain private submission.

No L4 DEV/CERT requirement and no model inference. The report records that
local validation is pending; cached-score, payload, and schema checks remain.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np

TRAIN = Path("/content/drive/MyDrive/DSC2026/stage07n_full_gold_qwen_a100")
os.environ["STAGE07N_CHECKPOINT_ROOT"] = str(TRAIN)

import run_gold_qwen_private_colab as p


def main():
    manifest, qids, docs, short, prior, _ = p.load_payload()
    meta_path = TRAIN / "TRAIN_META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("status") != "PASS" or meta.get("training_folds") != [0, 1, 2, 3, 4]
            or meta.get("model_id") != "Qwen/Qwen3-Reranker-0.6B"
            or meta.get("teacher_used") is not False or meta.get("distillation_used") is not False):
        raise RuntimeError("fulltrain provenance mismatch")

    out = TRAIN / "private_inference"
    contract = json.loads((out / "SCORE_CONTRACT.json").read_text(encoding="utf-8"))
    expected = {"schema": "stage07n.private_score_cache.v1",
                "private_sha256": manifest["private_sha256"],
                "evidence_sha256": manifest["files"]["evidence_top20.pkl"]["sha256"],
                "train_meta_sha256": p.sha(meta_path),
                "maxlen": int(meta["config"]["maxlen"])}
    if contract != expected:
        raise RuntimeError("private score cache/checkpoint contract mismatch")
    score = np.load(out / "qwen_scores_top20.f32.npy")
    done = np.load(out / "qwen_done.u1.npy")
    if (score.shape != (2080, 20) or done.shape != (2080,)
            or int(done.sum()) != 2080 or not np.isfinite(score).all()):
        raise RuntimeError("private scores incomplete")

    final = p.zrows(prior)
    final[:, :20] += float(meta["alpha"]) * score
    if not np.isfinite(final).all() or np.min(short) < 0 or np.max(short) >= len(docs):
        raise RuntimeError("invalid ranking inputs")
    output = {}
    for i, qid in enumerate(qids):
        order = np.argsort(-final[i], kind="stable")[:5]
        answer = [docs[int(short[i, j])] for j in order]
        if len(answer) != 5 or len(set(answer)) != 5:
            raise RuntimeError(f"invalid top5 for {qid}")
        output[qid] = {"answer": answer}
    if len(output) != 2080:
        raise RuntimeError("duplicate or missing private query IDs")

    label = "GOLD_QWEN06B_FULLTRAIN_PRIVATE_K5"
    submission_json = out / f"{label}.json"
    submission_zip = out / f"{label}.zip"
    submission_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(submission_zip, "w", compresslevel=9) as z:
        z.writestr(info, submission_json.read_bytes())
    with zipfile.ZipFile(submission_zip) as z:
        if z.namelist() != ["submission.json"] or z.read("submission.json") != submission_json.read_bytes():
            raise RuntimeError("ZIP payload mismatch")

    report = {"schema": "stage07n.private_submission.ungated.v1",
              "status": "READY_FOR_SUBMISSION_LOCAL_DEV_CERT_PENDING",
              "validation_status": "L4_DEV_CERT_PENDING_AT_MATERIALIZATION",
              "teacher_used": False, "distillation_used": False,
              "model_id": meta["model_id"], "training_folds": meta["training_folds"],
              "private_sha256": manifest["private_sha256"],
              "queries": len(output), "mean_returned": 5,
              "alpha": float(meta["alpha"]),
              "score_contract_sha256": p.sha(out / "SCORE_CONTRACT.json"),
              "json": str(submission_json), "json_sha256": p.sha(submission_json),
              "zip": str(submission_zip), "zip_sha256": p.sha(submission_zip)}
    (out / "PRIVATE_SUBMISSION_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("READY:", submission_zip, flush=True)
    print("SHA256:", report["zip_sha256"], "queries:", len(output), "K: 5", flush=True)
    print("LOCAL DEV/CERT: PENDING", flush=True)


if __name__ == "__main__":
    main()

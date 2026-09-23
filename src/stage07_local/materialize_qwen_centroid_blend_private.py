#!/usr/bin/env python
"""Prepare an equal blend of clean gold Qwen and AIT centroid rank scores."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from src.stage07_local import materialize_qwen_centered_private as q


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    qids = json.loads((q.PAYLOAD / "qids.json").read_text(encoding="utf-8"))
    docs = json.loads((q.PAYLOAD / "doc_ids.json").read_text(encoding="utf-8"))
    short = np.load(q.PAYLOAD / "shortlist_top30.i4.npy")
    prior = np.load(q.PAYLOAD / "ce_lr_prior.f32.npy")
    raw = np.load(q.SCORES / "qwen_scores_top20.f32.npy")
    meta = json.loads((q.SCORES / "TRAIN_META.json").read_text(encoding="utf-8"))
    centroid_report = json.loads((root / "reports/stage07o_gold_centroid_head/PRIVATE_SUBMISSION_REPORT.json").read_text(encoding="utf-8"))
    if (meta.get("teacher_used") is not False or meta.get("distillation_used") is not False
            or centroid_report.get("teacher_used") is not False
            or centroid_report.get("distillation_used") is not False):
        raise RuntimeError("Unclean score lineage")
    centroid = np.load(root / "cache/stage07o_gold_centroid_head/private_score30.f32.npy")
    if centroid.shape != prior.shape or raw.shape != (len(qids), 20):
        raise RuntimeError("Score shape mismatch")
    with zipfile.ZipFile(root / centroid_report["zip"]) as z:
        previous = json.loads(z.read("submission.json"))
    if q.materialize(qids, docs, short, centroid) != previous:
        raise RuntimeError("Submitted centroid score parity failure")
    centered = q.zrows(prior)
    centered[:, :20] += float(meta["alpha"]) * (raw - raw.mean(1, keepdims=True))
    blend = .5 * (centered + centroid)
    answers = q.materialize(qids, docs, short, blend)
    label = "GOLD_QWEN06B_CENTERED_CENTROID_EQUAL_K5"
    q.SUB.mkdir(parents=True, exist_ok=True)
    jp = q.SUB / (label + ".json")
    zp = q.SUB / (label + ".zip")
    data = (json.dumps(answers, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    jp.write_bytes(data)
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zp, "w", compresslevel=9) as z:
        z.writestr(info, data)
    with zipfile.ZipFile(zp) as z:
        if z.namelist() != ["submission.json"] or z.read("submission.json") != data:
            raise RuntimeError("ZIP payload mismatch")
    centered_answers = q.materialize(qids, docs, short, centered)
    print(f"CANDIDATE {zp} sha256={q.sha(zp)} changed_vs_centered="
          f"{sum(set(answers[k]['answer']) != set(centered_answers[k]['answer']) for k in qids)}", flush=True)


if __name__ == "__main__":
    main()

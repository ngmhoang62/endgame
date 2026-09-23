#!/usr/bin/env python
"""Check top20-only residual offset in the clean centroid OOF scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.stage07_breakthrough.run_residual_setwise_oof import fit_lr_score, zrows


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/stage07b_colab_payload"
CACHE = ROOT / "cache/stage07o_gold_centroid_head"
OUT = ROOT / "reports/stage07o_gold_centroid_head"


def rank_recall(score, y, count):
    rank = np.argsort(-score, axis=1, kind="stable")[:, :5]
    hit = np.take_along_axis(y, rank, axis=1).sum(1)
    recall = hit / count
    return recall, rank


def main():
    x = np.load(BASE / "X45.f32.npy", mmap_mode="r")
    y = np.load(BASE / "labels_top30.u1.npy")
    count = np.load(BASE / "gold_count.i2.npy")
    fold = np.load(BASE / "fold_id.i1.npy")
    head = np.load(CACHE / "gold_head_oof_scores30.f32.npy")
    if x.shape != (6991, 30, 45) or head.shape != (6991, 30):
        raise RuntimeError("OOF shape drift")
    prior = np.empty_like(head)
    for f in range(5):
        tr = np.where(fold != f)[0].astype(np.int32)
        va = np.where(fold == f)[0].astype(np.int32)
        prior[va] = fit_lr_score(x, y, tr, va)
    base = zrows(prior)
    base_rec, _ = rank_recall(base, y, count)
    head_rec, _ = rank_recall(head, y, count)
    oof_report = json.loads((OUT / "OOF_REPORT.json").read_text(encoding="utf-8"))
    if (abs(float(base_rec.mean()) - oof_report["baseline"]["overall"]["recall_at_5"]) > 3e-6
            or abs(float(head_rec.mean()) - oof_report["candidate"]["overall"]["recall_at_5"]) > 3e-6):
        raise RuntimeError("Stage07O OOF metric parity failure")
    residual = head[:, :20] - base[:, :20]
    centered = base.copy()
    centered[:, :20] += residual - residual.mean(1, keepdims=True)
    centered_rec, centered_rank = rank_recall(centered, y, count)
    single = count == 1
    metrics = {}
    for name, rec in (("ce_lr", base_rec), ("centroid_raw", head_rec), ("centroid_centered", centered_rec)):
        metrics[name] = {"overall": float(rec.mean()), "single": float(rec[single].mean()),
                         "multi": float(rec[~single].mean()),
                         "per_fold": [float(rec[fold == f].mean()) for f in range(5)]}
    delta = centered_rec - head_rec
    report = {"schema": "stage07o.offset_oof.v1", "teacher_used": False,
              "distillation_used": False, "metrics": metrics,
              "effect": {"delta_vs_raw": float(delta.mean()), "wins": int((delta > 0).sum()),
                         "losses": int((delta < 0).sum()),
                         "outside20_top5_slots": int((centered_rank >= 20).sum()),
                         "residual_mean_quantiles": np.quantile(residual.mean(1), [0, .1, .5, .9, 1]).tolist()},
              "decision": "PROMOTE" if (metrics["centroid_centered"]["overall"] >= .95
                         and delta.mean() >= .001
                         and int((delta > 0).sum()) >= int((delta < 0).sum())
                         and metrics["centroid_centered"]["multi"] >= metrics["centroid_raw"]["multi"])
                         else "KILL"}
    (OUT / "OFFSET_OOF_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["decision"] == "PROMOTE":
        np.save(CACHE / "gold_head_centered_oof_scores30.f32.npy", centered)
    print(json.dumps({"decision": report["decision"], "metrics": metrics,
                      "effect": report["effect"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

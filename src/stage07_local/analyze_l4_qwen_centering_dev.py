#!/usr/bin/env python
"""Single predeclared fold3 test of Qwen logit mean centering.

Does not touch fold4. Raw held scores come from the unchanged L4 checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/stage07b_colab_payload"
SCORES = ROOT / "cache/stage07n_l4_held_scores"
OUT = ROOT / "reports/stage07n_centered_qwen_private"


def zrows(x):
    x = np.asarray(x, np.float32)
    sd = x.std(1, keepdims=True)
    return (x - x.mean(1, keepdims=True)) / np.where(sd < 1e-5, 1., sd)


def per_query(score, y, gold_count, idx):
    rank = np.argsort(-score, axis=1, kind="stable")[:, :5]
    hits = np.take_along_axis(y[idx], rank, axis=1).sum(1)
    recall = hits / gold_count[idx]
    return recall, rank


def arm(score, y, gold_count, idx):
    recall, rank = per_query(score, y, gold_count, idx)
    one = gold_count[idx] == 1
    return {"recall_at_5": float(recall.mean()),
            "single_recall": float(recall[one].mean()),
            "multi_recall": float(recall[~one].mean()),
            "outside20_top5_slots": int((rank >= 20).sum()),
            "outside20_gold_hits": int(np.take_along_axis(y[idx], rank, axis=1)[rank >= 20].sum())}


def main():
    rep = json.loads((SCORES / "FOLD3_SCORE_REPORT.json").read_text(encoding="utf-8"))
    if rep.get("fold") != 3 or rep.get("checkpoint_train_folds") != [0, 1, 2]:
        raise RuntimeError("held score provenance mismatch")
    idx = np.load(SCORES / "stage07n_fold3_indices.npy")
    prior = np.load(SCORES / "stage07n_fold3_prior.npy")
    qwen = np.load(SCORES / "stage07n_fold3_raw.npy")
    y = np.load(BASE / "labels_top30.u1.npy")
    gc = np.load(BASE / "gold_count.i2.npy")
    folds = np.load(BASE / "fold_id.i1.npy")
    if (len(idx), prior.shape, qwen.shape) != (1398, (1398, 30), (1398, 20)):
        raise RuntimeError("held array shape drift")
    if not np.all(folds[idx] == 3) or not np.isfinite(qwen).all() or not np.isfinite(prior).all():
        raise RuntimeError("held fold identity/value drift")
    alpha = float(rep["alpha"])
    base = zrows(prior)
    raw = base.copy()
    raw[:, :20] += alpha * qwen
    centered = base.copy()
    centered[:, :20] += alpha * (qwen - qwen.mean(1, keepdims=True))
    scores = {"ce_lr": base, "qwen_raw": raw, "qwen_centered": centered}
    metrics = {name: arm(score, y, gc, idx) for name, score in scores.items()}
    if (abs(metrics["ce_lr"]["recall_at_5"] - .947186457) > 3e-6
            or abs(metrics["qwen_raw"]["recall_at_5"] - .953862661) > 3e-6):
        raise RuntimeError("L4 DEV official parity failure")
    rb, _ = per_query(raw, y, gc, idx)
    cb, _ = per_query(centered, y, gc, idx)
    effects = {"delta_vs_qwen_raw": metrics["qwen_centered"]["recall_at_5"] - metrics["qwen_raw"]["recall_at_5"],
               "delta_vs_ce_lr": metrics["qwen_centered"]["recall_at_5"] - metrics["ce_lr"]["recall_at_5"],
               "wins_vs_qwen_raw": int((cb > rb).sum()),
               "losses_vs_qwen_raw": int((cb < rb).sum()),
               "qwen_mean_quantiles": np.quantile(qwen.mean(1), [0, .1, .5, .9, 1]).tolist(),
               "qwen_sd_quantiles": np.quantile(qwen.std(1), [0, .1, .5, .9, 1]).tolist()}
    decision = ("TEST_ON_CERT" if effects["delta_vs_qwen_raw"] >= .002
                and effects["wins_vs_qwen_raw"] >= effects["losses_vs_qwen_raw"]
                and metrics["qwen_centered"]["multi_recall"] >= metrics["qwen_raw"]["multi_recall"]
                else "KILL_CENTERING")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "stage07n.centered_qwen_dev.v1", "fold": 3,
              "teacher_used": False, "distillation_used": False,
              "validation_checkpoint": "L4 gold-supervised train folds 0,1,2",
              "metric": "official macro recall at 5", "alpha": alpha,
              "metrics": metrics, "effect": effects, "decision": decision}
    (OUT / "DEV_CENTERING_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "RAW", metrics["qwen_raw"]["recall_at_5"],
          "CENTERED", metrics["qwen_centered"]["recall_at_5"],
          "DELTA", effects["delta_vs_qwen_raw"],
          "W/L", effects["wins_vs_qwen_raw"], effects["losses_vs_qwen_raw"], flush=True)


if __name__ == "__main__":
    main()

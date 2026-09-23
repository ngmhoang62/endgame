#!/usr/bin/env python
"""Gold-only centroid head with frozen alternate-evidence AIT cross-encoder scores.

The extra scores come from Stage06A0's registered AIT reranker on two
retrieval-aligned windows. They are raw model outputs, not distilled targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT = ROOT / "reports/stage07t_gold_centroid_alt_evidence"
CACHE = ROOT / "cache/stage07t_gold_centroid_alt_evidence"
ALT = ROOT / "cache/stage06a0_evidence_packaging/ce/full15/token_full_overlap50__title0.f32.npy"
ALT_META = ALT.parent / "token_full_overlap50__title0.json"


def augmented_features(x: np.ndarray) -> np.ndarray:
    meta = json.loads(ALT_META.read_text(encoding="utf-8"))
    contract = meta["contract"]
    if (meta.get("status") != "PASS" or meta.get("completed") != len(x)
            or contract.get("nq") != len(x) or contract.get("depth") != 15
            or contract.get("mode") != "token_full_overlap50"
            or contract.get("title_on") is not False):
        raise RuntimeError("Stage06A0 alternate evidence cache contract drift")
    raw = np.load(ALT)
    if raw.shape != (len(x), 15, 2) or not np.isfinite(raw).all():
        raise RuntimeError("Stage06A0 alternate CE score shape/value drift")
    z = ait.zrows(raw.transpose(2, 0, 1).reshape(2 * len(x), 15)).reshape(2, len(x), 15).transpose(1, 2, 0)
    hi = np.maximum(raw[:, :, 0], raw[:, :, 1])
    hi_z = ait.zrows(hi)
    old_z = x[:, :15, 42]
    extra = np.zeros((*x.shape[:2], 6), np.float32)
    extra[:, :15, 0] = z[:, :, 0]
    extra[:, :15, 1] = z[:, :, 1]
    extra[:, :15, 2] = hi_z
    extra[:, :15, 3] = np.abs(z[:, :, 0] - z[:, :, 1])
    extra[:, :15, 4] = hi_z - old_z
    extra[:, :15, 5] = 1.0
    return np.concatenate((x, extra), axis=2)


def perq(rank, qids, golds, docs):
    return np.asarray([len(set(golds[q]) & {docs[int(d)] for d in rank[i, :5]}) / len(golds[q])
                       for i, q in enumerate(qids)], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=276)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    qids, golds, folds, stress, docs, short, x45, x57, y, qe, cent = ait.make_world()
    x = augmented_features(x57)
    score = np.empty((len(qids), ait.FULL), np.float32)
    base = np.empty_like(score)
    fold_meta = {}
    for outer, fn in enumerate(folds):
        tr, held, btr, bhe = sw.inner_crossfit_prior(x45, y, folds, qids, fn)
        pred, meta = ait.fit_predict(outer, tr, held, btr, short, x, y, qe, cent, args)
        combined = ait.zrows(bhe)
        combined[:, :ait.DEPTH] += meta["alpha"] * pred
        score[held] = combined
        base[held] = bhe
        fold_meta[fn] = meta
    baseline = ait.rank(short, base)
    original_score = np.load(ait.CACHE / "gold_head_oof_scores30.f32.npy")
    original = ait.rank(short, original_score)
    candidate = ait.rank(short, score)
    metrics = {
        "ce_lr": b1.eval_rank(baseline, qids, golds, docs, folds, stress),
        "centroid": b1.eval_rank(original, qids, golds, docs, folds, stress),
        "alt_evidence_centroid": b1.eval_rank(candidate, qids, golds, docs, folds, stress),
    }
    if abs(metrics["ce_lr"]["overall"]["recall_at_5"] - ait.BASE_R) > 3e-6:
        raise RuntimeError("CE-LR baseline parity failed")
    oldq, newq = perq(original, qids, golds, docs), perq(candidate, qids, golds, docs)
    old, new = metrics["centroid"], metrics["alt_evidence_centroid"]
    delta = new["overall"]["recall_at_5"] - old["overall"]["recall_at_5"]
    effect = {"delta_vs_centroid": delta, "wins": int((newq > oldq).sum()),
              "losses": int((newq < oldq).sum()),
              "fold_deltas": {f: new["per_fold"][f]["recall_at_5"] - old["per_fold"][f]["recall_at_5"] for f in folds}}
    decision = ("PROMOTE_FOR_PRIVATE" if new["overall"]["recall_at_5"] >= .95
                and effect["wins"] >= effect["losses"]
                and new["overall"]["multi_gold_recall_at_5"] >= old["overall"]["multi_gold_recall_at_5"]
                else "KILL")
    np.save(CACHE / "alt_evidence_centroid_oof_scores30.f32.npy", score)
    report = {"schema": "stage07t.gold_centroid_alt_evidence.v1", "teacher_used": False,
              "distillation_used": False, "supervision": "gold_labels_only",
              "alternate_score_source": str(ALT.relative_to(ROOT)).replace("\\", "/"),
              "feature_names": ["alt_ait_z", "alt_lal_z", "alt_max_z", "alt_disagreement",
                                "alt_max_minus_old_ce_z", "alt_available"],
              "metrics": metrics, "effect": effect, "fold_meta": fold_meta, "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "R", new["overall"]["recall_at_5"],
          "delta", delta, "W/L", effect["wins"], effect["losses"], flush=True)


if __name__ == "__main__":
    main()

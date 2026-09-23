#!/usr/bin/env python
"""Gold-supervised AIT centroid head with fold-clean document propensity.

Adds two label-memory features that Stage07O did not see. Each training query
gets document statistics from the other three outer-train folds; the held fold
gets statistics from all four outer-train folds. No teacher/distillation input.
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
import src.stage07_local.run_gold_doc_propensity_oof as prop

OUT = ROOT / "reports/stage07s_gold_centroid_propensity"
CACHE = ROOT / "cache/stage07s_gold_centroid_propensity"


def augment(outer, qids, golds, folds, docs, short, X57, counts, indices):
    names = [f for f in folds if f != outer]
    x = np.empty((len(qids), ait.FULL, X57.shape[2] + 2), np.float32)
    x[:, :, :X57.shape[2]] = X57
    x[:, :, X57.shape[2]:] = 0
    for f in names:
        other = [z for z in names if z != f]
        g = sum((counts[z][0] for z in other), np.zeros(len(docs), np.float32))
        e = sum((counts[z][1] for z in other), np.zeros(len(docs), np.float32))
        x[indices[f], :, X57.shape[2]:] = prop.propensity(short[indices[f]], g, e, 3)
    g = sum((counts[z][0] for z in names), np.zeros(len(docs), np.float32))
    e = sum((counts[z][1] for z in names), np.zeros(len(docs), np.float32))
    x[indices[outer], :, X57.shape[2]:] = prop.propensity(short[indices[outer]], g, e, 4)
    return x


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
    qids, golds, folds, stress, docs, short, X45, X57, y, qe, cent = ait.make_world()
    q2i = {q: i for i, q in enumerate(qids)}
    d2i = {d: i for i, d in enumerate(docs)}
    indices = {f: np.asarray([q2i[q] for q in folds[f]], np.int32) for f in folds}
    counts = {f: prop.counts_for_fold(indices[f], short, qids, golds, d2i, len(docs))
              for f in folds}
    final = np.empty((len(qids), ait.FULL), np.float32)
    base = np.empty_like(final)
    fold_meta = {}
    for outer_num, fn in enumerate(folds):
        tr, held, btr, bhe = sw.inner_crossfit_prior(X45, y, folds, qids, fn)
        X59 = augment(fn, qids, golds, folds, docs, short, X57, counts, indices)
        pred, meta = ait.fit_predict(outer_num, tr, held, btr, short, X59, y, qe, cent, args)
        score = ait.zrows(bhe)
        score[:, :ait.DEPTH] += meta["alpha"] * pred
        final[held] = score
        base[held] = bhe
        fold_meta[fn] = meta
        print(f"[{fn}] candidate_dim={X59.shape[-1]} train={len(tr)} held={len(held)}", flush=True)
    rank = ait.rank(short, final)
    base_rank = ait.rank(short, base)
    original = np.load(ait.CACHE / "gold_head_oof_scores30.f32.npy")
    original_rank = ait.rank(short, original)
    metrics = {"ce_lr": b1.eval_rank(base_rank, qids, golds, docs, folds, stress),
               "ait_centroid": b1.eval_rank(original_rank, qids, golds, docs, folds, stress),
               "centroid_propensity": b1.eval_rank(rank, qids, golds, docs, folds, stress)}
    if abs(metrics["ce_lr"]["overall"]["recall_at_5"] - ait.BASE_R) > 3e-6:
        raise RuntimeError("CE-LR parity failed")
    apq = perq(original_rank, qids, golds, docs)
    cpq = perq(rank, qids, golds, docs)
    delta = metrics["centroid_propensity"]["overall"]["recall_at_5"] - metrics["ait_centroid"]["overall"]["recall_at_5"]
    effects = {"delta_vs_ait_centroid": delta,
               "wins": int((cpq > apq).sum()), "losses": int((cpq < apq).sum()),
               "fold_deltas": {f: metrics["centroid_propensity"]["per_fold"][f]["recall_at_5"] -
                               metrics["ait_centroid"]["per_fold"][f]["recall_at_5"] for f in folds}}
    decision = ("PROMOTE_FOR_PRIVATE" if metrics["centroid_propensity"]["overall"]["recall_at_5"] >= .95
                and effects["wins"] >= effects["losses"]
                and metrics["centroid_propensity"]["overall"]["multi_gold_recall_at_5"] >= metrics["ait_centroid"]["overall"]["multi_gold_recall_at_5"]
                else "KILL")
    np.save(CACHE / "centroid_propensity_oof_scores30.f32.npy", final)
    report = {"schema": "stage07s.gold_centroid_propensity.v1", "teacher_used": False,
              "distillation_used": False, "supervision": "gold_labels_only",
              "propensity_training_policy": "exclude own fold from outer-train, exclude outer-held always",
              "metrics": metrics, "effect": effects, "fold_meta": fold_meta,
              "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "R", metrics["centroid_propensity"]["overall"]["recall_at_5"],
          "delta", delta, "W/L", effects["wins"], effects["losses"], flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Strict outer OOF document relevance propensity from gold labels only.

For each training row, propensity is estimated from other training folds, so
its own gold label never enters its feature. Held queries use outer-train folds.
This adds information absent from the existing 45D retrieval/CE feature set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT = ROOT / "reports/stage07r_gold_doc_propensity"
BASE_R = ait.BASE_R


def make_lr():
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=.15, class_weight="balanced",
                                               solver="lbfgs", max_iter=300, tol=1e-5))])


def counts_for_fold(ids, short, qids, golds, doc_to_idx, ndocs):
    exposure = np.bincount(short[ids].reshape(-1), minlength=ndocs).astype(np.float32)
    targets = []
    for qi in ids:
        targets.extend(doc_to_idx[d] for d in set(golds[qids[int(qi)]]) if d in doc_to_idx)
    gold = np.bincount(targets, minlength=ndocs).astype(np.float32)
    return gold, exposure


def propensity(short_rows, gold, exposure, source_folds):
    # Rescale counts to the four-fold outer-train population before smoothing.
    factor = 4.0 / source_folds
    g = gold[short_rows] * factor
    e = exposure[short_rows] * factor
    log_frequency = np.log1p(g)
    log_rate = np.log((g + 0.5) / (e + 10.0))
    return np.stack((log_frequency, log_rate), axis=-1).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qids, _, golds, folds, stress, docs, _ = b1.load_world()
    short, _ = b1.load_shortlist(len(qids))
    sources = b1.load_sources(len(qids))
    ce = np.load(b1.CACHE / "oof_top30_ce_scores.f32.npy")
    if ce.shape != short.shape or int(np.load(b1.CACHE / "oof_top30_done.u1.npy").sum()) != len(qids):
        raise RuntimeError("CE cache incomplete")
    Xflat, _, _, names = b1.build_source_ce_features(short, ce, sources)
    X45 = Xflat.reshape(len(qids), 30, 45).astype(np.float32)
    y = qg.labels_for(short, qids, golds, docs).astype(np.uint8)
    q2i = {q: i for i, q in enumerate(qids)}
    d2i = {d: i for i, d in enumerate(docs)}
    fold_names = list(folds)
    indices = {f: np.asarray([q2i[q] for q in folds[f]], np.int32) for f in fold_names}
    counts = {f: counts_for_fold(indices[f], short, qids, golds, d2i, len(docs))
              for f in fold_names}
    base_oof = np.empty((len(qids), 30), np.float32)
    prop_oof = np.empty_like(base_oof)
    stats = {}
    for outer in fold_names:
        train_folds = [f for f in fold_names if f != outer]
        tr = np.concatenate([indices[f] for f in train_folds])
        held = indices[outer]
        train_extra = np.empty((len(tr), 30, 2), np.float32)
        offsets = {}
        start = 0
        for f in train_folds:
            count = len(indices[f]); offsets[f] = slice(start, start + count); start += count
        for f in train_folds:
            others = [x for x in train_folds if x != f]
            gold = sum((counts[x][0] for x in others), np.zeros(len(docs), np.float32))
            exposure = sum((counts[x][1] for x in others), np.zeros(len(docs), np.float32))
            train_extra[offsets[f]] = propensity(short[indices[f]], gold, exposure, 3)
        gold = sum((counts[x][0] for x in train_folds), np.zeros(len(docs), np.float32))
        exposure = sum((counts[x][1] for x in train_folds), np.zeros(len(docs), np.float32))
        held_extra = propensity(short[held], gold, exposure, 4)
        xtr = np.concatenate((X45[tr], train_extra), axis=2)
        xhe = np.concatenate((X45[held], held_extra), axis=2)
        baseline = make_lr().fit(X45[tr].reshape(-1, 45), y[tr].reshape(-1))
        model = make_lr().fit(xtr.reshape(-1, 47), y[tr].reshape(-1))
        base_oof[held] = baseline.decision_function(X45[held].reshape(-1, 45)).reshape(len(held), 30)
        prop_oof[held] = model.decision_function(xhe.reshape(-1, 47)).reshape(len(held), 30)
        stats[outer] = {"train_queries": len(tr), "held_queries": len(held),
                        "held_nonzero_gold_frequency_fraction": float((held_extra[..., 0] > 0).mean())}
        print(f"[{outer}] trained and scored", flush=True)
    br = ait.rank(short, base_oof)
    pr = ait.rank(short, prop_oof)
    bm = b1.eval_rank(br, qids, golds, docs, folds, stress)
    pm = b1.eval_rank(pr, qids, golds, docs, folds, stress)
    if abs(bm["overall"]["recall_at_5"] - BASE_R) > 3e-6:
        raise RuntimeError("baseline OOF parity failed")
    bq = np.asarray([len(set(golds[q]) & {docs[int(d)] for d in br[i, :5]}) / len(golds[q])
                     for i, q in enumerate(qids)])
    pq = np.asarray([len(set(golds[q]) & {docs[int(d)] for d in pr[i, :5]}) / len(golds[q])
                     for i, q in enumerate(qids)])
    delta = pm["overall"]["recall_at_5"] - bm["overall"]["recall_at_5"]
    effects = {"delta": delta, "wins": int((pq > bq).sum()), "losses": int((pq < bq).sum()),
               "fold_deltas": {f: pm["per_fold"][f]["recall_at_5"] - bm["per_fold"][f]["recall_at_5"]
                               for f in fold_names}}
    report = {"schema": "stage07r.gold_doc_propensity.v1", "teacher_used": False,
              "distillation_used": False, "training_feature_policy": "inner fold excluded",
              "feature_names": names + ["gold_doc_log_frequency", "gold_doc_log_propensity"],
              "baseline": bm, "candidate": pm, "effect": effects, "fold_stats": stats,
              "decision": "PROMOTE" if pm["overall"]["recall_at_5"] >= .95 and delta >= .002 and effects["wins"] >= effects["losses"] else "KILL"}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.save(OUT / "propensity_oof_scores30.f32.npy", prop_oof)
    print("DECISION", report["decision"], "R", pm["overall"]["recall_at_5"],
          "delta", delta, "W/L", effects["wins"], effects["losses"], flush=True)


if __name__ == "__main__":
    main()

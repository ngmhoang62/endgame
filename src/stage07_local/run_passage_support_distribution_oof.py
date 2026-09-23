#!/usr/bin/env python
"""Gold-only strict OOF using top-3 passage support from frozen AIT embeddings.

New information: the second/third relevant region and the number of regions
that nearly match the best region. Stage03 uses only parent max retrieval score.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage06_evidence.benchmark_evidence_packaging as ev
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage07_local.run_gold_centroid_head_oof as ait

CACHE = ROOT / "cache/stage07w_passage_support_distribution"
OUT = ROOT / "reports/stage07w_passage_support_distribution"
VIEWS = ("ait_atomic", "ait_coarse")
WIDTH = 6


def compute(short, docs, qemb, device):
    import torch

    cache = CACHE / "support_features.f32.npy"
    meta_path = CACHE / "support_features.json"
    if cache.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "PASS" and meta.get("views") == list(VIEWS):
            result = np.load(cache)
            if result.shape == (len(short), short.shape[1], WIDTH * len(VIEWS)) and np.isfinite(result).all():
                print("[support] cache hit", flush=True)
                return result
        raise RuntimeError("support cache contract drift")
    CACHE.mkdir(parents=True, exist_ok=True)
    result = np.zeros((len(short), short.shape[1], WIDTH * len(VIEWS)), np.float32)
    tq = torch.from_numpy(np.array(qemb, copy=True)).to(device)
    t0 = time.perf_counter()
    for vi, name in enumerate(VIEWS):
        emb, st, en = ev.load_ranges(name, docs)
        te = torch.from_numpy(np.array(emb, copy=True)).to(device)
        print(f"[support] {name} regions={len(emb)}", flush=True)
        for qi in range(len(short)):
            cand = short[qi]
            starts = st[cand]
            lens = en[cand] - starts
            if np.any(lens <= 0):
                raise RuntimeError("candidate without region")
            inds = np.concatenate([np.arange(a, a + n, dtype=np.int64) for a, n in zip(starts, lens)])
            with torch.inference_mode():
                scores = (te[torch.as_tensor(inds, device=device)] @ tq[qi]).float().cpu().numpy()
            pos = 0
            for j, n0 in enumerate(lens):
                n = int(n0)
                vals = scores[pos:pos + n]
                pos += n
                top = np.sort(vals)[-3:][::-1]
                best, second, third = float(top[0]), float(top[min(1, len(top) - 1)]), float(top[min(2, len(top) - 1)])
                result[qi, j, vi * WIDTH:(vi + 1) * WIDTH] = (
                    best, second, third, best - second,
                    np.log1p(n), np.log1p(int(np.sum(vals >= best - .025))))
            if (qi + 1) % 250 == 0 or qi + 1 == len(short):
                print(f"[support] {name} {qi + 1}/{len(short)} elapsed_min={(time.perf_counter() - t0) / 60:.1f}", flush=True)
        del te
        torch.cuda.empty_cache()
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite support features")
    np.save(cache, result)
    meta_path.write_text(json.dumps({"schema": "stage07w.support.v1", "status": "PASS", "views": list(VIEWS),
                                    "shape": list(result.shape), "teacher_used": False,
                                    "distillation_used": False}, indent=2) + "\n", encoding="utf-8")
    return result


def make_features(raw):
    n, k, m = raw.shape
    out = raw.copy()
    # Across-candidate z-score makes passage support comparable within a query.
    for vi in range(len(VIEWS)):
        off = vi * WIDTH
        for j in (0, 1, 2, 3, 5):
            v = raw[:, :, off + j]
            sd = v.std(1, keepdims=True)
            out[:, :, off + j] = (v - v.mean(1, keepdims=True)) / np.where(sd < 1e-6, 1., sd)
    return out


def perq(rank, qids, golds, docs):
    return np.asarray([len(set(golds[q]) & {docs[int(d)] for d in rank[i, :5]}) / len(golds[q])
                       for i, q in enumerate(qids)], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    qids, golds, folds, stress, docs, short, x45, _, y, qe, _ = ait.make_world()
    raw = compute(short, docs, qe, args.device)
    x = np.concatenate((x45, make_features(raw)), axis=2)
    base = np.empty((len(qids), ait.FULL), np.float32)
    final = np.empty_like(base)
    q2i = {q: i for i, q in enumerate(qids)}
    for fn in folds:
        held = np.asarray([q2i[q] for q in folds[fn]], np.int32)
        train = np.asarray([q2i[q] for f in folds if f != fn for q in folds[f]], np.int32)
        base[held] = sw.fit_lr_score(x45, y, train, held)
        final[held] = sw.fit_lr_score(x, y, train, held)
    br = ait.rank(short, base)
    cr = ait.rank(short, final)
    bm = b1.eval_rank(br, qids, golds, docs, folds, stress)
    cm = b1.eval_rank(cr, qids, golds, docs, folds, stress)
    if abs(bm["overall"]["recall_at_5"] - ait.BASE_R) > 3e-6:
        raise RuntimeError("CE-LR baseline parity failed")
    bq, cq = perq(br, qids, golds, docs), perq(cr, qids, golds, docs)
    effect = {"delta_recall": cm["overall"]["recall_at_5"] - bm["overall"]["recall_at_5"],
              "wins": int((cq > bq).sum()), "losses": int((cq < bq).sum()),
              "fold_deltas": {f: cm["per_fold"][f]["recall_at_5"] - bm["per_fold"][f]["recall_at_5"] for f in folds}}
    decision = ("PROMOTE_FOR_PRIVATE" if cm["overall"]["recall_at_5"] >= .95
                and effect["wins"] >= effect["losses"]
                and cm["overall"]["multi_gold_recall_at_5"] >= bm["overall"]["multi_gold_recall_at_5"]
                else "KILL")
    np.save(CACHE / "support_lr_oof_scores30.f32.npy", final)
    report = {"schema": "stage07w.passage_support_distribution.v1", "teacher_used": False,
              "distillation_used": False, "supervision": "gold_labels_only",
              "feature_names": [f"{v}_{f}" for v in VIEWS for f in
                                ("top1_z", "top2_z", "top3_z", "gap12_z", "log_region_count", "nearbest_count_z")],
              "baseline": bm, "candidate": cm, "effect": effect, "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "R", cm["overall"]["recall_at_5"],
          "delta", effect["delta_recall"], "W/L", effect["wins"], effect["losses"], flush=True)


if __name__ == "__main__":
    main()

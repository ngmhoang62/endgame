#!/usr/bin/env python
"""Gold-only LAL semantic head and fixed AIT/LAL ensemble, strict outer OOF.

Uses raw frozen vnlegal-lal query/region embeddings, no teacher artifacts.
The 50/50 combination of two independently trained semantic worlds is fixed
before evaluation; no tuning on held-fold labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage02_candidate_generation.screen_vietnamese_retrievers as s2
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT = ROOT / "reports/stage07p_lal_gold_centroid"
CACHE = ROOT / "cache/stage07p_lal_gold_centroid"
REGION = ROOT / "cache/stage02b4_vi_screen/vnlegal_lal/region_embeddings.f32.npy"
QUERY = ROOT / "cache/stage02b4_vi_screen/vnlegal_lal/query_embeddings.f32.npy"
CENTROID = CACHE / "lal_parent_centroids.f32.npy"
META = CACHE / "lal_parent_centroids.json"


def parent_centroids(docs):
    CACHE.mkdir(parents=True, exist_ok=True)
    source_sha = ait.sha256(REGION)
    if CENTROID.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        c = np.load(CENTROID)
        if meta.get("source_sha256") == source_sha and c.shape == (len(docs), 1024):
            return np.asarray(c, np.float32)
        raise RuntimeError("LAL centroid cache contract mismatch")
    import torch
    geom = s2.load_geometry()
    if list(map(str, geom["doc_ids"])) != list(map(str, docs)):
        raise RuntimeError("document order drift")
    pidx = np.asarray(geom["parent_index"], np.int64)
    reg = np.load(REGION, mmap_mode="r")
    if reg.shape != (len(pidx), 1024):
        raise RuntimeError("LAL region shape drift")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sums = torch.zeros((len(docs), 1024), dtype=torch.float32, device=dev)
    counts = torch.zeros((len(docs), 1), dtype=torch.float32, device=dev)
    for st in range(0, len(reg), 4096):
        en = min(st + 4096, len(reg))
        e = torch.from_numpy(np.asarray(reg[st:en], np.float32).copy()).to(dev)
        ii = torch.from_numpy(pidx[st:en].copy()).to(dev)
        sums.index_add_(0, ii, e)
        counts.index_add_(0, ii, torch.ones((en-st, 1), device=dev))
    c = torch.nn.functional.normalize(sums / counts.clamp_min(1), dim=1).cpu().numpy()
    np.save(CENTROID, c.astype(np.float32))
    META.write_text(json.dumps({"source": str(REGION.relative_to(ROOT)).replace("\\", "/"),
                                "source_sha256": source_sha, "docs": len(docs),
                                "aggregation": "mean(region embeddings) then L2 normalize",
                                "teacher_used": False}, indent=2) + "\n", encoding="utf-8")
    return c.astype(np.float32)


def per_query(rank, qids, golds, docs):
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
    print("[1/4] load fold-clean CE world and raw LAL embeddings", flush=True)
    qids, golds, folds, stress, docs, short, X45, X, y, _, _ = ait.make_world()
    qe = np.asarray(np.load(QUERY, mmap_mode="r"), np.float32)
    if qe.shape != (len(qids), 1024) or not np.isfinite(qe).all():
        raise RuntimeError("LAL query embedding drift")
    cent = parent_centroids(docs)
    print("[2/4] train LAL-only gold head across strict folds", flush=True)
    base_oof = np.zeros((len(qids), ait.FULL), np.float32)
    lal_oof = np.zeros_like(base_oof)
    fold_meta = {}
    for outer, fn in enumerate(folds):
        tr, held, btr, bhe = sw.inner_crossfit_prior(X45, y, folds, qids, fn)
        base_oof[held] = bhe
        residual, meta = ait.fit_predict(outer, tr, held, btr, short, X, y, qe, cent, args)
        score = ait.zrows(bhe)
        score[:, :ait.DEPTH] += meta["alpha"] * residual
        lal_oof[held] = score
        fold_meta[fn] = meta
    np.save(CACHE / "lal_gold_oof_scores30.f32.npy", lal_oof)
    ait_oof = np.load(ait.CACHE / "gold_head_oof_scores30.f32.npy")
    if ait_oof.shape != lal_oof.shape:
        raise RuntimeError("AIT OOF score shape mismatch")
    combo = .5 * ait.zrows(ait_oof) + .5 * ait.zrows(lal_oof)
    print("[3/4] evaluate frozen LAL and equal-weight AIT/LAL", flush=True)
    scores = {"baseline": base_oof, "ait": ait_oof, "lal": lal_oof, "ait_lal_fixed": combo}
    ranks = {key: ait.rank(short, value) for key, value in scores.items()}
    metrics = {key: b1.eval_rank(value, qids, golds, docs, folds, stress)
               for key, value in ranks.items()}
    if abs(metrics["baseline"]["overall"]["recall_at_5"] - ait.BASE_R) > 3e-6:
        raise RuntimeError("CE-LR baseline parity failure")
    perq = {key: per_query(value, qids, golds, docs) for key, value in ranks.items()}
    effects = {}
    for key in ("lal", "ait_lal_fixed"):
        anchor = "baseline" if key == "lal" else "ait"
        delta = metrics[key]["overall"]["recall_at_5"] - metrics[anchor]["overall"]["recall_at_5"]
        effects[key] = {"anchor": anchor, "delta": delta,
                        "wins": int((perq[key] > perq[anchor]).sum()),
                        "losses": int((perq[key] < perq[anchor]).sum()),
                        "fold_deltas": {f: metrics[key]["per_fold"][f]["recall_at_5"] -
                                        metrics[anchor]["per_fold"][f]["recall_at_5"] for f in folds}}
    promote_lal = effects["lal"]["delta"] >= .002 and effects["lal"]["wins"] >= effects["lal"]["losses"]
    promote_combo = effects["ait_lal_fixed"]["delta"] >= .001 and effects["ait_lal_fixed"]["wins"] >= effects["ait_lal_fixed"]["losses"]
    decision = "PROMOTE_FIXED_ENSEMBLE" if promote_combo else "PROMOTE_LAL" if promote_lal else "KILL"
    report = {"schema": "stage07p.lal_gold_centroid.v1", "teacher_used": False,
              "distillation_used": False, "supervision": "competition gold labels only",
              "lal_query_sha256": ait.sha256(QUERY), "lal_region_sha256": ait.sha256(REGION),
              "lal_centroid_sha256": ait.sha256(CENTROID), "metrics": metrics,
              "effects": effects, "fold_meta": fold_meta, "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[4/4] DECISION", decision, "LAL", effects["lal"],
          "FIXED_COMBO", effects["ait_lal_fixed"], flush=True)


if __name__ == "__main__":
    main()

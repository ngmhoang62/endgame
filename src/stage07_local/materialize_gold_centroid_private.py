#!/usr/bin/env python
"""Train the gold-only centroid head on all labeled folds and materialize private K5.

Runs only after the strict OOF promotion gate. Every training prior is
cross-fitted; the private prior uses the frozen Stage03B1 full-train LR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage04_private.materialize_endgame_private as pm
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_breakthrough.run_residual_setwise_oof as sw
from src.stage07_local.run_gold_centroid_head_oof import (
    CACHE, CENTROID, OUT, DEPTH, FULL, fit_predict, load_raw_centroids,
    make_model, make_world, rank, sha256, zrows,
)

PRIVATE_CACHE = ROOT / "cache/stage04_private_submission_v1_2"
SUB = ROOT / "submissions/endgame_20260923"
CHECKPOINT = CACHE / "gold_centroid_fulltrain.pt"


def train_full(args):
    report = json.loads((OUT / "OOF_REPORT.json").read_text(encoding="utf-8"))
    if report.get("decision") != "PROMOTE_FOR_PRIVATE":
        raise RuntimeError("OOF gate did not pass")
    qids, golds, folds, stress, docs, short, X45, X, y, qe, cent = make_world()
    prior = np.empty((len(qids), FULL), np.float32)
    q2i = {q: i for i, q in enumerate(qids)}
    allidx = np.arange(len(qids), dtype=np.int32)
    for fn, ids in folds.items():
        held = np.asarray([q2i[q] for q in ids], np.int32)
        tr = allidx[np.isin(allidx, held, invert=True)]
        prior[held] = sw.fit_lr_score(X45, y, tr, held)
    if not np.isfinite(prior).all():
        raise RuntimeError("fulltrain crossfit prior incomplete")
    fit_predict(100, allidx, np.empty(0, np.int32), prior, short, X,
                y, qe, cent, args, save_checkpoint=CHECKPOINT)
    return report


def private_features():
    qids, questions = pm.load_private()
    docs, _ = pm.load_corpus()
    short = np.load(PRIVATE_CACHE / "private_shortlist_top30_idx.npy")
    ce = np.load(PRIVATE_CACHE / "reranker/scores.npy")
    done = np.load(PRIVATE_CACHE / "reranker/done.npy")
    if short.shape != (len(qids), FULL) or ce.shape != short.shape or int(done.sum()) != len(qids):
        raise RuntimeError("private shortlist or CE cache incomplete")
    sources = {}
    for name in b1.SOURCE_NAMES:
        paths = PRIVATE_CACHE / "retrieval"
        ix = np.load(paths / f"{name}_idx.npy", mmap_mode="r")
        sc = np.load(paths / f"{name}_scores.npy", mmap_mode="r")
        if ix.shape != (len(qids), 100) or sc.shape != ix.shape:
            raise RuntimeError(f"private {name} source drift")
        sources[name] = (ix, sc)
    xflat, _, _, names45 = b1.build_source_ce_features(short, ce, sources)
    X45 = xflat.reshape(len(qids), FULL, 45)
    title = qg.make_title_features(qids, questions, short, qg.load_names(docs))
    X = np.concatenate([X45, title], axis=2).astype(np.float32)
    qe = np.load(PRIVATE_CACHE / "query/aiteam.npy")
    if qe.shape != (len(qids), 1024) or not np.isfinite(qe).all():
        raise RuntimeError("private AIT query embedding drift")
    cent = load_raw_centroids(len(docs))
    return qids, docs, short, X45, X, qe, cent, names45


def score_private(short, X, qe, cent, model, args):
    import torch

    device = torch.device(args.device)
    tq = torch.from_numpy(np.asarray(qe, np.float32).copy()).to(device)
    td = torch.from_numpy(np.asarray(cent, np.float32).copy()).to(device)
    tx = torch.from_numpy(X[:, :DEPTH].copy()).to(device)
    result = np.empty((len(short), DEPTH), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(short), args.batch_queries):
            end = min(start + args.batch_queries, len(short))
            row = short[start:end, :DEPTH]
            n = len(row)
            di = torch.as_tensor(row.reshape(-1), device=device)
            q = tq[start:end, None, :].expand(-1, DEPTH, -1).reshape(n * DEPTH, 1024)
            d = td[di]
            c = tx[start:end].reshape(n * DEPTH, -1)
            result[start:end] = model(q, d, c).reshape(n, DEPTH).cpu().numpy()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=276)
    args = ap.parse_args()
    print("[1/5] strict OOF gate and fulltrain gold-only head", flush=True)
    oof = train_full(args)
    print("[2/5] frozen private feature world", flush=True)
    qids, docs, short, X45, X, qe, cent, names45 = private_features()
    obj = joblib.load(b1.CACHE / "fulltrain_ce_fusion.joblib")
    expected_names = json.loads((ROOT / "reports/stage03b1_aiteam_reranker/RERANK_OOF.json").read_text(encoding="utf-8"))["ce_lr"]["features"]
    if obj["feature_names"] != expected_names or names45 != expected_names:
        raise RuntimeError("feature contract drift")
    prior = obj["model"].decision_function(X45.reshape(-1, 45)).reshape(len(qids), FULL)
    base_rank = rank(short, prior)
    existing = json.loads((ROOT / "submissions/endgame_20260922/ENDGAME_PRIVATE_CE_LR_K5.json").read_text(encoding="utf-8"))
    if any([docs[int(d)] for d in base_rank[i, :5]] != existing[q]["answer"] for i, q in enumerate(qids)):
        raise RuntimeError("private CE-LR baseline prediction parity failed")
    print("[3/5] private head inference", flush=True)
    import torch
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if ckpt.get("teacher_used") is not False or ckpt.get("supervision") != "gold_labels_only":
        raise RuntimeError("checkpoint lineage drift")
    model = make_model(X.shape[-1]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    residual = score_private(short, X, qe, cent, model, args)
    final_score = zrows(prior)
    final_score[:, :DEPTH] += float(model.alpha().detach().cpu()) * residual
    final_rank = rank(short, final_score)
    print("[4/5] package deterministic submission", flush=True)
    if len(qids) != 2080 or len(set(qids)) != len(qids):
        raise RuntimeError("private qid population drift")
    valid = set(docs)
    answers = {}
    for i, q in enumerate(qids):
        ans = [docs[int(d)] for d in final_rank[i, :5]]
        if len(ans) != 5 or len(set(ans)) != 5 or any(d not in valid for d in ans):
            raise RuntimeError(f"invalid answer for {q}")
        answers[q] = {"answer": ans}
    SUB.mkdir(parents=True, exist_ok=True)
    label = "GOLD_CENTROID_HEAD_K5"
    jp = SUB / (label + ".json")
    zp = SUB / (label + ".zip")
    payload = (json.dumps(answers, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    jp.write_bytes(payload)
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(info, payload)
    with zipfile.ZipFile(zp) as z:
        if z.read("submission.json") != payload:
            raise RuntimeError("ZIP/JSON parity failed")
    manifest = {"schema": "dsc2026.endgame.stage07o.private.v1", "status": "READY_FOR_PRIVATE_SUBMISSION",
                "private_sha256": sha256(pm.PRIVATE), "queries": len(qids),
                "checkpoint_sha256": sha256(CHECKPOINT), "centroid_sha256": sha256(CENTROID),
                "oof_recall_at5": oof["candidate"]["overall"]["recall_at_5"],
                "private_baseline_prediction_parity": True,
                "teacher_used": False, "distillation_used": False,
                "json": str(jp.relative_to(ROOT)).replace("\\", "/"), "json_sha256": sha256(jp),
                "zip": str(zp.relative_to(ROOT)).replace("\\", "/"), "zip_sha256": sha256(zp),
                "set_churn_vs_ce_lr": sum(set(answers[q]["answer"]) != set(existing[q]["answer"]) for q in qids)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PRIVATE_SUBMISSION_REPORT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[5/5] READY", zp, "set_churn", manifest["set_churn_vs_ce_lr"], flush=True)


if __name__ == "__main__":
    main()

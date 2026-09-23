#!/usr/bin/env python
"""Fulltrain and private K5 for Stage07S after strict OOF clears 0.95."""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_gold_centroid_propensity_oof as stage
import src.stage07_local.run_gold_doc_propensity_oof as prop
import src.stage07_local.materialize_gold_centroid_private as mp

SUB = ROOT / "submissions/endgame_20260923"
CK = stage.CACHE / "centroid_propensity_fulltrain.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=276)
    args = ap.parse_args()
    report = json.loads((stage.OUT / "OOF_REPORT.json").read_text(encoding="utf-8"))
    if report["decision"] != "PROMOTE_FOR_PRIVATE" or report["teacher_used"] is not False:
        raise RuntimeError(f"Stage07S OOF gate failed: {report['decision']}")
    print("[1/5] gold-only fulltrain with crossfit propensity", flush=True)
    qids, golds, folds, _, docs, short, X45, X57, y, qe, cent = ait.make_world()
    q2i = {q: i for i, q in enumerate(qids)}
    d2i = {d: i for i, d in enumerate(docs)}
    indices = {f: np.asarray([q2i[q] for q in folds[f]], np.int32) for f in folds}
    counts = {f: prop.counts_for_fold(indices[f], short, qids, golds, d2i, len(docs)) for f in folds}
    X59 = np.empty((len(qids), ait.FULL, 59), np.float32)
    X59[:, :, :57] = X57
    prior = np.empty((len(qids), ait.FULL), np.float32)
    allidx = np.arange(len(qids), dtype=np.int32)
    for f in folds:
        others = [x for x in folds if x != f]
        gold = sum((counts[x][0] for x in others), np.zeros(len(docs), np.float32))
        exposure = sum((counts[x][1] for x in others), np.zeros(len(docs), np.float32))
        X59[indices[f], :, 57:] = prop.propensity(short[indices[f]], gold, exposure, 4)
        tr = allidx[np.isin(allidx, indices[f], invert=True)]
        prior[indices[f]] = sw.fit_lr_score(X45, y, tr, indices[f])
    if not np.isfinite(X59).all() or not np.isfinite(prior).all():
        raise RuntimeError("fulltrain feature incomplete")
    ait.fit_predict(100, allidx, np.empty(0, np.int32), prior, short, X59,
                    y, qe, cent, args, save_checkpoint=CK)
    print("[2/5] private features and label-only document statistics", flush=True)
    pqids, pdocs, pshort, PX45, PX57, pqe, pcent, names45 = mp.private_features()
    if pdocs != docs:
        raise RuntimeError("private corpus document order drift")
    gold = sum((counts[f][0] for f in folds), np.zeros(len(docs), np.float32))
    exposure = sum((counts[f][1] for f in folds), np.zeros(len(docs), np.float32))
    PX59 = np.concatenate((PX57, prop.propensity(pshort, gold, exposure, 5)), axis=2)
    obj = joblib.load(b1.CACHE / "fulltrain_ce_fusion.joblib")
    if obj["feature_names"] != names45:
        raise RuntimeError("private CE-LR feature contract drift")
    pprior = obj["model"].decision_function(PX45.reshape(-1, 45)).reshape(len(pqids), 30)
    existing = json.loads((ROOT / "submissions/endgame_20260922/ENDGAME_PRIVATE_CE_LR_K5.json").read_text(encoding="utf-8"))
    base_rank = ait.rank(pshort, pprior)
    if any([pdocs[int(d)] for d in base_rank[i, :5]] != existing[q]["answer"] for i, q in enumerate(pqids)):
        raise RuntimeError("private baseline parity failed")
    print("[3/5] private model inference", flush=True)
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    if ck.get("teacher_used") is not False or ck.get("supervision") != "gold_labels_only" or ck.get("candidate_dim") != 59:
        raise RuntimeError("checkpoint lineage or feature drift")
    model = ait.make_model(59).to(args.device)
    model.load_state_dict(ck["state_dict"])
    residual = mp.score_private(pshort, PX59, pqe, pcent, model, args)
    final = ait.zrows(pprior)
    final[:, :ait.DEPTH] += float(model.alpha().detach().cpu()) * residual
    ranks = ait.rank(pshort, final)
    print("[4/5] deterministic K5 submission", flush=True)
    if len(pqids) != 2080 or len(set(pqids)) != 2080:
        raise RuntimeError("private query population drift")
    valid = set(pdocs)
    answers = {}
    for i, q in enumerate(pqids):
        ans = [pdocs[int(d)] for d in ranks[i, :5]]
        if len(ans) != 5 or len(set(ans)) != 5 or any(d not in valid for d in ans):
            raise RuntimeError(f"invalid answer: {q}")
        answers[q] = {"answer": ans}
    SUB.mkdir(parents=True, exist_ok=True)
    label = "GOLD_CENTROID_PROPENSITY_K5"
    jp = SUB / f"{label}.json"
    zp = SUB / f"{label}.zip"
    payload = (json.dumps(answers, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    jp.write_bytes(payload)
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(info, payload)
    with zipfile.ZipFile(zp) as z:
        if z.namelist() != ["submission.json"] or z.read("submission.json") != payload:
            raise RuntimeError("ZIP parity failed")
    manifest = {"schema": "stage07s.private.v1", "status": "READY_FOR_SUBMISSION",
                "private_sha256": ait.sha256(mp.pm.PRIVATE), "queries": len(pqids),
                "teacher_used": False, "distillation_used": False,
                "oof_recall_at5": report["metrics"]["centroid_propensity"]["overall"]["recall_at_5"],
                "checkpoint_sha256": ait.sha256(CK),
                "zip": str(zp.relative_to(ROOT)).replace("\\", "/"), "zip_sha256": ait.sha256(zp),
                "json_sha256": ait.sha256(jp),
                "set_churn_vs_ce_lr": sum(set(answers[q]["answer"]) != set(existing[q]["answer"]) for q in pqids)}
    (stage.OUT / "PRIVATE_SUBMISSION_REPORT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[5/5] READY", zp, "sha", manifest["zip_sha256"], flush=True)


if __name__ == "__main__":
    main()

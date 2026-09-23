#!/usr/bin/env python
"""Fulltrain and materialize Stage07P only after its strict OOF promotion gate."""
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
import src.stage07_local.materialize_gold_centroid_private as mp
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_lal_gold_centroid_oof as lal

SUB = ROOT / "submissions/endgame_20260923"
CK = lal.CACHE / "lal_gold_fulltrain.pt"


def train_lal(args, report):
    qids, _, folds, _, docs, short, X45, X, y, _, _ = ait.make_world()
    lq = np.asarray(np.load(lal.QUERY, mmap_mode="r"), np.float32).copy()
    lc = lal.parent_centroids(docs)
    prior = np.empty((len(qids), ait.FULL), np.float32)
    q2i = {q: i for i, q in enumerate(qids)}
    allidx = np.arange(len(qids), dtype=np.int32)
    for _, ids in folds.items():
        held = np.asarray([q2i[q] for q in ids], np.int32)
        tr = allidx[np.isin(allidx, held, invert=True)]
        prior[held] = sw.fit_lr_score(X45, y, tr, held)
    if not np.isfinite(prior).all():
        raise RuntimeError("fulltrain crossfit prior incomplete")
    ait.fit_predict(100, allidx, np.empty(0, np.int32), prior, short, X,
                    y, lq, lc, args, save_checkpoint=CK)
    return report


def load_head(path, cdim, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("teacher_used") is not False or ck.get("supervision") != "gold_labels_only":
        raise RuntimeError("checkpoint provenance failed")
    model = ait.make_model(cdim).to(device)
    model.load_state_dict(ck["state_dict"])
    return model


def write_submission(label, score, qids, docs, short, existing, manifest):
    rank = ait.rank(short, score)
    valid = set(docs)
    answers = {}
    for i, qid in enumerate(qids):
        row = [docs[int(x)] for x in rank[i, :5]]
        if len(row) != 5 or len(set(row)) != 5 or any(x not in valid for x in row):
            raise RuntimeError(f"invalid answer at {qid}")
        answers[qid] = {"answer": row}
    SUB.mkdir(parents=True, exist_ok=True)
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
    entry = {"zip": str(zp.relative_to(ROOT)).replace("\\", "/"),
             "zip_sha256": ait.sha256(zp), "json_sha256": ait.sha256(jp),
             "set_churn_vs_baseline": sum(set(answers[q]["answer"]) != set(existing[q]["answer"]) for q in qids)}
    manifest["submissions"][label] = entry
    print("READY", zp, "set_churn", entry["set_churn_vs_baseline"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=276)
    args = ap.parse_args()
    report = json.loads((lal.OUT / "OOF_REPORT.json").read_text(encoding="utf-8"))
    decision = report["decision"]
    if decision not in ("PROMOTE_FIXED_ENSEMBLE", "PROMOTE_LAL") or report["teacher_used"] is not False:
        raise RuntimeError(f"LAL OOF gate failed: {decision}")
    print("[1/4] train full gold-only LAL head", flush=True)
    train_lal(args, report)
    print("[2/4] private AIT/LAL feature and baseline parity", flush=True)
    qids, docs, short, X45, X, aq, ac, names45 = mp.private_features()
    if len(qids) != 2080:
        raise RuntimeError("private population drift")
    prior = joblib.load(b1.CACHE / "fulltrain_ce_fusion.joblib")["model"].decision_function(
        X45.reshape(-1, 45)).reshape(len(qids), ait.FULL)
    existing = json.loads((ROOT / "submissions/endgame_20260922/ENDGAME_PRIVATE_CE_LR_K5.json").read_text(encoding="utf-8"))
    rank0 = ait.rank(short, prior)
    if any([docs[int(d)] for d in rank0[i, :5]] != existing[q]["answer"] for i, q in enumerate(qids)):
        raise RuntimeError("baseline private parity failed")
    lq = np.asarray(np.load(mp.PRIVATE_CACHE / "query/lal.npy"), np.float32).copy()
    lc = lal.parent_centroids(docs)
    if lq.shape != (len(qids), 1024):
        raise RuntimeError("private LAL query embedding drift")
    print("[3/4] private head inference", flush=True)
    lm = load_head(CK, X.shape[-1], args.device)
    lr = mp.score_private(short, X, lq, lc, lm, args)
    lal_score = ait.zrows(prior)
    lal_score[:, :ait.DEPTH] += float(lm.alpha().detach().cpu()) * lr
    manifest = {"schema": "stage07p.private.v1", "private_sha256": ait.sha256(mp.pm.PRIVATE),
                "queries": len(qids), "teacher_used": False, "distillation_used": False,
                "oof_decision": decision, "lal_checkpoint_sha256": ait.sha256(CK),
                "lal_centroid_sha256": ait.sha256(lal.CENTROID), "submissions": {}}
    if decision == "PROMOTE_LAL":
        write_submission("GOLD_LAL_CENTROID_K5", lal_score, qids, docs, short, existing, manifest)
    else:
        am = load_head(mp.CHECKPOINT, X.shape[-1], args.device)
        ar = mp.score_private(short, X, aq, ac, am, args)
        ait_score = ait.zrows(prior)
        ait_score[:, :ait.DEPTH] += float(am.alpha().detach().cpu()) * ar
        old = json.loads((ROOT / "submissions/endgame_20260923/GOLD_CENTROID_HEAD_K5.json").read_text(encoding="utf-8"))
        ait_rank = ait.rank(short, ait_score)
        if any([docs[int(d)] for d in ait_rank[i, :5]] != old[q]["answer"] for i, q in enumerate(qids)):
            raise RuntimeError("AIT gold private parity failed")
        combo = .5 * ait.zrows(ait_score) + .5 * ait.zrows(lal_score)
        write_submission("GOLD_AIT_LAL_FIXED_K5", combo, qids, docs, short, existing, manifest)
        manifest["ait_checkpoint_sha256"] = ait.sha256(mp.CHECKPOINT)
    print("[4/4] save manifest", flush=True)
    (lal.OUT / "PRIVATE_SUBMISSION_REPORT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

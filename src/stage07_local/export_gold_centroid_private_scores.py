#!/usr/bin/env python
"""Reconstruct the already submitted clean centroid scores for fixed fusion tests."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch

from src.stage07_local import materialize_gold_centroid_private as m


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "reports/stage07o_gold_centroid_head/PRIVATE_SUBMISSION_REPORT.json").read_text(encoding="utf-8"))
    if report.get("teacher_used") is not False or report.get("distillation_used") is not False:
        raise RuntimeError("Centroid lineage drift")
    ckpt = torch.load(m.CHECKPOINT, map_location="cpu", weights_only=True)
    if ckpt.get("teacher_used") is not False or ckpt.get("supervision") != "gold_labels_only":
        raise RuntimeError("Centroid checkpoint lineage drift")
    qids, docs, short, X45, X, qe, cent, names45 = m.private_features()
    obj = joblib.load(m.b1.CACHE / "fulltrain_ce_fusion.joblib")
    if obj["feature_names"] != names45:
        raise RuntimeError("Prior feature contract drift")
    prior = obj["model"].decision_function(X45.reshape(-1, 45)).reshape(len(qids), m.FULL)
    model = m.make_model(X.shape[-1]).to("cuda")
    model.load_state_dict(ckpt["state_dict"])
    args = type("Args", (), {"device": "cuda", "batch_queries": 32})()
    residual = m.score_private(short, X, qe, cent, model, args)
    final = m.zrows(prior)
    final[:, :m.DEPTH] += float(model.alpha().detach().cpu()) * residual
    rank = m.rank(short, final)
    path = root / report["zip"]
    with zipfile.ZipFile(path) as z:
        submitted = json.loads(z.read("submission.json"))
    for qi, q in enumerate(qids):
        if [docs[int(d)] for d in rank[qi, :5]] != submitted[q]["answer"]:
            raise RuntimeError(f"Submitted centroid ranking mismatch at {q}")
    out = root / "cache/stage07o_gold_centroid_head/private_score30.f32.npy"
    np.save(out, final.astype(np.float32))
    print(f"READY {out} shape={final.shape} submitted_zip_parity=True", flush=True)


if __name__ == "__main__":
    main()

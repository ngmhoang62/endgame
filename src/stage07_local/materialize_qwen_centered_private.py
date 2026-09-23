#!/usr/bin/env python
"""Correct the top20-only Qwen logit offset; retain all model/ranking contracts.

Prepares a candidate ZIP from existing A100 fulltrain private scores. The
candidate is not certified until the held-fold calibration check completes.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "artifacts/stage07n_private_payload"
SCORES = ROOT / "cache/stage07n_private_analysis"
SUB = ROOT / "submissions/endgame_20260923"
REPORT = ROOT / "reports/stage07n_centered_qwen_private"
RAW_ZIP = SCORES / "GOLD_QWEN06B_FULLTRAIN_PRIVATE_K5.zip"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def zrows(x):
    x = np.asarray(x, np.float32)
    sd = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(sd < 1e-5, 1., sd)


def materialize(qids, docs, short, score):
    out = {}
    for i, qid in enumerate(qids):
        row = [docs[int(short[i, j])] for j in np.argsort(-score[i], kind="stable")[:5]]
        if len(row) != 5 or len(set(row)) != 5:
            raise RuntimeError(f"invalid top5 {qid}")
        out[qid] = {"answer": row}
    if len(out) != 2080:
        raise RuntimeError("query count drift")
    return out


def main():
    manifest = json.loads((PAYLOAD / "MANIFEST.json").read_text(encoding="utf-8"))
    meta = json.loads((SCORES / "TRAIN_META.json").read_text(encoding="utf-8"))
    contract = json.loads((SCORES / "SCORE_CONTRACT.json").read_text(encoding="utf-8"))
    if (manifest["private_sha256"] != "9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc"
            or manifest["teacher_used"] or manifest["distillation_used"]):
        raise RuntimeError("private population/provenance drift")
    if (meta.get("status") != "PASS" or meta.get("training_folds") != [0, 1, 2, 3, 4]
            or meta.get("model_id") != "Qwen/Qwen3-Reranker-0.6B"
            or meta.get("teacher_used") is not False or meta.get("distillation_used") is not False):
        raise RuntimeError("A100 checkpoint provenance drift")
    if (contract.get("private_sha256") != manifest["private_sha256"]
            or contract.get("evidence_sha256") != manifest["files"]["evidence_top20.pkl"]["sha256"]
            or contract.get("train_meta_sha256") != sha(SCORES / "TRAIN_META.json")
            or contract.get("maxlen") != meta["config"]["maxlen"]):
        raise RuntimeError("score contract drift")

    qids = json.loads((PAYLOAD / "qids.json").read_text(encoding="utf-8"))
    docs = json.loads((PAYLOAD / "doc_ids.json").read_text(encoding="utf-8"))
    short = np.load(PAYLOAD / "shortlist_top30.i4.npy")
    prior = np.load(PAYLOAD / "ce_lr_prior.f32.npy")
    qwen = np.load(SCORES / "qwen_scores_top20.f32.npy")
    done = np.load(SCORES / "qwen_done.u1.npy")
    if (len(qids), len(set(qids)), len(docs), short.shape, prior.shape,
            qwen.shape, done.shape, int(done.sum())) != (2080, 2080, 8512,
                                                        (2080, 30), (2080, 30),
                                                        (2080, 20), (2080,), 2080):
        raise RuntimeError("shape/completeness drift")
    if not np.isfinite(qwen).all() or not np.isfinite(prior).all():
        raise RuntimeError("nonfinite score")

    alpha = float(meta["alpha"])
    base = zrows(prior)
    raw = base.copy()
    raw[:, :20] += alpha * qwen
    baseline = materialize(qids, docs, short, raw)
    with zipfile.ZipFile(RAW_ZIP) as z:
        original = json.loads(z.read("submission.json"))
    if baseline != original or sha(RAW_ZIP) != "d213fd17ff26e2adf991703299dc520d3fee530ef786d0180bde230f96041f5a":
        raise RuntimeError("original public submission parity failure")

    centered = base.copy()
    centered[:, :20] += alpha * (qwen - qwen.mean(axis=1, keepdims=True))
    output = materialize(qids, docs, short, centered)
    label = "GOLD_QWEN06B_CENTERED_PRIVATE_K5"
    SUB.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    jp = SUB / f"{label}.json"
    zp = SUB / f"{label}.zip"
    data = (json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    jp.write_bytes(data)
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zp, "w", compresslevel=9) as z:
        z.writestr(info, data)
    with zipfile.ZipFile(zp) as z:
        if z.namelist() != ["submission.json"] or z.read("submission.json") != data:
            raise RuntimeError("ZIP payload mismatch")
    changed = sum(set(output[q]["answer"]) != set(baseline[q]["answer"]) for q in qids)
    report = {"schema": "stage07n.centered_private_candidate.v1",
              "status": "CANDIDATE_AWAITING_HELD_FOLD_VALIDATION",
              "hypothesis": "remove arbitrary per-query Qwen logit offset before top20-only fusion",
              "teacher_used": False, "distillation_used": False,
              "private_sha256": manifest["private_sha256"], "queries": len(qids),
              "alpha": alpha, "changed_top5_sets_vs_raw": changed,
              "raw_top5_outside20": int((np.argsort(-raw, axis=1, kind="stable")[:, :5] >= 20).sum()),
              "centered_top5_outside20": int((np.argsort(-centered, axis=1, kind="stable")[:, :5] >= 20).sum()),
              "original_public_recall": .946889227,
              "zip": str(zp.relative_to(ROOT)).replace("\\", "/"), "zip_sha256": sha(zp)}
    (REPORT / "CANDIDATE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CANDIDATE", zp, "changed_sets", changed,
          "outside20", report["raw_top5_outside20"], "->", report["centered_top5_outside20"],
          "SHA256", report["zip_sha256"], flush=True)


if __name__ == "__main__":
    main()

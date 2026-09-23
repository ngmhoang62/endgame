#!/usr/bin/env python
"""Resume-safe private inference for the gold-supervised Stage07N checkpoint.

Upload stage07n_private_payload.tar.gz and this file to /content, then run this
script only after the Stage07N DEV/CERT report promotes the frozen checkpoint.
The script imports the exact Stage07N prompt, tokenizer and score path.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("/content")
ARCHIVE = BASE / "stage07n_private_payload.tar.gz"
PAYLOAD = BASE / "stage07n_private_payload"
GATE = BASE / "drive/MyDrive/DSC2026/stage07n_t4_gold_qwen"
TRAIN = Path(os.environ.get("STAGE07N_CHECKPOINT_ROOT", str(GATE)))
OUT = TRAIN / "private_inference"

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def qwen_features(tok, pairs, maxlen):
    pre = tok.encode(PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    avail = maxlen - len(pre) - len(suf)
    if avail < 256:
        raise RuntimeError("maxlen too small")
    raw = tok(pairs, padding=False, truncation=True, max_length=avail,
              add_special_tokens=False, return_attention_mask=False)
    return [{"input_ids": pre + row + suf} for row in raw["input_ids"]]


def yes_no_ids(tok):
    yes = tok.encode("yes", add_special_tokens=False)
    no = tok.encode("no", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError(f"yes/no tokenizer drift: {yes}, {no}")
    return yes[0], no[0]


def score_pairs(model, tok, yes_id, no_id, pairs, maxlen, amp_dtype):
    batch = tok.pad(qwen_features(tok, pairs, maxlen), padding=True,
                    pad_to_multiple_of=8, return_tensors="pt")
    batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=amp_dtype):
        out = model.model(**batch, return_dict=True)
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        return (logits[:, yes_id] - logits[:, no_id]).float()


def zrows(x):
    x = np.asarray(x, np.float32)
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    return (x - mu) / np.where(sd < 1e-5, 1., sd)


def load_saved_model():
    ck = TRAIN / "train012_model"
    tok = AutoTokenizer.from_pretrained(ck, padding_side="left", use_fast=True)
    kwargs = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(ck, attn_implementation="sdpa", **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(ck, **kwargs)
    model.config.use_cache = False
    return tok, model.to("cuda").eval()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_payload():
    if not PAYLOAD.is_dir():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            for item in tar.getmembers():
                p = Path(item.name)
                if p.is_absolute() or ".." in p.parts or p.parts[0] != PAYLOAD.name:
                    raise RuntimeError(f"unsafe payload member {item.name}")
            tar.extractall(BASE)
    manifest = json.loads((PAYLOAD / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["private_sha256"] != "9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc":
        raise RuntimeError("unexpected private population")
    if manifest["teacher_used"] or manifest["distillation_used"]:
        raise RuntimeError("ineligible evidence payload")
    for name, entry in manifest["files"].items():
        if sha(PAYLOAD / name) != entry["sha256"]:
            raise RuntimeError(f"payload hash mismatch: {name}")
    qids = json.loads((PAYLOAD / "qids.json").read_text(encoding="utf-8"))
    docs = json.loads((PAYLOAD / "doc_ids.json").read_text(encoding="utf-8"))
    short = np.load(PAYLOAD / "shortlist_top30.i4.npy")
    prior = np.load(PAYLOAD / "ce_lr_prior.f32.npy")
    with (PAYLOAD / "evidence_top20.pkl").open("rb") as f:
        evidence = pickle.load(f)
    if (len(qids), len(docs), short.shape, prior.shape, len(evidence)) != (2080, 8512, (2080, 30), (2080, 30), 2080):
        raise RuntimeError("private payload shape drift")
    if any(len(row) != 20 for row in evidence) or not np.isfinite(prior).all():
        raise RuntimeError("private payload content drift")
    return manifest, qids, docs, short, prior, evidence


def verify_train():
    meta = json.loads((TRAIN / "TRAIN_META.json").read_text(encoding="utf-8"))
    report = json.loads((GATE / "REPORT.json").read_text(encoding="utf-8"))
    if meta.get("status") != "PASS" or meta.get("teacher_used") is not False or meta.get("distillation_used") is not False:
        raise RuntimeError("Stage07N training provenance gate failed")
    if report.get("decision") != "PROMOTE_TO_FULLTRAIN" or report["eligibility"].get("distillation_used") is not False:
        raise RuntimeError(f"Stage07N DEV/CERT gate failed: {report.get('decision', report.get('status'))}")
    if TRAIN == GATE and abs(float(meta["alpha"]) - float(report["alpha"])) > 1e-7:
        raise RuntimeError("frozen alpha mismatch")
    if TRAIN != GATE and meta.get("training_folds") != [0, 1, 2, 3, 4]:
        raise RuntimeError("fulltrain checkpoint must contain all five gold folds")
    if not (TRAIN / "train012_model").is_dir():
        raise RuntimeError("Stage07N checkpoint missing")
    return meta, report


def scores(model, tok, evidence, maxlen, contract):
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "qwen_scores_top20.f32.npy"
    dp = OUT / "qwen_done.u1.npy"
    cp = OUT / "SCORE_CONTRACT.json"
    if len({sp.exists(), dp.exists(), cp.exists()}) != 1:
        raise RuntimeError("partial score cache")
    if sp.exists():
        if json.loads(cp.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("score cache contract mismatch")
        result = np.lib.format.open_memmap(sp, mode="r+")
        done = np.lib.format.open_memmap(dp, mode="r+")
    else:
        result = np.lib.format.open_memmap(sp, mode="w+", dtype=np.float32, shape=(len(evidence), 20))
        done = np.lib.format.open_memmap(dp, mode="w+", dtype=np.uint8, shape=(len(evidence),))
        done[:] = 0
        done.flush()
        cp.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if result.shape != (len(evidence), 20) or done.shape != (len(evidence),):
        raise RuntimeError("score cache shape drift")
    yes_id, no_id = yes_no_ids(tok)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.eval()
    start = time.time()
    with torch.inference_mode():
        for qi, row in enumerate(evidence):
            if done[qi]:
                continue
            s = score_pairs(model, tok, yes_id, no_id, row, maxlen, amp_dtype)
            result[qi] = s.float().cpu().numpy()
            done[qi] = 1
            if (qi + 1) % 25 == 0 or qi + 1 == len(evidence):
                result.flush()
                done.flush()
                elapsed = (time.time() - start) / 60
                print(f"[private] {int(done.sum())}/{len(evidence)} elapsed_min={elapsed:.1f}", flush=True)
    if int(done.sum()) != len(evidence) or not np.isfinite(result).all():
        raise RuntimeError("Qwen private scores incomplete")
    return np.asarray(result)


def materialize(manifest, meta, report, qids, docs, short, prior, score):
    final = zrows(prior)
    final[:, :20] += float(meta["alpha"]) * score
    output = {}
    valid = set(docs)
    for i, qid in enumerate(qids):
        order = np.argsort(-final[i], kind="stable")[:5]
        answer = [docs[int(short[i, j])] for j in order]
        if len(answer) != 5 or len(set(answer)) != 5 or any(x not in valid for x in answer):
            raise RuntimeError(f"invalid private answer: {qid}")
        output[qid] = {"answer": answer}
    label = "GOLD_QWEN06B_TRAIN012_PRIVATE_K5" if TRAIN == GATE else "GOLD_QWEN06B_FULLTRAIN_PRIVATE_K5"
    path = OUT / f"{label}.json"
    zp = OUT / f"{label}.zip"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = zipfile.ZipInfo("submission.json", date_time=(2026, 9, 23, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zp, "w", compresslevel=9) as z:
        z.writestr(info, path.read_bytes())
    with zipfile.ZipFile(zp) as z:
        if z.namelist() != ["submission.json"] or z.read("submission.json") != path.read_bytes():
            raise RuntimeError("ZIP payload mismatch")
    m = {"schema": "stage07n.private_submission.v1", "status": "READY_FOR_SUBMISSION",
         "private_sha256": manifest["private_sha256"], "queries": len(qids),
         "teacher_used": False, "distillation_used": False,
         "model_id": meta["model_id"], "training_folds": meta["training_folds"],
         "dev_delta": report["dev"]["delta_recall"], "cert_delta": report["cert"]["delta_recall"],
         "gate_report": str(GATE / "REPORT.json"),
         "frozen_alpha": meta["alpha"], "checkpoint_dir": str(TRAIN / "train012_model"),
         "json": str(path), "json_sha256": sha(path), "zip": str(zp), "zip_sha256": sha(zp)}
    (OUT / "PRIVATE_SUBMISSION_REPORT.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print("READY:", zp, "SHA256", m["zip_sha256"], flush=True)


def main():
    manifest, qids, docs, short, prior, evidence = load_payload()
    meta, report = verify_train()
    tok, model = load_saved_model()
    maxlen = int(meta["config"]["maxlen"])
    contract = {"schema": "stage07n.private_score_cache.v1",
                "private_sha256": manifest["private_sha256"],
                "evidence_sha256": manifest["files"]["evidence_top20.pkl"]["sha256"],
                "train_meta_sha256": sha(TRAIN / "TRAIN_META.json"),
                "maxlen": maxlen}
    score = scores(model, tok, evidence, maxlen, contract)
    materialize(manifest, meta, report, qids, docs, short, prior, score)


if __name__ == "__main__":
    main()

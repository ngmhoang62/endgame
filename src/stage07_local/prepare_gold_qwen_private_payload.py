#!/usr/bin/env python
"""Prepare teacher-free private evidence for Stage07N on the current Colab runtime.

Uses the same title + highest cosine AIT/LAL witness contract as the training
payload, with the frozen private query embeddings and Stage03B1 45D features.
No teacher target or distilled prediction is read.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage04_private.materialize_endgame_private as pm
import src.stage06_evidence.benchmark_evidence_packaging as ev
from src.stage07_breakthrough.prepare_stage07b_colab_payload import bundle, format_pair

PC = ROOT / "cache/stage04_private_submission_v1_2"
OUT = ROOT / "artifacts/stage07n_private_payload"
ARCHIVE = ROOT / "artifacts/stage07n_private_payload.tar.gz"
DEPTH = 20
FULL = 30


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def stable(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_private_world():
    qids, qs = pm.load_private()
    docs, _ = pm.load_corpus()
    short = np.load(PC / "private_shortlist_top30_idx.npy")
    ce = np.load(PC / "reranker/scores.npy")
    done = np.load(PC / "reranker/done.npy")
    if short.shape != (len(qids), FULL) or ce.shape != short.shape or int(done.sum()) != len(qids):
        raise RuntimeError("private shortlist/CE cache incomplete")
    sources = {}
    for name in b1.SOURCE_NAMES:
        ix = np.load(PC / "retrieval" / f"{name}_idx.npy", mmap_mode="r")
        sc = np.load(PC / "retrieval" / f"{name}_scores.npy", mmap_mode="r")
        if ix.shape != (len(qids), 100) or sc.shape != ix.shape:
            raise RuntimeError(f"private {name} retrieval drift")
        sources[name] = (ix, sc)
    flat, _, _, names45 = b1.build_source_ce_features(short, ce, sources)
    expected = json.loads((ROOT / "reports/stage03b1_aiteam_reranker/RERANK_OOF.json").read_text(encoding="utf-8"))["ce_lr"]["features"]
    if names45 != expected:
        raise RuntimeError("45D feature contract drift")
    X45 = flat.reshape(len(qids), FULL, 45)
    model = joblib.load(b1.CACHE / "fulltrain_ce_fusion.joblib")["model"]
    prior = model.decision_function(X45.reshape(-1, 45)).reshape(len(qids), FULL)
    baseline = json.loads((ROOT / "submissions/endgame_20260922/ENDGAME_PRIVATE_CE_LR_K5.json").read_text(encoding="utf-8"))
    for i, q in enumerate(qids):
        order = np.lexsort((short[i], -prior[i]))
        if [docs[int(d)] for d in short[i, order[:5]]] != baseline[q]["answer"]:
            raise RuntimeError(f"private CE-LR baseline parity failed at {q}")
    aq = np.load(PC / "query/aiteam.npy", mmap_mode="r")
    lq = np.load(PC / "query/lal.npy", mmap_mode="r")
    if aq.shape != (len(qids), 1024) or lq.shape != (len(qids), 1024):
        raise RuntimeError("private query embedding shape drift")
    return qids, qs, docs, short, X45, aq, lq


def witness_cache(qids, short):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / "witness_view.i1.npy", OUT / "witness_row.i4.npy",
             OUT / "witness_done.u1.npy", OUT / "WITNESS_META.json"]
    contract = {"schema": "stage07n.private_witness.v1", "private_sha256": sha(pm.PRIVATE),
                "shortlist_sha256": sha(PC / "private_shortlist_top30_idx.npy"),
                "query_aiteam_sha256": sha(PC / "query/aiteam.npy"),
                "query_lal_sha256": sha(PC / "query/lal.npy"),
                "depth": DEPTH,
                "views": {v: {"chunks_sha256": sha(x["chunks"]),
                              "embedding_shape": list(np.load(x["emb"], mmap_mode="r").shape)}
                          for v, x in ev.VIEWS.items()}}
    ch = stable(contract)
    exists = [p.exists() for p in paths]
    if any(exists) and not all(exists):
        raise RuntimeError("partial private witness cache")
    if all(exists):
        meta = json.loads(paths[3].read_text(encoding="utf-8"))
        if meta.get("contract_hash") != ch:
            raise RuntimeError("private witness contract mismatch")
        vv, rr, done = [np.lib.format.open_memmap(p, mode="r+") for p in paths[:3]]
    else:
        vv = np.lib.format.open_memmap(paths[0], mode="w+", dtype=np.int8, shape=(len(qids), DEPTH, 2))
        rr = np.lib.format.open_memmap(paths[1], mode="w+", dtype=np.int32, shape=(len(qids), DEPTH, 2))
        done = np.lib.format.open_memmap(paths[2], mode="w+", dtype=np.uint8, shape=(len(qids),))
        done[:] = 0
        for arr in (vv, rr, done):
            arr.flush()
        paths[3].write_text(json.dumps({"contract_hash": ch, "contract": contract,
                                         "status": "IN_PROGRESS", "completed": 0}, indent=2) + "\n", encoding="utf-8")
    if vv.shape != (len(qids), DEPTH, 2) or rr.shape != vv.shape or done.shape != (len(qids),):
        raise RuntimeError("private witness shape drift")
    return vv, rr, done, paths[3], contract, ch


def select_witnesses(qids, docs, short, aq, lq):
    vv, rr, done, mp, contract, ch = witness_cache(qids, short)
    if int(done.sum()) == len(qids):
        print("[witness] complete cache hit", flush=True)
        return np.asarray(vv), np.asarray(rr), list(ev.VIEWS)
    view_names = list(ev.VIEWS)
    stores = {}
    for name in view_names:
        stores[name] = ev.load_ranges(name, docs)
        print(f"[witness] loaded {name}", flush=True)
    families = {fam: [v for v in view_names if ev.VIEWS[v]["family"] == fam]
                for fam in ("ait", "lal")}
    for qi in range(len(qids)):
        if done[qi]:
            continue
        for pos, d0 in enumerate(short[qi, :DEPTH]):
            d = int(d0)
            for fi, fam in enumerate(("ait", "lal")):
                query = np.asarray(aq[qi] if fam == "ait" else lq[qi], np.float32)
                best = (-1e30, -1, -1)
                for view in families[fam]:
                    emb, starts, ends = stores[view]
                    start, end = int(starts[d]), int(ends[d])
                    sims = np.asarray(emb[start:end], np.float32) @ query
                    loc = int(np.argmax(sims))
                    score = float(sims[loc])
                    if score > best[0]:
                        best = (score, view_names.index(view), start + loc)
                vv[qi, pos, fi], rr[qi, pos, fi] = best[1], best[2]
        done[qi] = 1
        if (qi + 1) % 50 == 0 or qi + 1 == len(qids):
            for arr in (vv, rr, done):
                arr.flush()
            completed = int(done.sum())
            mp.write_text(json.dumps({"contract_hash": ch, "contract": contract,
                                      "status": "PASS" if completed == len(qids) else "IN_PROGRESS",
                                      "completed": completed}, indent=2) + "\n", encoding="utf-8")
            print(f"[witness] {completed}/{len(qids)}", flush=True)
    return np.asarray(vv), np.asarray(rr), view_names


def main():
    print("[1/5] private population, retrieval and CE-LR parity", flush=True)
    qids, qs, docs, short, X45, aq, lq = load_private_world()
    print("[2/5] select AIT/LAL semantic evidence", flush=True)
    vv, rr, views = select_witnesses(qids, docs, short, aq, lq)
    print("[3/5] materialize private evidence text", flush=True)
    texts = ev.load_selected_texts(vv, rr, views)
    names = ev.load_names(docs)
    evidence = []
    for qi, q in enumerate(qids):
        row = [format_pair(qs[q], bundle(qi, pos, int(d), names, vv, rr, views, texts))
               for pos, d in enumerate(short[qi, :DEPTH])]
        evidence.append(row)
        if (qi + 1) % 500 == 0 or qi + 1 == len(qids):
            print(f"[evidence] {qi+1}/{len(qids)}", flush=True)
    print("[4/5] save portable private payload", flush=True)
    np.save(OUT / "shortlist_top30.i4.npy", short.astype(np.int32))
    np.save(OUT / "X45.f32.npy", X45.astype(np.float32))
    lr = joblib.load(b1.CACHE / "fulltrain_ce_fusion.joblib")["model"]
    prior = lr.decision_function(X45.reshape(-1, 45)).reshape(len(qids), FULL).astype(np.float32)
    np.save(OUT / "ce_lr_prior.f32.npy", prior)
    (OUT / "qids.json").write_text(json.dumps(qids, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "doc_ids.json").write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "evidence_top20.pkl").open("wb") as f:
        pickle.dump(evidence, f, protocol=5)
    manifest = {"schema": "stage07n.private_payload.v1", "private_sha256": sha(pm.PRIVATE),
                "queries": len(qids), "depth": DEPTH, "full_depth": FULL,
                "teacher_used": False, "distillation_used": False,
                "evidence": "title + AIT/LAL max-cosine witness, same as train payload",
                "files": {}}
    for name in ("shortlist_top30.i4.npy", "X45.f32.npy", "ce_lr_prior.f32.npy", "qids.json", "doc_ids.json", "evidence_top20.pkl"):
        path = OUT / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for name in ["shortlist_top30.i4.npy", "X45.f32.npy", "ce_lr_prior.f32.npy", "qids.json", "doc_ids.json", "evidence_top20.pkl", "MANIFEST.json"]:
            tar.add(OUT / name, arcname="stage07n_private_payload/" + name)
    print("[5/5] READY", ARCHIVE, "bytes", ARCHIVE.stat().st_size, flush=True)


if __name__ == "__main__":
    main()

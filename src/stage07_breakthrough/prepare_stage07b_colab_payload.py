#!/usr/bin/env python
"""Build the ONLY large file that must be transferred to Colab for Stage07B.

Run on the LOCAL full ENDGAME repo:
    python src/stage07_breakthrough/prepare_stage07b_colab_payload.py

It reads the existing local caches, pre-materializes all top-20 evidence bundles,
and creates:
    artifacts/stage07b_colab_payload.tar.gz

The remote Colab runtime does NOT need the whole endgame repo.
"""
from __future__ import annotations
import hashlib, io, json, pickle, shutil, sys, tarfile, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage06_evidence.benchmark_evidence_packaging as a0

OUT=ROOT/"artifacts/stage07b_colab_payload"
ARCHIVE=ROOT/"artifacts/stage07b_colab_payload.tar.gz"
DEPTH=20
FULL=30

INSTRUCTION=(
    "Given a Vietnamese legal question, judge whether the candidate legal document "
    "contains provisions that are sufficient and directly relevant for answering the question. "
    "Consider legal scope, actors, actions, conditions, exceptions, procedures, sanctions, "
    "and explicit article or instrument references. Prefer legally applicable evidence over "
    "superficial lexical overlap."
)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def labels(short,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    y=np.zeros(short.shape,np.uint8)
    gc=np.zeros(len(qids),np.int16)
    for i,q in enumerate(qids):
        gs={d2i[d] for d in golds[q]}
        gc[i]=len(gs)
        y[i]=np.asarray([int(int(x) in gs) for x in short[i]],np.uint8)
    return y,gc

def bundle(li,pos,d,names,vv,rr,views,texts):
    parts=[f"[LEGAL DOCUMENT TITLE]\n{a0.clean_title(names[int(d)])}"]
    seen=set()
    for fi,label in enumerate(("SEMANTIC EVIDENCE A","SEMANTIC EVIDENCE B")):
        v=views[int(vv[li,pos,fi])]
        raw=str(texts[v][int(rr[li,pos,fi])]).strip()
        key=" ".join(raw.split())
        if key and key not in seen:
            seen.add(key);parts.append(f"[{label}]\n{raw}")
    return "\n\n".join(parts)

def format_pair(q,doc):
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {doc}"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("[1/7] frozen OOF world",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy")
    done=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy")
    if short.shape!=(len(qids),FULL) or ce.shape!=(len(qids),FULL) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03 caches incomplete")

    print("[2/7] build exact current 45D feature world",flush=True)
    Xflat,cq,cand,names45=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),FULL,-1).astype(np.float32)
    if X45.shape[-1]!=45:raise RuntimeError(f"45D drift {X45.shape}")

    print("[3/7] labels/folds",flush=True)
    y,gold_count=labels(short,qids,golds,docs)
    q2i={q:i for i,q in enumerate(qids)}
    fold_id=np.full(len(qids),-1,np.int8)
    for fi,fn in enumerate(sorted(folds)):
        for q in folds[fn]:fold_id[q2i[q]]=fi
    if np.any(fold_id<0):raise RuntimeError("fold coverage incomplete")

    np.save(OUT/"shortlist_top30.i4.npy",short.astype(np.int32))
    np.save(OUT/"X45.f32.npy",X45)
    np.save(OUT/"labels_top30.u1.npy",y)
    np.save(OUT/"gold_count.i2.npy",gold_count)
    np.save(OUT/"fold_id.i1.npy",fold_id)

    print("[4/7] select top20 semantic witnesses LOCALLY",flush=True)
    qidx=np.arange(len(qids),dtype=np.int32)
    s20=short[:,:DEPTH]
    vv,rr,sim,views=a0.select_witnesses("stage07b_portable_top20_v1",qidx,s20,docs)
    texts=a0.load_selected_texts(vv,rr,views)
    names=a0.load_names(docs)

    print("[5/7] materialize portable evidence strings",flush=True)
    evidence=[]
    t0=time.perf_counter()
    for qi,q in enumerate(qids):
        question=questions[q]
        row=[]
        for pos,d in enumerate(s20[qi]):
            row.append(format_pair(question,bundle(qi,pos,d,names,vv,rr,views,texts)))
        evidence.append(row)
        if (qi+1)%500==0 or qi+1==len(qids):
            print(f"  evidence {qi+1}/{len(qids)} elapsed={(time.perf_counter()-t0)/60:.1f}m",flush=True)
    with (OUT/"evidence_top20.pkl").open("wb") as f:
        pickle.dump(evidence,f,protocol=5)

    (OUT/"qids.json").write_text(json.dumps(qids,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={
        "schema":"dsc2026.endgame.stage07b.portable_payload.v1",
        "queries":len(qids),"depth":DEPTH,"full_depth":FULL,
        "candidate_feature_dim":45,
        "files":{},
        "claim":"self-contained Colab payload; no full ENDGAME repo required remotely"
    }
    print("[6/7] hashes",flush=True)
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!="MANIFEST.json":
            manifest["files"][p.name]={"bytes":p.stat().st_size,"sha256":sha(p)}
    (OUT/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Runtime scripts must already have been copied into local repo from this package.
    runtime_files=[
        ROOT/"src/stage07_breakthrough/stage07b_portable_common.py",
        ROOT/"src/stage07_breakthrough/run_qwen4b_teacher_portable_a100.py",
        ROOT/"src/stage07_breakthrough/run_qwen06b_student_portable_a100.py",
        ROOT/"src/stage07_breakthrough/shutdown_colab_runtime.py",
        ROOT/"src/stage07_breakthrough/run_stage07b_unattended.sh",
    ]
    missing=[str(p) for p in runtime_files if not p.is_file()]
    if missing:raise FileNotFoundError("Missing runtime files: "+repr(missing))

    print("[7/7] create ONE upload archive",flush=True)
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(ARCHIVE,"w:gz") as tar:
        for p in OUT.iterdir():
            if p.is_file():tar.add(p,arcname=f"stage07b/payload/{p.name}")
        for p in runtime_files:
            tar.add(p,arcname=f"stage07b/{p.name}")
    print("="*100)
    print("READY:",ARCHIVE)
    print(f"SIZE: {ARCHIVE.stat().st_size/2**20:.1f} MiB")
    print("Upload ONLY this .tar.gz to /content (plus keep the notebook locally as the Colab anchor).")
    print("="*100)

if __name__=="__main__":main()

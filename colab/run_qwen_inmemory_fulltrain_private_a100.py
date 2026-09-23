#!/usr/bin/env python
"""
Stage07X-B — Fulltrain Qwen0.6B and infer private WITHOUT reloading the model.

Use ONLY if Stage07X-A reports PROMOTE_INMEMORY_FULLTRAIN_PRIVATE.

Eligibility:
  original Qwen/Qwen3-Reranker-0.6B
  competition gold labels only
  no teacher / no distillation

Required:
  /content/stage07b/run_qwen06b_gold_supervised_l4.py
  /content/stage07b/payload/*
  /content/run_gold_qwen_private_colab.py
  /content/stage07n_private_payload/  OR /content/stage07n_private_payload.tar.gz

Run:
  python /content/run_qwen_inmemory_fulltrain_private_a100.py
"""
from __future__ import annotations

import argparse, gc, json, shutil, sys, time, zipfile
from pathlib import Path

import numpy as np
import torch

BASE=Path("/content")
sys.path.insert(0,str(BASE))
from stage07b import run_qwen06b_gold_supervised_l4 as n
import run_gold_qwen_private_colab as p

OUT=BASE/"drive/MyDrive/DSC2026/stage07x_qwen_inmemory_fulltrain"
DEPTH=20

def batch_private(model,tok,evidence,maxlen,batch_queries=4):
    yes_id,no_id=n.yes_no_ids(tok)
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    result=np.empty((len(evidence),DEPTH),np.float32)
    model.eval()
    t=time.time()
    with torch.inference_mode():
        for st in range(0,len(evidence),batch_queries):
            rows=evidence[st:st+batch_queries]
            pairs=[x for row in rows for x in row[:DEPTH]]
            feats=n.qwen_features(tok,pairs,maxlen)
            batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda",dtype=amp_dtype):
                s=n.last_score(model,batch,yes_id,no_id).float()
            result[st:st+len(rows)]=s.reshape(len(rows),DEPTH).cpu().numpy()
            if (st//batch_queries+1)%20==0 or st+len(rows)==len(evidence):
                print(f"[private] {min(st+len(rows),len(evidence))}/{len(evidence)} elapsed_min={(time.time()-t)/60:.1f}",flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--batch-queries",type=int,default=4)
    ap.add_argument("--force-retrain",action="store_true")
    args=ap.parse_args()

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A100 required")
    if args.force_retrain and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True,exist_ok=True)

    n.OUT=OUT
    n.TRAIN_FOLDS=(0,1,2,3,4)
    qids,short,X45,y,gold_count,fold_id,evidence=n.load_payload()
    cfg=argparse.Namespace(
        epochs=1,grad_accum=4,maxlen=1024,
        lr=8e-6,alpha_lr=8e-4,weight_decay=.01,
        hard_negs=4,max_negs=4,
        listwise_weight=.55,pair_weight=.25,bce_weight=.20,
        resid_l2=5e-4,gold_temp=1.0,margin=.4,
        hard_query_bonus=1.25,multi_bonus=.15,
        dev_min_delta=.002,seed=276,
    )
    print("[1/5] fulltrain all 6991, KEEP MODEL IN MEMORY",flush=True)
    tok,model,alpha,train_idx=n.train_model(qids,X45,y,fold_id,evidence,cfg)
    if len(train_idx)!=len(qids):
        raise RuntimeError("fulltrain population drift")

    print("[2/5] load frozen private payload",flush=True)
    manifest,pqids,docs,pshort,prior,pevidence=p.load_payload()
    if manifest.get("teacher_used") or manifest.get("distillation_used"):
        raise RuntimeError("private payload lineage failed")

    print("[3/5] private inference directly from in-memory model",flush=True)
    raw=batch_private(model,tok,pevidence,cfg.maxlen,args.batch_queries)
    np.save(OUT/"private_raw_inmemory_top20.f32.npy",raw)

    final=p.zrows(prior)
    final[:,:DEPTH]+=float(alpha)*raw
    output={}
    for i,q in enumerate(pqids):
        order=np.argsort(-final[i],kind="stable")[:5]
        ans=[docs[int(pshort[i,j])] for j in order]
        if len(ans)!=5 or len(set(ans))!=5:
            raise RuntimeError(f"invalid answer {q}")
        output[q]={"answer":ans}

    print("[4/5] deterministic zip",flush=True)
    label="GOLD_QWEN06B_INMEMORY_FULLTRAIN_PRIVATE_K5"
    jp=OUT/f"{label}.json"
    zp=OUT/f"{label}.zip"
    payload=(json.dumps(output,ensure_ascii=False,indent=2)+"\n").encode()
    jp.write_bytes(payload)
    info=zipfile.ZipInfo("submission.json",date_time=(2026,9,23,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o644<<16
    with zipfile.ZipFile(zp,"w",compresslevel=9) as z:
        z.writestr(info,payload)
    with zipfile.ZipFile(zp) as z:
        if z.namelist()!=["submission.json"] or z.read("submission.json")!=payload:
            raise RuntimeError("zip parity failed")

    report={
        "schema":"stage07x.qwen_inmemory_fulltrain_private.v1",
        "status":"READY_FOR_SUBMISSION",
        "teacher_used":False,"distillation_used":False,
        "model_id":n.MODEL_ID,
        "training_folds":[0,1,2,3,4],
        "queries":len(pqids),"mean_returned":5,
        "alpha":float(alpha),
        "private_sha256":manifest["private_sha256"],
        "inference_model_state":"same live in-memory object returned by fulltrain; no checkpoint reload",
        "zip":str(zp),"json":str(jp),
    }
    (OUT/"PRIVATE_REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print("[5/5] READY",zp,flush=True)

if __name__=="__main__":
    main()

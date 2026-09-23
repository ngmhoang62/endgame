#!/usr/bin/env python
"""
Stage07AD — FINAL SHOT: fresh gold-Qwen(train012) + private-winning W75.

Use the already-finished fresh Stage07X checkpoint. No retraining.

Required:
  /content/stage07b/run_qwen06b_gold_supervised_l4.py
  /content/stage07b/payload/*
  /content/w75_qwen_bridge.npz
  /content/stage07n_private_payload/ OR /content/stage07n_private_payload.tar.gz

Drive checkpoint:
  /content/drive/MyDrive/DSC2026/stage07x_qwen_persistence_probe/train012_model

Flow:
  1) infer only fold4 (fold3 raw is already cached);
  2) audit W75/Qwen complement on BOTH untouched folds 3 & 4;
  3) try conservative global blend + W75-top4 anchored boundary blend;
  4) only if a config improves the mean and does not materially damage either fold,
     infer private and materialize ONE final ZIP.

Gold labels only. No teacher. No distillation.
"""
from __future__ import annotations
import argparse, gc, json, pickle, sys, tarfile, time, zipfile
from pathlib import Path

import numpy as np
import torch

BASE=Path("/content")
sys.path.insert(0,str(BASE))
from stage07b import run_qwen06b_gold_supervised_l4 as n

PROBE=BASE/"drive/MyDrive/DSC2026/stage07x_qwen_persistence_probe"
BRIDGE=BASE/"w75_qwen_bridge.npz"
PRIVDIR=BASE/"stage07n_private_payload"
PRIVTAR=BASE/"stage07n_private_payload.tar.gz"
DEPTH=20

W75_F3=0.9579160705770149
W75_F4=0.9463519313304721

def zrows(x):
    x=np.asarray(x,np.float32)
    sd=x.std(1,keepdims=True)
    return (x-x.mean(1,keepdims=True))/np.where(sd<1e-6,1,sd)

def batch_infer(model,tok,evidence,idx,maxlen,bq=4):
    yes_id,no_id=n.yes_no_ids(tok)
    amp=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out=np.empty((len(idx),DEPTH),np.float32)
    model.eval()
    with torch.inference_mode():
        for st in range(0,len(idx),bq):
            ids=idx[st:st+bq]
            pairs=[]
            for qi in ids: pairs.extend(evidence[int(qi)][:DEPTH])
            feats=n.qwen_features(tok,pairs,maxlen)
            batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            with torch.autocast("cuda",dtype=amp):
                s=n.last_score(model,batch,yes_id,no_id).float()
            out[st:st+len(ids)]=s.reshape(len(ids),DEPTH).cpu().numpy()
            if (st//bq+1)%25==0 or st+len(ids)==len(idx):
                print(f"[infer] {min(st+len(ids),len(idx))}/{len(idx)}",flush=True)
    return out

def qwen_score(prior,raw,alpha):
    s=zrows(prior)
    s[:,:DEPTH]+=float(alpha)*raw
    return s.astype(np.float32)

def top5_indices(score):
    return np.argsort(-score,axis=1,kind="stable")[:,:5]

def anchored_top5(w75,blend,k):
    # Preserve W75 top-k exactly, fill remaining slots using blend.
    nrow=w75.shape[0]
    ow=np.argsort(-w75,axis=1,kind="stable")
    ob=np.argsort(-blend,axis=1,kind="stable")
    out=np.empty((nrow,5),np.int32)
    for i in range(nrow):
        chosen=list(map(int,ow[i,:k]))
        seen=set(chosen)
        for j in ob[i]:
            jj=int(j)
            if jj not in seen:
                chosen.append(jj); seen.add(jj)
                if len(chosen)==5: break
        out[i]=chosen
    return out

def recall_from_pos(pos,y,gold_count,idx):
    hits=np.take_along_axis(y[idx],pos,axis=1).sum(1)
    return float(np.mean(hits/np.asarray(gold_count[idx],np.float32)))

def eval_config(w75,qw,w,anchor,y,gold_count,idx):
    blend=(1-w)*zrows(w75)+w*zrows(qw)
    pos=top5_indices(blend) if anchor==0 else anchored_top5(w75,blend,anchor)
    return recall_from_pos(pos,y,gold_count,idx),pos

def load_private():
    if not PRIVDIR.is_dir():
        if not PRIVTAR.is_file(): raise FileNotFoundError(PRIVTAR)
        with tarfile.open(PRIVTAR,"r:gz") as tar:
            for item in tar.getmembers():
                p=Path(item.name)
                if p.is_absolute() or ".." in p.parts:
                    raise RuntimeError("unsafe private payload")
            tar.extractall(BASE)
    qids=json.loads((PRIVDIR/"qids.json").read_text(encoding="utf-8"))
    docs=json.loads((PRIVDIR/"doc_ids.json").read_text(encoding="utf-8"))
    short=np.load(PRIVDIR/"shortlist_top30.i4.npy")
    prior=np.load(PRIVDIR/"ce_lr_prior.f32.npy")
    with (PRIVDIR/"evidence_top20.pkl").open("rb") as f: ev=pickle.load(f)
    if len(qids)!=2080 or short.shape!=(2080,30) or prior.shape!=(2080,30) or len(ev)!=2080:
        raise RuntimeError("private payload shape drift")
    return qids,docs,short,prior,ev

def private_raw(model,tok,ev,maxlen,bq):
    p=PROBE/"private_train012_raw.f32.npy"
    if p.exists():
        a=np.load(p)
        if a.shape==(len(ev),20) and np.isfinite(a).all():
            print("[private] raw cache hit",p,flush=True); return a
    idx=np.arange(len(ev),dtype=np.int32)
    a=batch_infer(model,tok,ev,idx,maxlen,bq)
    np.save(p,a); return a

def package(label,pos,qids,docs,short):
    answers={}
    for i,q in enumerate(qids):
        row=[docs[int(short[i,j])] for j in pos[i]]
        if len(row)!=5 or len(set(row))!=5: raise RuntimeError(f"bad row {q}")
        answers[q]={"answer":row}
    jp=PROBE/f"{label}.json"; zp=PROBE/f"{label}.zip"
    payload=(json.dumps(answers,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    jp.write_bytes(payload)
    info=zipfile.ZipInfo("submission.json",date_time=(2026,9,23,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16
    with zipfile.ZipFile(zp,"w",compresslevel=9) as z:z.writestr(info,payload)
    print("READY",zp,flush=True)
    return zp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--batch-queries",type=int,default=4)
    args=ap.parse_args()

    if not BRIDGE.exists(): raise FileNotFoundError(BRIDGE)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")

    n.OUT=PROBE
    n.TRAIN_FOLDS=(0,1,2)
    qids,short,X45,y,gold_count,fold_id,evidence=n.load_payload()
    train=np.where(np.isin(fold_id,(0,1,2)))[0].astype(np.int32)
    h3=np.where(fold_id==3)[0].astype(np.int32)
    h4=np.where(fold_id==4)[0].astype(np.int32)

    b=np.load(BRIDGE)
    if not np.array_equal(short,b["public_short"]):
        raise RuntimeError("public shortlist bridge drift")
    W=b["public_w75"].astype(np.float32)

    meta=json.loads((PROBE/"TRAIN_META.json").read_text(encoding="utf-8"))
    alpha=float(meta["alpha"])
    maxlen=int(meta["config"]["maxlen"])
    print("[checkpoint] alpha",alpha,"maxlen",maxlen,flush=True)

    tok,model=n.load_saved_model()

    raw3=np.load(PROBE/"fold3_raw_reloaded.f32.npy")
    if raw3.shape!=(len(h3),20): raise RuntimeError("fold3 raw shape drift")
    p3=n.held_prior(X45,y,train,h3)
    q3=qwen_score(p3,raw3,alpha)

    print("[1/5] infer fresh checkpoint on untouched fold4",flush=True)
    raw4=batch_infer(model,tok,evidence,h4,maxlen,args.batch_queries)
    np.save(PROBE/"fold4_raw_reloaded.f32.npy",raw4)
    p4=n.held_prior(X45,y,train,h4)
    q4=qwen_score(p4,raw4,alpha)

    qm3,_=n.evaluate(q3,y,gold_count,h3)
    qm4,_=n.evaluate(q4,y,gold_count,h4)
    print(f"[QWEN] fold3={qm3['recall_at_5']:.9f} fold4={qm4['recall_at_5']:.9f}",flush=True)
    print(f"[W75 ] fold3={W75_F3:.9f} fold4={W75_F4:.9f}",flush=True)

    print("[2/5] W75+Qwen conservative fusion audit",flush=True)
    rows=[]
    # w=0 parity, then modest Qwen influence. anchor=4 is the main candidate.
    for anchor in (0,4,3):
        for w in np.round(np.arange(0,0.501,.025),3):
            r3,_=eval_config(W[h3],q3,float(w),anchor,y,gold_count,h3)
            r4,_=eval_config(W[h4],q4,float(w),anchor,y,gold_count,h4)
            mean=(len(h3)*r3+len(h4)*r4)/(len(h3)+len(h4))
            worst_delta=min(r3-W75_F3,r4-W75_F4)
            rows.append({"anchor":anchor,"w_qwen":float(w),"fold3":r3,"fold4":r4,
                         "mean":mean,"worst_delta":worst_delta})

    best_mean=max(rows,key=lambda x:(x["mean"],x["worst_delta"],-x["w_qwen"]))
    best_safe=max(rows,key=lambda x:(x["worst_delta"],x["mean"],-x["w_qwen"]))
    # Cross-held: choose on one held fold, test the other.
    best3=max(rows,key=lambda x:(x["fold3"],x["fold4"],-x["w_qwen"]))
    best4=max(rows,key=lambda x:(x["fold4"],x["fold3"],-x["w_qwen"]))
    cross_mean=(best3["fold4"]+best4["fold3"])/2

    print("[BEST mean]",best_mean,flush=True)
    print("[BEST safe]",best_safe,flush=True)
    print("[select fold3 -> test fold4]",best3,"test_R",best3["fold4"],flush=True)
    print("[select fold4 -> test fold3]",best4,"test_R",best4["fold3"],flush=True)
    print("[cross-held mean]",cross_mean,flush=True)

    w75mean=(len(h3)*W75_F3+len(h4)*W75_F4)/(len(h3)+len(h4))
    # Final gate: positive aggregate held gain; neither held fold may fall more than one
    # single-query step (~0.00072); cross-held diagnostic may not collapse.
    cand=best_mean
    pass_gate=(
        cand["mean"] >= w75mean + .00025 and
        cand["fold3"] >= W75_F3 - .00072 and
        cand["fold4"] >= W75_F4 - .00072 and
        cross_mean >= (W75_F3+W75_F4)/2 - .00020
    )

    audit={"schema":"stage07ad.qwen_w75_lastshot.v1","teacher_used":False,"distillation_used":False,
           "alpha":alpha,"qwen":{"fold3":qm3,"fold4":qm4},
           "w75":{"fold3":W75_F3,"fold4":W75_F4,"weighted_mean":w75mean},
           "best_mean":best_mean,"best_safe":best_safe,
           "cross":{"select3":best3,"select4":best4,"cross_mean":cross_mean},
           "pass_gate":bool(pass_gate)}
    (PROBE/"STAGE07AD_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")

    if not pass_gate:
        print("[3/5] KILL FINAL FUSION — keep W75 champion. Do NOT spend last slot.",flush=True)
        return

    print("[3/5] gate PASS -> load private payload",flush=True)
    pqids,docs,pshort,pprior,pev=load_private()
    if not np.array_equal(pshort,b["private_short"]):
        raise RuntimeError("private shortlist bridge drift")
    PW=b["private_w75"].astype(np.float32)

    print("[4/5] infer private Qwen using current fresh train012 checkpoint",flush=True)
    praw=private_raw(model,tok,pev,maxlen,args.batch_queries)
    PQ=qwen_score(pprior,praw,alpha)
    blend=(1-cand["w_qwen"])*zrows(PW)+cand["w_qwen"]*zrows(PQ)
    pos=top5_indices(blend) if cand["anchor"]==0 else anchored_top5(PW,blend,int(cand["anchor"]))

    label=f"GOLD_LASTSHOT_W75_QWEN_A{int(cand['anchor'])}_W{int(round(cand['w_qwen']*100)):02d}_K5"
    zp=package(label,pos,pqids,docs,pshort)
    audit["private_zip"]=str(zp)
    (PROBE/"STAGE07AD_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("[5/5] CANDIDATE_FOR_FINAL_SUBMISSION",zp,flush=True)

if __name__=="__main__":
    main()

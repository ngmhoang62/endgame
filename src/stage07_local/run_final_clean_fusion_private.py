#!/usr/bin/env python
"""
Stage07Y — final clean local fusion audit + private materialization.

Signals:
  Stage07R gold document propensity (gold-only, no distillation)
  Stage07O AIT raw-centroid gold head (gold-only, no distillation)
  Stage07P LAL raw-centroid gold head (optional analysis)

Outputs:
  OOF weight audit
  private propensity standalone
  private 50/50 propensity+AIT
  private robust-weight propensity+AIT

No new foundation model.
"""
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_gold_doc_propensity_oof as prop
import src.stage07_local.materialize_gold_centroid_private as mp

OUT=ROOT/"reports/stage07y_final_clean_fusion"
CACHE=ROOT/"cache/stage07y_final_clean_fusion"
SUB=ROOT/"submissions/endgame_20260923"
P_PROP=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
P_AIT=ait.CACHE/"gold_head_oof_scores30.f32.npy"
P_LAL=ROOT/"cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy"

def rank(short,s):
    return ait.rank(short,s)

def perq(rank_,qids,golds,docs):
    return np.asarray([
        len(set(golds[q]) & {docs[int(d)] for d in rank_[i,:5]})/len(golds[q])
        for i,q in enumerate(qids)
    ],np.float64)

def write_zip(label,score,qids,docs,short):
    rr=rank(short,score)
    ans={}
    for i,q in enumerate(qids):
        row=[docs[int(x)] for x in rr[i,:5]]
        if len(row)!=5 or len(set(row))!=5:
            raise RuntimeError(f"bad row {q}")
        ans[q]={"answer":row}
    SUB.mkdir(parents=True,exist_ok=True)
    jp=SUB/f"{label}.json"; zp=SUB/f"{label}.zip"
    payload=(json.dumps(ans,ensure_ascii=False,indent=2)+"\n").encode()
    jp.write_bytes(payload)
    info=zipfile.ZipInfo("submission.json",date_time=(2026,9,23,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o644<<16
    with zipfile.ZipFile(zp,"w") as z:z.writestr(info,payload)
    print("READY",zp,flush=True)
    return zp

def private_propensity():
    qids,_,golds,folds,_,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(b1.CACHE/"oof_top30_ce_scores.f32.npy")
    flat,_,_,_=b1.build_source_ce_features(short,ce,sources)
    X45=flat.reshape(len(qids),30,45).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.uint8)
    q2i={q:i for i,q in enumerate(qids)}
    d2i={d:i for i,d in enumerate(docs)}
    indices={f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}
    counts={f:prop.counts_for_fold(indices[f],short,qids,golds,d2i,len(docs)) for f in folds}

    X47=np.empty((len(qids),30,47),np.float32)
    X47[:,:,:45]=X45
    for f in folds:
        others=[x for x in folds if x!=f]
        g=sum((counts[x][0] for x in others),np.zeros(len(docs),np.float32))
        e=sum((counts[x][1] for x in others),np.zeros(len(docs),np.float32))
        X47[indices[f],:,45:]=prop.propensity(short[indices[f]],g,e,4)
    model=prop.make_lr().fit(X47.reshape(-1,47),y.reshape(-1))

    pqids,pdocs,pshort,PX45,PX57,pqe,pcent,names45=mp.private_features()
    if pdocs!=docs:raise RuntimeError("doc order drift")
    g=sum((counts[f][0] for f in folds),np.zeros(len(docs),np.float32))
    e=sum((counts[f][1] for f in folds),np.zeros(len(docs),np.float32))
    extra=prop.propensity(pshort,g,e,5)
    PX47=np.concatenate((PX45,extra),axis=2)
    ps=model.decision_function(PX47.reshape(-1,47)).reshape(len(pqids),30).astype(np.float32)
    return pqids,pdocs,pshort,ps,PX45,PX57,pqe,pcent

def private_ait_score(pqids,pdocs,pshort,PX45,PX57,pqe,pcent):
    path=ait.CACHE/"private_score30.f32.npy"
    if path.exists():
        s=np.load(path)
        if s.shape==(len(pqids),30) and np.isfinite(s).all():
            print("[AIT private] cache hit",path,flush=True)
            return np.asarray(s,np.float32)
    import torch
    ck=ait.CACHE/"gold_centroid_fulltrain.pt"
    if not ck.exists():
        raise FileNotFoundError(
            f"{ck} missing. Run src/stage07_local/materialize_gold_centroid_private.py once "
            "or restore its checkpoint."
        )
    cp=torch.load(ck,map_location="cpu",weights_only=True)
    model=ait.make_model(PX57.shape[-1]).to("cuda")
    model.load_state_dict(cp["state_dict"])
    args=type("Args",(),{"device":"cuda","batch_queries":32})()
    residual=mp.score_private(pshort,PX57,pqe,pcent,model,args)
    prior=joblib.load(b1.CACHE/"fulltrain_ce_fusion.joblib")["model"].decision_function(
        PX45.reshape(-1,45)).reshape(len(pqids),30)
    s=ait.zrows(prior); s[:,:ait.DEPTH]+=float(model.alpha().detach().cpu())*residual
    path.parent.mkdir(parents=True,exist_ok=True); np.save(path,s.astype(np.float32))
    return s.astype(np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    for p in (P_PROP,P_AIT):
        if not p.exists():raise FileNotFoundError(p)
    qids,_,golds,folds,stress,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    prop_o=np.load(P_PROP); ait_o=np.load(P_AIT)
    lal_o=np.load(P_LAL) if P_LAL.exists() else None
    q2i={q:i for i,q in enumerate(qids)}
    fidx={f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

    grid=np.round(np.arange(0,1.0001,.05),2)
    rows=[]
    for w in grid:
        s=w*ait.zrows(prop_o)+(1-w)*ait.zrows(ait_o)
        rr=rank(short,s)
        met=b1.eval_rank(rr,qids,golds,docs,folds,stress)
        rows.append({
            "w_prop":float(w),
            "R":met["overall"]["recall_at_5"],
            "multi":met["overall"]["multi_gold_recall_at_5"],
            "folds":{f:met["per_fold"][f]["recall_at_5"] for f in folds},
        })
    # Robust final weight: maximize mean - 0.35*std of fold recalls.
    for x in rows:
        vals=np.asarray(list(x["folds"].values()))
        x["robust_obj"]=float(vals.mean()-.35*vals.std())
    best=max(rows,key=lambda x:(x["robust_obj"],x["R"],-abs(x["w_prop"]-.5)))
    print("[OOF] best robust",best,flush=True)

    # Leave-one-fold-out selection diagnostic.
    nested=[]
    for held in folds:
        candidates=[]
        for x in rows:
            trainvals=[v for f,v in x["folds"].items() if f!=held]
            obj=float(np.mean(trainvals)-.35*np.std(trainvals))
            candidates.append((obj,x["R"],-abs(x["w_prop"]-.5),x["w_prop"]))
        w=max(candidates)[-1]
        x=next(z for z in rows if z["w_prop"]==w)
        nested.append({"held":held,"selected_w_prop":w,"held_R":x["folds"][held]})
    nested_R=float(np.average(
        [x["held_R"] for x in nested],
        weights=[len(folds[x["held"]]) for x in nested]
    ))
    report={"schema":"stage07y.final_clean_fusion.v1","teacher_used":False,"distillation_used":False,
            "grid":rows,"robust_choice":best,"nested":nested,"nested_weighted_R":nested_R}
    (OUT/"FUSION_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    print("[OOF] nested weighted R",nested_R,flush=True)

    print("[private] build propensity",flush=True)
    pqids,pdocs,pshort,pps,PX45,PX57,pqe,pcent=private_propensity()
    print("[private] build/load AIT",flush=True)
    pas=private_ait_score(pqids,pdocs,pshort,PX45,PX57,pqe,pcent)

    write_zip("GOLD_DOC_PROPENSITY_K5",pps,pqids,pdocs,pshort)
    s50=.5*ait.zrows(pps)+.5*ait.zrows(pas)
    write_zip("GOLD_PROP_AIT_50_50_K5",s50,pqids,pdocs,pshort)
    w=float(best["w_prop"])
    sr=w*ait.zrows(pps)+(1-w)*ait.zrows(pas)
    write_zip(f"GOLD_PROP_AIT_W{int(round(100*w)):02d}_K5",sr,pqids,pdocs,pshort)
    print("DECISION: use robust candidate first; keep standalone/50-50 as backups.",flush=True)

if __name__=="__main__":
    main()

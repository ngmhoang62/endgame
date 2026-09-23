#!/usr/bin/env python
"""
Stage07AC — strict nested safe-switch router around W75.

Purpose
-------
Stage07AB shows fixed residuals do not beat W75, but W75 + top residual arms
has a meaningful oracle. Test whether the *decision to switch* is learnable.

Everything is competition-clean:
  Base W75 = Stage07R propensity + Stage07O AIT centroid.
  Candidate residual signals:
    L = Stage07P LAL gold centroid.
    S = Stage07S centroid+propensity gold head.
  No teacher, no distillation, no new foundation model.

Fixed arm library (NOT selected from held labels):
  W75 + beta * orth(L, W75)
  W75 + beta * orth(S, W75)
  beta in {-0.15,-0.10,-0.05,+0.05,+0.10,+0.15}

Router
------
A small ExtraTrees regressor predicts utility delta of each arm vs W75 from
label-free score/rank geometry + query structural cues.

For every OUTER fold:
  - router is trained on the other 4 folds only;
  - switching threshold is selected by INNER OOF across those 4 folds;
  - the held outer fold is touched once.

So the reported outer OOF is strict.

Run:
  python src/stage07_local/run_stage07ac_safe_switch_router.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT=ROOT/"reports/stage07ac_safe_switch_router"

P_PATH=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
A_PATH=ROOT/"cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"
L_PATH=ROOT/"cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy"
S_PATH=ROOT/"cache/stage07s_gold_centroid_propensity/centroid_propensity_oof_scores30.f32.npy"

BETAS=(-.15,-.10,-.05,.05,.10,.15)
W75_R=0.9505364039479332

def ranks(s):
    order=np.argsort(-s,axis=1,kind="stable")
    r=np.empty_like(order)
    rr=np.arange(len(s))[:,None]
    r[rr,order]=np.arange(s.shape[1])[None,:]
    return r

def orth(r,b):
    den=(b*b).sum(1,keepdims=True)
    coef=(r*b).sum(1,keepdims=True)/np.where(den<1e-8,1,den)
    x=r-coef*b
    sd=x.std(1,keepdims=True)
    return x/np.where(sd<1e-6,1,sd)

def rank_docs(short,s):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-s[i]))]
    return out

def recall_q(rank,qids,golds,docs):
    x=np.empty(len(qids),np.float32)
    for i,q in enumerate(qids):
        g=set(golds[q])
        x[i]=len(g & {docs[int(d)] for d in rank[i,:5]})/len(g)
    return x

def topk_mask(r,k=5):
    return r<k

def gap_features(s):
    so=np.sort(s,axis=1)[:,::-1]
    return np.stack([
        so[:,0]-so[:,4],
        so[:,3]-so[:,4],
        so[:,4]-so[:,5],
        so[:,4]-so[:,6],
        so[:,4]-so[:,9],
        so[:,:5].std(1),
        so[:,5:10].std(1),
    ],axis=1).astype(np.float32)

def base_query_features(B,P,A,L,S,questions,qids):
    rB,rP,rA,rL,rS=map(ranks,(B,P,A,L,S))
    sets=[topk_mask(r) for r in (rB,rP,rA,rL,rS)]
    G=np.concatenate([gap_features(x) for x in (B,P,A,L,S)],axis=1)
    overlaps=[]
    for i in range(1,5):
        inter=(sets[0]&sets[i]).sum(1)
        union=(sets[0]|sets[i]).sum(1)
        overlaps.append((inter/np.maximum(union,1))[:,None])
    pa=((sets[1]&sets[2]).sum(1)/np.maximum((sets[1]|sets[2]).sum(1),1))[:,None]
    qh=np.stack([qg.q_hand(questions[q]) for q in qids]).astype(np.float32)
    return np.concatenate([G,*overlaps,pa,qh],axis=1).astype(np.float32)

def arm_library(B,L,S):
    zL,zS=orth(L,B),orth(S,B)
    arms=[]
    for sig,z in (("L",zL),("S",zS)):
        for beta in BETAS:
            arms.append({"signal":sig,"beta":float(beta),"score":B+float(beta)*z})
    return arms

def candidate_features(B,P,A,L,S,arm,baseX):
    C=arm["score"]
    rB,rC,rP,rA,rL,rS=map(ranks,(B,C,P,A,L,S))
    mB,mC=topk_mask(rB),topk_mask(rC)
    entering=mC&~mB
    leaving=mB&~mC
    churn=(entering.sum(1)+leaving.sum(1)).astype(np.float32)[:,None]

    def masked_stats(val,mask,default=0.):
        out=[]
        for fn in ("mean","max","min"):
            z=np.empty(len(val),np.float32)
            for i in range(len(val)):
                vv=val[i,mask[i]]
                if len(vv)==0: z[i]=default
                elif fn=="mean": z[i]=vv.mean()
                elif fn=="max": z[i]=vv.max()
                else: z[i]=vv.min()
            out.append(z[:,None])
        return out

    feats=[baseX,churn]
    # Candidate-arm boundary geometry.
    feats.append(gap_features(C))
    # Entering/leaving candidate score/rank evidence from all clean worlds.
    for score,rk in ((B,rB),(P,rP),(A,rA),(L,rL),(S,rS),(C,rC)):
        feats += masked_stats(score,entering)
        feats += masked_stats(score,leaving)
        feats += masked_stats((-rk/30.0).astype(np.float32),entering)
        feats += masked_stats((-rk/30.0).astype(np.float32),leaving)
    # Arm constants.
    sigL=np.full((len(B),1),1.0 if arm["signal"]=="L" else 0.0,np.float32)
    beta=np.full((len(B),1),arm["beta"],np.float32)
    feats += [sigL,beta]
    return np.concatenate(feats,axis=1).astype(np.float32)

def make_model(seed):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=6,
        max_features=.75,
        random_state=seed,
        n_jobs=-1,
    )

def fit_predict(train_idx,test_idx,Xarms,Ydelta,seed):
    # Flatten query-arm rows; keep all arms so model can learn arm-specific utility.
    na=len(Xarms)
    Xtr=np.concatenate([Xarms[a][train_idx] for a in range(na)],axis=0)
    ytr=np.concatenate([Ydelta[train_idx,a] for a in range(na)],axis=0)
    sw=1.0+7.0*(np.abs(ytr)>1e-8)
    m=make_model(seed)
    m.fit(Xtr,ytr,sample_weight=sw)
    pred=np.stack([m.predict(Xarms[a][test_idx]) for a in range(na)],axis=1).astype(np.float32)
    return pred

def choose_threshold(pred,delta):
    # pred/delta [N,A]; select predicted-best arm or W75.
    mx=pred.max(1); ai=pred.argmax(1)
    vals=np.unique(np.quantile(mx,[.0,.25,.40,.50,.60,.70,.80,.85,.90,.925,.95,.975,.99]))
    vals=np.concatenate(([-1e-6],vals,[float(mx.max()+1e-6)]))
    best=None
    for t in vals:
        use=mx>t
        d=np.where(use,delta[np.arange(len(delta)),ai],0.0)
        gain=float(d.mean()); wins=int((d>0).sum()); losses=int((d<0).sum())
        # Conservative objective: gain, then lower losses, then lower switch rate.
        obj=gain-0.00002*max(losses-wins,0)
        cand=(obj,gain,wins-losses,-int(use.sum()),float(t),wins,losses,int(use.sum()))
        if best is None or cand[:4]>best[:4]: best=cand
    return {
        "threshold":best[4],"inner_gain":best[1],"wins":best[5],
        "losses":best[6],"switches":best[7]
    }

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in (P_PATH,A_PATH,L_PATH,S_PATH):
        if not p.exists(): raise FileNotFoundError(p)

    qids,questions,golds,folds,stress,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    q2i={q:i for i,q in enumerate(qids)}
    fidx={f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

    P,A,L,S=[ait.zrows(np.load(p)) for p in (P_PATH,A_PATH,L_PATH,S_PATH)]
    B=.75*P+.25*A
    base_rank=rank_docs(short,B)
    base_rq=recall_q(base_rank,qids,golds,docs)
    if abs(float(base_rq.mean())-W75_R)>1e-6:
        raise RuntimeError(f"W75 parity {base_rq.mean()}")

    arms=arm_library(B,L,S)
    baseX=base_query_features(B,P,A,L,S,questions,qids)
    print("[1/5] build per-arm label-free geometry",flush=True)
    Xarms=[candidate_features(B,P,A,L,S,a,baseX) for a in arms]
    arm_rq=np.stack([recall_q(rank_docs(short,a["score"]),qids,golds,docs) for a in arms],axis=1)
    delta=arm_rq-base_rq[:,None]
    oracle=float(np.maximum(delta.max(1),0).mean()+base_rq.mean())
    print(f"[arm oracle L/S library] {oracle:.9f}",flush=True)

    out_rq=base_rq.copy()
    meta={}
    print("[2/5] strict nested outer routing",flush=True)
    fold_names=list(folds)
    for oi,held_name in enumerate(fold_names):
        held=fidx[held_name]
        train_folds=[f for f in fold_names if f!=held_name]
        train=np.concatenate([fidx[f] for f in train_folds])

        # Inner OOF predictions on the four outer-train folds.
        ip=np.zeros((len(train),len(arms)),np.float32)
        pos={int(q):i for i,q in enumerate(train)}
        for ii,inner_name in enumerate(train_folds):
            iv=fidx[inner_name]
            itr=np.concatenate([fidx[f] for f in train_folds if f!=inner_name])
            pp=fit_predict(itr,iv,Xarms,delta,276+oi*10+ii)
            loc=np.asarray([pos[int(q)] for q in iv],np.int32)
            ip[loc]=pp
        tinfo=choose_threshold(ip,delta[train])

        # Final outer model.
        hp=fit_predict(train,held,Xarms,delta,911+oi)
        mx=hp.max(1); ai=hp.argmax(1); use=mx>tinfo["threshold"]
        chosen_delta=np.where(use,delta[held,ai],0.0)
        out_rq[held]=base_rq[held]+chosen_delta

        meta[held_name]={
            "threshold":tinfo,
            "held_switches":int(use.sum()),
            "held_wins":int((chosen_delta>0).sum()),
            "held_losses":int((chosen_delta<0).sum()),
            "held_R":float(out_rq[held].mean()),
            "base_R":float(base_rq[held].mean()),
            "delta_R":float(chosen_delta.mean()),
        }
        print("[fold]",held_name,meta[held_name],flush=True)

    R=float(out_rq.mean())
    wins=int((out_rq>base_rq).sum()); losses=int((out_rq<base_rq).sum())
    print(f"[3/5] ROUTED R={R:.9f} delta={R-W75_R:+.9f} W/L={wins}/{losses}",flush=True)

    report={
        "schema":"stage07ac.safe_switch_router.v1",
        "teacher_used":False,"distillation_used":False,
        "base_R":W75_R,"arm_oracle":oracle,
        "routed_R":R,"delta":R-W75_R,"wins":wins,"losses":losses,
        "arms":[{"signal":a["signal"],"beta":a["beta"]} for a in arms],
        "folds":meta,
    }
    (OUT/"OOF_REPORT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("[4/5] wrote",OUT/"OOF_REPORT.json",flush=True)

    actionable=(R>=W75_R+.00025 and wins>=losses and sum(v["delta_R"]>0 for v in meta.values())>=3)
    print("[5/5] ACTIONABLE",actionable,flush=True)
    print("DECISION:","BUILD_PRIVATE_ROUTER" if actionable else "KILL_ROUTER_WAIT_FOR_QWEN",flush=True)

if __name__=="__main__":
    main()

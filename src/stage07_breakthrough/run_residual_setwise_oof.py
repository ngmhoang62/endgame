#!/usr/bin/env python
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg

CACHE=ROOT/"cache/stage07a_residual_setwise"
OUT=ROOT/"reports/stage07a_residual_setwise"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"

SHORT=30
SEEDS=(276,911)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rank_from_scores(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-score[i]))]
    return out

def eval_metrics(rank,qids,golds,docs,folds):
    pred={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    overall=official_metrics(pred,golds,qids)
    per={f:official_metrics(pred,golds,ids) for f,ids in folds.items()}
    return overall,per

def labels(short,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    y=np.zeros(short.shape,np.float32)
    for i,q in enumerate(qids):
        gs={d2i[d] for d in golds[q]}
        y[i]=np.asarray([float(int(x) in gs) for x in short[i]],np.float32)
    return y

def load_world():
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH)
    done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("AITeam CE cache incomplete")
    Xflat,cq,cand,names45=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    if X45.shape[-1]!=45: raise RuntimeError(f"45D drift: {X45.shape}")
    names=qg.load_names(docs)
    T=qg.make_title_features(qids,questions,short,names).astype(np.float32)
    X=np.concatenate([X45,T],axis=2)
    qe=qg.query_embed(qids)
    qh=np.stack([qg.q_hand(questions[q]) for q in qids]).astype(np.float32)
    Q=np.concatenate([qe,qh],axis=1).astype(np.float32)
    y=labels(short,qids,golds,docs)
    return qids,questions,golds,folds,docs,short,X45,X,Q,y

def fit_lr_score(X45,y,train_idx,score_idx):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=.15,class_weight="balanced",solver="lbfgs",
                                  max_iter=300,tol=1e-5))
    ])
    m.fit(X45[train_idx].reshape(-1,X45.shape[-1]),y[train_idx].reshape(-1))
    s=m.decision_function(X45[score_idx].reshape(-1,X45.shape[-1]))
    return np.asarray(s,np.float32).reshape(len(score_idx),SHORT)

def inner_crossfit_prior(X45,y,folds,qids,outer_name):
    q2i={q:i for i,q in enumerate(qids)}
    train_folds=[f for f in folds if f!=outer_name]
    train_idx=np.asarray([q2i[q] for f in train_folds for q in folds[f]],np.int32)
    held_idx=np.asarray([q2i[q] for q in folds[outer_name]],np.int32)

    prior_all=np.full((len(qids),SHORT),np.nan,np.float32)
    for inner in train_folds:
        val=np.asarray([q2i[q] for q in folds[inner]],np.int32)
        fit=np.asarray([q2i[q] for f in train_folds if f!=inner for q in folds[f]],np.int32)
        prior_all[val]=fit_lr_score(X45,y,fit,val)
    if not np.isfinite(prior_all[train_idx]).all():
        raise RuntimeError("inner crossfit prior incomplete")
    prior_held=fit_lr_score(X45,y,train_idx,held_idx)
    return train_idx,held_idx,prior_all[train_idx],prior_held

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(axis=1,keepdims=True); sd=x.std(axis=1,keepdims=True)
    sd=np.where(sd<1e-6,1.0,sd)
    return (x-mu)/sd

def train_fold(X,Q,y,train_idx,held_idx,base_tr,base_he,args,seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    set_seed(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"

    flat=X[train_idx].reshape(-1,X.shape[-1]).astype(np.float64)
    mu=flat.mean(0); sd=flat.std(0); sd=np.where(sd<1e-6,1.0,sd)
    xtr=((X[train_idx]-mu)/sd).astype(np.float32)
    xhe=((X[held_idx]-mu)/sd).astype(np.float32)

    qtr=Q[train_idx].astype(np.float32); qhe=Q[held_idx].astype(np.float32)
    qtr=qtr/np.maximum(np.linalg.norm(qtr,axis=1,keepdims=True),1e-6)
    qhe=qhe/np.maximum(np.linalg.norm(qhe,axis=1,keepdims=True),1e-6)

    btr=zrows(base_tr); bhe=zrows(base_he)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            h=args.hidden
            self.cand=nn.Sequential(nn.Linear(X.shape[-1]+2,h),nn.GELU(),nn.LayerNorm(h))
            self.query=nn.Sequential(nn.Linear(Q.shape[-1],h),nn.GELU(),nn.LayerNorm(h))
            self.pos=nn.Embedding(SHORT,h)
            layer=nn.TransformerEncoderLayer(
                d_model=h,nhead=args.heads,dim_feedforward=h*3,dropout=.08,
                activation="gelu",batch_first=True,norm_first=True
            )
            self.enc=nn.TransformerEncoder(layer,num_layers=args.layers,norm=nn.LayerNorm(h))
            self.head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1))
            self.raw_alpha=nn.Parameter(torch.tensor(-1.5))
        def forward(self,c,q,b):
            B,K,_=c.shape
            rank=torch.linspace(0,1,K,device=c.device,dtype=c.dtype)[None,:,None].expand(B,-1,-1)
            z=torch.cat([c,b[:,:,None],rank],dim=-1)
            h=self.cand(z)+self.query(q)[:,None,:]
            p=torch.arange(K,device=c.device)
            h=h+self.pos(p)[None,:,:]
            h=self.enc(h)
            delta=self.head(h).squeeze(-1)
            alpha=F.softplus(self.raw_alpha)
            total=b+alpha*delta
            return total,delta,alpha

    model=Model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)
    rng=np.random.default_rng(seed)
    valid_local=np.where(y[train_idx].sum(axis=1)>0)[0]
    losses=[]

    for ep in range(args.epochs):
        order=valid_local.copy(); rng.shuffle(order); model.train(); ep_losses=[]
        for st in range(0,len(order),args.batch_q):
            loc=order[st:st+args.batch_q]
            c=torch.from_numpy(xtr[loc]).to(device)
            q=torch.from_numpy(qtr[loc]).to(device)
            yy=torch.from_numpy(y[train_idx[loc]]).to(device)
            bb=torch.from_numpy(btr[loc]).to(device)
            score,delta,alpha=model(c,q,bb)

            cnt=yy.sum(1,keepdim=True).clamp_min(1)
            target=yy/cnt
            listwise=-(target*torch.log_softmax(score/args.temperature,dim=1)).sum(1).mean()

            pair_losses=[]
            for bi in range(len(loc)):
                pos=torch.where(yy[bi]>0)[0]
                neg=torch.where(yy[bi]==0)[0]
                if len(pos)==0 or len(neg)==0: continue
                hard=neg[torch.topk(bb[bi,neg],k=min(8,len(neg))).indices]
                diff=score[bi,pos][:,None]-score[bi,hard][None,:]
                pair_losses.append(F.softplus(args.margin-diff).mean())
            pair=torch.stack(pair_losses).mean() if pair_losses else torch.zeros((),device=device)
            reg=(delta**2).mean()
            loss=listwise+args.pair_weight*pair+args.delta_l2*reg

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            ep_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(ep_losses)))
        if ep==0 or ep+1==args.epochs or (ep+1)%4==0:
            av=float(torch.nn.functional.softplus(model.raw_alpha).detach().cpu())
            print(f"    seed={seed} ep={ep+1}/{args.epochs} loss={losses[-1]:.5f} alpha={av:.4f}",flush=True)

    model.eval(); pred=[]
    with torch.inference_mode():
        for st in range(0,len(held_idx),args.batch_q):
            c=torch.from_numpy(xhe[st:st+args.batch_q]).to(device)
            q=torch.from_numpy(qhe[st:st+args.batch_q]).to(device)
            bb=torch.from_numpy(bhe[st:st+args.batch_q]).to(device)
            s,_,_=model(c,q,bb); pred.append(s.float().cpu().numpy())
    score=np.concatenate(pred,axis=0).astype(np.float32)
    alpha=float(torch.nn.functional.softplus(model.raw_alpha).detach().cpu())
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return score,{"seed":seed,"loss_tail":losses[-1],"alpha":alpha}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--batch-q",type=int,default=128)
    ap.add_argument("--hidden",type=int,default=192)
    ap.add_argument("--heads",type=int,default=8)
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--lr",type=float,default=1.5e-3)
    ap.add_argument("--temperature",type=float,default=.8)
    ap.add_argument("--pair-weight",type=float,default=.25)
    ap.add_argument("--margin",type=float,default=.5)
    ap.add_argument("--delta-l2",type=float,default=.002)
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()

    print("[1/6] Load frozen OOF world/features",flush=True)
    qids,questions,golds,folds,docs,short,X45,X,Q,y=load_world()
    oof=np.full((len(qids),SHORT),np.nan,np.float32)
    base_oof=np.full_like(oof,np.nan)
    fold_meta={}

    print("[2/6] Nested-crossfit LR priors + residual setwise OOF",flush=True)
    for fn in folds:
        print(f"  [{fn}] leakage-safe LR priors",flush=True)
        tr,he,btr,bhe=inner_crossfit_prior(X45,y,folds,qids,fn)
        base_oof[he]=bhe
        ens=np.zeros((len(he),SHORT),np.float64); metas=[]
        for seed in SEEDS:
            s,m=train_fold(X,Q,y,tr,he,btr,bhe,args,seed)
            ens+=s/len(SEEDS); metas.append(m)
        oof[he]=ens.astype(np.float32)
        fold_meta[fn]={"train":len(tr),"held":len(he),"seeds":metas}

    if not np.isfinite(oof).all() or not np.isfinite(base_oof).all():
        raise RuntimeError("OOF scores incomplete")

    print("[3/6] Baseline parity",flush=True)
    base_rank=rank_from_scores(short,base_oof)
    bm,bpf=eval_metrics(base_rank,qids,golds,docs,folds)
    if abs(bm["recall_at_5"]-0.9425976255185238)>3e-6:
        raise RuntimeError(f"CE-LR parity failed: {bm['recall_at_5']}")

    print("[4/6] Setwise evaluation",flush=True)
    srank=rank_from_scores(short,oof)
    sm,spf=eval_metrics(srank,qids,golds,docs,folds)
    delta=float(sm["recall_at_5"]-bm["recall_at_5"])

    def pq(rank):
        z=[]
        for i,q in enumerate(qids):
            pred={docs[int(x)] for x in rank[i,:5]}; g=set(golds[q])
            z.append(len(pred&g)/len(g))
        return np.asarray(z,np.float64)
    bq=pq(base_rank); sq=pq(srank)
    wins=int(np.sum(sq>bq)); losses=int(np.sum(sq<bq))

    print("[5/6] Save reusable scores",flush=True)
    np.save(CACHE/"setwise_oof_scores.f32.npy",oof)
    np.save(CACHE/"ce_lr_oof_prior_scores.f32.npy",base_oof)

    if sm["recall_at_5"]>=.960: decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.010: decision="MAJOR_GAIN_EXPAND_IMMEDIATELY"
    elif delta>=.005: decision="STRONG_GAIN_KEEP"
    elif delta>=.002: decision="KEEP_AS_COMPLEMENT"
    else: decision="KILL_SETWISE"

    print("[6/6] Report",flush=True)
    rep={
        "schema":"dsc2026.endgame.stage07a.residual_setwise_oof.v1",
        "status":"COMPLETE","claim_boundary":"strict 5-fold OOF",
        "architecture":{"candidate_dim":int(X.shape[-1]),"query_dim":int(Q.shape[-1]),
                        "hidden":args.hidden,"heads":args.heads,"layers":args.layers,
                        "seeds":list(SEEDS),
                        "loss":"multi-positive listwise + hard-boundary pairwise + residual L2"},
        "baseline_ce_lr":{"overall":bm,"per_fold":bpf},
        "residual_setwise":{"overall":sm,"per_fold":spf},
        "effect":{"delta_recall":delta,
                  "delta_single":float(sm["single_gold_recall_at_5"]-bm["single_gold_recall_at_5"]),
                  "delta_multi":float(sm["multi_gold_recall_at_5"]-bm["multi_gold_recall_at_5"]),
                  "wins":wins,"losses":losses},
        "fold_training":fold_meta,"decision":decision,
        "wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"OOF_REPORT.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("="*112)
    print(f"BASE CE-LR       R={bm['recall_at_5']:.9f} P={bm['precision_at_5']:.9f} single={bm['single_gold_recall_at_5']:.9f} multi={bm['multi_gold_recall_at_5']:.9f}")
    print(f"SETWISE RESIDUAL R={sm['recall_at_5']:.9f} P={sm['precision_at_5']:.9f} single={sm['single_gold_recall_at_5']:.9f} multi={sm['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: base={bpf[f]['recall_at_5']:.9f} setwise={spf[f]['recall_at_5']:.9f} delta={spf[f]['recall_at_5']-bpf[f]['recall_at_5']:+.9f}")
    print("DECISION:",decision)
    print("="*112)

if __name__=="__main__": main()

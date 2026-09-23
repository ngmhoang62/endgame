#!/usr/bin/env python
"""
Stage07D — frozen-embedding teacher distillation head.

Purpose
-------
Distill the already-computed Qwen3-Reranker-4B training-time teacher targets
into a *tiny* residual reranker that uses only already-approved/frozen ENDGAME
signals at inference:
  - AITeam query embedding (already active retriever)
  - AITeam document parent centroid built from cached region embeddings
  - frozen 45D retrieval/CE features + 12 title/legal features

The 4B teacher is NEVER used at inference and is NEVER evaluated as a ranking arm.
Outer held labels are never used to fit the student.

Default teacher snapshot:
  results/stage07b/teacher_targets_snapshot.npz

Run:
  python src/stage07_local/run_teacher_distilled_embedding_head_oof.py

Expected runtime: minutes to tens of minutes on the local RTX 4050, not hours.
"""
from __future__ import annotations
import argparse, gc, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage02_candidate_generation.screen_vietnamese_retrievers as s2

OUT=ROOT/"reports/stage07d_teacher_distill_head"
CACHE=ROOT/"cache/stage07d_teacher_distill_head"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
REGION_EMB=ROOT/"cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy"
TEACHER_DEFAULT=ROOT/"results/stage07b/teacher_targets_snapshot.npz"
SHORT=30
DEPTH=20
BASE_R=0.9425976255185238

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-6,1.,sd)
    return (x-mu)/sd

def load_world(teacher_path):
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH); done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("CE cache drift/incomplete")
    Xflat,_,_,_=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    names=qg.load_names(docs)
    T=qg.make_title_features(qids,questions,short,names).astype(np.float32)
    Xext=np.concatenate([X45,T],axis=2).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)
    qe=qg.query_embed(qids).astype(np.float32)

    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing teacher snapshot: {teacher_path}\n"
            "Copy teacher_targets_snapshot.npz from Google Drive into results/stage07b/."
        )
    z=np.load(teacher_path)
    teacher=np.asarray(z["scores"],np.float32)
    done_t=np.asarray(z["done"],np.uint8)
    if teacher.shape!=(len(qids),DEPTH) or done_t.shape!=(len(qids),) or int(done_t.sum())!=len(qids):
        raise RuntimeError(f"teacher snapshot drift scores={teacher.shape} done={int(done_t.sum())}/{len(qids)}")
    if not np.isfinite(teacher).all():
        raise RuntimeError("teacher targets contain nonfinite values")
    return qids,questions,golds,folds,stress,docs,short,X45,Xext,y,qe,zrows(teacher)

def build_parent_centroids(docs,device):
    CACHE.mkdir(parents=True,exist_ok=True)
    cp=CACHE/"aiteam_parent_centroids.f32.npy"
    mp=CACHE/"aiteam_parent_centroids.json"
    if cp.is_file() and mp.is_file():
        x=np.load(cp,mmap_mode="r")
        if x.shape==(len(docs),1024) and np.isfinite(x).all():
            print(f"[centroid] cache hit {cp}",flush=True)
            return np.asarray(x,np.float32)

    if not REGION_EMB.is_file():
        raise FileNotFoundError(REGION_EMB)
    geom=s2.load_geometry()
    if list(map(str,geom["doc_ids"]))!=list(map(str,docs)):
        raise RuntimeError("Stage02 geometry document order != evaluation corpus order")
    pidx=np.asarray(geom["parent_index"],np.int64)
    reg=np.load(REGION_EMB,mmap_mode="r")
    if reg.shape[0]!=len(pidx) or reg.shape[1]!=1024:
        raise RuntimeError(f"region embedding drift {reg.shape} pidx={pidx.shape}")

    import torch
    dev=torch.device(device)
    sums=torch.zeros((len(docs),1024),dtype=torch.float32,device=dev)
    cnt=torch.zeros((len(docs),1),dtype=torch.float32,device=dev)
    chunk=4096 if dev.type=="cuda" else 2048
    print(f"[centroid] build from {len(reg):,} cached regions on {dev}",flush=True)
    for st in range(0,len(reg),chunk):
        en=min(st+chunk,len(reg))
        e=torch.from_numpy(np.asarray(reg[st:en],np.float32).copy()).to(dev)
        ii=torch.from_numpy(pidx[st:en].copy()).to(dev)
        sums.index_add_(0,ii,e)
        cnt.index_add_(0,ii,torch.ones((en-st,1),device=dev))
        if st==0 or en==len(reg) or (st//chunk)%10==0:
            print(f"  regions {en:,}/{len(reg):,}",flush=True)
    c=sums/cnt.clamp_min(1)
    c=torch.nn.functional.normalize(c,dim=1)
    out=c.cpu().numpy().astype(np.float32)
    np.save(cp,out)
    mp.write_text(json.dumps({
        "schema":"stage07d.aiteam_parent_centroid.v1",
        "docs":len(docs),"dim":1024,
        "source":"cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy",
        "aggregation":"mean(region embeddings) then L2 normalize"
    },indent=2)+"\n",encoding="utf-8")
    del sums,cnt,c
    if dev.type=="cuda": torch.cuda.empty_cache()
    return out

def rank_from_score(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        o=np.lexsort((short[i],-score[i]))
        out[i]=short[i,o]
    return out

def eval_rank(rank,qids,golds,docs,folds,stress):
    return qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def perq(rank,qids,golds,docs):
    z=[]
    for i,q in enumerate(qids):
        g=set(golds[q]);p={docs[int(x)] for x in rank[i,:5]}
        z.append(len(g&p)/len(g))
    return np.asarray(z,float)

def train_fold(outer,tr,he,btr,bhe,short,Xext,y,qe,cent,teacher,args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_all(args.seed+outer)
    dev=torch.device(args.device)

    class Head(nn.Module):
        def __init__(self,cdim):
            super().__init__()
            self.q=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.LayerNorm(128))
            self.d=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.LayerNorm(128))
            self.c=nn.Sequential(nn.Linear(cdim,96),nn.GELU(),nn.LayerNorm(96))
            self.h=nn.Sequential(
                nn.Linear(128*4+96,256),nn.GELU(),nn.Dropout(.08),
                nn.Linear(256,64),nn.GELU(),
                nn.Linear(64,1)
            )
            self.raw_alpha=nn.Parameter(torch.tensor(-1.5))
        def forward(self,q,d,c):
            qq=self.q(q); dd=self.d(d); cc=self.c(c)
            z=torch.cat([qq,dd,qq*dd,torch.abs(qq-dd),cc],dim=-1)
            return self.h(z).squeeze(-1)

    model=Head(Xext.shape[-1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    # Frozen arrays copied once to GPU; well below 6GB.
    tq=torch.from_numpy(qe).to(dev)
    tc=torch.from_numpy(cent).to(dev)
    tx=torch.from_numpy(Xext[:,:DEPTH]).to(dev)
    ty=torch.from_numpy(y[:,:DEPTH]).to(dev)
    tt=torch.from_numpy(teacher).to(dev)
    btr_t=torch.from_numpy(zrows(btr)[:,:DEPTH]).to(dev)

    # map global qi -> local row in btr
    trpos={int(q):i for i,q in enumerate(tr)}
    valid=np.asarray([int(q) for q in tr if y[int(q),:DEPTH].sum()>0],np.int32)
    rng=np.random.default_rng(args.seed+outer)
    t0=time.perf_counter(); loss_tail=[]

    for ep in range(args.epochs):
        order=valid.copy();rng.shuffle(order);ep_loss=[]
        model.train()
        for st in range(0,len(order),args.batch_queries):
            qis=order[st:st+args.batch_queries]
            B=len(qis)
            docs20=short[qis,:DEPTH].reshape(-1)
            qrep=tq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            demb=tc[torch.as_tensor(docs20,device=dev)]
            cfeat=tx[qis].reshape(B*DEPTH,-1)
            pred=model(qrep,demb,cfeat).reshape(B,DEPTH)

            teach=tt[qis]
            yy=ty[qis]
            base=torch.stack([btr_t[trpos[int(q)]] for q in qis],dim=0)
            alpha=F.softplus(model.raw_alpha)
            total=base+alpha*pred

            kd=F.smooth_l1_loss(pred,teach,beta=.5)
            cnt=yy.sum(1,keepdim=True).clamp_min(1)
            target=yy/cnt
            gold=-(target*torch.log_softmax(total/args.gold_temp,dim=1)).sum(1).mean()

            pair_losses=[]
            for bi in range(B):
                pos=torch.where(yy[bi]>0)[0]
                neg=torch.where(yy[bi]==0)[0]
                if len(pos)==0 or len(neg)==0: continue
                hard=neg[torch.topk(base[bi,neg],k=min(6,len(neg))).indices]
                diff=total[bi,pos][:,None]-total[bi,hard][None,:]
                pair_losses.append(F.softplus(args.margin-diff).mean())
            pair=torch.stack(pair_losses).mean() if pair_losses else torch.zeros((),device=dev)

            loss=args.kd_weight*kd+args.gold_weight*gold+args.pair_weight*pair
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            ep_loss.append(float(loss.detach().cpu()))
        loss_tail=ep_loss[-100:]
        print(f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(ep_loss):.5f} "
              f"alpha={float(F.softplus(model.raw_alpha).detach().cpu()):.4f} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m",flush=True)

    model.eval(); hs=np.empty((len(he),DEPTH),np.float32)
    with torch.inference_mode():
        for st in range(0,len(he),args.eval_batch_queries):
            qis=he[st:st+args.eval_batch_queries];B=len(qis)
            docs20=short[qis,:DEPTH].reshape(-1)
            qrep=tq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            demb=tc[torch.as_tensor(docs20,device=dev)]
            cfeat=tx[qis].reshape(B*DEPTH,-1)
            p=model(qrep,demb,cfeat).reshape(B,DEPTH)
            hs[st:st+B]=p.float().cpu().numpy()
    alpha=float(F.softplus(model.raw_alpha).detach().cpu())
    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.
    meta={"alpha":alpha,"loss_tail":float(np.mean(loss_tail)),
          "train_queries":len(tr),"held_queries":len(he),
          "peak_reserved_gib":float(peak)}
    del model,opt,tq,tc,tx,ty,tt,btr_t
    gc.collect()
    if dev.type=="cuda":torch.cuda.empty_cache()
    return hs,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,default=TEACHER_DEFAULT)
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch-queries",type=int,default=48)
    ap.add_argument("--eval-batch-queries",type=int,default=96)
    ap.add_argument("--lr",type=float,default=8e-4)
    ap.add_argument("--kd-weight",type=float,default=.55)
    ap.add_argument("--gold-weight",type=float,default=.35)
    ap.add_argument("--pair-weight",type=float,default=.10)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--margin",type=float,default=.4)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/6] frozen world + teacher targets",flush=True)
    qids,questions,golds,folds,stress,docs,short,X45,Xext,y,qe,teacher=load_world(args.teacher)
    print(f"  teacher targets restored: {teacher.shape}; teacher is TRAINING-TIME ONLY",flush=True)

    print("[2/6] parent semantic centroids",flush=True)
    cent=build_parent_centroids(docs,args.device)

    print("[3/6] strict nested CE-LR priors + distilled head OOF",flush=True)
    oof=np.zeros((len(qids),SHORT),np.float32)
    base_oof=np.zeros_like(oof)
    fold_meta={}
    for outer,fn in enumerate(folds):
        tr,he,btr,bhe=sw.inner_crossfit_prior(X45,y,folds,qids,fn)
        base_oof[he]=bhe
        if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
        hs,meta=train_fold(outer,tr,he,btr,bhe,short,Xext,y,qe,cent,teacher,args)
        fs=zrows(bhe)
        fs[:,:DEPTH]+=meta["alpha"]*hs
        oof[he]=fs
        fold_meta[fn]=meta

    print("[4/6] parity + official eval",flush=True)
    br=rank_from_score(short,base_oof); rr=rank_from_score(short,oof)
    bm=eval_rank(br,qids,golds,docs,folds,stress)
    rm=eval_rank(rr,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_R)>3e-6:
        raise RuntimeError(f"baseline parity failed {bm['overall']['recall_at_5']}")
    b=bm["overall"];r=rm["overall"];delta=r["recall_at_5"]-b["recall_at_5"]
    bq=perq(br,qids,golds,docs);rq=perq(rr,qids,golds,docs)
    fd={f:rm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}
    wins=int((rq>bq).sum());losses=int((rq<bq).sum())

    print("[5/6] save reusable OOF signal",flush=True)
    np.save(CACHE/"distilled_head_oof_scores30.f32.npy",oof)
    np.save(CACHE/"distilled_head_oof_residual20.f32.npy",
            np.stack([oof[:,i]-base_oof[:,i] for i in range(DEPTH)],axis=1).astype(np.float32))

    pos=sum(v>0 for v in fd.values())
    if r["recall_at_5"]>=.960:decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:decision="KEEP_AS_COMPLEMENT"
    else:decision="KILL_DISTILLED_HEAD"

    rep={
        "schema":"dsc2026.endgame.stage07d.teacher_distilled_embedding_head.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; 4B teacher used only as training target",
        "teacher":{"model":"Qwen/Qwen3-Reranker-4B","training_time_only":True,"evaluated_as_arm":False},
        "student":{"architecture":"tiny MLP over frozen AIT query/doc-centroid embeddings + 57D candidate features",
                   "new_foundation_model":False},
        "baseline_ce_lr":bm,"distilled_head":rm,
        "effect":{"delta_recall":delta,
                  "delta_single":r["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
                  "delta_multi":r["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
                  "wins":wins,"losses":losses,"fold_deltas":fd},
        "fold_meta":fold_meta,"decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("[6/6] result")
    print("="*112)
    print(f"BASE          R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    print(f"DISTILL-HEAD  R={r['recall_at_5']:.9f} single={r['single_gold_recall_at_5']:.9f} multi={r['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: delta={fd[f]:+.9f} alpha={fold_meta[f]['alpha']:.4f}")
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*112)

if __name__=="__main__":
    main()

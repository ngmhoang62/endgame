#!/usr/bin/env python
"""
Stage07E — Evidence-aware teacher distillation head.

Key hypothesis:
Stage07D distilled the 4B teacher into a tiny head using whole-document parent
centroids. But the 4B teacher was actually shown two query-selected evidence
witnesses per candidate. Parent averaging can wash out the decisive clause.

This experiment distills the SAME already-computed teacher targets into a tiny
head that sees the exact frozen AIT + LAL witness embeddings used to build the
teacher evidence bundle.

No new foundation model. No new retrieval/reranker inference.
The 4B teacher remains training-time only.

Run:
  python src/stage07_local/run_evidence_distilled_head_oof.py
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
import src.stage06_evidence.benchmark_evidence_packaging as a0

OUT=ROOT/"reports/stage07e_evidence_distill_head"
CACHE=ROOT/"cache/stage07e_evidence_distill_head"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
TEACHER_DEFAULT=ROOT/"results/stage07b/teacher_targets_snapshot.npz"
STAGE07D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"

SHORT=30
DEPTH=20
BASE_R=0.9425976255185238

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True)
    sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-6,1.,sd)
    return ((x-mu)/sd).astype(np.float32)

def rank_from_score(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        o=np.lexsort((short[i],-score[i]))
        out[i]=short[i,o]
    return out

def perq(rank,qids,golds,docs):
    z=[]
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        z.append(len(g&p)/len(g))
    return np.asarray(z,float)

def load_world(teacher_path):
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH)
    done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03 CE cache incomplete/drifted")

    Xflat,_,_,_=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    names=qg.load_names(docs)
    title=qg.make_title_features(qids,questions,short,names).astype(np.float32)
    Xext=np.concatenate([X45,title],axis=2).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)

    aq=np.asarray(np.load(a0.AIT_Q,mmap_mode="r"),np.float32)
    lq=np.asarray(np.load(a0.LAL_Q,mmap_mode="r"),np.float32)
    if aq.shape!=(len(qids),1024) or lq.shape!=(len(qids),1024):
        raise RuntimeError(f"query embedding drift AIT={aq.shape} LAL={lq.shape}")

    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing teacher snapshot: {teacher_path}\n"
            "Copy teacher_targets_snapshot.npz from Drive to results/stage07b/."
        )
    z=np.load(teacher_path)
    teacher=np.asarray(z["scores"],np.float32)
    td=np.asarray(z["done"],np.uint8)
    if teacher.shape!=(len(qids),DEPTH) or int(td.sum())!=len(qids):
        raise RuntimeError(f"teacher snapshot drift {teacher.shape} done={int(td.sum())}")
    teacher=zrows(teacher)

    return qids,questions,golds,folds,stress,docs,short,X45,Xext,y,aq,lq,teacher

def load_witness_index(short,docs):
    qidx=np.arange(len(short),dtype=np.int32)
    vv,rr,sim,view_names=a0.select_witnesses(
        "stage07b_portable_top20_v1",qidx,short[:,:DEPTH],docs
    )
    if vv.shape!=(len(short),DEPTH,2) or rr.shape!=vv.shape:
        raise RuntimeError("witness index drift")

    stores={}
    dims={}
    for v in view_names:
        arr=np.load(a0.VIEWS[v]["emb"],mmap_mode="r")
        stores[v]=arr
        dims[v]=arr.shape[1]
    if any(d!=1024 for d in dims.values()):
        raise RuntimeError(f"expected 1024D witness embeddings, got {dims}")
    print("[witness] exact Stage07B witness cache restored",flush=True)
    return vv,rr,np.asarray(sim,np.float32),view_names,stores

def gather_family(qis,fi,vv,rr,view_names,stores):
    """Return [B,20,1024] float32 for family fi with grouped memmap gathers."""
    qis=np.asarray(qis,np.int32)
    B=len(qis)
    out=np.empty((B,DEPTH,1024),np.float32)
    vsub=vv[qis,:,fi]
    rsub=rr[qis,:,fi]
    for vid,v in enumerate(view_names):
        mask=(vsub==vid)
        if not np.any(mask):
            continue
        rows=rsub[mask].astype(np.int64)
        out[mask]=np.asarray(stores[v][rows],np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("nonfinite witness gather")
    return out

def train_fold(
    outer,tr,he,btr,bhe,short,Xext,y,aq,lq,teacher,
    vv,rr,sim,view_names,stores,args
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_all(args.seed+outer)
    dev=torch.device(args.device)

    class Head(nn.Module):
        def __init__(self,cdim):
            super().__init__()
            h=args.proj
            self.aq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.aw=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.lq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.lw=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.c=nn.Sequential(nn.Linear(cdim+2,96),nn.GELU(),nn.LayerNorm(96))
            # per family: q,w,q*w,|q-w| => 4h; two families => 8h
            self.h=nn.Sequential(
                nn.Linear(8*h+96,256),nn.GELU(),nn.Dropout(.08),
                nn.Linear(256,64),nn.GELU(),
                nn.Linear(64,1),
            )
            self.raw_alpha=nn.Parameter(torch.tensor(-1.5))
        def fam(self,q,w,pq,pw):
            q=pq(q); w=pw(w)
            return torch.cat([q,w,q*w,torch.abs(q-w)],dim=-1)
        def forward(self,qa,wa,ql,wl,c):
            za=self.fam(qa,wa,self.aq,self.aw)
            zl=self.fam(ql,wl,self.lq,self.lw)
            cc=self.c(c)
            return self.h(torch.cat([za,zl,cc],dim=-1)).squeeze(-1)

    model=Head(Xext.shape[-1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    # Small static tensors only. Witness embeddings are streamed from frozen memmaps.
    taq=torch.from_numpy(aq).to(dev)
    tlq=torch.from_numpy(lq).to(dev)
    tx=torch.from_numpy(Xext[:,:DEPTH]).to(dev)
    ty=torch.from_numpy(y[:,:DEPTH]).to(dev)
    tt=torch.from_numpy(teacher).to(dev)
    tsim=torch.from_numpy(sim).to(dev)
    btrz=torch.from_numpy(zrows(btr)[:,:DEPTH]).to(dev)

    trpos={int(q):i for i,q in enumerate(tr)}
    valid=np.asarray([int(q) for q in tr if y[int(q),:DEPTH].sum()>0],np.int32)
    rng=np.random.default_rng(args.seed+outer)
    t0=time.perf_counter();tail=[]

    for ep in range(args.epochs):
        order=valid.copy();rng.shuffle(order);ep_losses=[]
        model.train()
        for st in range(0,len(order),args.batch_queries):
            qis=order[st:st+args.batch_queries]
            B=len(qis)

            wa_np=gather_family(qis,0,vv,rr,view_names,stores)
            wl_np=gather_family(qis,1,vv,rr,view_names,stores)
            wa=torch.from_numpy(wa_np).to(dev,non_blocking=True)
            wl=torch.from_numpy(wl_np).to(dev,non_blocking=True)

            qa=taq[qis][:,None,:].expand(-1,DEPTH,-1)
            ql=tlq[qis][:,None,:].expand(-1,DEPTH,-1)
            # Add exact witness cosine from both families as two cheap features.
            c=torch.cat([tx[qis],tsim[qis]],dim=-1)
            pred=model(
                qa.reshape(B*DEPTH,1024),wa.reshape(B*DEPTH,1024),
                ql.reshape(B*DEPTH,1024),wl.reshape(B*DEPTH,1024),
                c.reshape(B*DEPTH,-1)
            ).reshape(B,DEPTH)

            teach=tt[qis]
            yy=ty[qis]
            base=torch.stack([btrz[trpos[int(q)]] for q in qis],dim=0)
            alpha=F.softplus(model.raw_alpha)
            total=base+alpha*pred

            # Teacher imitation on the exact same witness contract.
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))

        tail=ep_losses[-100:]
        print(
            f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(ep_losses):.5f} "
            f"alpha={float(F.softplus(model.raw_alpha).detach().cpu()):.4f} "
            f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
            flush=True
        )

    model.eval()
    hs=np.empty((len(he),DEPTH),np.float32)
    with torch.inference_mode():
        for st in range(0,len(he),args.eval_batch_queries):
            qis=he[st:st+args.eval_batch_queries]
            B=len(qis)
            wa=torch.from_numpy(gather_family(qis,0,vv,rr,view_names,stores)).to(dev)
            wl=torch.from_numpy(gather_family(qis,1,vv,rr,view_names,stores)).to(dev)
            qa=taq[qis][:,None,:].expand(-1,DEPTH,-1)
            ql=tlq[qis][:,None,:].expand(-1,DEPTH,-1)
            c=torch.cat([tx[qis],tsim[qis]],dim=-1)
            p=model(
                qa.reshape(B*DEPTH,1024),wa.reshape(B*DEPTH,1024),
                ql.reshape(B*DEPTH,1024),wl.reshape(B*DEPTH,1024),
                c.reshape(B*DEPTH,-1)
            ).reshape(B,DEPTH)
            hs[st:st+B]=p.float().cpu().numpy()

    alpha=float(F.softplus(model.raw_alpha).detach().cpu())
    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.
    meta={
        "alpha":alpha,
        "loss_tail":float(np.mean(tail)),
        "train_queries":int(len(tr)),
        "held_queries":int(len(he)),
        "peak_reserved_gib":float(peak),
    }

    del model,opt,taq,tlq,tx,ty,tt,tsim,btrz
    gc.collect()
    if dev.type=="cuda":
        torch.cuda.empty_cache()
    return hs,meta

def evaluate(name,score,short,qids,golds,docs,folds,stress):
    rank=rank_from_score(short,score)
    m=qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)
    return rank,m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,default=TEACHER_DEFAULT)
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch-queries",type=int,default=20)
    ap.add_argument("--eval-batch-queries",type=int,default=40)
    ap.add_argument("--proj",type=int,default=80)
    ap.add_argument("--lr",type=float,default=7e-4)
    ap.add_argument("--kd-weight",type=float,default=.58)
    ap.add_argument("--gold-weight",type=float,default=.32)
    ap.add_argument("--pair-weight",type=float,default=.10)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--margin",type=float,default=.4)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/7] Load frozen ENDGAME world + teacher targets",flush=True)
    qids,questions,golds,folds,stress,docs,short,X45,Xext,y,aq,lq,teacher=load_world(args.teacher)
    print(f"  teacher={teacher.shape}; TRAINING-TIME ONLY",flush=True)

    print("[2/7] Restore exact Stage07B semantic witness indices",flush=True)
    vv,rr,sim,view_names,stores=load_witness_index(short,docs)

    print("[3/7] Strict 5-fold evidence-aware distillation",flush=True)
    oof=np.zeros((len(qids),SHORT),np.float32)
    base_oof=np.zeros_like(oof)
    fold_meta={}
    for outer,fn in enumerate(folds):
        tr,he,btr,bhe=sw.inner_crossfit_prior(X45,y,folds,qids,fn)
        base_oof[he]=bhe
        if args.device=="cuda":
            torch.cuda.reset_peak_memory_stats()
        hs,meta=train_fold(
            outer,tr,he,btr,bhe,short,Xext,y,aq,lq,teacher,
            vv,rr,sim,view_names,stores,args
        )
        fs=zrows(bhe)
        fs[:,:DEPTH]+=meta["alpha"]*hs
        oof[he]=fs
        fold_meta[fn]=meta

    print("[4/7] Official evaluation",flush=True)
    br,bm=evaluate("base",base_oof,short,qids,golds,docs,folds,stress)
    er,em=evaluate("evidence",oof,short,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_R)>3e-6:
        raise RuntimeError(f"baseline parity failed {bm['overall']['recall_at_5']}")

    b=bm["overall"]; e=em["overall"]
    delta=e["recall_at_5"]-b["recall_at_5"]
    bq=perq(br,qids,golds,docs); eq=perq(er,qids,golds,docs)
    fd={f:em["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}
    wins=int((eq>bq).sum()); losses=int((eq<bq).sum())

    print("[5/7] Optional fixed fusion with Stage07D",flush=True)
    fusion=None
    if STAGE07D_SCORE.is_file():
        dscore=np.load(STAGE07D_SCORE)
        if dscore.shape!=oof.shape or not np.isfinite(dscore).all():
            raise RuntimeError(f"Stage07D score cache drift: {dscore.shape}")
        # Precommitted 50/50 row-normalized fusion. No tuning on OOF labels.
        fused=.5*zrows(dscore)+.5*zrows(oof)
        fr,fm=evaluate("fusion",fused,short,qids,golds,docs,folds,stress)
        fq=perq(fr,qids,golds,docs)
        fusion={
            "contract":"fixed 50/50 row-z fusion; no OOF weight tuning",
            "metrics":fm,
            "effect_vs_base":{
                "delta_recall":fm["overall"]["recall_at_5"]-b["recall_at_5"],
                "delta_single":fm["overall"]["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
                "delta_multi":fm["overall"]["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
                "wins":int((fq>bq).sum()),"losses":int((fq<bq).sum()),
            }
        }
        np.save(CACHE/"fixed_fusion_stage07d_stage07e.f32.npy",fused)
    else:
        print("  Stage07D score cache not found; fusion skipped",flush=True)

    print("[6/7] Save reusable scores/report",flush=True)
    np.save(CACHE/"evidence_head_oof_scores30.f32.npy",oof)
    pos=sum(v>0 for v in fd.values())
    if e["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_EVIDENCE_HEAD"

    report={
        "schema":"dsc2026.endgame.stage07e.evidence_distill_head.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; teacher used only as training target",
        "hypothesis":"exact query-selected witness embeddings preserve teacher-relevant local evidence better than parent centroids",
        "new_foundation_model":False,
        "teacher":{"model":"Qwen/Qwen3-Reranker-4B","training_time_only":True,"evaluated_as_arm":False},
        "student":{"architecture":"tiny dual-family witness MLP over frozen AIT/LAL query+witness embeddings + 57D candidate features"},
        "baseline_ce_lr":bm,
        "evidence_distilled_head":em,
        "effect":{
            "delta_recall":delta,
            "delta_single":e["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
            "delta_multi":e["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
            "wins":wins,"losses":losses,"fold_deltas":fd,
        },
        "fold_meta":fold_meta,
        "fixed_fusion_with_stage07d":fusion,
        "decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[7/7] Result")
    print("="*118)
    print(f"BASE          R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    print(f"EVIDENCE-HEAD R={e['recall_at_5']:.9f} single={e['single_gold_recall_at_5']:.9f} multi={e['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: delta={fd[f]:+.9f} alpha={fold_meta[f]['alpha']:.4f}")
    if fusion is not None:
        fm=fusion["metrics"]["overall"]; fe=fusion["effect_vs_base"]
        print("-"*118)
        print(f"FIXED 07D+07E R={fm['recall_at_5']:.9f} single={fm['single_gold_recall_at_5']:.9f} multi={fm['multi_gold_recall_at_5']:.9f}")
        print(f"FUSION DELTA={fe['delta_recall']:+.9f} W/L={fe['wins']}/{fe['losses']}")
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*118)

if __name__=="__main__":
    main()

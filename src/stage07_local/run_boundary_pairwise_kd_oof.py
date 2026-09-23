#!/usr/bin/env python
"""
Stage07K — Boundary-Aware Pairwise Distillation.

Evidence behind this experiment
-------------------------------
- Stage07D is the strongest local distilled head (~0.94585 OOF).
- Stage07E adds local evidence but is weaker overall.
- Stage07J's larger joint/listwise student is weaker again.
Therefore we keep Stage07D's *simple global semantic representation* and change
only the training objective.

Hypothesis
----------
The actual task is top-5 selection, but Stage07D regressed teacher scores
pointwise. Most teacher information that agrees with the current CE-LR ranking
is redundant. What matters is teacher evidence that says:

    a candidate OUTSIDE the current top5 should beat
    a candidate INSIDE the current top5.

Stage07K trains directly on those boundary-crossing preferences.

Training losses
---------------
1) Gold boundary pairwise:
   missed gold outside base top5 vs wrong non-gold inside base top5.
   Retention pairs are added for already-correct golds to prevent destructive
   swaps.

2) Teacher disagreement pairwise:
   on OUTER-TRAIN queries only, take the strongest Qwen4B preferences that
   reverse the inner-crossfit CE-LR top5 boundary.
   Distill them as soft Bradley-Terry targets.

3) Small residual regularizer.

Inference
---------
CE-LR prior + learned alpha * tiny student residual.
No teacher model, no teacher logits, no teacher snapshot feature.

Distillation audit
------------------
For each outer fold, the only teacher tensor materialized for optimization is
teacher[outer_train]. Outer-held teacher rows do not exist in the training
tensor and are not referenced during held inference.

No new foundation model is introduced.

Run:
  python src/stage07_local/run_boundary_pairwise_kd_oof.py
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
import src.stage07_local.run_wide_slot5_rescue_oof as f7

OUT=ROOT/"reports/stage07k_boundary_pairwise_kd"
CACHE=ROOT/"cache/stage07k_boundary_pairwise_kd"

TEACHER_DEFAULT=ROOT/"results/stage07b/teacher_targets_snapshot.npz"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"

SHORT=30
DEPTH=20
BASE_R=0.9425976255185238
D_EXPECT=0.9458518094693177

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-6,1.,sd)
    return ((x-mu)/sd).astype(np.float32)

def rank_from_score(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        o=np.lexsort((short[i],-score[i]))
        out[i]=short[i,o]
    return out

def perq(rank,qids,golds,docs):
    vals=[]
    for i,q in enumerate(qids):
        g=set(golds[q]);p={docs[int(x)] for x in rank[i,:5]}
        vals.append(len(g&p)/len(g))
    return np.asarray(vals,np.float64)

def metrics(score,short,qids,golds,docs,folds,stress):
    rank=rank_from_score(short,score)
    return rank,qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def load_world(teacher_path):
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH);done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("CE cache incomplete/drifted")

    Xflat,_,_,_=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    names=qg.load_names(docs)
    title=qg.make_title_features(qids,questions,short,names).astype(np.float32)
    Xext=np.concatenate([X45,title],axis=2).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)

    aq=np.array(np.load(a0.AIT_Q,mmap_mode="r"),dtype=np.float32,copy=True,order="C")
    if aq.shape!=(len(qids),1024):
        raise RuntimeError(f"AIT query embedding drift {aq.shape}")

    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    z=np.load(teacher_path)
    teacher=np.asarray(z["scores"],np.float32)
    td=np.asarray(z["done"],np.uint8)
    if teacher.shape!=(len(qids),DEPTH) or int(td.sum())!=len(qids):
        raise RuntimeError(f"teacher drift {teacher.shape} done={int(td.sum())}")
    teacher=zrows(teacher)

    return qids,golds,folds,stress,docs,short,X45,Xext,y,aq,teacher

def build_pair_plan(base,yrow,trow,max_teacher_pairs,max_gold_pairs):
    """
    base/yrow/trow are length DEPTH.

    Returns:
      teacher_pairs: (outside_idx, inside_idx, soft_target, confidence)
      gold_pairs:    (positive_idx, negative_idx, weight)
    """
    order=np.argsort(-base,kind="stable")
    inside=list(map(int,order[:5]))
    outside=list(map(int,order[5:]))
    in_set=set(inside)

    teacher_pairs=[]
    # Only boundary reversals: teacher prefers outside candidate over current top5.
    for j in outside:
        for i in inside:
            delta=float(trow[j]-trow[i])
            if delta>0:
                # Bradley-Terry soft target; >0.5 by construction.
                pt=1.0/(1.0+np.exp(-delta))
                teacher_pairs.append((j,i,float(pt),float(delta)))
    teacher_pairs.sort(key=lambda x:x[3],reverse=True)
    teacher_pairs=teacher_pairs[:max_teacher_pairs]

    gold_pairs=[]
    gold=np.where(yrow>0)[0].tolist()
    nongold=np.where(yrow<=0)[0].tolist()

    # Highest-value training signal: a gold currently outside top5 against a
    # wrong non-gold currently inside top5.
    missed=[p for p in gold if p not in in_set]
    wrong_inside=[n for n in inside if yrow[n]<=0]
    for p in missed:
        for n in wrong_inside:
            gold_pairs.append((int(p),int(n),2.0))

    # Retention: protect correct top5 golds against strongest outside non-golds.
    correct_inside=[p for p in inside if yrow[p]>0]
    hard_out=[n for n in outside if yrow[n]<=0][:6]
    for p in correct_inside:
        for n in hard_out:
            gold_pairs.append((int(p),int(n),0.50))

    # If no crossing error exists, retain a few generic hard pairs so the
    # residual does not learn only promotion behavior.
    if not gold_pairs and gold and nongold:
        ng=sorted(nongold,key=lambda n:-base[n])[:6]
        for p in gold:
            for n in ng:
                if p!=n:
                    gold_pairs.append((int(p),int(n),0.35))

    # Deterministic cap: high-value crossing pairs were appended first.
    return teacher_pairs,gold_pairs[:max_gold_pairs]

def train_fold(
    outer,tr,he,btr,bhe,short,Xext,y,aq,teacher,cent,args
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
            self.q=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.d=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.c=nn.Sequential(nn.Linear(cdim,96),nn.GELU(),nn.LayerNorm(96))
            self.h=nn.Sequential(
                nn.Linear(4*h+96,256),nn.GELU(),nn.Dropout(.06),
                nn.Linear(256,64),nn.GELU(),
                nn.Linear(64,1),
            )
            self.raw_alpha=nn.Parameter(torch.tensor(-1.5))
        def forward(self,q,d,c):
            qq=self.q(q);dd=self.d(d);cc=self.c(c)
            z=torch.cat([qq,dd,qq*dd,torch.abs(qq-dd),cc],dim=-1)
            return self.h(z).squeeze(-1)

    model=Head(Xext.shape[-1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    tq=torch.from_numpy(np.array(aq,copy=True,order="C")).to(dev)
    tc=torch.from_numpy(np.array(cent,copy=True,order="C")).to(dev)
    tx=torch.from_numpy(np.array(Xext[:,:DEPTH],copy=True,order="C")).to(dev)
    ty=torch.from_numpy(np.array(y[:,:DEPTH],copy=True,order="C")).to(dev)

    # Strict distillation boundary.
    teacher_train=np.array(teacher[tr],dtype=np.float32,copy=True,order="C")
    tt=torch.from_numpy(teacher_train).to(dev)
    trpos={int(q):i for i,q in enumerate(tr)}
    btrz=torch.from_numpy(np.array(zrows(btr)[:,:DEPTH],copy=True,order="C")).to(dev)

    rng=np.random.default_rng(args.seed+outer)
    train=np.asarray(tr,np.int32)
    t0=time.perf_counter();tail=[]
    diag={"teacher_pairs":0,"gold_pairs":0,"queries_teacher_disagree":0,
          "queries_gold_boundary_error":0}

    for ep in range(args.epochs):
        order=train.copy();rng.shuffle(order)
        ep_losses=[];ep_t=ep_g=ep_qt=ep_qg=0
        model.train()

        for st in range(0,len(order),args.batch_queries):
            qis=order[st:st+args.batch_queries]
            B=len(qis)
            docs20=short[qis,:DEPTH].reshape(-1)

            qrep=tq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            demb=tc[torch.as_tensor(docs20,device=dev)]
            cfeat=tx[qis].reshape(B*DEPTH,-1)
            pred=model(qrep,demb,cfeat).reshape(B,DEPTH)

            base=torch.stack([btrz[trpos[int(q)]] for q in qis],dim=0)
            teach=torch.stack([tt[trpos[int(q)]] for q in qis],dim=0)
            yy=ty[qis]
            alpha=F.softplus(model.raw_alpha)
            total=base+alpha*pred

            losses=[]
            for bi,q in enumerate(qis):
                bp=base[bi].detach().cpu().numpy()
                yp=yy[bi].detach().cpu().numpy()
                tp=teach[bi].detach().cpu().numpy()
                tpair,gpair=build_pair_plan(
                    bp,yp,tp,args.max_teacher_pairs,args.max_gold_pairs
                )

                qloss=torch.zeros((),device=dev)
                has=False

                if tpair:
                    ep_qt+=1;ep_t+=len(tpair)
                    oi=torch.tensor([x[0] for x in tpair],dtype=torch.long,device=dev)
                    ii=torch.tensor([x[1] for x in tpair],dtype=torch.long,device=dev)
                    target=torch.tensor([x[2] for x in tpair],dtype=torch.float32,device=dev)
                    conf=torch.tensor([min(x[3],3.0)/3.0 for x in tpair],
                                      dtype=torch.float32,device=dev)
                    # Student residual preference. Teacher corrections should
                    # appear in residual space because base already provides
                    # the current ranking.
                    logits=(pred[bi,oi]-pred[bi,ii])/args.teacher_temp
                    kd=F.binary_cross_entropy_with_logits(
                        logits,target,reduction="none"
                    )
                    kd=(kd*(0.5+conf)).mean()
                    qloss=qloss+args.kd_weight*kd
                    has=True

                if gpair:
                    ep_qg+=1;ep_g+=len(gpair)
                    pi=torch.tensor([x[0] for x in gpair],dtype=torch.long,device=dev)
                    ni=torch.tensor([x[1] for x in gpair],dtype=torch.long,device=dev)
                    wt=torch.tensor([x[2] for x in gpair],dtype=torch.float32,device=dev)
                    diff=total[bi,pi]-total[bi,ni]
                    gl=(F.softplus(args.margin-diff)*wt).mean()
                    # Exact macro-Recall marginal weighting.
                    ngold=max(float(yp.sum()),1.0)
                    qloss=qloss+args.gold_weight*(gl/ngold)
                    has=True

                if has:
                    # Keep correction small unless pair supervision justifies it.
                    qloss=qloss+args.residual_l2*(pred[bi]**2).mean()
                    losses.append(qloss)

            if not losses:
                continue
            loss=torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))

        diag["teacher_pairs"]+=ep_t
        diag["gold_pairs"]+=ep_g
        diag["queries_teacher_disagree"]+=ep_qt
        diag["queries_gold_boundary_error"]+=ep_qg
        tail=ep_losses[-100:]
        print(
            f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(ep_losses):.5f} "
            f"alpha={float(F.softplus(model.raw_alpha).detach().cpu()):.4f} "
            f"teacherPairs={ep_t} goldPairs={ep_g} "
            f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
            flush=True
        )

    model.eval()
    hs=np.empty((len(he),DEPTH),np.float32)
    with torch.inference_mode():
        for st in range(0,len(he),args.eval_batch_queries):
            qis=he[st:st+args.eval_batch_queries]
            B=len(qis)
            docs20=short[qis,:DEPTH].reshape(-1)
            qrep=tq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            demb=tc[torch.as_tensor(docs20,device=dev)]
            p=model(qrep,demb,tx[qis].reshape(B*DEPTH,-1)).reshape(B,DEPTH)
            hs[st:st+B]=p.float().cpu().numpy()

    alpha=float(F.softplus(model.raw_alpha).detach().cpu())
    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.
    meta={
        "alpha":alpha,
        "loss_tail":float(np.mean(tail)) if tail else None,
        "train_queries":int(len(tr)),"held_queries":int(len(he)),
        "peak_reserved_gib":float(peak),
        "pair_diagnostics":diag,
    }

    del model,opt,tq,tc,tx,ty,tt,btrz
    gc.collect()
    if dev.type=="cuda":torch.cuda.empty_cache()
    return hs,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,default=TEACHER_DEFAULT)
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=7)
    ap.add_argument("--batch-queries",type=int,default=48)
    ap.add_argument("--eval-batch-queries",type=int,default=96)
    ap.add_argument("--proj",type=int,default=80)
    ap.add_argument("--lr",type=float,default=6e-4)
    ap.add_argument("--kd-weight",type=float,default=.55)
    ap.add_argument("--gold-weight",type=float,default=.45)
    ap.add_argument("--residual-l2",type=float,default=.003)
    ap.add_argument("--teacher-temp",type=float,default=.80)
    ap.add_argument("--margin",type=float,default=.35)
    ap.add_argument("--max-teacher-pairs",type=int,default=12)
    ap.add_argument("--max-gold-pairs",type=int,default=18)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/6] Load frozen world + teacher",flush=True)
    qids,golds,folds,stress,docs,short,X45,Xext,y,aq,teacher=load_world(args.teacher)

    print("[2/6] Reuse/build AIT parent centroids",flush=True)
    cent=f7.build_centroid("ait",f7.AIT_REGION,docs,args.device)

    print("[3/6] Strict 5-fold boundary-pairwise KD",flush=True)
    oof=np.zeros((len(qids),SHORT),np.float32)
    base_oof=np.zeros_like(oof)
    fold_meta={}

    for outer,fn in enumerate(folds):
        tr,he,btr,bhe=sw.inner_crossfit_prior(X45,y,folds,qids,fn)
        base_oof[he]=bhe
        if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
        hs,meta=train_fold(
            outer,tr,he,btr,bhe,short,Xext,y,aq,teacher,cent,args
        )
        fs=zrows(bhe)
        fs[:,:DEPTH]+=meta["alpha"]*hs
        oof[he]=fs
        fold_meta[fn]=meta

    print("[4/6] Official evaluation",flush=True)
    br,bm=metrics(base_oof,short,qids,golds,docs,folds,stress)
    kr,km=metrics(oof,short,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_R)>3e-6:
        raise RuntimeError(f"CE-LR parity failed {bm['overall']['recall_at_5']}")

    dref=None
    if D_SCORE.is_file():
        ds=np.load(D_SCORE)
        dr,dm=metrics(ds,short,qids,golds,docs,folds,stress)
        if abs(dm["overall"]["recall_at_5"]-D_EXPECT)>3e-6:
            raise RuntimeError(f"Stage07D parity failed {dm['overall']['recall_at_5']}")
        dref=dm

    b=bm["overall"];k=km["overall"]
    delta=k["recall_at_5"]-b["recall_at_5"]
    bq=perq(br,qids,golds,docs);kq=perq(kr,qids,golds,docs)
    wins=int((kq>bq).sum());losses=int((kq<bq).sum())
    fd={f:km["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}

    np.save(CACHE/"boundary_pairwise_oof_scores30.f32.npy",oof)
    np.save(CACHE/"boundary_pairwise_oof_rank.i32.npy",kr.astype(np.int32))

    pos=sum(v>0 for v in fd.values())
    if k["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_BOUNDARY_PAIRWISE_KD"

    print("[5/6] Save report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage07k.boundary_pairwise_kd.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; held teacher rows structurally absent from optimization",
        "teacher":{"model":"Qwen/Qwen3-Reranker-4B","training_time_only":True,"evaluated_as_arm":False},
        "new_foundation_model":False,
        "student":{
            "architecture":"Stage07D-style tiny global semantic residual head",
            "distillation":"soft pairwise teacher preferences only on CE-LR top5 boundary reversals",
        },
        "baseline_ce_lr":bm,
        "stage07d_reference":dref,
        "boundary_pairwise":km,
        "effect":{
            "delta_recall":delta,
            "delta_single":k["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
            "delta_multi":k["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
            "wins":wins,"losses":losses,"fold_deltas":fd,
        },
        "fold_meta":fold_meta,
        "decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[6/6] Result")
    print("="*120)
    print(f"CE-LR BASE   R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    if dref is not None:
        d=dref["overall"]
        print(f"STAGE07D     R={d['recall_at_5']:.9f} single={d['single_gold_recall_at_5']:.9f} multi={d['multi_gold_recall_at_5']:.9f}")
    print(f"BOUNDARY-KD  R={k['recall_at_5']:.9f} single={k['single_gold_recall_at_5']:.9f} multi={k['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA vs CE={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        pd=fold_meta[f]["pair_diagnostics"]
        print(
            f"{f}: CE-delta={fd[f]:+.9f} alpha={fold_meta[f]['alpha']:.4f} "
            f"teacherPairs={pd['teacher_pairs']} goldPairs={pd['gold_pairs']}"
        )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*120)

if __name__=="__main__":
    main()

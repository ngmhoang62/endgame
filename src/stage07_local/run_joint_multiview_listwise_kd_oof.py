#!/usr/bin/env python
"""
Stage07J — Joint Multi-View Listwise Distillation.

Goal
----
Stage07D (whole-document semantic centroid) and Stage07E (exact local evidence)
are both positive but a fixed score fusion is not. Train ONE tiny student that
sees both views and learns a query/candidate-conditioned gate between them.

Crucially, this also changes the distillation objective:
  - previous local heads regressed teacher z-scores pointwise (SmoothL1);
  - Stage07J distills the teacher's LISTWISE ranking distribution (KL), which
    matches the retrieval objective better;
  - gold listwise + hard-boundary pairwise supervision still anchors the model
    to official labels.

Teacher contract
----------------
Qwen3-Reranker-4B is TRAINING-TIME ONLY.
For each outer fold, only teacher targets for OUTER-TRAIN queries are
materialized into the optimization tensor/dictionary. Outer-held teacher
targets are structurally inaccessible inside the training loop.
Held inference uses only the tiny student + frozen approved embeddings/features.

No new foundation model is introduced.

Run:
  python src/stage07_local/run_joint_multiview_listwise_kd_oof.py
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

OUT=ROOT/"reports/stage07j_joint_multiview_kd"
CACHE=ROOT/"cache/stage07j_joint_multiview_kd"

TEACHER_DEFAULT=ROOT/"results/stage07b/teacher_targets_snapshot.npz"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"
E_SCORE=ROOT/"cache/stage07e_evidence_distill_head/evidence_head_oof_scores30.f32.npy"

SHORT=30
DEPTH=20
BASE_R=0.9425976255185238
D_EXPECT=0.9458518094693177
E_EXPECT=0.9452558050827254

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
    out=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        out[i]=len(g&p)/len(g)
    return out

def load_world(teacher_path):
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH); done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03 CE cache incomplete/drifted")

    Xflat,_,_,_=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    names=qg.load_names(docs)
    title=qg.make_title_features(qids,questions,short,names).astype(np.float32)
    Xext=np.concatenate([X45,title],axis=2).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)

    aq=np.array(np.load(a0.AIT_Q,mmap_mode="r"),dtype=np.float32,copy=True,order="C")
    lq=np.array(np.load(a0.LAL_Q,mmap_mode="r"),dtype=np.float32,copy=True,order="C")
    if aq.shape!=(len(qids),1024) or lq.shape!=(len(qids),1024):
        raise RuntimeError("query embedding drift")

    qhand=np.stack([qg.q_hand(questions[q]) for q in qids]).astype(np.float32)

    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    z=np.load(teacher_path)
    teacher=np.asarray(z["scores"],np.float32)
    td=np.asarray(z["done"],np.uint8)
    if teacher.shape!=(len(qids),DEPTH) or int(td.sum())!=len(qids):
        raise RuntimeError(f"teacher drift {teacher.shape} done={int(td.sum())}")
    teacher=zrows(teacher)

    return qids,questions,golds,folds,stress,docs,short,X45,Xext,y,aq,lq,qhand,teacher

def load_witnesses(short,docs):
    qidx=np.arange(len(short),dtype=np.int32)
    vv,rr,sim,view_names=a0.select_witnesses(
        "stage07b_portable_top20_v1",qidx,short[:,:DEPTH],docs
    )
    stores={}
    for v in view_names:
        arr=np.load(a0.VIEWS[v]["emb"],mmap_mode="r")
        if arr.shape[1]!=1024: raise RuntimeError(f"{v} dim drift {arr.shape}")
        stores[v]=arr
    return vv,rr,np.asarray(sim,np.float32),view_names,stores

def gather_family(qis,fi,vv,rr,view_names,stores):
    qis=np.asarray(qis,np.int32)
    out=np.empty((len(qis),DEPTH,1024),np.float32)
    vsub=vv[qis,:,fi]; rsub=rr[qis,:,fi]
    for vid,v in enumerate(view_names):
        m=(vsub==vid)
        if np.any(m):
            out[m]=np.asarray(stores[v][rsub[m].astype(np.int64)],np.float32)
    if not np.isfinite(out).all(): raise RuntimeError("nonfinite witness gather")
    return out

def metrics(score,short,qids,golds,docs,folds,stress):
    rank=rank_from_score(short,score)
    return rank,qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def train_fold(
    outer,tr,he,btr,bhe,short,Xext,y,aq,lq,qhand,teacher,
    vv,rr,sim,view_names,stores,acent,lcent,args
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_all(args.seed+outer)
    dev=torch.device(args.device)

    class JointStudent(nn.Module):
        def __init__(self,cdim,qhdim):
            super().__init__()
            h=args.proj
            # Global document view.
            self.gaq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.gad=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.glq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.gld=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.global_head=nn.Sequential(
                nn.Linear(8*h,192),nn.GELU(),nn.Dropout(.06),
                nn.Linear(192,64),nn.GELU(),nn.Linear(64,1),
            )

            # Local exact-evidence view.
            self.laq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.law=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.llq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.llw=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.local_head=nn.Sequential(
                nn.Linear(8*h,192),nn.GELU(),nn.Dropout(.06),
                nn.Linear(192,64),nn.GELU(),nn.Linear(64,1),
            )

            # Query/candidate-conditioned gate + a small metadata residual.
            self.meta=nn.Sequential(
                nn.Linear(cdim+2+qhdim,128),nn.GELU(),nn.LayerNorm(128),
            )
            self.gate=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
            self.meta_resid=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
            self.raw_alpha=nn.Parameter(torch.tensor(-1.5))

        @staticmethod
        def fam(q,d,pq,pd):
            q=pq(q); d=pd(d)
            return torch.cat([q,d,q*d,torch.abs(q-d)],dim=-1)

        def forward(self,qa,da,ql,dl,wa,wl,c,qh):
            zg=torch.cat([
                self.fam(qa,da,self.gaq,self.gad),
                self.fam(ql,dl,self.glq,self.gld),
            ],dim=-1)
            zl=torch.cat([
                self.fam(qa,wa,self.laq,self.law),
                self.fam(ql,wl,self.llq,self.llw),
            ],dim=-1)
            gs=self.global_head(zg).squeeze(-1)
            ls=self.local_head(zl).squeeze(-1)

            m=self.meta(torch.cat([c,qh],dim=-1))
            gate=torch.sigmoid(self.gate(m).squeeze(-1))
            mr=self.meta_resid(m).squeeze(-1)
            pred=gate*gs+(1.0-gate)*ls+0.25*mr
            return pred,gate,F.softplus(self.raw_alpha)

    model=JointStudent(Xext.shape[-1],qhand.shape[1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    # Writable arrays; no mmap warning.
    taq=torch.from_numpy(np.array(aq,copy=True,order="C")).to(dev)
    tlq=torch.from_numpy(np.array(lq,copy=True,order="C")).to(dev)
    tac=torch.from_numpy(np.array(acent,copy=True,order="C")).to(dev)
    tlc=torch.from_numpy(np.array(lcent,copy=True,order="C")).to(dev)
    tx=torch.from_numpy(np.array(Xext[:,:DEPTH],copy=True,order="C")).to(dev)
    ty=torch.from_numpy(np.array(y[:,:DEPTH],copy=True,order="C")).to(dev)
    tsim=torch.from_numpy(np.array(sim,copy=True,order="C")).to(dev)
    tqh=torch.from_numpy(np.array(qhand,copy=True,order="C")).to(dev)

    # Hard distillation boundary: held-fold teacher rows do not exist in tt.
    teacher_train=np.array(teacher[tr],dtype=np.float32,copy=True,order="C")
    tt=torch.from_numpy(teacher_train).to(dev)
    trpos={int(q):i for i,q in enumerate(tr)}

    btrz=torch.from_numpy(np.array(zrows(btr)[:,:DEPTH],copy=True,order="C")).to(dev)
    valid=np.asarray([int(q) for q in tr if y[int(q),:DEPTH].sum()>0],np.int32)

    rng=np.random.default_rng(args.seed+outer)
    t0=time.perf_counter();tail=[];gate_tail=[]

    for ep in range(args.epochs):
        order=valid.copy();rng.shuffle(order)
        ep_losses=[];ep_gates=[]
        model.train()

        for st in range(0,len(order),args.batch_queries):
            qis=order[st:st+args.batch_queries]
            B=len(qis)

            wa=torch.from_numpy(gather_family(qis,0,vv,rr,view_names,stores)).to(dev)
            wl=torch.from_numpy(gather_family(qis,1,vv,rr,view_names,stores)).to(dev)

            docs20=short[qis,:DEPTH]
            di=torch.as_tensor(docs20.reshape(-1),device=dev)

            qa=taq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            ql=tlq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            da=tac[di]; dl=tlc[di]

            c=torch.cat([tx[qis],tsim[qis]],dim=-1).reshape(B*DEPTH,-1)
            qh=tqh[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,-1)

            pred,gate,alpha=model(
                qa,da,ql,dl,
                wa.reshape(B*DEPTH,1024),
                wl.reshape(B*DEPTH,1024),
                c,qh
            )
            pred=pred.reshape(B,DEPTH)
            gate=gate.reshape(B,DEPTH)
            ep_gates.append(float(gate.mean().detach().cpu()))

            teach=torch.stack([tt[trpos[int(q)]] for q in qis],dim=0)
            yy=ty[qis]
            base=torch.stack([btrz[trpos[int(q)]] for q in qis],dim=0)
            total=base+alpha*pred

            # LISTWISE KD: match teacher preference distribution, not raw logits.
            logp=F.log_softmax(pred/args.teacher_temp,dim=1)
            pt=F.softmax(teach/args.teacher_temp,dim=1)
            kd=F.kl_div(logp,pt,reduction="batchmean")*(args.teacher_temp**2)

            cnt=yy.sum(1,keepdim=True).clamp_min(1)
            gold_target=yy/cnt
            gold_each=-(gold_target*F.log_softmax(total/args.gold_temp,dim=1)).sum(1)

            # Emphasize queries where the current inner-CV prior still misses
            # at least one available gold, without changing evaluation weights.
            with torch.no_grad():
                k5=torch.topk(base,k=5,dim=1).indices
                got=torch.gather(yy,1,k5).sum(1)
                avail=yy.sum(1)
                hard=(got<avail).float()
                multi=(avail>1).float()
                qweight=1.0+args.hard_boost*hard+args.multi_boost*multi
            gold=(gold_each*qweight).mean()

            pair_losses=[]
            for bi in range(B):
                pos=torch.where(yy[bi]>0)[0]
                neg=torch.where(yy[bi]==0)[0]
                if len(pos)==0 or len(neg)==0:continue
                hardneg=neg[torch.topk(base[bi,neg],k=min(6,len(neg))).indices]
                diff=total[bi,pos][:,None]-total[bi,hardneg][None,:]
                pair_losses.append(F.softplus(args.margin-diff).mean()*qweight[bi])
            pair=torch.stack(pair_losses).mean() if pair_losses else torch.zeros((),device=dev)

            # Mild gate entropy regularizer prevents immediate all-global/all-local collapse.
            ge=(gate*(1-gate)).mean()

            loss=(args.kd_weight*kd
                  +args.gold_weight*gold
                  +args.pair_weight*pair
                  -args.gate_entropy_weight*ge)

            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            ep_losses.append(float(loss.detach().cpu()))

        tail=ep_losses[-100:]; gate_tail=ep_gates[-100:]
        print(
            f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(ep_losses):.5f} "
            f"alpha={float(torch.nn.functional.softplus(model.raw_alpha).detach().cpu()):.4f} "
            f"gate_global={np.mean(ep_gates):.3f} elapsed={(time.perf_counter()-t0)/60:.1f}m",
            flush=True
        )

    model.eval()
    hs=np.empty((len(he),DEPTH),np.float32)
    gate_he=np.empty((len(he),DEPTH),np.float32)

    with torch.inference_mode():
        for st in range(0,len(he),args.eval_batch_queries):
            qis=he[st:st+args.eval_batch_queries]
            B=len(qis)
            wa=torch.from_numpy(gather_family(qis,0,vv,rr,view_names,stores)).to(dev)
            wl=torch.from_numpy(gather_family(qis,1,vv,rr,view_names,stores)).to(dev)
            docs20=short[qis,:DEPTH]
            di=torch.as_tensor(docs20.reshape(-1),device=dev)

            qa=taq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            ql=tlq[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,1024)
            c=torch.cat([tx[qis],tsim[qis]],dim=-1).reshape(B*DEPTH,-1)
            qh=tqh[qis][:,None,:].expand(-1,DEPTH,-1).reshape(B*DEPTH,-1)

            p,g,_=model(
                qa,tac[di],ql,tlc[di],
                wa.reshape(B*DEPTH,1024),
                wl.reshape(B*DEPTH,1024),
                c,qh
            )
            hs[st:st+B]=p.reshape(B,DEPTH).float().cpu().numpy()
            gate_he[st:st+B]=g.reshape(B,DEPTH).float().cpu().numpy()

    alpha=float(torch.nn.functional.softplus(model.raw_alpha).detach().cpu())
    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.

    meta={
        "alpha":alpha,
        "loss_tail":float(np.mean(tail)),
        "train_gate_global_mean":float(np.mean(gate_tail)),
        "held_gate_global_mean":float(gate_he.mean()),
        "held_gate_global_std":float(gate_he.std()),
        "train_queries":int(len(tr)),
        "held_queries":int(len(he)),
        "peak_reserved_gib":float(peak),
    }

    del model,opt,taq,tlq,tac,tlc,tx,ty,tsim,tqh,tt,btrz
    gc.collect()
    if dev.type=="cuda":torch.cuda.empty_cache()
    return hs,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,default=TEACHER_DEFAULT)
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch-queries",type=int,default=16)
    ap.add_argument("--eval-batch-queries",type=int,default=32)
    ap.add_argument("--proj",type=int,default=64)
    ap.add_argument("--lr",type=float,default=6e-4)
    ap.add_argument("--teacher-temp",type=float,default=.80)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--kd-weight",type=float,default=.50)
    ap.add_argument("--gold-weight",type=float,default=.38)
    ap.add_argument("--pair-weight",type=float,default=.12)
    ap.add_argument("--hard-boost",type=float,default=.35)
    ap.add_argument("--multi-boost",type=float,default=.15)
    ap.add_argument("--gate-entropy-weight",type=float,default=.01)
    ap.add_argument("--margin",type=float,default=.4)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/7] Load frozen world + teacher targets",flush=True)
    qids,questions,golds,folds,stress,docs,short,X45,Xext,y,aq,lq,qhand,teacher=(
        load_world(args.teacher)
    )

    print("[2/7] Restore exact local evidence witnesses",flush=True)
    vv,rr,sim,view_names,stores=load_witnesses(short,docs)

    print("[3/7] Reuse/build global AIT + LAL parent centroids",flush=True)
    ac=f7.build_centroid("ait",f7.AIT_REGION,docs,args.device)
    lc=f7.build_centroid("lal",f7.LAL_REGION,docs,args.device)

    print("[4/7] Strict 5-fold joint multi-view KD",flush=True)
    oof=np.zeros((len(qids),SHORT),np.float32)
    base_oof=np.zeros_like(oof)
    fold_meta={}

    for outer,fn in enumerate(folds):
        tr,he,btr,bhe=sw.inner_crossfit_prior(X45,y,folds,qids,fn)
        base_oof[he]=bhe
        if args.device=="cuda":torch.cuda.reset_peak_memory_stats()

        hs,meta=train_fold(
            outer,tr,he,btr,bhe,short,Xext,y,aq,lq,qhand,teacher,
            vv,rr,sim,view_names,stores,ac,lc,args
        )
        fs=zrows(bhe)
        fs[:,:DEPTH]+=meta["alpha"]*hs
        oof[he]=fs
        fold_meta[fn]=meta

    print("[5/7] Official evaluation + frozen baselines",flush=True)
    br,bm=metrics(base_oof,short,qids,golds,docs,folds,stress)
    jr,jm=metrics(oof,short,qids,golds,docs,folds,stress)

    if abs(bm["overall"]["recall_at_5"]-BASE_R)>3e-6:
        raise RuntimeError(f"CE-LR parity failed {bm['overall']['recall_at_5']}")

    baselines={}
    for name,path,expected in [
        ("stage07d",D_SCORE,D_EXPECT),
        ("stage07e",E_SCORE,E_EXPECT),
    ]:
        if path.is_file():
            s=np.load(path)
            rr0,mm=metrics(s,short,qids,golds,docs,folds,stress)
            baselines[name]=mm
            if abs(mm["overall"]["recall_at_5"]-expected)>3e-6:
                raise RuntimeError(f"{name} parity failed {mm['overall']['recall_at_5']}")

    b=bm["overall"];j=jm["overall"]
    delta=j["recall_at_5"]-b["recall_at_5"]
    bq=perq(br,qids,golds,docs);jq=perq(jr,qids,golds,docs)
    fd={f:jm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}
    wins=int((jq>bq).sum());losses=int((jq<bq).sum())

    np.save(CACHE/"joint_multiview_oof_scores30.f32.npy",oof)
    np.save(CACHE/"joint_multiview_oof_rank.i32.npy",jr.astype(np.int32))

    pos=sum(v>0 for v in fd.values())
    if j["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_JOINT_MULTIVIEW_KD"

    print("[6/7] Save report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage07j.joint_multiview_listwise_kd.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; held teacher targets structurally absent from optimization",
        "teacher":{"model":"Qwen/Qwen3-Reranker-4B","training_time_only":True,"evaluated_as_arm":False},
        "new_foundation_model":False,
        "student":{
            "architecture":"tiny gated global-centroid + local-witness dual-family student",
            "distillation":"listwise KL over teacher ranking distribution",
        },
        "baseline_ce_lr":bm,
        "frozen_reference_baselines":baselines,
        "joint_student":jm,
        "effect":{
            "delta_recall":delta,
            "delta_single":j["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
            "delta_multi":j["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
            "wins":wins,"losses":losses,"fold_deltas":fd,
        },
        "fold_meta":fold_meta,
        "decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[7/7] Result")
    print("="*122)
    print(f"CE-LR BASE   R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    for name,mm in baselines.items():
        o=mm["overall"]
        print(f"{name.upper():12s} R={o['recall_at_5']:.9f} single={o['single_gold_recall_at_5']:.9f} multi={o['multi_gold_recall_at_5']:.9f}")
    print(f"JOINT-KD     R={j['recall_at_5']:.9f} single={j['single_gold_recall_at_5']:.9f} multi={j['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA vs CE={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(
            f"{f}: CE-delta={fd[f]:+.9f} "
            f"alpha={fold_meta[f]['alpha']:.4f} "
            f"gate_global={fold_meta[f]['held_gate_global_mean']:.3f}"
        )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*122)

if __name__=="__main__":
    main()

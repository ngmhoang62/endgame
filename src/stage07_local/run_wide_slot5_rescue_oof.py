#!/usr/bin/env python
"""
Stage07F — Wide-pool semantic Slot-5 Rescue.

Motivation
----------
Stage07D reached ~0.94585 OOF and Stage07E improved multi-gold, but both still
rank only the old top-30 world. Acquisition evidence says a much wider union
already contains almost all golds.

Instead of destabilizing all five output slots, lock Stage07D's top-4 and learn
ONLY the fifth slot from a wide cached candidate pool.

Candidate pool:
  union(top50) from
    - AIT atomic
    - AIT coarse1024
    - LAL b4
    - BM25
  + old Stage03 top30
  capped after deterministic RRF ordering.

Student features:
  - frozen AIT query + parent-centroid interactions
  - frozen LAL query + parent-centroid interactions
  - all six cached retrieval source geometries
  - Stage07D/Stage07E/CE score/rank features when candidate is in old top30
  - optional Qwen4B teacher KD only on candidates that are in teacher top20

The 4B teacher is NEVER evaluated or loaded at inference. This script consumes
only its already-produced training targets.

Strict 5-fold OOF:
  Stage07D OOF top4 is frozen for every query.
  For each outer fold, rescue model trains only on other folds' gold labels.
  Outer-held labels never affect training.

Run:
  python src/stage07_local/run_wide_slot5_rescue_oof.py
"""
from __future__ import annotations

import argparse, gc, json, math, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage02_candidate_generation.screen_vietnamese_retrievers as s2
import src.stage06_evidence.benchmark_evidence_packaging as a0

OUT=ROOT/"reports/stage07f_wide_slot5_rescue"
CACHE=ROOT/"cache/stage07f_wide_slot5_rescue"
D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"
E_SCORE=ROOT/"cache/stage07e_evidence_distill_head/evidence_head_oof_scores30.f32.npy"
CE_SCORE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
TEACHER_DEFAULT=ROOT/"results/stage07b/teacher_targets_snapshot.npz"

AIT_REGION=ROOT/"cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy"
LAL_REGION=ROOT/"cache/stage02b4_vi_screen/vnlegal_lal/region_embeddings.f32.npy"

BASE_EXPECT=0.9458518094693177
OLD=30
TEACH_DEPTH=20
POOL_DEPTH=50
POOL_CAP=160

CORE_POOL_SOURCES=("ait_atomic","ait_coarse1024","lal_b4","bm25")

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def zrow(x):
    x=np.asarray(x,np.float32)
    sd=float(x.std())
    if sd<1e-6: sd=1.
    return (x-float(x.mean()))/sd

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-6,1.,sd)
    return ((x-mu)/sd).astype(np.float32)

def rank_old(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        o=np.lexsort((short[i],-score[i]))
        out[i]=short[i,o]
    return out

def metrics(rank,qids,golds,docs,folds,stress):
    return qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def perq(rank,qids,golds,docs):
    z=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        z[i]=len(g&p)/len(g)
    return z

def build_centroid(name,emb_path,docs,device):
    CACHE.mkdir(parents=True,exist_ok=True)
    cp=CACHE/f"{name}_parent_centroids.f32.npy"
    if cp.is_file():
        x=np.load(cp,mmap_mode="r")
        if x.shape==(len(docs),1024) and np.isfinite(x).all():
            print(f"[centroid/{name}] cache hit",flush=True)
            return np.asarray(x,np.float32)

    if not emb_path.is_file():
        raise FileNotFoundError(emb_path)
    geom=s2.load_geometry()
    if list(map(str,geom["doc_ids"]))!=list(map(str,docs)):
        raise RuntimeError(f"{name}: geometry/doc order drift")
    pidx=np.asarray(geom["parent_index"],np.int64)
    reg=np.load(emb_path,mmap_mode="r")
    if reg.shape!=(len(pidx),1024):
        raise RuntimeError(f"{name}: region embedding drift {reg.shape}")

    import torch
    dev=torch.device(device)
    sums=torch.zeros((len(docs),1024),dtype=torch.float32,device=dev)
    cnt=torch.zeros((len(docs),1),dtype=torch.float32,device=dev)
    chunk=4096 if dev.type=="cuda" else 2048
    print(f"[centroid/{name}] building from {len(reg):,} regions",flush=True)
    for st in range(0,len(reg),chunk):
        en=min(st+chunk,len(reg))
        e=torch.from_numpy(np.asarray(reg[st:en],np.float32).copy()).to(dev)
        ii=torch.from_numpy(pidx[st:en].copy()).to(dev)
        sums.index_add_(0,ii,e)
        cnt.index_add_(0,ii,torch.ones((en-st,1),device=dev))
    c=torch.nn.functional.normalize(sums/cnt.clamp_min(1),dim=1)
    out=c.cpu().numpy().astype(np.float32)
    np.save(cp,out)
    del sums,cnt,c
    if dev.type=="cuda": torch.cuda.empty_cache()
    return out

def load_world(teacher_path):
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))

    if not D_SCORE.is_file():
        raise FileNotFoundError(D_SCORE)
    ds=np.asarray(np.load(D_SCORE),np.float32)
    if ds.shape!=(len(qids),OLD) or not np.isfinite(ds).all():
        raise RuntimeError(f"Stage07D score drift {ds.shape}")

    es=None
    if E_SCORE.is_file():
        es=np.asarray(np.load(E_SCORE),np.float32)
        if es.shape!=ds.shape or not np.isfinite(es).all():
            raise RuntimeError(f"Stage07E score drift {es.shape}")

    ce=np.asarray(np.load(CE_SCORE),np.float32)
    if ce.shape!=ds.shape or not np.isfinite(ce).all():
        raise RuntimeError(f"CE score drift {ce.shape}")

    aq=np.array(np.load(a0.AIT_Q,mmap_mode="r"),dtype=np.float32,copy=True,order="C")
    lq=np.array(np.load(a0.LAL_Q,mmap_mode="r"),dtype=np.float32,copy=True,order="C")
    if aq.shape!=(len(qids),1024) or lq.shape!=(len(qids),1024):
        raise RuntimeError("query embedding drift")

    teacher=None
    if teacher_path.is_file():
        z=np.load(teacher_path)
        teacher=np.asarray(z["scores"],np.float32)
        td=np.asarray(z["done"],np.uint8)
        if teacher.shape!=(len(qids),TEACH_DEPTH) or int(td.sum())!=len(qids):
            raise RuntimeError("teacher target drift")
        teacher=zrows(teacher)
        print("[teacher] restored training targets; never used at held inference",flush=True)
    else:
        print("[teacher] snapshot absent -> KD disabled, gold-only rescue",flush=True)

    return qids,questions,golds,folds,stress,docs,short,sources,ds,es,ce,aq,lq,teacher

def source_query_maps(sources,qi):
    rms={}; zms={}; gms={}
    for s,(idx,scr) in sources.items():
        ids=np.asarray(idx[qi,:100],np.int32)
        ss=np.asarray(scr[qi,:100],np.float64)
        mu=float(ss.mean()); sd=float(ss.std())
        if sd<1e-8: sd=1.
        top=float(ss[0])
        rms[s]={int(d):r+1 for r,d in enumerate(ids)}
        zms[s]={int(d):float((float(ss[r])-mu)/sd) for r,d in enumerate(ids)}
        gms[s]={int(d):float((top-float(ss[r]))/sd) for r,d in enumerate(ids)}
    return rms,zms,gms

def build_pools(short,sources,ds,es,ce):
    pools=[]; feats=[]
    src_names=list(b1.SOURCE_NAMES)
    print("[pool] build deterministic wide rescue pools",flush=True)
    for qi in range(len(short)):
        rms,zms,gms=source_query_maps(sources,qi)
        union=set(map(int,short[qi]))
        for s in CORE_POOL_SOURCES:
            union.update(map(int,np.asarray(sources[s][0][qi,:POOL_DEPTH],np.int32)))

        # deterministic RRF preliminary order
        rr=[]
        for d in union:
            rrf=sum(1.0/(60.0+rms[s][d]) for s in src_names if d in rms[s])
            rr.append((rrf,d))
        rr.sort(key=lambda x:(-x[0],x[1]))
        ordered=[d for _,d in rr[:POOL_CAP]]

        oldpos={int(d):j for j,d in enumerate(short[qi])}
        # Guarantee all old top30 survive cap.
        present=set(ordered)
        for d in map(int,short[qi]):
            if d not in present:
                ordered.append(d);present.add(d)

        row=[]
        dsz=zrow(ds[qi])
        esz=zrow(es[qi]) if es is not None else None
        cez=zrow(ce[qi])
        d_order=np.empty(OLD,np.int32)
        do=np.lexsort((short[qi],-ds[qi])); d_order[do]=np.arange(1,OLD+1)
        if es is not None:
            e_order=np.empty(OLD,np.int32)
            eo=np.lexsort((short[qi],-es[qi])); e_order[eo]=np.arange(1,OLD+1)
        c_order=np.empty(OLD,np.int32)
        co=np.lexsort((short[qi],-ce[qi])); c_order[co]=np.arange(1,OLD+1)

        for d in ordered:
            f=[]; ranks=[]; c5=c10=c20=c50=0; rrf60=0.
            for s in src_names:
                r=rms[s].get(d)
                if r is None:
                    f += [0.,0.,1.2,-3.,4.]
                else:
                    f += [1.,1./(10.+r),r/100.,zms[s][d],gms[s][d]]
                    ranks.append(r);rrf60+=1./(60.+r)
                    c5+=r<=5;c10+=r<=10;c20+=r<=20;c50+=r<=50
            f += [
                float(len(ranks)),rrf60,
                min(ranks)/100. if ranks else 1.2,
                float(np.mean(ranks))/100. if ranks else 1.2,
                float(c5),float(c10),float(c20),float(c50),
            ]
            j=oldpos.get(d)
            if j is None:
                f += [0.,-4.,1.2, 0.,-4.,1.2, 0.,-4.,1.2]
            else:
                f += [1.,float(dsz[j]),float(d_order[j]/OLD)]
                if es is not None:
                    f += [1.,float(esz[j]),float(e_order[j]/OLD)]
                else:
                    f += [0.,-4.,1.2]
                f += [1.,float(cez[j]),float(c_order[j]/OLD)]
            row.append(f)

        pools.append(np.asarray(ordered,np.int32))
        feats.append(np.asarray(row,np.float32))
        if (qi+1)%1000==0 or qi+1==len(short):
            print(f"  pools {qi+1}/{len(short)}",flush=True)
    return pools,feats

def oracle_slot5(base_rank,pools,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    out=base_rank.copy()
    changed=0
    for qi,q in enumerate(qids):
        top4=list(map(int,base_rank[qi,:4]))
        g={d2i[d] for d in golds[q]}
        candidates=[int(d) for d in pools[qi] if int(d) not in set(top4)]
        pick=None
        for d in candidates:
            if d in g:
                pick=d;break
        if pick is not None:
            out[qi,4]=pick
            if pick!=int(base_rank[qi,4]): changed+=1
    return out,changed

def teacher_map_for_query(qi,short,teacher):
    if teacher is None: return {}
    return {int(short[qi,j]):float(teacher[qi,j]) for j in range(TEACH_DEPTH)}

def train_fold(
    outer,train,held,base_rank,pools,feats,qids,golds,docs,short,
    aq,lq,acent,lcent,teacher,args
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_all(args.seed+outer)
    dev=torch.device(args.device)
    d2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{d2i[d] for d in golds[q]} for q in qids]
    gold_count=np.asarray([len(golds[q]) for q in qids],np.float32)

    fdim=feats[0].shape[1]

    class Rescue(nn.Module):
        def __init__(self):
            super().__init__()
            h=args.proj
            self.aq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.ad=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.lq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.ld=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.c=nn.Sequential(nn.Linear(fdim+3,96),nn.GELU(),nn.LayerNorm(96))
            self.h=nn.Sequential(
                nn.Linear(8*h+96,256),nn.GELU(),nn.Dropout(.08),
                nn.Linear(256,64),nn.GELU(),nn.Linear(64,1)
            )
        def fam(self,q,d,pq,pd):
            q=pq(q);d=pd(d)
            return torch.cat([q,d,q*d,torch.abs(q-d)],dim=-1)
        def forward(self,qa,da,ql,dl,c):
            return self.h(torch.cat([
                self.fam(qa,da,self.aq,self.ad),
                self.fam(ql,dl,self.lq,self.ld),
                self.c(c)
            ],dim=-1)).squeeze(-1)

    model=Rescue().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    taq=torch.from_numpy(np.array(aq,copy=True,order="C")).to(dev)
    tlq=torch.from_numpy(np.array(lq,copy=True,order="C")).to(dev)
    tac=torch.from_numpy(np.array(acent,copy=True,order="C")).to(dev)
    tlc=torch.from_numpy(np.array(lcent,copy=True,order="C")).to(dev)

    rng=np.random.default_rng(args.seed+outer)
    t0=time.perf_counter();tail=[]
    train=np.asarray(train,np.int32)

    # Distillation audit boundary: materialize teacher targets ONLY for
    # outer-training queries. Held-fold teacher targets are not addressable
    # anywhere inside the optimization loop.
    teacher_local=None
    if teacher is not None and args.kd_weight>0:
        teacher_local={
            int(qi): teacher_map_for_query(int(qi),short,teacher)
            for qi in train
        }

    for ep in range(args.epochs):
        order=train.copy();rng.shuffle(order);losses=[]
        model.train()
        for st in range(0,len(order),args.batch_queries):
            qs=order[st:st+args.batch_queries]
            total_loss=torch.zeros((),device=dev)
            used_queries=0
            for qi in qs:
                top4=set(map(int,base_rank[qi,:4]))
                cand=np.asarray([int(d) for d in pools[qi] if int(d) not in top4],np.int32)
                if len(cand)==0: continue

                yy=np.asarray([float(int(d) in gold_idx[int(qi)]) for d in cand],np.float32)
                # A fifth-slot decision only changes Recall when at least one
                # remaining gold exists in the rescue pool. Queries without one
                # are deliberately excluded from the optimization objective.
                if yy.sum()<=0:
                    continue

                # numeric features aligned to filtered candidate order
                posmap={int(d):j for j,d in enumerate(pools[qi])}
                nf=np.stack([feats[qi][posmap[int(d)]] for d in cand],axis=0)
                keep5=(cand==int(base_rank[qi,4])).astype(np.float32)[:,None]
                prelim=np.asarray([1./(1.+j) for j in range(len(cand))],np.float32)[:,None]
                oldpresent=np.asarray([[float(int(d) in set(map(int,short[qi])))] for d in cand],np.float32)
                nf=np.concatenate([nf,keep5,prelim,oldpresent],axis=1)

                qa=taq[qi][None,:].expand(len(cand),-1)
                ql=tlq[qi][None,:].expand(len(cand),-1)
                di=torch.as_tensor(cand,device=dev)
                da=tac[di];dl=tlc[di]
                cc=torch.from_numpy(nf).to(dev)
                sc=model(qa,da,ql,dl,cc)

                yt=torch.from_numpy(yy).to(dev)
                target=yt/yt.sum()
                listwise=-(target*torch.log_softmax(sc/args.temperature,dim=0)).sum()
                qloss=listwise

                # Optional KD is structurally limited to outer-training queries.
                # The teacher remains an auxiliary training target only.
                if teacher_local is not None:
                    tm=teacher_local[int(qi)]
                    loc=[j for j,d in enumerate(cand) if int(d) in tm]
                    if len(loc)>=2:
                        lt=torch.as_tensor(loc,device=dev)
                        target_t=torch.tensor(
                            [tm[int(cand[j])] for j in loc],
                            dtype=torch.float32,device=dev
                        )
                        pred_t=sc[lt]
                        pred_t=(pred_t-pred_t.mean())/pred_t.std(unbiased=False).clamp_min(1e-4)
                        kd=F.smooth_l1_loss(pred_t,target_t,beta=.5)
                        qloss=qloss+args.kd_weight*kd

                # Exact marginal contribution of a correct slot-5 rescue to
                # macro Recall is 1 / number_of_gold_documents.
                qweight=1.0/max(float(gold_count[int(qi)]),1.0)
                total_loss=total_loss+qweight*qloss
                used_queries+=1

            if used_queries==0: continue
            loss=total_loss/used_queries
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach().cpu()))
        tail=losses[-100:]
        print(f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(losses):.5f} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m",flush=True)

    model.eval()
    picks={}
    with torch.inference_mode():
        for qi in held:
            top4=set(map(int,base_rank[qi,:4]))
            cand=np.asarray([int(d) for d in pools[qi] if int(d) not in top4],np.int32)
            posmap={int(d):j for j,d in enumerate(pools[qi])}
            nf=np.stack([feats[qi][posmap[int(d)]] for d in cand],axis=0)
            keep5=(cand==int(base_rank[qi,4])).astype(np.float32)[:,None]
            prelim=np.asarray([1./(1.+j) for j in range(len(cand))],np.float32)[:,None]
            oldpresent=np.asarray([[float(int(d) in set(map(int,short[qi])))] for d in cand],np.float32)
            nf=np.concatenate([nf,keep5,prelim,oldpresent],axis=1)

            scores=[]
            for st in range(0,len(cand),args.eval_candidates):
                dd=cand[st:st+args.eval_candidates]
                B=len(dd)
                qa=taq[qi][None,:].expand(B,-1);ql=tlq[qi][None,:].expand(B,-1)
                di=torch.as_tensor(dd,device=dev)
                cc=torch.from_numpy(nf[st:st+B]).to(dev)
                s=model(qa,tac[di],ql,tlc[di],cc)
                scores.append(s.float().cpu().numpy())
            scores=np.concatenate(scores)
            j=int(np.lexsort((cand,-scores))[0])
            picks[int(qi)]=int(cand[j])

    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.
    meta={"train_queries":int(len(train)),"held_queries":int(len(held)),
          "loss_tail":float(np.mean(tail)) if tail else None,
          "peak_reserved_gib":float(peak)}
    del model,opt,taq,tlq,tac,tlc
    gc.collect()
    if dev.type=="cuda":torch.cuda.empty_cache()
    return picks,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,default=TEACHER_DEFAULT)
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--batch-queries",type=int,default=10)
    ap.add_argument("--eval-candidates",type=int,default=256)
    ap.add_argument("--proj",type=int,default=64)
    ap.add_argument("--lr",type=float,default=7e-4)
    ap.add_argument("--temperature",type=float,default=1.0)
    ap.add_argument("--kd-weight",type=float,default=.20)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/8] Load frozen world",flush=True)
    qids,questions,golds,folds,stress,docs,short,sources,ds,es,ce,aq,lq,teacher=load_world(args.teacher)
    base_rank=rank_old(short,ds)
    bm=metrics(base_rank,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_EXPECT)>3e-6:
        raise RuntimeError(f"Stage07D parity failed {bm['overall']['recall_at_5']}")
    print(f"  Stage07D parity R={bm['overall']['recall_at_5']:.9f}",flush=True)

    print("[2/8] Build wide pools/features",flush=True)
    pools,feats=build_pools(short,sources,ds,es,ce)
    sizes=np.asarray([len(x) for x in pools])
    print(f"  pool mean={sizes.mean():.1f} median={np.median(sizes):.0f} max={sizes.max()}",flush=True)

    print("[3/8] Fixed-top4 slot5 oracle",flush=True)
    orank,ochanged=oracle_slot5(base_rank,pools,qids,golds,docs)
    om=metrics(orank,qids,golds,docs,folds,stress)
    print(f"  SLOT5 ORACLE R={om['overall']['recall_at_5']:.9f} "
          f"single={om['overall']['single_gold_recall_at_5']:.9f} "
          f"multi={om['overall']['multi_gold_recall_at_5']:.9f}",flush=True)

    print("[4/8] Build/reuse AIT+LAL parent centroids",flush=True)
    ac=build_centroid("ait",AIT_REGION,docs,args.device)
    lc=build_centroid("lal",LAL_REGION,docs,args.device)

    print("[5/8] Strict 5-fold rescue training",flush=True)
    q2i={q:i for i,q in enumerate(qids)}
    rescued=base_rank.copy()
    fold_meta={}
    for outer,(fn,held_ids) in enumerate(folds.items()):
        he=np.asarray([q2i[q] for q in held_ids],np.int32)
        hs=set(map(int,he))
        tr=np.asarray([i for i in range(len(qids)) if i not in hs],np.int32)
        if args.device=="cuda":torch.cuda.reset_peak_memory_stats()
        picks,meta=train_fold(
            outer,tr,he,base_rank,pools,feats,qids,golds,docs,short,
            aq,lq,ac,lc,teacher,args
        )
        changed=0
        for qi,d in picks.items():
            if int(rescued[qi,4])!=int(d): changed+=1
            rescued[qi,4]=int(d)
        meta["changed_rank5"]=changed
        fold_meta[fn]=meta
        print(f"[{fn}] rank5 changes={changed}/{len(he)}",flush=True)

    print("[6/8] Official evaluation",flush=True)
    rm=metrics(rescued,qids,golds,docs,folds,stress)
    b=bm["overall"];r=rm["overall"]
    delta=r["recall_at_5"]-b["recall_at_5"]
    bq=perq(base_rank,qids,golds,docs);rq=perq(rescued,qids,golds,docs)
    fd={f:rm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}
    wins=int((rq>bq).sum());losses=int((rq<bq).sum())

    print("[7/8] Save report/ranking",flush=True)
    np.save(CACHE/"rescued_oof_rank.i32.npy",rescued.astype(np.int32))
    pos=sum(v>0 for v in fd.values())
    if r["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_WIDE_SLOT5_RESCUE"

    report={
        "schema":"dsc2026.endgame.stage07f.wide_slot5_rescue.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; Stage07D top4 frozen; teacher targets outer-train only",
        "new_foundation_model":False,
        "pool":{"sources":CORE_POOL_SOURCES,"source_depth":POOL_DEPTH,"cap":POOL_CAP,
                "mean_size":float(sizes.mean()),"median_size":float(np.median(sizes)),
                "max_size":int(sizes.max())},
        "baseline_stage07d":bm,
        "slot5_oracle":{"metrics":om,"changed_queries":ochanged},
        "rescued":rm,
        "effect":{"delta_recall":delta,
                  "delta_single":r["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
                  "delta_multi":r["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
                  "wins":wins,"losses":losses,"fold_deltas":fd},
        "fold_meta":fold_meta,
        "teacher_kd":{"enabled":teacher is not None and args.kd_weight>0,
                      "weight":args.kd_weight,
                      "teacher_runtime_inference":False},
        "decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("[8/8] Result")
    print("="*120)
    print(f"BASE Stage07D R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    oo=om["overall"]
    print(f"SLOT5 ORACLE  R={oo['recall_at_5']:.9f} single={oo['single_gold_recall_at_5']:.9f} multi={oo['multi_gold_recall_at_5']:.9f}")
    print(f"RESCUED       R={r['recall_at_5']:.9f} single={r['single_gold_recall_at_5']:.9f} multi={r['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: delta={fd[f]:+.9f} changes={fold_meta[f]['changed_rank5']}")
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*120)

if __name__=="__main__":
    main()

#!/usr/bin/env python
"""
Stage07G — Conservative Anchor-Pairwise Slot-5 Rescue.

Why this exists
---------------
Stage07F proved enormous fixed-top4 headroom:
    Stage07D ~0.94585 -> slot5 oracle ~0.98926
but its listwise scorer changed ~80% of rank-5 choices and LOST overall Recall.

This experiment changes the learning target. It does NOT learn "rank the wide
pool". It learns exactly one binary decision:

    Is challenger c better than the CURRENT rank-5 anchor for official Recall?

Only decisive training pairs are used:
  +1 : challenger is gold, current rank5 is not gold
  -1 : current rank5 is gold, challenger is not gold
   0 : ignored (swap is Recall-neutral)

The model learns a global positive KEEP BIAS on the current rank5. At inference,
a replacement happens only if the best challenger beats current rank5 + keep_bias.

No teacher. No self-distillation. No new foundation model.
Uses only already-approved/frozen AIT/LAL embeddings + cached retrieval features.

Run:
  python src/stage07_local/run_anchor_pairwise_slot5_rescue_oof.py
"""
from __future__ import annotations
import argparse, gc, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage06_evidence.benchmark_evidence_packaging as a0
import src.stage07_local.run_wide_slot5_rescue_oof as f7

OUT=ROOT/"reports/stage07g_anchor_pairwise_rescue"
CACHE=ROOT/"cache/stage07g_anchor_pairwise_rescue"
BASE_EXPECT=0.9458518094693177

def seed_all(s):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def metrics(rank,qids,golds,docs,folds,stress):
    return qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def perq(rank,qids,golds,docs):
    out=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        out[i]=len(g&p)/len(g)
    return out

def candidate_numeric(qi,cand,pools,feats,short,base_rank):
    posmap={int(d):j for j,d in enumerate(pools[qi])}
    nf=np.stack([feats[qi][posmap[int(d)]] for d in cand],axis=0)
    keep5=(cand==int(base_rank[qi,4])).astype(np.float32)[:,None]
    prelim=np.asarray([1./(1.+posmap[int(d)]) for d in cand],np.float32)[:,None]
    oldset=set(map(int,short[qi]))
    oldpresent=np.asarray([[float(int(d) in oldset)] for d in cand],np.float32)
    return np.concatenate([nf,keep5,prelim,oldpresent],axis=1).astype(np.float32)

def train_fold(
    outer,train,held,base_rank,pools,feats,qids,golds,docs,short,
    aq,lq,acent,lcent,args
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_all(args.seed+outer)
    dev=torch.device(args.device)
    d2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{d2i[d] for d in golds[q]} for q in qids]
    gold_count=np.asarray([len(golds[q]) for q in qids],np.float32)
    fdim=feats[0].shape[1]+3

    class Rescue(nn.Module):
        def __init__(self):
            super().__init__()
            h=args.proj
            self.aq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.ad=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.lq=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.ld=nn.Sequential(nn.Linear(1024,h),nn.GELU(),nn.LayerNorm(h))
            self.c=nn.Sequential(nn.Linear(fdim,96),nn.GELU(),nn.LayerNorm(96))
            self.h=nn.Sequential(
                nn.Linear(8*h+96,256),nn.GELU(),nn.Dropout(.08),
                nn.Linear(256,64),nn.GELU(),
                nn.Linear(64,1),
            )
            # Positive by construction: current rank5 receives a learned
            # conservative prior. softplus(-1.0) ~= 0.313 initially.
            self.raw_keep=nn.Parameter(torch.tensor(-1.0))
        def fam(self,q,d,pq,pd):
            q=pq(q); d=pd(d)
            return torch.cat([q,d,q*d,torch.abs(q-d)],dim=-1)
        def score(self,qa,da,ql,dl,c):
            z=torch.cat([
                self.fam(qa,da,self.aq,self.ad),
                self.fam(ql,dl,self.lq,self.ld),
                self.c(c),
            ],dim=-1)
            return self.h(z).squeeze(-1)
        def keep_bias(self):
            return F.softplus(self.raw_keep)

    model=Rescue().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)

    # Writable CPU copies => no NumPy/PyTorch mmap warning.
    taq=torch.from_numpy(np.array(aq,dtype=np.float32,copy=True,order="C")).to(dev)
    tlq=torch.from_numpy(np.array(lq,dtype=np.float32,copy=True,order="C")).to(dev)
    tac=torch.from_numpy(np.array(acent,dtype=np.float32,copy=True,order="C")).to(dev)
    tlc=torch.from_numpy(np.array(lcent,dtype=np.float32,copy=True,order="C")).to(dev)

    train=np.asarray(train,np.int32)
    rng=np.random.default_rng(args.seed+outer)
    t0=time.perf_counter(); tail=[]
    pair_stats={"positive":0,"negative":0}

    for ep in range(args.epochs):
        order=train.copy(); rng.shuffle(order)
        losses=[]; ep_pos=ep_neg=0
        model.train()

        for st in range(0,len(order),args.batch_queries):
            qs=order[st:st+args.batch_queries]
            qlosses=[]

            for qi in qs:
                cur=int(base_rank[qi,4])
                top4=set(map(int,base_rank[qi,:4]))
                g=gold_idx[int(qi)]
                cur_gold=cur in g

                challengers=np.asarray(
                    [int(d) for d in pools[qi]
                     if int(d) not in top4 and int(d)!=cur],
                    np.int32
                )
                if len(challengers)==0:
                    continue

                if cur_gold:
                    # Only non-gold challengers can make Recall worse.
                    decisive=np.asarray([d for d in challengers if int(d) not in g],np.int32)
                    sign=-1.0
                    ep_neg+=len(decisive)
                else:
                    # Only gold challengers can improve Recall.
                    decisive=np.asarray([d for d in challengers if int(d) in g],np.int32)
                    sign=+1.0
                    ep_pos+=len(decisive)

                if len(decisive)==0:
                    continue

                # Hard cap for speed. Positives are usually tiny; negatives can be large.
                if len(decisive)>args.max_decisive:
                    # deterministic-ish hard sample using wide-pool order
                    decisive=decisive[:args.max_decisive]

                cand=np.concatenate([np.asarray([cur],np.int32),decisive])
                nf=candidate_numeric(int(qi),cand,pools,feats,short,base_rank)
                di=torch.as_tensor(cand,device=dev)
                B=len(cand)

                qa=taq[qi][None,:].expand(B,-1)
                ql=tlq[qi][None,:].expand(B,-1)
                cc=torch.from_numpy(nf).to(dev)
                sc=model.score(qa,tac[di],ql,tlc[di],cc)

                # Current rank5 is the anchor. Its learned keep-bias is part of
                # the actual inference rule, so training/inference are aligned.
                s_cur=sc[0]+model.keep_bias()
                s_ch=sc[1:]

                if sign>0:
                    # Challenger gold, current not: challenger must beat anchor.
                    pl=F.softplus(args.margin-(s_ch-s_cur)).mean()
                else:
                    # Current gold, challenger not: anchor must beat challenger.
                    pl=F.softplus(args.margin-(s_cur-s_ch)).mean()

                # Exact marginal macro-Recall value of one correct slot.
                w=1.0/max(float(gold_count[int(qi)]),1.0)
                qlosses.append(w*pl)

            if not qlosses:
                continue
            loss=torch.stack(qlosses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        pair_stats["positive"]+=ep_pos
        pair_stats["negative"]+=ep_neg
        tail=losses[-100:]
        kb=float(model.keep_bias().detach().cpu())
        print(
            f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(losses):.5f} "
            f"keep_bias={kb:.4f} decisive_pos={ep_pos} decisive_neg={ep_neg} "
            f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
            flush=True
        )

    model.eval()
    picks={}
    margins={}
    with torch.inference_mode():
        for qi in held:
            cur=int(base_rank[qi,4])
            top4=set(map(int,base_rank[qi,:4]))
            challengers=np.asarray(
                [int(d) for d in pools[qi]
                 if int(d) not in top4 and int(d)!=cur],
                np.int32
            )
            if len(challengers)==0:
                picks[int(qi)]=cur; margins[int(qi)]=float("-inf")
                continue

            cand=np.concatenate([np.asarray([cur],np.int32),challengers])
            nf=candidate_numeric(int(qi),cand,pools,feats,short,base_rank)

            all_sc=[]
            for cs in range(0,len(cand),args.eval_candidates):
                dd=cand[cs:cs+args.eval_candidates]
                B=len(dd)
                di=torch.as_tensor(dd,device=dev)
                qa=taq[qi][None,:].expand(B,-1)
                ql=tlq[qi][None,:].expand(B,-1)
                cc=torch.from_numpy(nf[cs:cs+B]).to(dev)
                s=model.score(qa,tac[di],ql,tlc[di],cc)
                all_sc.append(s.float().cpu().numpy())
            sc=np.concatenate(all_sc)

            current_score=float(sc[0])+float(model.keep_bias().detach().cpu())
            chal_sc=sc[1:]
            j=int(np.lexsort((challengers,-chal_sc))[0])
            best=int(challengers[j])
            margin=float(chal_sc[j]-current_score)

            # Precommitted gate: replace iff challenger actually beats the
            # keep-biased current anchor. No post-hoc OOF threshold search.
            if margin>0.0:
                picks[int(qi)]=best
            else:
                picks[int(qi)]=cur
            margins[int(qi)]=margin

    peak=torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0.
    meta={
        "train_queries":int(len(train)),
        "held_queries":int(len(held)),
        "loss_tail":float(np.mean(tail)) if tail else None,
        "keep_bias":float(model.keep_bias().detach().cpu()),
        "decisive_pairs_seen":pair_stats,
        "peak_reserved_gib":float(peak),
    }

    del model,opt,taq,tlq,tac,tlc
    gc.collect()
    if dev.type=="cuda":
        torch.cuda.empty_cache()
    return picks,margins,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch-queries",type=int,default=12)
    ap.add_argument("--eval-candidates",type=int,default=256)
    ap.add_argument("--max-decisive",type=int,default=24)
    ap.add_argument("--proj",type=int,default=64)
    ap.add_argument("--lr",type=float,default=7e-4)
    ap.add_argument("--margin",type=float,default=.5)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    import torch
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)

    print("[1/7] Load frozen world WITHOUT teacher",flush=True)
    # Explicit nonexistent teacher path => Stage07F loader disables KD.
    no_teacher=ROOT/"results/stage07b/__NO_TEACHER_FOR_STAGE07G__.npz"
    qids,questions,golds,folds,stress,docs,short,sources,ds,es,ce,aq,lq,teacher=(
        f7.load_world(no_teacher)
    )
    if teacher is not None:
        raise RuntimeError("Stage07G must be teacher-free")

    base_rank=f7.rank_old(short,ds)
    bm=metrics(base_rank,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_EXPECT)>3e-6:
        raise RuntimeError(f"Stage07D parity failed {bm['overall']['recall_at_5']}")
    print(f"  Stage07D parity R={bm['overall']['recall_at_5']:.9f}",flush=True)

    print("[2/7] Rebuild/reuse deterministic wide pools",flush=True)
    pools,feats=f7.build_pools(short,sources,ds,es,ce)
    sizes=np.asarray([len(x) for x in pools])

    print("[3/7] Verify fixed-top4 slot5 oracle",flush=True)
    orank,ochanged=f7.oracle_slot5(base_rank,pools,qids,golds,docs)
    om=metrics(orank,qids,golds,docs,folds,stress)
    print(
        f"  SLOT5 ORACLE R={om['overall']['recall_at_5']:.9f} "
        f"single={om['overall']['single_gold_recall_at_5']:.9f} "
        f"multi={om['overall']['multi_gold_recall_at_5']:.9f}",
        flush=True
    )

    print("[4/7] Reuse AIT/LAL parent centroids",flush=True)
    ac=f7.build_centroid("ait",f7.AIT_REGION,docs,args.device)
    lc=f7.build_centroid("lal",f7.LAL_REGION,docs,args.device)

    print("[5/7] Strict 5-fold anchor-pairwise rescue",flush=True)
    q2i={q:i for i,q in enumerate(qids)}
    rescued=base_rank.copy()
    all_margin=np.full(len(qids),np.nan,np.float32)
    fold_meta={}

    for outer,(fn,held_ids) in enumerate(folds.items()):
        he=np.asarray([q2i[q] for q in held_ids],np.int32)
        hs=set(map(int,he))
        tr=np.asarray([i for i in range(len(qids)) if i not in hs],np.int32)

        if args.device=="cuda":
            torch.cuda.reset_peak_memory_stats()

        picks,margins,meta=train_fold(
            outer,tr,he,base_rank,pools,feats,qids,golds,docs,short,
            aq,lq,ac,lc,args
        )

        changed=0
        for qi,d in picks.items():
            if int(rescued[qi,4])!=int(d):
                changed+=1
            rescued[qi,4]=int(d)
            all_margin[qi]=float(margins[qi])

        meta["changed_rank5"]=int(changed)
        meta["change_rate"]=float(changed/len(he))
        fold_meta[fn]=meta
        print(
            f"[{fn}] rank5 changes={changed}/{len(he)} "
            f"({100*changed/len(he):.1f}%) keep_bias={meta['keep_bias']:.4f}",
            flush=True
        )

    print("[6/7] Official evaluation",flush=True)
    rm=metrics(rescued,qids,golds,docs,folds,stress)
    b=bm["overall"]; r=rm["overall"]
    delta=r["recall_at_5"]-b["recall_at_5"]
    bq=perq(base_rank,qids,golds,docs)
    rq=perq(rescued,qids,golds,docs)
    wins=int((rq>bq).sum()); losses=int((rq<bq).sum())
    fd={
        fn:rm["per_fold"][fn]["recall_at_5"]-bm["per_fold"][fn]["recall_at_5"]
        for fn in folds
    }

    np.save(CACHE/"rescued_oof_rank.i32.npy",rescued.astype(np.int32))
    np.save(CACHE/"held_swap_margin.f32.npy",all_margin)

    pos=sum(v>0 for v in fd.values())
    if r["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_ANCHOR_PAIRWISE_RESCUE"

    report={
        "schema":"dsc2026.endgame.stage07g.anchor_pairwise_slot5.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict 5-fold OOF; no teacher; no new foundation model",
        "hypothesis":"learn only decisive challenger-vs-current-rank5 swaps with a conservative learned keep bias",
        "teacher_used":False,
        "new_foundation_model":False,
        "pool":{
            "sources":f7.CORE_POOL_SOURCES,
            "source_depth":f7.POOL_DEPTH,
            "cap":f7.POOL_CAP,
            "mean_size":float(sizes.mean()),
            "median_size":float(np.median(sizes)),
            "max_size":int(sizes.max()),
        },
        "baseline_stage07d":bm,
        "slot5_oracle":{"metrics":om,"changed_queries":ochanged},
        "rescued":rm,
        "effect":{
            "delta_recall":delta,
            "delta_single":r["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
            "delta_multi":r["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
            "wins":wins,"losses":losses,
            "fold_deltas":fd,
        },
        "fold_meta":fold_meta,
        "decision":decision,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )

    print("[7/7] Result")
    print("="*120)
    print(f"BASE Stage07D R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    oo=om["overall"]
    print(f"SLOT5 ORACLE  R={oo['recall_at_5']:.9f} single={oo['single_gold_recall_at_5']:.9f} multi={oo['multi_gold_recall_at_5']:.9f}")
    print(f"ANCHOR-RESCUE R={r['recall_at_5']:.9f} single={r['single_gold_recall_at_5']:.9f} multi={r['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for fn in folds:
        print(
            f"{fn}: delta={fd[fn]:+.9f} "
            f"changes={fold_meta[fn]['changed_rank5']} "
            f"keep_bias={fold_meta[fn]['keep_bias']:.4f}"
        )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*120)

if __name__=="__main__":
    main()

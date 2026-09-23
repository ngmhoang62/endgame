#!/usr/bin/env python
"""
Stage07H — Nested-CV Calibrated Utility Rescue.

Stage07F/G failure diagnosis:
- both searched a very wide challenger pool;
- max-over-many noisy scores created an extreme-value problem;
- Stage07G's keep_bias lived on an arbitrary neural score scale and swapped
  essentially every query.

This experiment directly predicts the *official Recall utility of a swap*:

    utility(q, challenger)
      = [I(challenger is gold) - I(current rank5 is gold)] / |gold(q)|

So targets are exactly {-1/g, 0, +1/g}. Neutral swaps are explicitly included.

For every OUTER fold:
1) build 4-way INNER OOF utility predictions on outer-train queries;
2) select challenger depth K and swap threshold ONLY from inner OOF;
3) require stable positive inner-fold deltas and <= max-change-rate;
4) fit one utility regressor on all outer-train queries;
5) score the untouched outer-held fold.

No teacher. No self-distillation. No new foundation model.
Stage07D top-4 remains locked.

Run:
  python src/stage07_local/run_calibrated_utility_slot5_oof.py

Requires xgboost (already used by Stage07C).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_wide_slot5_rescue_oof as f7

OUT=ROOT/"reports/stage07h_calibrated_utility_rescue"
CACHE=ROOT/"cache/stage07h_calibrated_utility_rescue"

BASE_EXPECT=0.9458518094693177
TRAIN_K=20
K_GRID=(3,5,8,12,20)
QUANTILES=(85.0,90.0,92.5,95.0,97.0,98.0,99.0,99.5)

def metrics(rank,qids,golds,docs,folds,stress):
    return qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def perq(rank,qids,golds,docs):
    out=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]);p={docs[int(x)] for x in rank[i,:5]}
        out[i]=len(g&p)/len(g)
    return out

def qidx_by_fold(qids,folds):
    q2i={q:i for i,q in enumerate(qids)}
    return {f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

def ordered_challengers(qi,base_rank,pools,k):
    top4=set(map(int,base_rank[qi,:4]))
    cur=int(base_rank[qi,4])
    out=[]
    for d0 in pools[qi]:
        d=int(d0)
        if d in top4 or d==cur:
            continue
        out.append(d)
        if len(out)>=k:
            break
    return np.asarray(out,np.int32)

def raw_feat(qi,d,pools,feats,short,base_rank):
    posmap={int(x):j for j,x in enumerate(pools[qi])}
    j=posmap[int(d)]
    f=np.asarray(feats[qi][j],np.float32)
    oldset=set(map(int,short[qi]))
    extra=np.asarray([
        1.0 if int(d)==int(base_rank[qi,4]) else 0.0,
        1.0/(1.0+j),
        1.0 if int(d) in oldset else 0.0,
    ],np.float32)
    return np.concatenate([f,extra])

def pair_feat(qi,d,cur,pools,feats,short,base_rank,qhand):
    ch=raw_feat(qi,d,pools,feats,short,base_rank)
    cu=raw_feat(qi,cur,pools,feats,short,base_rank)
    return np.concatenate([ch,cu,ch-cu,np.abs(ch-cu),qhand[qi]]).astype(np.float32)

def build_examples(indices,base_rank,pools,feats,short,qhand,gold_idx,gold_count,k=TRAIN_K):
    X=[]; y=[]; w=[]; owner=[]; cand=[]
    for qi in indices:
        qi=int(qi); cur=int(base_rank[qi,4]); cg=float(cur in gold_idx[qi])
        chs=ordered_challengers(qi,base_rank,pools,k)
        for d in chs:
            dg=float(int(d) in gold_idx[qi])
            util=(dg-cg)/max(float(gold_count[qi]),1.0)
            X.append(pair_feat(qi,int(d),cur,pools,feats,short,base_rank,qhand))
            y.append(util)
            # Neutral examples are essential for calibration, but decisive
            # examples receive extra weight because they are rare.
            w.append(1.0 + (4.0 if util!=0 else 0.0))
            owner.append(qi); cand.append(int(d))
    return (
        np.asarray(X,np.float32),
        np.asarray(y,np.float32),
        np.asarray(w,np.float32),
        np.asarray(owner,np.int32),
        np.asarray(cand,np.int32),
    )

def train_xgb(xgb,X,y,w,args):
    d=xgb.DMatrix(X,label=y,weight=w)
    params={
        "objective":"reg:pseudohubererror",
        "eval_metric":"rmse",
        "tree_method":"hist",
        "device":args.device,
        "max_depth":4,
        "eta":0.045,
        "min_child_weight":24,
        "subsample":0.90,
        "colsample_bytree":0.86,
        "lambda":12.0,
        "alpha":0.10,
        "gamma":0.0,
        "seed":args.seed,
        "nthread":args.threads,
        "verbosity":0,
    }
    try:
        return xgb.train(params,d,num_boost_round=args.rounds,verbose_eval=False)
    except xgb.core.XGBoostError:
        if args.device!="cuda": raise
        params["device"]="cpu"
        return xgb.train(params,d,num_boost_round=args.rounds,verbose_eval=False)

def predict_xgb(xgb,bst,X):
    return np.asarray(bst.predict(xgb.DMatrix(X)),np.float32)

def predictions_by_query(indices,base_rank,pools,feats,short,qhand,bst,xgb,k=TRAIN_K):
    X,_,_,owner,cand=build_examples(
        indices,base_rank,pools,feats,short,qhand,
        [set() for _ in range(len(base_rank))], # labels not needed for features
        np.ones(len(base_rank),np.float32),k
    )
    # build_examples above computes util using empty golds, but util is discarded.
    pred=predict_xgb(xgb,bst,X)
    out={int(q):[] for q in indices}
    for q,d,s in zip(owner,cand,pred):
        out[int(q)].append((int(d),float(s)))
    return out

def apply_rule(base_rank,predmap,indices,k,threshold):
    rank=base_rank.copy()
    changed=0
    for qi0 in indices:
        qi=int(qi0)
        vals=predmap.get(qi,[])[:k]
        if not vals: continue
        # Stable tie-break by doc id.
        d,s=max(vals,key=lambda x:(x[1],-x[0]))
        if s>threshold:
            rank[qi,4]=int(d)
            if int(d)!=int(base_rank[qi,4]):changed+=1
    return rank,changed

def fold_recall(rank,ids,qids,golds,docs):
    vals=[]
    for qi in ids:
        q=qids[int(qi)]
        g=set(golds[q]); p={docs[int(x)] for x in rank[int(qi),:5]}
        vals.append(len(g&p)/len(g))
    return float(np.mean(vals))

def oracle_prefix(base_rank,pools,k,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    out=base_rank.copy()
    for qi,q in enumerate(qids):
        g={d2i[d] for d in golds[q]}
        cur=int(base_rank[qi,4])
        cg=cur in g
        if cg: continue
        for d in ordered_challengers(qi,base_rank,pools,k):
            if int(d) in g:
                out[qi,4]=int(d);break
    return out

def select_rule(inner_pred,outer_train,inner_folds,base_rank,qids,golds,docs,args):
    # Candidate thresholds are prediction-only quantiles. Labels are used only
    # to evaluate candidates on inner held folds.
    candidates=[]
    base_fold={f:fold_recall(base_rank,idx,qids,golds,docs) for f,idx in inner_folds.items()}

    for k in K_GRID:
        best_scores=[]
        for qi in outer_train:
            vals=inner_pred.get(int(qi),[])[:k]
            best_scores.append(max((s for _,s in vals),default=-np.inf))
        arr=np.asarray([x for x in best_scores if np.isfinite(x)],np.float64)
        thresholds=[float(np.percentile(arr,q)) for q in QUANTILES]
        thresholds.append(float("inf")) # explicit KEEP-ALL fallback

        for th in sorted(set(thresholds)):
            r,changed=apply_rule(base_rank,inner_pred,outer_train,k,th)
            deltas=[]
            for f,idx in inner_folds.items():
                deltas.append(fold_recall(r,idx,qids,golds,docs)-base_fold[f])
            change_rate=changed/max(len(outer_train),1)
            mean=float(np.mean(deltas)); sd=float(np.std(deltas))
            positive=sum(x>0 for x in deltas)
            # Conservative selection score. Large unstable gains do not win.
            score=mean-0.50*sd
            valid=change_rate<=args.max_change_rate and positive>=2 and mean>0
            candidates.append({
                "k":int(k),"threshold":float(th),"mean_delta":mean,"sd_delta":sd,
                "positive_inner_folds":int(positive),"change_rate":float(change_rate),
                "selection_score":float(score),"valid":bool(valid),
                "fold_deltas":[float(x) for x in deltas],
            })

    valid=[x for x in candidates if x["valid"]]
    if not valid:
        return {"k":0,"threshold":float("inf"),"mean_delta":0.0,"sd_delta":0.0,
                "positive_inner_folds":0,"change_rate":0.0,"selection_score":0.0,
                "valid":True,"fold_deltas":[]},candidates
    valid.sort(key=lambda x:(x["selection_score"],x["mean_delta"],-x["change_rate"]),reverse=True)
    return valid[0],candidates

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=["cuda","cpu"],default="cpu")
    ap.add_argument("--rounds",type=int,default=320)
    ap.add_argument("--threads",type=int,default=10)
    ap.add_argument("--max-change-rate",type=float,default=.15)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit('Install first: pip install -U "xgboost>=2.1,<4"')

    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()

    print("[1/7] Load frozen teacher-free world",flush=True)
    no_teacher=ROOT/"results/stage07b/__NO_TEACHER_FOR_STAGE07H__.npz"
    qids,questions,golds,folds,stress,docs,short,sources,ds,es,ce,aq,lq,teacher=(
        f7.load_world(no_teacher)
    )
    if teacher is not None:raise RuntimeError("Stage07H must be teacher-free")
    base_rank=f7.rank_old(short,ds)
    bm=metrics(base_rank,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_EXPECT)>3e-6:
        raise RuntimeError(f"Stage07D parity failed {bm['overall']['recall_at_5']}")
    print(f"  Stage07D parity R={bm['overall']['recall_at_5']:.9f}",flush=True)

    print("[2/7] Wide pools + pair features",flush=True)
    pools,feats=f7.build_pools(short,sources,ds,es,ce)
    qhand=np.stack([qg.q_hand(questions[q]) for q in qids]).astype(np.float32)
    d2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{d2i[d] for d in golds[q]} for q in qids]
    gold_count=np.asarray([len(golds[q]) for q in qids],np.float32)
    fi=qidx_by_fold(qids,folds)

    print("[3/7] Prefix oracle ladder (diagnostic only; never used to tune outer fold)",flush=True)
    oracle={}
    for k in K_GRID:
        rr=oracle_prefix(base_rank,pools,k,qids,golds,docs)
        mm=metrics(rr,qids,golds,docs,folds,stress)
        oracle[str(k)]=mm["overall"]
        print(f"  K={k:2d} oracle R={mm['overall']['recall_at_5']:.9f}",flush=True)

    print("[4/7] Strict nested-CV utility rescue",flush=True)
    rescued=base_rank.copy()
    fold_meta={}
    all_pred=np.full((len(qids),TRAIN_K),np.nan,np.float32)

    for outer,fn in enumerate(folds):
        held=fi[fn]
        inner_names=[f for f in folds if f!=fn]
        outer_train=np.concatenate([fi[f] for f in inner_names])

        print(f"\n[{fn}] inner OOF utility predictions",flush=True)
        inner_pred={int(q):[] for q in outer_train}
        inner_fold_map={f:fi[f] for f in inner_names}

        for inner in inner_names:
            va=fi[inner]
            tr=np.concatenate([fi[f] for f in inner_names if f!=inner])
            Xtr,ytr,wtr,_,_=build_examples(
                tr,base_rank,pools,feats,short,qhand,gold_idx,gold_count,TRAIN_K
            )
            bst=train_xgb(xgb,Xtr,ytr,wtr,args)

            Xv,_,_,own,can=build_examples(
                va,base_rank,pools,feats,short,qhand,gold_idx,gold_count,TRAIN_K
            )
            pv=predict_xgb(xgb,bst,Xv)
            for q,d,s in zip(own,can,pv):
                inner_pred[int(q)].append((int(d),float(s)))
            print(f"  inner={inner} train_rows={len(Xtr):,} val_rows={len(Xv):,}",flush=True)

        selected,grid=select_rule(
            inner_pred,outer_train,inner_fold_map,base_rank,qids,golds,docs,args
        )
        print(
            f"[{fn}] SELECT K={selected['k']} th={selected['threshold']:.6g} "
            f"inner_delta={selected['mean_delta']:+.6f} sd={selected['sd_delta']:.6f} "
            f"pos={selected['positive_inner_folds']}/4 "
            f"change={100*selected['change_rate']:.2f}%",
            flush=True
        )

        if selected["k"]==0:
            fold_meta[fn]={
                "selected":selected,"held_changes":0,"grid":grid,
                "train_rows":0,"decision":"KEEP_ALL_NO_STABLE_INNER_GAIN",
            }
            continue

        # Final outer-train model.
        Xtr,ytr,wtr,_,_=build_examples(
            outer_train,base_rank,pools,feats,short,qhand,gold_idx,gold_count,TRAIN_K
        )
        bst=train_xgb(xgb,Xtr,ytr,wtr,args)
        Xh,_,_,own,can=build_examples(
            held,base_rank,pools,feats,short,qhand,gold_idx,gold_count,TRAIN_K
        )
        ph=predict_xgb(xgb,bst,Xh)
        held_pred={int(q):[] for q in held}
        for q,d,s in zip(own,can,ph):
            held_pred[int(q)].append((int(d),float(s)))

        tmp,changed=apply_rule(
            base_rank,held_pred,held,selected["k"],selected["threshold"]
        )
        rescued[held,4]=tmp[held,4]

        # persist top20 held utility predictions aligned to candidate order
        for qi in held:
            vals=held_pred[int(qi)]
            for j,(_,s) in enumerate(vals[:TRAIN_K]):
                all_pred[int(qi),j]=float(s)

        held_delta=fold_recall(rescued,held,qids,golds,docs)-fold_recall(base_rank,held,qids,golds,docs)
        fold_meta[fn]={
            "selected":selected,
            "held_changes":int(changed),
            "held_change_rate":float(changed/len(held)),
            "held_delta_recall":float(held_delta),
            "train_rows":int(len(Xtr)),
            "grid":grid,
        }
        print(
            f"[{fn}] HELD delta={held_delta:+.6f} changes={changed}/{len(held)} "
            f"({100*changed/len(held):.2f}%)",
            flush=True
        )

    print("[5/7] Official evaluation",flush=True)
    rm=metrics(rescued,qids,golds,docs,folds,stress)
    b=bm["overall"];r=rm["overall"]
    delta=r["recall_at_5"]-b["recall_at_5"]
    bq=perq(base_rank,qids,golds,docs);rq=perq(rescued,qids,golds,docs)
    wins=int((rq>bq).sum());losses=int((rq<bq).sum())
    fd={f:rm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}

    print("[6/7] Save artifacts",flush=True)
    np.save(CACHE/"rescued_oof_rank.i32.npy",rescued.astype(np.int32))
    np.save(CACHE/"held_utility_predictions_top20.f32.npy",all_pred)
    pos=sum(v>0 for v in fd.values())
    if r["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_CALIBRATED_UTILITY_RESCUE"

    report={
        "schema":"dsc2026.endgame.stage07h.calibrated_utility_rescue.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict outer 5-fold OOF with 4-way inner calibration; teacher-free",
        "new_foundation_model":False,
        "teacher_used":False,
        "target":"exact marginal official Recall utility of replacing current rank5",
        "baseline_stage07d":bm,
        "oracle_prefix_diagnostic":oracle,
        "rescued":rm,
        "effect":{
            "delta_recall":delta,
            "delta_single":r["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
            "delta_multi":r["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
            "wins":wins,"losses":losses,"fold_deltas":fd,
        },
        "fold_meta":fold_meta,
        "decision":decision,
        "wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[7/7] Result")
    print("="*120)
    print(f"BASE Stage07D R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    print(f"UTILITY-RESCUE R={r['recall_at_5']:.9f} single={r['single_gold_recall_at_5']:.9f} multi={r['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        fm=fold_meta[f]
        print(
            f"{f}: delta={fd[f]:+.9f} "
            f"K={fm['selected']['k']} "
            f"changes={fm.get('held_changes',0)} "
            f"inner={fm['selected']['mean_delta']:+.6f}"
        )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*120)

if __name__=="__main__":
    main()

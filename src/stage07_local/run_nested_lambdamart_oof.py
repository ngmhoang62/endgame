#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg

OUT=ROOT/"reports/stage07c_lambdamart"
CACHE=ROOT/"cache/stage07c_lambdamart"
CE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
DONE_PATH=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
SHORT=30
BASE_R=0.9425976255185238

CONFIGS=[
    dict(name="raw45_d3",feature_mode="raw45",max_depth=3,eta=.045,min_child_weight=16,
         subsample=.90,colsample_bytree=.90,reg_lambda=8.,reg_alpha=.05,gamma=0.),
    dict(name="localz90_d3",feature_mode="localz90",max_depth=3,eta=.040,min_child_weight=16,
         subsample=.90,colsample_bytree=.88,reg_lambda=10.,reg_alpha=.08,gamma=0.),
    dict(name="localz90_d5",feature_mode="localz90",max_depth=5,eta=.030,min_child_weight=24,
         subsample=.88,colsample_bytree=.82,reg_lambda=14.,reg_alpha=.10,gamma=.02),
]

def load_world():
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_PATH)
    done=np.load(DONE_PATH)
    if ce.shape!=(len(qids),SHORT) or int(done.sum())!=len(qids):
        raise RuntimeError("CE cache incomplete")
    Xflat,_,_,names=b1.build_source_ce_features(short,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1).astype(np.float32)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)
    return qids,golds,folds,stress,docs,short,X45,y

def local_z(X):
    mu=X.mean(1,keepdims=True); sd=X.std(1,keepdims=True)
    sd=np.where(sd<1e-6,1.,sd)
    return ((X-mu)/sd).astype(np.float32)

def feat(X45,mode):
    return X45 if mode=="raw45" else np.concatenate([X45,local_z(X45)],axis=2).astype(np.float32)

def fold_idx(qids,folds):
    q2i={q:i for i,q in enumerate(qids)}
    return {f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

def dmatrix(xgb,X,y,qidx,drop_zero=False):
    qidx=np.asarray(qidx,np.int32)
    if drop_zero:
        qidx=qidx[y[qidx].sum(1)>0]
    d=xgb.DMatrix(X[qidx].reshape(-1,X.shape[-1]),label=y[qidx].reshape(-1))
    d.set_group(np.full(len(qidx),SHORT,dtype=np.uint32))
    return d,qidx

def params(cfg,args):
    return {
        "objective":"rank:ndcg","eval_metric":"ndcg@5","tree_method":"hist","device":args.device,
        "max_depth":cfg["max_depth"],"eta":cfg["eta"],"min_child_weight":cfg["min_child_weight"],
        "subsample":cfg["subsample"],"colsample_bytree":cfg["colsample_bytree"],
        "lambda":cfg["reg_lambda"],"alpha":cfg["reg_alpha"],"gamma":cfg["gamma"],
        "lambdarank_pair_method":"topk","lambdarank_num_pair_per_sample":10,
        "lambdarank_normalization":True,"seed":args.seed,"nthread":args.threads,"verbosity":0,
    }

def train_inner(xgb,X,y,tr,va,cfg,args):
    dtr,_=dmatrix(xgb,X,y,tr,True); dva,va2=dmatrix(xgb,X,y,va,False)
    p=params(cfg,args)
    try:
        bst=xgb.train(p,dtr,num_boost_round=args.rounds,evals=[(dva,"valid")],
                      early_stopping_rounds=args.early_stopping,verbose_eval=False)
    except xgb.core.XGBoostError:
        if args.device!="cuda": raise
        p["device"]="cpu"
        bst=xgb.train(p,dtr,num_boost_round=args.rounds,evals=[(dva,"valid")],
                      early_stopping_rounds=args.early_stopping,verbose_eval=False)
    bi=int(getattr(bst,"best_iteration",args.rounds-1))+1
    pred=bst.predict(dva,iteration_range=(0,bi)).reshape(len(va2),SHORT)
    return pred,bi,va2

def fit_final(xgb,X,y,tr,cfg,nround,args):
    dtr,_=dmatrix(xgb,X,y,tr,True); p=params(cfg,args)
    try:
        return xgb.train(p,dtr,num_boost_round=nround,verbose_eval=False)
    except xgb.core.XGBoostError:
        if args.device!="cuda": raise
        p["device"]="cpu"
        return xgb.train(p,dtr,num_boost_round=nround,verbose_eval=False)

def score_recall(score,short,y,qidx):
    vals=[]
    for qi in qidx:
        order=np.lexsort((short[qi],-score[qi]))
        denom=float(y[qi].sum())
        vals.append(0. if denom<=0 else float(y[qi,order[:5]].sum())/denom)
    return float(np.mean(vals))

def rank_from_score(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        o=np.lexsort((short[i],-score[i])); out[i]=short[i,o]
    return out

def perq(rank,qids,golds,docs):
    z=[]
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        z.append(len(g&p)/len(g))
    return np.asarray(z,float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--rounds",type=int,default=700)
    ap.add_argument("--early-stopping",type=int,default=55)
    ap.add_argument("--threads",type=int,default=10)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit('Install first: pip install -U "xgboost>=2.1,<4"')

    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    qids,golds,folds,stress,docs,short,X45,y=load_world()
    fi=fold_idx(qids,folds); fns=list(folds)

    print(f"[xgb] {xgb.__version__} device={args.device}",flush=True)
    print("[1/4] CE-LR parity",flush=True)
    base_rank=qg.current_lr_oof(X45,short,y,qids,folds)
    bm=qg.metrics_from_rank(base_rank,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_R)>3e-6:
        raise RuntimeError(f"baseline drift {bm['overall']['recall_at_5']}")

    score=np.full((len(qids),SHORT),np.nan,np.float32); meta={}
    print("[2/4] strict nested CV",flush=True)
    for outer in fns:
        held=fi[outer]; train_f=[f for f in fns if f!=outer]
        summaries=[]
        for cfg in CONFIGS:
            X=feat(X45,cfg["feature_mode"]); vals=[]; rounds=[]
            for inner in train_f:
                va=fi[inner]; tr=np.concatenate([fi[f] for f in train_f if f!=inner])
                pred,bi,va2=train_inner(xgb,X,y,tr,va,cfg,args)
                tmp=np.zeros((len(qids),SHORT),np.float32); tmp[va2]=pred
                r=score_recall(tmp,short,y,va2)
                vals.append(r); rounds.append(bi)
                print(f"[{outer}] {cfg['name']} inner={inner} R={r:.6f} rounds={bi}",flush=True)
            summaries.append((float(np.mean(vals)),float(np.std(vals)),cfg,vals,rounds))
        summaries.sort(key=lambda x:(x[0],-x[1]),reverse=True)
        mean,sd,cfg,vals,rounds=summaries[0]
        nround=max(20,int(round(np.median(rounds))))
        X=feat(X45,cfg["feature_mode"]); tr=np.concatenate([fi[f] for f in train_f])
        bst=fit_final(xgb,X,y,tr,cfg,nround,args)
        dhe,he2=dmatrix(xgb,X,y,held,False)
        pred=bst.predict(dhe,iteration_range=(0,nround)).reshape(len(he2),SHORT)
        score[he2]=pred
        tmp=np.zeros_like(score); tmp[he2]=pred
        held_r=score_recall(tmp,short,y,he2)
        meta[outer]={
            "selected_config":cfg["name"],"selected_rounds":nround,
            "inner_mean_recall_at5":mean,"inner_sd":sd,
            "outer_shortlist_recall_at5":held_r,
            "all_configs":{
                s[2]["name"]:{"mean":s[0],"sd":s[1],"fold_recalls":s[3],"rounds":s[4]}
                for s in summaries
            }
        }
        print(f"[{outer}] SELECT {cfg['name']} rounds={nround} heldR={held_r:.6f}",flush=True)

    if not np.isfinite(score).all(): raise RuntimeError("OOF incomplete")

    print("[3/4] official eval",flush=True)
    rank=rank_from_score(short,score)
    lm=qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)
    b=bm["overall"]; m=lm["overall"]; delta=m["recall_at_5"]-b["recall_at_5"]
    bq=perq(base_rank,qids,golds,docs); mq=perq(rank,qids,golds,docs)
    fd={f:lm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in fns}
    wins=int((mq>bq).sum()); losses=int((mq<bq).sum())

    np.save(CACHE/"lambdamart_oof_scores.f32.npy",score)
    np.save(CACHE/"lambdamart_oof_rank.i32.npy",rank.astype(np.int32))
    pos=sum(v>0 for v in fd.values())
    if m["recall_at_5"]>=.960: dec="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4: dec="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3: dec="KEEP_AS_COMPLEMENT"
    else: dec="KILL_LAMBDAMART"

    rep={
        "schema":"dsc2026.endgame.stage07c.nested_lambdamart.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict outer 5-fold OOF; hyperparameters selected only inside outer-train folds",
        "hypothesis":"nonlinear groupwise ranking over frozen 45D retrieval+CE evidence",
        "new_foundation_model":False,
        "configs":CONFIGS,
        "baseline_ce_lr":bm,"lambdamart":lm,
        "effect":{"delta_recall":delta,
                  "delta_single":m["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
                  "delta_multi":m["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
                  "wins":wins,"losses":losses,"fold_deltas":fd},
        "fold_meta":meta,
        "selected_config_counts":dict(Counter(x["selected_config"] for x in meta.values())),
        "decision":dec,"wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"OOF_REPORT.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("[4/4] result")
    print("="*110)
    print(f"BASE       R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    print(f"LAMBDAMART R={m['recall_at_5']:.9f} single={m['single_gold_recall_at_5']:.9f} multi={m['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in fns:
        print(f"{f}: delta={fd[f]:+.9f} cfg={meta[f]['selected_config']}")
    print("DECISION:",dec)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*110)

if __name__=="__main__":
    main()

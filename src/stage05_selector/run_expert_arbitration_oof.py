#!/usr/bin/env python
"""Stage 05C — Whole-list expert arbitration OOF tournament.

Why this exists
---------------
Stage05B showed:
  CE-LR OOF                  ~= .94260
  query-conditioned doc QGate ~= .94285
  10-arm best-list oracle     ~= .96779
  all-arm cap-5 oracle        ~= .96906

Therefore test the hypothesis at the *correct level*:
predict which whole retrieval/ranking expert is trustworthy for each query.

Arms:
  rankonly, ce_lr, blend015, ce_only,
  ait_atomic, ait_coarse1024,
  lal_coarse1024, lal_atomic, lal_b4, bm25

Router features are label-free:
- 1024D AIT query embedding -> fold-clean PCA;
- query structural/legal cues;
- per-arm list confidence;
- pairwise top5/top10 overlaps;
- cross-source support of each arm's top5;
- CE support of each arm's top5;
- retrieval-score shape for native source arms.

Targets are per-arm Recall@5 utilities, seen only on the training folds.

Models:
- fold-clean MultiOutput ExtraTrees utility regressor;
- routed best arm;
- confidence-weighted RRF over predicted top-2 arms;
- confidence-weighted RRF over all arms.

Promotion gate: frozen 5-fold OOF Recall@5 >= 0.960.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as b2

EVAL=ROOT/"data/evaluation_v2"
B4=ROOT/"cache/stage02b4_vi_screen"
CACHE=ROOT/"cache/stage05c_expert_arbitration"
OUT=ROOT/"reports/stage05c_expert_arbitration"

QEMB=B4/"aiteamvn_v1/query_embeddings.f32.npy"
CE_CACHE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
CE_DONE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"

PROMOTE=0.960
TOP5=5
TOP10=10
ARM_NAMES=[
    "rankonly","ce_lr","blend015","ce_only",
    "ait_atomic","ait_coarse1024",
    "lal_coarse1024","lal_atomic","lal_b4","bm25",
]
SOURCE_TO_ARM={
    "ait_atomic":"ait_atomic",
    "ait_coarse1024":"ait_coarse1024",
    "lal_coarse1024":"lal_coarse1024",
    "lal_atomic":"lal_atomic",
    "lal_b4":"lal_b4",
    "bm25":"bm25",
}


def metrics(rank,qids,golds,docs,folds,stress):
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    return {
        "overall":official_metrics(preds,golds,qids),
        "per_fold":{f:official_metrics(preds,golds,ids) for f,ids in folds.items()},
        "stress":{
            n:official_metrics(preds,golds,[q for q in ids if q in preds])
            for n,ids in stress.items() if any(q in preds for q in ids)
        },
    }


def recall_matrix(arms,qids,golds,docs):
    Y=np.empty((len(qids),len(ARM_NAMES)),dtype=np.float32)
    for ai,a in enumerate(ARM_NAMES):
        r=arms[a]
        for qi,q in enumerate(qids):
            p={docs[int(x)] for x in r[qi,:5]}
            g=set(golds[q])
            Y[qi,ai]=len(p&g)/len(g)
    return Y


def make_arms(shortlist,ce,sources,ce_lr):
    rankonly=shortlist.copy()
    ce_only=b1.rank_by_scores(shortlist,ce)

    zr=b1.zrows(np.broadcast_to(
        -np.arange(1,31,dtype=np.float32)[None,:],shortlist.shape
    ))
    zc=b1.zrows(ce)
    sc=.85*zr+.15*zc
    blend=np.empty_like(shortlist)
    for i in range(len(shortlist)):
        blend[i]=shortlist[i,np.lexsort((shortlist[i],-sc[i]))]

    arms={
        "rankonly":rankonly,
        "ce_lr":ce_lr,
        "blend015":blend,
        "ce_only":ce_only,
    }
    for s,a in SOURCE_TO_ARM.items():
        arms[a]=np.asarray(sources[s][0],dtype=np.int32)
    return arms


def rank_maps(arr,qi,depth=50):
    return {int(d):r+1 for r,d in enumerate(arr[qi,:depth])}


def list_diag_features(arms,sources,ce,shortlist):
    """Return [N, A, D] per-arm diagnostics + pairwise overlaps flattened separately."""
    n=len(shortlist); A=len(ARM_NAMES)

    # CE lookup on the shared top30.
    ce_z=b1.zrows(ce)
    ce_rank=np.empty_like(shortlist)
    for qi in range(n):
        o=np.lexsort((shortlist[qi],-ce[qi]))
        inv=np.empty(30,np.int32); inv[o]=np.arange(1,31)
        ce_rank[qi]=inv

    # Native source rank maps are expensive as dicts but manageable for 6991.
    D=18
    F=np.zeros((n,A,D),dtype=np.float32)
    O5=np.zeros((n,A,A),dtype=np.float32)
    O10=np.zeros((n,A,A),dtype=np.float32)

    for qi in range(n):
        src_maps={s:rank_maps(sources[s][0],qi,50) for s in b1.SOURCE_NAMES}
        src_top5={s:set(map(int,sources[s][0][qi,:5])) for s in b1.SOURCE_NAMES}
        src_top10={s:set(map(int,sources[s][0][qi,:10])) for s in b1.SOURCE_NAMES}
        src_top20={s:set(map(int,sources[s][0][qi,:20])) for s in b1.SOURCE_NAMES}

        sl_pos={int(d):j for j,d in enumerate(shortlist[qi])}

        top5_sets=[]
        top10_sets=[]
        for a in ARM_NAMES:
            top5_sets.append(set(map(int,arms[a][qi,:5])))
            top10_sets.append(set(map(int,arms[a][qi,:10])))

        for ai,a in enumerate(ARM_NAMES):
            a5=list(map(int,arms[a][qi,:5]))
            supports5=[]; supports10=[]; supports20=[]
            mean_rr=[]; cevals=[]; cezvals=[]; ceranks=[]
            for d in a5:
                supports5.append(sum(d in src_top5[s] for s in b1.SOURCE_NAMES))
                supports10.append(sum(d in src_top10[s] for s in b1.SOURCE_NAMES))
                supports20.append(sum(d in src_top20[s] for s in b1.SOURCE_NAMES))
                rr=[1/(10+src_maps[s][d]) for s in b1.SOURCE_NAMES if d in src_maps[s]]
                mean_rr.append(float(np.mean(rr)) if rr else 0.)
                if d in sl_pos:
                    j=sl_pos[d]
                    cevals.append(float(ce[qi,j]))
                    cezvals.append(float(ce_z[qi,j]))
                    ceranks.append(float(ce_rank[qi,j]))
                else:
                    cevals.append(-20.); cezvals.append(-4.); ceranks.append(35.)

            # Native confidence if this is a raw source arm.
            native_gap15=native_gap56=native_std10=native_z1=0.
            if a in SOURCE_TO_ARM.values():
                s=a
                scr=np.asarray(sources[s][1][qi],dtype=np.float64)
                sd=float(scr.std())
                if sd<1e-8: sd=1.
                native_gap15=float((scr[0]-scr[4])/sd)
                native_gap56=float((scr[4]-scr[5])/sd)
                native_std10=float(np.std(scr[:10])/sd)
                native_z1=float((scr[0]-scr.mean())/sd)

            F[qi,ai]=[
                float(np.mean(supports5)),
                float(np.min(supports5)),
                float(np.max(supports5)),
                float(np.mean(supports10)),
                float(np.mean(supports20)),
                float(np.mean(mean_rr)),
                float(np.mean(cevals)),
                float(np.max(cevals)),
                float(np.mean(cezvals)),
                float(np.mean(ceranks))/30.,
                float(len(set(a5))),
                float(len(set(a5)&set(map(int,shortlist[qi,:5]))))/5.,
                float(len(set(a5)&set(map(int,shortlist[qi,:10]))))/5.,
                native_gap15,native_gap56,native_std10,native_z1,
                float(sum(x>=4 for x in supports10))/5.,
            ]

        for i in range(A):
            for j in range(A):
                u=top5_sets[i]|top5_sets[j]
                O5[qi,i,j]=len(top5_sets[i]&top5_sets[j])/max(len(u),1)
                u=top10_sets[i]|top10_sets[j]
                O10[qi,i,j]=len(top10_sets[i]&top10_sets[j])/max(len(u),1)

        if (qi+1)%1000==0 or qi+1==n:
            print(f"[diag] {qi+1}/{n}",flush=True)

    # Query-level flattened diagnostic tensor.
    Xdiag=np.concatenate([
        F.reshape(n,-1),
        O5.reshape(n,-1),
        O10.reshape(n,-1),
    ],axis=1)
    return F,O5,O10,Xdiag


def qhand(questions,qids):
    return np.stack([b2.q_hand(questions[q]) for q in qids],axis=0).astype(np.float32)


def route_oof(Xdiag,Qemb,Qhand,Y,arms,qids,folds,n_estimators,min_leaf):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.preprocessing import StandardScaler

    q2i={q:i for i,q in enumerate(qids)}
    pred=np.zeros_like(Y,dtype=np.float32)
    fold_meta={}

    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        hs=set(map(int,held))
        train=np.asarray([i for i in range(len(qids)) if i not in hs],dtype=np.int32)

        # Fold-clean PCA on query embedding.
        pca=PCA(n_components=64,random_state=276)
        qtr=pca.fit_transform(Qemb[train]).astype(np.float32)
        qhe=pca.transform(Qemb[held]).astype(np.float32)

        scaler=StandardScaler()
        dtr=scaler.fit_transform(Xdiag[train]).astype(np.float32)
        dhe=scaler.transform(Xdiag[held]).astype(np.float32)

        Xtr=np.concatenate([qtr,Qhand[train],dtr],axis=1)
        Xhe=np.concatenate([qhe,Qhand[held],dhe],axis=1)

        model=ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf,
            max_features=.65,
            random_state=276,
            n_jobs=-1,
        )
        model.fit(Xtr,Y[train])
        pred[held]=model.predict(Xhe).astype(np.float32)
        fold_meta[fn]={
            "train_queries":len(train),
            "held_queries":len(held),
            "pca_explained_variance":float(pca.explained_variance_ratio_.sum()),
        }
        print(f"[router/{fn}] done",flush=True)

    # Route whole list.
    route=np.empty((len(qids),100),dtype=np.int32)
    chosen=np.argmax(pred,axis=1)
    for qi,ai in enumerate(chosen):
        a=ARM_NAMES[int(ai)]
        arr=arms[a][qi]
        k=min(len(arr),100)
        route[qi,:k]=arr[:k]
        if k<100:
            # Fill with rank-only unique docs to keep a stable array shape.
            seen=set(map(int,route[qi,:k]))
            pos=k
            for d in arms["rankonly"][qi]:
                d=int(d)
                if d not in seen:
                    route[qi,pos]=d; seen.add(d); pos+=1
                    if pos==100: break
            if pos<100:
                route[qi,pos:]=route[qi,0]
    return pred,route,chosen,fold_meta


def weighted_rrf(pred,arms,mode="top2"):
    n=len(pred)
    out=np.empty((n,100),dtype=np.int32)
    for qi in range(n):
        util=np.asarray(pred[qi],dtype=np.float64)
        # Stable utility softmax; predicted recall lives roughly [0,1].
        z=np.exp((util-util.max())*12.0)
        w=z/z.sum()
        if mode=="top2":
            keep=np.argsort(-w)[:2]
        else:
            keep=np.arange(len(ARM_NAMES))

        score={}
        for ai in keep:
            a=ARM_NAMES[int(ai)]
            # top30 sufficient for learned ranking arms, raw retrievers expose 100.
            depth=min(50,arms[a].shape[1])
            for r,d0 in enumerate(arms[a][qi,:depth],1):
                d=int(d0)
                score[d]=score.get(d,0.0)+float(w[ai])/(20.0+r)
        ranked=sorted(score,key=lambda d:(-score[d],d))
        if len(ranked)<100:
            seen=set(ranked)
            for d0 in arms["rankonly"][qi]:
                d=int(d0)
                if d not in seen:
                    ranked.append(d); seen.add(d)
                if len(ranked)==100: break
        if len(ranked)<100:
            ranked += [ranked[0]]*(100-len(ranked))
        out[qi]=np.asarray(ranked[:100],dtype=np.int32)
    return out


def oracle_report(Y,golds):
    best=Y.max(axis=1)
    chosen=Y.argmax(axis=1)
    wins=Counter(ARM_NAMES[int(x)] for x in chosen)
    single=np.asarray([len(golds[q])==1 for q in qids_global])
    multi=~single
    return {
        "overall":float(best.mean()),
        "single_gold":float(best[single].mean()),
        "multi_gold":float(best[multi].mean()),
        "argmax_tiebroken_arm_counts":dict(wins),
        "queries_with_arm_utility_disagreement":int(np.sum(Y.max(axis=1)>Y.min(axis=1))),
    }


def paired_gain(base,other,qids,golds,docs):
    b=[]; o=[]
    for qi,q in enumerate(qids):
        g=set(golds[q])
        b.append(len({docs[int(x)] for x in base[qi,:5]}&g)/len(g))
        o.append(len({docs[int(x)] for x in other[qi,:5]}&g)/len(g))
    b=np.asarray(b); o=np.asarray(o)
    return {
        "mean_delta":float(np.mean(o-b)),
        "wins":int(np.sum(o>b)),
        "losses":int(np.sum(o<b)),
        "ties":int(np.sum(o==b)),
    }


def main():
    global qids_global
    ap=argparse.ArgumentParser()
    ap.add_argument("--trees",type=int,default=500)
    ap.add_argument("--min-leaf",type=int,default=8)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()

    print("[1/7] Load frozen world + current selector",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    qids_global=qids
    shortlist,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_CACHE); done=np.load(CE_DONE)
    if ce.shape!=(len(qids),30) or int(done.sum())!=len(qids):
        raise RuntimeError("CE cache incomplete")

    Xflat,cq,cand,names=b1.build_source_ce_features(shortlist,ce,sources)
    X45=Xflat.reshape(len(qids),30,-1)
    ydoc=b2.labels_for(shortlist,qids,golds,docs)
    ce_lr=b2.current_lr_oof(X45,shortlist,ydoc,qids,folds)
    base_metrics=metrics(ce_lr,qids,golds,docs,folds,stress)
    if abs(base_metrics["overall"]["recall_at_5"]-0.9425976255185238)>2e-6:
        raise RuntimeError("CE-LR parity failed")

    print("[2/7] Build 10 whole-list arms + utility targets",flush=True)
    arms=make_arms(shortlist,ce,sources,ce_lr)
    if list(arms)!=ARM_NAMES:
        raise RuntimeError(f"arm order drift {list(arms)}")
    Y=recall_matrix(arms,qids,golds,docs)
    oracle=oracle_report(Y,golds)
    print("  oracle",oracle["overall"],"single",oracle["single_gold"],"multi",oracle["multi_gold"])

    print("[3/7] Build label-free arm diagnostics",flush=True)
    F,O5,O10,Xdiag=list_diag_features(arms,sources,ce,shortlist)
    np.save(CACHE/"arm_diagnostics.f32.npy",F)
    np.save(CACHE/"arm_overlap5.f32.npy",O5)
    np.save(CACHE/"arm_overlap10.f32.npy",O10)

    print("[4/7] Load query semantics",flush=True)
    Qemb=np.load(QEMB)
    if Qemb.shape!=(len(qids),1024):
        raise RuntimeError(f"query embedding shape drift {Qemb.shape}")
    Qhand=qhand(questions,qids)

    print("[5/7] Fold-clean ExtraTrees utility router",flush=True)
    pred,routed,chosen,fold_meta=route_oof(
        Xdiag,Qemb,Qhand,Y,arms,qids,folds,args.trees,args.min_leaf
    )
    np.save(CACHE/"oof_predicted_arm_utilities.f32.npy",pred)
    np.save(CACHE/"oof_chosen_arm.u1.npy",chosen.astype(np.uint8))
    route_metrics=metrics(routed,qids,golds,docs,folds,stress)

    print("[6/7] Conditional fusion variants",flush=True)
    top2=weighted_rrf(pred,arms,"top2")
    allw=weighted_rrf(pred,arms,"all")
    top2_metrics=metrics(top2,qids,golds,docs,folds,stress)
    allw_metrics=metrics(allw,qids,golds,docs,folds,stress)

    methods={
        "ce_lr_baseline":base_metrics,
        "expert_router_top1":route_metrics,
        "expert_router_top2_rrf":top2_metrics,
        "expert_router_all_weighted_rrf":allw_metrics,
    }
    best=max(methods,key=lambda k:methods[k]["overall"]["recall_at_5"])
    best_r=methods[best]["overall"]["recall_at_5"]

    print("[7/7] Report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage05c.expert_arbitration.v1",
        "status":"COMPLETE",
        "arms":ARM_NAMES,
        "oracle":oracle,
        "methods":methods,
        "paired_vs_ce_lr":{
            "expert_router_top1":paired_gain(ce_lr,routed,qids,golds,docs),
            "expert_router_top2_rrf":paired_gain(ce_lr,top2,qids,golds,docs),
            "expert_router_all_weighted_rrf":paired_gain(ce_lr,allw,qids,golds,docs),
        },
        "router":{
            "model":"MultiOutput ExtraTreesRegressor",
            "trees":args.trees,
            "min_samples_leaf":args.min_leaf,
            "query_pca_dim":64,
            "arm_diag_dim":int(F.shape[-1]),
            "query_diag_flat_dim":int(Xdiag.shape[-1]),
            "fold_meta":fold_meta,
        },
        "promotion":{
            "gate":PROMOTE,
            "best_method":best,
            "best_recall_at5":best_r,
            "decision":"PROMOTE" if best_r>=PROMOTE else "DO_NOT_PROMOTE",
        },
        "wall_seconds":time.perf_counter()-start,
    }
    (OUT/"EXPERT_ARBITRATION.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    lines=[
        "# Stage 05C — Whole-List Expert Arbitration","",
        f"- Best-list oracle: **{oracle['overall']:.6f}**",
        f"- Oracle single-gold: **{oracle['single_gold']:.6f}**",
        f"- Oracle multi-gold: **{oracle['multi_gold']:.6f}**","",
        "| Method | R@5 | P@5 | Single | Multi |",
        "|---|---:|---:|---:|---:|",
    ]
    for n,m in methods.items():
        o=m["overall"]
        lines.append(
            f"| {n} | {o['recall_at_5']:.6f} | {o['precision_at_5']:.6f} | "
            f"{o['single_gold_recall_at_5']:.6f} | {o['multi_gold_recall_at_5']:.6f} |"
        )
    lines += ["",f"**Decision: {report['promotion']['decision']}** "
              f"(gate {PROMOTE:.3f}; best={best} {best_r:.6f})",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*100)
    print(f"ORACLE overall={oracle['overall']:.9f} single={oracle['single_gold']:.9f} multi={oracle['multi_gold']:.9f}")
    for n,m in methods.items():
        o=m["overall"]
        p=report["paired_vs_ce_lr"].get(n)
        extra="" if p is None else f" W/L={p['wins']}/{p['losses']}"
        print(f"{n:32s} R={o['recall_at_5']:.9f} P={o['precision_at_5']:.9f}{extra}")
    print("DECISION:",report["promotion"]["decision"])
    print("REPORT:",OUT/"REPORT.md")
    print("="*100)


if __name__=="__main__":
    from collections import Counter
    main()

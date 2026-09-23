#!/usr/bin/env python
"""
Stage07I — Strict Co-Gold Graph Slot-5 Rescue.

Hypothesis
----------
Stage07D/07E show the largest remaining weakness on multi-gold queries. The
retrieval/semantic features in Stage07F-H could not tell which rare wide-pool
challenger should replace rank5.

A different signal is available in the training labels themselves: documents
that repeatedly occur together as gold answers define a document co-relevance
graph. For a held query, if one of Stage07D's top anchors has historically
co-occurred with a wide-pool challenger, that edge may identify a companion
gold document.

This is NOT query-memory / nearest-query matching:
- no held query text is compared to training queries;
- the graph contains only document-node co-occurrence statistics;
- every outer-held fold is scored with a graph built without its labels;
- hyperparameters are chosen by inner CV with inner graphs rebuilt without the
  corresponding validation fold.

No teacher. No self-distillation. No new foundation model.
Stage07D top-4 is frozen; only slot5 may change.

Run:
  python src/stage07_local/run_cogold_graph_slot5_oof.py
"""
from __future__ import annotations
import argparse, itertools, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_wide_slot5_rescue_oof as f7

OUT=ROOT/"reports/stage07i_cogold_graph_rescue"
CACHE=ROOT/"cache/stage07i_cogold_graph_rescue"
BASE_EXPECT=0.9458518094693177

ANCHOR_GRID=(1,2,4)
K_GRID=(5,10,20,40)
MIN_SUPPORT_GRID=(1,2,3)
SCORE_MODES=("cond","jaccard","hybrid")
QUANTILES=(0.0,50.0,70.0,80.0,90.0,95.0)

def metrics(rank,qids,golds,docs,folds,stress):
    return qg.metrics_from_rank(rank,qids,golds,docs,folds,stress)

def perq(rank,qids,golds,docs):
    out=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]); p={docs[int(x)] for x in rank[i,:5]}
        out[i]=len(g&p)/len(g)
    return out

def fold_indices(qids,folds):
    q2i={q:i for i,q in enumerate(qids)}
    return {f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

def build_graph(train_idx,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    freq=Counter()
    pair=Counter()
    multi_q=0
    for qi0 in train_idx:
        qi=int(qi0); gs=sorted({d2i[d] for d in golds[qids[qi]]})
        for d in gs: freq[d]+=1
        if len(gs)>1:
            multi_q+=1
            for a,b in itertools.combinations(gs,2):
                pair[(a,b)]+=1
    nbr=defaultdict(dict)
    for (a,b),c in pair.items():
        nbr[a][b]=c
        nbr[b][a]=c
    return freq,nbr,{"queries":int(len(train_idx)),"multi_queries":int(multi_q),
                     "nodes":int(len(freq)),"edges":int(len(pair))}

def edge_stats(a,c,freq,nbr,ntrain):
    pc=int(nbr.get(int(a),{}).get(int(c),0))
    if pc<=0:return 0,0.,0.,0.
    fa=int(freq.get(int(a),0)); fc=int(freq.get(int(c),0))
    cond=pc/max(fa,1)
    jac=pc/max(fa+fc-pc,1)
    # Smoothed positive PMI-like association. Clamp at 0 because negative
    # association is not useful for proposing companion gold documents.
    pmi=math.log(((pc+0.5)*(ntrain+1.0))/((fa+0.5)*(fc+0.5)))
    return pc,float(cond),float(jac),float(max(pmi,0.0))

def challengers(qi,base_rank,pools,k):
    top4=set(map(int,base_rank[qi,:4]));cur=int(base_rank[qi,4])
    out=[]
    for d0 in pools[qi]:
        d=int(d0)
        if d in top4 or d==cur:continue
        out.append(d)
        if len(out)>=k:break
    return out

def graph_score(qi,c,base_rank,pools,freq,nbr,ntrain,
                anchors_n,min_support,mode):
    anchors=list(map(int,base_rank[qi,:anchors_n]))
    best=-1.0; best_support=0; linked=0
    for ar, a in enumerate(anchors):
        pc,cond,jac,pmi=edge_stats(a,int(c),freq,nbr,ntrain)
        if pc<min_support:continue
        linked+=1
        aw=1.0/(1.0+0.25*ar)
        if mode=="cond":
            s=cond
        elif mode=="jaccard":
            s=jac
        elif mode=="hybrid":
            # association + reliability; bounded-ish and support-aware.
            s=math.sqrt(max(cond,0.)*max(jac,0.))*math.log1p(pc)*(1.0+0.10*pmi)
        else: raise KeyError(mode)
        best=max(best,aw*s)
        best_support=max(best_support,pc)
    return float(best),int(best_support),int(linked)

def best_candidate(qi,base_rank,pools,freq,nbr,ntrain,cfg):
    vals=[]
    for r,c in enumerate(challengers(qi,base_rank,pools,cfg["k"])):
        s,sup,linked=graph_score(
            qi,c,base_rank,pools,freq,nbr,ntrain,
            cfg["anchors"],cfg["min_support"],cfg["mode"]
        )
        if s<0:continue
        # Tiny deterministic retrieval-rank tie-break only.
        vals.append((s, sup, linked, -r, -int(c), int(c)))
    if not vals:return None,-1.0,0,0
    x=max(vals)
    return x[-1],float(x[0]),int(x[1]),int(x[2])

def apply_cfg(indices,base_rank,pools,freq,nbr,ntrain,cfg,threshold):
    out=base_rank.copy();changes=0; scores=[]
    for qi0 in indices:
        qi=int(qi0)
        d,s,sup,linked=best_candidate(qi,base_rank,pools,freq,nbr,ntrain,cfg)
        scores.append(s)
        if d is not None and s>threshold:
            out[qi,4]=int(d);changes+=1
    return out,int(changes),np.asarray(scores,np.float64)

def fold_recall(rank,idx,qids,golds,docs):
    vals=[]
    for qi0 in idx:
        qi=int(qi0);q=qids[qi]
        g=set(golds[q]);p={docs[int(x)] for x in rank[qi,:5]}
        vals.append(len(g&p)/len(g))
    return float(np.mean(vals))

def strict_graph_oracle(base_rank,pools,fi,qids,golds,docs):
    """Diagnostic: each outer fold uses graph built only on other 4 folds."""
    d2i={d:i for i,d in enumerate(docs)}
    out=base_rank.copy()
    coverage={"multi_queries":0,"gold_challenger_graph_linked":0}
    for fn,held in fi.items():
        train=np.concatenate([x for f,x in fi.items() if f!=fn])
        freq,nbr,_=build_graph(train,qids,golds,docs)
        for qi0 in held:
            qi=int(qi0);g={d2i[d] for d in golds[qids[qi]]}
            if len(g)>1:coverage["multi_queries"]+=1
            cur=int(base_rank[qi,4])
            if cur in g:continue
            found=None
            for c in challengers(qi,base_rank,pools,40):
                if c not in g:continue
                linked=False
                for a in map(int,base_rank[qi,:4]):
                    if nbr.get(a,{}).get(int(c),0)>0:
                        linked=True;break
                if linked:
                    found=int(c);break
            if found is not None:
                out[qi,4]=found
                coverage["gold_challenger_graph_linked"]+=1
    return out,coverage

def inner_select(outer_fn,fi,base_rank,pools,qids,golds,docs,args):
    inner_names=[f for f in fi if f!=outer_fn]
    outer_train=np.concatenate([fi[f] for f in inner_names])

    # Prebuild leakage-clean graph for each inner held fold.
    graphs={}
    base_r={}
    for inner in inner_names:
        tr=np.concatenate([fi[f] for f in inner_names if f!=inner])
        graphs[inner]=build_graph(tr,qids,golds,docs)
        base_r[inner]=fold_recall(base_rank,fi[inner],qids,golds,docs)

    configs=[]
    for anchors in ANCHOR_GRID:
        for k in K_GRID:
            for ms in MIN_SUPPORT_GRID:
                for mode in SCORE_MODES:
                    configs.append({"anchors":anchors,"k":k,"min_support":ms,"mode":mode})

    candidates=[]
    for cfg in configs:
        # First collect graph-score distribution across leakage-clean inner folds.
        fold_scores={}
        for inner in inner_names:
            freq,nbr,meta=graphs[inner]
            _,_,sc=apply_cfg(
                fi[inner],base_rank,pools,freq,nbr,meta["queries"],cfg,float("inf")
            )
            fold_scores[inner]=sc
        finite=np.concatenate([x[x>=0] for x in fold_scores.values()
                               if np.any(x>=0)]) if any(np.any(x>=0) for x in fold_scores.values()) else np.asarray([])
        if len(finite)==0:continue
        thresholds=sorted(set(float(np.percentile(finite,q)) for q in QUANTILES))
        # inf = explicit keep-all is unnecessary here because configs must have >0 gain.
        for th in thresholds:
            deltas=[];changes=0;total=0
            for inner in inner_names:
                freq,nbr,meta=graphs[inner]
                r,ch,_=apply_cfg(
                    fi[inner],base_rank,pools,freq,nbr,meta["queries"],cfg,th
                )
                deltas.append(fold_recall(r,fi[inner],qids,golds,docs)-base_r[inner])
                changes+=ch;total+=len(fi[inner])
            mean=float(np.mean(deltas));sd=float(np.std(deltas))
            pos=sum(x>0 for x in deltas)
            cr=changes/max(total,1)
            valid=(mean>0 and pos>=args.min_positive_inner
                   and cr<=args.max_change_rate)
            candidates.append({
                **cfg,"threshold":float(th),"mean_delta":mean,"sd_delta":sd,
                "positive_inner_folds":int(pos),"change_rate":float(cr),
                "selection_score":float(mean-0.5*sd),"fold_deltas":[float(x) for x in deltas],
                "valid":bool(valid),
            })

    valid=[x for x in candidates if x["valid"]]
    if not valid:
        return None,candidates
    valid.sort(key=lambda x:(x["selection_score"],x["mean_delta"],-x["change_rate"]),reverse=True)
    return valid[0],candidates

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-change-rate",type=float,default=.12)
    ap.add_argument("--min-positive-inner",type=int,default=3)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()

    print("[1/7] Load Stage07D + frozen wide pool",flush=True)
    no_teacher=ROOT/"results/stage07b/__NO_TEACHER_FOR_STAGE07I__.npz"
    qids,questions,golds,folds,stress,docs,short,sources,ds,es,ce,aq,lq,teacher=(
        f7.load_world(no_teacher)
    )
    if teacher is not None:raise RuntimeError("Stage07I must be teacher-free")
    base_rank=f7.rank_old(short,ds)
    bm=metrics(base_rank,qids,golds,docs,folds,stress)
    if abs(bm["overall"]["recall_at_5"]-BASE_EXPECT)>3e-6:
        raise RuntimeError(f"Stage07D parity failed {bm['overall']['recall_at_5']}")
    pools,feats=f7.build_pools(short,sources,ds,es,ce)
    fi=fold_indices(qids,folds)
    print(f"  Stage07D R={bm['overall']['recall_at_5']:.9f}",flush=True)

    print("[2/7] Co-gold graph global descriptive statistics",flush=True)
    all_idx=np.arange(len(qids),dtype=np.int32)
    gf,gn,gm=build_graph(all_idx,qids,golds,docs)
    supports=[c for a in gn.values() for c in a.values()]
    # undirected edges appear twice in gn
    unique_support=[c for (a,b),c in Counter({
        tuple(sorted((a,b))):c for a,nb in gn.items() for b,c in nb.items()
    }).items()]
    desc={
        **gm,
        "max_doc_frequency":int(max(gf.values()) if gf else 0),
        "edge_support_ge2":int(sum(c>=2 for c in unique_support)),
        "edge_support_ge3":int(sum(c>=3 for c in unique_support)),
        "max_edge_support":int(max(unique_support) if unique_support else 0),
    }
    print(" ",desc,flush=True)

    print("[3/7] Strict graph-linked slot5 oracle",flush=True)
    grank,gcov=strict_graph_oracle(base_rank,pools,fi,qids,golds,docs)
    gmo=metrics(grank,qids,golds,docs,folds,stress)
    print(
        f"  GRAPH-LINK ORACLE R={gmo['overall']['recall_at_5']:.9f} "
        f"single={gmo['overall']['single_gold_recall_at_5']:.9f} "
        f"multi={gmo['overall']['multi_gold_recall_at_5']:.9f} "
        f"linked_gold_rescues={gcov['gold_challenger_graph_linked']}",
        flush=True
    )

    print("[4/7] Strict nested graph rescue",flush=True)
    rescued=base_rank.copy()
    fold_meta={}
    for outer_fn in folds:
        sel,grid=inner_select(
            outer_fn,fi,base_rank,pools,qids,golds,docs,args
        )
        held=fi[outer_fn]
        train=np.concatenate([x for f,x in fi.items() if f!=outer_fn])
        freq,nbr,gmeta=build_graph(train,qids,golds,docs)

        if sel is None:
            fold_meta[outer_fn]={
                "selected":None,"held_changes":0,"held_delta_recall":0.0,
                "graph":gmeta,"decision":"KEEP_ALL_NO_STABLE_INNER_GAIN",
                "valid_config_count":0,
            }
            print(f"[{outer_fn}] KEEP ALL — no stable inner graph rule",flush=True)
            continue

        cfg={k:sel[k] for k in ("anchors","k","min_support","mode")}
        r,ch,_=apply_cfg(
            held,base_rank,pools,freq,nbr,gmeta["queries"],cfg,sel["threshold"]
        )
        rescued[held,4]=r[held,4]
        hd=fold_recall(rescued,held,qids,golds,docs)-fold_recall(base_rank,held,qids,golds,docs)
        fold_meta[outer_fn]={
            "selected":sel,"held_changes":int(ch),"held_change_rate":float(ch/len(held)),
            "held_delta_recall":float(hd),"graph":gmeta,
            "valid_config_count":int(sum(x["valid"] for x in grid)),
        }
        print(
            f"[{outer_fn}] SELECT anchors={sel['anchors']} K={sel['k']} "
            f"support>={sel['min_support']} mode={sel['mode']} th={sel['threshold']:.6g} "
            f"inner={sel['mean_delta']:+.6f} pos={sel['positive_inner_folds']}/4 "
            f"change={100*sel['change_rate']:.2f}% | "
            f"HELD={hd:+.6f} changes={ch}",
            flush=True
        )

    print("[5/7] Official evaluation",flush=True)
    rm=metrics(rescued,qids,golds,docs,folds,stress)
    b=bm["overall"];r=rm["overall"]
    delta=r["recall_at_5"]-b["recall_at_5"]
    bq=perq(base_rank,qids,golds,docs);rq=perq(rescued,qids,golds,docs)
    wins=int((rq>bq).sum());losses=int((rq<bq).sum())
    fd={f:rm["per_fold"][f]["recall_at_5"]-bm["per_fold"][f]["recall_at_5"] for f in folds}

    print("[6/7] Save",flush=True)
    np.save(CACHE/"rescued_oof_rank.i32.npy",rescued.astype(np.int32))
    pos=sum(v>0 for v in fd.values())
    if r["recall_at_5"]>=.960:
        decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.005 and pos>=4:
        decision="STRONG_GAIN_PROMOTE"
    elif delta>=.002 and pos>=3:
        decision="KEEP_AS_COMPLEMENT"
    else:
        decision="KILL_COGOLD_GRAPH_RESCUE"

    report={
        "schema":"dsc2026.endgame.stage07i.cogold_graph_slot5.v1",
        "status":"COMPLETE",
        "claim_boundary":"strict outer 5-fold with leakage-clean inner graph rebuilding; teacher-free",
        "teacher_used":False,"new_foundation_model":False,
        "hypothesis":"training-label document co-relevance graph can identify companion gold documents for Stage07D anchors",
        "graph_descriptive":desc,
        "baseline_stage07d":bm,
        "strict_graph_link_oracle":{"metrics":gmo,"coverage":gcov},
        "rescued":rm,
        "effect":{"delta_recall":delta,
                  "delta_single":r["single_gold_recall_at_5"]-b["single_gold_recall_at_5"],
                  "delta_multi":r["multi_gold_recall_at_5"]-b["multi_gold_recall_at_5"],
                  "wins":wins,"losses":losses,"fold_deltas":fd},
        "fold_meta":fold_meta,
        "decision":decision,
        "wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[7/7] Result")
    print("="*124)
    print(f"BASE Stage07D   R={b['recall_at_5']:.9f} single={b['single_gold_recall_at_5']:.9f} multi={b['multi_gold_recall_at_5']:.9f}")
    go=gmo["overall"]
    print(f"GRAPH-LINK ORCL R={go['recall_at_5']:.9f} single={go['single_gold_recall_at_5']:.9f} multi={go['multi_gold_recall_at_5']:.9f}")
    print(f"GRAPH-RESCUE    R={r['recall_at_5']:.9f} single={r['single_gold_recall_at_5']:.9f} multi={r['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        m=fold_meta[f]
        if m["selected"] is None:
            print(f"{f}: delta={fd[f]:+.9f} KEEP_ALL")
        else:
            s=m["selected"]
            print(
                f"{f}: delta={fd[f]:+.9f} changes={m['held_changes']} "
                f"A={s['anchors']} K={s['k']} support={s['min_support']} mode={s['mode']}"
            )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*124)

if __name__=="__main__":
    main()

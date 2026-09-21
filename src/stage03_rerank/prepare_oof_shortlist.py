#!/usr/bin/env python
"""Stage 03B0 — prepare fold-clean OOF reranker shortlist.

No neural model is loaded. Rebuilds the best Stage03A-v2 all_cached6 selector,
but preserves rankings out to top50 so reranker depth can be chosen from measured
candidate ceiling rather than guessed.

Outputs:
- cache/stage03b_reranker/oof_rank_selector_top50_idx.npy
- cache/stage03b_reranker/fulltrain_rank_selector.joblib
- reports/stage03b0_reranker_shortlist/SHORTLIST_REPORT.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RAW=ROOT/"data/official_v1"
EVAL=ROOT/"data/evaluation_v2"
C1=ROOT/"cache/stage02c1_aiteam_representation"
C2=ROOT/"cache/stage02c2_lal_representation"
B4=ROOT/"cache/stage02b4_vi_screen"
A2=ROOT/"cache/stage02a_parent_anchor"
CACHE=ROOT/"cache/stage03b_reranker"
OUT=ROOT/"reports/stage03b0_reranker_shortlist"

DEPTH=50
KEEP=50
CVAL=.15
KS=(5,10,15,20,30,40,50)

SOURCES={
 "ait_atomic":(C1/"atomic_split_2048/parent_top100_idx.npy",C1/"atomic_split_2048/parent_top100_scores.npy"),
 "ait_coarse1024":(C1/"coarse_pack_1024/parent_top100_idx.npy",C1/"coarse_pack_1024/parent_top100_scores.npy"),
 "lal_coarse1024":(C2/"lal_coarse_pack_1024/parent_top100_idx.npy",C2/"lal_coarse_pack_1024/parent_top100_scores.npy"),
 "lal_atomic":(C2/"lal_atomic_split_2048/parent_top100_idx.npy",C2/"lal_atomic_split_2048/parent_top100_scores.npy"),
 "lal_b4":(B4/"vnlegal_lal/parent_top100_idx.npy",B4/"vnlegal_lal/parent_top100_scores.npy"),
 "bm25":(A2/"bm25_idx.npy",A2/"bm25_scores.npy"),
}
SN=list(SOURCES)

def rj(p): return json.loads(p.read_text(encoding="utf-8"))

def load():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    docs=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): docs.append(str(json.loads(line)["id"]))
    if len(qids)!=6991 or len(docs)!=8512: raise RuntimeError("frozen population drift")
    src={}
    for n,(ip,sp) in SOURCES.items():
        i=np.load(ip,mmap_mode="r"); s=np.load(sp,mmap_mode="r")
        if i.shape!=(6991,100) or s.shape!=(6991,100): raise RuntimeError(f"{n} shape drift")
        src[n]=(i,s)
    return qids,golds,folds,docs,src

def names():
    x=[]
    for s in SN: x += [s+"__present",s+"__rr10",s+"__rank50",s+"__z",s+"__gapz"]
    return x+["source_count","best_rr10","mean_rr10","rrf60","min_rank50","mean_rank50",
             "count_top5","count_top10","count_top20"]

FEATURE_NAMES=names()

def build(qids,golds,docs,src):
    d2i={d:i for i,d in enumerate(docs)}
    gold=[{d2i[d] for d in golds[q]} for q in qids]
    pools=[]; total=0
    for qi in range(len(qids)):
        u=set()
        for s in SN: u.update(int(x) for x in src[s][0][qi,:DEPTH])
        a=np.asarray(sorted(u),dtype=np.int32); pools.append(a); total+=len(a)
    X=np.empty((total,len(FEATURE_NAMES)),np.float32)
    y=np.empty(total,np.uint8); cand=np.empty(total,np.int32); cq=np.empty(total,np.int32)
    off=np.empty(len(qids)+1,np.int64)
    pos=0
    for qi,pool in enumerate(pools):
        off[qi]=pos
        rm={}; zm={}; gm={}
        for s in SN:
            idx,scr=src[s]; ids=idx[qi,:DEPTH]; s100=np.asarray(scr[qi,:],np.float64)
            mu=float(s100.mean()); sd=float(s100.std())
            if not np.isfinite(sd) or sd<1e-8: sd=1.
            top=float(s100[0])
            rm[s]={int(d):r+1 for r,d in enumerate(ids)}
            zm[s]={int(d):float((float(scr[qi,r])-mu)/sd) for r,d in enumerate(ids)}
            gm[s]={int(d):float((top-float(scr[qi,r]))/sd) for r,d in enumerate(ids)}
        for dd in pool:
            d=int(dd); f=[]; rrs=[]; ranks=[]; c5=c10=c20=0; rrf=0.
            for s in SN:
                r=rm[s].get(d)
                if r is None: f += [0.,0.,1.2,-3.,4.]
                else:
                    rr=1/(10+r); f += [1.,rr,r/DEPTH,zm[s][d],gm[s][d]]
                    rrs.append(rr); ranks.append(r); rrf += 1/(60+r)
                    c5+=r<=5; c10+=r<=10; c20+=r<=20
            f += [len(rrs),max(rrs) if rrs else 0.,float(np.mean(rrs)) if rrs else 0.,
                  rrf,min(ranks)/DEPTH if ranks else 1.2,
                  float(np.mean(ranks))/DEPTH if ranks else 1.2,c5,c10,c20]
            X[pos]=f; y[pos]=d in gold[qi]; cand[pos]=d; cq[pos]=qi; pos+=1
        if (qi+1)%1000==0 or qi+1==len(qids):
            print(f"[features] {qi+1}/{len(qids)} rows={pos}",flush=True)
    off[-1]=pos
    return dict(X=X,y=y,cand=cand,cq=cq,off=off)

def fit_model(X,y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=Pipeline([("scale",StandardScaler()),
                ("lr",LogisticRegression(C=CVAL,class_weight="balanced",solver="lbfgs",
                                         max_iter=300,tol=1e-5))])
    m.fit(X,y); return m

def rank_queries(model,data,qis):
    out=np.empty((len(qis),KEEP),dtype=np.int32)
    for oi,qi in enumerate(qis):
        a,b=int(data["off"][qi]),int(data["off"][qi+1])
        p=model.predict_proba(data["X"][a:b])[:,1]; di=data["cand"][a:b]
        order=np.lexsort((di,-p))
        if len(order)<KEEP: raise RuntimeError(f"q={qi}: pool {len(order)} < {KEEP}")
        out[oi]=di[order[:KEEP]]
    return out

def eval_ceiling(arr,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    gold=[{d2i[d] for d in golds[q]} for q in qids]
    out={}
    for k in KS:
        vals=[]; full=0; hits=tot=0
        for i,g in enumerate(gold):
            h=len(set(map(int,arr[i,:k]))&g)
            vals.append(h/len(g)); full+=h==len(g); hits+=h; tot+=len(g)
        out[str(k)]={"macro_oracle_recall":float(np.mean(vals)),
                     "full_gold_coverage_rate":full/len(qids),
                     "gold_occurrence_coverage":hits/tot}
    return out

def main():
    import joblib
    CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    qids,golds,folds,docs,src=load(); data=build(qids,golds,docs,src); q2i={q:i for i,q in enumerate(qids)}
    oof=np.empty((len(qids),KEEP),dtype=np.int32)
    seen=np.zeros(len(qids),bool)
    fold_metrics={}
    for fn,ids in folds.items():
        held=np.asarray([q2i[q] for q in ids],np.int32); hs=set(map(int,held))
        mask=np.asarray([int(q) not in hs for q in data["cq"]],bool)
        print(f"[{fn}] fit rows={mask.sum()} positives={data['y'][mask].sum()}",flush=True)
        m=fit_model(data["X"][mask],data["y"][mask]); rr=rank_queries(m,data,held)
        oof[held]=rr; seen[held]=True
    if not seen.all(): raise RuntimeError("OOF coverage incomplete")
    ceiling=eval_ceiling(oof,qids,golds,docs)
    np.save(CACHE/"oof_rank_selector_top50_idx.npy",oof)

    full=fit_model(data["X"],data["y"])
    joblib.dump({"model":full,"feature_names":FEATURE_NAMES,"sources":SN,"depth":DEPTH},
                CACHE/"fulltrain_rank_selector.joblib")

    report={"schema_version":"dsc2026.endgame.stage03b0.shortlist.v1","status":"PASS",
            "selector":"Stage03A-v2 all_cached6 LogisticRegression",
            "shortlist_depths":ceiling,
            "oof_top50_path":str((CACHE/"oof_rank_selector_top50_idx.npy").resolve()),
            "fulltrain_model_path":str((CACHE/"fulltrain_rank_selector.joblib").resolve())}
    (OUT/"SHORTLIST_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 03B0 — Reranker shortlist ceiling","",
           "| K | Macro oracle recall | Full coverage | Gold occurrence coverage |",
           "|---:|---:|---:|---:|"]
    for k in map(str,KS):
        r=ceiling[k]
        lines.append(f"| {k} | {r['macro_oracle_recall']:.6f} | {r['full_gold_coverage_rate']:.4%} | {r['gold_occurrence_coverage']:.4%} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","ceiling":ceiling,"out":str(OUT)},indent=2))

if __name__=="__main__":
    main()

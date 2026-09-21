#!/usr/bin/env python
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics

RAW=ROOT/"data/official_v1"
EVAL=ROOT/"data/evaluation_v2"
C1=ROOT/"cache/stage02c1_aiteam_representation"
C2=ROOT/"cache/stage02c2_lal_representation"
B4=ROOT/"cache/stage02b4_vi_screen"
A2=ROOT/"cache/stage02a_parent_anchor"
OUT=ROOT/"reports/stage03a_fast_oof_fusion_v2"

DEPTH=50
TOPK=5
CVAL=0.15

SRC_PATHS={
    "ait_atomic":(
        C1/"atomic_split_2048/parent_top100_idx.npy",
        C1/"atomic_split_2048/parent_top100_scores.npy"),
    "ait_coarse1024":(
        C1/"coarse_pack_1024/parent_top100_idx.npy",
        C1/"coarse_pack_1024/parent_top100_scores.npy"),
    "lal_coarse1024":(
        C2/"lal_coarse_pack_1024/parent_top100_idx.npy",
        C2/"lal_coarse_pack_1024/parent_top100_scores.npy"),
    "lal_atomic":(
        C2/"lal_atomic_split_2048/parent_top100_idx.npy",
        C2/"lal_atomic_split_2048/parent_top100_scores.npy"),
    "lal_b4":(
        B4/"vnlegal_lal/parent_top100_idx.npy",
        B4/"vnlegal_lal/parent_top100_scores.npy"),
    "bm25":(
        A2/"bm25_idx.npy",
        A2/"bm25_scores.npy"),
}

CONFIGS={
    "compact3":("ait_coarse1024","lal_coarse1024","bm25"),
    "dual_ait4":("ait_atomic","ait_coarse1024","lal_coarse1024","bm25"),
    "dual_ait_dual_lal5":(
        "ait_atomic","ait_coarse1024","lal_coarse1024","lal_atomic","bm25"),
    "all_cached6":(
        "ait_atomic","ait_coarse1024","lal_coarse1024","lal_atomic","lal_b4","bm25"),
}

def rj(p): return json.loads(p.read_text(encoding="utf-8"))

def load_eval():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}
    if len(qids)!=6991: raise RuntimeError(f"Expected 6991 qids, got {len(qids)}")
    return qids,golds,folds,stress

def load_docs():
    docs=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): docs.append(str(json.loads(line)["id"]))
    if len(docs)!=8512: raise RuntimeError(f"Expected 8512 docs, got {len(docs)}")
    return docs

def load_sources(nq):
    out={}
    for name,(ip,sp) in SRC_PATHS.items():
        if not ip.exists() or not sp.exists(): raise FileNotFoundError(f"{name}: {ip} / {sp}")
        idx=np.load(ip,mmap_mode="r")
        scr=np.load(sp,mmap_mode="r")
        if idx.shape!=(nq,100) or scr.shape!=(nq,100):
            raise RuntimeError(f"{name} shape drift idx={idx.shape} score={scr.shape}")
        out[name]=(idx,scr)
    return out

def fnames(sources):
    out=[]
    for s in sources:
        out += [f"{s}__present",f"{s}__rr10",f"{s}__rank_norm50",
                f"{s}__score_z",f"{s}__top_gap_z"]
    out += ["source_count","best_rr10","mean_rr10_present","rrf60",
            "min_rank_norm50","mean_rank_norm50_present",
            "count_top5","count_top10","count_top20"]
    return out

def build_data(qids,golds,docs,src,sources):
    doc2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{doc2i[d] for d in golds[q]} for q in qids]
    names=fnames(sources)

    pools=[]; total=0; oracle=[]
    for qi in range(len(qids)):
        u=set()
        for s in sources: u.update(int(x) for x in src[s][0][qi,:DEPTH])
        arr=np.asarray(sorted(u),dtype=np.int32)
        pools.append(arr); total+=len(arr)
        oracle.append(len(set(map(int,arr))&gold_idx[qi])/len(gold_idx[qi]))

    X=np.empty((total,len(names)),dtype=np.float32)
    y=np.empty(total,dtype=np.uint8)
    cand=np.empty(total,dtype=np.int32)
    cq=np.empty(total,dtype=np.int32)
    offsets=np.empty(len(qids)+1,dtype=np.int64)

    pos=0
    for qi,pool in enumerate(pools):
        offsets[qi]=pos
        rank_maps={}; z_maps={}; gap_maps={}
        for s in sources:
            idx,scr=src[s]
            ids=np.asarray(idx[qi,:DEPTH],dtype=np.int32)
            s100=np.asarray(scr[qi,:],dtype=np.float64)
            mu=float(s100.mean()); sd=float(s100.std())
            if not np.isfinite(sd) or sd<1e-8: sd=1.0
            top=float(s100[0])
            rank_maps[s]={int(d):r+1 for r,d in enumerate(ids)}
            z_maps[s]={int(d):float((float(scr[qi,r])-mu)/sd) for r,d in enumerate(ids)}
            gap_maps[s]={int(d):float((top-float(scr[qi,r]))/sd) for r,d in enumerate(ids)}

        g=gold_idx[qi]
        for d0 in pool:
            d=int(d0); feat=[]; rrs=[]; ranks=[]; c5=c10=c20=0; rrf=0.0
            for s in sources:
                r=rank_maps[s].get(d)
                if r is None:
                    feat += [0.0,0.0,1.2,-3.0,4.0]
                else:
                    rr=1.0/(10.0+r)
                    feat += [1.0,rr,r/DEPTH,z_maps[s][d],gap_maps[s][d]]
                    rrs.append(rr); ranks.append(r); rrf+=1.0/(60.0+r)
                    c5+=r<=5; c10+=r<=10; c20+=r<=20
            feat += [
                float(len(rrs)),
                max(rrs) if rrs else 0.0,
                float(np.mean(rrs)) if rrs else 0.0,
                rrf,
                min(ranks)/DEPTH if ranks else 1.2,
                float(np.mean(ranks))/DEPTH if ranks else 1.2,
                float(c5),float(c10),float(c20),
            ]
            X[pos]=feat; y[pos]=d in g; cand[pos]=d; cq[pos]=qi; pos+=1

        if (qi+1)%1000==0 or qi+1==len(qids):
            print(f"[features] {qi+1}/{len(qids)} rows={pos}",flush=True)

    offsets[len(qids)]=pos
    return dict(X=X,y=y,cand=cand,cq=cq,offsets=offsets,names=names,
                mean_pool=total/len(qids),oracle_r50=float(np.mean(oracle)))

def predict(model,data,qis,docs):
    out={}
    for qi in qis:
        a,b=int(data["offsets"][qi]),int(data["offsets"][qi+1])
        p=model.predict_proba(data["X"][a:b])[:,1]
        di=data["cand"][a:b]
        order=np.lexsort((di,-p))
        out[int(qi)]=[docs[int(x)] for x in di[order[:TOPK]]]
    return out

def run_config(name,sources,qids,golds,folds,stress,docs,src):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n=== {name}: {sources} ===",flush=True)
    data=build_data(qids,golds,docs,src,sources)
    q2i={q:i for i,q in enumerate(qids)}
    oof={}; fold_metrics={}; t0=time.perf_counter()

    for fn,held_qids in folds.items():
        held=np.asarray([q2i[q] for q in held_qids],dtype=np.int32)
        held_set=set(map(int,held))
        mask=np.asarray([int(q) not in held_set for q in data["cq"]],dtype=bool)
        model=Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(
                C=CVAL,class_weight="balanced",solver="lbfgs",
                max_iter=300,tol=1e-5)),
        ])
        print(f"[{name}/{fn}] fit rows={int(mask.sum())} pos={int(data['y'][mask].sum())}",flush=True)
        model.fit(data["X"][mask],data["y"][mask])
        pred=predict(model,data,held,docs)
        oof.update(pred)
        pmap={qids[i]:pred[int(i)] for i in held}
        fold_metrics[fn]=official_metrics(pmap,golds,[qids[i] for i in held])

    full={qids[i]:oof[i] for i in range(len(qids))}
    overall=official_metrics(full,golds,qids)
    stress_out={}
    for sn,ids in stress.items():
        use=[q for q in ids if q in full]
        if use: stress_out[sn]=official_metrics(full,golds,use)

    return {
        "config":name,"sources":list(sources),
        "candidate_oracle_r50":data["oracle_r50"],
        "mean_unique_pool":data["mean_pool"],
        "oof_overall":overall,"oof_per_fold":fold_metrics,
        "oof_stress":stress_out,
        "wall_seconds":time.perf_counter()-t0,
    }

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qids,golds,folds,stress=load_eval()
    docs=load_docs(); src=load_sources(len(qids))

    results={}
    for name,sources in CONFIGS.items():
        results[name]=run_config(name,sources,qids,golds,folds,stress,docs,src)
        m=results[name]["oof_overall"]
        print(json.dumps({
            "config":name,
            "oracle_r50":results[name]["candidate_oracle_r50"],
            "oof_r5":m["recall_at_5"],
            "oof_p5":m["precision_at_5"],
            "single_r5":m["single_gold_recall_at_5"],
            "multi_r5":m["multi_gold_recall_at_5"],
        },indent=2),flush=True)

    order=sorted(results,key=lambda n:results[n]["oof_overall"]["recall_at_5"],reverse=True)
    summary={
        "schema_version":"dsc2026.endgame.stage03a_fast_oof_fusion.v2",
        "status":"COMPLETE","configs":results,
        "report_order_by_oof_r5":order,
        "selector":{"type":"StandardScaler+LogisticRegression",
                    "C":CVAL,"class_weight":"balanced","fold_clean":True},
        "claim_boundary":"cached rank/score OOF selector only",
    }
    (OUT/"FAST_FUSION_OOF.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    rows=[]
    for n in order:
        r=results[n]; m=r["oof_overall"]
        rows.append({
            "config":n,"sources":"|".join(r["sources"]),
            "oracle_r50":r["candidate_oracle_r50"],
            "mean_pool":r["mean_unique_pool"],
            "oof_r5":m["recall_at_5"],"oof_p5":m["precision_at_5"],
            "single_r5":m["single_gold_recall_at_5"],
            "multi_r5":m["multi_gold_recall_at_5"],
            "seconds":r["wall_seconds"],
        })
    with (OUT/"CONFIG_COMPARISON.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines=["# Stage 03A v2 — Fast OOF Fusion","",
           "| Config | Oracle@50 | Pool | OOF R@5 | OOF P@5 | Single | Multi |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['oracle_r50']:.6f} | {r['mean_pool']:.1f} | "
                     f"{r['oof_r5']:.6f} | {r['oof_p5']:.6f} | "
                     f"{r['single_r5']:.6f} | {r['multi_r5']:.6f} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    print(json.dumps({"status":"COMPLETE","order":order,
                      "best_oof_r5":rows[0]["oof_r5"],"out":str(OUT)},indent=2))

if __name__=="__main__":
    main()

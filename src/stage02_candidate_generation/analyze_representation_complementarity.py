#!/usr/bin/env python
from __future__ import annotations
import csv, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics

RAW=ROOT/"data/official_v1"
EVAL=ROOT/"data/evaluation_v2"
C1=ROOT/"cache/stage02c1_aiteam_representation"
B4=ROOT/"cache/stage02b4_vi_screen"
A2=ROOT/"cache/stage02a_parent_anchor"
OUT=ROOT/"reports/stage02c1b_representation_complementarity"
TOPK=100
KS=(5,10,20,50,100)

PATHS={
 "aiteam_b4":B4/"aiteamvn_v1/parent_top100_idx.npy",
 "aiteam_atomic_split_2048":C1/"atomic_split_2048/parent_top100_idx.npy",
 "aiteam_coarse_1024":C1/"coarse_pack_1024/parent_top100_idx.npy",
 "aiteam_coarse_2048":C1/"coarse_pack_2048/parent_top100_idx.npy",
 "lal_b4":B4/"vnlegal_lal/parent_top100_idx.npy",
 "bm25":A2/"bm25_idx.npy",
}

def rj(p): return json.loads(p.read_text(encoding="utf-8"))

def load_eval():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    if len(qids)!=6991: raise RuntimeError("6991 eval contract drift")
    return qids,golds,folds,stress

def load_docids():
    x=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): x.append(str(json.loads(line)["id"]))
    if len(x)!=8512: raise RuntimeError("8512 doc contract drift")
    return x

def load_ranks(qids,docids):
    out={}
    for n,p in PATHS.items():
        if not p.exists(): raise FileNotFoundError(p)
        a=np.load(p,mmap_mode="r")
        if a.shape!=(len(qids),100): raise RuntimeError(f"{n}: {a.shape}")
        out[n]=[[docids[int(j)] for j in a[i]] for i in range(len(qids))]
    return out

def curve(rank,qids,golds):
    out={}
    for k in KS:
        vals=[]; full=0
        for i,q in enumerate(qids):
            g=set(golds[q]); h=len(set(rank[i][:k])&g)
            vals.append(h/len(g)); full+=h==len(g)
        out[str(k)]={"macro_recall":float(np.mean(vals)),
                     "full_gold_coverage_rate":full/len(qids)}
    return out

def union_oracle(ranks,names,depth,qids,golds):
    vals=[]; full=0; sizes=[]; hit=tot=0
    for i,q in enumerate(qids):
        p=set()
        for n in names: p.update(ranks[n][i][:depth])
        g=set(golds[q]); h=len(p&g)
        vals.append(h/len(g)); full+=h==len(g); sizes.append(len(p)); hit+=h; tot+=len(g)
    return {"sources":list(names),"depth":depth,
            "macro_oracle_recall":float(np.mean(vals)),
            "full_gold_coverage_rate":full/len(qids),
            "mean_unique_pool_size":float(np.mean(sizes)),
            "gold_occurrence_coverage":hit/tot}

def pairwise(ranks,a,b,qids,golds):
    ar=br=ag=bg=0; jac=[]; uni=[]
    for i,q in enumerate(qids):
        A=set(ranks[a][i]); B=set(ranks[b][i]); G=set(golds[q])
        ao=(A&G)-B; bo=(B&G)-A
        ar+=bool(ao); br+=bool(bo); ag+=len(ao); bg+=len(bo)
        jac.append(len(A&B)/len(A|B))
        uni.append(len((A|B)&G)/len(G))
    return {"source_a":a,"source_b":b,
            "a_rescue_queries":int(ar),"b_rescue_queries":int(br),
            "a_unique_gold_occurrences":int(ag),"b_unique_gold_occurrences":int(bg),
            "mean_top100_jaccard":float(np.mean(jac)),
            "union_oracle_r100":float(np.mean(uni))}

def rrf(ranks,names,k0=60):
    out=[]
    N=len(next(iter(ranks.values())))
    for i in range(N):
        s={}
        for n in names:
            for r,d in enumerate(ranks[n][i],1):
                s[d]=s.get(d,0.0)+1.0/(k0+r)
        out.append(sorted(s,key=lambda d:(-s[d],d))[:TOPK])
    return out

def per_fold_r5(rank,qids,golds,folds):
    preds={q:rank[i][:5] for i,q in enumerate(qids)}
    return {f:official_metrics(preds,golds,ids)["recall_at_5"] for f,ids in folds.items()}

def stress_r5(rank,qids,golds,stress):
    preds={q:rank[i][:5] for i,q in enumerate(qids)}
    present=set(qids); out={}
    for n,ids in stress.items():
        use=[q for q in ids if q in present]
        if use: out[n]=official_metrics(preds,golds,use)["recall_at_5"]
    return out

def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qids,golds,folds,stress=load_eval()
    docids=load_docids()
    ranks=load_ranks(qids,docids)

    views=("aiteam_b4","aiteam_atomic_split_2048","aiteam_coarse_1024","aiteam_coarse_2048")
    pairs=[pairwise(ranks,a,b,qids,golds) for a,b in combinations(views,2)]
    pairs.sort(key=lambda x:-x["union_oracle_r100"])

    combos=[
      ("baseline_core",("aiteam_b4","lal_b4","bm25")),
      ("coarse1024_core",("aiteam_coarse_1024","lal_b4","bm25")),
      ("atomic_core",("aiteam_atomic_split_2048","lal_b4","bm25")),
      ("dual_aiteam_core",("aiteam_atomic_split_2048","aiteam_coarse_1024","lal_b4","bm25")),
      ("dual_aiteam_no_lal",("aiteam_atomic_split_2048","aiteam_coarse_1024","bm25")),
      ("all_three_aiteam_core",("aiteam_atomic_split_2048","aiteam_coarse_1024","aiteam_coarse_2048","lal_b4","bm25")),
    ]

    oracle=[]
    for label,names in combos:
        for d in (20,50,100):
            x=union_oracle(ranks,names,d,qids,golds); x["label"]=label; oracle.append(x)
            print(f"[oracle] {label} d={d} R={x['macro_oracle_recall']:.6f} pool={x['mean_unique_pool_size']:.1f}",flush=True)

    rrf_rows=[]; details={}
    for label,names in combos:
        rr=rrf(ranks,names,60)
        m5=official_metrics({q:rr[i][:5] for i,q in enumerate(qids)},golds,qids)
        cv=curve(rr,qids,golds)
        row={"label":label,"sources":"|".join(names),
             "recall_at5":m5["recall_at_5"],"precision_at5":m5["precision_at_5"],
             "recall_at20":cv["20"]["macro_recall"],
             "recall_at100":cv["100"]["macro_recall"],
             "full_at100":cv["100"]["full_gold_coverage_rate"]}
        rrf_rows.append(row)
        details[label]={"summary":row,"curve":cv,
                        "per_fold_recall_at5":per_fold_r5(rr,qids,golds,folds),
                        "stress_recall_at5":stress_r5(rr,qids,golds,stress)}
        print(f"[rrf60] {label} R5={row['recall_at5']:.6f} R100={row['recall_at100']:.6f}",flush=True)

    base=("aiteam_atomic_split_2048","aiteam_coarse_1024","lal_b4","bm25")
    plus=base+("aiteam_coarse_2048",)
    marginal={}
    for d in (20,50,100):
        a=union_oracle(ranks,base,d,qids,golds)
        b=union_oracle(ranks,plus,d,qids,golds)
        marginal[str(d)]={"base":a,"plus_coarse2048":b,
                          "delta_macro_oracle_recall":b["macro_oracle_recall"]-a["macro_oracle_recall"],
                          "delta_mean_pool_size":b["mean_unique_pool_size"]-a["mean_unique_pool_size"]}

    write_csv(OUT/"AIT_VIEW_PAIRWISE.csv",pairs)
    write_csv(OUT/"UNION_ORACLE.csv",oracle)
    write_csv(OUT/"RRF60.csv",rrf_rows)

    summary={"schema_version":"dsc2026.endgame.stage02c1b_representation_complementarity.v1",
             "status":"COMPLETE",
             "claim_boundary":"cached complementarity + untuned fixed RRF60; no learned fusion",
             "aiteam_view_pairwise":pairs,
             "union_oracle":oracle,
             "rrf60":details,
             "coarse2048_marginal_after_dual_view_core":marginal}
    (OUT/"REPRESENTATION_COMPLEMENTARITY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    o100={r["label"]:r for r in oracle if r["depth"]==100}
    rmap={r["label"]:r for r in rrf_rows}
    lines=["# Stage 02C1b — Representation Complementarity","",
           "Cache-only analysis; no encoding.","",
           "| Configuration | Oracle R@100 | Mean pool | RRF60 R@5 | RRF60 R@100 |",
           "|---|---:|---:|---:|---:|"]
    for label,_ in combos:
        lines.append(f"| {label} | {o100[label]['macro_oracle_recall']:.6f} | "
                     f"{o100[label]['mean_unique_pool_size']:.1f} | "
                     f"{rmap[label]['recall_at5']:.6f} | {rmap[label]['recall_at100']:.6f} |")
    lines += ["","RRF60 is an untuned sanity check, not final fusion.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","out":str(OUT)},indent=2))

if __name__=="__main__":
    main()

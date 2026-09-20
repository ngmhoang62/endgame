#!/usr/bin/env python
from __future__ import annotations
import csv, json
from itertools import combinations
from pathlib import Path
from statistics import mean
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/"data/official_v1"
EVAL = ROOT/"data/evaluation_v2"
B4 = ROOT/"cache/stage02b4_vi_screen"
A2 = ROOT/"cache/stage02a_parent_anchor"
GEOM = ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"
OUT = ROOT/"reports/stage02b5_acquisition_oracle"

DEPTHS = (10,20,50,100)
CORE = ("aiteamvn_v1","vnlegal_lal","vietlegal_harrier","bm25")
DROPPED = ("aiteamvn_v2","vietlegal_e5","vietnamese_legal_embedding")
B4_PATHS = {
    "aiteamvn_v1": B4/"aiteamvn_v1/parent_top100_idx.npy",
    "vnlegal_lal": B4/"vnlegal_lal/parent_top100_idx.npy",
    "vietlegal_harrier": B4/"vietlegal_harrier/parent_top100_idx.npy",
    "aiteamvn_v2": B4/"aiteamvn_v2/parent_top100_idx.npy",
    "vietlegal_e5": B4/"vietlegal_e5/parent_top100_idx.npy",
    "vietnamese_legal_embedding": B4/"vietnamese_legal_embedding/parent_top100_idx.npy",
}

def rj(p): return json.loads(p.read_text(encoding="utf-8"))

def load_docs():
    geom_docs=[]; seen=set()
    with GEOM.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d=str(json.loads(line)["doc_id"])
            if d not in seen:
                seen.add(d); geom_docs.append(d)
    corpus_docs=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): corpus_docs.append(str(json.loads(line)["id"]))
    if geom_docs != corpus_docs:
        raise RuntimeError("Stage02B4 geometry doc order != Stage02A corpus doc order")
    if len(geom_docs) != 8512:
        raise RuntimeError(f"Expected 8512 docs, got {len(geom_docs)}")
    return geom_docs

def load_eval():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    if len(qids)!=6991: raise RuntimeError(f"Expected 6991 qids, got {len(qids)}")
    questions={q:str(train[q]["question"]) for q in qids}
    q2fold={str(q):fold for fold,ids in rj(EVAL/"folds_v2.json").items() for q in ids}
    return qids,golds,questions,q2fold

def load_rankings(docids,qids):
    ranks={}
    for name,path in B4_PATHS.items():
        if not path.exists(): raise FileNotFoundError(path)
        idx=np.load(path,mmap_mode="r")
        if idx.shape!=(len(qids),100): raise RuntimeError(f"{name}: {idx.shape}")
        ranks[name]=[[docids[int(j)] for j in idx[i]] for i in range(len(qids))]
    bp=A2/"bm25_idx.npy"
    if not bp.exists(): raise FileNotFoundError(bp)
    idx=np.load(bp,mmap_mode="r")
    if idx.shape!=(len(qids),100): raise RuntimeError(f"bm25: {idx.shape}")
    ranks["bm25"]=[[docids[int(j)] for j in idx[i]] for i in range(len(qids))]
    return ranks

def pool(ranks,sources,i,depth):
    p=set()
    for s in sources: p.update(ranks[s][i][:depth])
    return p

def eval_union(ranks,sources,depth,qids,golds):
    rs=[]; full=0; sizes=[]; hits=tot=0
    for i,q in enumerate(qids):
        p=pool(ranks,sources,i,depth); g=set(golds[q]); h=len(p&g)
        rs.append(h/len(g)); full += h==len(g); sizes.append(len(p)); hits+=h; tot+=len(g)
    return {
        "sources":list(sources),"per_source_depth":depth,
        "macro_oracle_recall":float(mean(rs)),
        "full_gold_coverage_rate":full/len(qids),
        "mean_unique_pool_size":float(mean(sizes)),
        "max_nominal_pool_size":depth*len(sources),
        "gold_occurrence_coverage":hits/tot,
        "hit_gold_occurrences":hits,"total_gold_occurrences":tot,
    }

def marginal(ranks,base,challenger,depth,qids,golds):
    rb=[]; re=[]; qr=fr=gr=0
    for i,q in enumerate(qids):
        b=pool(ranks,base,i,depth); e=b|set(ranks[challenger][i][:depth]); g=set(golds[q])
        hb=len(b&g); he=len(e&g); rb.append(hb/len(g)); re.append(he/len(g))
        if he>hb: qr+=1; gr+=he-hb
        if hb<len(g) and he==len(g): fr+=1
    return {
        "base_sources":list(base),"challenger":challenger,"per_source_depth":depth,
        "base_macro_recall":float(mean(rb)),"expanded_macro_recall":float(mean(re)),
        "delta_macro_recall":float(mean(re)-mean(rb)),
        "query_rescues":qr,"full_query_rescues":fr,"gold_occurrence_rescues":gr,
    }

def failures(ranks,sources,depth,qids,golds,questions,q2fold):
    out=[]
    for i,q in enumerate(qids):
        p=pool(ranks,sources,i,depth)
        missing=[d for d in golds[q] if d not in p]
        if not missing: continue
        per={}
        for d in missing:
            per[d]={}
            for s in ranks:
                try: per[d][s]=ranks[s][i].index(d)+1
                except ValueError: per[d][s]=None
        out.append({"qid":q,"fold":q2fold[q],"question":questions[q],
                    "gold_count":len(golds[q]),"missing_gold_ids":missing,
                    "missing_gold_source_ranks":per})
    return out

def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        if not rows: return
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docids=load_docs()
    qids,golds,questions,q2fold=load_eval()
    ranks=load_rankings(docids,qids)

    combos=[
        ("aiteamvn_v1",),
        ("aiteamvn_v1","vnlegal_lal"),
        ("aiteamvn_v1","vietlegal_harrier"),
        ("aiteamvn_v1","bm25"),
        ("aiteamvn_v1","vnlegal_lal","vietlegal_harrier"),
        ("aiteamvn_v1","vnlegal_lal","bm25"),
        ("aiteamvn_v1","vietlegal_harrier","bm25"),
        CORE,
    ]
    curves=[]
    for src in combos:
        for d in DEPTHS:
            row=eval_union(ranks,src,d,qids,golds); curves.append(row)
            print(f"[union] {','.join(src)} d={d} R={row['macro_oracle_recall']:.6f} "
                  f"full={row['full_gold_coverage_rate']:.4%} pool={row['mean_unique_pool_size']:.1f}",flush=True)

    subsets=[]
    for n in range(1,len(CORE)+1):
        for src in combinations(CORE,n):
            subsets.append(eval_union(ranks,src,100,qids,golds))
    subsets.sort(key=lambda x:(-x["macro_oracle_recall"],len(x["sources"]),x["mean_unique_pool_size"]))

    marg=[]
    for d in DEPTHS:
        for c in DROPPED:
            row=marginal(ranks,CORE,c,d,qids,golds); marg.append(row)
            print(f"[kill] +{c} d={d} dR={row['delta_macro_recall']:+.6f} "
                  f"q={row['query_rescues']} gold={row['gold_occurrence_rescues']}",flush=True)

    replace=[]
    base2=("aiteamvn_v1","vnlegal_lal")
    for c in ("vietlegal_harrier","bm25",*DROPPED):
        replace.append(marginal(ranks,base2,c,100,qids,golds))

    core100=eval_union(ranks,CORE,100,qids,golds)
    miss=failures(ranks,CORE,100,qids,golds,questions,q2fold)

    write_csv(OUT/"CORE_UNION_CURVES.csv",curves)
    write_csv(OUT/"CORE_SUBSETS_AT100.csv",subsets)
    write_csv(OUT/"DROPPED_SOURCE_MARGINAL.csv",marg)
    write_csv(OUT/"THIRD_SOURCE_REPLACEMENT_AT100.csv",replace)
    (OUT/"CORE_UNCOVERED_FORENSICS.json").write_text(
        json.dumps({"sources":list(CORE),"per_source_depth":100,
                    "uncovered_query_count":len(miss),"queries":miss},
                   ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    summary={
        "schema_version":"dsc2026.endgame.stage02b5_acquisition_oracle.v1",
        "status":"COMPLETE",
        "claim_boundary":"cached candidate-acquisition oracle only; no fusion/promotion",
        "document_order_alignment":"PASS",
        "population":{"queries":len(qids),"documents":len(docids)},
        "sources_available":list(ranks),
        "retained_core":list(CORE),
        "core_at100":core100,
        "uncovered_queries_core_at100":len(miss),
        "core_union_curves":curves,
        "core_subsets_at100":subsets,
        "dropped_source_marginal":marg,
        "third_source_replacement_at100":replace,
    }
    (OUT/"ACQUISITION_ORACLE.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    lines=[
        "# Stage 02B5 — Cached Acquisition Oracle","",
        "**No models loaded; no embeddings recomputed.**","",
        "Core: `AITeamVN-v1 + VNLegal-LAL + Harrier + BM25`.","",
        f"- Macro oracle Recall@100: **{core100['macro_oracle_recall']:.6f}**",
        f"- Full-gold query coverage: **{core100['full_gold_coverage_rate']:.4%}**",
        f"- Gold occurrence coverage: **{core100['gold_occurrence_coverage']:.4%}**",
        f"- Mean unique pool size: **{core100['mean_unique_pool_size']:.1f}**",
        f"- Queries not fully covered: **{len(miss)}**","",
        "## Proposed-kill challenge @100","",
        "| Challenger | Δ macro recall | Query rescues | Gold rescues | Full-query rescues |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in marg:
        if r["per_source_depth"]==100:
            lines.append(f"| {r['challenger']} | {r['delta_macro_recall']:+.6f} | "
                         f"{r['query_rescues']} | {r['gold_occurrence_rescues']} | {r['full_query_rescues']} |")
    lines += ["","No source is promoted automatically from this oracle analysis.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print(json.dumps({"status":"COMPLETE",
                      "core_recall_at100":core100["macro_oracle_recall"],
                      "core_full_coverage_at100":core100["full_gold_coverage_rate"],
                      "core_mean_pool_size":core100["mean_unique_pool_size"],
                      "uncovered_queries":len(miss),"out":str(OUT)},
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

#!/usr/bin/env python
"""Stage 02C0 v2 — representation geometry/token-budget audit.

Fix vs v1:
- explicitly disables Hugging Face tokenizers backend truncation before raw
  token counting. This is required for VNLegal-LAL because its tokenizer
  snapshot carries a 512-token truncation preset.
- verifies with a synthetic long-text probe that raw counting can exceed 512.

NO model forward pass. Tokenizers only.

Models:
- AITeamVN/Vietnamese_Embedding v1
- darklethelong/vnlegal-lal

Representations:
1) current_atomic_truncate_2048
2) atomic_split_2048
3) coarse_pack_1024
4) coarse_pack_1792

Run:
    python src/stage02_candidate_generation/audit_representation_geometry_v2.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEOM = ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"
EVAL = ROOT/"data/evaluation_v2"
OUT = ROOT/"reports/stage02c0_representation_geometry_v2"

MODELS = {
    "aiteamvn_v1": {
        "path": ROOT/"models/retrievers_vi/aiteamvn-vietnamese-embedding",
        "fix_mistral_regex": False,
        "max_input": 2048,
    },
    "vnlegal_lal": {
        "path": ROOT/"models/retrievers/vnlegal-lal",
        "fix_mistral_regex": True,
        "max_input": 2048,
    },
}

def rj(p): return json.loads(p.read_text(encoding="utf-8"))

def desc(xs):
    x=np.asarray(list(xs),dtype=np.float64)
    return {k:float(v) for k,v in {
        "min":x.min(),"p25":np.quantile(x,.25),"median":np.median(x),
        "mean":x.mean(),"p75":np.quantile(x,.75),"p90":np.quantile(x,.90),
        "p95":np.quantile(x,.95),"p99":np.quantile(x,.99),"max":x.max()
    }.items()}

def load_regions():
    rows=[]
    with GEOM.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                rows.append({
                    "region_id":str(x["region_id"]),
                    "doc_id":str(x["doc_id"]),
                    "kind":str(x["kind"]),
                    "raw_text":str(x["raw_text"]),
                })
    if len(rows)!=179737: raise RuntimeError(f"Expected 179737, got {len(rows)}")
    return rows

def gold_docs():
    return {str(d) for ds in rj(EVAL/"primary_golds_6991.json").values() for d in ds}

def load_tokenizer(spec):
    from transformers import AutoTokenizer
    kw={"local_files_only":True,"use_fast":True}
    if spec["fix_mistral_regex"]: kw["fix_mistral_regex"]=True
    tok=AutoTokenizer.from_pretrained(str(spec["path"]),**kw)
    if not hasattr(tok,"backend_tokenizer"):
        raise RuntimeError("Fast tokenizer required")

    # CRITICAL: tokenizer.json may carry a baked-in truncation preset. Disable it
    # before any "raw token length" measurement.
    tok.backend_tokenizer.no_truncation()

    probe=("Điều 49 quy định quyền và nghĩa vụ của người lao động theo pháp luật Việt Nam. " * 400)
    n=len(tok.backend_tokenizer.encode(probe,add_special_tokens=False).ids)
    if n<=512:
        raise RuntimeError(
            f"Untruncated-tokenizer probe failed: only {n} tokens after no_truncation()."
        )
    return tok,n

def token_count(tok,text):
    return len(tok.backend_tokenizer.encode(text,add_special_tokens=False).ids)

def atomic_truncate(counts,docs,golds,max_input,specials):
    budget=max_input-specials
    vpd=defaultdict(int); trunc_docs=set(); trunc_gold=set()
    retained=0; trunc_regions=0
    for n,d in zip(counts,docs):
        vpd[d]+=1; retained+=min(n,budget)
        if n>budget:
            trunc_regions+=1; trunc_docs.add(d)
            if d in golds: trunc_gold.add(d)
    total=sum(counts)
    return {
        "vector_count":len(counts),"vectors_per_document":desc(vpd.values()),
        "truncated_region_count":trunc_regions,
        "truncated_region_rate":trunc_regions/len(counts),
        "documents_with_truncation":len(trunc_docs),
        "gold_documents_with_truncation":len(trunc_gold),
        "estimated_content_token_coverage":retained/total,
        "estimated_input_tokens_to_embed":retained+len(counts)*specials,
        "coverage_contract":"truncates oversized regions",
    }

def atomic_split(counts,docs,golds,max_input,specials):
    budget=max_input-specials
    vpd=defaultdict(int); split_docs=set(); split_gold=set()
    vectors=0; split_regions=0
    for n,d in zip(counts,docs):
        k=max(1,math.ceil(n/budget)); vectors+=k; vpd[d]+=k
        if k>1:
            split_regions+=1; split_docs.add(d)
            if d in golds: split_gold.add(d)
    return {
        "vector_count":vectors,"vectors_per_document":desc(vpd.values()),
        "split_region_count":split_regions,"split_region_rate":split_regions/len(counts),
        "documents_with_split_region":len(split_docs),
        "gold_documents_with_split_region":len(split_gold),
        "estimated_content_token_coverage":1.0,
        "estimated_input_tokens_to_embed":sum(counts)+vectors*specials,
        "coverage_contract":"full coverage; oversized region split without overlap",
    }

def coarse_pack(counts,docs,golds,max_input,specials):
    budget=max_input-specials
    vpd=defaultdict(int); oversized_docs=set(); oversized_gold=set()
    vectors=0; oversized=0; cur_doc=None; fill=0
    def flush():
        nonlocal vectors,fill
        if fill>0:
            vectors+=1; vpd[cur_doc]+=1; fill=0
    for n,d in zip(counts,docs):
        if cur_doc is None: cur_doc=d
        elif d!=cur_doc:
            flush(); cur_doc=d
        if n>budget:
            flush()
            k=max(1,math.ceil(n/budget)); vectors+=k; vpd[d]+=k
            oversized+=1; oversized_docs.add(d)
            if d in golds: oversized_gold.add(d)
        elif fill==0:
            fill=n
        elif fill+n<=budget:
            fill+=n
        else:
            flush(); fill=n
    flush()
    return {
        "vector_count":vectors,"vectors_per_document":desc(vpd.values()),
        "oversized_region_count":oversized,"oversized_region_rate":oversized/len(counts),
        "documents_with_oversized_region":len(oversized_docs),
        "gold_documents_with_oversized_region":len(oversized_gold),
        "estimated_content_token_coverage":1.0,
        "estimated_input_tokens_to_embed":sum(counts)+vectors*specials,
        "coverage_contract":"full coverage; greedy consecutive pack; oversized split; no overlap",
    }

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_regions(); gold=gold_docs()
    docs=[x["doc_id"] for x in rows]; kinds=[x["kind"] for x in rows]
    texts=[x["raw_text"] for x in rows]
    out_models={}

    for key,spec in MODELS.items():
        print(f"=== {key} ===",flush=True)
        tok,probe_tokens=load_tokenizer(spec)
        specials=int(tok.num_special_tokens_to_add(pair=False))
        counts=[]; bykind=defaultdict(list)
        for i,(t,k) in enumerate(zip(texts,kinds),1):
            n=token_count(tok,t); counts.append(n); bykind[k].append(n)
            if i%10000==0 or i==len(texts):
                print(f"[{key}] {i}/{len(texts)}",flush=True)
        a=np.asarray(counts,dtype=np.int64)
        reps={
            "current_atomic_truncate_2048":atomic_truncate(counts,docs,gold,2048,specials),
            "atomic_split_2048":atomic_split(counts,docs,gold,2048,specials),
            "coarse_pack_1024":coarse_pack(counts,docs,gold,1024,specials),
            "coarse_pack_1792":coarse_pack(counts,docs,gold,1792,specials),
        }
        for r in reps.values():
            r["vector_count_ratio_vs_stage02b4_atomic"]=r["vector_count"]/len(rows)
        result={
            "model_key":key,"max_input_tokens":2048,
            "special_tokens_per_single_input":specials,
            "backend_truncation_explicitly_disabled":True,
            "long_probe_token_count":probe_tokens,
            "base_regions":len(rows),"documents":len(set(docs)),
            "unique_gold_documents":len(gold),
            "content_token_count":{
                "all_regions":desc(counts),
                "by_kind":{k:desc(v) for k,v in bykind.items()},
                "total":int(a.sum()),
            },
            "thresholds":{
                str(t):{"regions_over":int(np.sum(a>(t-specials))),
                        "rate_over":float(np.mean(a>(t-specials)))}
                for t in (256,512,1024,1536,1792,2048)
            },
            "representations":reps,
        }
        out_models[key]=result
        (OUT/f"{key}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    summary={
        "schema_version":"dsc2026.endgame.stage02c0_representation_geometry.v2",
        "status":"COMPLETE",
        "fix":"explicit tokenizer backend no_truncation; v1 LAL measurements invalidated",
        "claim_boundary":"tokenizer/geometry audit only; no retrieval conclusion",
        "base_geometry":{"regions":len(rows),"documents":len(set(docs)),"gold_documents":len(gold)},
        "models":out_models,
    }
    (OUT/"REPRESENTATION_GEOMETRY_AUDIT.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    lines=[
        "# Stage 02C0 v2 — Representation Geometry Audit","",
        "v1 LAL token counts are invalidated; v2 explicitly disables backend truncation.","",
        "| Model | Representation | Vectors | Ratio | Median vec/doc | P95 vec/doc | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for k,m in out_models.items():
        for n,r in m["representations"].items():
            v=r["vectors_per_document"]
            lines.append(f"| {k} | {n} | {r['vector_count']} | "
                         f"{r['vector_count_ratio_vs_stage02b4_atomic']:.3f}x | "
                         f"{v['median']:.1f} | {v['p95']:.1f} | "
                         f"{r['estimated_content_token_coverage']:.4%} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","out":str(OUT)},indent=2))

if __name__=="__main__":
    main()

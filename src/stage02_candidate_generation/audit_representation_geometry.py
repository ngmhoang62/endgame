#!/usr/bin/env python
"""Stage 02C0 — representation geometry/token-budget audit.

NO model forward pass. Tokenizers only.

Models:
- AITeamVN/Vietnamese_Embedding v1
- darklethelong/vnlegal-lal

Both are audited at the frozen 2048-token inference/training contract.

Representations:
1) current_atomic_truncate_2048
   - exactly the Stage02B4 geometry: one vector per moderate region
   - regions longer than context are truncated
2) atomic_split_2048
   - keep every moderate region atomic when it fits
   - split only oversized regions into non-overlapping token pieces
3) coarse_pack_1024
   - greedily pack consecutive fitting moderate regions within a parent
   - oversized regions are split standalone; no overlap
4) coarse_pack_1792
   - same, with a larger practical input budget

This stage estimates vector counts, fragmentation, token coverage, and embedding
token cost. It does not create embeddings and does not promote a representation.

Run:
    python src/stage02_candidate_generation/audit_representation_geometry.py
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEOM = ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"
EVAL = ROOT/"data/evaluation_v2"
OUT = ROOT/"reports/stage02c0_representation_geometry"

MODELS = {
    "aiteamvn_v1": {
        "path": ROOT/"models/retrievers_vi/aiteamvn-vietnamese-embedding",
        "kind": "auto",
        "fix_mistral_regex": False,
        "max_input": 2048,
    },
    "vnlegal_lal": {
        "path": ROOT/"models/retrievers/vnlegal-lal",
        "kind": "auto",
        "fix_mistral_regex": True,
        "max_input": 2048,
    },
}

TARGETS = {
    "atomic_split_2048": 2048,
    "coarse_pack_1024": 1024,
    "coarse_pack_1792": 1792,
}


def rj(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def desc(xs):
    x=np.asarray(list(xs),dtype=np.float64)
    if len(x)==0:
        return {}
    return {
        "min":float(x.min()),
        "p25":float(np.quantile(x,.25)),
        "median":float(np.median(x)),
        "mean":float(x.mean()),
        "p75":float(np.quantile(x,.75)),
        "p90":float(np.quantile(x,.90)),
        "p95":float(np.quantile(x,.95)),
        "p99":float(np.quantile(x,.99)),
        "max":float(x.max()),
    }


def load_regions():
    rows=[]
    with GEOM.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                rows.append({
                    "region_id":str(x["region_id"]),
                    "doc_id":str(x["doc_id"]),
                    "kind":str(x["kind"]),
                    "raw_text":str(x["raw_text"]),
                })
    if len(rows)!=179737:
        raise RuntimeError(f"Expected 179737 regions, got {len(rows)}")
    return rows


def gold_docs():
    golds=rj(EVAL/"primary_golds_6991.json")
    return {str(d) for ds in golds.values() for d in ds}


def load_tokenizer(spec):
    from transformers import AutoTokenizer
    kwargs=dict(local_files_only=True,use_fast=True)
    if spec["fix_mistral_regex"]:
        kwargs["fix_mistral_regex"]=True
    tok=AutoTokenizer.from_pretrained(str(spec["path"]),**kwargs)
    if not hasattr(tok,"backend_tokenizer"):
        raise RuntimeError(f"Fast tokenizer required: {spec['path']}")
    return tok


def content_token_count(tok,text):
    # backend tokenizer avoids model_max_length warnings and truncation.
    return len(tok.backend_tokenizer.encode(text,add_special_tokens=False).ids)


def simulate_atomic_truncate(counts,doc_ids,gold_set,max_input,specials):
    budget=max_input-specials
    vec_per_doc=defaultdict(int)
    truncated_docs=set()
    truncated_gold_docs=set()
    total_content=sum(counts)
    retained=0
    truncated_regions=0

    for n,d in zip(counts,doc_ids):
        vec_per_doc[d]+=1
        retained += min(n,budget)
        if n>budget:
            truncated_regions+=1
            truncated_docs.add(d)
            if d in gold_set:
                truncated_gold_docs.add(d)

    vectors=len(counts)
    return {
        "vector_count":vectors,
        "vectors_per_document":desc(vec_per_doc.values()),
        "truncated_region_count":truncated_regions,
        "truncated_region_rate":truncated_regions/len(counts),
        "documents_with_truncation":len(truncated_docs),
        "gold_documents_with_truncation":len(truncated_gold_docs),
        "estimated_content_token_coverage":retained/total_content,
        "estimated_input_tokens_to_embed":retained + vectors*specials,
        "coverage_contract":"truncates oversized regions",
    }


def simulate_atomic_split(counts,doc_ids,gold_set,max_input,specials):
    budget=max_input-specials
    vec_per_doc=defaultdict(int)
    split_regions=0
    split_docs=set()
    split_gold_docs=set()
    vectors=0
    total_content=sum(counts)

    for n,d in zip(counts,doc_ids):
        pieces=max(1,math.ceil(n/budget))
        vectors+=pieces
        vec_per_doc[d]+=pieces
        if pieces>1:
            split_regions+=1
            split_docs.add(d)
            if d in gold_set:
                split_gold_docs.add(d)

    return {
        "vector_count":vectors,
        "vectors_per_document":desc(vec_per_doc.values()),
        "split_region_count":split_regions,
        "split_region_rate":split_regions/len(counts),
        "documents_with_split_region":len(split_docs),
        "gold_documents_with_split_region":len(split_gold_docs),
        "estimated_content_token_coverage":1.0,
        "estimated_input_tokens_to_embed":total_content + vectors*specials,
        "coverage_contract":"full token coverage; oversized region split without overlap",
    }


def simulate_coarse_pack(counts,doc_ids,gold_set,max_input,specials):
    budget=max_input-specials
    vec_per_doc=defaultdict(int)
    packed_vectors=0
    total_content=sum(counts)
    oversized_regions=0
    oversized_docs=set()
    oversized_gold_docs=set()

    current_doc=None
    current_fill=0

    def flush():
        nonlocal packed_vectors,current_fill,current_doc
        if current_fill>0:
            packed_vectors+=1
            vec_per_doc[current_doc]+=1
            current_fill=0

    for n,d in zip(counts,doc_ids):
        if current_doc is None:
            current_doc=d
        elif d!=current_doc:
            flush()
            current_doc=d

        if n>budget:
            flush()
            pieces=max(1,math.ceil(n/budget))
            packed_vectors+=pieces
            vec_per_doc[d]+=pieces
            oversized_regions+=1
            oversized_docs.add(d)
            if d in gold_set:
                oversized_gold_docs.add(d)
            continue

        if current_fill==0:
            current_fill=n
        elif current_fill+n<=budget:
            current_fill+=n
        else:
            flush()
            current_fill=n

    flush()

    return {
        "vector_count":packed_vectors,
        "vectors_per_document":desc(vec_per_doc.values()),
        "oversized_region_count":oversized_regions,
        "oversized_region_rate":oversized_regions/len(counts),
        "documents_with_oversized_region":len(oversized_docs),
        "gold_documents_with_oversized_region":len(oversized_gold_docs),
        "estimated_content_token_coverage":1.0,
        "estimated_input_tokens_to_embed":total_content + packed_vectors*specials,
        "coverage_contract":(
            "full token coverage; consecutive fitting regions greedily packed "
            "within parent; oversized regions split standalone; no overlap"
        ),
    }


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    regions=load_regions()
    gold=gold_docs()
    doc_ids=[r["doc_id"] for r in regions]
    kinds=[r["kind"] for r in regions]
    texts=[r["raw_text"] for r in regions]
    unique_docs=len(set(doc_ids))
    if unique_docs!=8512:
        raise RuntimeError(f"Expected 8512 docs, got {unique_docs}")

    all_results={}

    for key,spec in MODELS.items():
        print(f"=== {key} tokenizer audit ===",flush=True)
        tok=load_tokenizer(spec)
        specials=int(tok.num_special_tokens_to_add(pair=False))
        if specials<0 or specials>16:
            raise RuntimeError(f"Unexpected special-token overhead: {specials}")

        counts=[]
        by_kind=defaultdict(list)
        for i,(text,kind) in enumerate(zip(texts,kinds),1):
            n=content_token_count(tok,text)
            counts.append(n)
            by_kind[kind].append(n)
            if i%10000==0 or i==len(texts):
                print(f"[{key}] tokenized {i}/{len(texts)}",flush=True)

        counts_np=np.asarray(counts,dtype=np.int64)
        total_content=int(counts_np.sum())
        model_result={
            "model_key":key,
            "max_input_tokens":spec["max_input"],
            "special_tokens_per_single_input":specials,
            "content_budget_at_2048":spec["max_input"]-specials,
            "base_regions":len(regions),
            "documents":unique_docs,
            "unique_gold_documents":len(gold),
            "content_token_count":{
                "all_regions":desc(counts),
                "by_kind":{k:desc(v) for k,v in by_kind.items()},
                "total":total_content,
            },
            "thresholds":{
                str(t):{
                    "regions_over":int(np.sum(counts_np>(t-specials))),
                    "rate_over":float(np.mean(counts_np>(t-specials))),
                }
                for t in (256,512,1024,1536,1792,2048)
            },
            "representations":{},
        }

        model_result["representations"]["current_atomic_truncate_2048"] = (
            simulate_atomic_truncate(
                counts,doc_ids,gold,spec["max_input"],specials
            )
        )
        model_result["representations"]["atomic_split_2048"] = (
            simulate_atomic_split(
                counts,doc_ids,gold,spec["max_input"],specials
            )
        )
        model_result["representations"]["coarse_pack_1024"] = (
            simulate_coarse_pack(
                counts,doc_ids,gold,1024,specials
            )
        )
        model_result["representations"]["coarse_pack_1792"] = (
            simulate_coarse_pack(
                counts,doc_ids,gold,1792,specials
            )
        )

        base=len(regions)
        for name,r in model_result["representations"].items():
            r["vector_count_ratio_vs_stage02b4_atomic"] = r["vector_count"]/base

        all_results[key]=model_result

        (OUT/f"{key}.json").write_text(
            json.dumps(model_result,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8",
        )

    report={
        "schema_version":"dsc2026.endgame.stage02c0_representation_geometry.v1",
        "status":"COMPLETE",
        "claim_boundary":"tokenizer/geometry audit only; no retrieval quality conclusion",
        "base_geometry":{
            "regions":len(regions),
            "documents":unique_docs,
            "gold_documents":len(gold),
            "source":"stage02b0 moderate structure",
        },
        "models":all_results,
    }
    (OUT/"REPRESENTATION_GEOMETRY_AUDIT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )

    lines=[
        "# Stage 02C0 — Representation Geometry Audit","",
        "Tokenizer-only audit. No model forward pass.","",
        "| Model | Representation | Vectors | Ratio vs B4 | Median vec/doc | P95 vec/doc | Token coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key,m in all_results.items():
        for name,r in m["representations"].items():
            vpd=r["vectors_per_document"]
            lines.append(
                f"| {key} | {name} | {r['vector_count']} | "
                f"{r['vector_count_ratio_vs_stage02b4_atomic']:.3f}x | "
                f"{vpd['median']:.1f} | {vpd['p95']:.1f} | "
                f"{r['estimated_content_token_coverage']:.4%} |"
            )
    lines += [
        "",
        "Selection of a representation happens only after reviewing fragmentation,",
        "coverage, and estimated embedding cost from this audit.",
        "",
    ]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print(json.dumps({
        "status":"COMPLETE",
        "out":str(OUT),
        "models":{
            k:{
                n:{
                    "vectors":r["vector_count"],
                    "ratio":r["vector_count_ratio_vs_stage02b4_atomic"],
                    "coverage":r["estimated_content_token_coverage"],
                }
                for n,r in m["representations"].items()
            }
            for k,m in all_results.items()
        }
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

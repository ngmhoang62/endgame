#!/usr/bin/env python
"""Stage 02C2 — VNLegal-LAL representation screen.

Purpose
-------
Port the representation fix from Stage02C1 to the second retained dense family.

Only two new LAL document views are tested:
- atomic_split_2048
- coarse_pack_1024

Controls
--------
- Existing LAL B4 query embeddings are reused read-only.
- Existing AITeam coarse1024 / atomic-split rankings are reused read-only.
- Existing BM25 ranking is reused read-only.
- No metadata/title prefix.
- No overlap.
- Literal document IDs.
- Parent score = exact max cosine over child vectors.

Safety
------
All new assets live under cache/stage02c2_lal_representation/.
Existing caches are never opened writable.
Any cache contract mismatch FAILS CLOSED; nothing is silently overwritten.

Run:
    python src/stage02_candidate_generation/screen_lal_representations.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.evaluation import official_metrics

HF_ROOT = ROOT/"cache/huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT/"hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT/"modules")

MODEL = ROOT/"models/retrievers/vnlegal-lal"
MODEL_MANIFEST = ROOT/"reports/stage02b1_model_materialization/vnlegal_lal/MODEL_MANIFEST.json"
BASE_GEOM = ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"
BASE_GEOM_REPORT = ROOT/"reports/stage02b0_structure_geometry/STRUCTURE_GEOMETRY.json"
RAW = ROOT/"data/official_v1"
EVAL = ROOT/"data/evaluation_v2"

B4 = ROOT/"cache/stage02b4_vi_screen"
C1 = ROOT/"cache/stage02c1_aiteam_representation"
A2 = ROOT/"cache/stage02a_parent_anchor"

CACHE = ROOT/"cache/stage02c2_lal_representation"
OUT = ROOT/"reports/stage02c2_lal_representation"

TOPK = 100
KS = (1,5,10,20,50,100)
CORE_DEPTHS = (20,50,100)
REP_ALGO_VERSION = "stage02c2.lal_representation.v1"

REPS = {
    "lal_atomic_split_2048": {
        "mode":"atomic_split",
        "max_input":2048,
        "batch_size":2,
    },
    "lal_coarse_pack_1024": {
        "mode":"coarse_pack",
        "max_input":1024,
        "batch_size":4,
    },
}


def rj(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    ).hexdigest()


def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def desc(values):
    a=np.asarray(list(values),dtype=np.float64)
    if not len(a): return {}
    return {
        "min":float(a.min()),"p25":float(np.quantile(a,.25)),
        "median":float(np.median(a)),"mean":float(a.mean()),
        "p75":float(np.quantile(a,.75)),"p90":float(np.quantile(a,.90)),
        "p95":float(np.quantile(a,.95)),"p99":float(np.quantile(a,.99)),
        "max":float(a.max()),
    }


def load_model_contract():
    m=rj(MODEL_MANIFEST)
    if m.get("status")!="PASS":
        raise RuntimeError("LAL manifest is not PASS")
    if m.get("resolved_revision_sha")!="de759324ef931a2475ae8db97137b6a6cbb98aa0":
        raise RuntimeError("LAL model SHA drift")
    ac=m.get("adapter_contract",{})
    if int(ac.get("operational_max_length",-1))!=2048:
        raise RuntimeError("LAL 2048 operational contract drift")
    if ac.get("pooling")!="last_non_padding_token":
        raise RuntimeError("LAL pooling contract drift")
    return m


def load_tokenizer_untruncated():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(
        str(MODEL),local_files_only=True,use_fast=True,fix_mistral_regex=True
    )
    if not hasattr(tok,"backend_tokenizer"):
        raise RuntimeError("Fast tokenizer required")
    tok.backend_tokenizer.no_truncation()
    probe=("Điều 49 quy định quyền và nghĩa vụ của người lao động. "*500)
    n=len(tok.backend_tokenizer.encode(probe,add_special_tokens=False).ids)
    if n<=2048:
        raise RuntimeError(f"LAL tokenizer no_truncation probe failed: {n}")
    return tok


def base_geometry_contract():
    g=rj(BASE_GEOM_REPORT)
    expected=g["artifact"]["regions_sha256"]
    actual=sha256_file(BASE_GEOM)
    if actual!=expected:
        raise RuntimeError(f"Base geometry SHA drift: {actual} != {expected}")
    return actual


def iter_base_regions():
    with BASE_GEOM.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                yield {
                    "region_id":str(x["region_id"]),
                    "doc_id":str(x["doc_id"]),
                    "kind":str(x["kind"]),
                    "raw_text":str(x["raw_text"]),
                }


def decode_token_slices(tok,text,budget):
    enc=tok.backend_tokenizer.encode(text,add_special_tokens=False)
    ids=enc.ids
    if len(ids)<=budget:
        return [(text,len(ids),len(ids))]
    out=[]
    for s in range(0,len(ids),budget):
        source_ids=ids[s:s+budget]
        chunk=tok.backend_tokenizer.decode(source_ids,skip_special_tokens=True)
        if not chunk.strip():
            continue
        actual=len(tok.backend_tokenizer.encode(chunk,add_special_tokens=False).ids)
        if actual>budget:
            hi=len(source_ids)
            while hi>1:
                hi-=1
                chunk=tok.backend_tokenizer.decode(source_ids[:hi],skip_special_tokens=True)
                actual=len(tok.backend_tokenizer.encode(chunk,add_special_tokens=False).ids)
                if actual<=budget:
                    break
            if actual>budget:
                raise RuntimeError("Could not back off decoded LAL slice below budget")
            out.append((chunk,hi,actual))
            rem=tok.backend_tokenizer.decode(source_ids[hi:],skip_special_tokens=True)
            if rem.strip():
                out.extend(decode_token_slices(tok,rem,budget))
        else:
            out.append((chunk,len(source_ids),actual))
    return out


def rep_contract(name,spec,geom_sha,manifest,specials):
    return {
        "schema":REP_ALGO_VERSION,
        "name":name,
        "mode":spec["mode"],
        "max_input":spec["max_input"],
        "content_budget":spec["max_input"]-specials,
        "separator":"\\n\\n",
        "overlap_tokens":0,
        "metadata_prefix":False,
        "same_parent_only":True,
        "oversized_policy":"tokenizer_id_slice_decode_no_overlap",
        "base_geometry_sha":geom_sha,
        "model_sha":manifest["resolved_revision_sha"],
        "fix_mistral_regex":True,
        "document_prefix":"",
    }


def emit_row(out,row,stats):
    out.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    stats["vectors"]+=1
    stats["doc_counts"][row["doc_id"]]+=1
    stats["source_tokens"]+=int(row["source_token_count"])
    stats["actual_tokens"]+=int(row["content_tokens"])
    stats["max_content_tokens"]=max(stats["max_content_tokens"],int(row["content_tokens"]))


def materialize_representation(name,spec,tok,geom_sha,manifest):
    d=CACHE/name
    d.mkdir(parents=True,exist_ok=True)
    data=d/"chunks.jsonl"
    meta=d/"representation.json"
    specials=int(tok.num_special_tokens_to_add(pair=False))
    contract=rep_contract(name,spec,geom_sha,manifest,specials)
    ch=stable_hash(contract)

    if data.exists()!=meta.exists():
        raise RuntimeError(f"[{name}] incomplete representation cache pair; REFUSING TO OVERWRITE")
    if data.exists():
        old=rj(meta)
        if old.get("contract_hash")!=ch:
            raise RuntimeError(f"[{name}] representation contract mismatch; REFUSING TO OVERWRITE")
        if old.get("status")!="PASS":
            raise RuntimeError(f"[{name}] representation cache not PASS; REFUSING TO OVERWRITE")
        if sha256_file(data)!=old.get("data_sha256"):
            raise RuntimeError(f"[{name}] chunks SHA mismatch")
        print(f"[{name}] representation cache hit vectors={old['vector_count']}",flush=True)
        return data,old

    tmp=d/"chunks.jsonl.tmp"
    if tmp.exists():
        raise RuntimeError(f"[{name}] stale temp exists: {tmp}")

    budget=spec["max_input"]-specials
    stats={"vectors":0,"doc_counts":defaultdict(int),"source_tokens":0,
           "actual_tokens":0,"max_content_tokens":0,"base_regions":0,
           "oversized_base_regions":0,"split_pieces":0,
           "additional_regions_packed":0}
    t0=time.perf_counter()

    with tmp.open("w",encoding="utf-8",newline="\n") as out:
        if spec["mode"]=="atomic_split":
            for i,r in enumerate(iter_base_regions(),1):
                stats["base_regions"]+=1
                n=len(tok.backend_tokenizer.encode(r["raw_text"],add_special_tokens=False).ids)
                if n<=budget:
                    emit_row(out,{
                        "chunk_id":f"{name}:{stats['vectors']:09d}",
                        "doc_id":r["doc_id"],"text":r["raw_text"],
                        "source_region_ids":[r["region_id"]],
                        "source_kinds":[r["kind"]],
                        "source_token_count":n,"content_tokens":n,
                        "piece_of_oversized_region":False,
                    },stats)
                else:
                    stats["oversized_base_regions"]+=1
                    pieces=decode_token_slices(tok,r["raw_text"],budget)
                    stats["split_pieces"]+=len(pieces)
                    for pi,(txt,srcn,actn) in enumerate(pieces):
                        emit_row(out,{
                            "chunk_id":f"{name}:{stats['vectors']:09d}",
                            "doc_id":r["doc_id"],"text":txt,
                            "source_region_ids":[r["region_id"]],
                            "source_kinds":[r["kind"]],
                            "source_token_count":srcn,"content_tokens":actn,
                            "piece_of_oversized_region":True,"piece_index":pi,
                        },stats)
                if i%5000==0 or i==179737:
                    print(f"[{name}] materialize {i}/179737 vectors={stats['vectors']}",flush=True)

        elif spec["mode"]=="coarse_pack":
            current_doc=None
            current_text=""
            current_ids=[]
            current_kinds=[]
            current_source_tokens=0

            def flush_pack():
                nonlocal current_text,current_ids,current_kinds,current_source_tokens
                if not current_text: return
                actual=len(tok.backend_tokenizer.encode(current_text,add_special_tokens=False).ids)
                if actual>budget:
                    raise RuntimeError(f"{name}: packed chunk {actual}>{budget}")
                emit_row(out,{
                    "chunk_id":f"{name}:{stats['vectors']:09d}",
                    "doc_id":current_doc,"text":current_text,
                    "source_region_ids":list(current_ids),
                    "source_kinds":list(current_kinds),
                    "source_token_count":current_source_tokens,
                    "content_tokens":actual,
                    "piece_of_oversized_region":False,
                },stats)
                current_text=""; current_ids=[]; current_kinds=[]; current_source_tokens=0

            for i,r in enumerate(iter_base_regions(),1):
                stats["base_regions"]+=1
                if current_doc is None:
                    current_doc=r["doc_id"]
                elif r["doc_id"]!=current_doc:
                    flush_pack(); current_doc=r["doc_id"]

                n=len(tok.backend_tokenizer.encode(r["raw_text"],add_special_tokens=False).ids)
                if n>budget:
                    flush_pack()
                    stats["oversized_base_regions"]+=1
                    pieces=decode_token_slices(tok,r["raw_text"],budget)
                    stats["split_pieces"]+=len(pieces)
                    for pi,(txt,srcn,actn) in enumerate(pieces):
                        emit_row(out,{
                            "chunk_id":f"{name}:{stats['vectors']:09d}",
                            "doc_id":r["doc_id"],"text":txt,
                            "source_region_ids":[r["region_id"]],
                            "source_kinds":[r["kind"]],
                            "source_token_count":srcn,"content_tokens":actn,
                            "piece_of_oversized_region":True,"piece_index":pi,
                        },stats)
                    continue

                candidate=r["raw_text"] if not current_text else current_text+"\n\n"+r["raw_text"]
                cn=len(tok.backend_tokenizer.encode(candidate,add_special_tokens=False).ids)
                if cn<=budget:
                    if current_text:
                        stats["additional_regions_packed"]+=1
                    current_text=candidate
                    current_ids.append(r["region_id"])
                    current_kinds.append(r["kind"])
                    current_source_tokens+=n
                else:
                    flush_pack()
                    current_text=r["raw_text"]
                    current_ids=[r["region_id"]]
                    current_kinds=[r["kind"]]
                    current_source_tokens=n

                if i%5000==0 or i==179737:
                    print(f"[{name}] materialize {i}/179737 vectors={stats['vectors']}",flush=True)
            flush_pack()
        else:
            raise RuntimeError(spec["mode"])

    tmp.replace(data)
    meta_obj={
        "schema_version":"dsc2026.endgame.stage02c2.representation.v1",
        "status":"PASS","contract_hash":ch,"contract":contract,
        "vector_count":stats["vectors"],"base_region_count":stats["base_regions"],
        "oversized_base_region_count":stats["oversized_base_regions"],
        "split_piece_count":stats["split_pieces"],
        "additional_regions_packed_into_existing_chunk":stats["additional_regions_packed"],
        "vectors_per_document":desc(stats["doc_counts"].values()),
        "source_token_count_total":stats["source_tokens"],
        "actual_retokenized_content_tokens_total":stats["actual_tokens"],
        "max_content_tokens":stats["max_content_tokens"],
        "data_sha256":sha256_file(data),
        "materialize_seconds":time.perf_counter()-t0,
    }
    meta.write_text(json.dumps(meta_obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[{name}] materialized vectors={stats['vectors']}",flush=True)
    return data,meta_obj


def load_rep_rows(data):
    texts=[]; docs=[]
    with data.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                texts.append(str(x["text"]))
                docs.append(str(x["doc_id"]))
    return texts,docs


def robust_last_token(last_hidden,attention_mask):
    """Correct for either left or right padding."""
    import torch
    if bool(torch.all(attention_mask[:,-1] == 1).item()):
        return last_hidden[:,-1]
    lengths=attention_mask.sum(dim=1)-1
    rows=torch.arange(last_hidden.shape[0],device=last_hidden.device)
    return last_hidden[rows,lengths]


def load_lal():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(
        str(MODEL),local_files_only=True,use_fast=True,fix_mistral_regex=True
    )
    model=AutoModel.from_pretrained(
        str(MODEL),local_files_only=True,dtype=torch.float16
    ).cuda().eval()
    if int(model.config.hidden_size)!=1024:
        raise RuntimeError("LAL hidden size drift")
    return tok,model


def validate_partial_embedding(arr,completed,total,name):
    if arr.shape!=(total,1024):
        raise RuntimeError(f"[{name}] embedding shape mismatch; REFUSING TO OVERWRITE")
    if not (0<=completed<=total):
        raise RuntimeError(f"[{name}] invalid completed={completed}")
    if completed:
        n=min(completed,1024)
        idx=np.linspace(0,completed-1,n,dtype=np.int64)
        x=np.asarray(arr[idx],dtype=np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError(f"[{name}] non-finite embeddings")
        err=float(np.max(np.abs(np.linalg.norm(x,axis=1)-1.0)))
        if err>5e-3:
            raise RuntimeError(f"[{name}] embedding norm error {err}")


def embed_rep(name,spec,data,repmeta,manifest):
    import torch
    import torch.nn.functional as F

    texts,docs=load_rep_rows(data)
    d=CACHE/name
    npy=d/"embeddings.f32.npy"
    meta=d/"embeddings.json"
    contract={
        "schema":"stage02c2.lal_embeddings.v1",
        "model_sha":manifest["resolved_revision_sha"],
        "representation_contract_hash":repmeta["contract_hash"],
        "representation_data_sha":repmeta["data_sha256"],
        "max_length":spec["max_input"],
        "query_or_document":"document",
        "document_prefix":"",
        "pooling":"robust_last_non_padding_token",
        "normalization":"float32_l2",
        "dimension":1024,
        "fix_mistral_regex":True,
    }
    ch=stable_hash(contract)

    if npy.exists()!=meta.exists():
        raise RuntimeError(f"[{name}] incomplete embedding cache pair; REFUSING TO OVERWRITE")

    completed=0
    if npy.exists():
        old=rj(meta)
        if old.get("contract_hash")!=ch:
            raise RuntimeError(f"[{name}] embedding contract mismatch; REFUSING TO OVERWRITE")
        completed=int(old.get("completed",0))
        arr=np.lib.format.open_memmap(npy,mode="r+")
        validate_partial_embedding(arr,completed,len(texts),name)
        if completed==len(texts):
            print(f"[{name}] embedding cache hit {completed}/{len(texts)}",flush=True)
            return np.load(npy,mmap_mode="r"),docs,old
        print(f"[{name}] embedding RESUME {completed}/{len(texts)}",flush=True)
    else:
        arr=np.lib.format.open_memmap(npy,mode="w+",dtype=np.float32,shape=(len(texts),1024))
        old={"contract_hash":ch,"contract":contract,"completed":0,"total":len(texts),
             "dimension":1024,"status":"IN_PROGRESS"}
        meta.write_text(json.dumps(old,indent=2)+"\n",encoding="utf-8")

    tok,model=load_lal()
    batch=int(spec["batch_size"])
    i=completed; last=completed
    reductions=list(old.get("oom_reductions",[]))
    t0=time.perf_counter()

    try:
        while i<len(texts):
            j=min(i+batch,len(texts))
            try:
                inp=tok(
                    texts[i:j],padding=True,truncation=True,
                    max_length=spec["max_input"],return_tensors="pt"
                ).to("cuda")
                with torch.inference_mode():
                    out=model(**inp)
                    emb=robust_last_token(out.last_hidden_state,inp["attention_mask"])
                    emb=F.normalize(emb.float(),p=2,dim=1)
                z=emb.cpu().numpy().astype(np.float32,copy=False)
                n=np.linalg.norm(z,axis=1,keepdims=True)
                if np.any(n<=0) or not np.isfinite(n).all():
                    raise RuntimeError("bad LAL embedding norm")
                z=z/n
                arr[i:j]=z; arr.flush(); i=j

                if i-last>=1000 or i==len(texts):
                    state={
                        "contract_hash":ch,"contract":contract,
                        "completed":i,"total":len(texts),"dimension":1024,
                        "status":"PASS" if i==len(texts) else "IN_PROGRESS",
                        "initial_batch_size":spec["batch_size"],
                        "current_batch_size":batch,
                        "oom_reductions":reductions,
                        "elapsed_this_run_seconds":time.perf_counter()-t0,
                    }
                    meta.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
                    last=i
                    print(f"[{name}] embed {i}/{len(texts)} batch={batch}",flush=True)
            except torch.cuda.OutOfMemoryError:
                clear_cuda()
                if batch<=1: raise
                new=max(1,batch//2)
                reductions.append({"at_index":i,"from":batch,"to":new})
                batch=new
                print(f"[{name}] OOM -> batch={batch}",flush=True)
    finally:
        del model
        clear_cuda()

    return np.load(npy,mmap_mode="r"),docs,rj(meta)


def load_eval():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    if len(qids)!=6991: raise RuntimeError("6991 eval contract drift")
    return qids,golds,folds,stress


def doc_universe():
    out=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(str(json.loads(line)["id"]))
    if len(out)!=8512: raise RuntimeError("8512 doc contract drift")
    return out


def load_query_embeddings(qids,manifest):
    npy=B4/"vnlegal_lal/query_embeddings.f32.npy"
    meta=B4/"vnlegal_lal/query_embeddings.json"
    if not npy.exists() or not meta.exists():
        raise FileNotFoundError("B4 LAL query embeddings missing")
    m=rj(meta)
    if int(m.get("completed",-1))!=len(qids) or int(m.get("dimension",-1))!=1024:
        raise RuntimeError("B4 LAL query embedding cache incomplete")
    a=np.load(npy,mmap_mode="r")
    if a.shape!=(len(qids),1024): raise RuntimeError(f"LAL query shape {a.shape}")
    return a,m


def exact_parent_max_search(name,emb,qemb,chunk_docs,docids,embedding_hash):
    import torch
    d=CACHE/name
    ip=d/"parent_top100_idx.npy"
    sp=d/"parent_top100_scores.npy"
    mp=d/"parent_top100.json"
    contract={
        "schema":"stage02c2.parent_amax.v1",
        "embedding_contract_hash":embedding_hash,
        "aggregation":"exact_parent_amax","topk":100,
        "doc_order_sha":stable_hash(docids),
    }
    ch=stable_hash(contract)

    if any(p.exists() for p in (ip,sp,mp)):
        if not all(p.exists() for p in (ip,sp,mp)):
            raise RuntimeError(f"[{name}] incomplete ranking cache; REFUSING TO OVERWRITE")
        old=rj(mp)
        if old.get("contract_hash")!=ch:
            raise RuntimeError(f"[{name}] ranking contract mismatch; REFUSING TO OVERWRITE")
        idx=np.load(ip); score=np.load(sp)
        if idx.shape!=(len(qemb),100) or score.shape!=idx.shape:
            raise RuntimeError(f"[{name}] ranking shape mismatch")
        print(f"[{name}] ranking cache hit",flush=True)
        return idx,score,old

    d2i={d:i for i,d in enumerate(docids)}
    pidx_np=np.asarray([d2i[d] for d in chunk_docs],dtype=np.int64)
    rv_np=np.array(emb,dtype=np.float32,copy=True,order="C")
    rv=torch.from_numpy(rv_np).cuda()
    pidx=torch.from_numpy(pidx_np).cuda()
    idx=np.empty((len(qemb),100),dtype=np.int32)
    scores=np.empty((len(qemb),100),dtype=np.float32)
    t0=time.perf_counter()

    with torch.inference_mode():
        for s in range(0,len(qemb),32):
            q_np=np.array(qemb[s:s+32],dtype=np.float32,copy=True,order="C")
            q=torch.from_numpy(q_np).cuda()
            sim=q@rv.T
            ps=torch.full((len(q),len(docids)),-torch.inf,device="cuda",dtype=torch.float32)
            ps.scatter_reduce_(1,pidx.unsqueeze(0).expand(len(q),-1),sim,
                               reduce="amax",include_self=True)
            v,j=torch.topk(ps,k=100,dim=1,largest=True,sorted=True)
            idx[s:s+len(q)]=j.cpu().numpy().astype(np.int32)
            scores[s:s+len(q)]=v.cpu().numpy().astype(np.float32)
            if s%512==0 or s+len(q)==len(qemb):
                print(f"[{name}] search {min(s+32,len(qemb))}/{len(qemb)}",flush=True)

    np.save(ip,idx); np.save(sp,scores)
    obj={"contract_hash":ch,"contract":contract,"search_seconds":time.perf_counter()-t0}
    mp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    del rv,rv_np,pidx
    clear_cuda()
    return idx,scores,obj


def idx_to_rank(idx,docids):
    return [[docids[int(j)] for j in idx[i]] for i in range(len(idx))]


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


def eval_rank(rank,qids,golds,folds,stress):
    preds={q:rank[i][:5] for i,q in enumerate(qids)}
    qi={q:i for i,q in enumerate(qids)}
    present=set(qids)
    pf={}
    for f,ids in folds.items():
        pf[f]={
            "at5":official_metrics(preds,golds,ids),
            "recall_at100":float(np.mean([
                len(set(rank[qi[q]][:100])&set(golds[q]))/len(set(golds[q]))
                for q in ids
            ])),
        }
    st={}
    for n,ids in stress.items():
        use=[q for q in ids if q in present]
        if use:
            st[n]={
                "n":len(use),
                "recall_at5":official_metrics(preds,golds,use)["recall_at_5"],
                "recall_at100":float(np.mean([
                    len(set(rank[qi[q]][:100])&set(golds[q]))/len(set(golds[q]))
                    for q in use
                ])),
            }
    return {"official_at5":official_metrics(preds,golds,qids),
            "curve":curve(rank,qids,golds),"per_fold":pf,"stress":st}


def load_fixed_rankings(docids,qids):
    paths={
        "lal_b4":B4/"vnlegal_lal/parent_top100_idx.npy",
        "aiteam_coarse_1024":C1/"coarse_pack_1024/parent_top100_idx.npy",
        "aiteam_atomic_split_2048":C1/"atomic_split_2048/parent_top100_idx.npy",
        "bm25":A2/"bm25_idx.npy",
    }
    out={}
    for n,p in paths.items():
        if not p.exists(): raise FileNotFoundError(p)
        a=np.load(p,mmap_mode="r")
        if a.shape!=(len(qids),100): raise RuntimeError(f"{n}: {a.shape}")
        out[n]=idx_to_rank(a,docids)
    return out


def union_oracle(ranks,names,depth,qids,golds):
    vals=[]; full=0; sizes=[]; hit=tot=0
    for i,q in enumerate(qids):
        p=set()
        for n in names: p.update(ranks[n][i][:depth])
        g=set(golds[q]); h=len(p&g)
        vals.append(h/len(g)); full+=h==len(g); sizes.append(len(p)); hit+=h; tot+=len(g)
    return {"sources":list(names),"per_source_depth":depth,
            "macro_oracle_recall":float(np.mean(vals)),
            "full_gold_coverage_rate":full/len(qids),
            "mean_unique_pool_size":float(np.mean(sizes)),
            "gold_occurrence_coverage":hit/tot}


def compare(base,cand,qids,golds):
    aqr=bqr=ag=bg=0; jac=[]
    for i,q in enumerate(qids):
        A=set(base[i]); B=set(cand[i]); G=set(golds[q])
        ao=(A&G)-B; bo=(B&G)-A
        aqr+=bool(ao); bqr+=bool(bo); ag+=len(ao); bg+=len(bo)
        jac.append(len(A&B)/len(A|B))
    return {"baseline_rescue_queries":int(aqr),"candidate_rescue_queries":int(bqr),
            "baseline_unique_gold_occurrences":int(ag),
            "candidate_unique_gold_occurrences":int(bg),
            "mean_top100_jaccard":float(np.mean(jac))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--representations",default="all")
    args=ap.parse_args()
    selected=list(REPS) if args.representations=="all" else [
        x.strip() for x in args.representations.split(",") if x.strip()
    ]
    bad=[x for x in selected if x not in REPS]
    if bad: raise ValueError(f"Unknown representations: {bad}")

    CACHE.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)

    manifest=load_model_contract()
    geom_sha=base_geometry_contract()
    tok=load_tokenizer_untruncated()
    qids,golds,folds,stress=load_eval()
    docids=doc_universe()
    qemb,qmeta=load_query_embeddings(qids,manifest)
    ranks=load_fixed_rankings(docids,qids)

    baseline_metrics=eval_rank(ranks["lal_b4"],qids,golds,folds,stress)
    results={}; failures={}

    for name in selected:
        try:
            print(f"\n=== {name} ===",flush=True)
            spec=REPS[name]
            data,repmeta=materialize_representation(name,spec,tok,geom_sha,manifest)
            emb,chunk_docs,emeta=embed_rep(name,spec,data,repmeta,manifest)
            idx,scores,rmeta=exact_parent_max_search(
                name,emb,qemb,chunk_docs,docids,emeta["contract_hash"]
            )
            rank=idx_to_rank(idx,docids)
            ranks[name]=rank
            metrics=eval_rank(rank,qids,golds,folds,stress)

            # Practical core: new LAL + AITeam coarse1024 + BM25.
            core={}
            dual={}
            for d in CORE_DEPTHS:
                core[str(d)]=union_oracle(
                    ranks,(name,"aiteam_coarse_1024","bm25"),d,qids,golds
                )
                dual[str(d)]=union_oracle(
                    ranks,(name,"aiteam_atomic_split_2048",
                           "aiteam_coarse_1024","bm25"),d,qids,golds
                )

            obj={
                "representation":name,"spec":spec,
                "representation_geometry":repmeta,
                "embedding_cache":{
                    "completed":emeta["completed"],"total":emeta["total"],
                    "current_batch_size":emeta.get("current_batch_size"),
                    "oom_reductions":emeta.get("oom_reductions",[]),
                },
                "metrics":metrics,
                "vs_lal_b4_at100":compare(ranks["lal_b4"],rank,qids,golds),
                "core_union_with_aiteam_coarse1024_bm25":core,
                "dual_aiteam_core_union":dual,
            }
            results[name]=obj
            (OUT/f"{name}.json").write_text(
                json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
            )
            print(json.dumps({
                "representation":name,
                "R5":metrics["official_at5"]["recall_at_5"],
                "R100":metrics["curve"]["100"]["macro_recall"],
                "core_R100":core["100"]["macro_oracle_recall"],
                "dual_core_R100":dual["100"]["macro_oracle_recall"],
                "vectors":repmeta["vector_count"],
            },indent=2),flush=True)
        except Exception as exc:
            failures[name]={"error_type":type(exc).__name__,
                            "error":str(exc),"traceback":traceback.format_exc()}
            print(json.dumps({"representation":name,"status":"FAIL","error":str(exc)},indent=2),flush=True)
            clear_cuda()

    baseline_core={
        str(d):union_oracle(ranks,("lal_b4","aiteam_coarse_1024","bm25"),d,qids,golds)
        for d in CORE_DEPTHS
    }
    baseline_dual={
        str(d):union_oracle(
            ranks,("lal_b4","aiteam_atomic_split_2048","aiteam_coarse_1024","bm25"),
            d,qids,golds
        ) for d in CORE_DEPTHS
    }

    summary={
        "schema_version":"dsc2026.endgame.stage02c2_lal_representation_screen.v1",
        "status":"COMPLETE" if not failures else "PARTIAL_FAIL",
        "claim_boundary":"LAL document-representation screen; AITeam/BM25 fixed",
        "model":{"id":manifest["model_id"],"sha":manifest["resolved_revision_sha"],
                 "parameter_count":manifest["parameter_count"]},
        "baseline":{"representation":"LAL B4 atomic truncate 2048",
                    "metrics":baseline_metrics,
                    "core_union_with_aiteam_coarse1024_bm25":baseline_core,
                    "dual_aiteam_core_union":baseline_dual},
        "representations":results,"failures":failures,
    }
    (OUT/"REPRESENTATION_SCREEN.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    rows=[]
    b5=baseline_metrics["official_at5"]["recall_at_5"]
    b100=baseline_metrics["curve"]["100"]["macro_recall"]
    bc=baseline_core["100"]["macro_oracle_recall"]
    bd=baseline_dual["100"]["macro_oracle_recall"]
    rows.append({"representation":"lal_b4_atomic_truncate_2048",
                 "vectors":179737,"R5":b5,"R100":b100,
                 "core_R100":bc,"dual_core_R100":bd,
                 "delta_R5":0.0,"delta_R100":0.0,
                 "delta_core_R100":0.0,"delta_dual_core_R100":0.0})
    for n,r in results.items():
        m=r["metrics"]
        c=r["core_union_with_aiteam_coarse1024_bm25"]["100"]["macro_oracle_recall"]
        d=r["dual_aiteam_core_union"]["100"]["macro_oracle_recall"]
        rows.append({"representation":n,
                     "vectors":r["representation_geometry"]["vector_count"],
                     "R5":m["official_at5"]["recall_at_5"],
                     "R100":m["curve"]["100"]["macro_recall"],
                     "core_R100":c,"dual_core_R100":d,
                     "delta_R5":m["official_at5"]["recall_at_5"]-b5,
                     "delta_R100":m["curve"]["100"]["macro_recall"]-b100,
                     "delta_core_R100":c-bc,
                     "delta_dual_core_R100":d-bd})
    with (OUT/"SUMMARY.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines=["# Stage 02C2 — VNLegal-LAL Representation Screen","",
           f"Status: **{summary['status']}**","",
           "| Representation | Vectors | R@5 | R@100 | +AIT coarse1024+BM25 oracle@100 | +dual AIT+BM25 oracle@100 |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['representation']} | {r['vectors']} | {r['R5']:.6f} | "
                     f"{r['R100']:.6f} | {r['core_R100']:.6f} | {r['dual_core_R100']:.6f} |")
    lines += ["","No representation is auto-promoted.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print(json.dumps({"status":summary["status"],
                      "completed":list(results),"failed":list(failures),
                      "out":str(OUT)},ensure_ascii=False,indent=2))
    raise SystemExit(0 if not failures else 2)


if __name__=="__main__":
    main()

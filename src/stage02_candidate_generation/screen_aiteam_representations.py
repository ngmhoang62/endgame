#!/usr/bin/env python
"""Stage 02C1 — AITeam-v1 representation retrieval screen.

Purpose
-------
Isolate document representation while holding the encoder fixed.

Existing baseline (read-only, never rewritten here):
- Stage02B4 AITeam-v1 moderate atomic regions, truncate at 2048.

New representations:
- atomic_split_2048:
    preserve every moderate region; split only an oversized region into
    non-overlapping tokenizer slices.
- coarse_pack_1024:
    greedily pack consecutive moderate regions within the same parent;
    split an oversized region; no overlap.
- coarse_pack_2048:
    same algorithm using the full 2048 input contract.

Scientific controls:
- same AITeamVN/Vietnamese_Embedding v1 SHA;
- same 6991 queries and B4 query embeddings;
- no metadata/title prefix;
- same exact parent aggregation: max cosine over child vectors;
- no overlap;
- literal document IDs;
- LAL B4 and BM25 Stage02A rankings are read-only complements.

Safety:
- all new runtime assets live under cache/stage02c1_aiteam_representation/;
- existing B4/BM25 caches are never opened writable;
- new representation/embedding/ranking caches FAIL CLOSED on contract mismatch;
- representation files are built to a temporary path then atomically renamed.

Run all three new representations:
    python src/stage02_candidate_generation/screen_aiteam_representations.py

Run selected:
    python src/stage02_candidate_generation/screen_aiteam_representations.py \
      --representations coarse_pack_1024,coarse_pack_2048
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.evaluation import official_metrics

HF_ROOT = ROOT/"cache/huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT/"hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT/"modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT/"cache/sentence_transformers")

MODEL = ROOT/"models/retrievers_vi/aiteamvn-vietnamese-embedding"
MODEL_MANIFEST = ROOT/"reports/stage02b3_vietnamese_model_materialization/aiteamvn_vietnamese_embedding/MODEL_MANIFEST.json"
BASE_GEOM = ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"
BASE_GEOM_REPORT = ROOT/"reports/stage02b0_structure_geometry/STRUCTURE_GEOMETRY.json"
RAW = ROOT/"data/official_v1"
EVAL = ROOT/"data/evaluation_v2"

B4 = ROOT/"cache/stage02b4_vi_screen"
B4_RESULT = ROOT/"reports/stage02b4_vi_screen/aiteamvn_v1/RESULT.json"
A2 = ROOT/"cache/stage02a_parent_anchor"

CACHE = ROOT/"cache/stage02c1_aiteam_representation"
OUT = ROOT/"reports/stage02c1_aiteam_representation"

TOPK = 100
KS = (1,5,10,20,50,100)
CORE_DEPTHS = (20,50,100)
REP_ALGO_VERSION = "stage02c1.representation.v1"

REPS = {
    "atomic_split_2048": {
        "mode":"atomic_split",
        "max_input":2048,
        "batch_size":4,
    },
    "coarse_pack_1024": {
        "mode":"coarse_pack",
        "max_input":1024,
        "batch_size":8,
    },
    "coarse_pack_2048": {
        "mode":"coarse_pack",
        "max_input":2048,
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
        raise RuntimeError("AITeam-v1 model manifest is not PASS")
    if m.get("resolved_revision_sha")!="dea33aa1ab339f38d66ae0a40e6c40e0a9249568":
        raise RuntimeError("AITeam-v1 SHA drift")
    return m


def load_tokenizer_untruncated():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(
        str(MODEL),local_files_only=True,use_fast=True
    )
    if not hasattr(tok,"backend_tokenizer"):
        raise RuntimeError("Fast tokenizer required")
    tok.backend_tokenizer.no_truncation()
    probe=("Điều 49 quy định quyền và nghĩa vụ của người lao động. "*500)
    n=len(tok.backend_tokenizer.encode(probe,add_special_tokens=False).ids)
    if n<=2048:
        raise RuntimeError(f"Tokenizer no_truncation probe failed: {n}")
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
                    "start":int(x["start"]),
                    "end":int(x["end"]),
                }


def decode_token_slices(tok, text, budget):
    """Split one oversized region into non-overlapping tokenizer-ID slices.

    The source token stream is partitioned exactly. Each decoded slice is then
    re-tokenized and validated <= budget before being emitted.
    """
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
            # Extremely defensive backoff for boundary re-tokenization.
            hi=len(source_ids)
            while hi>1:
                hi-=1
                chunk=tok.backend_tokenizer.decode(source_ids[:hi],skip_special_tokens=True)
                actual=len(tok.backend_tokenizer.encode(chunk,add_special_tokens=False).ids)
                if actual<=budget:
                    break
            if actual>budget:
                raise RuntimeError("Could not back off decoded tokenizer slice below budget")
            # The rare tail IDs not represented by this shortened slice must not be lost.
            # Recurse on their decoded text.
            out.append((chunk,hi,actual))
            rem=tok.backend_tokenizer.decode(source_ids[hi:],skip_special_tokens=True)
            if rem.strip():
                for t,srcn,actn in decode_token_slices(tok,rem,budget):
                    out.append((t,srcn,actn))
        else:
            out.append((chunk,len(source_ids),actual))
    return out


def rep_contract(name,spec,geom_sha,model_manifest,specials):
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
        "model_sha":model_manifest["resolved_revision_sha"],
        "tokenizer_from_model_snapshot":True,
    }


def emit_row(out,row,stats):
    out.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    stats["vectors"]+=1
    stats["doc_counts"][row["doc_id"]]+=1
    stats["actual_tokens"]+=int(row["content_tokens"])
    stats["source_tokens"]+=int(row["source_token_count"])
    stats["max_content_tokens"]=max(stats["max_content_tokens"],int(row["content_tokens"]))


def materialize_representation(name,spec,tok,geom_sha,model_manifest):
    repdir=CACHE/name
    repdir.mkdir(parents=True,exist_ok=True)
    data=repdir/"chunks.jsonl"
    meta=repdir/"representation.json"
    specials=int(tok.num_special_tokens_to_add(pair=False))
    contract=rep_contract(name,spec,geom_sha,model_manifest,specials)
    ch=stable_hash(contract)

    if data.exists() != meta.exists():
        raise RuntimeError(
            f"[{name}] incomplete representation cache pair; REFUSING TO OVERWRITE"
        )
    if data.exists():
        old=rj(meta)
        if old.get("contract_hash")!=ch:
            raise RuntimeError(
                f"[{name}] representation contract mismatch; REFUSING TO OVERWRITE"
            )
        if old.get("status")!="PASS":
            raise RuntimeError(
                f"[{name}] representation metadata not PASS; REFUSING TO OVERWRITE"
            )
        if sha256_file(data)!=old.get("data_sha256"):
            raise RuntimeError(f"[{name}] chunks.jsonl SHA mismatch")
        print(f"[{name}] representation cache hit vectors={old['vector_count']}",flush=True)
        return data,old

    tmp=data.with_suffix(".jsonl.tmp")
    if tmp.exists():
        raise RuntimeError(f"[{name}] stale temp file exists: {tmp}")

    budget=spec["max_input"]-specials
    stats={
        "vectors":0,"doc_counts":defaultdict(int),
        "source_tokens":0,"actual_tokens":0,"max_content_tokens":0,
        "base_regions":0,"oversized_base_regions":0,
        "split_pieces":0,"packed_base_regions":0,
    }
    started=time.perf_counter()

    try:
        with tmp.open("w",encoding="utf-8",newline="\n") as out:
            if spec["mode"]=="atomic_split":
                for i,r in enumerate(iter_base_regions(),1):
                    stats["base_regions"]+=1
                    enc=tok.backend_tokenizer.encode(r["raw_text"],add_special_tokens=False)
                    n=len(enc.ids)
                    if n<=budget:
                        row={
                            "chunk_id":f"{name}:{stats['vectors']:09d}",
                            "doc_id":r["doc_id"],"text":r["raw_text"],
                            "source_region_ids":[r["region_id"]],
                            "source_kinds":[r["kind"]],
                            "source_token_count":n,"content_tokens":n,
                            "piece_of_oversized_region":False,
                        }
                        emit_row(out,row,stats)
                    else:
                        stats["oversized_base_regions"]+=1
                        pieces=decode_token_slices(tok,r["raw_text"],budget)
                        stats["split_pieces"]+=len(pieces)
                        for pi,(txt,srcn,actn) in enumerate(pieces):
                            row={
                                "chunk_id":f"{name}:{stats['vectors']:09d}",
                                "doc_id":r["doc_id"],"text":txt,
                                "source_region_ids":[r["region_id"]],
                                "source_kinds":[r["kind"]],
                                "source_token_count":srcn,"content_tokens":actn,
                                "piece_of_oversized_region":True,
                                "piece_index":pi,
                            }
                            emit_row(out,row,stats)
                    if i%5000==0 or i==179737:
                        print(f"[{name}] materialize base {i}/179737 vectors={stats['vectors']}",flush=True)

            elif spec["mode"]=="coarse_pack":
                current_doc=None
                current_text=""
                current_ids=[]
                current_kinds=[]
                current_source_tokens=0

                def flush_pack():
                    nonlocal current_text,current_ids,current_kinds,current_source_tokens
                    if not current_text:
                        return
                    actual=len(tok.backend_tokenizer.encode(
                        current_text,add_special_tokens=False
                    ).ids)
                    if actual>budget:
                        raise RuntimeError(
                            f"{name}: packed chunk exceeds budget {actual}>{budget}"
                        )
                    row={
                        "chunk_id":f"{name}:{stats['vectors']:09d}",
                        "doc_id":current_doc,"text":current_text,
                        "source_region_ids":list(current_ids),
                        "source_kinds":list(current_kinds),
                        "source_token_count":current_source_tokens,
                        "content_tokens":actual,
                        "piece_of_oversized_region":False,
                    }
                    emit_row(out,row,stats)
                    current_text=""; current_ids=[]; current_kinds=[]; current_source_tokens=0

                for i,r in enumerate(iter_base_regions(),1):
                    stats["base_regions"]+=1
                    if current_doc is None:
                        current_doc=r["doc_id"]
                    elif r["doc_id"]!=current_doc:
                        flush_pack()
                        current_doc=r["doc_id"]

                    n=len(tok.backend_tokenizer.encode(
                        r["raw_text"],add_special_tokens=False
                    ).ids)

                    if n>budget:
                        flush_pack()
                        stats["oversized_base_regions"]+=1
                        pieces=decode_token_slices(tok,r["raw_text"],budget)
                        stats["split_pieces"]+=len(pieces)
                        for pi,(txt,srcn,actn) in enumerate(pieces):
                            row={
                                "chunk_id":f"{name}:{stats['vectors']:09d}",
                                "doc_id":r["doc_id"],"text":txt,
                                "source_region_ids":[r["region_id"]],
                                "source_kinds":[r["kind"]],
                                "source_token_count":srcn,"content_tokens":actn,
                                "piece_of_oversized_region":True,
                                "piece_index":pi,
                            }
                            emit_row(out,row,stats)
                        continue

                    candidate=r["raw_text"] if not current_text else current_text+"\n\n"+r["raw_text"]
                    cn=len(tok.backend_tokenizer.encode(candidate,add_special_tokens=False).ids)
                    if cn<=budget:
                        if current_text:
                            stats["packed_base_regions"]+=1
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
                        print(f"[{name}] materialize base {i}/179737 vectors={stats['vectors']}",flush=True)
                flush_pack()
            else:
                raise RuntimeError(spec["mode"])

        tmp.replace(data)
    except Exception:
        # Do not silently delete a partial temp file: preserving it helps forensics.
        raise

    vpd=list(stats["doc_counts"].values())
    metadata={
        "schema_version":"dsc2026.endgame.stage02c1.representation.v1",
        "status":"PASS",
        "contract_hash":ch,
        "contract":contract,
        "vector_count":stats["vectors"],
        "base_region_count":stats["base_regions"],
        "oversized_base_region_count":stats["oversized_base_regions"],
        "split_piece_count":stats["split_pieces"],
        "additional_regions_packed_into_existing_chunk":stats["packed_base_regions"],
        "vectors_per_document":desc(vpd),
        "source_token_count_total":stats["source_tokens"],
        "actual_retokenized_content_tokens_total":stats["actual_tokens"],
        "max_content_tokens":stats["max_content_tokens"],
        "data_sha256":sha256_file(data),
        "materialize_seconds":time.perf_counter()-started,
    }
    meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[{name}] materialized vectors={stats['vectors']}",flush=True)
    return data,metadata


def load_rep_rows(data):
    texts=[]; docs=[]
    with data.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                texts.append(str(x["text"]))
                docs.append(str(x["doc_id"]))
    return texts,docs


def load_encoder(max_input):
    import torch
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(
        str(MODEL),device="cuda",local_files_only=True,
        trust_remote_code=False,model_kwargs={"dtype":torch.float16},
    )
    if int(m.get_embedding_dimension())!=1024:
        raise RuntimeError("AITeam embedding dimension drift")
    m.max_seq_length=int(max_input)
    return m


def validate_partial_embedding(arr,completed,total,dim,name):
    if arr.shape!=(total,dim):
        raise RuntimeError(
            f"[{name}] embedding shape mismatch {arr.shape}; REFUSING TO OVERWRITE"
        )
    if not (0<=completed<=total):
        raise RuntimeError(f"[{name}] bad completed={completed}")
    if completed:
        n=min(completed,1024)
        idx=np.linspace(0,completed-1,n,dtype=np.int64)
        x=np.asarray(arr[idx],dtype=np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError(f"[{name}] non-finite cached embeddings")
        err=float(np.max(np.abs(np.linalg.norm(x,axis=1)-1.0)))
        if err>5e-3:
            raise RuntimeError(f"[{name}] cached embedding norm error {err}")


def embed_rep(name,spec,data,repmeta,model_manifest):
    import torch
    texts,docs=load_rep_rows(data)
    repdir=CACHE/name
    npy=repdir/"embeddings.f32.npy"
    meta=repdir/"embeddings.json"
    contract={
        "schema":"stage02c1.aiteam_embeddings.v1",
        "model_sha":model_manifest["resolved_revision_sha"],
        "representation_contract_hash":repmeta["contract_hash"],
        "representation_data_sha":repmeta["data_sha256"],
        "max_seq_length":spec["max_input"],
        "normalize":"SentenceTransformer + explicit float32 L2",
        "dimension":1024,
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
        validate_partial_embedding(arr,completed,len(texts),1024,name)
        if completed==len(texts):
            print(f"[{name}] embedding cache hit {completed}/{len(texts)}",flush=True)
            return np.load(npy,mmap_mode="r"),docs,old
        print(f"[{name}] embedding RESUME {completed}/{len(texts)}",flush=True)
    else:
        arr=np.lib.format.open_memmap(
            npy,mode="w+",dtype=np.float32,shape=(len(texts),1024)
        )
        old={"contract_hash":ch,"completed":0,"total":len(texts),
             "dimension":1024,"status":"IN_PROGRESS","contract":contract}
        meta.write_text(json.dumps(old,indent=2)+"\n",encoding="utf-8")

    model=load_encoder(spec["max_input"])
    batch=int(spec["batch_size"])
    start_time=time.perf_counter()
    i=completed; last=completed; reductions=list(old.get("oom_reductions",[]))
    try:
        while i<len(texts):
            j=min(i+batch,len(texts))
            try:
                z=model.encode(
                    texts[i:j],convert_to_numpy=True,
                    normalize_embeddings=True,show_progress_bar=False
                )
                z=np.asarray(z,dtype=np.float32)
                n=np.linalg.norm(z,axis=1,keepdims=True)
                if np.any(n<=0) or not np.isfinite(n).all():
                    raise RuntimeError("bad embedding norm")
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
                        "elapsed_this_run_seconds":time.perf_counter()-start_time,
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


def load_query_embeddings(qids,model_manifest):
    npy=B4/"aiteamvn_v1/query_embeddings.f32.npy"
    meta=B4/"aiteamvn_v1/query_embeddings.json"
    if not npy.exists() or not meta.exists():
        raise FileNotFoundError("Stage02B4 AITeam query embedding cache missing")
    m=rj(meta)
    if int(m.get("completed",-1))!=len(qids) or int(m.get("dimension",-1))!=1024:
        raise RuntimeError("Stage02B4 AITeam query cache incomplete")
    z=np.load(npy,mmap_mode="r")
    if z.shape!=(len(qids),1024):
        raise RuntimeError(f"query shape drift {z.shape}")
    result=rj(B4_RESULT)
    if result.get("model_sha")!=model_manifest["resolved_revision_sha"]:
        raise RuntimeError("B4 query model SHA mismatch")
    return z,m


def doc_universe_from_base():
    docs=[]; seen=set()
    for r in iter_base_regions():
        d=r["doc_id"]
        if d not in seen:
            seen.add(d); docs.append(d)
    corpus=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): corpus.append(str(json.loads(line)["id"]))
    if docs!=corpus:
        raise RuntimeError("moderate geometry parent order != retrieval corpus order")
    return docs


def exact_parent_max_search(name,emb,qemb,chunk_docs,docids,contract_hash):
    import torch
    repdir=CACHE/name
    ip=repdir/"parent_top100_idx.npy"
    sp=repdir/"parent_top100_scores.npy"
    mp=repdir/"parent_top100.json"
    contract={
        "schema":"stage02c1.parent_amax.v1",
        "embedding_contract_hash":contract_hash,
        "aggregation":"exact_parent_amax",
        "topk":TOPK,
        "doc_order_sha":stable_hash(docids),
    }
    ch=stable_hash(contract)

    if any(p.exists() for p in (ip,sp,mp)):
        if not all(p.exists() for p in (ip,sp,mp)):
            raise RuntimeError(f"[{name}] incomplete ranking cache; REFUSING TO OVERWRITE")
        old=rj(mp)
        if old.get("contract_hash")!=ch:
            raise RuntimeError(f"[{name}] ranking contract mismatch; REFUSING TO OVERWRITE")
        idx=np.load(ip); scores=np.load(sp)
        if idx.shape!=(len(qemb),TOPK) or scores.shape!=idx.shape:
            raise RuntimeError(f"[{name}] ranking shape mismatch")
        print(f"[{name}] ranking cache hit",flush=True)
        return idx,scores,old

    d2i={d:i for i,d in enumerate(docids)}
    try:
        pidx_np=np.asarray([d2i[d] for d in chunk_docs],dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"unknown parent in representation: {exc}") from exc

    rv_np=np.array(emb,dtype=np.float32,copy=True,order="C")
    rv=torch.from_numpy(rv_np).cuda()
    pidx=torch.from_numpy(pidx_np).cuda()
    idx=np.empty((len(qemb),TOPK),dtype=np.int32)
    scores=np.empty((len(qemb),TOPK),dtype=np.float32)
    started=time.perf_counter()

    with torch.inference_mode():
        for s in range(0,len(qemb),32):
            q_np=np.array(qemb[s:s+32],dtype=np.float32,copy=True,order="C")
            q=torch.from_numpy(q_np).cuda()
            sim=q@rv.T
            ps=torch.full((len(q),len(docids)),-torch.inf,device="cuda",dtype=torch.float32)
            ps.scatter_reduce_(1,pidx.unsqueeze(0).expand(len(q),-1),sim,
                               reduce="amax",include_self=True)
            v,j=torch.topk(ps,k=TOPK,dim=1,largest=True,sorted=True)
            idx[s:s+len(q)]=j.cpu().numpy().astype(np.int32)
            scores[s:s+len(q)]=v.cpu().numpy().astype(np.float32)
            if s%512==0 or s+len(q)==len(qemb):
                print(f"[{name}] search {min(s+32,len(qemb))}/{len(qemb)}",flush=True)

    np.save(ip,idx); np.save(sp,scores)
    meta={"contract_hash":ch,"contract":contract,
          "search_seconds":time.perf_counter()-started}
    mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    del rv,rv_np,pidx
    clear_cuda()
    return idx,scores,meta


def load_eval():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    if len(qids)!=6991: raise RuntimeError("6991 eval contract drift")
    return qids,golds,folds,stress


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
    pset=set(qids)
    per_fold={}
    for f,ids in folds.items():
        per_fold[f]={
            "at5":official_metrics(preds,golds,ids),
            "recall_at100":float(np.mean([
                len(set(rank[qi[q]][:100])&set(golds[q]))/len(set(golds[q]))
                for q in ids
            ]))
        }
    st={}
    for n,ids in stress.items():
        use=[q for q in ids if q in pset]
        if use:
            st[n]={
                "n":len(use),
                "recall_at5":official_metrics(preds,golds,use)["recall_at_5"],
                "recall_at100":float(np.mean([
                    len(set(rank[qi[q]][:100])&set(golds[q]))/len(set(golds[q]))
                    for q in use
                ]))
            }
    return {"official_at5":official_metrics(preds,golds,qids),
            "curve":curve(rank,qids,golds),
            "per_fold":per_fold,"stress":st}


def load_existing_rankings(docids,qids):
    paths={
        "baseline_atomic_truncate_2048":B4/"aiteamvn_v1/parent_top100_idx.npy",
        "vnlegal_lal_b4":B4/"vnlegal_lal/parent_top100_idx.npy",
        "bm25":A2/"bm25_idx.npy",
    }
    out={}
    for k,p in paths.items():
        if not p.exists(): raise FileNotFoundError(p)
        a=np.load(p,mmap_mode="r")
        if a.shape!=(len(qids),100): raise RuntimeError(f"{k} ranking shape {a.shape}")
        out[k]=idx_to_rank(a,docids)
    return out


def union_oracle(ranks,names,depth,qids,golds):
    vals=[]; full=0; sizes=[]; hits=tot=0
    for i,q in enumerate(qids):
        p=set()
        for n in names: p.update(ranks[n][i][:depth])
        g=set(golds[q]); h=len(p&g)
        vals.append(h/len(g)); full+=h==len(g); sizes.append(len(p))
        hits+=h; tot+=len(g)
    return {
        "sources":list(names),"per_source_depth":depth,
        "macro_oracle_recall":float(np.mean(vals)),
        "full_gold_coverage_rate":full/len(qids),
        "mean_unique_pool_size":float(np.mean(sizes)),
        "gold_occurrence_coverage":hits/tot,
    }


def compare_to_baseline(base,cand,qids,golds):
    a_resc=b_resc=ag=bg=0; jac=[]
    for i,q in enumerate(qids):
        A=set(base[i][:100]); B=set(cand[i][:100]); G=set(golds[q])
        a=(A&G); b=(B&G)
        a_resc+=int(len(a-b)>0); b_resc+=int(len(b-a)>0)
        ag+=len(a-b); bg+=len(b-a)
        jac.append(len(A&B)/len(A|B))
    return {
        "baseline_rescues_candidate_queries":a_resc,
        "candidate_rescues_baseline_queries":b_resc,
        "baseline_unique_gold_occurrences":ag,
        "candidate_unique_gold_occurrences":bg,
        "mean_top100_jaccard":float(np.mean(jac)),
    }


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

    model_manifest=load_model_contract()
    geom_sha=base_geometry_contract()
    tok=load_tokenizer_untruncated()
    qids,golds,folds,stress=load_eval()
    docids=doc_universe_from_base()
    qemb,qmeta=load_query_embeddings(qids,model_manifest)
    ranks=load_existing_rankings(docids,qids)

    results={}
    rep_reports={}
    failures={}

    for name in selected:
        try:
            print(f"\n=== {name} ===",flush=True)
            spec=REPS[name]
            data,repmeta=materialize_representation(
                name,spec,tok,geom_sha,model_manifest
            )
            emb,chunk_docs,emeta=embed_rep(
                name,spec,data,repmeta,model_manifest
            )
            idx,scores,rmeta=exact_parent_max_search(
                name,emb,qemb,chunk_docs,docids,emeta["contract_hash"]
            )
            rank=idx_to_rank(idx,docids)
            ranks[name]=rank
            metrics=eval_rank(rank,qids,golds,folds,stress)
            comp=compare_to_baseline(
                ranks["baseline_atomic_truncate_2048"],rank,qids,golds
            )
            core={}
            for depth in CORE_DEPTHS:
                core[str(depth)]=union_oracle(
                    ranks,(name,"vnlegal_lal_b4","bm25"),depth,qids,golds
                )
            result={
                "representation":name,
                "spec":spec,
                "representation_geometry":repmeta,
                "embedding_cache":{
                    "completed":emeta["completed"],
                    "total":emeta["total"],
                    "current_batch_size":emeta.get("current_batch_size"),
                    "oom_reductions":emeta.get("oom_reductions",[]),
                },
                "metrics":metrics,
                "vs_b4_baseline_at100":comp,
                "core_union_with_lal_bm25":core,
            }
            results[name]=result
            rep_reports[name]={
                "vector_count":repmeta["vector_count"],
                "vectors_per_document":repmeta["vectors_per_document"],
                "source_token_count_total":repmeta["source_token_count_total"],
                "actual_retokenized_content_tokens_total":repmeta["actual_retokenized_content_tokens_total"],
                "max_content_tokens":repmeta["max_content_tokens"],
            }
            (OUT/f"{name}.json").write_text(
                json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
            )
            print(json.dumps({
                "representation":name,
                "R5":metrics["official_at5"]["recall_at_5"],
                "R20":metrics["curve"]["20"]["macro_recall"],
                "R100":metrics["curve"]["100"]["macro_recall"],
                "core_R100":core["100"]["macro_oracle_recall"],
                "vectors":repmeta["vector_count"],
            },indent=2),flush=True)
        except Exception as exc:
            failures[name]={
                "error_type":type(exc).__name__,
                "error":str(exc),
                "traceback":traceback.format_exc(),
            }
            print(json.dumps({"representation":name,"status":"FAIL","error":str(exc)},indent=2),flush=True)
            clear_cuda()

    baseline_metrics=eval_rank(
        ranks["baseline_atomic_truncate_2048"],qids,golds,folds,stress
    )
    baseline_core={
        str(d):union_oracle(
            ranks,("baseline_atomic_truncate_2048","vnlegal_lal_b4","bm25"),
            d,qids,golds
        )
        for d in CORE_DEPTHS
    }

    summary={
        "schema_version":"dsc2026.endgame.stage02c1_aiteam_representation_screen.v1",
        "status":"COMPLETE" if not failures else "PARTIAL_FAIL",
        "claim_boundary":"AITeam-v1 document-representation screen; LAL/BM25 kept fixed",
        "model":{
            "id":model_manifest["model_id"],
            "sha":model_manifest["resolved_revision_sha"],
            "parameter_count":model_manifest["parameter_count"],
        },
        "baseline":{
            "representation":"B4 atomic truncate 2048",
            "metrics":baseline_metrics,
            "core_union_with_lal_bm25":baseline_core,
        },
        "representations":results,
        "failures":failures,
    }
    (OUT/"REPRESENTATION_SCREEN.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    rows=[]
    b5=baseline_metrics["official_at5"]["recall_at_5"]
    b100=baseline_metrics["curve"]["100"]["macro_recall"]
    bc100=baseline_core["100"]["macro_oracle_recall"]
    rows.append({
        "representation":"baseline_atomic_truncate_2048",
        "vectors":179737,"R5":b5,"R20":baseline_metrics["curve"]["20"]["macro_recall"],
        "R100":b100,"core_union_R100":bc100,"delta_R5_vs_baseline":0.0,
        "delta_R100_vs_baseline":0.0,"delta_core_R100_vs_baseline":0.0,
    })
    for n,r in results.items():
        m=r["metrics"]
        rows.append({
            "representation":n,
            "vectors":r["representation_geometry"]["vector_count"],
            "R5":m["official_at5"]["recall_at_5"],
            "R20":m["curve"]["20"]["macro_recall"],
            "R100":m["curve"]["100"]["macro_recall"],
            "core_union_R100":r["core_union_with_lal_bm25"]["100"]["macro_oracle_recall"],
            "delta_R5_vs_baseline":m["official_at5"]["recall_at_5"]-b5,
            "delta_R100_vs_baseline":m["curve"]["100"]["macro_recall"]-b100,
            "delta_core_R100_vs_baseline":r["core_union_with_lal_bm25"]["100"]["macro_oracle_recall"]-bc100,
        })
    with (OUT/"SUMMARY.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines=[
        "# Stage 02C1 — AITeam-v1 Representation Screen","",
        f"Status: **{summary['status']}**","",
        "LAL and BM25 are frozen read-only complements; only AITeam document representation changes.","",
        "| Representation | Vectors | R@5 | R@20 | R@100 | AITeam+LAL+BM25 oracle@100 | Δ core |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['representation']} | {r['vectors']} | {r['R5']:.6f} | "
            f"{r['R20']:.6f} | {r['R100']:.6f} | {r['core_union_R100']:.6f} | "
            f"{r['delta_core_R100_vs_baseline']:+.6f} |"
        )
    lines += ["","No representation is auto-promoted; inspect head recall, acquisition depth, complementarity, and vector count together.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print(json.dumps({
        "status":summary["status"],
        "completed":list(results),
        "failed":list(failures),
        "report":str((OUT/"REPORT.md").resolve()),
    },ensure_ascii=False,indent=2))
    raise SystemExit(0 if not failures else 2)


if __name__=="__main__":
    main()

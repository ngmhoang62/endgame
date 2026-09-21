#!/usr/bin/env python
"""Stage 03B1 — AITeam Vietnamese reranker OOF benchmark.

Deadline-sprint design:
- rerank only the Stage03B0 fold-clean top-30 shortlist;
- historical passage contract: 2 lexical-selected windows/doc, max score;
- fresh pinned AITeamVN/Vietnamese_Reranker;
- resumable fail-closed score cache;
- evaluate selector-only, CE-only, fold-clean tuned rank/CE blend, and
  fold-clean rank+score+CE LogisticRegression;
- save a full-train CE-fusion model for immediate private materialization.

Run:
    python src/stage03_rerank/run_aiteam_reranker_oof.py

Resume after interruption by running the same command again.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
B0=ROOT/"cache/stage03b_reranker"
MODEL=ROOT/"models/rerankers/aiteamvn-vietnamese-reranker"
MANIFEST=ROOT/"reports/stage03b0_reranker_materialization/MODEL_MANIFEST.json"

CACHE=ROOT/"cache/stage03b1_aiteam_reranker"
OUT=ROOT/"reports/stage03b1_aiteam_reranker"

SHORTLIST_DEPTH=30
PASSAGES_PER_DOC=2
WINDOW=220
OVERLAP=70
MAX_LENGTH=512
INITIAL_BATCH=8
QUERY_CHUNK=32
LR_C=.15
ALPHAS=[round(x/20,2) for x in range(21)]

SOURCE_PATHS={
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
SOURCE_NAMES=list(SOURCE_PATHS)

TOKEN_RE=re.compile(r"\w+",re.UNICODE)
SPACE_RE=re.compile(r"\S+",re.UNICODE)
STOPWORDS={
    "bị","các","có","của","cho","được","để","đến","đối","gì",
    "hay","khi","không","là","làm","một","nào","những","như",
    "phải","ra","sẽ","theo","thì","thế","trong","trên","từ",
    "và","về","với","việc","bao","nhiêu","người","quy","định",
}


def rj(p:Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def top_passages(question,text,count=2,window=WINDOW,overlap=OVERLAP):
    """Exact historical lexical-window contract used by Huy reranker pipeline."""
    words=SPACE_RE.findall(text or "")
    if len(words)<=window+80:
        return [" ".join(words)]
    query_tokens=tokens(question)
    content={t for t in query_tokens if len(t)>=3 and t not in STOPWORDS}
    numbers={t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams={" ".join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)}
    header=" ".join(words[:70])
    scored=[]
    step=window-overlap
    for start in range(0,len(words),step):
        end=min(start+window,len(words))
        part_words=words[start:end]
        part=" ".join(part_words)
        normalized=tokens(part)
        token_set=set(normalized)
        norm_text=" ".join(normalized)
        coverage=sum(1.0+.20*min(normalized.count(t),3)
                     for t in content if t in token_set)
        numeric=3.0*sum(t in token_set for t in numbers)
        phrase=1.8*sum(p in norm_text for p in bigrams)
        density=(coverage+numeric+phrase)/math.sqrt(max(len(normalized),1))
        scored.append((density,coverage+numeric+phrase,-start,part))
        if end==len(words):
            break
    scored.sort(reverse=True)
    passages=[]
    for _,_,neg_start,part in scored:
        candidate=part if -neg_start<70 else header+"\n[ĐOẠN PHÙ HỢP]\n"+part
        if candidate not in passages:
            passages.append(candidate)
        if len(passages)>=count:
            break
    return passages


def load_world():
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds]
           for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    questions={q:str(train[q]["question"]) for q in qids}
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}

    docs=[]; passages=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                docs.append(str(x["id"]))
                passages.append(str(x["passage"]))
    if len(qids)!=6991 or len(docs)!=8512:
        raise RuntimeError("frozen population drift")
    return qids,questions,golds,folds,stress,docs,passages


def load_shortlist(nq):
    p=B0/"oof_rank_selector_top50_idx.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    a=np.load(p,mmap_mode="r")
    if a.shape!=(nq,50):
        raise RuntimeError(f"shortlist shape drift: {a.shape}")
    return np.asarray(a[:,:SHORTLIST_DEPTH],dtype=np.int32),sha256_file(p)


def load_sources(nq):
    out={}
    for n,(ip,sp) in SOURCE_PATHS.items():
        if not ip.exists() or not sp.exists():
            raise FileNotFoundError(f"{n}: {ip} / {sp}")
        i=np.load(ip,mmap_mode="r"); s=np.load(sp,mmap_mode="r")
        if i.shape!=(nq,100) or s.shape!=(nq,100):
            raise RuntimeError(f"{n} shape drift idx={i.shape} score={s.shape}")
        if not np.isfinite(s).all():
            raise RuntimeError(f"{n}: non-finite source score")
        out[n]=(i,s)
    return out


def load_model_manifest():
    m=rj(MANIFEST)
    if m.get("status")!="PASS":
        raise RuntimeError("reranker manifest not PASS")
    if m.get("resolved_revision_sha")!="f536976248403314225d7fdfdbc87f0e9516a54e":
        raise RuntimeError("reranker SHA drift")
    if int(m["contract"]["max_length"])!=512:
        raise RuntimeError("reranker max_length drift")
    return m


def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(
        MODEL,local_files_only=True,trust_remote_code=True
    )
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL,local_files_only=True,trust_remote_code=True,dtype=torch.float16
    ).eval().to("cuda")
    return tok,model


def forward_pairs(tok,model,pairs,batch):
    """Return logits with controlled dynamic batch size."""
    import torch
    vals=[]
    i=0
    current=batch
    while i<len(pairs):
        j=min(i+current,len(pairs))
        try:
            enc=tok(
                [a for a,_ in pairs[i:j]],
                [b for _,b in pairs[i:j]],
                max_length=MAX_LENGTH,truncation=True,padding=True,
                return_tensors="pt",
            )
            enc={k:v.to("cuda") for k,v in enc.items()}
            with torch.inference_mode():
                logits=model(**enc).logits
            if logits.ndim==2 and logits.shape[1]==1:
                v=logits[:,0]
            elif logits.ndim==2 and logits.shape[1]>=2:
                v=logits[:,-1]
            else:
                v=logits.reshape(-1)
            vals.extend(float(x) for x in v.float().cpu().tolist())
            i=j
        except torch.cuda.OutOfMemoryError:
            clear_cuda()
            if current<=1:
                raise
            current=max(1,current//2)
            print(f"[reranker] CUDA OOM -> batch={current}",flush=True)
    return vals,current


def score_contract(shortlist_sha,manifest):
    return {
        "schema":"stage03b1.aiteam_reranker_scores.v1",
        "model_sha":manifest["resolved_revision_sha"],
        "shortlist_top50_sha":shortlist_sha,
        "shortlist_depth":SHORTLIST_DEPTH,
        "max_length":MAX_LENGTH,
        "passages_per_doc":PASSAGES_PER_DOC,
        "passage_selector":{
            "window_words":WINDOW,"overlap_words":OVERLAP,
            "header_words":70,
            "historical_contract":"benchmark_jina_reranker_holdouts.top_passages",
        },
        "aggregation":"max_passage_logit_per_document",
    }


def open_score_cache(nq,contract):
    CACHE.mkdir(parents=True,exist_ok=True)
    sp=CACHE/"oof_top30_ce_scores.f32.npy"
    dp=CACHE/"oof_top30_done.u1.npy"
    mp=CACHE/"oof_top30_ce_scores.json"
    ch=stable_hash(contract)

    exists=[p.exists() for p in (sp,dp,mp)]
    if any(exists) and not all(exists):
        raise RuntimeError("partial reranker cache triplet exists; REFUSING TO OVERWRITE")

    if all(exists):
        meta=rj(mp)
        if meta.get("contract_hash")!=ch:
            raise RuntimeError("reranker cache contract mismatch; REFUSING TO OVERWRITE")
        scores=np.lib.format.open_memmap(sp,mode="r+")
        done=np.lib.format.open_memmap(dp,mode="r+")
        if scores.shape!=(nq,SHORTLIST_DEPTH) or done.shape!=(nq,):
            raise RuntimeError("reranker cache shape mismatch")
        mask=np.asarray(done,dtype=bool)
        if mask.any() and not np.isfinite(np.asarray(scores[mask])).all():
            raise RuntimeError("non-finite completed reranker scores")
        print(f"[reranker] RESUME completed={int(mask.sum())}/{nq}",flush=True)
        return scores,done,meta,sp,dp,mp

    scores=np.lib.format.open_memmap(
        sp,mode="w+",dtype=np.float32,shape=(nq,SHORTLIST_DEPTH)
    )
    scores[:]=np.nan; scores.flush()
    done=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(nq,))
    done[:]=0; done.flush()
    meta={
        "contract_hash":ch,"contract":contract,
        "status":"IN_PROGRESS","completed_queries":0,
        "total_queries":nq,"current_batch":INITIAL_BATCH,
        "wall_seconds_accumulated":0.0,
    }
    mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return scores,done,meta,sp,dp,mp


def _prepare_pair_chunk(qis,qids,questions,shortlist,passages):
    """CPU producer: exact historical passage selection for one query chunk."""
    pairs=[]
    owners=[]  # (qi, candidate_position)
    for qi in qis:
        q=questions[qids[qi]]
        for pos,didx in enumerate(shortlist[qi]):
            ps=top_passages(q,passages[int(didx)],count=PASSAGES_PER_DOC)
            if not ps:
                ps=[""]
            for p in ps:
                owners.append((qi,pos))
                pairs.append((q,p))
    return qis,pairs,owners


def score_all(qids,questions,shortlist,passages,manifest,shortlist_sha):
    contract=score_contract(shortlist_sha,manifest)
    scores,done,meta,sp,dp,mp=open_score_cache(len(qids),contract)
    pending=[i for i in range(len(qids)) if int(done[i])==0]
    if not pending:
        print("[reranker] score cache complete",flush=True)
        return np.asarray(scores,dtype=np.float32),rj(mp)

    tok,model=load_model()
    batch=int(meta.get("current_batch",INITIAL_BATCH))
    started=time.perf_counter()
    newly=0

    # CPU passage selection for chunk N+1 is overlapped with GPU scoring of chunk N.
    # Scheduling parameters are intentionally NOT part of the scientific cache contract.
    chunks=[pending[st:st+QUERY_CHUNK] for st in range(0,len(pending),QUERY_CHUNK)]
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="passage-prefetch")
    future=executor.submit(
        _prepare_pair_chunk,chunks[0],qids,questions,shortlist,passages
    )

    try:
        for ci in range(len(chunks)):
            wait0=time.perf_counter()
            qis,pairs,owners=future.result()
            prep_wait=time.perf_counter()-wait0

            # Start CPU preparation of the next chunk BEFORE GPU scoring current chunk.
            if ci+1<len(chunks):
                future=executor.submit(
                    _prepare_pair_chunk,chunks[ci+1],qids,questions,shortlist,passages
                )

            gpu0=time.perf_counter()
            vals,batch=forward_pairs(tok,model,pairs,batch)
            gpu_wall=time.perf_counter()-gpu0
            if len(vals)!=len(owners):
                raise RuntimeError("reranker pair-score cardinality drift")

            tmp={qi:np.full(SHORTLIST_DEPTH,-np.inf,dtype=np.float32) for qi in qis}
            for (qi,pos),v in zip(owners,vals):
                if v>tmp[qi][pos]:
                    tmp[qi][pos]=float(v)

            for qi in qis:
                if not np.isfinite(tmp[qi]).all():
                    raise RuntimeError(f"non-finite aggregate q={qids[qi]}")
                scores[qi]=tmp[qi]
                done[qi]=1
                newly+=1
            scores.flush(); done.flush()

            completed=int(np.asarray(done,dtype=np.uint8).sum())
            # Save after every chunk: at QUERY_CHUNK=32, interruption loses at most
            # the in-flight chunk and all already completed chunks are resumable.
            meta={
                "contract_hash":stable_hash(contract),"contract":contract,
                "status":"PASS" if completed==len(qids) else "IN_PROGRESS",
                "completed_queries":completed,"total_queries":len(qids),
                "current_batch":batch,
                "scheduler":{
                    "query_chunk":QUERY_CHUNK,
                    "cpu_prefetch_workers":1,
                    "last_uncovered_prep_wait_seconds":prep_wait,
                    "last_gpu_score_seconds":gpu_wall,
                },
                "wall_seconds_accumulated":float(
                    meta.get("wall_seconds_accumulated",0.0)
                    + (time.perf_counter()-started)
                ),
            }
            mp.write_text(
                json.dumps(meta,ensure_ascii=False,indent=2)+"\n",
                encoding="utf-8"
            )

            if completed%64<len(qis) or ci+1==len(chunks):
                rate=newly/max(time.perf_counter()-started,1e-9)
                print(
                    f"[reranker] {completed}/{len(qids)} batch={batch} "
                    f"new_rate={rate:.2f} q/s prep_wait={prep_wait:.2f}s "
                    f"gpu={gpu_wall:.2f}s pairs={len(pairs)}",
                    flush=True
                )
    finally:
        executor.shutdown(wait=False,cancel_futures=True)
        del model,tok
        clear_cuda()

    return np.asarray(scores,dtype=np.float32),rj(mp)

def rank_by_scores(shortlist,scores):
    out=np.empty_like(shortlist)
    for i in range(len(shortlist)):
        order=np.lexsort((shortlist[i],-scores[i]))
        out[i]=shortlist[i,order]
    return out


def eval_rank(rank,qids,golds,docs,folds,stress):
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    overall=official_metrics(preds,golds,qids)
    pf={f:official_metrics(preds,golds,ids) for f,ids in folds.items()}
    st={}
    for n,ids in stress.items():
        use=[q for q in ids if q in preds]
        if use: st[n]=official_metrics(preds,golds,use)
    return {"overall":overall,"per_fold":pf,"stress":st}


def macro_r5_indices(rank,query_indices,qids,golds,docs):
    vals=[]
    for qi in query_indices:
        g=set(golds[qids[qi]])
        p={docs[int(x)] for x in rank[qi,:5]}
        vals.append(len(p&g)/len(g))
    return float(np.mean(vals))


def zrows(a):
    a=np.asarray(a,dtype=np.float64)
    mu=a.mean(axis=1,keepdims=True)
    sd=a.std(axis=1,keepdims=True)
    sd=np.where(sd<1e-8,1.0,sd)
    return ((a-mu)/sd).astype(np.float32)


def nested_blend(shortlist,ce_scores,qids,golds,docs,folds):
    n=len(qids)
    ranks=np.arange(1,SHORTLIST_DEPTH+1,dtype=np.float32)[None,:]
    rank_raw=np.broadcast_to(-ranks,(n,SHORTLIST_DEPTH))
    rank_z=zrows(rank_raw)
    ce_z=zrows(ce_scores)
    q2i={q:i for i,q in enumerate(qids)}
    out=np.empty_like(shortlist)
    selected={}

    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        held_set=set(map(int,held))
        train=np.asarray([i for i in range(n) if i not in held_set],dtype=np.int32)

        best=None
        for alpha in ALPHAS:
            blend=(1-alpha)*rank_z + alpha*ce_z
            rr=np.empty_like(shortlist)
            for qi in train:
                order=np.lexsort((shortlist[qi],-blend[qi]))
                rr[qi]=shortlist[qi,order]
            score=macro_r5_indices(rr,train,qids,golds,docs)
            key=(score,-alpha)  # stable tie-break: less CE weight
            if best is None or key>best[0]:
                best=(key,alpha,score)
        alpha=best[1]
        selected[fn]={"alpha":alpha,"train_macro_recall_at5":best[2]}
        blend=(1-alpha)*rank_z + alpha*ce_z
        for qi in held:
            order=np.lexsort((shortlist[qi],-blend[qi]))
            out[qi]=shortlist[qi,order]

    return out,selected


def build_source_ce_features(shortlist,ce_scores,sources):
    n=len(shortlist)
    # 6 sources * 5 + aggregate 9 + selector 2 + CE 4 = 45D
    names=[]
    for s in SOURCE_NAMES:
        names += [s+"__present",s+"__rr10",s+"__rank50",
                  s+"__z",s+"__gapz"]
    names += ["source_count","best_rr10","mean_rr10","rrf60",
              "min_rank50","mean_rank50","count_top5","count_top10","count_top20",
              "selector_rr10","selector_rank_norm30",
              "ce_raw","ce_z","ce_gap_z","ce_rr10"]

    X=np.empty((n*SHORTLIST_DEPTH,len(names)),dtype=np.float32)
    cq=np.repeat(np.arange(n,dtype=np.int32),SHORTLIST_DEPTH)
    cand=shortlist.reshape(-1).astype(np.int32,copy=True)
    ce_z=zrows(ce_scores)
    ce_order=np.empty_like(shortlist)
    ce_rank=np.empty_like(shortlist)
    for qi in range(n):
        order=np.lexsort((shortlist[qi],-ce_scores[qi]))
        ce_order[qi]=shortlist[qi,order]
        inv=np.empty(SHORTLIST_DEPTH,dtype=np.int32)
        inv[order]=np.arange(1,SHORTLIST_DEPTH+1,dtype=np.int32)
        ce_rank[qi]=inv

    pos=0
    for qi in range(n):
        rm={}; zm={}; gm={}
        for s in SOURCE_NAMES:
            idx,scr=sources[s]
            ids=np.asarray(idx[qi,:50],dtype=np.int32)
            s100=np.asarray(scr[qi,:],dtype=np.float64)
            mu=float(s100.mean()); sd=float(s100.std())
            if not np.isfinite(sd) or sd<1e-8: sd=1.0
            top=float(s100[0])
            rm[s]={int(d):r+1 for r,d in enumerate(ids)}
            zm[s]={int(d):float((float(scr[qi,r])-mu)/sd) for r,d in enumerate(ids)}
            gm[s]={int(d):float((top-float(scr[qi,r]))/sd) for r,d in enumerate(ids)}
        ce_sd=float(np.std(ce_scores[qi]))
        if ce_sd<1e-8: ce_sd=1.0
        ce_top=float(np.max(ce_scores[qi]))

        for sp,d0 in enumerate(shortlist[qi]):
            d=int(d0); f=[]; rrs=[]; rs=[]; c5=c10=c20=0; rrf=0.
            for s in SOURCE_NAMES:
                r=rm[s].get(d)
                if r is None:
                    f += [0.,0.,1.2,-3.,4.]
                else:
                    rr=1/(10+r)
                    f += [1.,rr,r/50.0,zm[s][d],gm[s][d]]
                    rrs.append(rr); rs.append(r); rrf+=1/(60+r)
                    c5+=r<=5; c10+=r<=10; c20+=r<=20
            cr=int(ce_rank[qi,sp])
            f += [
                float(len(rrs)),max(rrs) if rrs else 0.,
                float(np.mean(rrs)) if rrs else 0.,rrf,
                min(rs)/50.0 if rs else 1.2,
                float(np.mean(rs))/50.0 if rs else 1.2,
                float(c5),float(c10),float(c20),
                1.0/(10.0+sp+1),(sp+1)/SHORTLIST_DEPTH,
                float(ce_scores[qi,sp]),float(ce_z[qi,sp]),
                float((ce_top-ce_scores[qi,sp])/ce_sd),
                1.0/(10.0+cr),
            ]
            X[pos]=f; pos+=1
    return X,cq,cand,names


def lr_oof(shortlist,ce_scores,qids,golds,docs,folds,sources):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X,cq,cand,names=build_source_ce_features(shortlist,ce_scores,sources)
    d2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{d2i[d] for d in golds[q]} for q in qids]
    y=np.empty(len(cand),dtype=np.uint8)
    for j,(qi,d) in enumerate(zip(cq,cand)):
        y[j]=int(int(d) in gold_idx[int(qi)])

    q2i={q:i for i,q in enumerate(qids)}
    out=np.empty_like(shortlist)
    fold_meta={}

    def make_model():
        return Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(
                C=LR_C,class_weight="balanced",solver="lbfgs",
                max_iter=300,tol=1e-5)),
        ])

    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        hs=set(map(int,held))
        mask=np.asarray([int(q) not in hs for q in cq],dtype=bool)
        model=make_model()
        print(f"[CE-LR/{fn}] fit rows={int(mask.sum())} pos={int(y[mask].sum())}",flush=True)
        model.fit(X[mask],y[mask])
        for qi in held:
            a=qi*SHORTLIST_DEPTH; b=a+SHORTLIST_DEPTH
            p=model.predict_proba(X[a:b])[:,1]
            di=cand[a:b]
            order=np.lexsort((di,-p))
            out[qi]=di[order]
        fold_meta[fn]={"train_rows":int(mask.sum()),"train_positives":int(y[mask].sum())}

    full=make_model()
    full.fit(X,y)
    CACHE.mkdir(parents=True,exist_ok=True)
    joblib.dump(
        {"model":full,"feature_names":names,"source_names":SOURCE_NAMES,
         "shortlist_depth":SHORTLIST_DEPTH,"lr_C":LR_C},
        CACHE/"fulltrain_ce_fusion.joblib"
    )
    return out,fold_meta,names


def main():
    ap=argparse.ArgumentParser()
    ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True)
    qids,questions,golds,folds,stress,docs,passages=load_world()
    shortlist,shortlist_sha=load_shortlist(len(qids))
    sources=load_sources(len(qids))
    manifest=load_model_manifest()

    ce_scores,cache_meta=score_all(
        qids,questions,shortlist,passages,manifest,shortlist_sha
    )

    selector_rank=shortlist.copy()
    ce_rank=rank_by_scores(shortlist,ce_scores)
    blend_rank,alphas=nested_blend(
        shortlist,ce_scores,qids,golds,docs,folds
    )
    lr_rank,lr_meta,feature_names=lr_oof(
        shortlist,ce_scores,qids,golds,docs,folds,sources
    )

    methods={
        "selector_top30_order":eval_rank(
            selector_rank,qids,golds,docs,folds,stress),
        "ce_only":eval_rank(
            ce_rank,qids,golds,docs,folds,stress),
        "nested_rank_ce_blend":eval_rank(
            blend_rank,qids,golds,docs,folds,stress),
        "ce_feature_lr":eval_rank(
            lr_rank,qids,golds,docs,folds,stress),
    }
    order=sorted(methods,key=lambda n:methods[n]["overall"]["recall_at_5"],reverse=True)

    result={
        "schema_version":"dsc2026.endgame.stage03b1.aiteam_reranker_oof.v1",
        "status":"COMPLETE",
        "reranker":{
            "model_id":manifest["model_id"],
            "sha":manifest["resolved_revision_sha"],
            "parameter_count":manifest["parameter_count"],
            "max_length":MAX_LENGTH,
            "passages_per_document":PASSAGES_PER_DOC,
            "shortlist_depth":SHORTLIST_DEPTH,
        },
        "score_cache":cache_meta,
        "methods":methods,
        "nested_blend_selected_alpha_by_fold":alphas,
        "ce_lr":{"C":LR_C,"feature_count":len(feature_names),
                 "features":feature_names,"fold_meta":lr_meta,
                 "fulltrain_model":"cache/stage03b1_aiteam_reranker/fulltrain_ce_fusion.joblib"},
        "order_by_oof_recall_at5":order,
        "claim_boundary":"OOF reranker evaluation on fold-clean Stage03B0 shortlist",
    }
    (OUT/"RERANK_OOF.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    rows=[]
    for n in order:
        m=methods[n]["overall"]
        rows.append({
            "method":n,"recall_at5":m["recall_at_5"],
            "precision_at5":m["precision_at_5"],
            "single_gold_r5":m["single_gold_recall_at_5"],
            "multi_gold_r5":m["multi_gold_recall_at_5"],
        })
    with (OUT/"METHOD_COMPARISON.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    active_params=567754752+596049920+int(manifest["parameter_count"])
    lines=[
        "# Stage 03B1 — AITeam Vietnamese Reranker OOF","",
        f"- Rerank depth: **{SHORTLIST_DEPTH}**",
        f"- Reranker params: **{int(manifest['parameter_count']):,}**",
        f"- Provisional active model ledger (AIT embedding + LAL + reranker): **{active_params/1e9:.3f}B / 4B**","",
        "| Method | OOF R@5 | P@5 | Single R@5 | Multi R@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['recall_at5']:.6f} | "
            f"{r['precision_at5']:.6f} | {r['single_gold_r5']:.6f} | "
            f"{r['multi_gold_r5']:.6f} |"
        )
    lines += ["","Best method is not automatically promoted until fold/stress behavior is inspected.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print(json.dumps({
        "status":"COMPLETE","order":order,
        "best":rows[0],
        "active_params_B":active_params/1e9,
        "out":str(OUT)
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

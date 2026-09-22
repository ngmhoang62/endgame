#!/usr/bin/env python
"""Stage 06A — Retrieval-aligned cross-encoder evidence.

Hypothesis
----------
The current reranker is evidence-disconnected:
dense retrievers promote a parent document because a specific legal region has
high semantic similarity, but Stage03B1 reranks two lexical windows selected
independently from the raw parent document.

This experiment reconnects the evidence path.

For every fold-clean top-30 candidate:
  AIT witness = best-scoring child region across
      atomic_split_2048 + coarse_pack_1024
  LAL witness = best-scoring child region across
      lal_atomic_split_2048 + lal_coarse_pack_1024 + LAL B4

Each witness is localized to one 220-word window *inside that semantic region*,
optionally prefixed with the document title, and scored by the same pinned
AITeamVN/Vietnamese_Reranker used in Stage03B1.

Thus neural model parameters are held fixed; only evidence selection changes.

Evaluation:
- semantic AIT witness CE only
- semantic LAL witness CE only
- max semantic CE only
- current 45D CE-LR + 19 retrieval-aligned evidence features (LR)
- same features with pairwise LR
- same features with HistGradientBoosting

Frozen five-fold OOF. Promotion gate: Recall@5 >= .960.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as b2

EVAL=ROOT/"data/evaluation_v2"
C1=ROOT/"cache/stage02c1_aiteam_representation"
C2=ROOT/"cache/stage02c2_lal_representation"
B4=ROOT/"cache/stage02b4_vi_screen"
BASE=ROOT/"cache/stage02b_structure/moderate_structure_regions.jsonl"

CACHE=ROOT/"cache/stage06a_retrieval_aligned_ce"
OUT=ROOT/"reports/stage06a_retrieval_aligned_ce"

AIT_Q=B4/"aiteamvn_v1/query_embeddings.f32.npy"
LAL_Q=B4/"vnlegal_lal/query_embeddings.f32.npy"

SHORT=30
FAMILIES=("ait","lal")
PROMOTE=.960
INITIAL_BATCH=64
QUERY_CHUNK=32

VIEWS={
    "ait_atomic":{
        "family":"ait",
        "chunks":C1/"atomic_split_2048/chunks.jsonl",
        "emb":C1/"atomic_split_2048/embeddings.f32.npy",
        "rep_meta":C1/"atomic_split_2048/representation.json",
        "emb_meta":C1/"atomic_split_2048/embeddings.json",
        "text_field":"text",
    },
    "ait_coarse":{
        "family":"ait",
        "chunks":C1/"coarse_pack_1024/chunks.jsonl",
        "emb":C1/"coarse_pack_1024/embeddings.f32.npy",
        "rep_meta":C1/"coarse_pack_1024/representation.json",
        "emb_meta":C1/"coarse_pack_1024/embeddings.json",
        "text_field":"text",
    },
    "lal_atomic":{
        "family":"lal",
        "chunks":C2/"lal_atomic_split_2048/chunks.jsonl",
        "emb":C2/"lal_atomic_split_2048/embeddings.f32.npy",
        "rep_meta":C2/"lal_atomic_split_2048/representation.json",
        "emb_meta":C2/"lal_atomic_split_2048/embeddings.json",
        "text_field":"text",
    },
    "lal_coarse":{
        "family":"lal",
        "chunks":C2/"lal_coarse_pack_1024/chunks.jsonl",
        "emb":C2/"lal_coarse_pack_1024/embeddings.f32.npy",
        "rep_meta":C2/"lal_coarse_pack_1024/representation.json",
        "emb_meta":C2/"lal_coarse_pack_1024/embeddings.json",
        "text_field":"text",
    },
    "lal_b4":{
        "family":"lal",
        "chunks":BASE,
        "emb":B4/"vnlegal_lal/region_embeddings.f32.npy",
        "rep_meta":ROOT/"reports/stage02b0_structure_geometry/STRUCTURE_GEOMETRY.json",
        "emb_meta":B4/"vnlegal_lal/region_embeddings.json",
        "text_field":"raw_text",
    },
}


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def stable(obj):
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    ).hexdigest()


def load_doc_names(docs):
    got=[]; names=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                got.append(str(x["id"]))
                names.append(str(x.get("name") or ""))
    if got!=docs:
        raise RuntimeError("document order drift")
    return names


def meta_identity(view):
    x=VIEWS[view]
    rm=rj(x["rep_meta"])
    em=rj(x["emb_meta"])
    return {
        "rep":rm.get("contract_hash") or rm.get("artifact",{}).get("regions_sha256")
              or rm.get("data_sha256"),
        "emb":em.get("contract_hash"),
    }


def build_ranges(view,docs):
    x=VIEWS[view]
    d2i={d:i for i,d in enumerate(docs)}
    emb=np.load(x["emb"],mmap_mode="r")
    starts=np.full(len(docs),-1,dtype=np.int64)
    ends=np.full(len(docs),-1,dtype=np.int64)

    n=0
    with x["chunks"].open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            d=str(row["doc_id"])
            if d not in d2i:
                raise RuntimeError(f"{view}: unknown parent {d}")
            j=d2i[d]
            if starts[j]<0:
                starts[j]=n
                ends[j]=n+1
            else:
                if ends[j]!=n:
                    raise RuntimeError(f"{view}: non-contiguous rows for doc {d}")
                ends[j]=n+1
            n+=1
    if n!=emb.shape[0]:
        raise RuntimeError(f"{view}: chunks/embedding count drift {n}!={emb.shape[0]}")
    if np.any(starts<0):
        missing=np.where(starts<0)[0][:20].tolist()
        raise RuntimeError(f"{view}: documents without child region sample={missing}")
    return emb,starts,ends


def selection_paths():
    CACHE.mkdir(parents=True,exist_ok=True)
    return (
        CACHE/"witness_view.i1.npy",
        CACHE/"witness_row.i4.npy",
        CACHE/"witness_similarity.f32.npy",
        CACHE/"witness_selection.json",
    )


def select_witnesses(qids,shortlist,docs,shortlist_sha):
    vp,rp,sp,mp=selection_paths()
    view_names=list(VIEWS)
    view_id={v:i for i,v in enumerate(view_names)}
    family_views={
        "ait":[v for v in view_names if VIEWS[v]["family"]=="ait"],
        "lal":[v for v in view_names if VIEWS[v]["family"]=="lal"],
    }
    contract={
        "schema":"stage06a.retrieval_aligned_witness.v1",
        "shortlist_sha":shortlist_sha,
        "ait_query_meta":rj(B4/"aiteamvn_v1/query_embeddings.json").get("contract_hash"),
        "lal_query_meta":rj(B4/"vnlegal_lal/query_embeddings.json").get("contract_hash"),
        "views":{v:meta_identity(v) for v in view_names},
        "selection":"max cosine child region within candidate parent, then max over family views",
    }
    ch=stable(contract)
    exists=[p.exists() for p in (vp,rp,sp,mp)]
    if any(exists) and not all(exists):
        raise RuntimeError("partial witness-selection cache; REFUSING TO OVERWRITE")
    if all(exists):
        m=rj(mp)
        if m.get("contract_hash")!=ch:
            raise RuntimeError("witness-selection contract mismatch; REFUSING TO OVERWRITE")
        vv=np.load(vp); rr=np.load(rp); ss=np.load(sp)
        if vv.shape!=(len(qids),SHORT,2) or rr.shape!=vv.shape or ss.shape!=vv.shape:
            raise RuntimeError("witness-selection shape drift")
        print("[witness] selection cache hit",flush=True)
        return vv,rr,ss,view_names

    print("[witness] load representation ranges",flush=True)
    stores={}
    for v in view_names:
        emb,starts,ends=build_ranges(v,docs)
        stores[v]=(emb,starts,ends)
        print(f"  {v}: vectors={emb.shape[0]}",flush=True)

    aitq=np.load(AIT_Q,mmap_mode="r")
    lalq=np.load(LAL_Q,mmap_mode="r")
    if aitq.shape!=(len(qids),1024) or lalq.shape!=(len(qids),1024):
        raise RuntimeError("query embedding shape drift")

    vv=np.empty((len(qids),SHORT,2),dtype=np.int8)
    rr=np.empty((len(qids),SHORT,2),dtype=np.int32)
    ss=np.empty((len(qids),SHORT,2),dtype=np.float32)

    for qi in range(len(qids)):
        for pos,d0 in enumerate(shortlist[qi]):
            d=int(d0)
            for fi,fam in enumerate(FAMILIES):
                q=np.asarray(aitq[qi] if fam=="ait" else lalq[qi],dtype=np.float32)
                best=(-np.inf,None,None)
                for v in family_views[fam]:
                    emb,starts,ends=stores[v]
                    a=int(starts[d]); b=int(ends[d])
                    block=np.asarray(emb[a:b],dtype=np.float32)
                    sc=block@q
                    local=int(np.argmax(sc))
                    val=float(sc[local])
                    if val>best[0]:
                        best=(val,v,a+local)
                ss[qi,pos,fi]=best[0]
                vv[qi,pos,fi]=view_id[best[1]]
                rr[qi,pos,fi]=best[2]
        if (qi+1)%100==0 or qi+1==len(qids):
            print(f"[witness] select {qi+1}/{len(qids)}",flush=True)

    np.save(vp,vv); np.save(rp,rr); np.save(sp,ss)
    mp.write_text(json.dumps({
        "contract_hash":ch,"contract":contract,"status":"PASS",
        "view_names":view_names,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return vv,rr,ss,view_names


def load_needed_texts(vv,rr,view_names):
    needed={v:set() for v in view_names}
    for fi in range(2):
        ids=vv[:,:,fi]
        rows=rr[:,:,fi]
        for vid,v in enumerate(view_names):
            vals=rows[ids==vid]
            if len(vals):
                needed[v].update(map(int,np.unique(vals)))

    text={}
    for v in view_names:
        want=needed[v]
        text[v]={}
        if not want:
            continue
        field=VIEWS[v]["text_field"]
        with VIEWS[v]["chunks"].open(encoding="utf-8") as f:
            rowi=0
            for line in f:
                if not line.strip(): continue
                if rowi in want:
                    x=json.loads(line)
                    text[v][rowi]=str(x[field])
                rowi+=1
        if len(text[v])!=len(want):
            raise RuntimeError(f"{v}: selected text load incomplete {len(text[v])}/{len(want)}")
        print(f"[text] {v}: loaded {len(text[v])} selected rows",flush=True)
    return text


def score_paths():
    d=CACHE/"semantic_ce"
    d.mkdir(parents=True,exist_ok=True)
    return d/"scores.f32.npy",d/"done.u1.npy",d/"meta.json"


def open_scores(nq,selection_sha,manifest):
    sp,dp,mp=score_paths()
    contract={
        "schema":"stage06a.retrieval_aligned_ce.v1",
        "selection_sha":selection_sha,
        "reranker_sha":manifest["resolved_revision_sha"],
        "families":["ait","lal"],
        "localizer":{
            "function":"Stage03B1 top_passages",
            "count":1,"window":220,"overlap":70,
        },
        "package":"[VAN BAN] normalized document title + [DOAN TRUY XUAT] localized semantic region",
        "max_length":512,
    }
    ch=stable(contract)
    ex=[p.exists() for p in (sp,dp,mp)]
    if any(ex) and not all(ex):
        raise RuntimeError("partial semantic CE cache; REFUSING TO OVERWRITE")
    if all(ex):
        m=rj(mp)
        if m.get("contract_hash")!=ch:
            raise RuntimeError("semantic CE contract mismatch; REFUSING TO OVERWRITE")
        s=np.lib.format.open_memmap(sp,mode="r+")
        d=np.lib.format.open_memmap(dp,mode="r+")
        if s.shape!=(nq,SHORT,2) or d.shape!=(nq,):
            raise RuntimeError("semantic CE cache shape drift")
        print(f"[semantic-ce] RESUME {int(np.asarray(d).sum())}/{nq}",flush=True)
        return s,d,m,sp,dp,mp
    s=np.lib.format.open_memmap(sp,mode="w+",dtype=np.float32,shape=(nq,SHORT,2))
    s[:]=np.nan; s.flush()
    d=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(nq,))
    d[:]=0; d.flush()
    m={"contract_hash":ch,"contract":contract,"status":"IN_PROGRESS",
       "completed":0,"total":nq,"current_batch":INITIAL_BATCH}
    mp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return s,d,m,sp,dp,mp


def clean_title(name):
    return " ".join(str(name or "").replace("-"," ").replace("_"," ").split())


def prepare_chunk(qis,qids,questions,shortlist,docs,names,vv,rr,texts,view_names):
    pairs=[]; owners=[]
    for qi in qis:
        q=questions[qids[qi]]
        for pos,d0 in enumerate(shortlist[qi]):
            didx=int(d0)
            title=clean_title(names[didx])
            for fi in range(2):
                v=view_names[int(vv[qi,pos,fi])]
                raw=texts[v][int(rr[qi,pos,fi])]
                ps=b1.top_passages(q,raw,count=1)
                local=ps[0] if ps else raw
                passage=(f"[VĂN BẢN] {title}\n[ĐOẠN TRUY XUẤT]\n{local}"
                         if title else local)
                pairs.append((q,passage))
                owners.append((qi,pos,fi))
    return qis,pairs,owners


def score_semantic(qids,questions,shortlist,docs,names,vv,rr,texts,view_names,selection_sha):
    manifest=b1.load_model_manifest()
    scores,done,meta,sp,dp,mp=open_scores(len(qids),selection_sha,manifest)
    pending=[i for i in range(len(qids)) if int(done[i])==0]
    if not pending:
        print("[semantic-ce] cache complete",flush=True)
        return np.asarray(scores,dtype=np.float32)

    tok,model=b1.load_model()
    batch=int(meta.get("current_batch",INITIAL_BATCH))
    chunks=[pending[s:s+QUERY_CHUNK] for s in range(0,len(pending),QUERY_CHUNK)]
    ex=ThreadPoolExecutor(max_workers=1,thread_name_prefix="semantic-witness-prefetch")
    future=ex.submit(prepare_chunk,chunks[0],qids,questions,shortlist,docs,names,vv,rr,texts,view_names)
    started=time.perf_counter(); newly=0
    try:
        for ci in range(len(chunks)):
            t=time.perf_counter()
            qis,pairs,owners=future.result()
            wait=time.perf_counter()-t
            if ci+1<len(chunks):
                future=ex.submit(prepare_chunk,chunks[ci+1],qids,questions,shortlist,docs,names,vv,rr,texts,view_names)
            t=time.perf_counter()
            vals,batch=b1.forward_pairs(tok,model,pairs,batch)
            gpu=time.perf_counter()-t
            tmp={qi:np.full((SHORT,2),-np.inf,np.float32) for qi in qis}
            for (qi,pos,fi),val in zip(owners,vals):
                tmp[qi][pos,fi]=float(val)
            for qi in qis:
                if not np.isfinite(tmp[qi]).all():
                    raise RuntimeError(f"nonfinite semantic CE q={qids[qi]}")
                scores[qi]=tmp[qi]; done[qi]=1; newly+=1
            scores.flush(); done.flush()
            completed=int(np.asarray(done).sum())
            meta={
                "contract_hash":meta["contract_hash"],"contract":meta["contract"],
                "status":"PASS" if completed==len(qids) else "IN_PROGRESS",
                "completed":completed,"total":len(qids),"current_batch":batch,
                "last_wait":wait,"last_gpu":gpu,
            }
            mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            rate=newly/max(time.perf_counter()-started,1e-9)
            print(f"[semantic-ce] {completed}/{len(qids)} batch={batch} rate={rate:.2f} q/s wait={wait:.2f}s gpu={gpu:.2f}s pairs={len(pairs)}",flush=True)
    finally:
        ex.shutdown(wait=False,cancel_futures=True)
        del model,tok
        b1.clear_cuda()
    return np.asarray(scores,dtype=np.float32)


def ranks(a):
    out=np.empty(a.shape,dtype=np.int32)
    for i in range(len(a)):
        o=np.argsort(-a[i],kind="stable")
        inv=np.empty(a.shape[1],dtype=np.int32)
        inv[o]=np.arange(1,a.shape[1]+1)
        out[i]=inv
    return out


def sem_features(sem,sim):
    feats=[]
    names=[]
    ce_rank=[]
    sim_rank=[]
    for fi,fam in enumerate(FAMILIES):
        raw=sem[:,:,fi]
        z=b1.zrows(raw)
        sd=np.std(raw,axis=1,keepdims=True); sd=np.where(sd<1e-8,1.,sd)
        gap=(np.max(raw,axis=1,keepdims=True)-raw)/sd
        r=ranks(raw)
        feats += [raw[:,:,None],z[:,:,None],gap[:,:,None],(1/(10+r))[:,:,None]]
        names += [f"{fam}_semce_raw",f"{fam}_semce_z",f"{fam}_semce_gapz",f"{fam}_semce_rr10"]
        ce_rank.append(r)

        sr=sim[:,:,fi]
        sz=b1.zrows(sr)
        rr=ranks(sr)
        feats += [sr[:,:,None],sz[:,:,None],(1/(10+rr))[:,:,None]]
        names += [f"{fam}_region_sim",f"{fam}_region_sim_z",f"{fam}_region_sim_rr10"]
        sim_rank.append(rr)

    ait=sem[:,:,0]; lal=sem[:,:,1]
    mx=np.maximum(ait,lal); mn=np.minimum(ait,lal); mean=.5*(ait+lal); diff=np.abs(ait-lal)
    both5=((ce_rank[0]<=5).astype(np.float32)+(ce_rank[1]<=5).astype(np.float32))
    rankdiff=np.abs(ce_rank[0]-ce_rank[1]).astype(np.float32)/SHORT
    feats += [mx[:,:,None],mn[:,:,None],mean[:,:,None],diff[:,:,None],both5[:,:,None],rankdiff[:,:,None]]
    names += ["semce_max","semce_min","semce_mean","semce_absdiff","semce_top5_support","semce_rankdiff"]
    X=np.concatenate(feats,axis=2).astype(np.float32)
    return X,names


def labels(shortlist,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    y=np.zeros(shortlist.shape,dtype=np.uint8)
    for qi,q in enumerate(qids):
        g={d2i[d] for d in golds[q]}
        y[qi]=np.asarray([int(int(x) in g) for x in shortlist[qi]],dtype=np.uint8)
    return y


def eval_rank(rank,qids,golds,docs,folds,stress):
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    return {
        "overall":official_metrics(preds,golds,qids),
        "per_fold":{f:official_metrics(preds,golds,ids) for f,ids in folds.items()},
        "stress":{n:official_metrics(preds,golds,[q for q in ids if q in preds])
                  for n,ids in stress.items() if any(q in preds for q in ids)},
    }


def lr_oof(X,y,shortlist,qids,folds,C=.15):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    q2i={q:i for i,q in enumerate(qids)}
    out=np.empty_like(shortlist)
    for fn,heldids in folds.items():
        held=np.asarray([q2i[q] for q in heldids],np.int32)
        hs=set(map(int,held))
        tr=np.asarray([i for i in range(len(qids)) if i not in hs],np.int32)
        m=Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(C=C,class_weight="balanced",max_iter=300,tol=1e-5))
        ])
        m.fit(X[tr].reshape(-1,X.shape[-1]),y[tr].reshape(-1))
        for qi in held:
            p=m.predict_proba(X[qi])[:,1]
            out[qi]=shortlist[qi,np.lexsort((shortlist[qi],-p))]
        print(f"[LR/{fn}] done",flush=True)
    return out


def pairwise_oof(X,y,shortlist,qids,folds):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    q2i={q:i for i,q in enumerate(qids)}
    out=np.empty_like(shortlist)
    for fn,heldids in folds.items():
        held=np.asarray([q2i[q] for q in heldids],np.int32)
        hs=set(map(int,held))
        tr=[i for i in range(len(qids)) if i not in hs]
        diffs=[]; labs=[]
        for qi in tr:
            pos=np.where(y[qi]==1)[0]; neg=np.where(y[qi]==0)[0]
            if not len(pos) or not len(neg): continue
            # deterministic cap keeps every query roughly equally represented
            pairs=[(int(p),int(n)) for p in pos for n in neg][:48]
            for p,n in pairs:
                d=X[qi,p]-X[qi,n]
                diffs.append(d); labs.append(1)
                diffs.append(-d); labs.append(0)
        DX=np.asarray(diffs,np.float32); L=np.asarray(labs,np.uint8)
        m=Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(C=.10,max_iter=300,tol=1e-5,fit_intercept=False))
        ])
        m.fit(DX,L)
        for qi in held:
            p=m.decision_function(X[qi])
            out[qi]=shortlist[qi,np.lexsort((shortlist[qi],-p))]
        print(f"[PAIR/{fn}] rows={len(L)}",flush=True)
    return out


def hgb_oof(X,y,shortlist,qids,folds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    q2i={q:i for i,q in enumerate(qids)}
    out=np.empty_like(shortlist)
    for fn,heldids in folds.items():
        held=np.asarray([q2i[q] for q in heldids],np.int32)
        hs=set(map(int,held))
        tr=np.asarray([i for i in range(len(qids)) if i not in hs],np.int32)
        XF=X[tr].reshape(-1,X.shape[-1]); YF=y[tr].reshape(-1)

        W=np.empty_like(YF,dtype=np.float32)
        off=0
        for qi in tr:
            yy=y[qi]; p=int(yy.sum()); n=len(yy)-p
            if p and n:
                ww=np.where(yy==1,.5/p,.5/n)
            else:
                ww=np.full(len(yy),1/len(yy))
            W[off:off+len(yy)]=ww
            off+=len(yy)

        m=HistGradientBoostingClassifier(
            learning_rate=.05,max_iter=160,max_leaf_nodes=15,
            min_samples_leaf=40,l2_regularization=1.0,random_state=276
        )
        m.fit(XF,YF,sample_weight=W)
        for qi in held:
            p=m.predict_proba(X[qi])[:,1]
            out[qi]=shortlist[qi,np.lexsort((shortlist[qi],-p))]
        print(f"[HGB/{fn}] done",flush=True)
    return out


def rank_score_only(scores,shortlist):
    out=np.empty_like(shortlist)
    for i in range(len(shortlist)):
        out[i]=shortlist[i,np.lexsort((shortlist[i],-scores[i]))]
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-hgb",action="store_true")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()

    print("[1/8] Load frozen OOF world",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    shortlist,shortlist_sha=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    names=load_doc_names(docs)

    current_ce=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy")
    current_done=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy")
    if current_ce.shape!=(len(qids),SHORT) or int(current_done.sum())!=len(qids):
        raise RuntimeError("Stage03B1 CE cache incomplete")

    print("[2/8] Select AIT/LAL retrieval witnesses",flush=True)
    vv,rr,sim,view_names=select_witnesses(qids,shortlist,docs,shortlist_sha)
    selection_sha=sha(selection_paths()[1])

    print("[3/8] Materialize only selected witness texts",flush=True)
    texts=load_needed_texts(vv,rr,view_names)

    print("[4/8] Score retrieval-aligned witnesses with same reranker",flush=True)
    sem=score_semantic(qids,questions,shortlist,docs,names,vv,rr,texts,view_names,selection_sha)

    print("[5/8] Build evidence features + parity baseline",flush=True)
    Xflat,cq,cand,names45=b1.build_source_ce_features(shortlist,current_ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1)
    y=labels(shortlist,qids,golds,docs)
    base=b2.current_lr_oof(X45,shortlist,y,qids,folds)
    base_m=eval_rank(base,qids,golds,docs,folds,stress)
    if abs(base_m["overall"]["recall_at_5"]-0.9425976255185238)>2e-6:
        raise RuntimeError("Stage03B1 parity failed")

    S,snames=sem_features(sem,sim)
    X=np.concatenate([X45,S],axis=2)

    print("[6/8] Semantic-only diagnostics",flush=True)
    ait_rank=rank_score_only(sem[:,:,0],shortlist)
    lal_rank=rank_score_only(sem[:,:,1],shortlist)
    max_rank=rank_score_only(np.maximum(sem[:,:,0],sem[:,:,1]),shortlist)

    print("[7/8] OOF fusion tournament",flush=True)
    lr=lr_oof(X,y,shortlist,qids,folds)
    pair=pairwise_oof(X,y,shortlist,qids,folds)
    hgb=None if args.skip_hgb else hgb_oof(X,y,shortlist,qids,folds)

    methods={
        "stage03b1_ce_lr":base_m,
        "ait_retrieval_witness_ce":eval_rank(ait_rank,qids,golds,docs,folds,stress),
        "lal_retrieval_witness_ce":eval_rank(lal_rank,qids,golds,docs,folds,stress),
        "max_retrieval_witness_ce":eval_rank(max_rank,qids,golds,docs,folds,stress),
        "lr_plus_retrieval_aligned_ce":eval_rank(lr,qids,golds,docs,folds,stress),
        "pairwise_plus_retrieval_aligned_ce":eval_rank(pair,qids,golds,docs,folds,stress),
    }
    if hgb is not None:
        methods["hgb_plus_retrieval_aligned_ce"]=eval_rank(hgb,qids,golds,docs,folds,stress)

    best=max(methods,key=lambda k:methods[k]["overall"]["recall_at_5"])
    best_r=methods[best]["overall"]["recall_at_5"]

    print("[8/8] Report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage06a.retrieval_aligned_ce.v1",
        "status":"COMPLETE",
        "hypothesis":"rerank the semantic child regions that caused parent retrieval instead of independent raw-document lexical windows",
        "families":{
            "ait":["atomic_split_2048","coarse_pack_1024"],
            "lal":["lal_atomic_split_2048","lal_coarse_pack_1024","lal_b4"],
        },
        "reranker":"same pinned AITeamVN/Vietnamese_Reranker as Stage03B1",
        "new_model_parameters":0,
        "features":{"base_dim":45,"new_dim":len(snames),"new_names":snames,"total_dim":int(X.shape[-1])},
        "methods":methods,
        "promotion":{
            "gate":PROMOTE,"best_method":best,"best_recall_at5":best_r,
            "decision":"PROMOTE" if best_r>=PROMOTE else "DO_NOT_PROMOTE",
        },
        "wall_seconds":time.perf_counter()-started,
    }
    (OUT/"RETRIEVAL_ALIGNED_CE.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    lines=[
        "# Stage 06A — Retrieval-Aligned Cross-Encoder Evidence","",
        "| Method | OOF R@5 | P@5 | Single | Multi |",
        "|---|---:|---:|---:|---:|",
    ]
    for n,m in methods.items():
        o=m["overall"]
        lines.append(
            f"| {n} | {o['recall_at_5']:.6f} | {o['precision_at_5']:.6f} | "
            f"{o['single_gold_recall_at_5']:.6f} | {o['multi_gold_recall_at_5']:.6f} |"
        )
    lines += ["",f"**Decision: {report['promotion']['decision']}** "
              f"(gate {PROMOTE:.3f}; best={best} {best_r:.6f})",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*108)
    for n,m in methods.items():
        o=m["overall"]
        print(f"{n:38s} R={o['recall_at_5']:.9f} P={o['precision_at_5']:.9f} single={o['single_gold_recall_at_5']:.9f} multi={o['multi_gold_recall_at_5']:.9f}")
    print("DECISION:",report["promotion"]["decision"])
    print("REPORT:",OUT/"REPORT.md")
    print("="*108)


if __name__=="__main__":
    main()

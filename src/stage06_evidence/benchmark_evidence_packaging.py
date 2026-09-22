#!/usr/bin/env python
"""Stage 06A0 — Evidence-packaging benchmark before retrieval-aligned reranking.

Protocol
========
DEV:
    folds 0,1,2 only.
    Used to choose ONE evidence-packaging policy.

CERT:
    folds 3,4 only.
    Never used to choose packaging/title/family aggregation.
    Exactly one frozen winner is evaluated here.

Why
===
Do not inherit historical `220 words / overlap 70` as truth.
Benchmark evidence localization from scratch under the CURRENT ENDGAME world.

Semantic witnesses
------------------
For each candidate parent:
  AIT witness = max-cosine child among current AIT atomic/coarse representations.
  LAL witness = max-cosine child among current LAL atomic/coarse representations.

DEV packaging screen (title OFF)
--------------------------------
  head_budget
      first tokenizer-aware passage budget.
  token_full_nonoverlap
      partition whole witness into max-length token windows, no overlap; CE max.
  token_full_overlap50
      same full budget, 50% overlap; CE max.
  token75_overlap50
      75% of available passage budget, 50% overlap; CE max.
  lexical220_top1
      historical control only; not privileged.

Passage budget is dynamic per query:
  reranker max_length(512) - query tokens - pair special tokens - optional title tokens.

DEV winner criterion
--------------------
1) mean per-query candidate AUC on a DEV screen population;
2) tie-break semantic-only Recall@5;
3) tie-break fewer CE pairs.

The DEV screen contains ALL queries where top-15 oracle can improve current CE-LR,
plus deterministic matched controls. Gold is used only on DEV.

After packaging is frozen:
  title OFF vs ON is decided on the same DEV screen.
  family aggregation (AIT / LAL / max raw / mean-z) is part of DEV selection.

Then score the ONE frozen winner on the full 6991 population, train ONE fixed
LogisticRegression fusion on DEV (folds0-2), and evaluate it ONCE on CERT
(folds3-4).

No CERT alternative/model tournament is produced.

Next-stage gate
---------------
Proceed to full 5-fold OOF certification only if:
  CERT delta Recall@5 vs current CE-LR >= +0.005
  and CERT single-gold recall does not decrease.

This is NOT a submission materializer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
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

CACHE=ROOT/"cache/stage06a0_evidence_packaging"
OUT=ROOT/"reports/stage06a0_evidence_packaging"

AIT_Q=B4/"aiteamvn_v1/query_embeddings.f32.npy"
LAL_Q=B4/"vnlegal_lal/query_embeddings.f32.npy"

DEV_FOLDS=("fold_0","fold_1","fold_2")
CERT_FOLDS=("fold_3","fold_4")
SCREEN_DEPTH=12
FULL_DEPTH=15
SCREEN_CONTROL_MULT=1
SCREEN_MAX=900
MAXLEN=512
QUERY_CHUNK=24
INITIAL_BATCH=64
SEED=276

VIEWS={
    "ait_atomic":{
        "family":"ait",
        "chunks":C1/"atomic_split_2048/chunks.jsonl",
        "emb":C1/"atomic_split_2048/embeddings.f32.npy",
        "text_field":"text",
    },
    "ait_coarse":{
        "family":"ait",
        "chunks":C1/"coarse_pack_1024/chunks.jsonl",
        "emb":C1/"coarse_pack_1024/embeddings.f32.npy",
        "text_field":"text",
    },
    "lal_atomic":{
        "family":"lal",
        "chunks":C2/"lal_atomic_split_2048/chunks.jsonl",
        "emb":C2/"lal_atomic_split_2048/embeddings.f32.npy",
        "text_field":"text",
    },
    "lal_coarse":{
        "family":"lal",
        "chunks":C2/"lal_coarse_pack_1024/chunks.jsonl",
        "emb":C2/"lal_coarse_pack_1024/embeddings.f32.npy",
        "text_field":"text",
    },
}

PACKAGES=[
    "head_budget",
    "token_full_nonoverlap",
    "token_full_overlap50",
    "token75_overlap50",
    "lexical220_top1",
]
AGGREGATIONS=("ait","lal","max_raw","mean_z")


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stable(obj):
    return hashlib.sha256(
        json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def load_names(docs):
    got=[]; names=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                got.append(str(x["id"]))
                names.append(str(x.get("name") or ""))
    if got!=docs:
        raise RuntimeError("corpus order drift")
    return names


def clean_title(s):
    return " ".join(str(s or "").replace("-"," ").replace("_"," ").split())


def load_ranges(view,docs):
    x=VIEWS[view]
    emb=np.load(x["emb"],mmap_mode="r")
    d2i={d:i for i,d in enumerate(docs)}
    starts=np.full(len(docs),-1,np.int64)
    ends=np.full(len(docs),-1,np.int64)
    n=0
    with x["chunks"].open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            d=str(row["doc_id"])
            if d not in d2i:
                raise RuntimeError(f"{view}: unknown doc {d}")
            j=d2i[d]
            if starts[j]<0:
                starts[j]=n; ends[j]=n+1
            else:
                if ends[j]!=n:
                    raise RuntimeError(f"{view}: non-contiguous parent {d}")
                ends[j]=n+1
            n+=1
    if n!=emb.shape[0]:
        raise RuntimeError(f"{view}: chunk/embedding count drift")
    if np.any(starts<0):
        raise RuntimeError(f"{view}: missing parent regions")
    return emb,starts,ends


def witness_cache(tag):
    d=CACHE/"witness"
    d.mkdir(parents=True,exist_ok=True)
    return d/f"{tag}_view.i1.npy",d/f"{tag}_row.i4.npy",d/f"{tag}_sim.f32.npy",d/f"{tag}.json"


def select_witnesses(tag,query_indices,shortlist,docs):
    vp,rp,sp,mp=witness_cache(tag)
    view_names=list(VIEWS)
    vid={v:i for i,v in enumerate(view_names)}
    contract={
        "schema":"stage06a0.witness.v1",
        "tag":tag,
        "query_indices_sha":stable(list(map(int,query_indices))),
        "depth":int(shortlist.shape[1]),
        "views":{v:{"chunks_sha":sha(VIEWS[v]["chunks"]),
                    "emb_shape":list(np.load(VIEWS[v]["emb"],mmap_mode="r").shape)}
                 for v in view_names},
        "selection":"max cosine child per family across atomic/coarse",
    }
    ch=stable(contract)
    ex=[p.exists() for p in (vp,rp,sp,mp)]
    if any(ex) and not all(ex):
        raise RuntimeError(f"{tag}: partial witness cache")
    if all(ex):
        meta=rj(mp)
        if meta.get("contract_hash")!=ch:
            raise RuntimeError(f"{tag}: witness cache contract mismatch")
        vv=np.load(vp); rr=np.load(rp); ss=np.load(sp)
        shape=(len(query_indices),shortlist.shape[1],2)
        if vv.shape!=shape or rr.shape!=shape or ss.shape!=shape:
            raise RuntimeError(f"{tag}: witness shape drift")
        print(f"[witness/{tag}] cache hit",flush=True)
        return vv,rr,ss,view_names

    stores={}
    for v in view_names:
        stores[v]=load_ranges(v,docs)
        print(f"[witness/{tag}] {v} vectors={stores[v][0].shape[0]}",flush=True)

    aq=np.load(AIT_Q,mmap_mode="r")
    lq=np.load(LAL_Q,mmap_mode="r")
    vv=np.empty((len(query_indices),shortlist.shape[1],2),np.int8)
    rr=np.empty_like(vv,dtype=np.int32)
    ss=np.empty_like(vv,dtype=np.float32)

    fam_views={
        "ait":[v for v in view_names if VIEWS[v]["family"]=="ait"],
        "lal":[v for v in view_names if VIEWS[v]["family"]=="lal"],
    }
    for li,qi in enumerate(query_indices):
        for pos,d0 in enumerate(shortlist[li]):
            d=int(d0)
            for fi,fam in enumerate(("ait","lal")):
                q=np.asarray(aq[qi] if fam=="ait" else lq[qi],dtype=np.float32)
                best=(-1e30,None,None)
                for v in fam_views[fam]:
                    emb,st,en=stores[v]
                    a=int(st[d]); b=int(en[d])
                    sc=np.asarray(emb[a:b],dtype=np.float32)@q
                    j=int(np.argmax(sc)); val=float(sc[j])
                    if val>best[0]:
                        best=(val,v,a+j)
                ss[li,pos,fi]=best[0]
                vv[li,pos,fi]=vid[best[1]]
                rr[li,pos,fi]=best[2]
        if (li+1)%100==0 or li+1==len(query_indices):
            print(f"[witness/{tag}] {li+1}/{len(query_indices)}",flush=True)

    np.save(vp,vv); np.save(rp,rr); np.save(sp,ss)
    mp.write_text(json.dumps(
        {"status":"PASS","contract_hash":ch,"contract":contract,"view_names":view_names},
        ensure_ascii=False,indent=2
    )+"\n",encoding="utf-8")
    return vv,rr,ss,view_names


def load_selected_texts(vv,rr,view_names):
    needed={v:set() for v in view_names}
    for fi in range(2):
        for vid,v in enumerate(view_names):
            vals=rr[:,:,fi][vv[:,:,fi]==vid]
            if len(vals):
                needed[v].update(map(int,np.unique(vals)))
    out={}
    for v in view_names:
        want=needed[v]; got={}
        field=VIEWS[v]["text_field"]
        if want:
            with VIEWS[v]["chunks"].open(encoding="utf-8") as f:
                i=0
                for line in f:
                    if not line.strip(): continue
                    if i in want:
                        got[i]=str(json.loads(line)[field])
                    i+=1
        if len(got)!=len(want):
            raise RuntimeError(f"{v}: text load incomplete {len(got)}/{len(want)}")
        out[v]=got
        print(f"[text] {v}: {len(got)} selected",flush=True)
    return out


def pair_budget(tok,query,title=""):
    qids=tok.encode(query,add_special_tokens=False)
    special=int(tok.num_special_tokens_to_add(pair=True))
    title_prefix=""
    title_tokens=0
    if title:
        title_prefix=f"[VĂN BẢN] {title}\n[ĐOẠN TRUY XUẤT]\n"
        title_tokens=len(tok.encode(title_prefix,add_special_tokens=False))
    budget=MAXLEN-len(qids)-special-title_tokens-2
    return max(64,budget),title_prefix


def token_windows(tok,text,width,stride):
    ids=tok.encode(text,add_special_tokens=False)
    if not ids:
        return [""]
    if len(ids)<=width:
        return [tok.decode(ids,skip_special_tokens=True)]
    wins=[]
    for s in range(0,len(ids),stride):
        part=ids[s:s+width]
        if not part: break
        wins.append(tok.decode(part,skip_special_tokens=True))
        if s+width>=len(ids): break
    return wins


def make_windows(tok,mode,query,text,title):
    budget,prefix=pair_budget(tok,query,title)
    if mode=="head_budget":
        wins=token_windows(tok,text,budget,budget)[:1]
    elif mode=="token_full_nonoverlap":
        wins=token_windows(tok,text,budget,budget)
    elif mode=="token_full_overlap50":
        wins=token_windows(tok,text,budget,max(1,budget//2))
    elif mode=="token75_overlap50":
        width=max(64,int(round(.75*budget)))
        wins=token_windows(tok,text,width,max(1,width//2))
    elif mode=="lexical220_top1":
        ps=b1.top_passages(query,text,count=1,window=220,overlap=70)
        wins=ps if ps else [text]
    else:
        raise RuntimeError(mode)
    return [prefix+w for w in wins]


def score_cache(tag,mode,title_on,nq,depth,contract_extra):
    d=CACHE/"ce"/tag
    d.mkdir(parents=True,exist_ok=True)
    name=f"{mode}__title{int(title_on)}"
    sp=d/f"{name}.f32.npy"; dp=d/f"{name}.done.npy"; mp=d/f"{name}.json"
    manifest=b1.load_model_manifest()
    contract={
        "schema":"stage06a0.ce.v1",
        "tag":tag,"mode":mode,"title_on":bool(title_on),
        "nq":nq,"depth":depth,
        "reranker_sha":manifest["resolved_revision_sha"],
        "max_length":MAXLEN,
        **contract_extra,
    }
    ch=stable(contract)
    ex=[p.exists() for p in (sp,dp,mp)]
    if any(ex) and not all(ex):
        raise RuntimeError(f"partial CE cache {tag}/{name}")
    if all(ex):
        m=rj(mp)
        if m.get("contract_hash")!=ch:
            raise RuntimeError(f"CE cache mismatch {tag}/{name}")
        scores=np.lib.format.open_memmap(sp,mode="r+")
        done=np.lib.format.open_memmap(dp,mode="r+")
        if scores.shape!=(nq,depth,2) or done.shape!=(nq,):
            raise RuntimeError("CE cache shape drift")
        return scores,done,m,sp,dp,mp
    scores=np.lib.format.open_memmap(sp,mode="w+",dtype=np.float32,shape=(nq,depth,2))
    scores[:]=np.nan; scores.flush()
    done=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(nq,))
    done[:]=0; done.flush()
    m={"contract_hash":ch,"contract":contract,"status":"IN_PROGRESS","completed":0,
       "pair_count":0,"current_batch":INITIAL_BATCH}
    mp.write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    return scores,done,m,sp,dp,mp


def prep_chunk(tok,mode,title_on,lis,query_indices,qids,questions,shortlist,docs,names,vv,rr,texts,view_names):
    pairs=[]; owners=[]
    for li in lis:
        qi=int(query_indices[li]); q=questions[qids[qi]]
        for pos,d0 in enumerate(shortlist[li]):
            didx=int(d0)
            title=clean_title(names[didx]) if title_on else ""
            for fi in range(2):
                v=view_names[int(vv[li,pos,fi])]
                text=texts[v][int(rr[li,pos,fi])]
                wins=make_windows(tok,mode,q,text,title)
                for w in wins:
                    pairs.append((q,w)); owners.append((li,pos,fi))
    return lis,pairs,owners


def score_mode(tag,mode,title_on,query_indices,qids,questions,shortlist,docs,names,vv,rr,texts,view_names,witness_hash,shared_model=None):
    contract_extra={"witness_hash":witness_hash}
    scores,done,meta,sp,dp,mp=score_cache(
        tag,mode,title_on,len(query_indices),shortlist.shape[1],contract_extra
    )
    pending=[i for i in range(len(query_indices)) if int(done[i])==0]
    if not pending:
        print(f"[ce/{tag}/{mode}/title{int(title_on)}] cache complete",flush=True)
        return np.asarray(scores,dtype=np.float32),int(meta.get("pair_count",0)),shared_model

    if shared_model is None:
        tok,model=b1.load_model()
    else:
        tok,model=shared_model
    batch=int(meta.get("current_batch",INITIAL_BATCH))
    chunks=[pending[s:s+QUERY_CHUNK] for s in range(0,len(pending),QUERY_CHUNK)]
    total_pairs=int(meta.get("pair_count",0)); start=time.perf_counter(); newq=0

    # IMPORTANT: Hugging Face fast tokenizers are backed by Rust objects that
    # cannot be borrowed concurrently. Stage06A0 originally prepared the next
    # chunk in a background thread while forward_pairs() tokenized the current
    # GPU batch with the same tokenizer instance, which can raise
    # RuntimeError("Already borrowed"). Preparation is intentionally sequential
    # here. This changes scheduling only, not the scoring contract, so existing
    # completed cache rows remain valid and resumable.
    for lis in chunks:
        lis,pairs,owners=prep_chunk(
            tok,mode,title_on,lis,query_indices,qids,questions,shortlist,
            docs,names,vv,rr,texts,view_names
        )
        vals,batch=b1.forward_pairs(tok,model,pairs,batch)
        tmp={li:np.full((shortlist.shape[1],2),-np.inf,np.float32) for li in lis}
        for (li,pos,fi),val in zip(owners,vals):
            if val>tmp[li][pos,fi]:
                tmp[li][pos,fi]=float(val)
        for li in lis:
            if not np.isfinite(tmp[li]).all():
                raise RuntimeError(f"nonfinite CE li={li}")
            scores[li]=tmp[li]; done[li]=1; newq+=1
        total_pairs+=len(pairs)
        scores.flush(); done.flush()
        completed=int(np.asarray(done).sum())
        meta={"contract_hash":meta["contract_hash"],"contract":meta["contract"],
              "status":"PASS" if completed==len(query_indices) else "IN_PROGRESS",
              "completed":completed,"pair_count":total_pairs,"current_batch":batch}
        mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        print(f"[ce/{tag}/{mode}/title{int(title_on)}] {completed}/{len(query_indices)} pairs={total_pairs} rate={newq/max(time.perf_counter()-start,1e-9):.2f} q/s",flush=True)
    return np.asarray(scores,dtype=np.float32),total_pairs,(tok,model)


def rank_from_score(score,shortlist):
    out=np.empty_like(shortlist)
    for i in range(len(shortlist)):
        out[i]=shortlist[i,np.lexsort((shortlist[i],-score[i]))]
    return out


def per_query_recall(rank,qids_sel,golds,docs):
    vals=[]
    for i,q in enumerate(qids_sel):
        p={docs[int(x)] for x in rank[i,:5]}
        g=set(golds[q])
        vals.append(len(p&g)/len(g))
    return np.asarray(vals,dtype=np.float64)


def query_auc(score,shortlist,qids_sel,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    vals=[]
    for i,q in enumerate(qids_sel):
        g={d2i[d] for d in golds[q]}
        pos=[score[i,j] for j,d in enumerate(shortlist[i]) if int(d) in g]
        neg=[score[i,j] for j,d in enumerate(shortlist[i]) if int(d) not in g]
        if not pos or not neg:
            continue
        wins=ties=tot=0
        for p in pos:
            for n in neg:
                wins+=p>n; ties+=p==n; tot+=1
        vals.append((wins+.5*ties)/tot)
    return float(np.mean(vals)) if vals else float("nan")


def zrows(a):
    a=np.asarray(a,dtype=np.float64)
    mu=a.mean(axis=1,keepdims=True); sd=a.std(axis=1,keepdims=True)
    sd=np.where(sd<1e-8,1.,sd)
    return ((a-mu)/sd).astype(np.float32)


def agg_scores(sem,kind):
    a=sem[:,:,0]; l=sem[:,:,1]
    if kind=="ait": return a
    if kind=="lal": return l
    if kind=="max_raw": return np.maximum(a,l)
    if kind=="mean_z": return .5*(zrows(a)+zrows(l))
    raise RuntimeError(kind)


def current_ce_lr(qids,questions,golds,folds,docs,stress,shortlist,sources,current_ce):
    Xflat,cq,cand,names45=b1.build_source_ce_features(shortlist,current_ce,sources)
    X45=Xflat.reshape(len(qids),30,-1)
    y=b2.labels_for(shortlist,qids,golds,docs)
    rank=b2.current_lr_oof(X45,shortlist,y,qids,folds)
    return rank,X45,y,names45


def choose_screen(dev_indices,baseline,short15,qids,golds,docs):
    base_r=per_query_recall(baseline[dev_indices], [qids[i] for i in dev_indices],golds,docs)
    oracle=[]
    for qi in dev_indices:
        q=qids[qi]
        g=set(golds[q])
        p={docs[int(x)] for x in short15[np.where(dev_indices==qi)[0][0],:5]}  # placeholder overwritten below
        # true cap5 oracle in top15 = min(5, number gold in candidate pool)/|gold|
        pool={docs[int(x)] for x in short15[np.where(dev_indices==qi)[0][0]]}
        oracle.append(min(5,len(pool&g))/len(g))
    oracle=np.asarray(oracle)
    rescue_pos=np.where(oracle>base_r+1e-12)[0]
    control_pos=np.where(oracle<=base_r+1e-12)[0]

    rng=np.random.default_rng(SEED)
    want=min(len(control_pos),max(len(rescue_pos)*SCREEN_CONTROL_MULT,200))
    if len(control_pos)>want:
        control_pos=np.sort(rng.choice(control_pos,size=want,replace=False))
    pos=np.unique(np.concatenate([rescue_pos,control_pos]))
    if len(pos)>SCREEN_MAX:
        # Never drop rescuables; cap controls first.
        rset=set(map(int,rescue_pos))
        controls=[int(x) for x in pos if int(x) not in rset]
        keep_controls=max(0,SCREEN_MAX-len(rescue_pos))
        controls=controls[:keep_controls]
        pos=np.asarray(sorted(list(rset)+controls),dtype=np.int32)
    return pos,{
        "dev_queries":int(len(dev_indices)),
        "rescuable":int(len(rescue_pos)),
        "controls":int(len(pos)-len(rescue_pos)),
        "screen_total":int(len(pos)),
        "baseline_screen_recall":float(base_r[pos].mean()),
        "oracle15_screen_recall":float(oracle[pos].mean()),
    }


def screen_table(sem,shortlist_screen,qids_screen,golds,docs,pair_count,mode,title_on):
    rows=[]
    for agg in AGGREGATIONS:
        s=agg_scores(sem,agg)
        auc=query_auc(s,shortlist_screen,qids_screen,golds,docs)
        rank=rank_from_score(s,shortlist_screen)
        rec=float(per_query_recall(rank,qids_screen,golds,docs).mean())
        rows.append({"mode":mode,"title_on":bool(title_on),"aggregation":agg,
                     "mean_query_auc":auc,"semantic_recall_at5":rec,
                     "pair_count":int(pair_count)})
    return rows


def semantic_features(sem,sim):
    feats=[]; names=[]
    for fi,fam in enumerate(("ait","lal")):
        raw=sem[:,:,fi]; z=zrows(raw)
        sd=np.std(raw,axis=1,keepdims=True); sd=np.where(sd<1e-8,1.,sd)
        gap=(np.max(raw,axis=1,keepdims=True)-raw)/sd
        r=np.empty_like(raw,dtype=np.int32)
        sr=np.empty_like(raw,dtype=np.int32)
        for i in range(len(raw)):
            o=np.argsort(-raw[i],kind="stable"); inv=np.empty(raw.shape[1],np.int32); inv[o]=np.arange(1,raw.shape[1]+1); r[i]=inv
            o=np.argsort(-sim[i,:,fi],kind="stable"); inv=np.empty(raw.shape[1],np.int32); inv[o]=np.arange(1,raw.shape[1]+1); sr[i]=inv
        feats += [raw[:,:,None],z[:,:,None],gap[:,:,None],(1/(10+r))[:,:,None],
                  sim[:,:,fi,None],zrows(sim[:,:,fi])[:,:,None],(1/(10+sr))[:,:,None]]
        names += [f"{fam}_ce_raw",f"{fam}_ce_z",f"{fam}_ce_gapz",f"{fam}_ce_rr10",
                  f"{fam}_sim_raw",f"{fam}_sim_z",f"{fam}_sim_rr10"]
    mx=np.maximum(sem[:,:,0],sem[:,:,1])
    meanz=.5*(zrows(sem[:,:,0])+zrows(sem[:,:,1]))
    feats += [mx[:,:,None],meanz[:,:,None],np.abs(sem[:,:,0]-sem[:,:,1])[:,:,None]]
    names += ["sem_max_raw","sem_mean_z","sem_absdiff"]
    return np.concatenate(feats,axis=2).astype(np.float32),names


def fit_dev_eval_cert(X,y,shortlist,qids,docs,golds,dev_idx,cert_idx):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=.15,class_weight="balanced",max_iter=350,tol=1e-5))
    ])
    m.fit(X[dev_idx].reshape(-1,X.shape[-1]),y[dev_idx].reshape(-1))
    out=np.empty((len(cert_idx),shortlist.shape[1]),dtype=np.int32)
    for li,qi in enumerate(cert_idx):
        p=m.predict_proba(X[qi])[:,1]
        out[li]=shortlist[qi,np.lexsort((shortlist[qi],-p))]
    cert_q=[qids[i] for i in cert_idx]
    preds={q:[docs[int(x)] for x in out[i,:5]] for i,q in enumerate(cert_q)}
    return official_metrics(preds,golds,cert_q),out


def subset_metrics(rank,qids,docs,golds,indices):
    qq=[qids[i] for i in indices]
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids) if i in set(map(int,indices))}
    return official_metrics(preds,golds,qq)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--screen-only",action="store_true",
                    help="Stop after DEV packaging/title selection; useful for a first fast run.")
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()

    print("[1/10] Load frozen OOF world + current baseline",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    shortlist30,shortlist_sha=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    current_ce=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy")
    done=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy")
    if current_ce.shape!=(len(qids),30) or int(done.sum())!=len(qids):
        raise RuntimeError("current CE cache incomplete")
    baseline,X45,y,names45=current_ce_lr(qids,questions,golds,folds,docs,stress,shortlist30,sources,current_ce)
    names=load_names(docs)

    q2i={q:i for i,q in enumerate(qids)}
    dev_idx=np.asarray([q2i[q] for f in DEV_FOLDS for q in folds[f]],np.int32)
    cert_idx=np.asarray([q2i[q] for f in CERT_FOLDS for q in folds[f]],np.int32)
    if len(set(dev_idx)&set(cert_idx)) or len(dev_idx)+len(cert_idx)!=len(qids):
        raise RuntimeError("DEV/CERT partition invalid")
    print(f"  DEV={len(dev_idx)} CERT={len(cert_idx)}",flush=True)

    print("[2/10] Build DEV screen: all rescuable + matched controls",flush=True)
    short15_dev=shortlist30[dev_idx,:FULL_DEPTH]
    pos,screen_meta=choose_screen(dev_idx,baseline,short15_dev,qids,golds,docs)
    screen_idx=dev_idx[pos]
    screen_short=shortlist30[screen_idx,:SCREEN_DEPTH]
    screen_qids=[qids[i] for i in screen_idx]
    print("  ",screen_meta,flush=True)

    print("[3/10] Select semantic witnesses for DEV screen",flush=True)
    vv,rr,sim,view_names=select_witnesses("dev_screen",screen_idx,screen_short,docs)
    texts=load_selected_texts(vv,rr,view_names)
    witness_hash=sha(witness_cache("dev_screen")[1])

    print("[4/10] DEV packaging screen — title OFF",flush=True)
    tok,model=b1.load_model()
    shared=(tok,model)
    table=[]
    sem_by_mode={}
    for mode in PACKAGES:
        sem,pairs,shared=score_mode(
            "dev_screen",mode,False,screen_idx,qids,questions,screen_short,docs,names,
            vv,rr,texts,view_names,witness_hash,shared
        )
        sem_by_mode[mode]=sem
        table.extend(screen_table(sem,screen_short,screen_qids,golds,docs,pairs,mode,False))
        best=max([x for x in table if x["mode"]==mode and not x["title_on"]],
                 key=lambda r:(r["mean_query_auc"],r["semantic_recall_at5"],-r["pair_count"]))
        print(f"  {mode}: best={best['aggregation']} AUC={best['mean_query_auc']:.6f} R={best['semantic_recall_at5']:.6f} pairs={best['pair_count']}",flush=True)

    best0=max(table,key=lambda r:(r["mean_query_auc"],r["semantic_recall_at5"],-r["pair_count"]))
    frozen_mode=best0["mode"]; frozen_agg=best0["aggregation"]
    print(f"[DEV FREEZE-1] mode={frozen_mode} aggregation={frozen_agg}",flush=True)

    print("[5/10] DEV title OFF vs ON for frozen mode",flush=True)
    sem_on,pairs_on,shared=score_mode(
        "dev_screen",frozen_mode,True,screen_idx,qids,questions,screen_short,docs,names,
        vv,rr,texts,view_names,witness_hash,shared
    )
    title_rows=screen_table(sem_on,screen_short,screen_qids,golds,docs,pairs_on,frozen_mode,True)
    cand_off=[x for x in table if x["mode"]==frozen_mode and x["aggregation"]==frozen_agg and not x["title_on"]][0]
    cand_on=[x for x in title_rows if x["aggregation"]==frozen_agg][0]
    title_on=(cand_on["mean_query_auc"],cand_on["semantic_recall_at5"],-cand_on["pair_count"]) > \
             (cand_off["mean_query_auc"],cand_off["semantic_recall_at5"],-cand_off["pair_count"])
    winner=cand_on if title_on else cand_off
    table.extend(title_rows)
    print(f"[DEV FREEZE-2] title_on={title_on} winner AUC={winner['mean_query_auc']:.6f} R={winner['semantic_recall_at5']:.6f}",flush=True)

    dev_report={
        "schema":"stage06a0.dev_screen.v1",
        "dev_folds":list(DEV_FOLDS),"cert_folds":list(CERT_FOLDS),
        "screen":screen_meta,
        "table":table,
        "frozen":{"mode":frozen_mode,"aggregation":frozen_agg,"title_on":title_on},
        "selection_key":"max(mean_query_auc, semantic_recall_at5, -pair_count)",
    }
    (OUT/"DEV_PACKAGING_SCREEN.json").write_text(
        json.dumps(dev_report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    del model,tok
    b1.clear_cuda()

    if args.screen_only:
        print("="*100)
        print("DEV SCREEN COMPLETE — CERT NOT TOUCHED")
        print("FROZEN",dev_report["frozen"])
        print("REPORT",OUT/"DEV_PACKAGING_SCREEN.json")
        print("="*100)
        return

    print("[6/10] Frozen config: select witnesses for full 6991 top15",flush=True)
    full_idx=np.arange(len(qids),dtype=np.int32)
    full_short=shortlist30[:,:FULL_DEPTH]
    fvv,frr,fsim,fviews=select_witnesses("full15",full_idx,full_short,docs)
    ftexts=load_selected_texts(fvv,frr,fviews)
    fwitness_hash=sha(witness_cache("full15")[1])

    print("[7/10] Score ONLY frozen winner on full population",flush=True)
    full_sem,full_pairs,shared=score_mode(
        "full15",frozen_mode,title_on,full_idx,qids,questions,full_short,docs,names,
        fvv,frr,ftexts,fviews,fwitness_hash,None
    )
    if shared is not None:
        tok,model=shared
        del model,tok
        b1.clear_cuda()

    print("[8/10] Build fixed DEV-trained fusion",flush=True)
    S,snames=semantic_features(full_sem,fsim)
    X=np.concatenate([X45[:,:FULL_DEPTH],S],axis=2)
    y15=y[:,:FULL_DEPTH]

    print("[9/10] ONE-SHOT CERT evaluation",flush=True)
    cert_m,cert_rank=fit_dev_eval_cert(X,y15,full_short,qids,docs,golds,dev_idx,cert_idx)
    base_cert=subset_metrics(baseline,qids,docs,golds,cert_idx)
    delta=float(cert_m["recall_at_5"]-base_cert["recall_at_5"])
    single_delta=float(cert_m["single_gold_recall_at_5"]-base_cert["single_gold_recall_at_5"])
    proceed=delta>=.005 and single_delta>=-1e-12

    print("[10/10] Write certification report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage06a0.evidence_packaging_cert.v1",
        "status":"COMPLETE",
        "certification_integrity":{
            "dev_folds":list(DEV_FOLDS),
            "cert_folds":list(CERT_FOLDS),
            "cert_alternatives_evaluated":1,
            "frozen_before_cert":dev_report["frozen"],
        },
        "dev_screen":dev_report,
        "full_winner_pair_count":int(full_pairs),
        "fusion":{
            "model":"LogisticRegression C=.15 balanced",
            "train_population":"DEV folds0-2 only",
            "candidate_depth":FULL_DEPTH,
            "base_feature_dim":45,
            "new_feature_dim":len(snames),
            "new_feature_names":snames,
        },
        "cert":{
            "baseline_ce_lr":base_cert,
            "frozen_evidence_fusion":cert_m,
            "delta_recall_at5":delta,
            "delta_single_gold_recall_at5":single_delta,
        },
        "next_stage_gate":{
            "require_delta_recall_at5":.005,
            "require_nonnegative_single_delta":True,
            "decision":"PROCEED_TO_FULL_5FOLD_OOF" if proceed else "KILL_OR_REDESIGN",
        },
        "wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"CERTIFICATION.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    lines=[
        "# Stage 06A0 — Evidence Packaging Certification","",
        f"Frozen config: `{json.dumps(dev_report['frozen'],ensure_ascii=False)}`","",
        "| CERT method | R@5 | P@5 | Single | Multi |",
        "|---|---:|---:|---:|---:|",
        f"| Current CE-LR | {base_cert['recall_at_5']:.6f} | {base_cert['precision_at_5']:.6f} | {base_cert['single_gold_recall_at_5']:.6f} | {base_cert['multi_gold_recall_at_5']:.6f} |",
        f"| Frozen evidence fusion | {cert_m['recall_at_5']:.6f} | {cert_m['precision_at_5']:.6f} | {cert_m['single_gold_recall_at_5']:.6f} | {cert_m['multi_gold_recall_at_5']:.6f} |",
        "",
        f"- Δ Recall@5: **{delta:+.6f}**",
        f"- Δ single-gold: **{single_delta:+.6f}**",
        f"- Decision: **{report['next_stage_gate']['decision']}**","",
    ]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*108)
    print("FROZEN CONFIG:",dev_report["frozen"])
    print(f"CERT baseline  R={base_cert['recall_at_5']:.9f} P={base_cert['precision_at_5']:.9f} single={base_cert['single_gold_recall_at_5']:.9f} multi={base_cert['multi_gold_recall_at_5']:.9f}")
    print(f"CERT evidence  R={cert_m['recall_at_5']:.9f} P={cert_m['precision_at_5']:.9f} single={cert_m['single_gold_recall_at_5']:.9f} multi={cert_m['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA           R={delta:+.9f} single={single_delta:+.9f}")
    print("DECISION:",report["next_stage_gate"]["decision"])
    print("REPORT:",OUT/"REPORT.md")
    print("="*108)


if __name__=="__main__":
    main()

#!/usr/bin/env python
"""Common utilities for Stage07B teacher->student distillation."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage06_evidence.benchmark_evidence_packaging as a0

DEPTH=20
FULL_DEPTH=30
MAXLEN=1536

INSTRUCTION=(
    "Given a Vietnamese legal question, judge whether the candidate legal document "
    "contains provisions that are sufficient and directly relevant for answering the question. "
    "Consider legal scope, actors, actions, conditions, exceptions, procedures, sanctions, "
    "and explicit article or instrument references. Prefer legally applicable evidence over "
    "superficial lexical overlap."
)

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

CACHE=ROOT/"cache/stage07b_qwen_distill"
OUT=ROOT/"reports/stage07b_qwen_distill"

def load_world():
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy")
    done=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy")
    if short.shape[1]<FULL_DEPTH or ce.shape!=(len(qids),FULL_DEPTH) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03 shortlist/CE cache incomplete")
    Xflat,cq,cand,names45=b1.build_source_ce_features(short[:,:FULL_DEPTH],ce,sources)
    X45=Xflat.reshape(len(qids),FULL_DEPTH,-1).astype(np.float32)
    if X45.shape[-1]!=45:
        raise RuntimeError(f"expected 45D, got {X45.shape}")
    return qids,questions,golds,folds,docs,short[:,:FULL_DEPTH],X45

def labels(short,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    y=np.zeros(short.shape,np.float32)
    for i,q in enumerate(qids):
        gs={d2i[d] for d in golds[q]}
        y[i]=np.asarray([float(int(x) in gs) for x in short[i]],np.float32)
    return y

def build_evidence_world():
    qids,questions,golds,folds,docs,short,X45=load_world()
    qidx=np.arange(len(qids),dtype=np.int32)
    s20=short[:,:DEPTH]
    vv,rr,sim,views=a0.select_witnesses("stage07b_distill_top20",qidx,s20,docs)
    texts=a0.load_selected_texts(vv,rr,views)
    names=a0.load_names(docs)
    return qids,questions,golds,folds,docs,short,X45,vv,rr,sim,views,texts,names

def bundle(li,pos,d,names,vv,rr,views,texts):
    parts=[f"[LEGAL DOCUMENT TITLE]\n{a0.clean_title(names[int(d)])}"]
    seen=set()
    for fi,label in enumerate(("SEMANTIC EVIDENCE A","SEMANTIC EVIDENCE B")):
        v=views[int(vv[li,pos,fi])]
        raw=str(texts[v][int(rr[li,pos,fi])]).strip()
        key=" ".join(raw.split())
        if key and key not in seen:
            seen.add(key)
            parts.append(f"[{label}]\n{raw}")
    return "\n\n".join(parts)

def format_pair(q,doc):
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {doc}"

def build_pairs_for_queries(indices,qids,questions,short,names,vv,rr,views,texts):
    pairs=[]; owners=[]
    for qi in indices:
        q=questions[qids[int(qi)]]
        for pos,d in enumerate(short[int(qi),:DEPTH]):
            pairs.append(format_pair(q,bundle(int(qi),pos,d,names,vv,rr,views,texts)))
            owners.append((int(qi),pos))
    return pairs,owners

def tokenize_qwen(tok,pairs,max_length=MAXLEN):
    pre=tok.encode(PREFIX,add_special_tokens=False)
    suf=tok.encode(SUFFIX,add_special_tokens=False)
    avail=max_length-len(pre)-len(suf)
    raw=tok(pairs,padding=False,truncation=True,max_length=avail,
            add_special_tokens=False,return_attention_mask=False)
    feats=[{"input_ids":pre+x+suf} for x in raw["input_ids"]]
    return feats

def last_yes_no_score(model,inputs,yes_id,no_id):
    """Compute last-token yes-no logit difference without materializing vocab logits at every token."""
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    decoder=getattr(base,"model",None)
    if decoder is None:
        raise RuntimeError("Cannot locate Qwen decoder model")
    out=decoder(**inputs,return_dict=True)
    last=out.last_hidden_state[:,-1,:]
    logits=base.lm_head(last)
    return logits[:,yes_id]-logits[:,no_id]

def zrows_torch(x,eps=1e-5):
    mu=x.mean(dim=1,keepdim=True)
    sd=x.std(dim=1,keepdim=True,unbiased=False).clamp_min(eps)
    return (x-mu)/sd

def zrows_np(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-5,1.,sd)
    return (x-mu)/sd

def fit_lr_score(X45,y,train_idx,score_idx):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(C=.15,class_weight="balanced",solver="lbfgs",
                                  max_iter=300,tol=1e-5))
    ])
    m.fit(X45[train_idx].reshape(-1,X45.shape[-1]),y[train_idx].reshape(-1))
    s=m.decision_function(X45[score_idx].reshape(-1,X45.shape[-1]))
    return np.asarray(s,np.float32).reshape(len(score_idx),FULL_DEPTH)

def outer_priors(X45,y,folds,qids,outer_name):
    """Leakage-safe stacking priors for one outer student fold."""
    q2i={q:i for i,q in enumerate(qids)}
    train_folds=[f for f in folds if f!=outer_name]
    train_idx=np.asarray([q2i[q] for f in train_folds for q in folds[f]],np.int32)
    held_idx=np.asarray([q2i[q] for q in folds[outer_name]],np.int32)
    prior_train=np.full((len(qids),FULL_DEPTH),np.nan,np.float32)
    for inner in train_folds:
        val=np.asarray([q2i[q] for q in folds[inner]],np.int32)
        fit=np.asarray([q2i[q] for f in train_folds if f!=inner for q in folds[f]],np.int32)
        prior_train[val]=fit_lr_score(X45,y,fit,val)
    if not np.isfinite(prior_train[train_idx]).all():
        raise RuntimeError("inner-crossfit prior incomplete")
    prior_held=fit_lr_score(X45,y,train_idx,held_idx)
    return train_idx,held_idx,prior_train[train_idx],prior_held

#!/usr/bin/env python
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np

BASE=Path("/content/stage07b")
PAYLOAD=BASE/"payload"
WORK=Path("/content/stage07b_work")
DEFAULT_DRIVE=Path("/content/drive/MyDrive/DSC2026/stage07b")
DEPTH=20
FULL=30
MAXLEN=1536

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def load_payload():
    short=np.load(PAYLOAD/"shortlist_top30.i4.npy")
    X45=np.load(PAYLOAD/"X45.f32.npy")
    y=np.load(PAYLOAD/"labels_top30.u1.npy")
    gc=np.load(PAYLOAD/"gold_count.i2.npy")
    fold=np.load(PAYLOAD/"fold_id.i1.npy")
    with (PAYLOAD/"evidence_top20.pkl").open("rb") as f:evidence=pickle.load(f)
    qids=json.loads((PAYLOAD/"qids.json").read_text(encoding="utf-8"))
    n=len(qids)
    if short.shape!=(n,FULL) or X45.shape!=(n,FULL,45) or y.shape!=(n,FULL):
        raise RuntimeError("payload shape drift")
    if len(evidence)!=n or any(len(x)!=DEPTH for x in evidence):
        raise RuntimeError("evidence shape drift")
    return qids,short,X45,y,gc,fold,evidence

def qwen_features(tok,pairs):
    pre=tok.encode(PREFIX,add_special_tokens=False)
    suf=tok.encode(SUFFIX,add_special_tokens=False)
    avail=MAXLEN-len(pre)-len(suf)
    raw=tok(pairs,padding=False,truncation=True,max_length=avail,
            add_special_tokens=False,return_attention_mask=False)
    return [{"input_ids":pre+x+suf} for x in raw["input_ids"]]

def yes_no_ids(tok):
    yes=tok.encode("yes",add_special_tokens=False)
    no=tok.encode("no",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1:
        raise RuntimeError(f"Expected one-token yes/no, got yes={yes} no={no}")
    return yes[0],no[0]

def last_score(model,batch,yes_id,no_id):
    base=model
    decoder=getattr(base,"model",None)
    lm=getattr(base,"lm_head",None)
    if decoder is None or lm is None:raise RuntimeError("Qwen CausalLM structure drift")
    out=decoder(**batch,return_dict=True)
    last=out.last_hidden_state[:,-1,:]
    logits=lm(last)
    return logits[:,yes_id]-logits[:,no_id]

def zrows_np(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True);sd=x.std(1,keepdims=True)
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
    m.fit(X45[train_idx].reshape(-1,45),y[train_idx].reshape(-1))
    return np.asarray(
        m.decision_function(X45[score_idx].reshape(-1,45)),
        np.float32
    ).reshape(len(score_idx),FULL)

def outer_priors(X45,y,fold_id,outer):
    train_idx=np.where(fold_id!=outer)[0].astype(np.int32)
    held_idx=np.where(fold_id==outer)[0].astype(np.int32)
    prior_train=np.full((len(fold_id),FULL),np.nan,np.float32)
    for inner in range(5):
        if inner==outer:continue
        val=np.where(fold_id==inner)[0].astype(np.int32)
        fit=np.where((fold_id!=outer)&(fold_id!=inner))[0].astype(np.int32)
        prior_train[val]=fit_lr_score(X45,y,fit,val)
    if not np.isfinite(prior_train[train_idx]).all():
        raise RuntimeError("inner prior incomplete")
    return train_idx,held_idx,prior_train[train_idx],fit_lr_score(X45,y,train_idx,held_idx)

def ranking(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=np.lexsort((short[i],-score[i]))
    return out

def metric(rank_pos,y,gold_count,fold_id=None,fold=None):
    ids=np.arange(len(y)) if fold is None else np.where(fold_id==fold)[0]
    vals=[];hits=[]
    for i in ids:
        h=int(y[i,rank_pos[i,:5]].sum())
        hits.append(h);vals.append(h/int(gold_count[i]))
    vals=np.asarray(vals,float);hits=np.asarray(hits,float)
    single=np.asarray([gold_count[i]==1 for i in ids])
    multi=np.asarray([gold_count[i]>1 for i in ids])
    return {
        "queries":int(len(ids)),
        "recall_at_5":float(vals.mean()),
        "precision_at_5":float((hits/5).mean()),
        "single_gold_recall_at_5":float(vals[single].mean()) if single.any() else 0.,
        "multi_gold_recall_at_5":float(vals[multi].mean()) if multi.any() else 0.,
    }

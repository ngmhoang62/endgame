#!/usr/bin/env python
"""Stage 05B — Query-conditioned listwise selector tournament.

Goal
----
Test whether the large gap between current OOF R@5 (~.9426) and current
top-30 candidate oracle (~.9813) is primarily *selector capacity*.

No new neural retrieval/reranking inference is performed. The script reuses:
- frozen 6991-query OOF shortlist;
- six source top100 rankings/scores;
- completed AITeam Vietnamese reranker scores;
- cached AITeam query embeddings.

New signals:
1) deterministic legal-title metadata features from the corpus `name` field;
2) query-conditioned gating from the existing 1024D AITeam query embedding;
3) listwise multi-positive training: each query receives equal loss weight.

Models evaluated OOF:
- current 45D LogisticRegression parity reconstruction;
- 45D + title-feature LogisticRegression;
- QGate-Listwise, a tiny learned selector conditioned on query semantics.

The 5 frozen Stage00C folds are never crossed.

Promotion gate:
    OOF Recall@5 >= 0.960000

Run:
    python src/stage05_selector/run_qgate_selector_oof.py

Expected runtime: mostly CPU feature construction + a small GPU MLP; no cross-encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1

RAW=ROOT/"data/official_v1"
EVAL=ROOT/"data/evaluation_v2"
B4=ROOT/"cache/stage02b4_vi_screen"
CACHE=ROOT/"cache/stage05b_qgate_selector"
OUT=ROOT/"reports/stage05b_qgate_selector"

QEMB=B4/"aiteamvn_v1/query_embeddings.f32.npy"
QEMB_META=B4/"aiteamvn_v1/query_embeddings.json"
CE_CACHE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
CE_DONE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"

SHORT=30
PROMOTE_R=0.960000
SEEDS=(276,911)
EPOCHS=18
BATCH_Q=256

WS=re.compile(r"\s+",re.UNICODE)
TOKEN_RE=re.compile(r"\b[\wÀ-ỹĐđ]+\b",re.UNICODE)
NUM_RE=re.compile(r"\d+",re.UNICODE)
YEAR_RE=re.compile(r"\b(?:19|20)\d{2}\b")
ARTICLE_RE=re.compile(r"\bđiều\s+\d+[a-zđ]?\b",re.I)
CLAUSE_RE=re.compile(r"\bkhoản\s+\d+\b",re.I)
POINT_RE=re.compile(r"\bđiểm\s+(?:[a-zđ]|\d+)\b",re.I)

INSTRUMENTS=[
    ("bo_luat",("bộ luật","bo luat")),
    ("luat",("luật","luat")),
    ("nghi_dinh",("nghị định","nghi dinh")),
    ("thong_tu",("thông tư","thong tu")),
    ("quyet_dinh",("quyết định","quyet dinh")),
    ("nghi_quyet",("nghị quyết","nghi quyet")),
    ("phap_lenh",("pháp lệnh","phap lenh")),
    ("hien_phap",("hiến pháp","hien phap")),
]

QUERY_CUES=[
    ("sanction",("xử phạt","mức phạt","phạt tiền","hình phạt","truy cứu")),
    ("definition",("là gì","được hiểu là","định nghĩa","khái niệm")),
    ("procedure",("thủ tục","hồ sơ","trình tự","đăng ký","cấp phép","cấp giấy","thẩm quyền")),
    ("condition",("điều kiện","khi nào","trường hợp nào","được phép","đủ điều kiện")),
    ("crossref",("theo quy định tại","căn cứ","theo điều","theo khoản","tại điều","tại khoản")),
    ("howmany",("bao nhiêu","mức bao nhiêu","thời hạn","bao lâu")),
    ("who",("ai ","người nào","đối tượng nào","cơ quan nào")),
    ("what",("là gì","gồm những gì","nội dung gì","quy định gì")),
]


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def ntext(s):
    s=unicodedata.normalize("NFKC",str(s or "")).lower()
    s=s.replace("-"," ").replace("_"," ")
    return WS.sub(" ",s).strip()


def toks(s):
    return TOKEN_RE.findall(ntext(s))


def q_hand(question):
    t=ntext(question); ts=toks(t)
    f=[
        min(len(ts),60)/60.0,
        min(len(t),300)/300.0,
        float(bool(ARTICLE_RE.search(t))),
        float(bool(CLAUSE_RE.search(t))),
        float(bool(POINT_RE.search(t))),
        float(bool(YEAR_RE.search(t))),
        float("không" in t or "chưa" in t),
        float("nếu " in t or "trong trường hợp" in t or "khi " in t),
    ]
    for _,phr in QUERY_CUES:
        f.append(float(any(x in t for x in phr)))
    return np.asarray(f,dtype=np.float32)


def instrument_types(text):
    t=ntext(text)
    return {name for name,phr in INSTRUMENTS if any(x in t for x in phr)}


def load_names(docs):
    got=[]; names=[]
    with (EVAL/"retrieval_corpus_8512.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                got.append(str(x["id"]))
                names.append(str(x.get("name") or ""))
    if got!=docs:
        raise RuntimeError("corpus doc order drift while loading title names")
    return names


def title_world(names):
    title_tokens=[toks(x) for x in names]
    df=Counter()
    for ts in title_tokens:
        df.update(set(ts))
    N=len(names)
    idf={t:math.log((N+1)/(df[t]+1))+1.0 for t in df}
    title_sets=[set(x) for x in title_tokens]
    title_nums=[set(NUM_RE.findall(ntext(x))) for x in names]
    title_years=[set(YEAR_RE.findall(ntext(x))) for x in names]
    title_inst=[instrument_types(x) for x in names]
    return title_tokens,title_sets,title_nums,title_years,title_inst,idf


def make_title_features(qids,questions,shortlist,names):
    print("[title] build deterministic metadata features",flush=True)
    title_tokens,title_sets,title_nums,title_years,title_inst,idf=title_world(names)
    X=np.zeros((len(qids),SHORT,12),dtype=np.float32)

    for qi,q in enumerate(qids):
        qt=toks(questions[q]); qs=set(qt)
        qnums=set(NUM_RE.findall(ntext(questions[q])))
        qyears=set(YEAR_RE.findall(ntext(questions[q])))
        qinst=instrument_types(questions[q])
        qidf=sum(idf.get(t,1.0) for t in qs) or 1.0
        qbig=set(zip(qt,qt[1:])) if len(qt)>1 else set()

        raw_scores=[]
        rows=[]
        for d0 in shortlist[qi]:
            d=int(d0)
            ts=title_sets[d]
            inter=qs & ts
            idfs=sum(idf.get(t,1.0) for t in inter)
            union=qs|ts
            title_len=max(len(ts),1)
            qcov=len(inter)/max(len(qs),1)
            tcov=len(inter)/title_len
            jac=len(inter)/max(len(union),1)
            nq=len(qnums & title_nums[d])
            ny=len(qyears & title_years[d])
            inst=float(bool(qinst & title_inst[d]))
            tt=title_tokens[d]
            tbig=set(zip(tt,tt[1:])) if len(tt)>1 else set()
            bigcov=len(qbig & tbig)/max(len(qbig),1)
            phrase=float(bool(ntext(questions[q]) and ntext(questions[q]) in ntext(names[d])))
            score=idfs/qidf
            raw_scores.append(score)
            rows.append([
                float(len(inter)),
                float(idfs),
                float(score),
                float(qcov),
                float(tcov),
                float(jac),
                float(nq),
                float(ny),
                inst,
                float(bigcov),
                phrase,
                0.0, # rank filled below
            ])

        arr=np.asarray(raw_scores,dtype=np.float64)
        order=np.argsort(-arr,kind="stable")
        inv=np.empty(SHORT,dtype=np.int32)
        inv[order]=np.arange(1,SHORT+1)
        mu=float(arr.mean()); sd=float(arr.std())
        if sd<1e-8: sd=1.0
        for j in range(SHORT):
            rows[j][10]=float((arr[j]-mu)/sd) # replace phrase slot with zscore
            rows[j][11]=float(inv[j]/SHORT)
            X[qi,j]=rows[j]

        if (qi+1)%1000==0 or qi+1==len(qids):
            print(f"[title] {qi+1}/{len(qids)}",flush=True)

    return X


def metrics_from_rank(rank,qids,golds,docs,folds,stress):
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    overall=official_metrics(preds,golds,qids)
    pf={f:official_metrics(preds,golds,ids) for f,ids in folds.items()}
    st={}
    for n,ids in stress.items():
        use=[q for q in ids if q in preds]
        if use:
            st[n]=official_metrics(preds,golds,use)
    return {"overall":overall,"per_fold":pf,"stress":st}


def labels_for(shortlist,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    gidx=[{d2i[d] for d in golds[q]} for q in qids]
    y=np.zeros(shortlist.shape,dtype=np.float32)
    for qi in range(len(qids)):
        for j,d in enumerate(shortlist[qi]):
            y[qi,j]=float(int(d) in gidx[qi])
    return y


def current_lr_oof(X45,shortlist,y,qids,folds):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    out=np.empty_like(shortlist)
    q2i={q:i for i,q in enumerate(qids)}
    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        hs=set(map(int,held))
        train=np.asarray([i for i in range(len(qids)) if i not in hs],dtype=np.int32)

        model=Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(
                C=.15,class_weight="balanced",solver="lbfgs",
                max_iter=300,tol=1e-5
            )),
        ])
        model.fit(
            X45[train].reshape(-1,X45.shape[-1]),
            y[train].reshape(-1)
        )
        for qi in held:
            p=model.predict_proba(X45[qi])[:,1]
            order=np.lexsort((shortlist[qi],-p))
            out[qi]=shortlist[qi,order]
        print(f"[LR45/{fn}] done",flush=True)
    return out


def extended_lr_oof(X,shortlist,y,qids,folds):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    out=np.empty_like(shortlist)
    q2i={q:i for i,q in enumerate(qids)}
    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        hs=set(map(int,held))
        train=np.asarray([i for i in range(len(qids)) if i not in hs],dtype=np.int32)
        model=Pipeline([
            ("scale",StandardScaler()),
            ("lr",LogisticRegression(
                C=.15,class_weight="balanced",solver="lbfgs",
                max_iter=300,tol=1e-5
            )),
        ])
        model.fit(X[train].reshape(-1,X.shape[-1]),y[train].reshape(-1))
        for qi in held:
            p=model.predict_proba(X[qi])[:,1]
            order=np.lexsort((shortlist[qi],-p))
            out[qi]=shortlist[qi,order]
        print(f"[LR+title/{fn}] done",flush=True)
    return out


def query_embed(qids):
    if not QEMB.exists() or not QEMB_META.exists():
        raise FileNotFoundError(f"Missing AIT query cache: {QEMB}")
    q=np.load(QEMB,mmap_mode="r")
    if q.shape!=(len(qids),1024):
        raise RuntimeError(f"AIT query embedding shape drift {q.shape}")
    if not np.isfinite(q).all():
        raise RuntimeError("nonfinite query embeddings")
    norm=np.linalg.norm(q,axis=1)
    if float(np.max(np.abs(norm-1)))>5e-3:
        raise RuntimeError("query embedding norm drift")
    return np.asarray(q,dtype=np.float32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qgate_oof(Xcand,Q,y,shortlist,qids,folds,epochs,batch_q,seeds):
    import torch
    import torch.nn as nn

    device="cuda" if torch.cuda.is_available() else "cpu"
    q2i={q:i for i,q in enumerate(qids)}
    ensemble=np.zeros(shortlist.shape,dtype=np.float64)
    fold_logs={}

    class QGate(nn.Module):
        def __init__(self,cdim,qdim):
            super().__init__()
            self.q=nn.Sequential(
                nn.Linear(qdim,96),nn.GELU(),
                nn.Linear(96,64),nn.GELU(),
            )
            self.c=nn.Sequential(
                nn.Linear(cdim,96),nn.GELU(),
                nn.Linear(96,64),nn.GELU(),
            )
            self.head=nn.Sequential(
                nn.Linear(64*3,64),nn.GELU(),
                nn.Linear(64,1),
            )
        def forward(self,c,q):
            # c [B,K,D], q [B,Q]
            qc=self.q(q)
            cc=self.c(c)
            qe=qc[:,None,:].expand(-1,c.shape[1],-1)
            z=torch.cat([cc,qe,cc*qe],dim=-1)
            return self.head(z).squeeze(-1)

    for fn,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32)
        hs=set(map(int,held))
        train=np.asarray([i for i in range(len(qids)) if i not in hs],dtype=np.int32)

        # Fold-clean scaling using training candidates only.
        flat=Xcand[train].reshape(-1,Xcand.shape[-1]).astype(np.float64)
        mu=flat.mean(axis=0)
        sd=flat.std(axis=0)
        sd=np.where(sd<1e-6,1.0,sd)
        Xtr=((Xcand[train]-mu)/sd).astype(np.float32)
        Xhe=((Xcand[held]-mu)/sd).astype(np.float32)

        fold_score=np.zeros((len(held),SHORT),dtype=np.float64)
        seed_meta=[]
        for seed in seeds:
            set_seed(seed)
            model=QGate(Xcand.shape[-1],Q.shape[-1]).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4)

            order_rng=np.random.default_rng(seed)
            losses=[]
            for ep in range(epochs):
                order=train.copy()
                order_rng.shuffle(order)
                ep_loss=[]
                model.train()
                for st in range(0,len(order),batch_q):
                    orig=order[st:st+batch_q]
                    # Xtr is indexed by training-array position, so map orig -> train positions.
                    pos=np.searchsorted(train,orig)
                    # train is sorted because constructed in natural order.
                    c=torch.from_numpy(Xtr[pos]).to(device)
                    q=torch.from_numpy(Q[orig]).to(device)
                    yy=torch.from_numpy(y[orig]).to(device)
                    cnt=yy.sum(dim=1,keepdim=True)
                    valid=cnt[:,0]>0
                    if not bool(valid.any()):
                        continue
                    score=model(c,q)
                    target=yy[valid]/cnt[valid]
                    logp=torch.log_softmax(score[valid],dim=1)
                    loss=-(target*logp).sum(dim=1).mean()

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    opt.step()
                    ep_loss.append(float(loss.detach().cpu()))
                losses.append(float(np.mean(ep_loss)))
            model.eval()
            pred=[]
            with torch.inference_mode():
                for st in range(0,len(held),batch_q):
                    c=torch.from_numpy(Xhe[st:st+batch_q]).to(device)
                    q=torch.from_numpy(Q[held[st:st+batch_q]]).to(device)
                    pred.append(model(c,q).float().cpu().numpy())
            fold_score += np.concatenate(pred,axis=0)/len(seeds)
            seed_meta.append({"seed":seed,"final_train_loss":losses[-1]})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ensemble[held]=fold_score
        fold_logs[fn]={"train_queries":len(train),"held_queries":len(held),"seeds":seed_meta}
        print(f"[QGate/{fn}] done",flush=True)

    rank=np.empty_like(shortlist)
    for qi in range(len(qids)):
        order=np.lexsort((shortlist[qi],-ensemble[qi]))
        rank[qi]=shortlist[qi,order]
    return rank,fold_logs,ensemble.astype(np.float32)


def recall_values(rank,qids,golds,docs):
    vals=[]
    for i,q in enumerate(qids):
        vals.append(len(set(docs[int(x)] for x in rank[i,:5]) & set(golds[q]))/len(set(golds[q])))
    return np.asarray(vals,dtype=np.float64)


def arm_oracle(arms,qids,golds,docs):
    rv={k:recall_values(v,qids,golds,docs) for k,v in arms.items()}
    mat=np.stack(list(rv.values()),axis=1)
    return {
        "arms":list(rv),
        "macro_best_arm_recall_at5":float(mat.max(axis=1).mean()),
        "queries_where_arms_disagree_on_recall":int(np.sum(np.max(mat,axis=1)>np.min(mat,axis=1))),
    }


def union_oracle(arms,qids,golds,docs):
    vals=[]
    sizes=[]
    for qi,q in enumerate(qids):
        u=set()
        for rank in arms.values():
            u.update(docs[int(x)] for x in rank[qi,:5])
        g=set(golds[q])
        # Oracle can pick at most 5 documents from union.
        hits=min(5,len(u & g))
        vals.append(hits/len(g))
        sizes.append(len(u))
    return {"macro_recall_cap5":float(np.mean(vals)),"mean_union_size":float(np.mean(sizes))}


def source_rank(source_idx):
    return np.asarray(source_idx[:,:SHORT],dtype=np.int32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=EPOCHS)
    ap.add_argument("--batch-q",type=int,default=BATCH_Q)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()

    print("[1/8] Load frozen OOF world",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    shortlist,_=b1.load_shortlist(len(qids))
    sources=b1.load_sources(len(qids))
    ce=np.load(CE_CACHE)
    done=np.load(CE_DONE)
    if ce.shape!=(len(qids),SHORT) or done.shape!=(len(qids),) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03B1 CE cache incomplete")
    if not np.isfinite(ce).all():
        raise RuntimeError("nonfinite CE scores")

    print("[2/8] Rebuild current 45D features",flush=True)
    Xflat,cq,cand,names45=b1.build_source_ce_features(shortlist,ce,sources)
    X45=Xflat.reshape(len(qids),SHORT,-1)
    if X45.shape[-1]!=45:
        raise RuntimeError(f"expected 45D, got {X45.shape}")
    y=labels_for(shortlist,qids,golds,docs)

    print("[3/8] Current LR parity",flush=True)
    lr45=current_lr_oof(X45,shortlist,y,qids,folds)
    lr45_metrics=metrics_from_rank(lr45,qids,golds,docs,folds,stress)
    parity=lr45_metrics["overall"]["recall_at_5"]
    if abs(parity-0.9425976255185238)>2e-6:
        raise RuntimeError(f"Stage03B1 parity failed {parity}")

    print("[4/8] Headroom among existing arms",flush=True)
    ce_rank=b1.rank_by_scores(shortlist,ce)
    rankonly=shortlist.copy()
    zr=b1.zrows(np.broadcast_to(-np.arange(1,SHORT+1,dtype=np.float32)[None,:],shortlist.shape))
    zc=b1.zrows(ce)
    blend015=np.empty_like(shortlist)
    score=.85*zr+.15*zc
    for i in range(len(qids)):
        blend015[i]=shortlist[i,np.lexsort((shortlist[i],-score[i]))]

    arms={
        "rankonly":rankonly,
        "ce_lr":lr45,
        "blend015":blend015,
        "ce_only":ce_rank,
    }
    for s in b1.SOURCE_NAMES:
        arms["src_"+s]=source_rank(sources[s][0])
    headroom={
        "best_arm_oracle":arm_oracle(arms,qids,golds,docs),
        "union_oracle_rank_ce_lr":union_oracle(
            {"rankonly":rankonly,"ce_lr":lr45},qids,golds,docs),
        "union_oracle_core3":union_oracle(
            {"rankonly":rankonly,"ce_lr":lr45,"blend015":blend015},qids,golds,docs),
        "union_oracle_all_arms":union_oracle(arms,qids,golds,docs),
    }

    print("[5/8] Title/legal-metadata features",flush=True)
    doc_names=load_names(docs)
    T=make_title_features(qids,questions,shortlist,doc_names)
    Xext=np.concatenate([X45,T],axis=2)
    lr_title=extended_lr_oof(Xext,shortlist,y,qids,folds)
    lr_title_metrics=metrics_from_rank(lr_title,qids,golds,docs,folds,stress)

    print("[6/8] Query-conditioned Listwise QGate",flush=True)
    qe=query_embed(qids)
    qh=np.stack([q_hand(questions[q]) for q in qids],axis=0)
    Q=np.concatenate([qe,qh],axis=1).astype(np.float32)
    qgate,train_meta,qscore=qgate_oof(
        Xext,Q,y,shortlist,qids,folds,args.epochs,args.batch_q,SEEDS
    )
    qgate_metrics=metrics_from_rank(qgate,qids,golds,docs,folds,stress)
    np.save(CACHE/"qgate_oof_scores.f32.npy",qscore)

    print("[7/8] Compare errors/headroom",flush=True)
    methods={
        "stage03b1_ce_lr":lr45_metrics,
        "lr_plus_title":lr_title_metrics,
        "qgate_listwise":qgate_metrics,
    }
    best=max(methods,key=lambda k:methods[k]["overall"]["recall_at_5"])
    best_r=methods[best]["overall"]["recall_at_5"]

    pair_union=union_oracle(
        {"ce_lr":lr45,"qgate":qgate},qids,golds,docs
    )
    pair_arm=arm_oracle(
        {"ce_lr":lr45,"qgate":qgate},qids,golds,docs
    )

    print("[8/8] Write report",flush=True)
    report={
        "schema_version":"dsc2026.endgame.stage05b.qgate_selector_oof.v1",
        "status":"COMPLETE",
        "claim_boundary":"5-fold frozen OOF only",
        "no_new_retrieval_or_reranker_inference":True,
        "features":{
            "current":names45,
            "title_feature_count":int(T.shape[-1]),
            "candidate_feature_dim":int(Xext.shape[-1]),
            "query_embedding_dim":1024,
            "query_handcrafted_dim":int(qh.shape[-1]),
        },
        "methods":methods,
        "existing_arm_headroom":headroom,
        "ce_lr_vs_qgate":{
            "best_arm_oracle":pair_arm,
            "union_oracle":pair_union,
        },
        "qgate_training":train_meta,
        "promotion":{
            "gate_recall_at5":PROMOTE_R,
            "best_method":best,
            "best_recall_at5":best_r,
            "decision":"PROMOTE" if best_r>=PROMOTE_R else "DO_NOT_PROMOTE",
        },
        "wall_seconds":time.perf_counter()-t0,
    }
    (OUT/"SELECTOR_TOURNAMENT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    lines=[
        "# Stage 05B — Query-Conditioned Selector Tournament","",
        "| Method | OOF R@5 | P@5 | Single | Multi |",
        "|---|---:|---:|---:|---:|",
    ]
    for name,m in methods.items():
        o=m["overall"]
        lines.append(
            f"| {name} | {o['recall_at_5']:.6f} | {o['precision_at_5']:.6f} | "
            f"{o['single_gold_recall_at_5']:.6f} | {o['multi_gold_recall_at_5']:.6f} |"
        )
    lines += [
        "",
        f"- Existing-arm best-arm oracle: **{headroom['best_arm_oracle']['macro_best_arm_recall_at5']:.6f}**",
        f"- Rank-only ∪ CE-LR top5 cap-5 oracle: **{headroom['union_oracle_rank_ce_lr']['macro_recall_cap5']:.6f}**",
        f"- All-arm top5 cap-5 oracle: **{headroom['union_oracle_all_arms']['macro_recall_cap5']:.6f}**",
        f"- CE-LR vs QGate best-arm oracle: **{pair_arm['macro_best_arm_recall_at5']:.6f}**",
        f"- CE-LR ∪ QGate cap-5 oracle: **{pair_union['macro_recall_cap5']:.6f}**",
        "",
        f"**Decision: {report['promotion']['decision']}** "
        f"(gate R@5 >= {PROMOTE_R:.3f}; best={best} {best_r:.6f})",
        "",
    ]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*100)
    for name,m in methods.items():
        o=m["overall"]
        print(f"{name:24s} R={o['recall_at_5']:.9f} P={o['precision_at_5']:.9f}")
    print("BEST ARM ORACLE :",headroom["best_arm_oracle"]["macro_best_arm_recall_at5"])
    print("ALL UNION ORACLE:",headroom["union_oracle_all_arms"]["macro_recall_cap5"])
    print("DECISION         :",report["promotion"]["decision"])
    print("REPORT           :",OUT/"REPORT.md")
    print("="*100)


if __name__=="__main__":
    main()

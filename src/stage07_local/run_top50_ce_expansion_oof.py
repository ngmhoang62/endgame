#!/usr/bin/env python
"""
Stage07M — Top-50 Cross-Encoder Expansion + Conservative Admission.

Core hypothesis
---------------
Stage07F-H showed that a wide pool has huge oracle headroom, but shallow
retrieval/embedding features cannot safely identify the right outside-top30
candidate. The missing signal is direct cross-encoder relevance outside top30.

Stage03B1 scored AITeamVN/Vietnamese_Reranker only on positions 1..30.
Stage07M scores ONLY positions 31..50 with the exact same pinned model and the
exact same historical two-lexical-window contract, then evaluates:

  1) CE-only top50
  2) strict 5-fold 45D CE-LR over top50
  3) conservative Stage07D top4 + one learned slot5 admission from top50,
     where admission depth/threshold are selected by inner CV only.

No new foundation model. No teacher. No distillation.
Existing approved model:
  AITeamVN/Vietnamese_Reranker
  SHA f536976248403314225d7fdfdbc87f0e9516a54e

The extra CE cache is resumable and never overwrites Stage03B1 top30 scores.

Run:
  python src/stage07_local/run_top50_ce_expansion_oof.py
"""
from __future__ import annotations

import argparse, gc, hashlib, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg

B0=ROOT/"cache/stage03b_reranker"
CACHE=ROOT/"cache/stage07m_top50_ce"
OUT=ROOT/"reports/stage07m_top50_ce"
TOP30=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy"
TOP30_DONE=ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_done.u1.npy"
D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"

DEPTH=50
EXTRA_START=30
EXTRA=20
QUERY_CHUNK=24
INITIAL_BATCH=8
BASE_D_R=0.9458518094693177
LR_C=.15

K_GRID=(20,30,40,50)
THRESH_QUANTILES=(50.,70.,80.,90.,95.,97.5,99.)

def stable(obj):
    return hashlib.sha256(
        json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

def sha256_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def zrows(a):
    a=np.asarray(a,np.float64)
    mu=a.mean(1,keepdims=True)
    sd=a.std(1,keepdims=True)
    sd=np.where(sd<1e-8,1.,sd)
    return ((a-mu)/sd).astype(np.float32)

def load_world():
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    top50=np.load(B0/"oof_rank_selector_top50_idx.npy",mmap_mode="r")
    if top50.shape!=(len(qids),DEPTH):
        raise RuntimeError(f"top50 drift {top50.shape}")
    top50=np.asarray(top50,np.int32)
    sources=b1.load_sources(len(qids))

    ce30=np.asarray(np.load(TOP30),np.float32)
    done=np.asarray(np.load(TOP30_DONE),np.uint8)
    if ce30.shape!=(len(qids),30) or int(done.sum())!=len(qids):
        raise RuntimeError("Stage03B1 top30 cache incomplete")

    ds=np.asarray(np.load(D_SCORE),np.float32)
    if ds.shape!=(len(qids),30) or not np.isfinite(ds).all():
        raise RuntimeError(f"Stage07D score drift {ds.shape}")

    return qids,questions,golds,folds,stress,docs,passages,top50,sources,ce30,ds

def cache_paths():
    CACHE.mkdir(parents=True,exist_ok=True)
    return (
        CACHE/"extra31_50_ce_scores.f32.npy",
        CACHE/"extra31_50_done.u1.npy",
        CACHE/"extra31_50_ce_scores.json",
    )

def open_extra_cache(nq,top50_sha,manifest):
    sp,dp,mp=cache_paths()
    contract={
        "schema":"stage07m.top50_extra_ce.v1",
        "reranker_sha":manifest["resolved_revision_sha"],
        "top50_sha":top50_sha,
        "positions":"31..50",
        "max_length":b1.MAX_LENGTH,
        "passages_per_doc":b1.PASSAGES_PER_DOC,
        "passage_contract":{
            "function":"stage03b1.top_passages",
            "window_words":b1.WINDOW,
            "overlap_words":b1.OVERLAP,
            "header_words":70,
        },
    }
    ch=stable(contract)
    ex=[p.exists() for p in (sp,dp,mp)]
    if any(ex) and not all(ex):
        raise RuntimeError("partial Stage07M CE cache; refusing overwrite")
    if all(ex):
        meta=json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("contract_hash")!=ch:
            raise RuntimeError("Stage07M CE cache contract mismatch")
        s=np.lib.format.open_memmap(sp,mode="r+")
        d=np.lib.format.open_memmap(dp,mode="r+")
        if s.shape!=(nq,EXTRA) or d.shape!=(nq,):
            raise RuntimeError("Stage07M CE cache shape drift")
        print(f"[CE50] RESUME {int(np.asarray(d).sum())}/{nq}",flush=True)
        return s,d,meta,sp,dp,mp,contract,ch

    s=np.lib.format.open_memmap(
        sp,mode="w+",dtype=np.float32,shape=(nq,EXTRA)
    )
    s[:]=np.nan;s.flush()
    d=np.lib.format.open_memmap(
        dp,mode="w+",dtype=np.uint8,shape=(nq,)
    )
    d[:]=0;d.flush()
    meta={
        "contract_hash":ch,"contract":contract,
        "status":"IN_PROGRESS","completed":0,"total":nq,
        "current_batch":INITIAL_BATCH,
    }
    mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return s,d,meta,sp,dp,mp,contract,ch

def prepare_chunk(qis,qids,questions,top50,passages):
    pairs=[];owners=[]
    for qi in qis:
        q=questions[qids[qi]]
        for ep,didx in enumerate(top50[qi,EXTRA_START:]):
            ps=b1.top_passages(
                q,passages[int(didx)],
                count=b1.PASSAGES_PER_DOC,
                window=b1.WINDOW,overlap=b1.OVERLAP
            )
            if not ps:ps=[""]
            for p in ps:
                pairs.append((q,p))
                owners.append((qi,ep))
    return qis,pairs,owners

def score_extra(qids,questions,top50,passages):
    manifest=b1.load_model_manifest()
    top50_sha=sha256_file(B0/"oof_rank_selector_top50_idx.npy")
    scores,done,meta,sp,dp,mp,contract,ch=open_extra_cache(
        len(qids),top50_sha,manifest
    )
    pending=[i for i in range(len(qids)) if int(done[i])==0]
    if not pending:
        print("[CE50] extra cache complete",flush=True)
        return np.asarray(scores,np.float32)

    tok,model=b1.load_model()
    batch=int(meta.get("current_batch",INITIAL_BATCH))
    chunks=[pending[s:s+QUERY_CHUNK] for s in range(0,len(pending),QUERY_CHUNK)]
    ex=ThreadPoolExecutor(max_workers=1,thread_name_prefix="stage07m-prefetch")
    fut=ex.submit(prepare_chunk,chunks[0],qids,questions,top50,passages)
    started=time.perf_counter();new=0

    try:
        for ci in range(len(chunks)):
            t=time.perf_counter()
            qis,pairs,owners=fut.result()
            wait=time.perf_counter()-t
            if ci+1<len(chunks):
                fut=ex.submit(
                    prepare_chunk,chunks[ci+1],qids,questions,top50,passages
                )

            t=time.perf_counter()
            vals,batch=b1.forward_pairs(tok,model,pairs,batch)
            gpu=time.perf_counter()-t
            if len(vals)!=len(owners):
                raise RuntimeError("pair cardinality drift")

            tmp={qi:np.full(EXTRA,-np.inf,np.float32) for qi in qis}
            for (qi,ep),v in zip(owners,vals):
                if v>tmp[qi][ep]:tmp[qi][ep]=float(v)

            for qi in qis:
                if not np.isfinite(tmp[qi]).all():
                    raise RuntimeError(f"nonfinite extra CE q={qids[qi]}")
                scores[qi]=tmp[qi];done[qi]=1;new+=1
            scores.flush();done.flush()

            completed=int(np.asarray(done).sum())
            meta={
                "contract_hash":ch,"contract":contract,
                "status":"PASS" if completed==len(qids) else "IN_PROGRESS",
                "completed":completed,"total":len(qids),
                "current_batch":batch,
                "last_prep_wait_seconds":wait,
                "last_gpu_seconds":gpu,
            }
            mp.write_text(
                json.dumps(meta,ensure_ascii=False,indent=2)+"\n",
                encoding="utf-8"
            )
            rate=new/max(time.perf_counter()-started,1e-9)
            print(
                f"[CE50] {completed}/{len(qids)} batch={batch} "
                f"rate={rate:.2f} q/s wait={wait:.2f}s gpu={gpu:.2f}s "
                f"pairs={len(pairs)}",
                flush=True
            )
    finally:
        ex.shutdown(wait=False,cancel_futures=True)
        del model,tok
        b1.clear_cuda()

    return np.asarray(scores,np.float32)

def build_features(top50,ce50,sources):
    """
    Exact Stage03B1 feature semantics generalized from shortlist depth 30 -> 50.
    6*5 + aggregate9 + selector2 + CE4 = 45D.
    """
    n=len(top50)
    names=[]
    for s in b1.SOURCE_NAMES:
        names += [s+"__present",s+"__rr10",s+"__rank50",s+"__z",s+"__gapz"]
    names += [
        "source_count","best_rr10","mean_rr10","rrf60",
        "min_rank50","mean_rank50","count_top5","count_top10","count_top20",
        "selector_rr10","selector_rank_norm50",
        "ce_raw","ce_z","ce_gap_z","ce_rr10",
    ]

    X=np.empty((n,DEPTH,len(names)),np.float32)
    cez=zrows(ce50)
    cerank=np.empty((n,DEPTH),np.int32)
    for qi in range(n):
        o=np.lexsort((top50[qi],-ce50[qi]))
        inv=np.empty(DEPTH,np.int32);inv[o]=np.arange(1,DEPTH+1)
        cerank[qi]=inv

    for qi in range(n):
        rm={};zm={};gm={}
        for s in b1.SOURCE_NAMES:
            idx,scr=sources[s]
            ids=np.asarray(idx[qi,:50],np.int32)
            s100=np.asarray(scr[qi,:],np.float64)
            mu=float(s100.mean());sd=float(s100.std())
            if not np.isfinite(sd) or sd<1e-8:sd=1.
            top=float(s100[0])
            rm[s]={int(d):r+1 for r,d in enumerate(ids)}
            zm[s]={int(d):float((float(scr[qi,r])-mu)/sd) for r,d in enumerate(ids)}
            gm[s]={int(d):float((top-float(scr[qi,r]))/sd) for r,d in enumerate(ids)}

        ce_sd=float(np.std(ce50[qi]))
        if ce_sd<1e-8:ce_sd=1.
        ce_top=float(np.max(ce50[qi]))

        for sp,d0 in enumerate(top50[qi]):
            d=int(d0);f=[];rrs=[];rs=[];c5=c10=c20=0;rrf=0.
            for s in b1.SOURCE_NAMES:
                r=rm[s].get(d)
                if r is None:
                    f += [0.,0.,1.2,-3.,4.]
                else:
                    rr=1/(10+r)
                    f += [1.,rr,r/50.,zm[s][d],gm[s][d]]
                    rrs.append(rr);rs.append(r);rrf+=1/(60+r)
                    c5+=r<=5;c10+=r<=10;c20+=r<=20
            cr=int(cerank[qi,sp])
            f += [
                float(len(rrs)),max(rrs) if rrs else 0.,
                float(np.mean(rrs)) if rrs else 0.,
                rrf,
                min(rs)/50. if rs else 1.2,
                float(np.mean(rs))/50. if rs else 1.2,
                float(c5),float(c10),float(c20),
                1./(10.+sp+1),(sp+1)/50.,
                float(ce50[qi,sp]),float(cez[qi,sp]),
                float((ce_top-ce50[qi,sp])/ce_sd),
                1./(10.+cr),
            ]
            X[qi,sp]=np.asarray(f,np.float32)

        if (qi+1)%1000==0 or qi+1==n:
            print(f"[features50] {qi+1}/{n}",flush=True)
    return X,names

def labels(top50,qids,golds,docs):
    d2i={d:i for i,d in enumerate(docs)}
    y=np.zeros(top50.shape,np.uint8)
    for qi,q in enumerate(qids):
        g={d2i[d] for d in golds[q]}
        y[qi]=np.asarray([int(int(d) in g) for d in top50[qi]],np.uint8)
    return y

def make_lr():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(
            C=LR_C,class_weight="balanced",solver="lbfgs",
            max_iter=300,tol=1e-5
        )),
    ])

def rank_scores(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-score[i]))]
    return out

def eval_rank(rank,qids,golds,docs,folds,stress):
    preds={q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)}
    return {
        "overall":official_metrics(preds,golds,qids),
        "per_fold":{f:official_metrics(preds,golds,ids) for f,ids in folds.items()},
        "stress":{
            n:official_metrics(preds,golds,[q for q in ids if q in preds])
            for n,ids in stress.items() if any(q in preds for q in ids)
        }
    }

def perq(rank,qids,golds,docs):
    out=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q]);p={docs[int(x)] for x in rank[i,:5]}
        out[i]=len(g&p)/len(g)
    return out

def fold_indices(qids,folds):
    q2i={q:i for i,q in enumerate(qids)}
    return {f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

def oof_lr50(X,y,top50,folds,qids):
    fi=fold_indices(qids,folds)
    score=np.empty((len(qids),DEPTH),np.float32)
    meta={}
    for fn,held in fi.items():
        hs=set(map(int,held))
        tr=np.asarray([i for i in range(len(qids)) if i not in hs],np.int32)
        m=make_lr()
        m.fit(X[tr].reshape(-1,X.shape[-1]),y[tr].reshape(-1))
        score[held]=m.predict_proba(X[held].reshape(-1,X.shape[-1]))[:,1].reshape(len(held),DEPTH)
        meta[fn]={"train_queries":int(len(tr)),"held_queries":int(len(held))}
        print(f"[LR50/{fn}] done",flush=True)
    return score,meta

def fold_recall(rank,idx,qids,golds,docs):
    vals=[]
    for qi0 in idx:
        qi=int(qi0);q=qids[qi]
        g=set(golds[q]);p={docs[int(x)] for x in rank[qi,:5]}
        vals.append(len(g&p)/len(g))
    return float(np.mean(vals))

def stage07d_rank(top50,ds):
    # Stage07D scores are aligned to selector positions 1..30.
    out=np.empty((len(top50),30),np.int32)
    for qi in range(len(top50)):
        o=np.lexsort((top50[qi,:30],-ds[qi]))
        out[qi]=top50[qi,:30][o]
    return out

def best_challenger(qi,p,top50,base_rank,k):
    top4=set(map(int,base_rank[qi,:4]))
    current=int(base_rank[qi,4])
    pos={int(d):j for j,d in enumerate(top50[qi,:k])}
    candidates=[int(d) for d in top50[qi,:k] if int(d) not in top4 and int(d)!=current]
    if not candidates:
        return None,-np.inf
    # compare in logit space for better calibration near 0/1.
    eps=1e-6
    curp=float(p[qi,pos[current]]) if current in pos else eps
    cur_log=math.log(max(curp,eps)/max(1-curp,eps))
    best=None
    for d in candidates:
        pp=float(p[qi,pos[d]])
        lg=math.log(max(pp,eps)/max(1-pp,eps))
        margin=lg-cur_log
        key=(margin,-d)
        if best is None or key>best[0]:
            best=(key,d,margin)
    return best[1],float(best[2])

def apply_gate(base_full,p,top50,indices,k,threshold):
    out=base_full.copy()
    changed=0
    for qi0 in indices:
        qi=int(qi0)
        d,m=best_challenger(qi,p,top50,base_full,k)
        if d is not None and m>threshold:
            out[qi,4]=int(d);changed+=1
    return out,changed

def nested_conservative(X,y,top50,base_full,folds,qids,golds,docs,args):
    """
    Outer-held prediction is scored by an LR trained on outer-train only.
    K + threshold are selected from 4 inner folds with LR refit each time.
    """
    fi=fold_indices(qids,folds)
    rescued=base_full.copy()
    fold_meta={}

    for outer_fn,held in fi.items():
        inner_names=[f for f in fi if f!=outer_fn]
        outer_train=np.concatenate([fi[f] for f in inner_names])

        # Inner OOF probabilities for rule selection.
        ip=np.full((len(qids),DEPTH),np.nan,np.float32)
        for inner in inner_names:
            va=fi[inner]
            tr=np.concatenate([fi[f] for f in inner_names if f!=inner])
            m=make_lr()
            m.fit(X[tr].reshape(-1,X.shape[-1]),y[tr].reshape(-1))
            ip[va]=m.predict_proba(X[va].reshape(-1,X.shape[-1]))[:,1].reshape(len(va),DEPTH)

        candidates=[]
        base_fold={f:fold_recall(base_full,fi[f],qids,golds,docs) for f in inner_names}
        for k in K_GRID:
            margins=[]
            for qi in outer_train:
                _,mg=best_challenger(int(qi),ip,top50,base_full,k)
                if np.isfinite(mg):margins.append(mg)
            arr=np.asarray(margins,np.float64)
            if len(arr)==0:continue
            thresholds=sorted(set(float(np.percentile(arr,q)) for q in THRESH_QUANTILES))
            thresholds.append(float("inf"))

            for th in thresholds:
                rr,ch=apply_gate(base_full,ip,top50,outer_train,k,th)
                ds=[]
                for f in inner_names:
                    ds.append(
                        fold_recall(rr,fi[f],qids,golds,docs)-base_fold[f]
                    )
                mean=float(np.mean(ds));sd=float(np.std(ds))
                pos=sum(x>0 for x in ds)
                cr=ch/max(len(outer_train),1)
                valid=(mean>0 and pos>=args.min_positive_inner
                       and cr<=args.max_change_rate)
                candidates.append({
                    "k":int(k),"threshold":float(th),
                    "mean_delta":mean,"sd_delta":sd,
                    "positive_inner_folds":int(pos),
                    "change_rate":float(cr),
                    "selection_score":float(mean-0.5*sd),
                    "fold_deltas":[float(x) for x in ds],
                    "valid":bool(valid),
                })

        valid=[x for x in candidates if x["valid"]]
        if not valid:
            fold_meta[outer_fn]={
                "selected":None,"held_changes":0,
                "held_delta":0.0,"valid_configs":0
            }
            print(f"[RESCUE/{outer_fn}] KEEP ALL",flush=True)
            continue
        valid.sort(
            key=lambda x:(x["selection_score"],x["mean_delta"],-x["change_rate"]),
            reverse=True
        )
        sel=valid[0]

        # Final outer-train LR, untouched outer-held.
        m=make_lr()
        m.fit(
            X[outer_train].reshape(-1,X.shape[-1]),
            y[outer_train].reshape(-1)
        )
        hp=m.predict_proba(X[held].reshape(-1,X.shape[-1]))[:,1].reshape(len(held),DEPTH)
        pfull=np.zeros((len(qids),DEPTH),np.float32)
        pfull[held]=hp

        tmp,ch=apply_gate(
            base_full,pfull,top50,held,sel["k"],sel["threshold"]
        )
        rescued[held,4]=tmp[held,4]
        hd=fold_recall(rescued,held,qids,golds,docs)-fold_recall(base_full,held,qids,golds,docs)
        fold_meta[outer_fn]={
            "selected":sel,"held_changes":int(ch),
            "held_change_rate":float(ch/len(held)),
            "held_delta":float(hd),
            "valid_configs":int(len(valid)),
        }
        print(
            f"[RESCUE/{outer_fn}] K={sel['k']} th={sel['threshold']:.4f} "
            f"inner={sel['mean_delta']:+.6f} pos={sel['positive_inner_folds']}/4 "
            f"change={100*sel['change_rate']:.2f}% | "
            f"HELD={hd:+.6f} changes={ch}",
            flush=True
        )

    return rescued,fold_meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-change-rate",type=float,default=.15)
    ap.add_argument("--min-positive-inner",type=int,default=3)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()

    print("[1/8] Load frozen world",flush=True)
    qids,questions,golds,folds,stress,docs,passages,top50,sources,ce30,ds=load_world()

    print("[2/8] Score ONLY positions 31..50 with existing AITeam reranker",flush=True)
    extra=score_extra(qids,questions,top50,passages)
    ce50=np.concatenate([ce30,extra],axis=1)
    if ce50.shape!=(len(qids),DEPTH) or not np.isfinite(ce50).all():
        raise RuntimeError("CE50 assembly drift")
    np.save(CACHE/"oof_top50_ce_scores.f32.npy",ce50)

    print("[3/8] Build generalized 45D features + labels",flush=True)
    X,names=build_features(top50,ce50,sources)
    y=labels(top50,qids,golds,docs)

    print("[4/8] CE-only and strict 5-fold LR50",flush=True)
    ce_rank=rank_scores(top50,ce50)
    lrscore,lrmeta=oof_lr50(X,y,top50,folds,qids)
    lr_rank=rank_scores(top50,lrscore)

    print("[5/8] Stage07D parity",flush=True)
    d30=stage07d_rank(top50,ds)
    base_full=np.empty((len(qids),DEPTH),np.int32)
    for qi in range(len(qids)):
        # Stage07D top30 order followed by untouched selector positions 31..50.
        seen=set(map(int,d30[qi]))
        tail=[int(d) for d in top50[qi] if int(d) not in seen]
        base_full[qi]=np.asarray(list(map(int,d30[qi]))+tail,np.int32)
    dm=eval_rank(base_full,qids,golds,docs,folds,stress)
    if abs(dm["overall"]["recall_at_5"]-BASE_D_R)>3e-6:
        raise RuntimeError(f"Stage07D parity failed {dm['overall']['recall_at_5']}")

    print("[6/8] Nested conservative slot5 admission using NEW CE50 signal",flush=True)
    rescued,rescue_meta=nested_conservative(
        X,y,top50,base_full,folds,qids,golds,docs,args
    )

    print("[7/8] Evaluate",flush=True)
    methods={
        "stage07d":dm,
        "ce_only_top50":eval_rank(ce_rank,qids,golds,docs,folds,stress),
        "lr45_top50":eval_rank(lr_rank,qids,golds,docs,folds,stress),
        "stage07d_top4_plus_ce50_rescue":eval_rank(rescued,qids,golds,docs,folds,stress),
    }

    base_q=perq(base_full,qids,golds,docs)
    effects={}
    for n,m in methods.items():
        if n=="stage07d":continue
        rank={"ce_only_top50":ce_rank,"lr45_top50":lr_rank,
              "stage07d_top4_plus_ce50_rescue":rescued}[n]
        qq=perq(rank,qids,golds,docs)
        effects[n]={
            "delta_recall":m["overall"]["recall_at_5"]-dm["overall"]["recall_at_5"],
            "wins":int((qq>base_q).sum()),
            "losses":int((qq<base_q).sum()),
        }

    # Diagnostic: how much oracle is actually available specifically outside top30.
    d2i={d:i for i,d in enumerate(docs)}
    outside_gold_queries=0;outside_gold_occ=0
    for qi,q in enumerate(qids):
        g={d2i[d] for d in golds[q]}
        t30=set(map(int,top50[qi,:30]));t50=set(map(int,top50[qi]))
        ext=(g&t50)-t30
        if ext:
            outside_gold_queries+=1
            outside_gold_occ+=len(ext)

    best=max(
        methods,key=lambda n:methods[n]["overall"]["recall_at_5"]
    )
    best_r=methods[best]["overall"]["recall_at_5"]
    decision=(
        "BREAKTHROUGH_TARGET_REACHED" if best_r>=.960 else
        "PROMOTE_TOP50_CE" if best_r>dm["overall"]["recall_at_5"]+.0015 else
        "KILL_TOP50_CE_EXPANSION"
    )

    print("[8/8] Save/report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage07m.top50_ce_expansion.v1",
        "status":"COMPLETE",
        "new_foundation_model":False,
        "teacher_used":False,
        "reranker":{
            "model":"AITeamVN/Vietnamese_Reranker",
            "new_scored_positions":"31..50 only",
            "same_stage03b1_contract":True,
        },
        "candidate_ceiling":{
            "top30_oracle_recall":0.9813498307347542,
            "top50_oracle_recall":0.9861345539503172,
            "queries_with_gold_in_31_50":int(outside_gold_queries),
            "gold_occurrences_in_31_50_not_top30":int(outside_gold_occ),
        },
        "features":{"dim":int(X.shape[-1]),"names":names},
        "methods":methods,
        "effects_vs_stage07d":effects,
        "lr50_fold_meta":lrmeta,
        "rescue_meta":rescue_meta,
        "decision":decision,
        "wall_seconds":time.perf_counter()-started,
    }
    (OUT/"OOF_REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    np.save(CACHE/"lr50_oof_scores.f32.npy",lrscore)
    np.save(CACHE/"rescued_oof_rank.i32.npy",rescued)

    print("="*124)
    for n,m in methods.items():
        o=m["overall"]
        if n=="stage07d":
            print(f"{n:36s} R={o['recall_at_5']:.9f} single={o['single_gold_recall_at_5']:.9f} multi={o['multi_gold_recall_at_5']:.9f}")
        else:
            e=effects[n]
            print(
                f"{n:36s} R={o['recall_at_5']:.9f} "
                f"DELTA={e['delta_recall']:+.9f} W/L={e['wins']}/{e['losses']} "
                f"single={o['single_gold_recall_at_5']:.9f} multi={o['multi_gold_recall_at_5']:.9f}"
            )
    print(
        f"outside30_gold_queries={outside_gold_queries} "
        f"outside30_gold_occurrences={outside_gold_occ}"
    )
    print("DECISION:",decision)
    print("REPORT:",OUT/"OOF_REPORT.json")
    print("="*124)

if __name__=="__main__":
    main()

#!/usr/bin/env python
"""
Stage07L — In-domain LoRA adaptation of the EXISTING AITeamVN/Vietnamese_Reranker.

This branch deliberately does NOT use the Qwen4B teacher.
It is therefore independent of the pending self-distillation rule clarification.

Why this branch
---------------
All tiny post-hoc selectors have plateaued around ~0.945. Instead of learning
another shallow head over frozen scores, adapt the strongest existing
cross-encoder itself to the 6991-query task.

Scientific split:
  TRAIN : fold_0 + fold_1
  DEV   : fold_2 (choose only blend alpha)
  CERT  : fold_3 + fold_4, touched only if DEV gate passes

Model:
  AITeamVN/Vietnamese_Reranker (already used/registered in ENDGAME)
  LoRA r=16 on attention query/value + trainable classifier.
  Final inference can merge the adapter into the same 0.568B base model.

Evidence package:
  one 512-token pair per candidate:
    [document title]
    + query-selected AIT witness
    + query-selected LAL witness
  with the available document-token budget split roughly 50/50 between the two
  witness families so one view cannot truncate the other completely.

Training:
  query-balanced hard pointwise BCE over top20:
    up to 2 hardest gold positives
    up to 3 strongest Stage07D non-gold negatives
  one pass over TRAIN queries, gradient accumulation by query.

Evaluation:
  adapted reranker scores top20.
  Frozen Stage07D scores remain the prior over top30.
  On DEV only, choose alpha in:
      z(Stage07D) + alpha * z(adapted_AIT_score) on first 20 candidates
  Then freeze alpha and evaluate CERT folds.

No new foundation model. No teacher. No distillation.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage06_evidence.benchmark_evidence_packaging as a0

MODEL=ROOT/"models/rerankers/aiteamvn-vietnamese-reranker"
MAN=ROOT/"reports/stage03b0_reranker_materialization/MODEL_MANIFEST.json"
D_SCORE=ROOT/"cache/stage07d_teacher_distill_head/distilled_head_oof_scores30.f32.npy"

CACHE=ROOT/"cache/stage07l_aiteam_gold_lora"
OUT=ROOT/"reports/stage07l_aiteam_gold_lora"
ADAPTER=CACHE/"adapter_train01"
TRAIN_META=CACHE/"adapter_train01.json"
RESUME=CACHE/"train_resume.pt"

TRAIN_FOLDS=("fold_0","fold_1")
DEV_FOLD="fold_2"
CERT_FOLDS=("fold_3","fold_4")

DEPTH=20
FULL=30
MAXLEN=512
POS_CAP=2
NEG_CAP=3
ACCUM_Q=8
LR=5e-5
WD=.01
SEED=276
ALPHAS=[round(i*.05,2) for i in range(21)]

def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def stable(x):
    return hashlib.sha256(
        json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def compute_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def world():
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    names=qg.load_names(docs)
    y=qg.labels_for(short,qids,golds,docs).astype(np.float32)
    if not D_SCORE.is_file():
        raise FileNotFoundError(D_SCORE)
    ds=np.asarray(np.load(D_SCORE),np.float32)
    if ds.shape!=(len(qids),FULL) or not np.isfinite(ds).all():
        raise RuntimeError(f"Stage07D score drift {ds.shape}")
    return qids,questions,golds,folds,stress,docs,short,names,y,ds

def split(qids,folds):
    q2i={q:i for i,q in enumerate(qids)}
    tr=np.asarray([q2i[q] for f in TRAIN_FOLDS for q in folds[f]],np.int32)
    dv=np.asarray([q2i[q] for q in folds[DEV_FOLD]],np.int32)
    cert={
        f:np.asarray([q2i[q] for q in folds[f]],np.int32)
        for f in CERT_FOLDS
    }
    return tr,dv,cert

def prepare_witnesses(tag,idx,short,docs):
    local=short[idx,:DEPTH]
    vv,rr,sim,views=a0.select_witnesses(tag,idx,local,docs)
    texts=a0.load_selected_texts(vv,rr,views)
    return local,vv,rr,views,texts

def model_load(trainable: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model

    tok=AutoTokenizer.from_pretrained(
        MODEL,local_files_only=True,trust_remote_code=True,use_fast=True
    )
    base=AutoModelForSequenceClassification.from_pretrained(
        MODEL,local_files_only=True,trust_remote_code=True,dtype=compute_dtype()
    )
    base.config.use_cache=False

    if trainable:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant":False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()

        model=get_peft_model(base,LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=["query","value"],
            modules_to_save=["classifier"],
            bias="none",
        ))
        # Trainables in FP32 for optimizer stability.
        for p in model.parameters():
            if p.requires_grad: p.data=p.data.float()
        try: model.enable_input_require_grads()
        except Exception: pass
    else:
        model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False)

    return tok,model.to("cuda")

def bundle(tok,q,title,li,pos,vv,rr,texts,views):
    """
    One deterministic query/document package.
    Budget is split across AIT and LAL witnesses so both survive truncation.
    """
    raw=[]
    for fi in range(2):
        v=views[int(vv[li,pos,fi])]
        raw.append(texts[v][int(rr[li,pos,fi])])

    title=" ".join(str(title or "").replace("-"," ").replace("_"," ").split())
    prefix=(f"[VĂN BẢN] {title}\n" if title else "")
    a_lab="[BẰNG CHỨNG AIT]\n"
    l_lab="\n[BẰNG CHỨNG LAL]\n"

    qn=len(tok.encode(q,add_special_tokens=False))
    special=int(tok.num_special_tokens_to_add(pair=True))
    fixed=len(tok.encode(prefix+a_lab+l_lab,add_special_tokens=False))
    budget=max(96,MAXLEN-qn-special-fixed-4)
    ait_budget=max(48,budget//2)
    lal_budget=max(48,budget-ait_budget)

    a_ids=tok.encode(raw[0],add_special_tokens=False)[:ait_budget]
    l_ids=tok.encode(raw[1],add_special_tokens=False)[:lal_budget]
    at=tok.decode(a_ids,skip_special_tokens=True)
    lt=tok.decode(l_ids,skip_special_tokens=True)
    return prefix+a_lab+at+l_lab+lt

def hard_group(shortrow,drow,yrow):
    """
    Up to 2 hardest golds (lowest Stage07D score) and 3 strongest wrong negatives.
    """
    pos=np.where(yrow[:DEPTH]>0)[0].tolist()
    neg=np.where(yrow[:DEPTH]<=0)[0].tolist()
    if not pos or not neg: return [],[]
    pos=sorted(pos,key=lambda j:(float(drow[j]),j))[:POS_CAP]
    neg=sorted(neg,key=lambda j:(-float(drow[j]),j))[:NEG_CAP]
    return pos,neg

def candidate_logit_train(tok,model,q,text):
    import torch
    enc=tok(
        q,text,padding=False,truncation=True,max_length=MAXLEN,
        return_tensors="pt"
    )
    enc={k:v.to("cuda") for k,v in enc.items()}
    with torch.autocast("cuda",dtype=compute_dtype()):
        x=model(**enc,return_dict=True).logits
    return x.view(-1)[0].float()

def train(tr,qids,questions,docs,short,names,y,ds,local,vv,rr,texts,views):
    import torch, torch.nn.functional as F
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup

    contract={
        "schema":"stage07l.aiteam_gold_lora.train.v1",
        "base_sha":rj(MAN)["resolved_revision_sha"],
        "train_qids_sha":stable([qids[int(i)] for i in tr]),
        "folds":list(TRAIN_FOLDS),
        "depth":DEPTH,"maxlen":MAXLEN,
        "evidence":"single title + balanced AIT/LAL witness bundle",
        "pos_cap":POS_CAP,"neg_cap":NEG_CAP,
        "accum_q":ACCUM_Q,"lr":LR,"wd":WD,"seed":SEED,
        "lora":"r16-a32-d05-query-value-classifier",
        "loss":"query-balanced hard BCE; Stage07D hard mining only",
        "teacher_used":False,
    }
    ch=stable(contract)

    if TRAIN_META.is_file() and ADAPTER.is_dir():
        m=rj(TRAIN_META)
        if m.get("contract_hash")==ch and m.get("status")=="PASS":
            print("[train] cache hit",flush=True)
            return m
        raise RuntimeError("training cache contract mismatch")

    random.seed(SEED);np.random.seed(SEED)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

    tok,model=model_load(True)
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    print(
        f"[train] dtype={compute_dtype()} trainable={sum(p.numel() for p in params):,}",
        flush=True
    )

    opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    order=list(range(len(tr)))
    random.Random(SEED).shuffle(order)
    total_steps=math.ceil(len(order)/ACCUM_Q)
    sch=get_cosine_schedule_with_warmup(
        opt,max(1,int(.1*total_steps)),total_steps
    )

    start=0
    if RESUME.is_file():
        st=torch.load(RESUME,map_location="cpu",weights_only=False)
        if st["contract_hash"]!=ch:
            raise RuntimeError("resume contract mismatch")
        set_peft_model_state_dict(model,st["adapter"])
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        start=int(st["position"])
        torch.set_rng_state(st["torch_cpu"])
        torch.cuda.set_rng_state_all(st["torch_cuda"])
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        print(f"[train] resume {start}/{len(order)}",flush=True)

    opt.zero_grad(set_to_none=True)
    t0=time.perf_counter()
    losses=[];cands=skipped=0

    for oi in range(start,len(order)):
        li=order[oi]
        qi=int(tr[li])
        q=questions[qids[qi]]
        ps,ns=hard_group(local[li],ds[qi],y[qi])
        if not ps or not ns:
            skipped+=1
        else:
            cs=ps+ns
            # Each query contributes 0.5 positive mass and 0.5 negative mass.
            weights=[.5/len(ps)]*len(ps)+[.5/len(ns)]*len(ns)
            targets=[1.]*len(ps)+[0.]*len(ns)

            for pos,w,tgt in zip(cs,weights,targets):
                d=int(local[li,pos])
                text=bundle(tok,q,names[d],li,pos,vv,rr,texts,views)
                score=candidate_logit_train(tok,model,q,text)
                target=torch.tensor(tgt,dtype=torch.float32,device="cuda")
                loss=F.binary_cross_entropy_with_logits(score,target)*w/ACCUM_Q
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss q={qids[qi]}")
                loss.backward()
                losses.append(float(loss.detach())*ACCUM_Q)
                cands+=1

        boundary=((oi+1)%ACCUM_Q==0 or oi+1==len(order))
        if boundary:
            torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            opt.step();sch.step();opt.zero_grad(set_to_none=True)

        if boundary and ((oi+1)%128==0 or oi+1==len(order)):
            st={
                "contract_hash":ch,"position":oi+1,
                "adapter":{k:v.detach().cpu()
                           for k,v in get_peft_model_state_dict(model).items()},
                "optimizer":opt.state_dict(),"scheduler":sch.state_dict(),
                "torch_cpu":torch.get_rng_state(),
                "torch_cuda":torch.cuda.get_rng_state_all(),
                "python":random.getstate(),"numpy":np.random.get_state(),
            }
            tmp=RESUME.with_suffix(".tmp")
            torch.save(st,tmp);tmp.replace(RESUME)

        if (oi+1)%32==0 or oi+1==len(order):
            elapsed=time.perf_counter()-t0
            done=oi+1-start
            eta=elapsed/max(done,1)*(len(order)-oi-1)
            print(
                f"[train] {oi+1}/{len(order)} loss128="
                f"{np.mean(losses[-128:]) if losses else float('nan'):.5f} "
                f"qps={done/max(elapsed,1e-9):.3f} candidates={cands} "
                f"skipped={skipped} eta_min={eta/60:.1f} "
                f"vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB",
                flush=True
            )

    ADAPTER.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(ADAPTER,safe_serialization=True)
    af=ADAPTER/"adapter_model.safetensors"
    meta={
        "status":"PASS","contract_hash":ch,"contract":contract,
        "trainable_parameters":sum(p.numel() for p in params),
        "queries":len(order),"skipped":skipped,"candidates":cands,
        "loss_tail":float(np.mean(losses[-128:])),
        "adapter_sha256":sha(af),
        "seconds":time.perf_counter()-t0,
        "peak_gib":torch.cuda.max_memory_reserved()/2**30,
    }
    TRAIN_META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    RESUME.unlink(missing_ok=True)
    del model,tok,opt,sch
    clear_cuda()
    return meta

def score_paths(tag,nq):
    d=CACHE/"scores"
    d.mkdir(parents=True,exist_ok=True)
    return (
        d/f"{tag}.f32.npy",
        d/f"{tag}.done.u1.npy",
        d/f"{tag}.json",
    )

def infer_forward(tok,model,qs,ts,batch):
    import torch
    vals=[];i=0;cur=batch
    while i<len(qs):
        j=min(i+cur,len(qs))
        try:
            enc=tok(
                qs[i:j],ts[i:j],padding=True,truncation=True,max_length=MAXLEN,
                return_tensors="pt"
            )
            enc={k:v.to("cuda") for k,v in enc.items()}
            with torch.inference_mode(),torch.autocast("cuda",dtype=compute_dtype()):
                x=model(**enc,return_dict=True).logits.view(-1).float()
            vals.extend(float(v) for v in x.cpu().tolist())
            i=j
        except torch.cuda.OutOfMemoryError:
            clear_cuda()
            if cur<=1:raise
            cur=max(1,cur//2)
            print(f"[infer] OOM -> batch={cur}",flush=True)
    return vals,cur

def score_split(tag,idx,qids,questions,docs,short,names,local,vv,rr,texts,views,
                adapter_meta,batch):
    sp,dp,mp=score_paths(tag,len(idx))
    contract={
        "schema":"stage07l.aiteam_gold_lora.score.v1",
        "tag":tag,
        "adapter_sha":adapter_meta["adapter_sha256"],
        "qids_sha":stable([qids[int(i)] for i in idx]),
        "depth":DEPTH,"maxlen":MAXLEN,
        "evidence":"balanced AIT/LAL bundle",
    }
    ch=stable(contract)
    ex=[sp.exists(),dp.exists(),mp.exists()]
    if any(ex) and not all(ex):
        raise RuntimeError(f"{tag}: partial score cache")
    if all(ex):
        meta=rj(mp)
        if meta["contract_hash"]!=ch:
            raise RuntimeError(f"{tag}: score cache mismatch")
        scores=np.lib.format.open_memmap(sp,mode="r+")
        done=np.lib.format.open_memmap(dp,mode="r+")
    else:
        scores=np.lib.format.open_memmap(
            sp,mode="w+",dtype=np.float32,shape=(len(idx),DEPTH)
        )
        scores[:]=np.nan;scores.flush()
        done=np.lib.format.open_memmap(
            dp,mode="w+",dtype=np.uint8,shape=(len(idx),)
        )
        done[:]=0;done.flush()
        meta={"contract_hash":ch,"contract":contract,
              "completed":0,"current_batch":batch}

    pending=[i for i in range(len(idx)) if int(done[i])==0]
    if not pending:
        print(f"[{tag}] score cache complete",flush=True)
        return np.asarray(scores,dtype=np.float32)

    tok,model=model_load(False)
    model.eval()
    cur=int(meta.get("current_batch",batch))
    started=time.perf_counter();new=0

    for li in pending:
        qi=int(idx[li]);q=questions[qids[qi]]
        qs=[];ts=[]
        for pos,d0 in enumerate(local[li]):
            d=int(d0)
            qs.append(q)
            ts.append(bundle(tok,q,names[d],li,pos,vv,rr,texts,views))
        vals,cur=infer_forward(tok,model,qs,ts,cur)
        if len(vals)!=DEPTH:raise RuntimeError("score cardinality drift")
        scores[li]=np.asarray(vals,np.float32)
        done[li]=1;new+=1

        if new%16==0 or new==len(pending):
            scores.flush();done.flush()
            completed=int(np.asarray(done).sum())
            meta={
                "contract_hash":ch,"contract":contract,
                "completed":completed,
                "status":"PASS" if completed==len(idx) else "IN_PROGRESS",
                "current_batch":cur,
            }
            mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
            print(
                f"[{tag}] {completed}/{len(idx)} batch={cur} "
                f"rate={new/max(time.perf_counter()-started,1e-9):.2f} q/s",
                flush=True
            )

    del model,tok
    clear_cuda()
    return np.asarray(scores,dtype=np.float32)

def rank_from_score(shortrow,score):
    out=np.empty_like(shortrow)
    for i in range(len(shortrow)):
        o=np.lexsort((shortrow[i],-score[i]))
        out[i]=shortrow[i,o]
    return out

def metric_for_indices(rank,idx,qids,golds,docs):
    preds={}
    ids=[]
    for li,qi0 in enumerate(idx):
        qi=int(qi0);q=qids[qi]
        preds[q]=[docs[int(x)] for x in rank[li,:5]]
        ids.append(q)
    return official_metrics(preds,golds,ids)

def perq_for_indices(rank,idx,qids,golds,docs):
    vals=[]
    for li,qi0 in enumerate(idx):
        qi=int(qi0);q=qids[qi]
        g=set(golds[q]);p={docs[int(x)] for x in rank[li,:5]}
        vals.append(len(g&p)/len(g))
    return np.asarray(vals,np.float64)

def blend_rank(short_local,dscore,adapt,alpha):
    """
    short_local: [n,20] same ids as first20 in full shortlist
    dscore: [n,30] Stage07D scores aligned to full shortlist
    adapt: [n,20]
    """
    dz=(dscore-dscore.mean(1,keepdims=True))/np.maximum(dscore.std(1,keepdims=True),1e-6)
    az=(adapt-adapt.mean(1,keepdims=True))/np.maximum(adapt.std(1,keepdims=True),1e-6)
    s=dz.copy()
    s[:,:DEPTH]+=alpha*az
    full_short=np.empty((len(short_local),FULL),np.int32)
    # Caller passes short_local that is a view of original first20; positions20:30
    # are filled outside this function by attaching original short.
    return s

def full_rank(short_full,score):
    out=np.empty_like(short_full)
    for i in range(len(short_full)):
        out[i]=short_full[i,np.lexsort((short_full[i],-score[i]))]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--infer-batch",type=int,default=24)
    ap.add_argument("--dev-min-delta",type=float,default=.0015)
    args=ap.parse_args()

    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)

    print("[1/9] Load frozen world",flush=True)
    qids,questions,golds,folds,stress,docs,short,names,y,ds=world()
    tr,dv,cert=split(qids,folds)
    print(
        f"  TRAIN={len(tr)} DEV={len(dv)} "
        f"CERT={sum(len(x) for x in cert.values())} (untouched initially)",
        flush=True
    )

    print("[2/9] TRAIN evidence witnesses",flush=True)
    tl,tv,trr,tviews,ttexts=prepare_witnesses(
        "stage07l_train01_top20",tr,short,docs
    )

    print("[3/9] Train gold-only AITeam LoRA",flush=True)
    tm=train(
        tr,qids,questions,docs,short,names,y,ds,
        tl,tv,trr,ttexts,tviews
    )
    del tl,tv,trr,ttexts
    gc.collect()

    print("[4/9] DEV fold2 evidence + scoring",flush=True)
    dl,dvv,drr,dviews,dtexts=prepare_witnesses(
        "stage07l_dev2_top20",dv,short,docs
    )
    adapt=score_split(
        "fold2",dv,qids,questions,docs,short,names,
        dl,dvv,drr,dtexts,dviews,tm,args.infer_batch
    )

    print("[5/9] DEV alpha selection vs frozen Stage07D",flush=True)
    short_d=short[dv]
    base_rank=full_rank(short_d,ds[dv])
    bm=metric_for_indices(base_rank,dv,qids,golds,docs)
    bq=perq_for_indices(base_rank,dv,qids,golds,docs)

    grid=[];best=None;best_rank=None
    for a in ALPHAS:
        score=blend_rank(short_d[:,:DEPTH],ds[dv],adapt,a)
        rr=full_rank(short_d,score)
        mm=metric_for_indices(rr,dv,qids,golds,docs)
        qq=perq_for_indices(rr,dv,qids,golds,docs)
        rec={
            "alpha":a,
            **mm,
            "wins":int((qq>bq).sum()),
            "losses":int((qq<bq).sum()),
            "delta":float(mm["recall_at_5"]-bm["recall_at_5"]),
        }
        grid.append(rec)
        key=(mm["recall_at_5"],mm["precision_at_5"],-a)
        if best is None or key>best[0]:
            best=(key,rec);best_rank=rr

    win=best[1]
    alpha=float(win["alpha"])
    dev_delta=float(win["delta"])
    dev_gate=(dev_delta>=args.dev_min_delta and win["wins"]>=win["losses"])

    print(
        f"  DEV Stage07D={bm['recall_at_5']:.9f} "
        f"best={win['recall_at_5']:.9f} delta={dev_delta:+.9f} "
        f"alpha={alpha:.2f} W/L={win['wins']}/{win['losses']}",
        flush=True
    )

    report={
        "schema":"dsc2026.endgame.stage07l.aiteam_gold_lora.v1",
        "status":"DEV_COMPLETE",
        "teacher_used":False,
        "new_foundation_model":False,
        "model":"AITeamVN/Vietnamese_Reranker",
        "partition":{
            "train_folds":list(TRAIN_FOLDS),
            "dev_fold":DEV_FOLD,
            "cert_folds":list(CERT_FOLDS),
        },
        "training":tm,
        "dev":{
            "stage07d":bm,
            "winner":win,
            "alpha_grid":grid,
            "gate":{
                "min_delta":args.dev_min_delta,
                "decision":"PROCEED_CERT" if dev_gate else "KILL_ON_DEV",
            }
        },
        "cert":{},
    }

    if not dev_gate:
        print("[6/9] CERT remains untouched — DEV gate failed",flush=True)
        report["status"]="KILL_ON_DEV_CERT_UNTOUCHED"
        (OUT/"REPORT.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
        )
        print("="*118)
        print(f"DEV Stage07D R={bm['recall_at_5']:.9f}")
        print(
            f"DEV ADAPTED-BLEND R={win['recall_at_5']:.9f} "
            f"DELTA={dev_delta:+.9f} alpha={alpha:.2f} "
            f"W/L={win['wins']}/{win['losses']}"
        )
        print("DECISION: KILL_ON_DEV_CERT_UNTOUCHED")
        print("REPORT:",OUT/"REPORT.json")
        print("="*118)
        return

    del dl,dvv,drr,dtexts,adapt
    clear_cuda()

    print("[6/9] DEV passed -> touch CERT with frozen alpha",flush=True)
    cert_deltas=[]
    for fn in CERT_FOLDS:
        idx=cert[fn]
        print(f"[CERT/{fn}] evidence",flush=True)
        cl,cv,cr,cviews,ctexts=prepare_witnesses(
            f"stage07l_{fn}_top20",idx,short,docs
        )
        cs=score_split(
            fn,idx,qids,questions,docs,short,names,
            cl,cv,cr,ctexts,cviews,tm,args.infer_batch
        )

        sr=short[idx]
        br=full_rank(sr,ds[idx])
        bmet=metric_for_indices(br,idx,qids,golds,docs)
        bqq=perq_for_indices(br,idx,qids,golds,docs)

        fs=blend_rank(sr[:,:DEPTH],ds[idx],cs,alpha)
        rr=full_rank(sr,fs)
        mm=metric_for_indices(rr,idx,qids,golds,docs)
        qq=perq_for_indices(rr,idx,qids,golds,docs)
        delta=float(mm["recall_at_5"]-bmet["recall_at_5"])
        cert_deltas.append(delta)
        report["cert"][fn]={
            "stage07d":bmet,
            "adapted_blend":mm,
            "delta":delta,
            "wins":int((qq>bqq).sum()),
            "losses":int((qq<bqq).sum()),
        }
        print(
            f"[CERT/{fn}] BASE={bmet['recall_at_5']:.9f} "
            f"ADAPT={mm['recall_at_5']:.9f} DELTA={delta:+.9f} "
            f"W/L={report['cert'][fn]['wins']}/{report['cert'][fn]['losses']}",
            flush=True
        )
        del cl,cv,cr,ctexts,cs
        clear_cuda()

    print("[7/9] CERT decision",flush=True)
    cert_positive=sum(x>0 for x in cert_deltas)
    cert_mean=float(np.mean(cert_deltas))
    if cert_positive==2 and cert_mean>=.0015:
        decision="PROMOTE_TO_FINAL_TRAIN"
    elif cert_positive>=1 and cert_mean>0:
        decision="KEEP_AS_RISKY_COMPLEMENT"
    else:
        decision="KILL_AFTER_CERT"

    report["status"]="CERT_COMPLETE"
    report["cert_summary"]={
        "frozen_alpha":alpha,
        "mean_delta":cert_mean,
        "positive_folds":cert_positive,
        "decision":decision,
    }

    print("[8/9] Save",flush=True)
    (OUT/"REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("[9/9] Result")
    print("="*120)
    print(
        f"DEV fold2: BASE={bm['recall_at_5']:.9f} "
        f"ADAPT={win['recall_at_5']:.9f} DELTA={dev_delta:+.9f} "
        f"alpha={alpha:.2f}"
    )
    for fn in CERT_FOLDS:
        c=report["cert"][fn]
        print(
            f"{fn}: BASE={c['stage07d']['recall_at_5']:.9f} "
            f"ADAPT={c['adapted_blend']['recall_at_5']:.9f} "
            f"DELTA={c['delta']:+.9f} W/L={c['wins']}/{c['losses']}"
        )
    print(
        f"CERT mean delta={cert_mean:+.9f} positive={cert_positive}/2"
    )
    print("DECISION:",decision)
    print("REPORT:",OUT/"REPORT.json")
    print("="*120)

if __name__=="__main__":
    main()

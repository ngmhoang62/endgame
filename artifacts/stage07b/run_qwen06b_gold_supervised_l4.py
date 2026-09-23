#!/usr/bin/env python
"""
Stage07N-L4/T4 — Gold-supervised Qwen3-Reranker-0.6B, T4-safe full fine-tuning.

NO DISTILLATION:
- no teacher model
- no teacher logits/targets/snapshot
- supervision = competition gold labels only
- CE-LR is used only as fold-clean prior + hard-negative miner

Why this T4 build differs from the A100 prototype
--------------------------------------------------
The A100 prototype loaded trainable weights directly in BF16. That is extremely
memory-efficient, but optimizer updates are then applied to low-precision
parameters. This T4 build keeps FP32 master/trainable parameters and uses FP16
autocast + GradScaler for compute. It is more numerically conventional and is
expected to use substantially more VRAM (~10-14 GiB rather than ~7 GiB).

Deadline-oriented defaults:
- MAXLEN 1024 (instead of 1536)
- 4 hard negatives/query (instead of up to 6)
- train folds 0,1,2 -> dev fold3 -> cert fold4
- gradient checkpointing
- AdamW foreach=False to avoid optimizer peak-memory spikes
- completed TRAIN checkpoint is saved to Drive immediately and auto-reused

Required payload:
  /content/stage07b/payload/
    shortlist_top30.i4.npy
    X45.f32.npy
    labels_top30.u1.npy
    gold_count.i2.npy
    fold_id.i1.npy
    evidence_top20.pkl
    qids.json

Run:
  python /content/stage07b/run_qwen06b_gold_supervised_l4.py

If OOM:
  python /content/stage07b/run_qwen06b_gold_supervised_l4.py --maxlen 768 --max-negs 3 --hard-negs 3
"""
from __future__ import annotations

import argparse, gc, json, math, pickle, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE=Path("/content/stage07b")
PAYLOAD=BASE/"payload"
WORK=Path("/content/stage07n_t4_work")
OUT=Path("/content/drive/MyDrive/DSC2026/stage07n_t4_gold_qwen")
MODEL_ID="Qwen/Qwen3-Reranker-0.6B"

TRAIN_FOLDS=(0,1,2)
DEV_FOLD=3
CERT_FOLD=4
DEPTH=20
FULL=30
SEED=276

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def load_payload():
    short=np.load(PAYLOAD/"shortlist_top30.i4.npy")
    X45=np.load(PAYLOAD/"X45.f32.npy")
    y=np.load(PAYLOAD/"labels_top30.u1.npy")
    gold_count=np.load(PAYLOAD/"gold_count.i2.npy")
    fold_id=np.load(PAYLOAD/"fold_id.i1.npy")
    with (PAYLOAD/"evidence_top20.pkl").open("rb") as f:
        evidence=pickle.load(f)
    qids=json.loads((PAYLOAD/"qids.json").read_text(encoding="utf-8"))
    n=len(qids)
    if short.shape!=(n,FULL) or X45.shape!=(n,FULL,45) or y.shape!=(n,FULL):
        raise RuntimeError(
            f"payload drift short={short.shape} X45={X45.shape} y={y.shape}"
        )
    if len(evidence)!=n or any(len(x)!=DEPTH for x in evidence):
        raise RuntimeError("evidence payload drift")
    return qids,short,X45,y,gold_count,fold_id,evidence

def qwen_features(tok,pairs,maxlen):
    pre=tok.encode(PREFIX,add_special_tokens=False)
    suf=tok.encode(SUFFIX,add_special_tokens=False)
    avail=maxlen-len(pre)-len(suf)
    if avail<256:
        raise RuntimeError(f"maxlen too small after prompt overhead: {avail}")
    raw=tok(
        pairs,padding=False,truncation=True,max_length=avail,
        add_special_tokens=False,return_attention_mask=False
    )
    return [{"input_ids":pre+x+suf} for x in raw["input_ids"]]

def yes_no_ids(tok):
    yes=tok.encode("yes",add_special_tokens=False)
    no=tok.encode("no",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1:
        raise RuntimeError(f"Expected one-token yes/no; yes={yes} no={no}")
    return yes[0],no[0]

def last_score(model,batch,yes_id,no_id):
    decoder=getattr(model,"model",None)
    lm=getattr(model,"lm_head",None)
    if decoder is None or lm is None:
        raise RuntimeError("Qwen CausalLM structure drift")
    out=decoder(**batch,return_dict=True)
    last=out.last_hidden_state[:,-1,:]
    logits=lm(last)
    return logits[:,yes_id]-logits[:,no_id]

def zrow(x):
    x=np.asarray(x,np.float32)
    sd=float(x.std())
    if sd<1e-5: sd=1.0
    return (x-float(x.mean()))/sd

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True)
    sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-5,1.,sd)
    return (x-mu)/sd

def fit_lr_score(X45,y,train_idx,score_idx):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m=Pipeline([
        ("scale",StandardScaler()),
        ("lr",LogisticRegression(
            C=.15,class_weight="balanced",solver="lbfgs",
            max_iter=300,tol=1e-5
        ))
    ])
    m.fit(X45[train_idx].reshape(-1,45),y[train_idx].reshape(-1))
    return np.asarray(
        m.decision_function(X45[score_idx].reshape(-1,45)),
        np.float32
    ).reshape(len(score_idx),FULL)

def train_prior_crossfit(X45,y,fold_id):
    train_idx=np.where(np.isin(fold_id,TRAIN_FOLDS))[0].astype(np.int32)
    prior=np.full((len(fold_id),FULL),np.nan,np.float32)
    for f in TRAIN_FOLDS:
        va=np.where(fold_id==f)[0].astype(np.int32)
        fit=np.where(
            np.isin(fold_id,[x for x in TRAIN_FOLDS if x!=f])
        )[0].astype(np.int32)
        prior[va]=fit_lr_score(X45,y,fit,va)
    if not np.isfinite(prior[train_idx]).all():
        raise RuntimeError("crossfit train prior incomplete")
    return train_idx,prior[train_idx]

def held_prior(X45,y,train_idx,idx):
    return fit_lr_score(X45,y,train_idx,idx)

def make_fresh_model():
    """
    FP32 trainable/master weights + FP16 autocast.
    This intentionally uses more memory than the earlier BF16-parameter build.
    """
    tok=AutoTokenizer.from_pretrained(
        MODEL_ID,padding_side="left",use_fast=True
    )
    kwargs=dict(torch_dtype=torch.float32,low_cpu_mem_usage=True)
    try:
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,attn_implementation="sdpa",**kwargs
        )
    except Exception:
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,**kwargs)
    model.config.use_cache=False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant":False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad_(True)
    return tok,model.to("cuda")

def load_saved_model():
    ck=OUT/"train012_model"
    tok=AutoTokenizer.from_pretrained(ck,padding_side="left",use_fast=True)
    kwargs=dict(torch_dtype=torch.float32,low_cpu_mem_usage=True)
    try:
        model=AutoModelForCausalLM.from_pretrained(
            ck,attn_implementation="sdpa",**kwargs
        )
    except Exception:
        model=AutoModelForCausalLM.from_pretrained(ck,**kwargs)
    model.config.use_cache=False
    return tok,model.to("cuda")

def precision_policy():
    # L4/A100 support BF16; T4 generally does not.
    # BF16 has FP32-like exponent range and is much less prone to gradient overflow.
    use_bf16=bool(torch.cuda.is_bf16_supported())
    return (torch.bfloat16 if use_bf16 else torch.float16), use_bf16

def make_scaler(use_bf16):
    if use_bf16:
        return None
    # Conservative scale for T4 FP16. Default 65536 can overflow immediately.
    try:
        return torch.amp.GradScaler(
            "cuda", init_scale=1024.0, growth_interval=200
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            init_scale=1024.0, growth_interval=200
        )

def score_pairs(model,tok,yes_id,no_id,pairs,maxlen,amp_dtype):
    feats=qwen_features(tok,pairs,maxlen)
    batch=tok.pad(
        feats,padding=True,pad_to_multiple_of=8,return_tensors="pt"
    )
    batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
    with torch.autocast("cuda",dtype=amp_dtype):
        s=last_score(model,batch,yes_id,no_id)
    return s.float()

def choose_group(base_row,yrow,args):
    yy=np.asarray(yrow[:DEPTH],np.uint8)
    pos=list(map(int,np.where(yy>0)[0]))
    neg=list(map(int,np.where(yy==0)[0]))
    if not pos or not neg:
        return None
    bz=zrow(base_row)[:DEPTH]
    hard=sorted(neg,key=lambda j:(-float(bz[j]),j))[:args.hard_negs]
    chosen=set(pos)|set(hard)
    # Optional extra negatives, still prior-only and teacher-free.
    for j in neg:
        if len([x for x in chosen if x not in set(pos)])>=args.max_negs:
            break
        chosen.add(j)
    neg_chosen=[j for j in chosen if j not in set(pos)]
    if len(neg_chosen)>args.max_negs:
        neg_chosen=sorted(
            neg_chosen,key=lambda j:(-float(bz[j]),j)
        )[:args.max_negs]
    return sorted(set(pos)|set(neg_chosen))

def query_weight(base_row,yrow,args):
    order=np.argsort(-np.asarray(base_row),kind="stable")[:5]
    total=float(np.asarray(yrow).sum())
    if total<=0:return 1.0
    rec=float(np.asarray(yrow)[order].sum())/total
    multi=1.0 if total>1 else 0.0
    return 1.0+args.hard_query_bonus*(1.0-rec)+args.multi_bonus*multi

def train_model(qids,X45,y,fold_id,evidence,args):
    OUT.mkdir(parents=True,exist_ok=True)

    complete_meta=OUT/"TRAIN_META.json"
    complete_model=OUT/"train012_model"
    if complete_meta.is_file() and complete_model.is_dir():
        meta=json.loads(complete_meta.read_text(encoding="utf-8"))
        expected={
            "maxlen":args.maxlen,
            "hard_negs":args.hard_negs,
            "max_negs":args.max_negs,
            "epochs":args.epochs,
            "lr":args.lr,
        }
        got={k:meta["config"].get(k) for k in expected}
        if got==expected and meta.get("status")=="PASS":
            print("[train] completed Drive checkpoint found -> REUSE",flush=True)
            tok,model=load_saved_model()
            train_idx=np.where(np.isin(fold_id,TRAIN_FOLDS))[0].astype(np.int32)
            return tok,model,float(meta["alpha"]),train_idx

    seed_all(args.seed)
    train_idx,prior_tr=train_prior_crossfit(X45,y,fold_id)
    tok,model=make_fresh_model()
    model.train()
    yes_id,no_id=yes_no_ids(tok)
    raw_alpha=torch.nn.Parameter(
        torch.tensor(-1.5,dtype=torch.float32,device="cuda")
    )

    # foreach=False avoids large temporary tensor lists during optimizer.step().
    opt=torch.optim.AdamW(
        [
            {"params":model.parameters(),"lr":args.lr},
            {"params":[raw_alpha],"lr":args.alpha_lr},
        ],
        weight_decay=args.weight_decay,
        foreach=False,
    )
    amp_dtype,use_bf16=precision_policy()
    scaler=make_scaler(use_bf16)
    print(
        f"[precision] autocast={str(amp_dtype).replace('torch.','')} "
        f"grad_scaler={'OFF' if scaler is None else 'ON'}",
        flush=True
    )

    valid=[]
    groups={}
    for li,qi in enumerate(train_idx):
        g=choose_group(prior_tr[li],y[int(qi)],args)
        if g is not None:
            valid.append(li);groups[li]=g
    valid=np.asarray(valid,np.int32)
    mean_group=float(np.mean([len(groups[int(x)]) for x in valid]))
    print(
        f"[train] queries={len(train_idx)} valid={len(valid)} "
        f"mean_group={mean_group:.2f} maxlen={args.maxlen}",
        flush=True
    )

    total_updates=math.ceil(len(valid)*args.epochs/args.grad_accum)
    sch=get_cosine_schedule_with_warmup(
        opt,max(1,int(.08*total_updates)),total_updates
    )
    rng=np.random.default_rng(args.seed)
    opt.zero_grad(set_to_none=True)
    global_micro=0
    t0=time.perf_counter()
    loss_tail=[]

    for ep in range(args.epochs):
        order=valid.copy();rng.shuffle(order)
        ep_losses=[]
        for step,li0 in enumerate(order):
            li=int(li0);qi=int(train_idx[li]);pp=groups[li]
            pairs=[evidence[qi][p] for p in pp]
            try:
                s=score_pairs(
                    model,tok,yes_id,no_id,pairs,args.maxlen,amp_dtype
                )
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                raise RuntimeError(
                    f"T4 OOM with group={len(pp)}, maxlen={args.maxlen}. "
                    f"Try --maxlen 768 --max-negs 3 --hard-negs 3"
                )

            p=np.asarray(pp,np.int32)
            base=torch.from_numpy(
                zrow(prior_tr[li])[p].astype(np.float32)
            ).to("cuda")
            yy=torch.from_numpy(y[qi,p].astype(np.float32)).to("cuda")

            alpha=F.softplus(raw_alpha)
            total=base+alpha*s

            cnt=yy.sum().clamp_min(1)
            target=yy/cnt
            listwise=-(target*torch.log_softmax(
                total/args.gold_temp,dim=0
            )).sum()

            pos=torch.where(yy>0)[0]
            neg=torch.where(yy==0)[0]
            if len(pos) and len(neg):
                diff=total[pos][:,None]-total[neg][None,:]
                pair=F.softplus(args.margin-diff).mean()
            else:
                pair=torch.zeros((),device="cuda")

            bce_each=F.binary_cross_entropy_with_logits(
                s,yy,reduction="none"
            )
            pc=yy.sum().clamp_min(1)
            nc=(1-yy).sum().clamp_min(1)
            weights=torch.where(
                yy>0,
                torch.full_like(yy,.5)/pc,
                torch.full_like(yy,.5)/nc,
            )
            bce=(bce_each*weights).sum()

            reg=(s**2).mean()
            qw=query_weight(prior_tr[li],y[qi],args)
            loss=qw*(
                args.listwise_weight*listwise
                +args.pair_weight*pair
                +args.bce_weight*bce
                +args.resid_l2*reg
            )/args.grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss q={qids[qi]}")

            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            ep_losses.append(float(loss.detach().cpu())*args.grad_accum)
            global_micro+=1

            boundary=(
                global_micro%args.grad_accum==0
                or step+1==len(order)
            )
            if boundary:
                params=list(model.parameters())+[raw_alpha]
                if scaler is None:
                    total_norm=torch.nn.utils.clip_grad_norm_(
                        params,1.0,error_if_nonfinite=False
                    )
                    if not bool(torch.isfinite(total_norm)):
                        raise RuntimeError(
                            "Non-finite BF16 gradient norm. Stop and report this."
                        )
                    opt.step()
                    sch.step()
                else:
                    scaler.unscale_(opt)
                    total_norm=torch.nn.utils.clip_grad_norm_(
                        params,1.0,error_if_nonfinite=False
                    )
                    if bool(torch.isfinite(total_norm)):
                        scaler.step(opt)
                        scaler.update()
                        sch.step()
                    else:
                        old_scale=float(scaler.get_scale())
                        scaler.update()
                        print(
                            f"[fp16-overflow] skipped optimizer step; "
                            f"scale {old_scale:.0f}->{float(scaler.get_scale()):.0f}",
                            flush=True
                        )
                opt.zero_grad(set_to_none=True)

            if (step+1)%64==0 or step+1==len(order):
                elapsed=time.perf_counter()-t0
                done=step+1
                eta=elapsed/max(done,1)*(len(order)-done)
                alloc=torch.cuda.memory_allocated()/2**30
                reserved=torch.cuda.max_memory_reserved()/2**30
                print(
                    f"[train] ep={ep+1}/{args.epochs} {done}/{len(order)} "
                    f"loss128={np.mean(ep_losses[-128:]):.5f} "
                    f"alpha={float(F.softplus(raw_alpha).detach().cpu()):.4f} "
                    f"eta_min={eta/60:.1f} "
                    f"alloc={alloc:.1f}GiB peak_reserved={reserved:.1f}GiB",
                    flush=True
                )

        loss_tail.extend(ep_losses[-128:])

    alpha=float(F.softplus(raw_alpha).detach().cpu())

    # Drop optimizer/grads BEFORE serializing/evaluation.
    opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.grad=None
    del opt,sch,raw_alpha
    if scaler is not None:
        del scaler
    gc.collect();torch.cuda.empty_cache()

    # Save completed train checkpoint immediately to persistent Drive.
    if complete_model.exists():
        import shutil
        shutil.rmtree(complete_model)
    model.save_pretrained(complete_model,safe_serialization=True)
    tok.save_pretrained(complete_model)

    meta={
        "schema":"stage07n.t4.gold_qwen.train.v1",
        "status":"PASS",
        "model_id":MODEL_ID,
        "teacher_used":False,
        "distillation_used":False,
        "training_folds":list(TRAIN_FOLDS),
        "train_queries":int(len(train_idx)),
        "valid_queries":int(len(valid)),
        "mean_group":mean_group,
        "alpha":alpha,
        "loss_tail":float(np.mean(loss_tail[-128:])),
        "peak_reserved_gib":float(
            torch.cuda.max_memory_reserved()/2**30
        ),
        "seconds":float(time.perf_counter()-t0),
        "precision":("FP32 params/optimizer + BF16 autocast" if use_bf16 else "FP32 params/optimizer + FP16 autocast + GradScaler"),
        "config":vars(args),
    }
    complete_meta.write_text(
        json.dumps(meta,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print(
        f"[train] SAVED persistent checkpoint -> {complete_model}",
        flush=True
    )
    model.eval()
    return tok,model,alpha,train_idx

def infer_scores(model,tok,evidence,idx,args):
    yes_id,no_id=yes_no_ids(tok)
    amp_dtype,_=precision_policy()
    out=np.empty((len(idx),DEPTH),np.float32)
    with torch.inference_mode():
        for li,qi0 in enumerate(idx):
            qi=int(qi0)
            pairs=evidence[qi][:DEPTH]
            try:
                s=score_pairs(
                    model,tok,yes_id,no_id,pairs,args.maxlen,amp_dtype
                )
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                raise RuntimeError(
                    f"inference OOM at 20 candidates x maxlen={args.maxlen}; "
                    f"rerun with --maxlen 768"
                )
            out[li]=s.float().cpu().numpy()
            if (li+1)%50==0 or li+1==len(idx):
                print(f"[infer] {li+1}/{len(idx)}",flush=True)
    return out

def evaluate(score,y,gold_count,idx):
    vals=[];hits=[]
    for li,qi0 in enumerate(idx):
        qi=int(qi0)
        order=np.argsort(-score[li],kind="stable")[:5]
        h=int(y[qi,order].sum())
        hits.append(h);vals.append(h/int(gold_count[qi]))
    vals=np.asarray(vals,float);hits=np.asarray(hits,float)
    gcnt=np.asarray([gold_count[int(q)] for q in idx])
    single=gcnt==1;multi=gcnt>1
    return {
        "queries":int(len(idx)),
        "recall_at_5":float(vals.mean()),
        "precision_at_5":float((hits/5).mean()),
        "single_gold_recall_at_5":float(vals[single].mean()),
        "multi_gold_recall_at_5":float(vals[multi].mean()),
    },vals

def eval_fold(name,fold,model,tok,alpha,train_idx,X45,y,gold_count,
              fold_id,evidence,args):
    idx=np.where(fold_id==fold)[0].astype(np.int32)
    prior=held_prior(X45,y,train_idx,idx)
    s=infer_scores(model,tok,evidence,idx,args)
    final=zrows(prior)
    final[:,:DEPTH]+=alpha*s

    bm,bq=evaluate(prior,y,gold_count,idx)
    sm,sq=evaluate(final,y,gold_count,idx)
    d=float(sm["recall_at_5"]-bm["recall_at_5"])
    w=int((sq>bq).sum());l=int((sq<bq).sum())

    print("="*112)
    print(
        f"[{name}] BASE={bm['recall_at_5']:.9f} "
        f"GOLD-QWEN={sm['recall_at_5']:.9f} "
        f"DELTA={d:+.9f} W/L={w}/{l}"
    )
    print(
        f"[{name}] single {bm['single_gold_recall_at_5']:.9f} -> "
        f"{sm['single_gold_recall_at_5']:.9f} | "
        f"multi {bm['multi_gold_recall_at_5']:.9f} -> "
        f"{sm['multi_gold_recall_at_5']:.9f}"
    )
    print("="*112)
    return {
        "baseline":bm,"gold_qwen":sm,
        "delta_recall":d,"wins":w,"losses":l
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--grad-accum",type=int,default=4)
    ap.add_argument("--maxlen",type=int,default=1024)
    ap.add_argument("--lr",type=float,default=8e-6)
    ap.add_argument("--alpha-lr",type=float,default=8e-4)
    ap.add_argument("--weight-decay",type=float,default=.01)
    ap.add_argument("--hard-negs",type=int,default=4)
    ap.add_argument("--max-negs",type=int,default=4)
    ap.add_argument("--listwise-weight",type=float,default=.55)
    ap.add_argument("--pair-weight",type=float,default=.25)
    ap.add_argument("--bce-weight",type=float,default=.20)
    ap.add_argument("--resid-l2",type=float,default=5e-4)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--margin",type=float,default=.4)
    ap.add_argument("--hard-query-bonus",type=float,default=1.25)
    ap.add_argument("--multi-bonus",type=float,default=.15)
    ap.add_argument("--dev-min-delta",type=float,default=.002)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(
        f"[device] {torch.cuda.get_device_name(0)} "
        f"VRAM={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB",
        flush=True
    )
    amp_dtype,use_bf16=precision_policy()
    print(
        f"[precision] FP32 trainable parameters + "
        f"{'BF16 autocast (no scaler)' if use_bf16 else 'FP16 autocast + GradScaler(init=1024)'}",
        flush=True
    )
    print(
        "[eligibility] teacher_used=False teacher_targets_used=False "
        "distillation_used=False supervision=gold_labels_only",
        flush=True
    )

    print("[1/6] Load portable payload",flush=True)
    qids,short,X45,y,gold_count,fold_id,evidence=load_payload()

    print("[2/6] Train/reuse gold-supervised Qwen0.6B",flush=True)
    tok,model,alpha,train_idx=train_model(
        qids,X45,y,fold_id,evidence,args
    )

    print("[3/6] DEV fold3",flush=True)
    dev=eval_fold(
        "DEV/fold3",DEV_FOLD,model,tok,alpha,train_idx,
        X45,y,gold_count,fold_id,evidence,args
    )
    dev_pass=(
        dev["delta_recall"]>=args.dev_min_delta
        and dev["wins"]>=dev["losses"]
    )

    report={
        "schema":"dsc2026.endgame.stage07n.t4_gold_qwen.v1",
        "status":"DEV_COMPLETE",
        "eligibility":{
            "base_model":MODEL_ID,
            "teacher_used":False,
            "teacher_targets_used":False,
            "distillation_used":False,
            "training_type":"supervised full fine-tuning on competition gold labels",
        },
        "precision":("FP32 params + BF16 autocast" if use_bf16 else "FP32 params + FP16 autocast + GradScaler"),
        "partition":{
            "train_folds":list(TRAIN_FOLDS),
            "dev_fold":DEV_FOLD,
            "cert_fold":CERT_FOLD,
        },
        "alpha":alpha,
        "dev":dev,
        "cert":None,
        "config":vars(args),
    }
    OUT.mkdir(parents=True,exist_ok=True)

    if not dev_pass:
        report["status"]="KILL_ON_DEV_CERT_UNTOUCHED"
        (OUT/"REPORT.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8"
        )
        print("DECISION: KILL_ON_DEV_CERT_UNTOUCHED")
        print("REPORT:",OUT/"REPORT.json")
        return

    print("[4/6] DEV passed -> CERT fold4",flush=True)
    cert=eval_fold(
        "CERT/fold4",CERT_FOLD,model,tok,alpha,train_idx,
        X45,y,gold_count,fold_id,evidence,args
    )
    report["cert"]=cert

    if cert["delta_recall"]>=.002 and cert["wins"]>=cert["losses"]:
        decision="PROMOTE_TO_FULLTRAIN"
    elif cert["delta_recall"]>0:
        decision="KEEP_AS_RISKY_COMPLEMENT"
    else:
        decision="KILL_AFTER_CERT"

    report["status"]="CERT_COMPLETE"
    report["decision"]=decision

    print("[5/6] Save report",flush=True)
    (OUT/"REPORT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )

    print("[6/6] Result")
    print("="*116)
    print(
        f"DEV delta={dev['delta_recall']:+.9f} "
        f"W/L={dev['wins']}/{dev['losses']}"
    )
    print(
        f"CERT delta={cert['delta_recall']:+.9f} "
        f"W/L={cert['wins']}/{cert['losses']}"
    )
    print(f"alpha={alpha:.4f}")
    print("DECISION:",decision)
    print("REPORT:",OUT/"REPORT.json")
    print("="*116)

if __name__=="__main__":
    main()

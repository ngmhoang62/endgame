#!/usr/bin/env python
"""Stage07B2-v3 — A100 full-finetune distilled Qwen3-Reranker-0.6B.

Why v3
------
Use the A100 where it matters:
- FULL fine-tune the 0.6B student (not LoRA);
- train only on gold + hard negatives / teacher-disagreement candidates;
- keep inference on top20 and CE-LR prior on full top30;
- strict 5-fold OOF with inner-cross-fitted CE-LR priors.

Teacher Qwen3-Reranker-4B is training-time distillation only.
It is never an active inference component and never evaluated as a pipeline arm.
"""
from __future__ import annotations
import argparse,gc,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics
import src.stage07_breakthrough.stage07b_distill_common as C

MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
TEACH=C.CACHE/"teacher_qwen4b_top20.f32.npy"
TEACH_DONE=C.CACHE/"teacher_qwen4b_done.u1.npy"
OOF=C.CACHE/"student_qwen06b_distilled_fullft_v3_oof_scores20.f32.npy"

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def rank(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-score[i]))]
    return out

def metrics(rankarr,qids,golds,docs,folds):
    pred={q:[docs[int(x)] for x in rankarr[i,:5]] for i,q in enumerate(qids)}
    return official_metrics(pred,golds,qids),{f:official_metrics(pred,golds,ids) for f,ids in folds.items()}

def zrows(x):
    x=np.asarray(x,np.float32)
    mu=x.mean(1,keepdims=True);sd=x.std(1,keepdims=True)
    sd=np.where(sd<1e-5,1.,sd)
    return (x-mu)/sd

def make_model():
    tok=AutoTokenizer.from_pretrained(MODEL_ID,padding_side="left",use_fast=True)
    kwargs=dict(torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    try:
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,attn_implementation="sdpa",**kwargs
        )
    except Exception:
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,**kwargs)
    # FULL FINETUNE: do not freeze backbone, do not use PEFT.
    for p in model.parameters():p.requires_grad_(True)
    model=model.to("cuda")
    return tok,model

def score_pairs(model,tok,yes_id,no_id,pairs):
    feats=C.tokenize_qwen(tok,pairs)
    batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
    batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        s=C.last_yes_no_score(model,batch,yes_id,no_id)
    return s

def evidence_pair(qi,pos,qids,questions,short,names,vv,rr,views,texts):
    q=questions[qids[int(qi)]]
    d=short[int(qi),int(pos)]
    return C.format_pair(q,C.bundle(int(qi),int(pos),d,names,vv,rr,views,texts))

def choose_train_positions(qi,base_row,teacher_row,yrow,args):
    """Gold + complementary hard negatives. Return sorted unique candidate positions in top20."""
    depth=C.DEPTH
    yy=yrow[:depth]
    pos=list(map(int,np.where(yy>0)[0]))
    neg=np.where(yy==0)[0]
    if not pos or len(neg)==0:return None

    bz=zrows(base_row[None,:])[0,:depth]
    tz=zrows(teacher_row[None,:])[0]

    chosen=set(pos)

    # Strongest CE-LR negatives: current decision boundary.
    ob=neg[np.argsort(-bz[neg],kind="stable")]
    chosen.update(map(int,ob[:args.base_negs]))

    # Strongest teacher negatives: semantic hard negatives.
    ot=neg[np.argsort(-tz[neg],kind="stable")]
    chosen.update(map(int,ot[:args.teacher_negs]))

    # Teacher/base disagreement negatives: information missing from current selector.
    disagree=tz-bz
    od=neg[np.argsort(-disagree[neg],kind="stable")]
    chosen.update(map(int,od[:args.disagree_negs]))

    # Keep all positives, cap only negatives.
    positives=set(pos)
    neg_chosen=[x for x in chosen if x not in positives]
    if len(positives)+len(neg_chosen)>args.max_group:
        # prioritize by max(base hardness, teacher hardness, disagreement)
        score={x:max(float(bz[x]),float(tz[x]),float(disagree[x])) for x in neg_chosen}
        neg_chosen=sorted(neg_chosen,key=lambda x:-score[x])[:max(0,args.max_group-len(positives))]
    return sorted(positives|set(neg_chosen))

def baseline_top5_miss_weight(base_row,yrow,args):
    top=np.argsort(-base_row,kind="stable")[:5]
    total=float(yrow.sum())
    if total<=0:return 1.0
    hit=float(yrow[top].sum())
    recall=hit/total
    return 1.0 + args.hard_query_bonus*(1.0-recall)

def train_outer(fn,train_idx,held_idx,prior_tr,prior_he,teacher,y,
                qids,questions,short,names,vv,rr,views,texts,args):
    seed_all(args.seed)
    tok,model=make_model();model.train()
    yes_id=tok.convert_tokens_to_ids("yes");no_id=tok.convert_tokens_to_ids("no")
    if yes_id is None or no_id is None:raise RuntimeError("yes/no token lookup failed")
    raw_alpha=torch.nn.Parameter(torch.tensor(-1.5,device="cuda"))
    opt=torch.optim.AdamW([
        {"params":model.parameters(),"lr":args.lr},
        {"params":[raw_alpha],"lr":args.alpha_lr},
    ],weight_decay=.01)
    rng=np.random.default_rng(args.seed)

    valid=[]
    groups={}
    for li,qi in enumerate(train_idx):
        p=choose_train_positions(int(qi),prior_tr[li],teacher[int(qi)],y[int(qi)],args)
        if p is not None:
            valid.append(li);groups[li]=p
    valid=np.asarray(valid,np.int32)
    print(f"    {fn}: train groups={len(valid)}/{len(train_idx)} "
          f"mean_group={np.mean([len(groups[int(x)]) for x in valid]):.2f}",flush=True)

    t0=time.perf_counter();losses=[]
    for ep in range(args.epochs):
        order=valid.copy();rng.shuffle(order);ep_loss=[]
        for st in range(0,len(order),args.batch_queries):
            locs=order[st:st+args.batch_queries]
            flat_pairs=[];slices=[];cursor=0
            for li0 in locs:
                li=int(li0);qi=int(train_idx[li]);pp=groups[li]
                for p in pp:
                    flat_pairs.append(evidence_pair(qi,p,qids,questions,short,names,vv,rr,views,texts))
                slices.append((li,qi,pp,cursor,cursor+len(pp)))
                cursor+=len(pp)

            try:
                all_s=score_pairs(model,tok,yes_id,no_id,flat_pairs)
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM at batch_queries={args.batch_queries}, pairs={len(flat_pairs)}. "
                    f"Rerun with --batch-queries {max(1,args.batch_queries//2)}"
                )

            alpha=F.softplus(raw_alpha)
            qloss=[]
            for li,qi,pp,a,b in slices:
                s=all_s[a:b]
                pp_np=np.asarray(pp,np.int32)
                base=torch.from_numpy(zrows(prior_tr[li:li+1])[0,pp_np]).to("cuda")
                teach=torch.from_numpy(teacher[qi,pp_np]).to("cuda")
                yy=torch.from_numpy(y[qi,pp_np]).to("cuda")

                total=base+alpha*s
                cnt=yy.sum().clamp_min(1)
                target=yy/cnt
                gold=-(target*torch.log_softmax(total/args.gold_temp,dim=0)).sum()

                tp=torch.softmax((teach-teach.mean())/args.kd_temp,dim=0)
                kd=F.kl_div(torch.log_softmax(s/args.kd_temp,dim=0),tp,
                            reduction="sum")*(args.kd_temp**2)

                pos=torch.where(yy>0)[0];neg=torch.where(yy==0)[0]
                if len(pos) and len(neg):
                    diff=total[pos][:,None]-total[neg][None,:]
                    pair=F.softplus(args.margin-diff).mean()
                else:
                    pair=torch.zeros((),device="cuda")

                reg=(s**2).mean()
                w=baseline_top5_miss_weight(prior_tr[li],y[qi],args)
                qloss.append(w*(args.gold_weight*gold+args.kd_weight*kd+
                                args.pair_weight*pair+args.resid_l2*reg))

            loss=torch.stack(qloss).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            ep_loss.append(float(loss.detach().cpu()))

        losses.append(float(np.mean(ep_loss)))
        print(f"    {fn} ep={ep+1}/{args.epochs} loss={losses[-1]:.5f} "
              f"alpha={float(F.softplus(raw_alpha).detach().cpu()):.4f} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m "
              f"vram={torch.cuda.max_memory_reserved()/2**30:.1f}GiB",flush=True)

    # Held-out inference: score ALL top20, retain full top30 CE-LR prior.
    model.eval();held_s=np.empty((len(held_idx),C.DEPTH),np.float32)
    with torch.inference_mode():
        for st in range(0,len(held_idx),args.eval_batch_queries):
            qis=held_idx[st:st+args.eval_batch_queries]
            pairs=[]
            for qi in qis:
                for p in range(C.DEPTH):
                    pairs.append(evidence_pair(int(qi),p,qids,questions,short,names,vv,rr,views,texts))
            s=score_pairs(model,tok,yes_id,no_id,pairs).reshape(len(qis),C.DEPTH)
            held_s[st:st+len(qis)]=s.float().cpu().numpy()

    alpha=float(F.softplus(raw_alpha).detach().cpu())
    del model,tok,opt,raw_alpha
    gc.collect();torch.cuda.empty_cache()
    return held_s,alpha,{"loss_tail":losses[-1],
                          "mean_group":float(np.mean([len(groups[int(x)]) for x in valid]))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-queries",type=int,default=4)
    ap.add_argument("--eval-batch-queries",type=int,default=8)
    ap.add_argument("--lr",type=float,default=8e-6)
    ap.add_argument("--alpha-lr",type=float,default=8e-4)
    ap.add_argument("--gold-weight",type=float,default=.62)
    ap.add_argument("--kd-weight",type=float,default=.28)
    ap.add_argument("--pair-weight",type=float,default=.10)
    ap.add_argument("--resid-l2",type=float,default=5e-4)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--kd-temp",type=float,default=1.5)
    ap.add_argument("--margin",type=float,default=.4)
    ap.add_argument("--hard-query-bonus",type=float,default=1.5)
    ap.add_argument("--base-negs",type=int,default=4)
    ap.add_argument("--teacher-negs",type=int,default=3)
    ap.add_argument("--disagree-negs",type=int,default=2)
    ap.add_argument("--max-group",type=int,default=10)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()
    C.CACHE.mkdir(parents=True,exist_ok=True);C.OUT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    total=torch.cuda.get_device_properties(0).total_memory/2**30
    if total<35:raise RuntimeError(f"Stage07B2-v3 requires large-VRAM CUDA; got {total:.1f}GB")

    print("[1/6] load world + teacher distillation targets",flush=True)
    qids,questions,golds,folds,docs,short,X45,vv,rr,sim,views,texts,names=C.build_evidence_world()
    teacher=np.load(TEACH);td=np.load(TEACH_DONE)
    if teacher.shape!=(len(qids),C.DEPTH) or int(td.sum())!=len(qids) or not np.isfinite(teacher).all():
        raise RuntimeError("teacher targets incomplete")
    y=C.labels(short,qids,golds,docs)

    print("[2/6] strict 5-fold FULL-FT distilled student OOF",flush=True)
    oof20=np.full((len(qids),C.DEPTH),np.nan,np.float32)
    final30=np.full((len(qids),C.FULL_DEPTH),np.nan,np.float32)
    base30=np.full_like(final30,np.nan);meta={}
    for fn in folds:
        print(f"  [{fn}] leakage-safe inner-crossfit CE-LR prior",flush=True)
        tr,he,btr,bhe=C.outer_priors(X45,y,folds,qids,fn)
        base30[he]=bhe
        hs,alpha,m=train_outer(fn,tr,he,btr,bhe,teacher,y,qids,questions,short,names,vv,rr,views,texts,args)
        oof20[he]=hs
        total=zrows(bhe)
        total[:,:C.DEPTH]+=alpha*hs
        final30[he]=total
        meta[fn]={"train":len(tr),"held":len(he),"alpha":alpha,**m}
        print(f"  [{fn}] alpha={alpha:.4f} complete",flush=True)

    if not np.isfinite(oof20).all() or not np.isfinite(final30).all():
        raise RuntimeError("OOF incomplete")
    np.save(OOF,oof20)

    print("[3/6] baseline parity",flush=True)
    br=rank(short,base30);bm,bpf=metrics(br,qids,golds,docs,folds)
    if abs(bm["recall_at_5"]-0.9425976255185238)>3e-6:
        raise RuntimeError(f"CE-LR parity failed {bm['recall_at_5']}")

    print("[4/6] student evaluation",flush=True)
    sr=rank(short,final30);sm,spf=metrics(sr,qids,golds,docs,folds)
    delta=float(sm["recall_at_5"]-bm["recall_at_5"])
    def pq(r):
        out=[]
        for i,q in enumerate(qids):
            out.append(len({docs[int(x)] for x in r[i,:5]}&set(golds[q]))/len(set(golds[q])))
        return np.asarray(out)
    bq=pq(br);sq=pq(sr);wins=int(np.sum(sq>bq));losses=int(np.sum(sq<bq))

    print("[5/6] active parameter ledger",flush=True)
    active_existing=1_731_560_449
    active_student=600_000_000
    active_total=active_existing+active_student
    if active_total>=4_000_000_000:raise RuntimeError("active parameter budget exceeded")

    if sm["recall_at_5"]>=.960:decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.010:decision="MAJOR_GAIN"
    elif delta>=.005:decision="STRONG_GAIN"
    elif delta>=.002:decision="KEEP_COMPLEMENT"
    else:decision="KILL_DISTILLED_STUDENT"

    print("[6/6] report",flush=True)
    rep={"schema":"dsc2026.endgame.stage07b.qwen4b_to_qwen06b_fullft_harddistill.v3",
         "status":"COMPLETE","claim_boundary":"strict 5-fold OOF",
         "teacher":{"model_id":"Qwen/Qwen3-Reranker-4B","role":"TRAINING_TIME_ONLY",
                    "evaluated_as_pipeline_arm":False},
         "student":{"model_id":MODEL_ID,"training":"FULL_FINETUNE",
                    "train_candidate_policy":"gold + CE-hard + teacher-hard + disagreement",
                    "inference_depth":C.DEPTH},
         "config":vars(args),
         "baseline":{"overall":bm,"per_fold":bpf},
         "student_distilled":{"overall":sm,"per_fold":spf},
         "effect":{"delta_recall":delta,
                   "delta_single":float(sm["single_gold_recall_at_5"]-bm["single_gold_recall_at_5"]),
                   "delta_multi":float(sm["multi_gold_recall_at_5"]-bm["multi_gold_recall_at_5"]),
                   "wins":wins,"losses":losses},
         "folds":meta,
         "active_parameter_ledger":{"existing":active_existing,"student_conservative":active_student,
                                    "active_total":active_total,"limit":4_000_000_000,
                                    "teacher_excluded_reason":"training-time distillation only; absent at inference"},
         "decision":decision}
    (C.OUT/"STUDENT_OOF_V3.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n")
    print("="*120)
    print(f"BASE CE-LR       R={bm['recall_at_5']:.9f} P={bm['precision_at_5']:.9f} single={bm['single_gold_recall_at_5']:.9f} multi={bm['multi_gold_recall_at_5']:.9f}")
    print(f"DISTILLED 0.6B   R={sm['recall_at_5']:.9f} P={sm['precision_at_5']:.9f} single={sm['single_gold_recall_at_5']:.9f} multi={sm['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: base={bpf[f]['recall_at_5']:.9f} student={spf[f]['recall_at_5']:.9f} delta={spf[f]['recall_at_5']-bpf[f]['recall_at_5']:+.9f}")
    print("ACTIVE PARAM TOTAL <=",active_total)
    print("DECISION:",decision)
    print("="*120)

if __name__=="__main__":main()

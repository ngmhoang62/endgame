#!/usr/bin/env python
"""Stage07B2 — Strict 5-fold OOF Qwen3-Reranker-0.6B residual distillation.

Teacher: Qwen3-Reranker-4B training-time soft targets only.
Student: Qwen3-Reranker-0.6B LoRA.
Final active pipeline uses ONLY the student, never the teacher.

Score:
  total_30 = z(CE-LR prior_30)
  total_30[:20] += softplus(alpha) * z(student_score_20)

Loss:
  supervised multi-positive listwise loss on total_30
  + teacher->student listwise KL on top20
  + boundary pairwise loss
  + residual L2

Strict OOF:
  outer held fold never contributes labels to student training;
  train-side CE-LR priors are INNER-cross-fitted to avoid stacking leakage.
"""
from __future__ import annotations
import argparse,gc,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics
import src.stage07_breakthrough.stage07b_distill_common as C

MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
TEACH=C.CACHE/"teacher_qwen4b_top20.f32.npy"
TEACH_DONE=C.CACHE/"teacher_qwen4b_done.u1.npy"
OOF=C.CACHE/"student_qwen06b_distilled_oof_scores20.f32.npy"

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def rank(short,fullscore):
    out=np.empty_like(short)
    for i in range(len(short)):out[i]=short[i,np.lexsort((short[i],-fullscore[i]))]
    return out

def metrics(rankarr,qids,golds,docs,folds):
    pred={q:[docs[int(x)] for x in rankarr[i,:5]] for i,q in enumerate(qids)}
    return official_metrics(pred,golds,qids),{f:official_metrics(pred,golds,ids) for f,ids in folds.items()}

def make_model():
    tok=AutoTokenizer.from_pretrained(MODEL_ID,padding_side="left",use_fast=True)
    try:
        base=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )
    except Exception:
        base=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
        )
    cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",
                   target_modules=["q_proj","k_proj","v_proj","o_proj"],
                   task_type="CAUSAL_LM")
    model=get_peft_model(base,cfg).to("cuda")
    return tok,model

def score_group(model,tok,yes_id,no_id,pairs):
    feats=C.tokenize_qwen(tok,pairs)
    batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
    batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return C.last_yes_no_score(model,batch,yes_id,no_id)

def query_pairs(qindices,qids,questions,short,names,vv,rr,views,texts):
    pairs=[]
    for qi in qindices:
        q=questions[qids[int(qi)]]
        for pos,d in enumerate(short[int(qi),:C.DEPTH]):
            pairs.append(C.format_pair(q,C.bundle(int(qi),pos,d,names,vv,rr,views,texts)))
    return pairs

def train_outer(fn,train_idx,held_idx,prior_tr,prior_he,teacher,y,
                qids,questions,short,names,vv,rr,views,texts,args):
    seed_all(args.seed)
    tok,model=make_model();model.train()
    yes_id=tok.convert_tokens_to_ids("yes");no_id=tok.convert_tokens_to_ids("no")
    raw_alpha=torch.nn.Parameter(torch.tensor(-1.5,device="cuda"))
    opt=torch.optim.AdamW([
        {"params":[p for p in model.parameters() if p.requires_grad],"lr":args.lr},
        {"params":[raw_alpha],"lr":args.alpha_lr},
    ],weight_decay=.01)
    rng=np.random.default_rng(args.seed)
    loc=np.arange(len(train_idx),dtype=np.int32)
    losses=[];t0=time.perf_counter()

    for ep in range(args.epochs):
        rng.shuffle(loc);ep_loss=[]
        for st in range(0,len(loc),args.batch_queries):
            lp=loc[st:st+args.batch_queries]
            qis=train_idx[lp]
            pairs=query_pairs(qis,qids,questions,short,names,vv,rr,views,texts)
            try:
                s=score_group(model,tok,yes_id,no_id,pairs).reshape(len(qis),C.DEPTH)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    f"OOM with batch_queries={args.batch_queries}; rerun with --batch-queries "
                    f"{max(1,args.batch_queries//2)}"
                )
            sz=C.zrows_torch(s)
            bz=torch.from_numpy(C.zrows_np(prior_tr[lp])).to("cuda")
            teach=torch.from_numpy(C.zrows_np(teacher[qis])).to("cuda")
            yy=torch.from_numpy(y[qis]).to("cuda")
            alpha=F.softplus(raw_alpha)

            total=bz.clone()
            total[:,:C.DEPTH]=total[:,:C.DEPTH]+alpha*sz

            cnt=yy.sum(1,keepdim=True).clamp_min(1)
            target=yy/cnt
            gold_loss=-(target*torch.log_softmax(total/args.gold_temp,dim=1)).sum(1).mean()

            tp=torch.softmax(teach/args.kd_temp,dim=1)
            kd=F.kl_div(torch.log_softmax(sz/args.kd_temp,dim=1),tp,
                        reduction="batchmean")*(args.kd_temp**2)

            pair_losses=[]
            for bi in range(len(qis)):
                pos=torch.where(yy[bi]>0)[0]
                neg=torch.where(yy[bi]==0)[0]
                if len(pos)==0 or len(neg)==0:continue
                hard=neg[torch.topk(bz[bi,neg],k=min(8,len(neg))).indices]
                diff=total[bi,pos][:,None]-total[bi,hard][None,:]
                pair_losses.append(F.softplus(args.margin-diff).mean())
            pair=torch.stack(pair_losses).mean() if pair_losses else torch.zeros((),device="cuda")
            reg=(sz**2).mean()

            loss=args.gold_weight*gold_loss+args.kd_weight*kd+args.pair_weight*pair+args.resid_l2*reg
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            ep_loss.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(ep_loss)))
        print(f"    {fn} ep={ep+1}/{args.epochs} loss={losses[-1]:.5f} "
              f"alpha={float(F.softplus(raw_alpha).detach().cpu()):.4f} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m",flush=True)

    # Held-out student score, query-batched.
    model.eval();held_s=np.empty((len(held_idx),C.DEPTH),np.float32)
    with torch.inference_mode():
        for st in range(0,len(held_idx),args.eval_batch_queries):
            qis=held_idx[st:st+args.eval_batch_queries]
            pairs=query_pairs(qis,qids,questions,short,names,vv,rr,views,texts)
            s=score_group(model,tok,yes_id,no_id,pairs).reshape(len(qis),C.DEPTH)
            held_s[st:st+len(qis)]=s.float().cpu().numpy()
    alpha=float(F.softplus(raw_alpha).detach().cpu())
    del model,tok,opt,raw_alpha
    gc.collect();torch.cuda.empty_cache()
    return held_s,alpha,{"loss_tail":losses[-1]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-queries",type=int,default=4)
    ap.add_argument("--eval-batch-queries",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--alpha-lr",type=float,default=1e-3)
    ap.add_argument("--gold-weight",type=float,default=.65)
    ap.add_argument("--kd-weight",type=float,default=.30)
    ap.add_argument("--pair-weight",type=float,default=.05)
    ap.add_argument("--resid-l2",type=float,default=.001)
    ap.add_argument("--gold-temp",type=float,default=1.0)
    ap.add_argument("--kd-temp",type=float,default=1.5)
    ap.add_argument("--margin",type=float,default=.5)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()
    C.CACHE.mkdir(parents=True,exist_ok=True);C.OUT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    total=torch.cuda.get_device_properties(0).total_memory/2**30
    if total<35:raise RuntimeError(f"Stage07B2 intended for large-VRAM CUDA; got {total:.1f}GB")

    print("[1/6] load world + teacher distillation targets",flush=True)
    qids,questions,golds,folds,docs,short,X45,vv,rr,sim,views,texts,names=C.build_evidence_world()
    teacher=np.load(TEACH)
    td=np.load(TEACH_DONE)
    if teacher.shape!=(len(qids),C.DEPTH) or int(td.sum())!=len(qids) or not np.isfinite(teacher).all():
        raise RuntimeError("teacher targets incomplete")
    y=C.labels(short,qids,golds,docs)

    print("[2/6] strict 5-fold student OOF",flush=True)
    oof20=np.full((len(qids),C.DEPTH),np.nan,np.float32)
    final30=np.full((len(qids),C.FULL_DEPTH),np.nan,np.float32)
    base30=np.full_like(final30,np.nan)
    meta={}
    for fn in folds:
        print(f"  [{fn}] leakage-safe inner-crossfit priors",flush=True)
        tr,he,btr,bhe=C.outer_priors(X45,y,folds,qids,fn)
        base30[he]=bhe
        hs,alpha,m=train_outer(fn,tr,he,btr,bhe,teacher,y,qids,questions,short,names,vv,rr,views,texts,args)
        oof20[he]=hs
        total=C.zrows_np(bhe)
        total[:,:C.DEPTH]+=alpha*C.zrows_np(hs)
        final30[he]=total
        meta[fn]={"train":len(tr),"held":len(he),"alpha":alpha,**m}
        print(f"  [{fn}] alpha={alpha:.4f} done",flush=True)

    if not np.isfinite(oof20).all() or not np.isfinite(final30).all():
        raise RuntimeError("OOF incomplete")
    np.save(OOF,oof20)

    print("[3/6] baseline parity",flush=True)
    br=rank(short,base30);bm,bpf=metrics(br,qids,golds,docs,folds)
    if abs(bm["recall_at_5"]-0.9425976255185238)>3e-6:
        raise RuntimeError(f"CE-LR parity failed: {bm['recall_at_5']}")

    print("[4/6] distilled student OOF evaluation",flush=True)
    sr=rank(short,final30);sm,spf=metrics(sr,qids,golds,docs,folds)
    delta=float(sm["recall_at_5"]-bm["recall_at_5"])

    def pq(r):
        vals=[]
        for i,q in enumerate(qids):
            vals.append(len({docs[int(x)] for x in r[i,:5]}&set(golds[q]))/len(set(golds[q])))
        return np.asarray(vals)
    bq=pq(br);sq=pq(sr);wins=int(np.sum(sq>bq));losses=int(np.sum(sq<bq))

    print("[5/6] parameter ledger",flush=True)
    active_existing=1731560449
    # Official family size is 0.6B; keep a conservative 0.6B ledger entry.
    active_student=600_000_000
    active_total=active_existing+active_student
    if active_total>=4_000_000_000:raise RuntimeError("active parameter budget exceeded")

    if sm["recall_at_5"]>=.960:decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.010:decision="MAJOR_GAIN"
    elif delta>=.005:decision="STRONG_GAIN"
    elif delta>=.002:decision="KEEP_COMPLEMENT"
    else:decision="KILL_DISTILLED_STUDENT"

    print("[6/6] report",flush=True)
    rep={"schema":"dsc2026.endgame.stage07b.qwen4b_to_qwen06b_distill_oof.v1",
         "status":"COMPLETE","claim_boundary":"strict 5-fold OOF",
         "teacher":{"model_id":"Qwen/Qwen3-Reranker-4B","role":"TRAINING_TIME_ONLY",
                    "evaluated_as_pipeline_arm":False},
         "student":{"model_id":MODEL_ID,"role":"ACTIVE_PIPELINE_COMPONENT","depth_scored":C.DEPTH},
         "loss":{"gold_weight":args.gold_weight,"kd_weight":args.kd_weight,
                 "pair_weight":args.pair_weight,"resid_l2":args.resid_l2},
         "baseline":{"overall":bm,"per_fold":bpf},
         "student_distilled":{"overall":sm,"per_fold":spf},
         "effect":{"delta_recall":delta,
                   "delta_single":float(sm["single_gold_recall_at_5"]-bm["single_gold_recall_at_5"]),
                   "delta_multi":float(sm["multi_gold_recall_at_5"]-bm["multi_gold_recall_at_5"]),
                   "wins":wins,"losses":losses},
         "folds":meta,
         "active_parameter_ledger":{"existing":active_existing,"student_conservative":active_student,
                                    "active_total":active_total,"limit":4_000_000_000,
                                    "teacher_excluded_reason":"training-time distillation only; not used at inference"},
         "decision":decision}
    (C.OUT/"STUDENT_OOF.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n")
    print("="*118)
    print(f"BASE CE-LR       R={bm['recall_at_5']:.9f} P={bm['precision_at_5']:.9f} single={bm['single_gold_recall_at_5']:.9f} multi={bm['multi_gold_recall_at_5']:.9f}")
    print(f"DISTILLED 0.6B   R={sm['recall_at_5']:.9f} P={sm['precision_at_5']:.9f} single={sm['single_gold_recall_at_5']:.9f} multi={sm['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in folds:
        print(f"{f}: base={bpf[f]['recall_at_5']:.9f} student={spf[f]['recall_at_5']:.9f} delta={spf[f]['recall_at_5']-bpf[f]['recall_at_5']:+.9f}")
    print("ACTIVE PARAM TOTAL <=",active_total)
    print("DECISION:",decision)
    print("="*118)

if __name__=="__main__":main()

#!/usr/bin/env python
from __future__ import annotations
import argparse,gc,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import stage07b_portable_common as C

MODEL_ID="Qwen/Qwen3-Reranker-0.6B"

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def load_teacher(drive_root,n):
    local=C.WORK/"teacher_scores.f32.npy"
    done=C.WORK/"teacher_done.u1.npy"
    if local.exists() and done.exists():
        s=np.asarray(np.load(local,mmap_mode="r"),np.float32);d=np.asarray(np.load(done,mmap_mode="r"))
        if s.shape==(n,C.DEPTH) and int(d.sum())==n:return s
    snap=drive_root/"teacher_targets_snapshot.npz"
    if not snap.is_file():raise FileNotFoundError("Complete teacher snapshot not found on local workdir or Drive")
    z=np.load(snap);s=np.asarray(z["scores"],np.float32);d=np.asarray(z["done"],np.uint8)
    if s.shape!=(n,C.DEPTH) or int(d.sum())!=n:raise RuntimeError("teacher snapshot incomplete")
    return s

def make_model():
    tok=AutoTokenizer.from_pretrained(MODEL_ID,padding_side="left",use_fast=True)
    try:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True).to("cuda")
    except Exception:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to("cuda")
    for p in model.parameters():p.requires_grad_(True)
    return tok,model

def score_pairs(model,tok,yes_id,no_id,pairs):
    feats=C.qwen_features(tok,pairs)
    batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
    batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return C.last_score(model,batch,yes_id,no_id)

def choose(base,teacher,yrow,args):
    yy=yrow[:C.DEPTH];pos=list(map(int,np.where(yy>0)[0]));neg=np.where(yy==0)[0]
    if not pos or len(neg)==0:return None
    bz=C.zrows_np(base[None,:])[0,:C.DEPTH];tz=C.zrows_np(teacher[None,:])[0]
    chosen=set(pos)
    chosen.update(map(int,neg[np.argsort(-bz[neg])[:args.base_negs]]))
    chosen.update(map(int,neg[np.argsort(-tz[neg])[:args.teacher_negs]]))
    dis=tz-bz
    chosen.update(map(int,neg[np.argsort(-dis[neg])[:args.disagree_negs]]))
    positives=set(pos);negc=[x for x in chosen if x not in positives]
    if len(positives)+len(negc)>args.max_group:
        score={x:max(float(bz[x]),float(tz[x]),float(dis[x])) for x in negc}
        negc=sorted(negc,key=lambda x:-score[x])[:max(0,args.max_group-len(positives))]
    return sorted(positives|set(negc))

def hard_weight(base,yrow,bonus):
    top=np.argsort(-base)[:5];total=float(yrow.sum())
    if total<=0:return 1.
    return 1.+bonus*(1.-float(yrow[top].sum())/total)

def save_fold(model,tok,drive_root,fold,held_scores,alpha,meta):
    d=drive_root/"checkpoints"/f"fold_{fold}"
    d.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(d,safe_serialization=True)
    tok.save_pretrained(d)
    np.save(d/"held_scores20.f32.npy",held_scores)
    (d/"fold_meta.json").write_text(json.dumps({"alpha":alpha,**meta},indent=2)+"\n")
    print(f"[drive] fold_{fold} checkpoint saved -> {d}",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",type=Path,default=C.DEFAULT_DRIVE)
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

    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if torch.cuda.get_device_properties(0).total_memory/2**30<35:raise RuntimeError("Large-VRAM CUDA required")
    args.drive_root.mkdir(parents=True,exist_ok=True);C.WORK.mkdir(parents=True,exist_ok=True)
    qids,short,X45,y,gold_count,fold_id,evidence=C.load_payload();n=len(qids)
    teacher=load_teacher(args.drive_root,n)

    final=np.full((n,C.FULL),np.nan,np.float32)
    base_oof=np.full_like(final,np.nan);student_oof=np.full((n,C.DEPTH),np.nan,np.float32)
    fold_meta={}

    for outer in range(5):
        ck=args.drive_root/"checkpoints"/f"fold_{outer}"
        hs_path=ck/"held_scores20.f32.npy";meta_path=ck/"fold_meta.json"
        tr,he,btr,bhe=C.outer_priors(X45,y,fold_id,outer);base_oof[he]=bhe

        if hs_path.is_file() and meta_path.is_file():
            hs=np.load(hs_path);fm=json.loads(meta_path.read_text())
            alpha=float(fm["alpha"])
            if hs.shape!=(len(he),C.DEPTH):raise RuntimeError(f"fold {outer} checkpoint score shape drift")
            print(f"[resume] fold_{outer} from Drive",flush=True)
        else:
            seed_all(args.seed+outer)
            tok,model=make_model();model.train();yes_id,no_id=C.yes_no_ids(tok)
            raw_alpha=torch.nn.Parameter(torch.tensor(-1.5,device="cuda"))
            opt=torch.optim.AdamW([
                {"params":model.parameters(),"lr":args.lr},
                {"params":[raw_alpha],"lr":args.alpha_lr},
            ],weight_decay=.01)

            groups={}
            locs=[]
            for li,qi in enumerate(tr):
                g=choose(btr[li],teacher[qi],y[qi],args)
                if g is not None:groups[li]=g;locs.append(li)
            locs=np.asarray(locs,np.int32)
            rng=np.random.default_rng(args.seed+outer);t0=time.perf_counter();loss_tail=None

            for ep in range(args.epochs):
                order=locs.copy();rng.shuffle(order);ep_losses=[]
                for st in range(0,len(order),args.batch_queries):
                    chunk=order[st:st+args.batch_queries]
                    pairs=[];slices=[];cur=0
                    for li0 in chunk:
                        li=int(li0);qi=int(tr[li]);pp=groups[li]
                        pairs.extend(evidence[qi][p] for p in pp)
                        slices.append((li,qi,pp,cur,cur+len(pp)));cur+=len(pp)
                    try:sall=score_pairs(model,tok,yes_id,no_id,pairs)
                    except torch.cuda.OutOfMemoryError:
                        raise RuntimeError(f"OOM: rerun with --batch-queries {max(1,args.batch_queries//2)}")
                    alpha=F.softplus(raw_alpha);ql=[]
                    for li,qi,pp,a,b in slices:
                        s=sall[a:b];ppn=np.asarray(pp,np.int32)
                        base=torch.from_numpy(C.zrows_np(btr[li:li+1])[0,ppn]).to("cuda")
                        teach=torch.from_numpy(C.zrows_np(teacher[qi:qi+1])[0,ppn]).to("cuda")
                        yy=torch.from_numpy(y[qi,ppn].astype(np.float32)).to("cuda")
                        total=base+alpha*s
                        target=yy/yy.sum().clamp_min(1)
                        gold=-(target*torch.log_softmax(total/args.gold_temp,dim=0)).sum()
                        tp=torch.softmax(teach/args.kd_temp,dim=0)
                        kd=F.kl_div(torch.log_softmax(s/args.kd_temp,dim=0),tp,reduction="sum")*(args.kd_temp**2)
                        pos=torch.where(yy>0)[0];neg=torch.where(yy==0)[0]
                        pair=F.softplus(args.margin-(total[pos][:,None]-total[neg][None,:])).mean() if len(pos) and len(neg) else torch.zeros((),device="cuda")
                        w=hard_weight(btr[li],y[qi],args.hard_query_bonus)
                        ql.append(w*(args.gold_weight*gold+args.kd_weight*kd+args.pair_weight*pair+args.resid_l2*(s**2).mean()))
                    loss=torch.stack(ql).mean()
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                    ep_losses.append(float(loss.detach().cpu()))
                loss_tail=float(np.mean(ep_losses))
                print(f"[fold{outer}] ep={ep+1}/{args.epochs} loss={loss_tail:.5f} alpha={float(F.softplus(raw_alpha).detach().cpu()):.4f} elapsed={(time.perf_counter()-t0)/60:.1f}m vram={torch.cuda.max_memory_reserved()/2**30:.1f}GiB",flush=True)

            model.eval();hs=np.empty((len(he),C.DEPTH),np.float32)
            with torch.inference_mode():
                for st in range(0,len(he),args.eval_batch_queries):
                    qis=he[st:st+args.eval_batch_queries]
                    pairs=[evidence[qi][p] for qi in qis for p in range(C.DEPTH)]
                    s=score_pairs(model,tok,yes_id,no_id,pairs).reshape(len(qis),C.DEPTH)
                    hs[st:st+len(qis)]=s.float().cpu().numpy()
            alpha=float(F.softplus(raw_alpha).detach().cpu())
            save_fold(model,tok,args.drive_root,outer,hs,alpha,
                      {"loss_tail":loss_tail,"train_queries":len(tr),"held_queries":len(he)})
            del model,tok,opt,raw_alpha;gc.collect();torch.cuda.empty_cache()

        student_oof[he]=hs
        fs=C.zrows_np(bhe);fs[:,:C.DEPTH]+=alpha*hs;final[he]=fs
        fold_meta[str(outer)]={"alpha":alpha,"train":len(tr),"held":len(he)}

    br=C.ranking(short,base_oof);sr=C.ranking(short,final)
    bm=C.metric(br,y,gold_count);sm=C.metric(sr,y,gold_count)
    bpf={str(f):C.metric(br,y,gold_count,fold_id,f) for f in range(5)}
    spf={str(f):C.metric(sr,y,gold_count,fold_id,f) for f in range(5)}
    if abs(bm["recall_at_5"]-0.9425976255185238)>3e-6:
        raise RuntimeError(f"baseline parity failed {bm['recall_at_5']}")
    delta=sm["recall_at_5"]-bm["recall_at_5"]
    # Paired query recall W/L.
    def perq(r):
        return np.asarray([y[i,r[i,:5]].sum()/gold_count[i] for i in range(n)],float)
    bq=perq(br);sq=perq(sr);wins=int((sq>bq).sum());losses=int((sq<bq).sum())

    if sm["recall_at_5"]>=.960:decision="BREAKTHROUGH_TARGET_REACHED"
    elif delta>=.010:decision="MAJOR_GAIN"
    elif delta>=.005:decision="STRONG_GAIN"
    elif delta>=.002:decision="KEEP_COMPLEMENT"
    else:decision="KILL_DISTILLED_STUDENT"

    report={
        "schema":"dsc2026.endgame.stage07b.portable_fullft_distill.v4",
        "teacher":{"model":"Qwen/Qwen3-Reranker-4B","role":"TRAINING_TIME_ONLY","active_inference":False},
        "student":{"model":MODEL_ID,"training":"FULL_FINETUNE","active_inference":True},
        "baseline":{"overall":bm,"per_fold":bpf},
        "student_distilled":{"overall":sm,"per_fold":spf},
        "effect":{"delta_recall":delta,
                  "delta_single":sm["single_gold_recall_at_5"]-bm["single_gold_recall_at_5"],
                  "delta_multi":sm["multi_gold_recall_at_5"]-bm["multi_gold_recall_at_5"],
                  "wins":wins,"losses":losses},
        "folds":fold_meta,
        "active_parameter_ledger":{"existing":1731560449,"student_conservative":600000000,
                                   "active_total":2331560449,"limit":4000000000},
        "decision":decision,
    }
    args.drive_root.mkdir(parents=True,exist_ok=True)
    (args.drive_root/"STUDENT_OOF_V4.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    np.save(args.drive_root/"student_oof_scores20.f32.npy",student_oof)

    print("="*120)
    print(f"BASE CE-LR       R={bm['recall_at_5']:.9f} P={bm['precision_at_5']:.9f} single={bm['single_gold_recall_at_5']:.9f} multi={bm['multi_gold_recall_at_5']:.9f}")
    print(f"DISTILLED 0.6B   R={sm['recall_at_5']:.9f} P={sm['precision_at_5']:.9f} single={sm['single_gold_recall_at_5']:.9f} multi={sm['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} W/L={wins}/{losses}")
    for f in range(5):
        print(f"fold_{f}: base={bpf[str(f)]['recall_at_5']:.9f} student={spf[str(f)]['recall_at_5']:.9f} delta={spf[str(f)]['recall_at_5']-bpf[str(f)]['recall_at_5']:+.9f}")
    print("DECISION:",decision)
    print("REPORT SAVED TO DRIVE:",args.drive_root/"STUDENT_OOF_V4.json")
    print("="*120)

if __name__=="__main__":main()

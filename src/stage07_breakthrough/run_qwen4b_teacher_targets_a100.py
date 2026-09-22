#!/usr/bin/env python
"""Stage07B1 — Qwen3-Reranker-4B teacher target generation on A100.

IMPORTANT:
- The 4B teacher is TRAINING-TIME ONLY.
- It is NEVER evaluated as a submission/diagnostic arm.
- It emits NO recall/precision metrics.
- Its scores are consumed only as soft targets by the 0.6B student.
"""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.stage07_breakthrough.stage07b_distill_common as C

MODEL_ID="Qwen/Qwen3-Reranker-4B"
SCORES=C.CACHE/"teacher_qwen4b_top20.f32.npy"
DONE=C.CACHE/"teacher_qwen4b_done.u1.npy"
META=C.CACHE/"teacher_qwen4b_meta.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--query-block",type=int,default=32)
    ap.add_argument("--max-batch",type=int,default=96)
    ap.add_argument("--token-budget",type=int,default=98304)
    args=ap.parse_args()
    C.CACHE.mkdir(parents=True,exist_ok=True);C.OUT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    total=torch.cuda.get_device_properties(0).total_memory/2**30
    if total<35:raise RuntimeError(f"This teacher job is reserved for >=35GB GPU; got {total:.1f}GB")

    print("[1/5] load OOF retrieval/evidence world (NO teacher metrics)",flush=True)
    qids,questions,golds,folds,docs,short,X45,vv,rr,sim,views,texts,names=C.build_evidence_world()
    n=len(qids)

    ex=[SCORES.exists(),DONE.exists(),META.exists()]
    if any(ex) and not all(ex):raise RuntimeError("partial teacher cache")
    if all(ex):
        score=np.lib.format.open_memmap(SCORES,mode="r+")
        done=np.lib.format.open_memmap(DONE,mode="r+")
        meta=json.loads(META.read_text())
    else:
        score=np.lib.format.open_memmap(SCORES,mode="w+",dtype=np.float32,shape=(n,C.DEPTH));score[:]=np.nan;score.flush()
        done=np.lib.format.open_memmap(DONE,mode="w+",dtype=np.uint8,shape=(n,));done[:]=0;done.flush()
        meta={"schema":"stage07b.teacher_targets.v1","model_id":MODEL_ID,"depth":C.DEPTH,
              "max_length":C.MAXLEN,"status":"IN_PROGRESS","completed":0,
              "claim":"TRAINING-TIME DISTILLATION TARGETS ONLY; NO TEACHER EVALUATION"}
        META.write_text(json.dumps(meta,indent=2)+"\n")

    pending=[i for i in range(n) if int(done[i])==0]
    if not pending:
        print("teacher target cache already complete");return

    print("[2/5] load Qwen3-Reranker-4B BF16",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,padding_side="left",use_fast=True)
    try:
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa",
            low_cpu_mem_usage=True
        ).eval().to("cuda")
    except Exception:
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
        ).eval().to("cuda")
    yes_id=tok.convert_tokens_to_ids("yes");no_id=tok.convert_tokens_to_ids("no")
    if yes_id is None or no_id is None:raise RuntimeError("yes/no token lookup failed")

    print("[3/5] batched teacher target generation",flush=True)
    t0=time.perf_counter();newq=0
    for st in range(0,len(pending),args.query_block):
        qis=pending[st:st+args.query_block]
        pairs,owners=C.build_pairs_for_queries(qis,qids,questions,short,names,vv,rr,views,texts)
        feats=C.tokenize_qwen(tok,pairs)
        lens=np.asarray([len(x["input_ids"]) for x in feats],np.int32)
        order=np.argsort(lens,kind="stable")
        vals=np.empty(len(feats),np.float32)
        i=0;oomcap=args.max_batch
        while i<len(order):
            b=max(1,min(oomcap,args.max_batch,args.token_budget//max(1,int(lens[order[i]]))))
            j=min(i+b,len(order))
            b2=max(1,min(oomcap,args.max_batch,args.token_budget//max(1,int(lens[order[j-1]]))))
            if b2<j-i:j=i+b2
            ids=order[i:j]
            batch=tok.pad([feats[int(k)] for k in ids],padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            try:
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    s=C.last_yes_no_score(model,batch,yes_id,no_id).float().cpu().numpy()
                for k,v in zip(ids,s):vals[int(k)]=float(v)
                i=j
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                if oomcap<=1:raise
                oomcap=max(1,oomcap//2)
                print(f"  OOM -> batch cap {oomcap}",flush=True)

        block={qi:np.empty(C.DEPTH,np.float32) for qi in qis}
        for (qi,pos),v in zip(owners,vals):block[qi][pos]=v
        for qi in qis:
            if not np.isfinite(block[qi]).all():raise RuntimeError(f"nonfinite teacher qi={qi}")
            score[qi]=block[qi];done[qi]=1;newq+=1
        score.flush();done.flush()
        completed=int(np.asarray(done).sum());elapsed=time.perf_counter()-t0
        meta["completed"]=completed;meta["status"]="PASS" if completed==n else "IN_PROGRESS"
        META.write_text(json.dumps(meta,indent=2)+"\n")
        print(f"  teacher {completed}/{n} q_rate={newq/max(elapsed,1e-9):.3f}/s "
              f"pair_rate={newq*C.DEPTH/max(elapsed,1e-9):.1f}/s "
              f"vram={torch.cuda.max_memory_reserved()/2**30:.1f}GiB",flush=True)

    print("[4/5] integrity",flush=True)
    arr=np.asarray(score,np.float32)
    if not np.isfinite(arr).all() or int(np.asarray(done).sum())!=n:
        raise RuntimeError("teacher target cache incomplete")

    print("[5/5] complete — deliberately NO teacher recall/precision was computed",flush=True)
    print("TARGETS:",SCORES)
    print("These scores are valid only as distillation targets for Stage07B2.")

if __name__=="__main__":main()

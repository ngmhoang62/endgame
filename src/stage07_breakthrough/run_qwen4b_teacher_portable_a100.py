#!/usr/bin/env python
from __future__ import annotations
import argparse,gc,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import stage07b_portable_common as C

MODEL_ID="Qwen/Qwen3-Reranker-4B"

def snapshot(local_score,local_done,drive_root,status):
    drive_root.mkdir(parents=True,exist_ok=True)
    arr=np.asarray(local_score,np.float32);done=np.asarray(local_done,np.uint8)
    np.savez_compressed(drive_root/"teacher_targets_snapshot.npz",scores=arr,done=done)
    (drive_root/"teacher_status.json").write_text(json.dumps(status,indent=2)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drive-root",type=Path,default=C.DEFAULT_DRIVE)
    ap.add_argument("--query-block",type=int,default=32)
    ap.add_argument("--max-batch",type=int,default=96)
    ap.add_argument("--token-budget",type=int,default=98304)
    ap.add_argument("--snapshot-every",type=int,default=256)
    args=ap.parse_args()
    C.WORK.mkdir(parents=True,exist_ok=True)
    qids,short,X45,y,gc,fold,evidence=C.load_payload();n=len(qids)
    sp=C.WORK/"teacher_scores.f32.npy";dp=C.WORK/"teacher_done.u1.npy"

    # Restore from Drive first if present.
    snap=args.drive_root/"teacher_targets_snapshot.npz"
    if snap.is_file() and not sp.exists():
        z=np.load(snap)
        score=np.lib.format.open_memmap(sp,mode="w+",dtype=np.float32,shape=(n,C.DEPTH));score[:]=z["scores"];score.flush()
        done=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(n,));done[:]=z["done"];done.flush()
        print(f"[restore] teacher snapshot from Drive completed={int(done.sum())}/{n}",flush=True)
    elif sp.exists() and dp.exists():
        score=np.lib.format.open_memmap(sp,mode="r+");done=np.lib.format.open_memmap(dp,mode="r+")
    else:
        score=np.lib.format.open_memmap(sp,mode="w+",dtype=np.float32,shape=(n,C.DEPTH));score[:]=np.nan;score.flush()
        done=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(n,));done[:]=0;done.flush()

    pending=[i for i in range(n) if int(done[i])==0]
    if not pending:
        print("teacher targets already complete");return

    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    vram=torch.cuda.get_device_properties(0).total_memory/2**30
    if vram<35:raise RuntimeError(f"Teacher reserved for >=35GiB GPU, got {vram:.1f}")

    print(f"[load] {MODEL_ID} BF16 on {torch.cuda.get_device_name(0)} {vram:.1f}GiB",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,padding_side="left",use_fast=True)
    try:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,attn_implementation="sdpa",low_cpu_mem_usage=True).eval().to("cuda")
    except Exception:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).eval().to("cuda")
    yes_id,no_id=C.yes_no_ids(tok)

    t0=time.perf_counter();newq=0;last_snap=int(done.sum())
    for st in range(0,len(pending),args.query_block):
        qis=pending[st:st+args.query_block]
        pairs=[evidence[qi][p] for qi in qis for p in range(C.DEPTH)]
        feats=C.qwen_features(tok,pairs)
        lens=np.asarray([len(x["input_ids"]) for x in feats],np.int32)
        order=np.argsort(lens,kind="stable");vals=np.empty(len(feats),np.float32)
        i=0;cap=args.max_batch
        while i<len(order):
            b=max(1,min(cap,args.token_budget//max(1,int(lens[order[i]]))))
            j=min(i+b,len(order))
            b2=max(1,min(cap,args.token_budget//max(1,int(lens[order[j-1]]))))
            if b2<j-i:j=i+b2
            ids=order[i:j]
            batch=tok.pad([feats[int(k)] for k in ids],padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            try:
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    s=C.last_score(model,batch,yes_id,no_id).float().cpu().numpy()
                for k,v in zip(ids,s):vals[int(k)]=float(v)
                i=j
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                cap//=2
                if cap<1:raise
                print(f"[OOM] cap -> {cap}",flush=True)

        vals=vals.reshape(len(qis),C.DEPTH)
        for r,qi in enumerate(qis):score[qi]=vals[r];done[qi]=1
        score.flush();done.flush();newq+=len(qis)
        completed=int(done.sum());elapsed=time.perf_counter()-t0
        print(f"[teacher] {completed}/{n} q_rate={newq/max(elapsed,1e-9):.3f}/s "
              f"pair_rate={newq*C.DEPTH/max(elapsed,1e-9):.1f}/s "
              f"vram={torch.cuda.max_memory_reserved()/2**30:.1f}GiB",flush=True)
        if completed-last_snap>=args.snapshot_every or completed==n:
            status={"status":"PASS" if completed==n else "IN_PROGRESS","completed":completed,
                    "total":n,"model":MODEL_ID,"training_time_teacher_only":True}
            snapshot(score,done,args.drive_root,status);last_snap=completed
            print(f"[drive] teacher snapshot saved -> {args.drive_root}",flush=True)

    print("TEACHER TARGETS COMPLETE. No teacher evaluation performed.")

if __name__=="__main__":main()

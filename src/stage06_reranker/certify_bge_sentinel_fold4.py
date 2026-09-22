#!/usr/bin/env python
from __future__ import annotations
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage06_evidence.benchmark_evidence_packaging as a0
import src.stage06_reranker.train_bge_lora_dev as s1

SRC_GATE=ROOT/"reports/stage06b1_bge_lora_dev_v1_2/DEV_GATE.json"
SRC_ADAPTER=ROOT/"cache/stage06b1_bge_lora_dev_v1_2/adapter_train01"

CACHE=ROOT/"cache/stage06b2_sentinel_fold4"
OUT=ROOT/"reports/stage06b2_sentinel_fold4"

FROZEN_ALPHA=.20
DEPTH=30
MAXLEN=1024
OVERLAP=.50
SENTINEL_FOLD="fold_4"

SCORES=CACHE/"fold4_scores.f32.npy"
DONE=CACHE/"fold4_done.u1.npy"
META=CACHE/"fold4_scores.json"

def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def stable(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load_world():
    qids,questions,golds,folds,docs,shortlist,baseline,y,names=s1.world()
    q2i={q:i for i,q in enumerate(qids)}
    idx=np.asarray([q2i[q] for q in folds[SENTINEL_FOLD]],np.int32)
    return qids,questions,golds,folds,docs,shortlist,baseline,names,idx

def verify(qids,idx):
    gate=rj(SRC_GATE)
    if gate.get("cert_touched") is not False:
        raise RuntimeError("Source Stage06B1 no longer reports CERT untouched")
    if abs(float(gate["blend"]["winner_alpha_bge"])-FROZEN_ALPHA)>1e-12:
        raise RuntimeError("Frozen alpha drift")
    af=SRC_ADAPTER/"adapter_model.safetensors"
    if not af.is_file(): raise FileNotFoundError(af)
    adsha=s1.sha(af)
    if adsha!=gate["training"]["adapter_sha256"]:
        raise RuntimeError("Adapter SHA drift")
    contract={
        "schema":"stage06b2.sentinel.fold4.v1",
        "adapter_sha256":adsha,
        "source_gate_sha":s1.sha(SRC_GATE),
        "fold":SENTINEL_FOLD,
        "qids_sha":stable([qids[int(i)] for i in idx]),
        "depth":DEPTH,"maxlen":MAXLEN,"overlap":OVERLAP,
        "title_on":True,"alpha_bge":FROZEN_ALPHA,
        "no_retraining":True,"no_tuning":True,
    }
    return gate,adsha,contract,stable(contract)

def prep(idx,shortlist,docs):
    local=shortlist[idx,:DEPTH]
    vv,rr,sim,views=a0.select_witnesses("bge_lora_sentinel_fold4_top30",idx,local,docs)
    texts=a0.load_selected_texts(vv,rr,views)
    return local,vv,rr,views,texts

def make_pairs(tok,li,qi,qids,questions,local,names,vv,rr,texts,views):
    q=questions[qids[int(qi)]]
    qs=[];ps=[];owners=[]
    for pos,d in enumerate(local[li]):
        title=a0.clean_title(names[int(d)])
        ws=s1.windows(tok,q,title,li,pos,vv,rr,texts,views)
        if not ws: raise RuntimeError(f"no windows li={li} pos={pos}")
        qs.extend([q]*len(ws));ps.extend(ws);owners.extend([(li,pos)]*len(ws))
    return qs,ps,owners

def tokenize_once(tok,qs,ps):
    enc=tok(qs,ps,padding=False,truncation=True,max_length=MAXLEN,
            add_special_tokens=True,return_attention_mask=True)
    lens=np.asarray([len(x) for x in enc["input_ids"]],np.int32)
    return enc,lens

def score_encoded(tok,model,enc,lens,args,oom_cap):
    import torch
    order=np.argsort(lens,kind="stable")
    vals=np.empty(len(order),np.float32)
    i=0
    while i<len(order):
        # hard safety ceiling: never exceed user-configured batch.
        b=max(1,min(args.batch,oom_cap,args.token_budget//max(1,int(lens[order[i]]))))
        j=min(i+b,len(order))
        maxlen=int(lens[order[j-1]])
        b2=max(1,min(args.batch,oom_cap,args.token_budget//max(1,maxlen)))
        if b2<j-i:j=i+b2
        ids=order[i:j]
        feats=[{k:enc[k][int(t)] for k in enc.keys()} for t in ids]
        try:
            batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            with torch.inference_mode(),torch.autocast("cuda",dtype=s1.compute_dtype()):
                x=model(**batch,return_dict=True).logits.view(-1).float().cpu().numpy()
            for t,v in zip(ids,x):vals[int(t)]=float(v)
            i=j
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            if oom_cap<=1:raise
            oom_cap=max(1,oom_cap//2)
            print(f"[sentinel] CUDA OOM -> oom_cap={oom_cap}",flush=True)
    return vals,oom_cap

def infer(idx,qids,questions,local,names,vv,rr,texts,views,contract,ch,args):
    import torch
    CACHE.mkdir(parents=True,exist_ok=True)
    ex=[SCORES.exists(),DONE.exists(),META.exists()]
    if any(ex) and not all(ex):raise RuntimeError("partial sentinel cache")
    if all(ex):
        meta=rj(META)
        if meta["contract_hash"]!=ch:raise RuntimeError("sentinel cache contract mismatch")
        scores=np.lib.format.open_memmap(SCORES,mode="r+")
        done=np.lib.format.open_memmap(DONE,mode="r+")
    else:
        scores=np.lib.format.open_memmap(SCORES,mode="w+",dtype=np.float32,shape=(len(idx),DEPTH));scores[:]=np.nan;scores.flush()
        done=np.lib.format.open_memmap(DONE,mode="w+",dtype=np.uint8,shape=(len(idx),));done[:]=0;done.flush()
        meta={"contract_hash":ch,"contract":contract,"completed":0,"pair_count":0,
              "oom_cap":args.batch,"status":"IN_PROGRESS"}
        META.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")

    pending=[i for i in range(len(idx)) if int(done[i])==0]
    if not pending:return np.asarray(scores,np.float32)

    tok,model=s1.load_model(False);model.eval()
    oom_cap=min(int(meta.get("oom_cap",args.batch)),args.batch)
    total=int(meta.get("pair_count",0));start_total=total
    t0=time.perf_counter();newq=0

    for st in range(0,len(pending),args.query_block):
        lis=pending[st:st+args.query_block]
        qs=[];ps=[];owners=[]
        for li in lis:
            a,b,c=make_pairs(tok,li,int(idx[li]),qids,questions,local,names,vv,rr,texts,views)
            qs.extend(a);ps.extend(b);owners.extend(c)

        enc,lens=tokenize_once(tok,qs,ps)
        vals,oom_cap=score_encoded(tok,model,enc,lens,args,oom_cap)
        block={li:np.full(DEPTH,-np.inf,np.float32) for li in lis}
        for (li,pos),v in zip(owners,vals):
            if v>block[li][pos]:block[li][pos]=v
        for li in lis:
            if not np.isfinite(block[li]).all():raise RuntimeError(f"nonfinite li={li}")
            scores[li]=block[li];done[li]=1;newq+=1

        total+=len(vals);scores.flush();done.flush()
        completed=int(np.asarray(done).sum())
        meta={"contract_hash":ch,"contract":contract,"completed":completed,
              "pair_count":total,"oom_cap":oom_cap,
              "status":"PASS" if completed==len(idx) else "IN_PROGRESS"}
        META.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        elapsed=time.perf_counter()-t0
        print(f"[sentinel] {completed}/{len(idx)} pairs={total} batch_cap={args.batch} oom_cap={oom_cap} "
              f"q_rate={newq/max(elapsed,1e-9):.3f}/s pair_rate={(total-start_total)/max(elapsed,1e-9):.1f}/s "
              f"vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB",flush=True)
        del enc,lens,vals,qs,ps,owners
        gc.collect()

    del model,tok;gc.collect();torch.cuda.empty_cache()
    return np.asarray(scores,np.float32)

def rank(short,score):
    out=np.empty_like(short)
    for i in range(len(short)):out[i]=short[i,np.lexsort((short[i],-score[i]))]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--query-block",type=int,default=4)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--token-budget",type=int,default=8192)
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)

    print("[1/5] load fold4 confirmation world",flush=True)
    qids,questions,golds,folds,docs,shortlist,baseline,names,idx=load_world()
    print(f"  {SENTINEL_FOLD} queries={len(idx)} (expected ~1398)",flush=True)
    if len(idx)!=1398:raise RuntimeError(f"fold3 size drift: {len(idx)}")

    print("[2/5] verify frozen Stage06B1",flush=True)
    gate,adsha,contract,ch=verify(qids,idx)
    print(f"  adapter={adsha}",flush=True)
    print("  alpha=0.20 frozen; no retraining; no tuning",flush=True)

    print("[3/5] witnesses",flush=True)
    local,vv,rr,views,texts=prep(idx,shortlist,docs)

    print("[4/5] safe fold3 inference",flush=True)
    score=infer(idx,qids,questions,local,names,vv,rr,texts,views,contract,ch,args)

    print("[5/5] evaluate sentinel once",flush=True)
    qsel=[qids[int(i)] for i in idx]
    base=baseline[idx,:DEPTH]
    brank=rank(local,score)
    bz=a0.zrows(score);rz=a0.zrows(s1.rrscore(base,local))
    blend=rank(local,FROZEN_ALPHA*bz+(1-FROZEN_ALPHA)*rz)

    bm=s1.met(base,qsel,golds,docs)
    gm=s1.met(brank,qsel,golds,docs)
    hm=s1.met(blend,qsel,golds,docs)
    bq=s1.pq(base,qsel,golds,docs);gq=s1.pq(brank,qsel,golds,docs);hq=s1.pq(blend,qsel,golds,docs)

    delta=float(hm["recall_at_5"]-bm["recall_at_5"])
    ds=float(hm["single_gold_recall_at_5"]-bm["single_gold_recall_at_5"])
    dm=float(hm["multi_gold_recall_at_5"]-bm["multi_gold_recall_at_5"])
    if delta<=-.0015:dec="ARCHIVE_UNSTABLE_COMPLEMENT"
    elif delta<0:dec="ARCHIVE_WEAK_NEGATIVE"
    elif delta<.0015:dec="ARCHIVE_WEAK_COMPLEMENT"
    else:dec="KEEP_COMPLEMENT_CONFIRMED"

    rep={"schema":"dsc2026.endgame.stage06b2.sentinel.fold3.v1","status":"COMPLETE",
         "fold":SENTINEL_FOLD,"frozen_alpha_bge":FROZEN_ALPHA,
         "baseline":bm,"bge_lora_standalone":gm,"frozen_blend_a020":hm,
         "effect":{"delta_recall":delta,"delta_single":ds,"delta_multi":dm},
         "complementarity":{"oracle":float(np.maximum(bq,gq).mean()),
                            "bge_wins":int(np.sum(gq>bq)),"bge_losses":int(np.sum(gq<bq)),
                            "blend_wins":int(np.sum(hq>bq)),"blend_losses":int(np.sum(hq<bq))},
         "decision":dec}
    (OUT/"CONFIRMATION.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"REPORT.md").write_text(
        "# Stage06B2 Fold4 Confirmation\n\n"
        f"- baseline R: {bm['recall_at_5']:.9f}\n"
        f"- BGE standalone R: {gm['recall_at_5']:.9f}\n"
        f"- frozen blend R: {hm['recall_at_5']:.9f}\n"
        f"- delta R: {delta:+.9f}\n"
        f"- delta single: {ds:+.9f}\n"
        f"- delta multi: {dm:+.9f}\n"
        f"- decision: **{dec}**\n",
        encoding="utf-8")

    print("="*105)
    print("FOLD4 CONFIRMATION — alpha(BGE)=0.20 FROZEN")
    print(f"baseline_ce_lr          R={bm['recall_at_5']:.9f} P={bm['precision_at_5']:.9f} single={bm['single_gold_recall_at_5']:.9f} multi={bm['multi_gold_recall_at_5']:.9f}")
    print(f"bge_lora_standalone     R={gm['recall_at_5']:.9f} P={gm['precision_at_5']:.9f} single={gm['single_gold_recall_at_5']:.9f} multi={gm['multi_gold_recall_at_5']:.9f}")
    print(f"frozen_blend_a020       R={hm['recall_at_5']:.9f} P={hm['precision_at_5']:.9f} single={hm['single_gold_recall_at_5']:.9f} multi={hm['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA={delta:+.9f} single={ds:+.9f} multi={dm:+.9f}")
    print(f"BGE W/L={rep['complementarity']['bge_wins']}/{rep['complementarity']['bge_losses']} "
          f"BLEND W/L={rep['complementarity']['blend_wins']}/{rep['complementarity']['blend_losses']} "
          f"ORACLE={rep['complementarity']['oracle']:.9f}")
    print("DECISION:",dec)
    print("="*105)

if __name__=="__main__":main()

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
CACHE=ROOT/"cache/stage06b2_lite_bge_cert"
OUT=ROOT/"reports/stage06b2_lite_bge_cert"

FROZEN_ALPHA=.20
DEPTH=30
MAXLEN=1024
OVERLAP=.50
CERT_FOLDS=("fold_3","fold_4")

SCORES=CACHE/"cert_scores.f32.npy"
DONE=CACHE/"cert_done.u1.npy"
META=CACHE/"cert_scores.json"

def rj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def stable(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def verify_frozen(qids,cert):
    gate=rj(SRC_GATE)
    if gate.get("cert_touched") is not False: raise RuntimeError("Stage06B1 CERT contract drift")
    alpha=float(gate["blend"]["winner_alpha_bge"])
    if abs(alpha-FROZEN_ALPHA)>1e-12: raise RuntimeError(f"alpha drift: {alpha}")
    af=SRC_ADAPTER/"adapter_model.safetensors"
    if not af.is_file(): raise FileNotFoundError(af)
    adapter_sha=s1.sha(af)
    if adapter_sha!=gate["training"]["adapter_sha256"]: raise RuntimeError("adapter SHA drift")
    contract={
        "schema":"stage06b2_lite.cert_scores.v1",
        "source_gate_sha":s1.sha(SRC_GATE),
        "adapter_sha256":adapter_sha,
        "cert_qids_sha":stable([qids[int(i)] for i in cert]),
        "cert_folds":list(CERT_FOLDS),
        "depth":DEPTH,"max_length":MAXLEN,"overlap":OVERLAP,
        "title_on":True,"witnesses":["AIT","LAL"],
        "parent_aggregation":"max_all_windows",
        "frozen_alpha_bge":FROZEN_ALPHA,
        "no_retraining":True,"no_cert_tuning":True,
    }
    return gate,adapter_sha,contract,stable(contract)

def prep_cert(cert,shortlist,docs):
    local=shortlist[cert,:DEPTH]
    vv,rr,sim,views=a0.select_witnesses("bge_lora_cert34_top30",cert,local,docs)
    texts=a0.load_selected_texts(vv,rr,views)
    return local,vv,rr,views,texts

def build_query_pairs(tok,li,qi,qids,questions,local,names,vv,rr,texts,views):
    q=questions[qids[int(qi)]]
    qs=[]; ps=[]; owners=[]
    for pos,d in enumerate(local[li]):
        title=a0.clean_title(names[int(d)])
        ws=s1.windows(tok,q,title,li,pos,vv,rr,texts,views)
        if not ws: raise RuntimeError(f"no windows li={li} pos={pos}")
        qs.extend([q]*len(ws)); ps.extend(ws); owners.extend([(li,pos)]*len(ws))
    return qs,ps,owners

def tokenize_once(tok,qs,ps):
    enc=tok(qs,ps,padding=False,truncation=True,max_length=MAXLEN,add_special_tokens=True,return_attention_mask=True)
    lens=np.asarray([len(x) for x in enc["input_ids"]],dtype=np.int32)
    return enc,lens

def choose_batch(max_len,args,oom_cap):
    by_tokens=max(1,args.token_budget//max(1,max_len))
    cap=args.long_batch if max_len>=768 else args.max_batch
    return max(1,min(cap,by_tokens,oom_cap))

def score_encoded(tok,model,enc,lens,args,oom_cap):
    import torch
    order=np.argsort(lens,kind="stable")
    vals=np.empty(len(order),dtype=np.float32)
    i=0
    while i<len(order):
        b=choose_batch(int(lens[order[i]]),args,oom_cap)
        j=min(i+b,len(order))
        b2=choose_batch(int(lens[order[j-1]]),args,oom_cap)
        if b2<j-i: j=i+b2
        idx=order[i:j]
        feats=[{k:enc[k][int(t)] for k in enc.keys()} for t in idx]
        try:
            batch=tok.pad(feats,padding=True,pad_to_multiple_of=8,return_tensors="pt")
            batch={k:v.to("cuda",non_blocking=True) for k,v in batch.items()}
            with torch.inference_mode(),torch.autocast("cuda",dtype=s1.compute_dtype()):
                x=model(**batch,return_dict=True).logits.view(-1).float().cpu().numpy()
            for t,v in zip(idx,x): vals[int(t)]=float(v)
            i=j
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if oom_cap<=1: raise
            oom_cap=max(1,oom_cap//2)
            print(f"[CERT infer] OOM -> oom_cap={oom_cap}",flush=True)
    return vals,oom_cap

def score_cert(cert,qids,questions,local,names,vv,rr,texts,views,contract,ch,args):
    import torch
    CACHE.mkdir(parents=True,exist_ok=True)
    ex=[SCORES.exists(),DONE.exists(),META.exists()]
    if any(ex) and not all(ex): raise RuntimeError("partial CERT cache")
    if all(ex):
        meta=rj(META)
        if meta["contract_hash"]!=ch: raise RuntimeError("CERT cache contract mismatch")
        scores=np.lib.format.open_memmap(SCORES,mode="r+")
        done=np.lib.format.open_memmap(DONE,mode="r+")
    else:
        scores=np.lib.format.open_memmap(SCORES,mode="w+",dtype=np.float32,shape=(len(cert),DEPTH));scores[:]=np.nan;scores.flush()
        done=np.lib.format.open_memmap(DONE,mode="w+",dtype=np.uint8,shape=(len(cert),));done[:]=0;done.flush()
        meta={"contract_hash":ch,"contract":contract,"completed":0,"pair_count":0,"oom_cap":args.max_batch,"status":"IN_PROGRESS"}
        META.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    pending=[i for i in range(len(cert)) if int(done[i])==0]
    if not pending: return np.asarray(scores,dtype=np.float32)

    tok,model=s1.load_model(False);model.eval()
    oom_cap=int(meta.get("oom_cap",args.max_batch))
    total_pairs=int(meta.get("pair_count",0));start_pairs=total_pairs
    t0=time.perf_counter();newq=0

    for st in range(0,len(pending),args.query_block):
        lis=pending[st:st+args.query_block]
        qs=[];ps=[];owners=[]
        for li in lis:
            a,b,c=build_query_pairs(tok,li,int(cert[li]),qids,questions,local,names,vv,rr,texts,views)
            qs.extend(a);ps.extend(b);owners.extend(c)

        enc,lens=tokenize_once(tok,qs,ps)
        vals,oom_cap=score_encoded(tok,model,enc,lens,args,oom_cap)

        block={li:np.full(DEPTH,-np.inf,np.float32) for li in lis}
        for (li,pos),v in zip(owners,vals):
            if v>block[li][pos]: block[li][pos]=v
        for li in lis:
            if not np.isfinite(block[li]).all(): raise RuntimeError(f"nonfinite scores li={li}")
            scores[li]=block[li];done[li]=1;newq+=1

        total_pairs+=len(vals);scores.flush();done.flush()
        completed=int(np.asarray(done).sum())
        meta={"contract_hash":ch,"contract":contract,"completed":completed,"pair_count":total_pairs,"oom_cap":oom_cap,
              "status":"PASS" if completed==len(cert) else "IN_PROGRESS"}
        META.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        elapsed=time.perf_counter()-t0
        print(f"[CERT infer] {completed}/{len(cert)} pairs={total_pairs} oom_cap={oom_cap} "
              f"q_rate={newq/max(elapsed,1e-9):.3f}/s pair_rate={(total_pairs-start_pairs)/max(elapsed,1e-9):.1f}/s "
              f"vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB",flush=True)
        del enc,lens,vals,qs,ps,owners;gc.collect()

    del model,tok;gc.collect();torch.cuda.empty_cache()
    return np.asarray(scores,dtype=np.float32)

def rank(short,score):
    out=np.empty_like(short)
    for i in range(len(short)): out[i]=short[i,np.lexsort((short[i],-score[i]))]
    return out

def decision(delta,fd):
    worst=min(fd.values())
    if delta<=0: return "KILL_BGE"
    if delta<.0015 or worst<-.0015: return "ARCHIVE_WEAK_COMPLEMENT"
    if delta>=.0035 and worst>=0: return "STRONG_COMPLEMENT"
    if delta>=.0015 and worst>=0: return "KEEP_COMPLEMENT_FULL_OOF_WORTHY"
    return "ARCHIVE_UNSTABLE_COMPLEMENT"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--query-block",type=int,default=32)
    ap.add_argument("--max-batch",type=int,default=96)
    ap.add_argument("--long-batch",type=int,default=64)
    ap.add_argument("--token-budget",type=int,default=65536)
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)

    print("[1/6] load world",flush=True)
    qids,questions,golds,folds,docs,shortlist,baseline,y,names=s1.world()
    tr,dv,cert=s1.split(qids,folds)
    print(f"  train01={len(tr)} dev2={len(dv)} CERT34={len(cert)}",flush=True)
    if len(cert)!=2796: raise RuntimeError(f"CERT size drift {len(cert)}")

    print("[2/6] verify frozen adapter/config",flush=True)
    gate,adapter_sha,contract,ch=verify_frozen(qids,cert)
    print(f"  adapter={adapter_sha}",flush=True)
    print("  alpha(BGE)=0.20; NO RETRAINING; NO CERT TUNING",flush=True)

    print("[3/6] CERT witnesses",flush=True)
    local,vv,rr,views,texts=prep_cert(cert,shortlist,docs)

    print("[4/6] optimized cross-query inference",flush=True)
    bs=score_cert(cert,qids,questions,local,names,vv,rr,texts,views,contract,ch,args)

    print("[5/6] one-shot CERT evaluation",flush=True)
    qcert=[qids[int(i)] for i in cert]
    base=baseline[cert,:DEPTH]
    brank=rank(local,bs)
    bz=a0.zrows(bs);rz=a0.zrows(s1.rrscore(base,local))
    hrank=rank(local,FROZEN_ALPHA*bz+(1-FROZEN_ALPHA)*rz)

    comb={
        "baseline_ce_lr":s1.met(base,qcert,golds,docs),
        "bge_lora_standalone":s1.met(brank,qcert,golds,docs),
        "frozen_blend_a020":s1.met(hrank,qcert,golds,docs),
    }
    q2li={q:i for i,q in enumerate(qcert)}
    frep={};fd={}
    for f in CERT_FOLDS:
        qf=list(folds[f]);pos=np.asarray([q2li[q] for q in qf],np.int32)
        bm=s1.met(base[pos],qf,golds,docs);gm=s1.met(brank[pos],qf,golds,docs);hm=s1.met(hrank[pos],qf,golds,docs)
        d=float(hm["recall_at_5"]-bm["recall_at_5"]);fd[f]=d
        frep[f]={"queries":len(qf),"baseline_ce_lr":bm,"bge_lora_standalone":gm,"frozen_blend_a020":hm,"blend_delta_recall":d}

    bq=s1.pq(base,qcert,golds,docs);gq=s1.pq(brank,qcert,golds,docs);hq=s1.pq(hrank,qcert,golds,docs)
    delta=float(comb["frozen_blend_a020"]["recall_at_5"]-comb["baseline_ce_lr"]["recall_at_5"])
    ds=float(comb["frozen_blend_a020"]["single_gold_recall_at_5"]-comb["baseline_ce_lr"]["single_gold_recall_at_5"])
    dm=float(comb["frozen_blend_a020"]["multi_gold_recall_at_5"]-comb["baseline_ce_lr"]["multi_gold_recall_at_5"])
    dec=decision(delta,fd)

    print("[6/6] report",flush=True)
    report={
        "schema":"dsc2026.endgame.stage06b2_lite.bge_cert.v1","status":"CERT_COMPLETE",
        "methodology":{"adapter_training_folds":["fold_0","fold_1"],"selection_fold":"fold_2","cert_folds":list(CERT_FOLDS),
                       "retrained_before_cert":False,"alpha_tuned_on_cert":False,"packaging_tuned_on_cert":False,
                       "frozen_alpha_bge":FROZEN_ALPHA,"adapter_sha256":adapter_sha},
        "combined":comb,"folds":frep,
        "complementarity":{"baseline_vs_bge_oracle_recall":float(np.maximum(bq,gq).mean()),
                           "bge_wins":int(np.sum(gq>bq)),"bge_losses":int(np.sum(gq<bq)),"bge_ties":int(np.sum(gq==bq)),
                           "blend_wins":int(np.sum(hq>bq)),"blend_losses":int(np.sum(hq<bq)),"blend_ties":int(np.sum(hq==bq))},
        "effect":{"combined_delta_recall":delta,"delta_single":ds,"delta_multi":dm,"fold_deltas":fd},
        "precommitted_rule":{"kill":"delta <= 0","archive_weak":"delta < +0.0015 OR any fold < -0.0015",
                             "keep_full_oof":"delta >= +0.0015 AND both folds >= 0",
                             "strong":"delta >= +0.0035 AND both folds >= 0"},
        "decision":dec,
    }
    (OUT/"CERTIFICATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 06B2-Lite — BGE LoRA One-Shot CERT","",
           "**No retraining. Alpha=0.20 frozen from fold2. No CERT tuning.**","",
           "| Population | Baseline R@5 | BGE standalone | Frozen blend | Delta |",
           "|---|---:|---:|---:|---:|"]
    cb=comb["baseline_ce_lr"]["recall_at_5"];cg=comb["bge_lora_standalone"]["recall_at_5"];chh=comb["frozen_blend_a020"]["recall_at_5"]
    lines.append(f"| folds3+4 | {cb:.6f} | {cg:.6f} | {chh:.6f} | {delta:+.6f} |")
    for f in CERT_FOLDS:
        x=frep[f];lines.append(f"| {f} | {x['baseline_ce_lr']['recall_at_5']:.6f} | {x['bge_lora_standalone']['recall_at_5']:.6f} | {x['frozen_blend_a020']['recall_at_5']:.6f} | {x['blend_delta_recall']:+.6f} |")
    lines+=["",f"- Delta single: **{ds:+.6f}**",f"- Delta multi: **{dm:+.6f}**",
            f"- BGE W/L: **{report['complementarity']['bge_wins']}/{report['complementarity']['bge_losses']}**",
            f"- Blend W/L: **{report['complementarity']['blend_wins']}/{report['complementarity']['blend_losses']}**",
            f"- Oracle: **{report['complementarity']['baseline_vs_bge_oracle_recall']:.6f}**",
            f"- Decision: **{dec}**",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*118)
    print("ONE-SHOT CERT — NO RETRAINING — alpha(BGE)=0.20 FROZEN")
    for n,m in comb.items():
        print(f"{n:25s} R={m['recall_at_5']:.9f} P={m['precision_at_5']:.9f} single={m['single_gold_recall_at_5']:.9f} multi={m['multi_gold_recall_at_5']:.9f}")
    print(f"DELTA combined={delta:+.9f} single={ds:+.9f} multi={dm:+.9f}")
    for f in CERT_FOLDS: print(f"{f} delta={fd[f]:+.9f}")
    print(f"BGE W/L={report['complementarity']['bge_wins']}/{report['complementarity']['bge_losses']} "
          f"BLEND W/L={report['complementarity']['blend_wins']}/{report['complementarity']['blend_losses']} "
          f"ORACLE={report['complementarity']['baseline_vs_bge_oracle_recall']:.9f}")
    print("DECISION:",dec)
    print("REPORT:",OUT/"REPORT.md")
    print("="*118)

if __name__=="__main__": main()

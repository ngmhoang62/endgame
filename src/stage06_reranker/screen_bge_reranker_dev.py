#!/usr/bin/env python
from __future__ import annotations
import gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage06_evidence.benchmark_evidence_packaging as a0

MODEL=ROOT/"models/rerankers/bge-reranker-v2-m3"
MANIFEST=ROOT/"reports/stage06b0_bge_reranker_materialization/MODEL_MANIFEST.json"
CACHE=ROOT/"cache/stage06b0_bge_reranker"
OUT=ROOT/"reports/stage06b0_bge_reranker"

DEV_FOLDS=("fold_0","fold_1","fold_2")
DEPTH=12
MAX_LENGTHS=(512,1024)
OVERLAPS=(0.0,0.5)
AGGS=("ait","lal","max_raw","mean_z")

def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def stable(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load_bge():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    m=rj(MANIFEST)
    if m.get("status")!="PASS": raise RuntimeError("BGE manifest not PASS")
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL,local_files_only=True,dtype=torch.float16
    ).eval().to("cuda")
    return tok,model,m

def pair_budget(tok,q,title,maxlen):
    qn=len(tok.encode(q,add_special_tokens=False))
    special=int(tok.num_special_tokens_to_add(pair=True))
    prefix=""
    if title:
        prefix=f"[VĂN BẢN] {title}\n[ĐOẠN TRUY XUẤT]\n"
    pn=len(tok.encode(prefix,add_special_tokens=False)) if prefix else 0
    return max(64,maxlen-qn-special-pn-2),prefix

def windows(tok,text,width,stride):
    ids=tok.encode(text,add_special_tokens=False)
    if not ids: return [""]
    if len(ids)<=width: return [tok.decode(ids,skip_special_tokens=True)]
    out=[]
    for s in range(0,len(ids),stride):
        part=ids[s:s+width]
        if not part: break
        out.append(tok.decode(part,skip_special_tokens=True))
        if s+width>=len(ids): break
    return out

def make_windows(tok,q,text,title,maxlen,overlap):
    budget,prefix=pair_budget(tok,q,title,maxlen)
    stride=budget if overlap<=0 else max(1,int(round(budget*(1-overlap))))
    return [prefix+x for x in windows(tok,text,budget,stride)]

def forward(tok,model,pairs,batch,maxlen):
    import torch
    vals=[]; i=0; cur=batch
    while i<len(pairs):
        j=min(i+cur,len(pairs))
        try:
            enc=tok(
                [x[0] for x in pairs[i:j]],[x[1] for x in pairs[i:j]],
                padding=True,truncation=True,max_length=maxlen,return_tensors="pt"
            )
            enc={k:v.to("cuda") for k,v in enc.items()}
            with torch.inference_mode():
                logits=model(**enc,return_dict=True).logits.view(-1)
            vals.extend(float(x) for x in logits.float().cpu().tolist())
            i=j
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if cur<=1: raise
            cur=max(1,cur//2)
            print(f"[BGE] CUDA OOM -> batch={cur}",flush=True)
    return vals,cur

def cache_paths(name):
    d=CACHE/"dev_screen"; d.mkdir(parents=True,exist_ok=True)
    return d/f"{name}.f32.npy",d/f"{name}.done.npy",d/f"{name}.json"

def score_config(name,tok,model,manifest,maxlen,overlap,title_on,
                 screen_idx,qids,questions,short,names,vv,rr,texts,view_names):
    sp,dp,mp=cache_paths(name)
    contract={
        "schema":"stage06b0.bge_dev_scores.v1","model_sha":manifest["resolved_revision_sha"],
        "max_length":maxlen,"overlap":overlap,"title_on":title_on,
        "query_indices_sha":stable(list(map(int,screen_idx))),"depth":short.shape[1],
        "witness_row_sha":a0.sha(a0.witness_cache("dev_screen")[1]),
    }
    ch=stable(contract); ex=[p.exists() for p in (sp,dp,mp)]
    if any(ex) and not all(ex): raise RuntimeError(f"partial cache {name}")
    if all(ex):
        meta=rj(mp)
        if meta["contract_hash"]!=ch: raise RuntimeError(f"cache contract mismatch {name}")
        scores=np.lib.format.open_memmap(sp,mode="r+")
        done=np.lib.format.open_memmap(dp,mode="r+")
    else:
        scores=np.lib.format.open_memmap(sp,mode="w+",dtype=np.float32,
                                        shape=(len(screen_idx),short.shape[1],2))
        scores[:]=np.nan; scores.flush()
        done=np.lib.format.open_memmap(dp,mode="w+",dtype=np.uint8,shape=(len(screen_idx),))
        done[:]=0; done.flush()
        meta={"contract_hash":ch,"contract":contract,"completed":0,"pair_count":0,
              "current_batch":32 if maxlen==512 else 8}
        mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")

    pending=[i for i in range(len(screen_idx)) if int(done[i])==0]
    batch=int(meta.get("current_batch",32 if maxlen==512 else 8))
    total_pairs=int(meta.get("pair_count",0)); start=time.perf_counter(); newq=0
    for st in range(0,len(pending),12):
        lis=pending[st:st+12]; pairs=[]; owners=[]
        for li in lis:
            qi=int(screen_idx[li]); q=questions[qids[qi]]
            for pos,d0 in enumerate(short[li]):
                didx=int(d0); title=a0.clean_title(names[didx]) if title_on else ""
                for fi in range(2):
                    v=view_names[int(vv[li,pos,fi])]
                    raw=texts[v][int(rr[li,pos,fi])]
                    for w in make_windows(tok,q,raw,title,maxlen,overlap):
                        pairs.append((q,w)); owners.append((li,pos,fi))
        vals,batch=forward(tok,model,pairs,batch,maxlen)
        tmp={li:np.full((short.shape[1],2),-np.inf,np.float32) for li in lis}
        for (li,pos,fi),val in zip(owners,vals):
            if val>tmp[li][pos,fi]: tmp[li][pos,fi]=val
        for li in lis:
            if not np.isfinite(tmp[li]).all(): raise RuntimeError(f"nonfinite scores li={li}")
            scores[li]=tmp[li]; done[li]=1; newq+=1
        total_pairs+=len(pairs); scores.flush(); done.flush()
        completed=int(np.asarray(done).sum())
        meta={"contract_hash":ch,"contract":contract,"completed":completed,
              "pair_count":total_pairs,"current_batch":batch,
              "status":"PASS" if completed==len(screen_idx) else "IN_PROGRESS"}
        mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        print(f"[{name}] {completed}/{len(screen_idx)} pairs={total_pairs} batch={batch} "
              f"rate={newq/max(time.perf_counter()-start,1e-9):.2f} q/s",flush=True)
    return np.asarray(scores,dtype=np.float32),total_pairs

def agg(sem,kind):
    a=sem[:,:,0]; l=sem[:,:,1]
    if kind=="ait": return a
    if kind=="lal": return l
    if kind=="max_raw": return np.maximum(a,l)
    if kind=="mean_z": return .5*(a0.zrows(a)+a0.zrows(l))
    raise RuntimeError(kind)

def main():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    print("[1/7] Reproduce exact Stage06A0 DEV screen",flush=True)
    qids,questions,golds,folds,stress,docs,passages=b1.load_world()
    shortlist30,_=b1.load_shortlist(len(qids)); sources=b1.load_sources(len(qids))
    current_ce=np.load(ROOT/"cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy")
    baseline,X45,y,names45=a0.current_ce_lr(
        qids,questions,golds,folds,docs,stress,shortlist30,sources,current_ce
    )
    q2i={q:i for i,q in enumerate(qids)}
    dev_idx=np.asarray([q2i[q] for f in DEV_FOLDS for q in folds[f]],np.int32)
    pos,meta=a0.choose_screen(
        dev_idx,baseline,shortlist30[dev_idx,:a0.FULL_DEPTH],qids,golds,docs
    )
    screen_idx=dev_idx[pos]; short=shortlist30[screen_idx,:DEPTH]
    qsel=[qids[i] for i in screen_idx]
    if len(screen_idx)!=353: raise RuntimeError(f"screen population drift: {len(screen_idx)}")
    print("  screen",meta,flush=True)

    print("[2/7] Reuse semantic witnesses",flush=True)
    vv,rr,sim,view_names=a0.select_witnesses("dev_screen",screen_idx,short,docs)
    texts=a0.load_selected_texts(vv,rr,view_names); names=a0.load_names(docs)

    print("[3/7] Load fresh pinned BGE reranker",flush=True)
    tok,model,manifest=load_bge()
    params=int(manifest["parameter_count"]); active=1731560449+params
    if active>4_000_000_000: raise RuntimeError(f"parameter budget exceeded: {active}")
    print(f"  BGE={params/1e9:.3f}B active_if_promoted={active/1e9:.3f}B/4B")

    print("[4/7] Model-specific packaging screen — title OFF",flush=True)
    rows=[]; score_store={}
    for ml in MAX_LENGTHS:
        for ov in OVERLAPS:
            name=f"ml{ml}_ov{int(ov*100)}_title0"
            sem,pairs=score_config(name,tok,model,manifest,ml,ov,False,
                screen_idx,qids,questions,short,names,vv,rr,texts,view_names)
            score_store[(ml,ov,False)]=sem
            for ag in AGGS:
                s=agg(sem,ag)
                rows.append({
                    "max_length":ml,"overlap":ov,"title_on":False,"aggregation":ag,
                    "mean_query_auc":a0.query_auc(s,short,qsel,golds,docs),
                    "semantic_recall_at5":float(a0.per_query_recall(
                        a0.rank_from_score(s,short),qsel,golds,docs).mean()),
                    "pair_count":pairs,
                })
            candidates=[r for r in rows if r["max_length"]==ml and r["overlap"]==ov and not r["title_on"]]
            b=max(candidates,key=lambda r:(r["mean_query_auc"],r["semantic_recall_at5"],-r["pair_count"]))
            print(f"  ml={ml} ov={ov}: {b['aggregation']} AUC={b['mean_query_auc']:.6f} "
                  f"R={b['semantic_recall_at5']:.6f} pairs={pairs}",flush=True)

    best0=max(rows,key=lambda r:(r["mean_query_auc"],r["semantic_recall_at5"],-r["pair_count"]))
    frozen={k:best0[k] for k in ("max_length","overlap","aggregation")}
    print("[FREEZE-1]",frozen,flush=True)

    print("[5/7] Title OFF vs ON only for frozen packaging",flush=True)
    ml=int(frozen["max_length"]); ov=float(frozen["overlap"]); ag=frozen["aggregation"]
    sem_on,pairs_on=score_config(f"ml{ml}_ov{int(ov*100)}_title1",tok,model,manifest,
        ml,ov,True,screen_idx,qids,questions,short,names,vv,rr,texts,view_names)
    on={"max_length":ml,"overlap":ov,"title_on":True,"aggregation":ag,
        "mean_query_auc":a0.query_auc(agg(sem_on,ag),short,qsel,golds,docs),
        "semantic_recall_at5":float(a0.per_query_recall(
            a0.rank_from_score(agg(sem_on,ag),short),qsel,golds,docs).mean()),
        "pair_count":pairs_on}
    off=[r for r in rows if r["max_length"]==ml and r["overlap"]==ov
         and r["aggregation"]==ag and not r["title_on"]][0]
    rows.append(on); key=lambda r:(r["mean_query_auc"],r["semantic_recall_at5"],-r["pair_count"])
    winner=on if key(on)>key(off) else off
    frozen["title_on"]=bool(winner["title_on"])
    print("[FREEZE-2]",frozen,flush=True)

    print("[6/7] Complementarity vs current CE-LR",flush=True)
    final_sem=sem_on if frozen["title_on"] else score_store[(ml,ov,False)]
    bge_rank=a0.rank_from_score(agg(final_sem,ag),short)
    base_rank=baseline[screen_idx,:DEPTH]
    base_r=a0.per_query_recall(base_rank,qsel,golds,docs)
    bge_r=a0.per_query_recall(bge_rank,qsel,golds,docs)
    comp={"baseline_screen_recall":float(base_r.mean()),"bge_screen_recall":float(bge_r.mean()),
          "best_of_two_oracle_recall":float(np.maximum(base_r,bge_r).mean()),
          "bge_wins":int(np.sum(bge_r>base_r)),"bge_losses":int(np.sum(bge_r<base_r)),
          "ties":int(np.sum(bge_r==base_r))}

    print("[7/7] Write DEV-only report",flush=True)
    report={"schema":"dsc2026.endgame.stage06b0.bge_independent_reranker_dev.v1",
        "status":"DEV_SCREEN_COMPLETE_CERT_UNTOUCHED","cert_touched":False,
        "model":{"id":manifest["model_id"],"sha":manifest["resolved_revision_sha"],
                 "parameter_count":params,"active_pipeline_params_if_promoted":active,
                 "budget":4_000_000_000},
        "screen_population":meta,"rows":rows,"frozen":frozen,"winner":winner,
        "complementarity":comp,
        "historical_evidence_policy":"historical BGE results are hypothesis prior only"}
    (OUT/"DEV_SCREEN.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    del model,tok; gc.collect()
    import torch; torch.cuda.empty_cache()

    print("="*108)
    print("DEV ONLY — CERT UNTOUCHED")
    print("FROZEN:",frozen)
    print(f"BGE winner AUC={winner['mean_query_auc']:.9f} semantic_R={winner['semantic_recall_at5']:.9f}")
    print(f"BASE/BGE/ORACLE R={comp['baseline_screen_recall']:.9f}/{comp['bge_screen_recall']:.9f}/{comp['best_of_two_oracle_recall']:.9f}")
    print(f"BGE W/L/T={comp['bge_wins']}/{comp['bge_losses']}/{comp['ties']}")
    print("REPORT:",OUT/"DEV_SCREEN.json")
    print("="*108)

if __name__=="__main__":
    main()

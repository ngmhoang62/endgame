#!/usr/bin/env python
"""
Stage07AA — final local tournament over ONLY the two clean winning signals.

Goal: try to beat the already-private-winning W75 without introducing a new
model or any teacher/distillation signal.

Signals:
  P = Stage07R gold document propensity
  A = Stage07O AIT raw-centroid gold head

Families:
  1) z-score linear blend
  2) weighted Borda/rank blend
  3) weighted reciprocal-rank fusion (RRF)
  4) agreement-aware dynamic z blend
  5) boundary-confidence dynamic z blend

Every routing condition uses ONLY prediction geometry (P/A scores/ranks),
never labels. Hyperparameters are checked with leave-one-fold-out selection.

Run from repo root:
  python src/stage07_local/run_stage07aa_final_fusion_tournament.py
"""
from __future__ import annotations

import json, math, sys, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_final_clean_fusion_private as yf

OUT=ROOT/"reports/stage07aa_final_fusion_tournament"
SUB=ROOT/"submissions/endgame_20260923"
P_PROP=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
P_AIT=ROOT/"cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"

W75_R=0.9505364039479332
W75_NESTED=0.9502503218423687
W75_FOLDS={
    "fold_0":0.9438527519656896,
    "fold_1":0.9464328322596038,
    "fold_2":0.9581248510841077,
    "fold_3":0.9579160705770149,
    "fold_4":0.9463519313304721,
}

def ranks_desc(s):
    # rank position 0..K-1 for each candidate column
    order=np.argsort(-s,axis=1,kind="stable")
    r=np.empty_like(order)
    rows=np.arange(len(s))[:,None]
    r[rows,order]=np.arange(s.shape[1])[None,:]
    return r

def top5_sets_from_ranks(r):
    return r<5

def score_config(cfg,P,A,rP,rA):
    kind=cfg["kind"]
    if kind=="z":
        w=cfg["w"]
        return w*P+(1-w)*A
    if kind=="borda":
        w=cfg["w"]
        # higher is better
        return -(w*rP+(1-w)*rA).astype(np.float32)
    if kind=="rrf":
        w=cfg["w"]; k=cfg["k"]
        return w/(k+1+rP)+(1-w)/(k+1+rA)
    if kind=="agree":
        base=cfg["base"]; hi=cfg["hi"]; t=cfg["overlap_le"]
        ov=(top5_sets_from_ranks(rP)&top5_sets_from_ranks(rA)).sum(axis=1)
        w=np.where(ov<=t,hi,base).astype(np.float32)
        return w[:,None]*P+(1-w[:,None])*A
    if kind=="weak_boundary":
        # If W75 boundary is uncertain, lean more on propensity; otherwise stay W75.
        base=cfg["base"]; hi=cfg["hi"]; q=cfg["quantile"]
        s0=base*P+(1-base)*A
        so=np.sort(s0,axis=1)[:,::-1]
        gap=so[:,4]-so[:,5]
        thr=float(np.quantile(gap,q))
        w=np.where(gap<=thr,hi,base).astype(np.float32)
        return w[:,None]*P+(1-w[:,None])*A
    if kind=="prop_plateau":
        # A broad propensity plateau is a label-free proxy for "several plausible docs".
        base=cfg["base"]; hi=cfg["hi"]; q=cfg["quantile"]
        po=np.sort(P,axis=1)[:,::-1]
        # top5-vs-top8 spread; small means many documents remain competitive.
        spread=po[:,4]-po[:,7]
        thr=float(np.quantile(spread,q))
        w=np.where(spread<=thr,hi,base).astype(np.float32)
        return w[:,None]*P+(1-w[:,None])*A
    raise ValueError(kind)

def rank_docs(short,s):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-s[i]))]
    return out

def per_query(rank,qids,golds,docs):
    vals=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q])
        vals[i]=len(g & {docs[int(x)] for x in rank[i,:5]})/len(g)
    return vals

def metric(score,short,qids,golds,docs,folds,q2i):
    rr=rank_docs(short,score)
    rq=per_query(rr,qids,golds,docs)
    pf={}
    for f,ids in folds.items():
        ix=np.asarray([q2i[q] for q in ids],np.int32)
        pf[f]=float(rq[ix].mean())
    return float(rq.mean()),pf,rq

def configs():
    cs=[]
    # global z blend around known optimum
    for w in np.arange(.65,.901,.025):
        cs.append({"kind":"z","w":round(float(w),3)})
    # rank-space blends can generalize better under score-calibration drift
    for w in np.arange(.55,.951,.05):
        cs.append({"kind":"borda","w":round(float(w),3)})
    for k in (0,2,5,10,20,60):
        for w in np.arange(.55,.951,.05):
            cs.append({"kind":"rrf","k":int(k),"w":round(float(w),3)})
    # modest dynamic routing only
    for t in (2,3,4):
        for hi in (.80,.85,.90,.95):
            cs.append({"kind":"agree","base":.75,"hi":hi,"overlap_le":t})
    for q in (.20,.35,.50):
        for hi in (.80,.85,.90,.95):
            cs.append({"kind":"weak_boundary","base":.75,"hi":hi,"quantile":q})
            cs.append({"kind":"prop_plateau","base":.75,"hi":hi,"quantile":q})
    return cs

def cfgname(c):
    if c["kind"]=="z": return f"z_w{int(round(100*c['w']))}"
    if c["kind"]=="borda": return f"borda_w{int(round(100*c['w']))}"
    if c["kind"]=="rrf": return f"rrf_k{c['k']}_w{int(round(100*c['w']))}"
    if c["kind"]=="agree": return f"agree_t{c['overlap_le']}_b75_h{int(round(100*c['hi']))}"
    if c["kind"]=="weak_boundary": return f"weak_q{int(100*c['quantile'])}_h{int(100*c['hi'])}"
    if c["kind"]=="prop_plateau": return f"plateau_q{int(100*c['quantile'])}_h{int(100*c['hi'])}"
    return str(c)

def write_zip(label,score,qids,docs,short):
    rr=rank_docs(short,score)
    ans={}
    for i,q in enumerate(qids):
        row=[docs[int(x)] for x in rr[i,:5]]
        if len(row)!=5 or len(set(row))!=5:
            raise RuntimeError(f"bad row {q}")
        ans[q]={"answer":row}
    SUB.mkdir(parents=True,exist_ok=True)
    jp=SUB/f"{label}.json"; zp=SUB/f"{label}.zip"
    payload=(json.dumps(ans,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    jp.write_bytes(payload)
    info=zipfile.ZipInfo("submission.json",date_time=(2026,9,23,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o644<<16
    with zipfile.ZipFile(zp,"w",compresslevel=9) as z:z.writestr(info,payload)
    with zipfile.ZipFile(zp) as z:
        if z.namelist()!=["submission.json"] or z.read("submission.json")!=payload:
            raise RuntimeError("zip parity failed")
    return jp,zp,ans

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qids,_,golds,folds,stress,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    q2i={q:i for i,q in enumerate(qids)}

    P=ait.zrows(np.load(P_PROP))
    A=ait.zrows(np.load(P_AIT))
    if P.shape!=A.shape or P.shape!=(len(qids),30):
        raise RuntimeError("score shape drift")
    rP,rA=ranks_desc(P),ranks_desc(A)

    rows=[]
    print("[1/4] evaluate clean fusion families",flush=True)
    for c in configs():
        s=score_config(c,P,A,rP,rA)
        R,pf,rq=metric(s,short,qids,golds,docs,folds,q2i)
        fv=np.asarray(list(pf.values()))
        robust=float(fv.mean()-.35*fv.std())
        rows.append({"name":cfgname(c),"config":c,"R":R,"folds":pf,"robust_obj":robust})

    # Make sure W75 parity is exact.
    w75=next(x for x in rows if x["name"]=="z_w75")
    if abs(w75["R"]-W75_R)>2e-9:
        raise RuntimeError(f"W75 parity failed {w75['R']}")

    robust_best=max(rows,key=lambda x:(x["robust_obj"],x["R"]))
    raw_best=max(rows,key=lambda x:(x["R"],x["robust_obj"]))

    # True leave-one-fold-out config selection.
    nested=[]
    for held in folds:
        choices=[]
        for x in rows:
            vv=np.asarray([v for f,v in x["folds"].items() if f!=held])
            obj=float(vv.mean()-.35*vv.std())
            choices.append((obj,x["R"],x))
        sel=max(choices,key=lambda z:z[:-1])[-1]
        nested.append({"held":held,"name":sel["name"],"held_R":sel["folds"][held]})
    nested_R=float(np.average(
        [x["held_R"] for x in nested],
        weights=[len(folds[x["held"]]) for x in nested]
    ))

    # Family-level nested selection is a harsher check: pick best within each family
    # on four folds, then apply to held fold.
    family_nested={}
    for family in ("z","borda","rrf","agree","weak_boundary","prop_plateau"):
        fr=[x for x in rows if x["config"]["kind"]==family]
        vals=[]
        sels=[]
        for held in folds:
            cands=[]
            for x in fr:
                vv=np.asarray([v for f,v in x["folds"].items() if f!=held])
                obj=float(vv.mean()-.35*vv.std())
                cands.append((obj,x["R"],x))
            sel=max(cands,key=lambda z:z[:-1])[-1]
            vals.append(sel["folds"][held]); sels.append((held,sel["name"],sel["folds"][held]))
        family_nested[family]={
            "R":float(np.average(vals,weights=[len(folds[f]) for f in folds])),
            "selections":sels
        }

    # Oracle among all configs: diagnosis only.
    # Recompute per-query recalls for top few candidates to avoid huge memory.
    top_candidates=sorted(rows,key=lambda x:x["R"],reverse=True)[:12]
    rv=[]
    for x in top_candidates:
        s=score_config(x["config"],P,A,rP,rA)
        rr=rank_docs(short,s)
        rv.append(per_query(rr,qids,golds,docs))
    oracle=float(np.stack(rv,axis=1).max(axis=1).mean())

    print("[BEST robust]",robust_best,flush=True)
    print("[BEST raw]",raw_best,flush=True)
    print("[NESTED all]",nested,flush=True)
    print(f"[NESTED all R] {nested_R:.9f}",flush=True)
    print("[FAMILY nested]",json.dumps(family_nested,indent=2),flush=True)
    print(f"[TOP12 arm oracle] {oracle:.9f}",flush=True)

    report={
        "schema":"stage07aa.final_clean_fusion_tournament.v1",
        "teacher_used":False,"distillation_used":False,
        "w75_reference":{"R":W75_R,"nested_R":W75_NESTED,"folds":W75_FOLDS},
        "robust_best":robust_best,"raw_best":raw_best,
        "nested":nested,"nested_R":nested_R,
        "family_nested":family_nested,
        "top12_arm_oracle":oracle,
        "rows":rows,
    }
    (OUT/"OOF_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

    # Action rule: don't spend last submission on a tiny apparent win.
    actionable=(
        raw_best["R"] >= W75_R + .00030
        and max(v["R"] for v in family_nested.values()) >= W75_NESTED + .00015
    )
    if not actionable:
        print("[2/4] No fusion family clears the last-shot robustness gate.",flush=True)
        print("DECISION: WAIT_FOR_QWEN",flush=True)
        return

    chosen=raw_best
    print("[2/4] build private P/A scores",flush=True)
    pqids,pdocs,pshort,pps,PX45,PX57,pqe,pcent=yf.private_propensity()
    pas=yf.private_ait_score(pqids,pdocs,pshort,PX45,PX57,pqe,pcent)
    PP=ait.zrows(pps); AA=ait.zrows(pas)
    rrP,rrA=ranks_desc(PP),ranks_desc(AA)
    final=score_config(chosen["config"],PP,AA,rrP,rrA)

    print("[3/4] package candidate",flush=True)
    label="GOLD_STAGE07AA_"+chosen["name"].upper()+"_K5"
    jp,zp,ans=write_zip(label,final,pqids,pdocs,pshort)

    churn=None
    oldp=SUB/"GOLD_PROP_AIT_W75_K5.json"
    if oldp.exists():
        old=json.loads(oldp.read_text(encoding="utf-8"))
        churn=sum(set(old[q]["answer"])!=set(ans[q]["answer"]) for q in pqids)
    report["private"]={"label":label,"zip":str(zp.relative_to(ROOT)).replace("\\","/"),
                       "set_churn_vs_W75":churn}
    (OUT/"OOF_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("[4/4] READY",zp,flush=True)
    print("set_churn_vs_W75",churn,flush=True)
    print("DECISION: CANDIDATE_FOR_LAST_SUBMISSION",flush=True)

if __name__=="__main__":
    main()

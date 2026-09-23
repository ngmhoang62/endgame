#!/usr/bin/env python
"""
LAST SHOT — clean 3-signal fusion audit and private materialization.

Signals are all competition-valid / gold-supervised:
  P = Stage07R document propensity
  A = Stage07O AIT raw-centroid gold head
  L = Stage07P LAL raw-centroid gold head

No teacher. No distillation. No Qwen. No new foundation model.

The script:
  1) searches a small simplex around the already-private-winning W75 solution,
  2) computes exact fold recalls and leave-one-fold-out weight selection,
  3) refuses to "promote" a 3-way blend unless it beats W75 robustly,
  4) materializes the best actionable private ZIP.

Run from repo root:
  python src/stage07_local/run_last_shot_clean_trifusion.py
"""
from __future__ import annotations

import argparse, json, sys, zipfile
from pathlib import Path

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_lal_gold_centroid_oof as lal
import src.stage07_local.materialize_gold_centroid_private as mp
import src.stage07_local.materialize_lal_gold_private as mlal
import src.stage07_local.run_final_clean_fusion_private as yfinal

OUT=ROOT/"reports/stage07z_last_shot_trifusion"
SUB=ROOT/"submissions/endgame_20260923"

P_PROP=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
P_AIT=ROOT/"cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"
P_LAL=ROOT/"cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy"

W75=(.75,.25,0.0)
W75_OOF=0.9505364039479332
W75_NESTED=0.9502503218423687

def fast_metrics(score,y,gcount,fold_index,multi_mask):
    # Recall only needs the SET of top5 positions; argpartition is much faster.
    pos=np.argpartition(-score,4,axis=1)[:,:5]
    hits=np.take_along_axis(y,pos,axis=1).sum(axis=1).astype(np.float64)
    rq=hits/gcount
    folds={f:float(rq[idx].mean()) for f,idx in fold_index.items()}
    return {
        "R":float(rq.mean()),
        "multi":float(rq[multi_mask].mean()),
        "folds":folds,
        "rq":rq,
    }

def grid():
    vals=np.round(np.arange(0,1.0001,.025),3)
    rows=[]
    # Deliberately local around the known good solution; don't overfit a huge simplex.
    for wp in vals:
        if wp < .55 or wp > .90: continue
        for wa in vals:
            wl=round(1.0-wp-wa,3)
            if wl < -1e-9: continue
            if wa < .05 or wa > .40: continue
            if wl < -1e-9 or wl > .25: continue
            if abs(wp+wa+wl-1)>1e-7: continue
            rows.append((float(wp),float(wa),float(wl)))
    if W75 not in rows: rows.append(W75)
    return sorted(set(rows))

def exact_rank(short,score):
    return ait.rank(short,score)

def write_zip(label,score,qids,docs,short):
    rank=exact_rank(short,score)
    ans={}
    for i,q in enumerate(qids):
        row=[docs[int(x)] for x in rank[i,:5]]
        if len(row)!=5 or len(set(row))!=5:
            raise RuntimeError(f"invalid answer {q}")
        ans[q]={"answer":row}
    SUB.mkdir(parents=True,exist_ok=True)
    jp=SUB/f"{label}.json"; zp=SUB/f"{label}.zip"
    payload=(json.dumps(ans,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    jp.write_bytes(payload)
    info=zipfile.ZipInfo("submission.json",date_time=(2026,9,23,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o644<<16
    with zipfile.ZipFile(zp,"w",compresslevel=9) as z:
        z.writestr(info,payload)
    with zipfile.ZipFile(zp) as z:
        if z.namelist()!=["submission.json"] or z.read("submission.json")!=payload:
            raise RuntimeError("ZIP parity failed")
    return jp,zp,ans

def get_private_lal(pqids,pdocs,pshort,PX45,PX57,args):
    # Reuse a clean fulltrain checkpoint if one already exists. Otherwise train it now.
    if not mlal.CK.exists():
        report=json.loads((lal.OUT/"OOF_REPORT.json").read_text(encoding="utf-8"))
        if report.get("teacher_used") is not False or report.get("decision") not in ("PROMOTE_LAL","PROMOTE_FIXED_ENSEMBLE"):
            raise RuntimeError("LAL OOF provenance/gate failed")
        print("[private-LAL] fulltrain small gold-only LAL head",flush=True)
        mlal.train_lal(args,report)
    lm=mlal.load_head(mlal.CK,PX57.shape[-1],args.device)
    lq=np.asarray(np.load(mp.PRIVATE_CACHE/"query/lal.npy"),np.float32).copy()
    lc=lal.parent_centroids(pdocs)
    if lq.shape!=(len(pqids),1024):
        raise RuntimeError("private LAL query shape drift")
    residual=mp.score_private(pshort,PX57,lq,lc,lm,args)
    prior=joblib.load(b1.CACHE/"fulltrain_ce_fusion.joblib")["model"].decision_function(
        PX45.reshape(-1,45)).reshape(len(pqids),ait.FULL
    )
    score=ait.zrows(prior)
    score[:,:ait.DEPTH]+=float(lm.alpha().detach().cpu())*residual
    return score.astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--batch-queries",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--lr",type=float,default=8e-4)
    ap.add_argument("--seed",type=int,default=276)
    args=ap.parse_args()

    OUT.mkdir(parents=True,exist_ok=True)
    for p in (P_PROP,P_AIT,P_LAL):
        if not p.exists(): raise FileNotFoundError(p)

    qids,_,golds,folds,stress,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    y=qg.labels_for(short,qids,golds,docs).astype(np.uint8)
    gcount=np.asarray([len(golds[q]) for q in qids],np.float64)
    multi=gcount>1
    q2i={q:i for i,q in enumerate(qids)}
    fidx={f:np.asarray([q2i[q] for q in ids],np.int32) for f,ids in folds.items()}

    P=ait.zrows(np.load(P_PROP))
    A=ait.zrows(np.load(P_AIT))
    L=ait.zrows(np.load(P_LAL))
    if P.shape!=A.shape or A.shape!=L.shape or P.shape!=(len(qids),30):
        raise RuntimeError("OOF score shape drift")

    configs=[]
    print("[1/5] controlled 3-way OOF search",flush=True)
    for wp,wa,wl in grid():
        s=wp*P+wa*A+wl*L
        m=fast_metrics(s,y,gcount,fidx,multi)
        fv=np.asarray(list(m["folds"].values()),np.float64)
        robust=float(fv.mean()-.35*fv.std())
        configs.append({
            "w_prop":wp,"w_ait":wa,"w_lal":wl,
            "R":m["R"],"multi":m["multi"],"folds":m["folds"],
            "robust_obj":robust,
        })

    # Full-fold robust choice.
    best=max(configs,key=lambda x:(x["robust_obj"],x["R"],-x["w_lal"],-abs(x["w_prop"]-.75)))

    # Nested: choose weights from the four non-held folds only.
    nested=[]
    for held in folds:
        cand=[]
        for x in configs:
            vals=np.asarray([v for f,v in x["folds"].items() if f!=held])
            obj=float(vals.mean()-.35*vals.std())
            cand.append((obj,x["R"],-x["w_lal"],-abs(x["w_prop"]-.75),x))
        sel=max(cand,key=lambda t:t[:-1])[-1]
        nested.append({
            "held":held,
            "weights":[sel["w_prop"],sel["w_ait"],sel["w_lal"]],
            "held_R":sel["folds"][held],
        })
    nested_R=float(np.average(
        [x["held_R"] for x in nested],
        weights=[len(folds[x["held"]]) for x in nested]
    ))

    # Also find the best raw OOF score, only as a diagnostic.
    raw_best=max(configs,key=lambda x:(x["R"],x["robust_obj"]))

    print("[OOF] robust_best",best,flush=True)
    print("[OOF] raw_best",raw_best,flush=True)
    print("[OOF] nested",nested,flush=True)
    print(f"[OOF] nested_weighted_R={nested_R:.9f}",flush=True)
    print(f"[OOF] W75 reference R={W75_OOF:.9f} nested={W75_NESTED:.9f}",flush=True)

    # Promotion requires a real nested gain, not a cosmetic grid win.
    promote=(
        best["R"] >= W75_OOF + .00025
        and nested_R >= W75_NESTED + .00015
        and sum(best["folds"][f] >= {
            "fold_0":0.9438527519656896,
            "fold_1":0.9464328322596038,
            "fold_2":0.9581248510841077,
            "fold_3":0.9579160705770149,
            "fold_4":0.9463519313304721,
        }[f] - .0010 for f in folds) >= 4
    )

    report={
        "schema":"stage07z.last_shot_clean_trifusion.v1",
        "teacher_used":False,"distillation_used":False,
        "signals":["gold_document_propensity","AIT_raw_centroid_gold_head","LAL_raw_centroid_gold_head"],
        "robust_best":best,"raw_best":raw_best,
        "nested":nested,"nested_weighted_R":nested_R,
        "w75_reference":{"R":W75_OOF,"nested_R":W75_NESTED},
        "promote_over_w75":bool(promote),
    }
    (OUT/"OOF_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

    if not promote:
        print("[2/5] NO robust local improvement over W75. Do not spend final submission on trifusion.",flush=True)
        print("DECISION: KEEP_W75_WAIT_FOR_QWEN_PROBE",flush=True)
        return

    print("[2/5] promoted -> build private propensity/AIT world",flush=True)
    pqids,pdocs,pshort,pps,PX45,PX57,pqe,pcent=yfinal.private_propensity()
    pas=yfinal.private_ait_score(pqids,pdocs,pshort,PX45,PX57,pqe,pcent)

    print("[3/5] build/load clean private LAL score",flush=True)
    pls=get_private_lal(pqids,pdocs,pshort,PX45,PX57,args)

    wp,wa,wl=best["w_prop"],best["w_ait"],best["w_lal"]
    final=wp*ait.zrows(pps)+wa*ait.zrows(pas)+wl*ait.zrows(pls)

    print("[4/5] deterministic private package",flush=True)
    label=f"GOLD_LASTSHOT_P{int(round(wp*100)):02d}_A{int(round(wa*100)):02d}_L{int(round(wl*100)):02d}_K5"
    jp,zp,ans=write_zip(label,final,pqids,pdocs,pshort)

    oldp=SUB/"GOLD_PROP_AIT_W75_K5.json"
    churn=None
    if oldp.exists():
        old=json.loads(oldp.read_text(encoding="utf-8"))
        churn=sum(set(old[q]["answer"])!=set(ans[q]["answer"]) for q in pqids)

    report["private"]={
        "label":label,"json":str(jp.relative_to(ROOT)).replace("\\","/"),
        "zip":str(zp.relative_to(ROOT)).replace("\\","/"),
        "set_churn_vs_W75":churn,
    }
    (OUT/"OOF_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("[5/5] READY",zp,flush=True)
    print("set_churn_vs_W75",churn,flush=True)
    print("DECISION: CANDIDATE_FOR_FINAL_SUBMISSION",flush=True)

if __name__=="__main__":
    main()

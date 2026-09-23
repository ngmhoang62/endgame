#!/usr/bin/env python
"""
Stage07AB — clean residual sweep around the private-winning W75.

No new model. No teacher. No distillation.

Base:
    B = 0.75 * z(Stage07R propensity) + 0.25 * z(Stage07O AIT centroid)

Potential clean residual signals already cached OOF:
    S = Stage07S centroid+propensity head
    T = Stage07T alternate-evidence centroid head
    Q = Stage07Q AIT witness head
    U = Stage07U setwise raw-embedding head
    L = Stage07P LAL centroid head

We test:
  raw:       B + beta * Z(R)
  orth:      B + beta * orthogonal_component(Z(R), B), row-wise
  boundary:  same orthogonal residual but only on B ranks 4..9

beta is selected by 5-fold leave-one-fold-out. This is diagnostic/OOF only.
If a deployable signal (S or L) clearly wins, materialize it separately.

Run:
  python src/stage07_local/run_stage07ab_clean_residual_sweep.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT=ROOT/"reports/stage07ab_clean_residual_sweep"
P=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
A=ROOT/"cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"
SIGS={
    "S_centroid_propensity": ROOT/"cache/stage07s_gold_centroid_propensity/centroid_propensity_oof_scores30.f32.npy",
    "T_alt_evidence": ROOT/"cache/stage07t_gold_centroid_alt_evidence/alt_evidence_centroid_oof_scores30.f32.npy",
    "Q_witness": ROOT/"cache/stage07q_gold_witness_head/gold_witness_oof_scores30.f32.npy",
    "U_setwise": ROOT/"cache/stage07u_gold_embedding_setwise/setwise_centroid_oof_scores30.f32.npy",
    "L_lal": ROOT/"cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy",
}
W75_R=0.9505364039479332
W75_F={
    "fold_0":0.9438527519656896,
    "fold_1":0.9464328322596038,
    "fold_2":0.9581248510841077,
    "fold_3":0.9579160705770149,
    "fold_4":0.9463519313304721,
}

def rank(short,s):
    out=np.empty_like(short)
    for i in range(len(short)):
        out[i]=short[i,np.lexsort((short[i],-s[i]))]
    return out

def perq(rr,qids,golds,docs):
    x=np.empty(len(qids),np.float64)
    for i,q in enumerate(qids):
        g=set(golds[q])
        x[i]=len(g & {docs[int(d)] for d in rr[i,:5]})/len(g)
    return x

def metrics(s,short,qids,golds,docs,folds,q2i):
    rq=perq(rank(short,s),qids,golds,docs)
    pf={}
    for f,ids in folds.items():
        ix=np.asarray([q2i[q] for q in ids],np.int32)
        pf[f]=float(rq[ix].mean())
    return float(rq.mean()),pf,rq

def row_orth(r,b):
    # Remove only the row-wise component parallel to B. This is label-free.
    den=(b*b).sum(1,keepdims=True)
    coef=(r*b).sum(1,keepdims=True)/np.where(den<1e-8,1,den)
    x=r-coef*b
    sd=x.std(1,keepdims=True)
    return x/np.where(sd<1e-6,1,sd)

def bmask(b):
    order=np.argsort(-b,axis=1,kind="stable")
    rk=np.empty_like(order)
    rows=np.arange(len(b))[:,None]
    rk[rows,order]=np.arange(b.shape[1])[None,:]
    # 0-indexed ranks 3..8 => nominal positions 4..9
    return ((rk>=3)&(rk<=8)).astype(np.float32)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qids,_,golds,folds,stress,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    q2i={q:i for i,q in enumerate(qids)}
    zp=ait.zrows(np.load(P)); za=ait.zrows(np.load(A))
    B=.75*zp+.25*za
    R0,pf0,rq0=metrics(B,short,qids,golds,docs,folds,q2i)
    if abs(R0-W75_R)>2e-9: raise RuntimeError(f"W75 parity {R0}")
    mask=bmask(B)

    betas=np.round(np.arange(-.30,.301,.025),3)
    rows=[]
    print("[1/3] sweep clean cached residuals",flush=True)
    for name,path in SIGS.items():
        if not path.exists():
            print("[skip missing]",name,path,flush=True); continue
        zr=ait.zrows(np.load(path))
        orth=row_orth(zr,B)
        for mode in ("raw","orth","boundary"):
            x=zr if mode=="raw" else orth if mode=="orth" else orth*mask
            for beta in betas:
                if abs(beta)<1e-12: continue
                s=B+float(beta)*x
                R,pf,rq=metrics(s,short,qids,golds,docs,folds,q2i)
                fv=np.asarray(list(pf.values()))
                rows.append({
                    "signal":name,"mode":mode,"beta":float(beta),
                    "R":R,"folds":pf,
                    "robust_obj":float(fv.mean()-.35*fv.std()),
                    "wins_vs_w75":int((rq>rq0).sum()),
                    "losses_vs_w75":int((rq<rq0).sum()),
                })

    raw_best=max(rows,key=lambda x:(x["R"],x["robust_obj"]))
    robust_best=max(rows,key=lambda x:(x["robust_obj"],x["R"]))

    # Nested selection independently inside each (signal,mode) family.
    fam={}
    for name in sorted({x["signal"] for x in rows}):
        for mode in ("raw","orth","boundary"):
            rr=[x for x in rows if x["signal"]==name and x["mode"]==mode]
            vals=[]; sels=[]
            for held in folds:
                c=[]
                for x in rr:
                    v=np.asarray([z for f,z in x["folds"].items() if f!=held])
                    c.append((float(v.mean()-.35*v.std()),x["R"],-abs(x["beta"]),x))
                sel=max(c,key=lambda z:z[:-1])[-1]
                vals.append(sel["folds"][held])
                sels.append({"held":held,"beta":sel["beta"],"held_R":sel["folds"][held]})
            fam[f"{name}:{mode}"]={
                "nested_R":float(np.average(vals,weights=[len(folds[f]) for f in folds])),
                "selections":sels,
                "raw_best":max(rr,key=lambda x:x["R"]),
            }

    best_family=max(fam.items(),key=lambda kv:(kv[1]["nested_R"],kv[1]["raw_best"]["R"]))
    print("[BEST raw]",json.dumps(raw_best,indent=2),flush=True)
    print("[BEST robust]",json.dumps(robust_best,indent=2),flush=True)
    print("[BEST nested family]",best_family[0],json.dumps(best_family[1],indent=2),flush=True)

    # Arm oracle W75 vs top residual configs: diagnostic headroom.
    top=sorted(rows,key=lambda x:x["R"],reverse=True)[:10]
    mat=[rq0]
    for x in top:
        zr=ait.zrows(np.load(SIGS[x["signal"]]))
        orth=row_orth(zr,B)
        xx=zr if x["mode"]=="raw" else orth if x["mode"]=="orth" else orth*mask
        _,_,rq=metrics(B+x["beta"]*xx,short,qids,golds,docs,folds,q2i)
        mat.append(rq)
    oracle=float(np.stack(mat,axis=1).max(1).mean())
    print(f"[ARM ORACLE W75+TOP10] {oracle:.9f}",flush=True)

    report={
        "schema":"stage07ab.clean_residual_sweep.v1",
        "teacher_used":False,"distillation_used":False,
        "w75":{"R":R0,"folds":pf0},
        "best_raw":raw_best,"best_robust":robust_best,
        "best_nested_family":{"name":best_family[0],**best_family[1]},
        "families":fam,"arm_oracle_w75_top10":oracle,
    }
    (OUT/"OOF_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("[2/3] wrote",OUT/"OOF_AUDIT.json",flush=True)

    deployable=raw_best["signal"] in ("S_centroid_propensity","L_lal")
    actionable=(
        raw_best["R"]>=W75_R+.00030 and
        best_family[1]["nested_R"]>=W75_R-.00010 and
        raw_best["wins_vs_w75"]>=raw_best["losses_vs_w75"] and
        deployable
    )
    print("[3/3] ACTIONABLE",actionable,flush=True)
    print("DECISION:",
          "MATERIALIZE_"+raw_best["signal"] if actionable else "WAIT_FOR_QWEN_OR_NEXT_DIAGNOSTIC",
          flush=True)

if __name__=="__main__":
    main()

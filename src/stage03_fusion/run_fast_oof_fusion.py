#!/usr/bin/env python
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics

RAW=ROOT/'data/official_v1'; EVAL=ROOT/'data/evaluation_v2'
C1=ROOT/'cache/stage02c1_aiteam_representation'
B4=ROOT/'cache/stage02b4_vi_screen'; A2=ROOT/'cache/stage02a_parent_anchor'
OUT=ROOT/'reports/stage03a_fast_oof_fusion'
DEPTH=50; TOPK=5; C_VALUE=0.15
SOURCES={
 'ait_atomic':(C1/'atomic_split_2048/parent_top100_idx.npy',C1/'atomic_split_2048/parent_top100_scores.npy'),
 'ait_coarse1024':(C1/'coarse_pack_1024/parent_top100_idx.npy',C1/'coarse_pack_1024/parent_top100_scores.npy'),
 'lal_b4':(B4/'vnlegal_lal/parent_top100_idx.npy',B4/'vnlegal_lal/parent_top100_scores.npy'),
 'bm25':(A2/'bm25_idx.npy',A2/'bm25_scores.npy'),
}
SN=list(SOURCES)

def rj(p): return json.loads(p.read_text(encoding='utf-8'))

def load_eval():
    train=rj(RAW/'train.json')
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/'primary_golds_6991.json').items()}
    qids=[str(q) for q in train if str(q) in golds]
    if len(qids)!=6991: raise RuntimeError(f'Expected 6991 qids, got {len(qids)}')
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/'folds_v2.json').items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/'stress_slices.json').items()}
    return qids,golds,folds,stress

def load_docs():
    out=[]
    with (EVAL/'retrieval_corpus_8512.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(str(json.loads(line)['id']))
    if len(out)!=8512: raise RuntimeError('8512 doc contract drift')
    return out

def load_sources(nq):
    out={}
    for n,(ip,sp) in SOURCES.items():
        if not ip.exists(): raise FileNotFoundError(ip)
        if not sp.exists(): raise FileNotFoundError(sp)
        idx=np.load(ip,mmap_mode='r'); scr=np.load(sp,mmap_mode='r')
        if idx.shape!=(nq,100) or scr.shape!=(nq,100):
            raise RuntimeError(f'{n}: idx={idx.shape} scores={scr.shape}')
        out[n]=(idx,scr)
    return out

def feature_names():
    x=[]
    for s in SN: x += [f'{s}__present',f'{s}__rr10',f'{s}__rank_norm50',f'{s}__score_z',f'{s}__top_gap_z']
    x += ['source_count','best_rr10','mean_rr10_present','rrf60','min_rank_norm50','mean_rank_norm50_present','count_top5','count_top10','count_top20']
    return x
FN=feature_names()

def build_data(qids,golds,docs,src):
    d2i={d:i for i,d in enumerate(docs)}
    gold_idx=[{d2i[d] for d in golds[q]} for q in qids]
    pools=[]; total=0
    for qi in range(len(qids)):
        u=set()
        for s in SN: u.update(int(x) for x in src[s][0][qi,:DEPTH])
        a=np.asarray(sorted(u),dtype=np.int32); pools.append(a); total+=len(a)
    X=np.empty((total,len(FN)),dtype=np.float32); y=np.empty(total,dtype=np.uint8)
    cand_doc=np.empty(total,dtype=np.int32); cand_q=np.empty(total,dtype=np.int32)
    offsets=np.empty(len(qids)+1,dtype=np.int64); pos=0; oracle=[]
    for qi,q in enumerate(qids):
        offsets[qi]=pos; pool=pools[qi]; g=gold_idx[qi]
        oracle.append(len(set(map(int,pool))&g)/len(g))
        rank_maps={}; z_maps={}; gap_maps={}
        for s in SN:
            idx,scr=src[s]; ids=np.asarray(idx[qi,:DEPTH],dtype=np.int32)
            s100=np.asarray(scr[qi,:],dtype=np.float64); mu=float(s100.mean()); sd=float(s100.std())
            if not math.isfinite(sd) or sd<1e-8: sd=1.0
            top=float(s100[0]); rank_maps[s]={int(d):r+1 for r,d in enumerate(ids)}
            z_maps[s]={int(d):float((float(scr[qi,r])-mu)/sd) for r,d in enumerate(ids)}
            gap_maps[s]={int(d):float((top-float(scr[qi,r]))/sd) for r,d in enumerate(ids)}
        for d0 in pool:
            d=int(d0); feat=[]; rrs=[]; ranks=[]; c5=c10=c20=0; rrf=0.0
            for s in SN:
                r=rank_maps[s].get(d)
                if r is None: feat += [0.0,0.0,1.2,-3.0,4.0]
                else:
                    rr=1/(10+r); feat += [1.0,rr,r/DEPTH,z_maps[s][d],gap_maps[s][d]]
                    rrs.append(rr); ranks.append(r); rrf += 1/(60+r)
                    c5+=r<=5; c10+=r<=10; c20+=r<=20
            feat += [float(len(rrs)),max(rrs) if rrs else 0.0,float(np.mean(rrs)) if rrs else 0.0,rrf,
                     min(ranks)/DEPTH if ranks else 1.2,float(np.mean(ranks))/DEPTH if ranks else 1.2,float(c5),float(c10),float(c20)]
            X[pos]=feat; y[pos]=1 if d in g else 0; cand_doc[pos]=d; cand_q[pos]=qi; pos+=1
        if (qi+1)%500==0 or qi+1==len(qids): print(f'[features] {qi+1}/{len(qids)} rows={pos}',flush=True)
    offsets[-1]=pos
    return {'X':X,'y':y,'cand_doc':cand_doc,'cand_q':cand_q,'offsets':offsets,
            'rows':int(total),'positive_rows':int(y.sum()),'mean_pool':total/len(qids),'oracle_r50':float(np.mean(oracle))}

def predict_top5(model,data,qidx,docs):
    out={}; X=data['X']; off=data['offsets']; cd=data['cand_doc']
    for qi in qidx:
        a,b=int(off[qi]),int(off[qi+1]); p=model.predict_proba(X[a:b])[:,1]; di=cd[a:b]
        order=np.lexsort((di,-p)); out[int(qi)]=[docs[int(x)] for x in di[order[:TOPK]]]
    return out

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True,exist_ok=True); t0=time.perf_counter()
    qids,golds,folds,stress=load_eval(); docs=load_docs(); src=load_sources(len(qids)); data=build_data(qids,golds,docs,src)
    q2i={q:i for i,q in enumerate(qids)}; oof={}; fold_reports={}; coefs=[]
    for fname,held_ids in folds.items():
        held=np.asarray([q2i[q] for q in held_ids],dtype=np.int32); held_set=set(map(int,held))
        train_q=np.asarray([i for i in range(len(qids)) if i not in held_set],dtype=np.int32)
        mask=np.isin(data['cand_q'],train_q); Xtr=data['X'][mask]; ytr=data['y'][mask]
        print(f'[{fname}] fit rows={len(ytr)} pos={int(ytr.sum())} held={len(held)}',flush=True)
        model=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(C=C_VALUE,class_weight='balanced',solver='lbfgs',max_iter=300,tol=1e-5))])
        model.fit(Xtr,ytr); pred=predict_top5(model,data,held,docs); oof.update(pred)
        pp={qids[qi]:pred[int(qi)] for qi in held}; fold_reports[fname]=official_metrics(pp,golds,held_ids)
        lr=model.named_steps['lr']; coefs.append({'fold':fname,'intercept':float(lr.intercept_[0]),'n_iter':int(lr.n_iter_[0]),'coef_standardized':{FN[i]:float(lr.coef_[0,i]) for i in range(len(FN))}})
    if len(oof)!=len(qids): raise RuntimeError(f'OOF coverage {len(oof)}')
    preds={qids[i]:oof[i] for i in range(len(qids))}; overall=official_metrics(preds,golds,qids)
    stress_out={}
    for n,ids in stress.items():
        use=[q for q in ids if q in preds]
        if use: stress_out[n]=official_metrics(preds,golds,use)
    standalone={}
    for s in SN:
        idx=src[s][0]; sp={q:[docs[int(x)] for x in idx[i,:5]] for i,q in enumerate(qids)}
        standalone[s]=official_metrics(sp,golds,qids)
    result={'schema_version':'dsc2026.endgame.stage03a_fast_oof_fusion.v1','status':'COMPLETE',
            'candidate_contract':{'sources':SN,'per_source_depth':DEPTH,'mean_unique_pool_size':data['mean_pool'],'candidate_rows':data['rows'],'positive_candidate_rows':data['positive_rows'],'acquisition_oracle_recall_at50':data['oracle_r50']},
            'selector':{'type':'StandardScaler + LogisticRegression','C':C_VALUE,'class_weight':'balanced','solver':'lbfgs','features':FN,'fold_clean':True},
            'oof':{'overall':overall,'per_fold':fold_reports,'stress':stress_out},'standalone_at5':standalone,'fold_coefficients':coefs,'wall_seconds':time.perf_counter()-t0,
            'claim_boundary':'OOF selector baseline only; no cross-encoder features'}
    (OUT/'FAST_FUSION_OOF.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for feat in FN:
        vals=[x['coef_standardized'][feat] for x in coefs]; rows.append({'feature':feat,'mean_coef':float(np.mean(vals)),'std_coef':float(np.std(vals)),'min_coef':float(np.min(vals)),'max_coef':float(np.max(vals))})
    rows.sort(key=lambda r:-abs(r['mean_coef']))
    with (OUT/'COEFFICIENTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=['# Stage 03A — Fast OOF Fusion','',f"- Candidate oracle @50: **{data['oracle_r50']:.6f}**",f"- Mean unique pool: **{data['mean_pool']:.1f}**",f"- OOF Recall@5: **{overall['recall_at_5']:.6f}**",f"- OOF Precision@5: **{overall['precision_at_5']:.6f}**",f"- Single-gold R@5: **{overall['single_gold_recall_at_5']:.6f}**",f"- Multi-gold R@5: **{overall['multi_gold_recall_at_5']:.6f}**",'', '## Standalone controls','', '| Source | R@5 | P@5 |','|---|---:|---:|']
    for s,m in standalone.items(): lines.append(f"| {s} | {m['recall_at_5']:.6f} | {m['precision_at_5']:.6f} |")
    lines += ['', 'Fold-clean OOF only. No private labels or cross-encoder features.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','candidate_oracle_r50':data['oracle_r50'],'mean_pool_size':data['mean_pool'],'oof_recall_at5':overall['recall_at_5'],'oof_precision_at5':overall['precision_at_5'],'single_gold_recall_at5':overall['single_gold_recall_at_5'],'multi_gold_recall_at5':overall['multi_gold_recall_at_5'],'out':str(OUT)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

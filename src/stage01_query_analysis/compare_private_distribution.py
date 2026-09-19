#!/usr/bin/env python
"""Stage 01B: label-free private distribution diagnostics, v2.

This is diagnostic only. Historical CAL600 is reported only as a secondary
reference coordinate and is NOT the ENDGAME evaluation protocol.

Requires PASS from preflight_query_data.py with matching input hashes.

Run:
  python src/stage01_query_analysis/compare_private_distribution.py \
    --sota-root D:/Study/DSC2026/sota
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.query_preprocessing import model_text, strict_match_key

DEFAULT_SOTA = Path("D:/Study/DSC2026/sota")
PREFLIGHT = ROOT/"reports/stage01_query_preflight/QUERY_PREFLIGHT.json"
OUT = ROOT/"reports/stage01_private_distribution"
CACHE = ROOT/"cache/stage01_query_analysis/private_distribution"

CAL = {
    "HIST_CAL_A": (750, 850),
    "HIST_CAL_B": (1250, 1350),
    "HIST_CAL_C": (1350, 1450),
    "HIST_CAL_D": (1450, 1750),
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_questions(path: Path, require_answer=False):
    raw=read_json(path); out={}
    for qid,row in raw.items():
        if isinstance(row,str):
            if require_answer: continue
            q=row
        else:
            if require_answer and not row.get("answer"): continue
            q=row.get("question","")
        q=model_text(q)
        if q: out[str(qid)]=q
    return out


def verify_preflight(private_path, train_path, public_path):
    if not PREFLIGHT.exists():
        raise SystemExit("Missing Stage 01B preflight. Run preflight_query_data.py first.")
    p=read_json(PREFLIGHT)
    if p.get("status")!="PASS":
        raise SystemExit("Stage 01B preflight status is not PASS.")
    expected={"PRIVATE":private_path,"TRAIN":train_path,"PUBLIC":public_path}
    for name,path in expected.items():
        got=p["inputs"][name]["sha256"]
        now=sha256(path)
        if got!=now:
            raise SystemExit(f"Input changed after preflight: {name}\npreflight={got}\ncurrent={now}")
    return p


def exact_overlap(left,right):
    keys={}
    for qid,q in right.items(): keys.setdefault(strict_match_key(q),[]).append(qid)
    hits=[]
    for qid,q in left.items():
        if strict_match_key(q) in keys:
            hits.append((qid,keys[strict_match_key(q)]))
    return len(hits), hits


def lexical_space(private,train,public):
    pops=[private,train,public]
    texts=sum([list(x.values()) for x in pops],[])
    lens=[len(x) for x in pops]
    v=TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5), min_df=2,
        max_features=180000, sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    x=v.fit_transform(texts)
    a=lens[0];b=a+lens[1]
    return x[:a],x[a:b],x[b:],len(v.vocabulary_)


def top1_sparse(target,ref):
    nn=NearestNeighbors(n_neighbors=1,metric="cosine",algorithm="brute",n_jobs=-1)
    nn.fit(ref);d,i=nn.kneighbors(target)
    return (1-d[:,0]).astype(np.float32),i[:,0]


def model_source(sota, requested):
    if requested:
        p=Path(requested)
        return str(p.resolve()) if p.exists() else requested, p.exists()
    p=sota/"cache/research_v2_e5_confirmation/bundle-v1/vietlegal-e5"
    if p.exists(): return str(p.resolve()),True
    return "mainguyen9/vietlegal-e5",False


def encode(texts, source, local_only, batch):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel,AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fix environment before Stage 01B.")
    tok=AutoTokenizer.from_pretrained(source,local_files_only=local_only,trust_remote_code=True)
    mdl=AutoModel.from_pretrained(
        source,local_files_only=local_only,trust_remote_code=True,torch_dtype=torch.float16
    ).cuda().eval()
    out=[]
    with torch.inference_mode():
        for s in range(0,len(texts),batch):
            z=tok(["query: "+x for x in texts[s:s+batch]],padding=True,truncation=True,
                  max_length=512,return_tensors="pt").to("cuda")
            h=mdl(**z).last_hidden_state.float()
            m=z["attention_mask"].unsqueeze(-1)
            e=F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),dim=-1)
            out.append(e.cpu().numpy().astype(np.float32))
            if s==0 or s+batch>=len(texts):
                print(f"[embed] {min(s+batch,len(texts))}/{len(texts)}",flush=True)
    del mdl; torch.cuda.empty_cache()
    return np.concatenate(out)


def cached_encode(name,qmap,source,local_only,batch,input_hash):
    CACHE.mkdir(parents=True,exist_ok=True)
    p=CACHE/f"{name}.npy";m=CACHE/f"{name}.json"
    contract={"name":name,"input_hash":input_hash,"qids":list(qmap),
              "model":source,"prefix":"query: ","pool":"mean_mask","max_length":512}
    ch=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if p.exists() and m.exists() and read_json(m).get("contract_hash")==ch:
        z=np.load(p)
        if len(z)==len(qmap):
            print(f"[embed] cache hit {name}",flush=True);return z
    z=encode(list(qmap.values()),source,local_only,batch)
    np.save(p,z)
    m.write_text(json.dumps({"contract_hash":ch,"contract":contract,"shape":list(z.shape)},indent=2),
                 encoding="utf-8")
    return z


def top1_dense(a,b,block=256):
    bt=np.ascontiguousarray(b.T);s=np.empty(len(a),np.float32);ix=np.empty(len(a),np.int64)
    for st in range(0,len(a),block):
        q=np.ascontiguousarray(a[st:st+block]);z=q@bt
        j=z.argmax(1);s[st:st+len(j)]=z[np.arange(len(j)),j];ix[st:st+len(j)]=j
    return s,ix


def desc(a):
    a=np.asarray(a,float)
    return {
        "min":float(a.min()),"p25":float(np.quantile(a,.25)),
        "median":float(np.median(a)),"mean":float(a.mean()),
        "p75":float(np.quantile(a,.75)),"p90":float(np.quantile(a,.9)),
        "p95":float(np.quantile(a,.95)),"max":float(a.max()),
        "ge_0.90":int((a>=.90).sum()),"ge_0.95":int((a>=.95).sum()),
        "ge_0.99":int((a>=.99).sum()),
    }


def cluster_private(vectors,qids,questions):
    ks=[8,12,16,24];scores={};runs={}
    for k in ks:
        km=MiniBatchKMeans(n_clusters=k,random_state=2026,n_init=10,batch_size=256)
        lab=km.fit_predict(vectors)
        sc=silhouette_score(vectors,lab,metric="cosine",
                            sample_size=min(1200,len(vectors)),random_state=2026)
        scores[str(k)]=float(sc);runs[k]=(km,lab)
    best=max(ks,key=lambda k:scores[str(k)])
    km,lab=runs[best]
    centers=km.cluster_centers_.astype(np.float32)
    centers/=np.linalg.norm(centers,axis=1,keepdims=True).clip(min=1e-12)
    rows=[]
    for c in range(best):
        ids=np.where(lab==c)[0]
        sims=vectors[ids]@centers[c]
        rep=ids[np.argsort(-sims)[:5]]
        rows.append({"cluster":c,"size":int(len(ids)),"representatives":[
            {"qid":qids[i],"question":questions[qids[i]],"centroid_cosine":float(vectors[i]@centers[c])}
            for i in rep
        ]})
    rows.sort(key=lambda x:-x["size"])
    return {"selected_k":best,"silhouette":scores,"clusters":rows},lab


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sota-root",type=Path,default=DEFAULT_SOTA)
    ap.add_argument("--private",type=Path,default=ROOT/"private-official.json")
    ap.add_argument("--model",default=None)
    ap.add_argument("--batch-size",type=int,default=64)
    a=ap.parse_args()
    sota=a.sota_root.resolve()
    data=sota/"DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path=a.private.resolve();train_path=data/"train.json";public_path=data/"public-official.json"
    pre=verify_preflight(private_path,train_path,public_path)

    private=load_questions(private_path)
    train=load_questions(train_path)
    train_eval=load_questions(train_path,require_answer=True)
    public=load_questions(public_path)
    eval_qids=list(train_eval)
    hist={}
    for name,(lo,hi) in CAL.items():
        ids=eval_qids[lo:hi]
        hist[name]={q:train_eval[q] for q in ids}
    if {k:len(v) for k,v in hist.items()} != {
        "HIST_CAL_A":100,"HIST_CAL_B":100,"HIST_CAL_C":100,"HIST_CAL_D":300
    }:
        raise RuntimeError("Historical CAL slice contract mismatch.")

    OUT.mkdir(parents=True,exist_ok=True)
    p_lex,t_lex,u_lex,vocab=lexical_space(private,train,public)
    refs={"TRAIN":(train,t_lex),"PUBLIC":(public,u_lex)}
    train_index={q:i for i,q in enumerate(train)}
    for name,qmap in hist.items():
        rows=np.asarray([train_index[q] for q in qmap])
        refs[name]=(qmap,t_lex[rows])

    lex={}
    for name,(qmap,mat) in refs.items():
        s,i=top1_sparse(p_lex,mat)
        lex[name]=(s,i,list(qmap))

    source,local=model_source(sota,a.model)
    pv=cached_encode("private",private,source,local,a.batch_size,sha256(private_path))
    tv=cached_encode("train",train,source,local,a.batch_size,sha256(train_path))
    uv=cached_encode("public",public,source,local,a.batch_size,sha256(public_path))
    erefs={"TRAIN":(train,tv),"PUBLIC":(public,uv)}
    for name,qmap in hist.items():
        rows=np.asarray([train_index[q] for q in qmap])
        erefs[name]=(qmap,tv[rows])

    emb={}
    for name,(qmap,mat) in erefs.items():
        s,i=top1_dense(pv,mat);emb[name]=(s,i,list(qmap))

    clusters,labels=cluster_private(pv,list(private),private)
    (OUT/"PRIVATE_CLUSTERS.json").write_text(json.dumps(clusters,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    exact={name:exact_overlap(private,qmap)[0] for name,qmap in {"TRAIN":train,"PUBLIC":public,**hist}.items()}
    summary={
        "schema_version":"dsc2026.endgame.stage01b_private_distribution.v2",
        "status":"COMPLETE_DIAGNOSTIC_ONLY",
        "private_labels_used":False,
        "evaluation_protocol_claim":"NONE; historical CAL600 is diagnostic only",
        "preflight_sha256":sha256(PREFLIGHT),
        "lexical_contract":"char_wb TF-IDF 3-5gram; transductive vocabulary; labels unused",
        "embedding_contract":{"model":source,"prefix":"query: ","pool":"mean_attention_mask",
                              "normalize":"L2","max_length":512},
        "population":{"PRIVATE":len(private),"TRAIN":len(train),"PUBLIC":len(public),
                      **{k:len(v) for k,v in hist.items()}},
        "exact_text_overlap":exact,
        "nearest_similarity":{name:{"lexical":desc(lex[name][0]),"embedding":desc(emb[name][0])}
                              for name in lex},
        "cluster_summary":{"selected_k":clusters["selected_k"],"silhouette":clusters["silhouette"],
                           "sizes":{str(x["cluster"]):x["size"] for x in clusters["clusters"]}},
        "notes":[
            "TRAIN is the primary reference population.",
            "PUBLIC is a secondary released-distribution reference.",
            "HIST_CAL_* are historical coordinates only and must not define ENDGAME promotion gates.",
        ],
    }
    (OUT/"PRIVATE_DISTRIBUTION_REPORT.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    fields=["private_qid","question","cluster_id"]
    for name in refs:
        x=name.lower()
        fields += [f"{x}_lex_qid",f"{x}_lex_sim",f"{x}_lex_question",
                   f"{x}_emb_qid",f"{x}_emb_sim",f"{x}_emb_question"]
    with (OUT/"PRIVATE_NEAREST_NEIGHBORS.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for n,(qid,q) in enumerate(private.items()):
            row={"private_qid":qid,"question":q,"cluster_id":int(labels[n])}
            for name in refs:
                x=name.lower();ls,li,lq=lex[name];es,ei,eq=emb[name]
                l=lq[int(li[n])];e=eq[int(ei[n])]
                row.update({f"{x}_lex_qid":l,f"{x}_lex_sim":f"{ls[n]:.8f}",
                            f"{x}_lex_question":refs[name][0][l],
                            f"{x}_emb_qid":e,f"{x}_emb_sim":f"{es[n]:.8f}",
                            f"{x}_emb_question":erefs[name][0][e]})
            w.writerow(row)

    md=["# Stage 01B — Private Distribution Diagnostic","","**Diagnostic only — not an evaluation protocol.**","",
        f"- PRIVATE: {len(private)}","- Primary comparison: full TRAIN","- Secondary: PUBLIC",
        "- Historical CAL A/B/C/D retained only as forensic coordinates.","",
        "## Nearest-neighbour summary","","| Reference | Lex median | Emb median | Emb >= .95 |",
        "|---|---:|---:|---:|"]
    for name in lex:
        md.append(f"| {name} | {summary['nearest_similarity'][name]['lexical']['median']:.4f} | "
                  f"{summary['nearest_similarity'][name]['embedding']['median']:.4f} | "
                  f"{summary['nearest_similarity'][name]['embedding']['ge_0.95']} |")
    md += ["","## Exact strict text overlap",""]
    for name,n in exact.items(): md.append(f"- {name}: {n}")
    md += ["","## Contract","","- Private labels used: false.","- No query identities were merged or dropped.",
           "- Aggressive normalization is used only in the preceding preflight duplicate audit.",""]
    (OUT/"REPORT.md").write_text("\n".join(md),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE_DIAGNOSTIC_ONLY","out":str(OUT),"model":source},indent=2))


if __name__=="__main__":
    main()

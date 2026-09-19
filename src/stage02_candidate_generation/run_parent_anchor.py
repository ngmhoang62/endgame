#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.evaluation import official_metrics

RAW = ROOT/"data/official_v1"
EVAL = ROOT/"data/evaluation_v2"
MODEL_DIR = ROOT/"models/vietlegal-e5"
MODEL_MANIFEST = ROOT/"reports/stage00_model_materialization/vietlegal-e5/MODEL_MANIFEST.json"
CACHE = ROOT/"cache/stage02a_parent_anchor"
OUT = ROOT/"reports/stage02a_parent_anchor"
KS=(1,5,10,20,50,100)
TOPK=100
WORD_RE=re.compile(r"\w+", re.UNICODE)
NORM_TOL=2e-6

os.environ["HF_HOME"]=str(ROOT/"cache/huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"]=str(ROOT/"cache/huggingface/hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"]=str(ROOT/"cache/sentence_transformers")

def rj(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def norm32(x):
    x=np.asarray(x,np.float32); n=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(n).all() or np.any(n<=0): raise RuntimeError("bad embedding norm")
    x=x/n
    if float(np.max(np.abs(np.linalg.norm(x,axis=1)-1)))>NORM_TOL:
        raise RuntimeError("float32 normalization failed")
    return x

def desc(v):
    x=np.asarray(list(v),float)
    return {k:float(val) for k,val in {
        "min":x.min(),"p25":np.quantile(x,.25),"median":np.median(x),
        "mean":x.mean(),"p75":np.quantile(x,.75),"p90":np.quantile(x,.9),
        "p95":np.quantile(x,.95),"p99":np.quantile(x,.99),"max":x.max()
    }.items()}

def load_all():
    corpus=[]
    cp=EVAL/"retrieval_corpus_8512.jsonl"
    with cp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); corpus.append({
                    "id":str(x["id"]),"passage":str(x["passage"]),
                    "name":str(x.get("name") or "")
                })
    if len(corpus)!=8512 or any(not x["passage"].strip() for x in corpus):
        raise RuntimeError("retrieval corpus contract failed")
    train=rj(RAW/"train.json")
    golds={str(q):[str(d) for d in ds] for q,ds in rj(EVAL/"primary_golds_6991.json").items()}
    folds={k:[str(q) for q in v] for k,v in rj(EVAL/"folds_v2.json").items()}
    stress={k:[str(q) for q in v] for k,v in rj(EVAL/"stress_slices.json").items()}
    qids=[str(q) for q in train if str(q) in golds]
    qs={q:str(train[q]["question"]).strip() for q in qids}
    if len(qids)!=6991: raise RuntimeError("expected 6991 qids")
    return cp, corpus, qids, qs, golds, folds, stress

def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    m=rj(MODEL_MANIFEST)
    if m.get("status")!="PASS" or m.get("historical_model_reused") is not False:
        raise RuntimeError("model manifest invalid")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    model=SentenceTransformer(str(MODEL_DIR),device="cuda",local_files_only=True,
        trust_remote_code=False,model_kwargs={"torch_dtype":torch.float16},
        processor_kwargs={"fix_mistral_regex":True})
    if int(model.max_seq_length)!=512: raise RuntimeError("max_seq_length drift")
    if int(model.get_embedding_dimension())!=int(m["model_contract"]["embedding_dimension"]):
        raise RuntimeError("embedding dimension drift")
    return model,m

def token_audit(model, corpus, golds):
    tok=model.tokenizer
    if not hasattr(tok,"backend_tokenizer"): raise RuntimeError("fast tokenizer required")
    lengths={}
    for i,x in enumerate(corpus,1):
        lengths[x["id"]]=len(tok.backend_tokenizer.encode("passage: "+x["passage"],add_special_tokens=True).ids)
        if i%1500==0 or i==len(corpus): print(f"[token] {i}/{len(corpus)}",flush=True)
    ug={d for ds in golds.values() for d in ds}
    if not ug<=set(lengths): raise RuntimeError("primary gold absent from corpus")
    occ=[lengths[d] for ds in golds.values() for d in ds]
    uniq=[lengths[d] for d in ug]
    over={d for d,n in lengths.items() if n>512}
    any_over=sum(any(d in over for d in ds) for ds in golds.values())
    all_over=sum(all(d in over for d in ds) for ds in golds.values())
    out={
      "representation":"passage: <raw passage>","max_seq_length":512,
      "all_documents":{"count":len(lengths),"tokens":desc(lengths.values()),
        "over_512_count":sum(n>512 for n in lengths.values()),
        "over_512_rate":sum(n>512 for n in lengths.values())/len(lengths),
        "over_1024_count":sum(n>1024 for n in lengths.values())},
      "unique_primary_gold_documents":{"count":len(ug),"tokens":desc(uniq),
        "over_512_count":sum(n>512 for n in uniq),
        "over_512_rate":sum(n>512 for n in uniq)/len(uniq)},
      "primary_gold_occurrences":{"count":len(occ),"tokens":desc(occ),
        "over_512_count":sum(n>512 for n in occ),
        "over_512_rate":sum(n>512 for n in occ)/len(occ)},
      "queries":{"count":len(golds),"any_gold_over_512_count":any_over,
        "any_gold_over_512_rate":any_over/len(golds),
        "all_golds_over_512_count":all_over,
        "all_golds_over_512_rate":all_over/len(golds)}
    }
    return lengths,out

def embed_cached(name,texts,ids,model,manifest,source_sha,batch):
    CACHE.mkdir(parents=True,exist_ok=True)
    npy=CACHE/f"{name}.f32.npy"; meta=CACHE/f"{name}.json"
    contract={"name":name,"source_sha":source_sha,"ids":ids,
      "model_sha":manifest["resolved_revision_sha"],"prefix_contract":True,
      "normalize":"ST_then_explicit_float32_l2","max_length":512}
    ch=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if npy.exists() and meta.exists() and rj(meta).get("contract_hash")==ch:
        z=np.asarray(np.load(npy),np.float32)
        if len(z)==len(ids):
            print(f"[embed] cache hit {name}",flush=True); return z
    z=norm32(model.encode(texts,batch_size=batch,convert_to_numpy=True,
        normalize_embeddings=True,show_progress_bar=True))
    np.save(npy,z); meta.write_text(json.dumps({"contract_hash":ch,"shape":list(z.shape)},indent=2))
    return z

def dense_top100(qv,dv):
    import torch
    docs=torch.from_numpy(np.ascontiguousarray(dv)).cuda().float()
    idx=np.empty((len(qv),TOPK),np.int32); scr=np.empty((len(qv),TOPK),np.float32)
    with torch.inference_mode():
        for s in range(0,len(qv),256):
            q=torch.from_numpy(np.ascontiguousarray(qv[s:s+256])).cuda().float()
            v,j=torch.topk(q@docs.T,k=TOPK,dim=1)
            idx[s:s+len(q)]=j.cpu().numpy(); scr[s:s+len(q)]=v.cpu().numpy()
            print(f"[dense] {min(s+256,len(qv))}/{len(qv)}",flush=True)
    return idx,scr

def toks(s): return WORD_RE.findall(s.casefold())

def bm25_top100(corpus,qids,questions):
    from rank_bm25 import BM25Okapi
    CACHE.mkdir(parents=True,exist_ok=True)
    ip=CACHE/"bm25_idx.npy"; sp=CACHE/"bm25_scores.npy"; mp=CACHE/"bm25.json"
    ch=hashlib.sha256((sha(EVAL/"retrieval_corpus_8512.jsonl")+sha(RAW/"train.json")).encode()).hexdigest()
    if ip.exists() and sp.exists() and mp.exists() and rj(mp).get("contract_hash")==ch:
        a=np.load(ip); b=np.load(sp)
        if a.shape==(len(qids),TOPK): print("[bm25] cache hit",flush=True); return a,b
    print("[bm25] tokenize corpus",flush=True)
    bm=BM25Okapi([toks(x["passage"]) for x in corpus])
    idx=np.empty((len(qids),TOPK),np.int32); scr=np.empty((len(qids),TOPK),np.float32)
    for i,qid in enumerate(qids):
        s=np.asarray(bm.get_scores(toks(questions[qid])),np.float32)
        cand=np.argpartition(s,-TOPK)[-TOPK:]
        order=sorted(map(int,cand),key=lambda j:(-float(s[j]),j))
        idx[i]=order; scr[i]=s[idx[i]]
        if (i+1)%1000==0 or i+1==len(qids): print(f"[bm25] {i+1}/{len(qids)}",flush=True)
    np.save(ip,idx); np.save(sp,scr); mp.write_text(json.dumps({"contract_hash":ch},indent=2))
    return idx,scr

def to_rank(idx,docids):
    return {i:[docids[int(j)] for j in idx[i]] for i in range(len(idx))}

def rrf(a,b):
    out={}
    for i in a:
        s={}
        for r,d in enumerate(a[i],1): s[d]=s.get(d,0)+1/(60+r)
        for r,d in enumerate(b[i],1): s[d]=s.get(d,0)+1/(60+r)
        out[i]=sorted(s,key=lambda d:(-s[d],d))[:TOPK]
    return out

def curve(rank,qids,golds):
    out={}
    for k in KS:
        vals=[]; full=0
        for i,q in enumerate(qids):
            g=set(golds[q]); h=len(set(rank[i][:k])&g)
            vals.append(h/len(g)); full+=h==len(g)
        out[str(k)]={"macro_recall":float(np.mean(vals)),"full_gold_coverage_rate":full/len(qids)}
    return out

def eval5(rank,qids,golds,folds,stress):
    preds={q:rank[i][:5] for i,q in enumerate(qids)}
    primary=set(qids)
    return {
      "overall":official_metrics(preds,golds,qids),
      "per_fold":{f:official_metrics(preds,golds,ids) for f,ids in folds.items()},
      "stress":{n:official_metrics(preds,golds,[q for q in ids if q in primary])
                for n,ids in stress.items() if any(q in primary for q in ids)}
    }

def dupmap(groups):
    m={}
    for g in groups:
        g=list(map(str,g))
        for d in g:m[d]=g
    return m

def union_oracle(a,b,qids,golds,dm):
    out={}; q100={}
    for k in KS:
        vals=[]; valsx=[]; sz=[]; szx=[]
        for i,q in enumerate(qids):
            p=set(a[i][:k])|set(b[i][:k]); px=set(p)
            for d in list(p): px.update(dm.get(d,[d]))
            g=set(golds[q]); vals.append(len(p&g)/len(g)); valsx.append(len(px&g)/len(g))
            sz.append(len(p)); szx.append(len(px))
            if k==100:q100[q]=(vals[-1],valsx[-1],len(p),len(px))
        out[str(k)]={"macro_oracle_recall":float(np.mean(vals)),
          "macro_oracle_recall_duplicate_expanded":float(np.mean(valsx)),
          "mean_pool_size":float(np.mean(sz)),"mean_expanded_pool_size":float(np.mean(szx)),
          "full_gold_coverage_rate":float(np.mean(np.asarray(vals)==1)),
          "full_gold_coverage_rate_duplicate_expanded":float(np.mean(np.asarray(valsx)==1))}
    return out,q100

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--doc-batch-size",type=int,default=12)
    ap.add_argument("--query-batch-size",type=int,default=64); args=ap.parse_args()
    cp,corpus,qids,questions,golds,folds,stress=load_all()
    model,manifest=load_model(); OUT.mkdir(parents=True,exist_ok=True)
    lengths,ta=token_audit(model,corpus,golds)
    (OUT/"CORPUS_TOKEN_AUDIT.json").write_text(json.dumps(ta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    docids=[x["id"] for x in corpus]
    dv=embed_cached("passages",["passage: "+x["passage"] for x in corpus],docids,model,manifest,sha(cp),args.doc_batch_size)
    qv=embed_cached("queries",["query: "+questions[q] for q in qids],qids,model,manifest,sha(RAW/"train.json"),args.query_batch_size)
    di,ds=dense_top100(qv,dv); np.save(CACHE/"dense_idx.npy",di); np.save(CACHE/"dense_scores.npy",ds)
    bi,bs=bm25_top100(corpus,qids,questions)
    dr=to_rank(di,docids); br=to_rank(bi,docids); rr=rrf(dr,br)

    methods={}
    for name,r in [("dense_vietlegal_e5_parent",dr),("bm25_parent",br),("rrf60_dense_bm25",rr)]:
        methods[name]={"curve":curve(r,qids,golds),"at5":eval5(r,qids,golds,folds,stress)}

    groups=rj(EVAL/"exact_duplicate_passage_groups.json"); dm=dupmap(groups)
    union,q100=union_oracle(dr,br,qids,golds,dm)

    bm_rescues=dense_rescues=0; failures=[]
    foldfor={q:f for f,ids in folds.items() for q in ids}
    rows=[]
    for i,q in enumerate(qids):
        g=set(golds[q]); dpos={d:r+1 for r,d in enumerate(dr[i])}; bpos={d:r+1 for r,d in enumerate(br[i])}
        drec=len(set(dr[i])&g)/len(g); brec=len(set(br[i])&g)/len(g); u,ux,sz,szx=q100[q]
        bm_rescues+=ux>drec+1e-12; dense_rescues+=ux>brec+1e-12
        rows.append({"qid":q,"fold":foldfor[q],"gold_count":len(g),
          "dense_best_rank":min([dpos[d] for d in g if d in dpos],default=">100"),
          "bm25_best_rank":min([bpos[d] for d in g if d in bpos],default=">100"),
          "union_recall_100":u,"union_expanded_recall_100":ux,
          "all_golds_over_512":int(all(lengths[d]>512 for d in g)),"question":questions[q]})
        if ux<1:
            pool=set(dr[i])|set(br[i]); px=set(pool)
            for d in list(pool):px.update(dm.get(d,[d]))
            failures.append({"qid":q,"fold":foldfor[q],"question":questions[q],
              "missing_gold":[{"doc_id":d,"token_length":lengths[d],
                "dense_rank":dpos.get(d),"bm25_rank":bpos.get(d)} for d in g if d not in px]})

    with (OUT/"QUERY_RETRIEVAL_AUDIT.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"MISSING_GOLD_FORENSICS.json").write_text(json.dumps({"count":len(failures),"queries":failures},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    report={"schema_version":"dsc2026.endgame.stage02a_parent_anchor.v1","status":"COMPLETE",
      "claim_boundary":"raw-parent acquisition anchor; no learned tuning",
      "population":{"queries":len(qids),"documents":len(corpus)},
      "model":{"id":manifest["model_id"],"sha":manifest["resolved_revision_sha"],
        "query_representation":"query: <question>","document_representation":"passage: <raw passage>","max_seq_length":512},
      "methods":methods,"candidate_union_oracle":union,
      "complementarity_top100":{"bm25_improves_dense":int(bm_rescues),"dense_improves_bm25":int(dense_rescues),
        "uncovered_after_union_duplicate_expansion":len(failures)},
      "duplicate_policy":"candidate expansion only; literal IDs preserved",
      "token_audit":ta}
    (OUT/"BASELINE_ANCHOR.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    d5=methods["dense_vietlegal_e5_parent"]["at5"]["overall"]; b5=methods["bm25_parent"]["at5"]["overall"]; r5=methods["rrf60_dense_bm25"]["at5"]["overall"]
    md=["# Stage 02A — Raw-Parent Acquisition Anchor","","**Status: COMPLETE**","",
      "## Truncation","",
      f"- Docs >512: **{ta['all_documents']['over_512_count']} ({ta['all_documents']['over_512_rate']:.2%})**",
      f"- Gold occurrences >512: **{ta['primary_gold_occurrences']['over_512_count']} ({ta['primary_gold_occurrences']['over_512_rate']:.2%})**",
      f"- Queries with all gold >512: **{ta['queries']['all_golds_over_512_count']} ({ta['queries']['all_golds_over_512_rate']:.2%})**","",
      "## Top-5","","| Method | Recall@5 | Precision@5 |","|---|---:|---:|",
      f"| Dense E5 parent | {d5['recall_at_5']:.6f} | {d5['precision_at_5']:.6f} |",
      f"| BM25 parent | {b5['recall_at_5']:.6f} | {b5['precision_at_5']:.6f} |",
      f"| RRF60 | {r5['recall_at_5']:.6f} | {r5['precision_at_5']:.6f} |","",
      "## Candidate acquisition","","| K | Dense | BM25 | Union oracle | Union+dup expansion |","|---:|---:|---:|---:|---:|"]
    for k in KS:
        kk=str(k); md.append(f"| {k} | {methods['dense_vietlegal_e5_parent']['curve'][kk]['macro_recall']:.6f} | {methods['bm25_parent']['curve'][kk]['macro_recall']:.6f} | {union[kk]['macro_oracle_recall']:.6f} | {union[kk]['macro_oracle_recall_duplicate_expanded']:.6f} |")
    md += ["",f"- BM25 improves dense top100 pool on **{bm_rescues}** queries.",
      f"- Dense improves BM25 top100 pool on **{dense_rescues}** queries.",
      f"- Union+duplicate expansion still fails full coverage on **{len(failures)}** queries.",""]
    (OUT/"REPORT.md").write_text("\n".join(md),encoding="utf-8")

    print(json.dumps({"status":"COMPLETE","dense_recall_at5":d5["recall_at_5"],
      "bm25_recall_at5":b5["recall_at_5"],"rrf_recall_at5":r5["recall_at_5"],
      "union_oracle100":union["100"]["macro_oracle_recall"],
      "union_oracle100_dup_expanded":union["100"]["macro_oracle_recall_duplicate_expanded"],
      "docs_over512_rate":ta["all_documents"]["over_512_rate"],
      "gold_occ_over512_rate":ta["primary_gold_occurrences"]["over_512_rate"],
      "uncovered_queries":len(failures),"out":str(OUT)},ensure_ascii=False,indent=2))

if __name__=="__main__": main()

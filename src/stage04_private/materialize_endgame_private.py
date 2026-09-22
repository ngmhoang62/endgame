#!/usr/bin/env python
from __future__ import annotations

import argparse, gc, hashlib, json, math, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/official_v1'; EVAL=ROOT/'data/evaluation_v2'
PRIVATE=RAW/'private-official.json'; CORPUS=EVAL/'retrieval_corpus_8512.jsonl'
MODERATE=ROOT/'cache/stage02b_structure/moderate_structure_regions.jsonl'
C1=ROOT/'cache/stage02c1_aiteam_representation'; C2=ROOT/'cache/stage02c2_lal_representation'
B4=ROOT/'cache/stage02b4_vi_screen'; B0=ROOT/'cache/stage03b_reranker'; B1=ROOT/'cache/stage03b1_aiteam_reranker'
A1_MODEL=ROOT/'models/retrievers_vi/aiteamvn-vietnamese-embedding'
A1_MAN=ROOT/'reports/stage02b3_vietnamese_model_materialization/aiteamvn_vietnamese_embedding/MODEL_MANIFEST.json'
LAL_MODEL=ROOT/'models/retrievers/vnlegal-lal'; LAL_MAN=ROOT/'reports/stage02b1_model_materialization/vnlegal_lal/MODEL_MANIFEST.json'
RR_MODEL=ROOT/'models/rerankers/aiteamvn-vietnamese-reranker'; RR_MAN=ROOT/'reports/stage03b0_reranker_materialization/MODEL_MANIFEST.json'
RANK_MODEL=B0/'fulltrain_rank_selector.joblib'; CE_MODEL=B1/'fulltrain_ce_fusion.joblib'
CACHE=ROOT/'cache/stage04_private_submission_v1_2'; OUT=ROOT/'reports/stage04_private_submission'; SUB=ROOT/'submissions/endgame_20260922'
EXPECTED_SHA='9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc'
A1_SHA='dea33aa1ab339f38d66ae0a40e6c40e0a9249568'; LAL_SHA='de759324ef931a2475ae8db97137b6a6cbb98aa0'; RR_SHA='f536976248403314225d7fdfdbc87f0e9516a54e'
NQ=2080; TOPK=100; SRC_DEPTH=50; SHORT=30
LEGAL_Q_PREFIX='Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: '
SOURCE_NAMES=['ait_atomic','ait_coarse1024','lal_coarse1024','lal_atomic','lal_b4','bm25']
DENSE={
 'ait_atomic':(C1/'atomic_split_2048/embeddings.f32.npy',C1/'atomic_split_2048/embeddings.json',C1/'atomic_split_2048/chunks.jsonl','aiteam'),
 'ait_coarse1024':(C1/'coarse_pack_1024/embeddings.f32.npy',C1/'coarse_pack_1024/embeddings.json',C1/'coarse_pack_1024/chunks.jsonl','aiteam'),
 'lal_coarse1024':(C2/'lal_coarse_pack_1024/embeddings.f32.npy',C2/'lal_coarse_pack_1024/embeddings.json',C2/'lal_coarse_pack_1024/chunks.jsonl','lal'),
 'lal_atomic':(C2/'lal_atomic_split_2048/embeddings.f32.npy',C2/'lal_atomic_split_2048/embeddings.json',C2/'lal_atomic_split_2048/chunks.jsonl','lal'),
 'lal_b4':(B4/'vnlegal_lal/region_embeddings.f32.npy',B4/'vnlegal_lal/region_embeddings.json',MODERATE,'lal'),
}
WORD_RE=re.compile(r'\w+',re.UNICODE); SPACE_RE=re.compile(r'\S+',re.UNICODE)
STOP={'bị','các','có','của','cho','được','để','đến','đối','gì','hay','khi','không','là','làm','một','nào','những','như','phải','ra','sẽ','theo','thì','thế','trong','trên','từ','và','về','với','việc','bao','nhiêu','người','quy','định'}

def rj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def shobj(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def clear_cuda():
 import torch; gc.collect(); torch.cuda.empty_cache()
def norm(x):
 x=np.asarray(x,np.float32); n=np.linalg.norm(x,axis=1,keepdims=True)
 if not np.isfinite(x).all() or np.any(n<=0): raise RuntimeError('bad embedding')
 return x/n

def load_private():
 if sha(PRIVATE)!=EXPECTED_SHA: raise RuntimeError('private snapshot SHA drift')
 raw=rj(PRIVATE); qids=[]; qs={}
 if not isinstance(raw,dict): raise RuntimeError('private top-level must be dict')
 for k,v in raw.items():
  if isinstance(v,str): q=str(k); text=v
  elif isinstance(v,dict): q=str(v.get('qid',k)); text=str(v['question'])
  else: raise RuntimeError('invalid private record')
  if q in qs: raise RuntimeError('duplicate private qid')
  qids.append(q); qs[q]=text.strip()
 if len(qids)!=NQ: raise RuntimeError(f'expected {NQ}, got {len(qids)}')
 return qids,qs

def load_corpus():
 docs=[]; texts=[]
 with CORPUS.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():
    x=json.loads(line); docs.append(str(x['id'])); texts.append(str(x['passage']))
 if len(docs)!=8512 or len(set(docs))!=8512: raise RuntimeError('corpus drift')
 return docs,texts

def preflight():
 a,l,r=rj(A1_MAN),rj(LAL_MAN),rj(RR_MAN)
 if a['resolved_revision_sha']!=A1_SHA or l['resolved_revision_sha']!=LAL_SHA or r['resolved_revision_sha']!=RR_SHA: raise RuntimeError('model SHA drift')
 for p in [A1_MODEL,LAL_MODEL,RR_MODEL,RANK_MODEL,CE_MODEL]:
  if not p.exists(): raise FileNotFoundError(p)
 return a,l,r

def qcache(fam):
 d=CACHE/'query'; d.mkdir(parents=True,exist_ok=True); return d/f'{fam}.npy',d/f'{fam}.json'

def encode_aiteam(qids,qs):
 import torch
 from sentence_transformers import SentenceTransformer
 npy,mp=qcache('aiteam'); contract={'sha':EXPECTED_SHA,'qids':qids,'model':A1_SHA,'max':2048}; ch=shobj(contract)
 if npy.exists() and mp.exists():
  if rj(mp)['contract_hash']!=ch: raise RuntimeError('AIT query cache mismatch')
  z=np.load(npy); print('[AIT] query cache hit'); return norm(z)
 if npy.exists() or mp.exists(): raise RuntimeError('AIT partial cache')
 m=SentenceTransformer(str(A1_MODEL),device='cuda',local_files_only=True,model_kwargs={'dtype':torch.float16}); m.max_seq_length=2048
 try: z=norm(m.encode([qs[q] for q in qids],batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True))
 finally: del m; clear_cuda()
 np.save(npy,z); mp.write_text(json.dumps({'contract_hash':ch,'contract':contract,'status':'PASS'},indent=2)+'\n')
 return z

def last_token(h,mask):
 import torch
 if bool(torch.all(mask[:,-1]==1).item()): return h[:,-1]
 lens=mask.sum(1)-1; return h[torch.arange(h.shape[0],device=h.device),lens]

def encode_lal(qids,qs):
 import torch, torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModel
 npy,mp=qcache('lal'); contract={'sha':EXPECTED_SHA,'qids':qids,'model':LAL_SHA,'max':2048,'prefix':LEGAL_Q_PREFIX}; ch=shobj(contract)
 if npy.exists() and mp.exists():
  if rj(mp)['contract_hash']!=ch: raise RuntimeError('LAL query cache mismatch')
  z=np.load(npy); print('[LAL] query cache hit'); return norm(z)
 if npy.exists() or mp.exists(): raise RuntimeError('LAL partial cache')
 tok=AutoTokenizer.from_pretrained(str(LAL_MODEL),local_files_only=True,use_fast=True,fix_mistral_regex=True)
 m=AutoModel.from_pretrained(str(LAL_MODEL),local_files_only=True,dtype=torch.float16).cuda().eval(); texts=[LEGAL_Q_PREFIX+qs[q] for q in qids]
 out=[]; i=0; batch=64
 try:
  while i<len(texts):
   j=min(i+batch,len(texts))
   try:
    x=tok(texts[i:j],padding=True,truncation=True,max_length=2048,return_tensors='pt').to('cuda')
    with torch.inference_mode(): e=F.normalize(last_token(m(**x).last_hidden_state,x['attention_mask']).float(),p=2,dim=1)
    out.append(e.cpu().numpy()); i=j
    if i%512<batch or i==len(texts): print(f'[LAL] query {i}/{len(texts)} batch={batch}')
   except torch.cuda.OutOfMemoryError:
    clear_cuda(); batch//=2
    if batch<1: raise
 finally: del m,tok; clear_cuda()
 z=norm(np.concatenate(out)); np.save(npy,z); mp.write_text(json.dumps({'contract_hash':ch,'contract':contract,'status':'PASS'},indent=2)+'\n'); return z

def parent_idx(path,docs):
 d2i={d:i for i,d in enumerate(docs)}; out=[]
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if line.strip(): out.append(d2i[str(json.loads(line)['doc_id'])])
 return np.asarray(out,np.int64)

def rpaths(name):
 d=CACHE/'retrieval'; d.mkdir(parents=True,exist_ok=True); return d/f'{name}_idx.npy',d/f'{name}_scores.npy',d/f'{name}.json'

def dense_search(name,q,docs):
 import torch
 ep,emp,cp,_=DENSE[name]
 for p in [ep,emp,cp]:
  if not p.exists(): raise FileNotFoundError(p)
 em=rj(emp); emb=np.load(ep,mmap_mode='r'); parent=parent_idx(cp,docs)
 if emb.shape[0]!=len(parent) or emb.shape[1]!=1024: raise RuntimeError(f'{name} shape mismatch')
 contract={'src':name,'emb_contract':em.get('contract_hash'),'private_q_sha':hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),'topk':100}; ch=shobj(contract)
 ip,sp,mp=rpaths(name)
 if ip.exists() and sp.exists() and mp.exists():
  if rj(mp)['contract_hash']!=ch: raise RuntimeError(f'{name} cache mismatch')
  print(f'[{name}] retrieval cache hit'); return np.load(ip),np.load(sp)
 if ip.exists() or sp.exists() or mp.exists(): raise RuntimeError(f'{name} partial retrieval cache')
 print(f'[{name}] search vectors={len(parent)}')
 rvnp=np.array(emb,np.float32,copy=True,order='C'); rv=torch.from_numpy(rvnp).cuda(); pi=torch.from_numpy(parent).cuda().long()
 idx=np.empty((len(q),100),np.int32); scr=np.empty((len(q),100),np.float32)
 with torch.inference_mode():
  for s in range(0,len(q),32):
   qq=torch.from_numpy(np.array(q[s:s+32],np.float32,copy=True,order='C')).cuda(); sim=qq@rv.T
   ps=torch.full((len(qq),len(docs)),-torch.inf,device='cuda'); ps.scatter_reduce_(1,pi.unsqueeze(0).expand(len(qq),-1),sim,reduce='amax',include_self=True)
   v,j=torch.topk(ps,100,dim=1); idx[s:s+len(qq)]=j.cpu().numpy(); scr[s:s+len(qq)]=v.cpu().numpy()
   if s%512==0: print(f'[{name}] {min(s+32,len(q))}/{len(q)}')
 np.save(ip,idx); np.save(sp,scr); mp.write_text(json.dumps({'contract_hash':ch,'contract':contract,'status':'PASS'},indent=2)+'\n')
 del rv,rvnp,pi; clear_cuda(); return idx,scr

def toks(s): return WORD_RE.findall((s or '').casefold())
def bm25(qids,qs,texts):
 from rank_bm25 import BM25Okapi
 ip,sp,mp=rpaths('bm25'); contract={'private':EXPECTED_SHA,'corpus':sha(CORPUS),'topk':100}; ch=shobj(contract)
 if ip.exists() and sp.exists() and mp.exists():
  if rj(mp)['contract_hash']!=ch: raise RuntimeError('BM25 cache mismatch')
  print('[BM25] cache hit'); return np.load(ip),np.load(sp)
 if ip.exists() or sp.exists() or mp.exists(): raise RuntimeError('BM25 partial cache')
 print('[BM25] tokenize corpus'); m=BM25Okapi([toks(x) for x in texts]); idx=np.empty((len(qids),100),np.int32); scr=np.empty((len(qids),100),np.float32)
 for i,q in enumerate(qids):
  s=np.asarray(m.get_scores(toks(qs[q])),np.float32); c=np.argpartition(s,-100)[-100:]; order=sorted(map(int,c),key=lambda j:(-float(s[j]),j)); idx[i]=order; scr[i]=s[idx[i]]
  if (i+1)%500==0: print(f'[BM25] {i+1}/{len(qids)}')
 np.save(ip,idx); np.save(sp,scr); mp.write_text(json.dumps({'contract_hash':ch,'contract':contract,'status':'PASS'},indent=2)+'\n'); return idx,scr

def rank_names():
 # EXACT Stage03B0 full-train feature-name contract.
 # Values are unchanged; these names must match the joblib metadata byte-for-byte.
 n=[]
 for s in SOURCE_NAMES: n += [s+'__present',s+'__rr10',s+'__rank50',s+'__z',s+'__gapz']
 return n+['source_count','best_rr10','mean_rr10','rrf60','min_rank50','mean_rank50','count_top5','count_top10','count_top20']

def shortlist(sources):
 import joblib
 obj=joblib.load(RANK_MODEL); names=rank_names()
 if obj['sources']!=SOURCE_NAMES or obj['feature_names']!=names or obj['depth']!=50:
  raise RuntimeError(
   'rank selector contract drift\n'
   f"sources actual={obj.get('sources')} expected={SOURCE_NAMES}\n"
   f"features actual={obj.get('feature_names')} expected={names}\n"
   f"depth actual={obj.get('depth')} expected=50"
  )
 model=obj['model']; nq=next(iter(sources.values()))[0].shape[0]; out=np.empty((nq,30),np.int32)
 for qi in range(nq):
  pool=np.asarray(sorted(set().union(*[set(map(int,sources[s][0][qi,:50])) for s in SOURCE_NAMES])),np.int32); rm={}; zm={}; gm={}
  for s in SOURCE_NAMES:
   idx,sc=sources[s]; ids=idx[qi,:50]; a=np.asarray(sc[qi],float); mu=float(a.mean()); sd=float(a.std()); sd=1. if (not np.isfinite(sd) or sd<1e-8) else sd; top=float(a[0]); rm[s]={int(d):r+1 for r,d in enumerate(ids)}; zm[s]={int(d):(float(sc[qi,r])-mu)/sd for r,d in enumerate(ids)}; gm[s]={int(d):(top-float(sc[qi,r]))/sd for r,d in enumerate(ids)}
  X=[]
  for d0 in pool:
   d=int(d0); f=[]; rrs=[]; rs=[]; c5=c10=c20=0; rrf=0.
   for s in SOURCE_NAMES:
    r=rm[s].get(d)
    if r is None: f += [0.,0.,1.2,-3.,4.]
    else:
     rr=1/(10+r); f += [1.,rr,r/50.,zm[s][d],gm[s][d]]; rrs.append(rr); rs.append(r); rrf+=1/(60+r); c5+=r<=5; c10+=r<=10; c20+=r<=20
   f += [len(rrs),max(rrs) if rrs else 0.,float(np.mean(rrs)) if rrs else 0.,rrf,min(rs)/50 if rs else 1.2,float(np.mean(rs))/50 if rs else 1.2,c5,c10,c20]; X.append(f)
  p=model.predict_proba(np.asarray(X,np.float32))[:,1]; order=np.lexsort((pool,-p)); out[qi]=pool[order[:30]]
  if (qi+1)%500==0: print(f'[selector] {qi+1}/{nq}')
 CACHE.mkdir(parents=True,exist_ok=True); np.save(CACHE/'private_shortlist_top30_idx.npy',out); return out

def rr_tokens(s): return WORD_RE.findall((s or '').lower())
def top_passages(q,text,count=2):
 words=SPACE_RE.findall(text or '')
 if len(words)<=300: return [' '.join(words)]
 qt=rr_tokens(q); content={t for t in qt if len(t)>=3 and t not in STOP}; nums={t for t in qt if any(c.isdigit() for c in t)}; bigrams={' '.join(qt[i:i+2]) for i in range(len(qt)-1)}; header=' '.join(words[:70]); scored=[]
 for start in range(0,len(words),150):
  end=min(start+220,len(words)); part=' '.join(words[start:end]); n=rr_tokens(part); S=set(n); nt=' '.join(n); cov=sum(1+.2*min(n.count(t),3) for t in content if t in S); num=3*sum(t in S for t in nums); ph=1.8*sum(p in nt for p in bigrams); scored.append(((cov+num+ph)/math.sqrt(max(len(n),1)),cov+num+ph,-start,part))
  if end==len(words): break
 scored.sort(reverse=True); out=[]
 for _,_,neg,p in scored:
  x=p if -neg<70 else header+'\n[ĐOẠN PHÙ HỢP]\n'+p
  if x not in out: out.append(x)
  if len(out)>=count: break
 return out

def rr_paths():
 d=CACHE/'reranker'; d.mkdir(parents=True,exist_ok=True); return d/'scores.npy',d/'done.npy',d/'meta.json'
def load_rr():
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 tok=AutoTokenizer.from_pretrained(str(RR_MODEL),local_files_only=True,trust_remote_code=True); m=AutoModelForSequenceClassification.from_pretrained(str(RR_MODEL),local_files_only=True,trust_remote_code=True,dtype=torch.float16).eval().cuda(); return tok,m
def rr_forward(tok,m,pairs,batch):
 import torch
 vals=[]; i=0
 while i<len(pairs):
  j=min(i+batch,len(pairs))
  try:
   x=tok([a for a,b in pairs[i:j]],[b for a,b in pairs[i:j]],max_length=512,truncation=True,padding=True,return_tensors='pt'); x={k:v.cuda() for k,v in x.items()}
   with torch.inference_mode(): z=m(**x).logits
   v=z[:,0] if z.ndim==2 and z.shape[1]==1 else (z[:,-1] if z.ndim==2 else z.reshape(-1)); vals.extend(map(float,v.float().cpu().tolist())); i=j
  except torch.cuda.OutOfMemoryError:
   clear_cuda(); batch//=2
   if batch<1: raise
 return vals,batch
def prep_chunk(qis,qids,qs,sl,texts):
 pairs=[]; owners=[]
 for qi in qis:
  q=qs[qids[qi]]
  for p,d in enumerate(sl[qi]):
   for passage in top_passages(q,texts[int(d)],2): owners.append((qi,p)); pairs.append((q,passage))
 return qis,pairs,owners

def rerank(qids,qs,sl,texts):
 sp,dp,mp=rr_paths(); slsha=hashlib.sha256(np.ascontiguousarray(sl).tobytes()).hexdigest(); contract={'private':EXPECTED_SHA,'shortlist':slsha,'model':RR_SHA,'depth':30,'max':512,'passages':2,'window':220,'overlap':70}; ch=shobj(contract)
 if sp.exists() and dp.exists() and mp.exists():
  meta=rj(mp)
  if meta['contract_hash']!=ch: raise RuntimeError('reranker cache mismatch')
  scores=np.lib.format.open_memmap(sp,mode='r+'); done=np.lib.format.open_memmap(dp,mode='r+')
 else:
  if sp.exists() or dp.exists() or mp.exists(): raise RuntimeError('reranker partial cache')
  scores=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(len(qids),30)); scores[:]=np.nan; scores.flush(); done=np.lib.format.open_memmap(dp,mode='w+',dtype=np.uint8,shape=(len(qids),)); done[:]=0; done.flush(); meta={'contract_hash':ch,'contract':contract,'current_batch':64}; mp.write_text(json.dumps(meta,indent=2)+'\n')
 pending=[i for i in range(len(qids)) if not int(done[i])]
 if not pending: print('[reranker] cache complete'); return np.asarray(scores,np.float32)
 tok,m=load_rr(); batch=int(meta.get('current_batch',64)); chunks=[pending[i:i+32] for i in range(0,len(pending),32)]; ex=ThreadPoolExecutor(max_workers=1); fut=ex.submit(prep_chunk,chunks[0],qids,qs,sl,texts); start=time.perf_counter(); new=0
 try:
  for ci in range(len(chunks)):
   w=time.perf_counter(); qis,pairs,owners=fut.result(); wait=time.perf_counter()-w
   if ci+1<len(chunks): fut=ex.submit(prep_chunk,chunks[ci+1],qids,qs,sl,texts)
   g=time.perf_counter(); vals,batch=rr_forward(tok,m,pairs,batch); gpu=time.perf_counter()-g; tmp={qi:np.full(30,-np.inf,np.float32) for qi in qis}
   for (qi,p),v in zip(owners,vals): tmp[qi][p]=max(tmp[qi][p],v)
   for qi in qis: scores[qi]=tmp[qi]; done[qi]=1; new+=1
   scores.flush(); done.flush(); completed=int(np.asarray(done).sum()); meta={'contract_hash':ch,'contract':contract,'current_batch':batch,'completed':completed,'status':'PASS' if completed==len(qids) else 'IN_PROGRESS'}; mp.write_text(json.dumps(meta,indent=2)+'\n'); print(f'[reranker] {completed}/{len(qids)} batch={batch} rate={new/max(time.perf_counter()-start,1e-9):.2f} q/s wait={wait:.2f}s gpu={gpu:.2f}s pairs={len(pairs)}')
 finally: ex.shutdown(wait=False,cancel_futures=True); del m,tok; clear_cuda()
 return np.asarray(scores,np.float32)

def zrows(a):
 a=np.asarray(a,float); mu=a.mean(1,keepdims=True); sd=a.std(1,keepdims=True); sd=np.where(sd<1e-8,1.,sd); return ((a-mu)/sd).astype(np.float32)
def ce_names():
 n=[]
 for s in SOURCE_NAMES: n += [s+'__present',s+'__rr10',s+'__rank50',s+'__z',s+'__gapz']
 return n+['source_count','best_rr10','mean_rr10','rrf60','min_rank50','mean_rank50','count_top5','count_top10','count_top20','selector_rr10','selector_rank_norm30','ce_raw','ce_z','ce_gap_z','ce_rr10']

def ce_lr(sl,ce,sources):
 import joblib
 obj=joblib.load(CE_MODEL); names=ce_names()
 if obj['source_names']!=SOURCE_NAMES or obj['feature_names']!=names or obj['shortlist_depth']!=30: raise RuntimeError('CE model contract drift')
 model=obj['model']; cez=zrows(ce); rank=np.empty_like(sl)
 for qi in range(len(sl)):
  rm={}; zm={}; gm={}
  for s in SOURCE_NAMES:
   idx,sc=sources[s]; ids=idx[qi,:50]; a=np.asarray(sc[qi],float); mu=float(a.mean()); sd=float(a.std()); sd=1. if (not np.isfinite(sd) or sd<1e-8) else sd; top=float(a[0]); rm[s]={int(d):r+1 for r,d in enumerate(ids)}; zm[s]={int(d):(float(sc[qi,r])-mu)/sd for r,d in enumerate(ids)}; gm[s]={int(d):(top-float(sc[qi,r]))/sd for r,d in enumerate(ids)}
  order=np.lexsort((sl[qi],-ce[qi])); inv=np.empty(30,np.int32); inv[order]=np.arange(1,31); cesd=float(np.std(ce[qi])); cesd=1. if (not np.isfinite(cesd) or cesd<1e-8) else cesd; cetop=float(np.max(ce[qi])); X=[]
  for sp,d0 in enumerate(sl[qi]):
   d=int(d0); f=[]; rrs=[]; rs=[]; c5=c10=c20=0; rrf=0.
   for s in SOURCE_NAMES:
    r=rm[s].get(d)
    if r is None: f += [0.,0.,1.2,-3.,4.]
    else:
     rr=1/(10+r); f += [1.,rr,r/50.,zm[s][d],gm[s][d]]; rrs.append(rr); rs.append(r); rrf+=1/(60+r); c5+=r<=5; c10+=r<=10; c20+=r<=20
   cr=int(inv[sp]); f += [len(rrs),max(rrs) if rrs else 0.,float(np.mean(rrs)) if rrs else 0.,rrf,min(rs)/50 if rs else 1.2,float(np.mean(rs))/50 if rs else 1.2,c5,c10,c20,1/(11+sp),(sp+1)/30,float(ce[qi,sp]),float(cez[qi,sp]),float((cetop-ce[qi,sp])/cesd),1/(10+cr)]; X.append(f)
  p=model.predict_proba(np.asarray(X,np.float32))[:,1]; o=np.lexsort((sl[qi],-p)); rank[qi]=sl[qi,o]
 return rank

def blend(sl,ce):
 zr=zrows(np.broadcast_to(-np.arange(1,31,dtype=np.float32)[None,:],sl.shape)); zc=zrows(ce); s=.85*zr+.15*zc; out=np.empty_like(sl)
 for i in range(len(sl)): out[i]=sl[i,np.lexsort((sl[i],-s[i]))]
 return out

def package(label,rank,qids,docs):
 SUB.mkdir(parents=True,exist_ok=True); valid=set(docs); sub={}
 for i,q in enumerate(qids):
  ans=[docs[int(x)] for x in rank[i,:5]]
  if len(set(ans))!=5 or any(x not in valid for x in ans): raise RuntimeError(f'invalid {q}')
  sub[q]={'answer':ans}
 jp=SUB/f'{label}.json'; zp=SUB/f'{label}.zip'; jp.write_text(json.dumps(sub,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z: z.write(jp,arcname=jp.name)
 return {'json':str(jp.relative_to(ROOT)).replace('\\','/'),'json_sha256':sha(jp),'zip':str(zp.relative_to(ROOT)).replace('\\','/'),'zip_sha256':sha(zp)}
def churn(a,b):
 return {'ordered_churn':int(np.sum(np.any(a[:,:5]!=b[:,:5],axis=1))),'set_churn':int(sum(set(map(int,a[i,:5]))!=set(map(int,b[i,:5])) for i in range(len(a))))}

def main():
 argparse.ArgumentParser().parse_args(); start=time.perf_counter(); CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
 print('[1/8] preflight'); qids,qs=load_private(); docs,texts=load_corpus(); am,lm,rm=preflight()
 print('[2/8] AIT queries/retrieval'); aq=encode_aiteam(qids,qs); sources={}; sources['ait_atomic']=dense_search('ait_atomic',aq,docs); sources['ait_coarse1024']=dense_search('ait_coarse1024',aq,docs); del aq; clear_cuda()
 print('[3/8] LAL queries/retrieval'); lq=encode_lal(qids,qs); sources['lal_coarse1024']=dense_search('lal_coarse1024',lq,docs); sources['lal_atomic']=dense_search('lal_atomic',lq,docs); sources['lal_b4']=dense_search('lal_b4',lq,docs); del lq; clear_cuda()
 print('[4/8] BM25'); sources['bm25']=bm25(qids,qs,texts)
 print('[5/8] fulltrain rank selector'); sl=shortlist(sources)
 print('[6/8] reranker'); ce=rerank(qids,qs,sl,texts)
 print('[7/8] final fusion'); primary=ce_lr(sl,ce,sources); control_blend=blend(sl,ce); rankonly=sl.copy()
 print('[8/8] package'); arts={'ENDGAME_PRIVATE_CE_LR_K5':package('ENDGAME_PRIVATE_CE_LR_K5',primary,qids,docs),'ENDGAME_PRIVATE_BLEND015_K5':package('ENDGAME_PRIVATE_BLEND015_K5',control_blend,qids,docs),'ENDGAME_PRIVATE_RANK_ONLY_K5':package('ENDGAME_PRIVATE_RANK_ONLY_K5',rankonly,qids,docs)}
 total=int(am['parameter_count'])+int(lm['parameter_count'])+int(rm['parameter_count']); report={'schema':'stage04_private_submission.v1','status':'READY_FOR_PRIVATE_SUBMISSION','private_labels_used':False,'private_queries':len(qids),'private_sha256':EXPECTED_SHA,'primary':'ENDGAME_PRIVATE_CE_LR_K5','primary_oof_recall_at5':0.9425976255185238,'active_parameter_ledger':{'aiteam_embedding':int(am['parameter_count']),'vnlegal_lal':int(lm['parameter_count']),'aiteam_reranker':int(rm['parameter_count']),'total':total,'budget':4_000_000_000},'churn':{'rank_to_primary':churn(rankonly,primary),'rank_to_blend':churn(rankonly,control_blend),'blend_to_primary':churn(control_blend,primary)},'artifacts':arts,'wall_seconds':time.perf_counter()-start}
 (OUT/'PRIVATE_SUBMISSION_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); lines=['# Stage 04 — ENDGAME Private Submission','','**READY_FOR_PRIVATE_SUBMISSION**','',f"Primary: `{arts['ENDGAME_PRIVATE_CE_LR_K5']['zip']}`",f"SHA256: `{arts['ENDGAME_PRIVATE_CE_LR_K5']['zip_sha256']}`",f'Active params: **{total/1e9:.3f}B / 4B**','','Submit CE-LR first; blend015 and rank-only are controls.']; (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('='*90); print('READY_FOR_PRIVATE_SUBMISSION'); print('PRIMARY',arts['ENDGAME_PRIVATE_CE_LR_K5']['zip']); print('SHA256',arts['ENDGAME_PRIVATE_CE_LR_K5']['zip_sha256']); print('='*90)
if __name__=='__main__': main()

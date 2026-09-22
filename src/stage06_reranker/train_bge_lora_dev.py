#!/usr/bin/env python
from __future__ import annotations
import argparse,gc,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.common.evaluation import official_metrics
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage06_evidence.benchmark_evidence_packaging as a0
import src.stage06_reranker.screen_bge_reranker_dev as b0

MODEL=ROOT/'models/rerankers/bge-reranker-v2-m3'
MAN=ROOT/'reports/stage06b0_bge_reranker_materialization/MODEL_MANIFEST.json'
CACHE=ROOT/'cache/stage06b1_bge_lora_dev_v1_2'; OUT=ROOT/'reports/stage06b1_bge_lora_dev_v1_2'
TRAIN_FOLDS=('fold_0','fold_1'); DEV_FOLD='fold_2'; CERT_FOLDS=('fold_3','fold_4')
DEPTH=30; MAXLEN=1024; OVERLAP=.5; NEG_CAP=4; ACCUM_Q=8; LR=5e-5; WD=.01; SEED=112
ADAPTER=CACHE/'adapter_train01'; TRAIN_META=CACHE/'adapter_train01.json'; RESUME=CACHE/'train_resume.pt'
DEV_S=CACHE/'fold2_scores.f32.npy'; DEV_D=CACHE/'fold2_done.u1.npy'; DEV_M=CACHE/'fold2_scores.json'
ALPHAS=[round(.05*i,2) for i in range(21)]

def compute_dtype():
 import torch
 return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def rj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def stable(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_model(trainable):
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 from peft import LoraConfig,PeftModel,TaskType,get_peft_model
 dtype=compute_dtype()
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
 base=AutoModelForSequenceClassification.from_pretrained(
  MODEL,local_files_only=True,dtype=dtype
 )
 base.config.use_cache=False
 if trainable:
  try: base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  except TypeError: base.gradient_checkpointing_enable()
  model=get_peft_model(base,LoraConfig(
   task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=.05,
   target_modules=['query','value'],modules_to_save=['classifier'],bias='none'
  ))
  for p in model.parameters():
   if p.requires_grad:
    p.data=p.data.float()
  try: model.enable_input_require_grads()
  except Exception: pass
 else:
  model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False)
 return tok,model.to('cuda')

def world():
 qids,questions,golds,folds,stress,docs,passages=b1.load_world(); shortlist,_=b1.load_shortlist(len(qids)); sources=b1.load_sources(len(qids))
 ce=np.load(ROOT/'cache/stage03b1_aiteam_reranker/oof_top30_ce_scores.f32.npy')
 baseline,X45,y,n45=a0.current_ce_lr(qids,questions,golds,folds,docs,stress,shortlist,sources,ce)
 return qids,questions,golds,folds,docs,shortlist,baseline,y,a0.load_names(docs)

def split(qids,folds):
 q2i={q:i for i,q in enumerate(qids)}
 tr=np.asarray([q2i[q] for f in TRAIN_FOLDS for q in folds[f]],np.int32); dv=np.asarray([q2i[q] for q in folds[DEV_FOLD]],np.int32); cert=np.asarray([q2i[q] for f in CERT_FOLDS for q in folds[f]],np.int32)
 return tr,dv,cert

def prep(tag,idx,shortlist,docs):
 local=shortlist[idx,:DEPTH]; vv,rr,sim,views=a0.select_witnesses(tag,idx,local,docs); texts=a0.load_selected_texts(vv,rr,views)
 return local,vv,rr,views,texts

def windows(tok,q,title,li,pos,vv,rr,texts,views):
 out=[]
 for fi in range(2):
  v=views[int(vv[li,pos,fi])]; raw=texts[v][int(rr[li,pos,fi])]; out.extend(b0.make_windows(tok,q,raw,title,MAXLEN,OVERLAP))
 return list(dict.fromkeys(out))

def cand_logit(model,tok,q,ws,micro):
 import torch
 mx=[]
 for s in range(0,len(ws),micro):
  z=ws[s:s+micro]; enc=tok([q]*len(z),z,padding=True,truncation=True,max_length=MAXLEN,return_tensors='pt'); enc={k:v.to('cuda') for k,v in enc.items()}
  with torch.autocast('cuda',dtype=compute_dtype()): x=model(**enc,return_dict=True).logits.view(-1).float()
  mx.append(x.max())
 return torch.stack(mx).max()

def group(shortrow,baserow,yrow):
 pos=[j for j,v in enumerate(yrow) if int(v)==1]
 if not pos:return [],[]
 mp={int(d):j for j,d in enumerate(shortrow)}; neg=[]; blocked=set(pos)
 for d in baserow:
  j=mp.get(int(d))
  if j is not None and j not in blocked and int(yrow[j])==0:
   neg.append(j);blocked.add(j)
   if len(neg)>=NEG_CAP:break
 if len(neg)<NEG_CAP:
  for j in range(len(shortrow)):
   if j not in blocked and int(yrow[j])==0:
    neg.append(j);blocked.add(j)
    if len(neg)>=NEG_CAP:break
 return pos,neg

def train(tr,qids,questions,docs,shortlist,baseline,y,names,local,vv,rr,texts,views,micro):
 import torch,torch.nn.functional as F
 from peft import get_peft_model_state_dict,set_peft_model_state_dict
 from transformers import get_cosine_schedule_with_warmup
 contract={'schema':'stage06b1.train.v1_2','base_sha':rj(MAN)['resolved_revision_sha'],'qids':stable([qids[int(i)] for i in tr]),'depth':DEPTH,'maxlen':MAXLEN,'overlap':OVERLAP,'neg':NEG_CAP,'accum':ACCUM_Q,'lr':LR,'wd':WD,'seed':SEED,'lora':'r16-a32-d05-query-value-classifier','loss':'query-balanced candidate MIL BCE','precision':'BF16 backbone when supported; FP32 trainables'}; ch=stable(contract)
 if TRAIN_META.is_file() and ADAPTER.is_dir():
  m=rj(TRAIN_META)
  if m.get('contract_hash')==ch and m.get('status')=='PASS': print('[train] cache hit');return m
  raise RuntimeError('training cache contract mismatch')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 tok,model=load_model(True);model.train();params=[p for p in model.parameters() if p.requires_grad]
 dtypes={}
 for p in params:dtypes[str(p.dtype)]=dtypes.get(str(p.dtype),0)+p.numel()
 print(f'[train] compute_dtype={compute_dtype()} trainable_dtypes={dtypes} trainable={sum(p.numel() for p in params):,}',flush=True)
 opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
 order=list(range(len(tr)));random.Random(SEED).shuffle(order);sch=get_cosine_schedule_with_warmup(opt,max(1,int(.1*math.ceil(len(order)/ACCUM_Q))),math.ceil(len(order)/ACCUM_Q));start=0
 if RESUME.is_file():
  st=torch.load(RESUME,map_location='cpu',weights_only=False)
  if st['contract_hash']!=ch:raise RuntimeError('resume contract mismatch')
  set_peft_model_state_dict(model,st['adapter']);opt.load_state_dict(st['optimizer']);sch.load_state_dict(st['scheduler']);start=int(st['position']);torch.set_rng_state(st['torch_cpu']);torch.cuda.set_rng_state_all(st['torch_cuda']);random.setstate(st['python']);np.random.set_state(st['numpy']);print(f'[train] resume {start}/{len(order)}')
 opt.zero_grad(set_to_none=True);t0=time.perf_counter();losses=[];cands=wins=skipped=0
 for oi in range(start,len(order)):
  li=order[oi];qi=int(tr[li]);q=questions[qids[qi]];ps,ns=group(local[li],baseline[qi],y[qi,:DEPTH])
  if not ps or not ns:skipped+=1
  else:
   cs=ps+ns;weights=[.5/len(ps)]*len(ps)+[.5/len(ns)]*len(ns);targets=[1.]*len(ps)+[0.]*len(ns)
   for pos,w,tgt in zip(cs,weights,targets):
    title=a0.clean_title(names[int(local[li,pos])]);ws=windows(tok,q,title,li,pos,vv,rr,texts,views);score=cand_logit(model,tok,q,ws,micro);target=torch.tensor(tgt,dtype=torch.float32,device='cuda');loss=F.binary_cross_entropy_with_logits(score,target)*w/ACCUM_Q
    if not torch.isfinite(loss):raise RuntimeError(f'non-finite loss q={qids[qi]} pos={pos}')
    loss.backward();losses.append(float(loss.detach())*ACCUM_Q);cands+=1;wins+=len(ws)
  boundary=((oi+1)%ACCUM_Q==0 or oi+1==len(order))
  if boundary:
   bad=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
   if bad:raise RuntimeError(f'non-finite gradients before clip: {bad[:12]}')
   torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
   opt.step();sch.step();opt.zero_grad(set_to_none=True)
  if boundary and ((oi+1)%128==0 or oi+1==len(order)):
   st={'contract_hash':ch,'position':oi+1,'adapter':{k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':np.random.get_state()};tmp=RESUME.with_suffix('.tmp');torch.save(st,tmp);tmp.replace(RESUME)
  if (oi+1)%32==0 or oi+1==len(order):
   elapsed=time.perf_counter()-t0;done=oi+1-start;eta=elapsed/max(done,1)*(len(order)-oi-1);print(f'[train] {oi+1}/{len(order)} loss128={np.mean(losses[-128:]) if losses else float("nan"):.5f} qps={done/max(elapsed,1e-9):.3f} cand={cands} windows={wins} skipped={skipped} eta_min={eta/60:.1f} vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB',flush=True)
 ADAPTER.mkdir(parents=True,exist_ok=True);model.save_pretrained(ADAPTER,safe_serialization=True);af=ADAPTER/'adapter_model.safetensors';m={'status':'PASS','contract_hash':ch,'contract':contract,'trainable_parameters':sum(p.numel() for p in params),'queries':len(order),'skipped':skipped,'candidates':cands,'windows':wins,'loss_tail':float(np.mean(losses[-128:])),'adapter_sha256':sha(af),'seconds':time.perf_counter()-t0,'peak_gib':torch.cuda.max_memory_reserved()/2**30};TRAIN_META.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');RESUME.unlink(missing_ok=True);del model,tok,opt,sch;gc.collect();torch.cuda.empty_cache();return m

def infer_forward(tok,model,pairs,batch):
 import torch
 vals=[];i=0;cur=batch
 while i<len(pairs):
  j=min(i+cur,len(pairs))
  try:
   enc=tok([x[0] for x in pairs[i:j]],[x[1] for x in pairs[i:j]],padding=True,truncation=True,max_length=MAXLEN,return_tensors='pt');enc={k:v.to('cuda') for k,v in enc.items()}
   with torch.inference_mode(),torch.autocast('cuda',dtype=compute_dtype()):x=model(**enc,return_dict=True).logits.view(-1).float()
   vals.extend(float(v) for v in x.cpu().tolist());i=j
  except torch.cuda.OutOfMemoryError:
   gc.collect();torch.cuda.empty_cache()
   if cur<=1: raise
   cur=max(1,cur//2);print(f'[infer] OOM -> batch={cur}')
 return vals,cur

def score_dev(dv,qids,questions,docs,local,names,vv,rr,texts,views,batch,tm):
 import torch
 contract={'schema':'stage06b1.devscore.v1_2','adapter':tm['adapter_sha256'],'qids':stable([qids[int(i)] for i in dv]),'depth':DEPTH,'maxlen':MAXLEN,'overlap':OVERLAP};ch=stable(contract);ex=[DEV_S.exists(),DEV_D.exists(),DEV_M.exists()]
 if any(ex) and not all(ex):raise RuntimeError('partial dev cache')
 if all(ex):
  meta=rj(DEV_M)
  if meta['contract_hash']!=ch:raise RuntimeError('dev cache mismatch')
  scores=np.lib.format.open_memmap(DEV_S,mode='r+');done=np.lib.format.open_memmap(DEV_D,mode='r+')
 else:
  scores=np.lib.format.open_memmap(DEV_S,mode='w+',dtype=np.float32,shape=(len(dv),DEPTH));scores[:]=np.nan;scores.flush();done=np.lib.format.open_memmap(DEV_D,mode='w+',dtype=np.uint8,shape=(len(dv),));done[:]=0;done.flush();meta={'contract_hash':ch,'contract':contract,'pair_count':0,'current_batch':batch}
 pending=[i for i in range(len(dv)) if int(done[i])==0]
 if not pending:return np.asarray(scores,dtype=np.float32)
 tok,model=load_model(False);model.eval();cur=int(meta.get('current_batch',batch));pairs_total=int(meta.get('pair_count',0));t0=time.perf_counter();new=0
 for li in pending:
  qi=int(dv[li]);q=questions[qids[qi]];pairs=[];owner=[]
  for pos,d in enumerate(local[li]):
   ws=windows(tok,q,a0.clean_title(names[int(d)]),li,pos,vv,rr,texts,views);pairs.extend((q,w) for w in ws);owner.extend([pos]*len(ws))
  vals,cur=infer_forward(tok,model,pairs,cur);mx=np.full(DEPTH,-np.inf,np.float32)
  for p,v in zip(owner,vals):mx[p]=max(mx[p],v)
  scores[li]=mx;done[li]=1;pairs_total+=len(pairs);new+=1
  if new%12==0 or new==len(pending):
   scores.flush();done.flush();completed=int(np.asarray(done).sum());meta={'contract_hash':ch,'contract':contract,'pair_count':pairs_total,'current_batch':cur,'completed':completed,'status':'PASS' if completed==len(dv) else 'IN_PROGRESS'};DEV_M.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print(f'[infer] {completed}/{len(dv)} pairs={pairs_total} batch={cur} rate={new/max(time.perf_counter()-t0,1e-9):.2f} q/s')
 del model,tok;gc.collect();torch.cuda.empty_cache();return np.asarray(scores,dtype=np.float32)

def rank(short,score):
 out=np.empty_like(short)
 for i in range(len(short)):out[i]=short[i,np.lexsort((short[i],-score[i]))]
 return out

def met(rank,qids,golds,docs):return official_metrics({q:[docs[int(x)] for x in rank[i,:5]] for i,q in enumerate(qids)},golds,qids)
def pq(rank,qids,golds,docs):return a0.per_query_recall(rank,qids,golds,docs)
def rrscore(base,short):
 out=np.empty(short.shape,np.float32)
 for i in range(len(short)):
  pos={int(d):r+1 for r,d in enumerate(base[i])};out[i]=np.asarray([1/(10+pos[int(d)]) for d in short[i]],np.float32)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-microbatch',type=int,default=2);ap.add_argument('--infer-batch',type=int,default=32);args=ap.parse_args();CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 import peft
 print('[1/8] load world');qids,questions,golds,folds,docs,shortlist,baseline,y,names=world();tr,dv,cert=split(qids,folds);print(f'  train={len(tr)} dev={len(dv)} cert={len(cert)} CERT untouched')
 print('[2/8] train witnesses');tl,tv,trr,tviews,ttexts=prep('bge_lora_train01_top30',tr,shortlist,docs)
 print('[3/8] train LoRA');tm=train(tr,qids,questions,docs,shortlist,baseline,y,names,tl,tv,trr,ttexts,tviews,args.train_microbatch);del tl,tv,trr,ttexts;gc.collect()
 print('[4/8] dev witnesses');dl,dv_vv,dv_rr,dviews,dtexts=prep('bge_lora_dev2_top30',dv,shortlist,docs)
 print('[5/8] score fold2');bs=score_dev(dv,qids,questions,docs,dl,names,dv_vv,dv_rr,dtexts,dviews,args.infer_batch,tm)
 print('[6/8] DEV ranking/blend');qdev=[qids[int(i)] for i in dv];base=baseline[dv,:DEPTH];bm=met(base,qdev,golds,docs);brank=rank(dl,bs);bgem=met(brank,qdev,golds,docs);bz=a0.zrows(bs);rz=a0.zrows(rrscore(base,dl));grid=[];best=None;bestr=None
 for a in ALPHAS:
  r=rank(dl,a*bz+(1-a)*rz);m=met(r,qdev,golds,docs);grid.append({'alpha_bge':a,**m});key=(m['recall_at_5'],m['precision_at_5'])
  if best is None or key>(best[0],best[1]):best=(m['recall_at_5'],m['precision_at_5'],a,m);bestr=r
 alpha=best[2];blend=best[3]
 print('[7/8] diagnostics');bq=pq(base,qdev,golds,docs);gq=pq(brank,qdev,golds,docs);hq=pq(bestr,qdev,golds,docs);oracle=float(np.maximum(bq,gq).mean());best_method='blend' if blend['recall_at_5']>=bgem['recall_at_5'] else 'bge_standalone';bestm=blend if best_method=='blend' else bgem;delta=bestm['recall_at_5']-bm['recall_at_5'];sd=bestm['single_gold_recall_at_5']-bm['single_gold_recall_at_5'];gate=bestm['recall_at_5']>=.955 and delta>=.005 and sd>=0
 report={'schema':'dsc2026.endgame.stage06b1.bge_lora_dev.v1_2','status':'DEV_COMPLETE_CERT_UNTOUCHED','cert_touched':False,'partition':{'train_folds':list(TRAIN_FOLDS),'dev_fold':DEV_FOLD,'cert_folds':list(CERT_FOLDS),'train_queries':len(tr),'dev_queries':len(dv),'cert_queries':len(cert)},'training':tm,'methods':{'baseline_ce_lr':bm,'bge_lora_standalone':bgem,'best_blend':blend},'blend':{'winner_alpha_bge':alpha,'grid':grid},'complementarity':{'oracle':oracle,'bge_wins':int(np.sum(gq>bq)),'bge_losses':int(np.sum(gq<bq)),'blend_wins':int(np.sum(hq>bq)),'blend_losses':int(np.sum(hq<bq))},'gate':{'min_recall':.955,'min_delta':.005,'best_method':best_method,'best_recall':bestm['recall_at_5'],'delta_recall':delta,'delta_single':sd,'decision':'PROCEED_TO_STAGE06B2_CERT' if gate else 'KILL_OR_REDESIGN'}}; (OUT/'DEV_GATE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# Stage 06B1 — BGE LoRA DEV Gate','','**CERT folds 3–4 were not evaluated.**','','| Method | R@5 | P@5 | Single | Multi |','|---|---:|---:|---:|---:|']
 for n,m in report['methods'].items():lines.append(f"| {n} | {m['recall_at_5']:.6f} | {m['precision_at_5']:.6f} | {m['single_gold_recall_at_5']:.6f} | {m['multi_gold_recall_at_5']:.6f} |")
 lines+=['',f'- Best alpha(BGE): **{alpha:.2f}**',f'- Delta R@5: **{delta:+.6f}**',f'- Delta single: **{sd:+.6f}**',f"- Decision: **{report['gate']['decision']}**",''];(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print('[8/8] report');print('='*112);print('CERT UNTOUCHED')
 for n,m in report['methods'].items():print(f"{n:24s} R={m['recall_at_5']:.9f} P={m['precision_at_5']:.9f} single={m['single_gold_recall_at_5']:.9f} multi={m['multi_gold_recall_at_5']:.9f}")
 print(f'BEST alpha_bge={alpha:.2f} delta_R={delta:+.9f} delta_single={sd:+.9f}');print(f"BGE W/L={report['complementarity']['bge_wins']}/{report['complementarity']['bge_losses']} BLEND W/L={report['complementarity']['blend_wins']}/{report['complementarity']['blend_losses']} ORACLE={oracle:.9f}");print('DECISION:',report['gate']['decision']);print('REPORT:',OUT/'REPORT.md');print('='*112)
if __name__=='__main__':main()

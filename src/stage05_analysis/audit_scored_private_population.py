#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,hashlib,json,re,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/official_v1'; EVAL=ROOT/'data/evaluation_v2'
CACHE=ROOT/'cache/stage04_private_submission_v1_2'
SUB=ROOT/'submissions/endgame_20260922'
OUT=ROOT/'reports/stage05a_private_failure_map'
PRIVATE=RAW/'private-official.json'; CORPUS=EVAL/'retrieval_corpus_8512.jsonl'
SOURCES=['ait_atomic','ait_coarse1024','lal_coarse1024','lal_atomic','lal_b4','bm25']
EXPECTED_SHA='9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc'
WS=re.compile(r'\s+',re.UNICODE); PUNCT=re.compile(r'[^\wÀ-ỹĐđ]+',re.UNICODE); WORDS=re.compile(r'\b[\wÀ-ỹĐđ]+\b',re.UNICODE)
ARTICLE=re.compile(r'\bđiều\s+\d+[a-zđ]?\b',re.I); CLAUSE=re.compile(r'\bkhoản\s+\d+\b',re.I); POINT=re.compile(r'\bđiểm\s+(?:[a-zđ]|\d+)\b',re.I)
SANCTION=('xử phạt','mức phạt','phạt bao nhiêu','tiền phạt','bị phạt','hình phạt','chế tài','phạt tiền','truy cứu')
DEFINITION=('là gì','được hiểu là','thế nào là','khái niệm','định nghĩa','có nghĩa là')
PROCEDURE=('thủ tục','hồ sơ','trình tự','nộp hồ sơ','đăng ký','cấp giấy','cấp phép','xin phép','giải quyết','thẩm quyền','cơ quan nào','gia hạn','cấp lại','cấp đổi','thu hồi','đề nghị')
CONDITION=('điều kiện','khi nào','trường hợp nào','trong trường hợp','có được','được phép','không được','phải đáp ứng','cần đáp ứng','đủ điều kiện')
CROSS=('theo quy định tại','quy định tại điều','quy định tại khoản','căn cứ','dẫn chiếu','theo điều','theo khoản','theo điểm','tại điều','tại khoản')
INSTR=('luật ','bộ luật','nghị định','thông tư','quyết định','nghị quyết','pháp lệnh','hiến pháp','quy chuẩn','tiêu chuẩn','văn bản')

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(s): return WS.sub(' ',unicodedata.normalize('NFKC',str(s or '')).strip().lower())
def loose(s): return WS.sub(' ',PUNCT.sub(' ',norm(s))).strip()
def loadq(p):
 o=rj(p); ids=[]; q={}
 if isinstance(o,dict):
  it=o.items()
 else: it=enumerate(o)
 for k,v in it:
  if isinstance(v,str): qid=str(k); text=v
  elif isinstance(v,dict): qid=str(v.get('qid',v.get('id',v.get('query_id',k)))); text=v.get('question') or v.get('query') or v.get('text') or v.get('content')
  else: continue
  if text is not None: ids.append(qid); q[qid]=str(text)
 if not ids: raise RuntimeError(f'no questions: {p}')
 return ids,q
def warmup_path(explicit):
 if explicit:
  p=Path(explicit).expanduser().resolve();
  if not p.is_file(): raise FileNotFoundError(p)
  return p
 par=ROOT.parent
 cand=[RAW/'warmup.json',ROOT/'warmup.json',par/'sota/DSC2026-LegalIR-main/v4_run/public_test_dataset/warmup.json',par/'sota/DSC2026-LegalIR-main/data/warmup.json',par/'sota/data/warmup.json',par/'LegalIR/DSC2026-LegalIR-main/v4_run/public_test_dataset/warmup.json',par/'LegalIR/DSC2026-LegalIR-main/data/warmup.json',par/'LegalIR/data/warmup.json']
 for p in cand:
  if p.is_file(): return p.resolve()
 raise FileNotFoundError('warmup.json not found; pass --warmup PATH\n'+'\n'.join(map(str,cand)))
def qfeat(text):
 t=norm(text); has=lambda xs:any(x in t for x in xs); a=bool(ARTICLE.search(t)); c=bool(CLAUSE.search(t)); p=bool(POINT.search(t))
 if has(SANCTION): arch='SANCTION'
 elif has(DEFINITION): arch='DEFINITION'
 elif has(PROCEDURE): arch='PROCEDURE'
 elif has(CONDITION): arch='CONDITION'
 elif has(CROSS) and (a or c or p): arch='CROSS_REFERENCE'
 elif a or c or p or has(INSTR): arch='DIRECT_REF'
 else: arch='GENERAL_SEMANTIC'
 cues={'ARTICLE':a,'CLAUSE':c,'POINT':p,'INSTRUMENT':has(INSTR),'CROSSREF':has(CROSS),'NEGATION':('không' in t or 'chưa' in t),'HYPOTHETICAL':('nếu ' in t or 'trong trường hợp' in t or 'khi ' in t)}
 return arch,len(WORDS.findall(t)),cues
def writecsv(p,rows):
 rows=list(rows)
 if not rows:return
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def loadsub(name,qids):
 o=rj(SUB/f'{name}.json')
 if set(o)!=set(qids): raise RuntimeError(f'{name} qid mismatch')
 return {q:[str(x) for x in o[q]['answer']] for q in qids}
def jacc(sets):
 v=[]
 for i in range(len(sets)):
  for j in range(i+1,len(sets)):
   u=sets[i]|sets[j]; v.append(len(sets[i]&sets[j])/len(u))
 return float(np.mean(v))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--warmup',type=Path); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
 if sha(PRIVATE)!=EXPECTED_SHA: raise RuntimeError('private SHA drift')
 pids,pq=loadq(PRIVATE); wp=warmup_path(a.warmup); wids,wq=loadq(wp)
 if len(pids)!=2080: raise RuntimeError(f'private count={len(pids)}')
 print('[1/5] warmup overlap',wp)
 wset=set(wids); nmap=defaultdict(list); lmap=defaultdict(list)
 for q in wids: nmap[norm(wq[q])].append(q); lmap[loose(wq[q])].append(q)
 excluded=set(); ov=[]
 for q in pids:
  n=norm(pq[q]); exact=nmap.get(n,[]); same=q in wset; exc=bool(same or exact)
  if exc: excluded.add(q)
  lo=[x for x in lmap.get(loose(pq[q]),[]) if x not in exact]
  if exc or lo: ov.append({'qid':q,'question':pq[q],'excluded':exc,'same_qid':same,'same_qid_same_text':bool(same and norm(wq[q])==n),'exact_warmup_qids':exact,'loose_only_qids':lo})
 scored=[q for q in pids if q not in excluded]
 (OUT/'OVERLAP_AUDIT.json').write_text(json.dumps({'private':len(pids),'warmup':len(wids),'excluded':len(excluded),'scored':len(scored),'warmup_path':str(wp),'warmup_sha256':sha(wp),'rows':ov},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if len(excluded)!=103 or len(scored)!=1977: raise RuntimeError(f'expected 103/1977 got {len(excluded)}/{len(scored)}; inspect OVERLAP_AUDIT.json')
 writecsv(OUT/'EXCLUDED_103.csv',[{'qid':x['qid'],'question':x['question'],'same_qid':int(x['same_qid']),'same_qid_same_text':int(x['same_qid_same_text']),'exact_warmup_qids':'|'.join(x['exact_warmup_qids'])} for x in ov if x['excluded']])
 print(' PASS 2080-103=1977')
 print('[2/5] current ENDGAME artifacts')
 docs=[]; names={}
 with CORPUS.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():
    x=json.loads(line); d=str(x['id']); docs.append(d); names[d]=str(x.get('name') or '')
 d2i={d:i for i,d in enumerate(docs)}; src={}
 for s in SOURCES:
  ip=CACHE/'retrieval'/f'{s}_idx.npy'; sp=CACHE/'retrieval'/f'{s}_scores.npy'; idx=np.load(ip); scr=np.load(sp)
  if idx.shape!=(2080,100) or scr.shape!=(2080,100): raise RuntimeError(f'{s} shape drift')
  src[s]=(idx,scr)
 sl=np.load(CACHE/'private_shortlist_top30_idx.npy'); ce=np.load(CACHE/'reranker/scores.npy'); done=np.load(CACHE/'reranker/done.npy')
 if sl.shape!=(2080,30) or ce.shape!=(2080,30) or int(done.sum())!=2080: raise RuntimeError('Stage04 cache incomplete')
 rank=loadsub('ENDGAME_PRIVATE_RANK_ONLY_K5',pids); blend=loadsub('ENDGAME_PRIVATE_BLEND015_K5',pids); primary=loadsub('ENDGAME_PRIVATE_CE_LR_K5',pids); q2i={q:i for i,q in enumerate(pids)}
 print('[3/5] build failure map')
 flat=[]; detail=[]
 for n,q in enumerate(scored,1):
  qi=q2i[q]; arch,wc,cues=qfeat(pq[q]); s5={s:[docs[int(x)] for x in src[s][0][qi,:5]] for s in SOURCES}; s10={s:[docs[int(x)] for x in src[s][0][qi,:10]] for s in SOURCES}; dis=1-jacc([set(v) for v in s5.values()]); union=len(set().union(*(set(v) for v in s5.values())))
  order=np.lexsort((sl[qi],-ce[qi])); css=ce[qi][order]; margin=float(css[4]-css[5]); ce5=[docs[int(sl[qi,j])] for j in order[:5]]
  r5=rank[q]; p5=primary[q]; b5=blend[q]; churn=set(r5)!=set(p5); ent=[d for d in p5 if d not in set(r5)]; leave=[d for d in r5 if d not in set(p5)]
  support=[]
  for d in p5:
   di=d2i[d]; support.append({'doc_id':d,'name':names.get(d,''),'top5':sum(di in set(map(int,src[s][0][qi,:5])) for s in SOURCES),'top10':sum(di in set(map(int,src[s][0][qi,:10])) for s in SOURCES),'top20':sum(di in set(map(int,src[s][0][qi,:20])) for s in SOURCES)})
  detail.append({'qid':q,'question':pq[q],'archetype':arch,'word_count':wc,'cues':cues,'source_disagreement_at5':dis,'source_top5_union_size':union,'ce_margin5_minus6':margin,'ce_score_std':float(np.std(ce[qi])),'rank_to_primary_set_churn':churn,'entering':ent,'leaving':leave,'rank_only_top5':r5,'blend015_top5':b5,'primary_top5':p5,'ce_only_top5':ce5,'source_top5':s5,'source_top10':s10,'primary_support':support})
  flat.append({'qid':q,'question':pq[q],'archetype':arch,'word_count':wc,'active_cues':'|'.join(k for k,v in cues.items() if v),'source_disagreement_at5':dis,'source_top5_union_size':union,'ce_margin5_minus6':margin,'ce_score_std':float(np.std(ce[qi])),'rank_to_primary_set_churn':int(churn),'entering':'|'.join(ent),'leaving':'|'.join(leave),'rank_only_top5':'|'.join(r5),'blend015_top5':'|'.join(b5),'primary_top5':'|'.join(p5),'ce_only_top5':'|'.join(ce5),'primary_top5_names':' || '.join(names.get(d,'') for d in p5),'primary_support5':'|'.join(str(x['top5']) for x in support),'primary_support10':'|'.join(str(x['top10']) for x in support),'primary_support20':'|'.join(str(x['top20']) for x in support)})
  if n%500==0 or n==len(scored): print(n,'/',len(scored))
 writecsv(OUT/'SCORED_1977.csv',flat)
 with (OUT/'SCORED_1977_DIAGNOSTICS.jsonl').open('w',encoding='utf-8') as f:
  for r in detail:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print('[4/5] slices')
 writecsv(OUT/'CE_UNCERTAINTY_TOP200.csv',sorted(flat,key=lambda r:(r['ce_margin5_minus6'],r['qid']))[:200]); writecsv(OUT/'SOURCE_DISAGREEMENT_TOP200.csv',sorted(flat,key=lambda r:(-r['source_disagreement_at5'],r['qid']))[:200]); cr=[r for r in flat if r['rank_to_primary_set_churn']]; cr.sort(key=lambda r:(r['ce_margin5_minus6'],-r['source_disagreement_at5'],r['qid'])); writecsv(OUT/'CE_CHURN_TOP200.csv',cr[:200])
 groups=defaultdict(list)
 for r in flat:groups[r['archetype']].append(r)
 ar=[]
 for k,rr in sorted(groups.items()): ar.append({'archetype':k,'queries':len(rr),'fraction':len(rr)/1977,'mean_source_disagreement_at5':float(np.mean([x['source_disagreement_at5'] for x in rr])),'mean_ce_margin5_minus6':float(np.mean([x['ce_margin5_minus6'] for x in rr])),'ce_set_churn_rate':float(np.mean([x['rank_to_primary_set_churn'] for x in rr])),'mean_top5_union_size':float(np.mean([x['source_top5_union_size'] for x in rr]))})
 writecsv(OUT/'ARCHETYPE_SUMMARY.csv',ar)
 print('[5/5] report')
 churn=sum(r['rank_to_primary_set_churn'] for r in flat); rep={'schema':'dsc2026.endgame.stage05a_private_failure_map.v1','status':'PASS','private_labels_used':False,'codabench_per_query_outcomes_used':False,'population':{'private':2080,'warmup_excluded':103,'scored':1977},'current_public_score':{'precision':0.201922105,'recall':0.941915361},'pipeline':{'rank_to_primary_set_churn':int(churn),'rank_to_primary_set_churn_rate':churn/1977,'mean_source_disagreement_at5':float(np.mean([r['source_disagreement_at5'] for r in flat])),'median_ce_margin5_minus6':float(np.median([r['ce_margin5_minus6'] for r in flat]))},'archetypes':ar,'claim_boundary':'label-free hypothesis generation only; validate every repair OOF'}
 (OUT/'REPORT.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 md=['# Stage 05A — Actually Scored PRIVATE Failure Map','','**Label-free. No private labels/per-query Codabench outcomes used.**','','- Population: **2080 - 103 = 1977**','- Current public: **R=0.941915361 / P=0.201922105**',f'- Rank-only → CE-LR set churn: **{churn}/1977 ({churn/1977:.1%})**','','| Archetype | N | Fraction | Source disagreement@5 | CE margin5-6 | CE churn |','|---|---:|---:|---:|---:|---:|']
 for r in ar:md.append(f"| {r['archetype']} | {r['queries']} | {r['fraction']:.1%} | {r['mean_source_disagreement_at5']:.3f} | {r['mean_ce_margin5_minus6']:.4f} | {r['ce_set_churn_rate']:.1%} |")
 md += ['','Inspect first: `SCORED_1977.csv`, `CE_UNCERTAINTY_TOP200.csv`, `SOURCE_DISAGREEMENT_TOP200.csv`, `CE_CHURN_TOP200.csv`.','','Any policy discovered here must beat frozen TRAIN OOF before promotion.','']
 (OUT/'REPORT.md').write_text('\n'.join(md),encoding='utf-8'); print('PASS',OUT/'REPORT.md')
if __name__=='__main__':main()

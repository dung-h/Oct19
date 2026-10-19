#!/usr/bin/env python3
"""Find abstracts and open repository full texts through the public OpenAIRE API."""
import csv, html, json, os, re, time, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import requests

ROOT=os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.join(ROOT,'abstracts_completed.csv')
CACHE=os.path.join(ROOT,'openaire_abstract_cache.jsonl')
UA='Academic paper abstract metadata retrieval'


def norm(s):
 s=html.unescape(s or '').lower()
 s=''.join(ch for ch in unicodedata.normalize('NFKD',s) if not unicodedata.combining(ch))
 return re.sub(r'[^a-z0-9]+',' ',s).strip()

def base_title(s):
 s=html.unescape(s or '').strip()
 s=re.sub(r'\s*\((?:short paper|invited talk|abstract|extended abstract|system description|poster|demo)\)\s*\.?$','',s,flags=re.I)
 s=re.sub(r'\s*(?:[-:]\s*)?(?:extended\s+)?abstract\s*\.?$','',s,flags=re.I)
 return s.rstrip(' .')

def sim(a,b):return SequenceMatcher(None,norm(base_title(a)),norm(base_title(b))).ratio()

def values(x):
 if x is None:return []
 if isinstance(x,dict):x=[x]
 if not isinstance(x,list):return []
 return [v.get('$','') for v in x if isinstance(v,dict) and v.get('$')]

def clean_text(x):
 x=html.unescape(re.sub(r'<[^>]+>',' ',str(x or '')))
 return re.sub(r'\s+',' ',x).strip()

def valid_description(x):
 x=clean_text(x)
 if len(x)<100:return ''
 if re.match(r'(?i)^(sponsorship|no description supplied|no abstract (?:available|supplied)|not available)\b',x):return ''
 return x

def surnames_from_result(result):
 out=[]
 creators=result.get('creator') or []
 if isinstance(creators,dict):creators=[creators]
 for c in creators:
  if not isinstance(c,dict):continue
  surname=c.get('@surname')
  if not surname:
   raw=c.get('$','')
   surname=raw.split(',',1)[0] if ',' in raw else raw.split()[-1] if raw.split() else ''
  if surname:out.append(surname)
 return out

def author_overlap(source_authors, result):
 src=[]
 for author in (source_authors or '').split(','):
  bits=norm(author).split()
  if bits:src.append(bits[-1])
 cand=[norm(x).split()[-1] for x in surnames_from_result(result) if norm(x)]
 return len(set(src)&set(cand))/max(1,min(len(set(src)),len(set(cand)))) if src and cand else 0.0

def extract_result_candidates(response, target):
 payload=response.get('response') or {}
 results=((payload.get('results') or {}).get('result') or [])
 if isinstance(results,dict):results=[results]
 out=[]
 for item in results:
  if not isinstance(item,dict):continue
  meta=item.get('metadata') or {}; entity=meta.get('oaf:entity') or {}; result=entity.get('oaf:result') or {}
  titles=values(result.get('title'))
  if not titles:continue
  best_title=max(titles,key=lambda x:sim(target.get('title',''),x))
  score=sim(target.get('title',''),best_title)
  overlap=author_overlap(target.get('authors',''),result)
  # Exact/near-exact titles are required. Author overlap breaks ties and protects short/common titles.
  if score<.96 and not (score>=.92 and overlap>=.5):continue
  descriptions=values(result.get('description'))
  description=next((a for a in (valid_description(d) for d in descriptions) if a), '')
  fulltext=values(result.get('fulltext'))
  pids=result.get('pid') or []
  if isinstance(pids,dict):pids=[pids]
  dois=[p.get('$','') for p in pids if isinstance(p,dict) and p.get('@classid')=='doi' and p.get('$')]
  sources=result.get('collectedfrom') or []
  if isinstance(sources,dict):sources=[sources]
  source_names=[s.get('@name','') for s in sources if isinstance(s,dict) and s.get('@name')]
  out.append({'title':best_title,'title_similarity':round(score,4),'author_overlap':round(overlap,4),
              'abstract':description,'fulltext_urls':fulltext[:8],'dois':dois[:8],
              'authors':surnames_from_result(result),'metadata_sources':source_names})
 out.sort(key=lambda x:(bool(x['abstract']),bool(x['fulltext_urls']),x['title_similarity'],x['author_overlap']),reverse=True)
 return out

def fetch_one(row):
 try:
  response=requests.get('https://api.openaire.eu/search/publications',
   params={'title':html.unescape(row.get('title','')),'format':'json','size':5},
   headers={'User-Agent':UA},timeout=35)
  if response.status_code==429:
   time.sleep(max(3,float(response.headers.get('Retry-After','3'))))
   response=requests.get('https://api.openaire.eu/search/publications',
    params={'title':html.unescape(row.get('title','')),'format':'json','size':5},
    headers={'User-Agent':UA},timeout=35)
  response.raise_for_status()
  payload=response.json()
  candidates=extract_result_candidates(payload,row)
  return {'split':row['split'],'id':row['id'],'method':'openaire_title','error':'','candidates':candidates}
 except Exception as e:
  return {'split':row['split'],'id':row['id'],'method':'openaire_title','error':str(e),'candidates':[]}

def main():
 with open(INPUT,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 missing=[r for r in rows if not r.get('abstract','').strip()]
 done=set()
 if os.path.exists(CACHE):
  for line in open(CACHE,encoding='utf-8'):
   try:
    x=json.loads(line)
    if x.get('method')=='openaire_title' and not x.get('error'):done.add((x['split'],x['id']))
   except Exception:pass
 todo=[r for r in missing if (r['split'],r['id']) not in done]
 print(f'{len(missing)} rows still need abstracts; querying OpenAIRE for {len(todo)} of them.',flush=True)
 if not todo:return
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(fetch_one,r) for r in todo]
  for i,future in enumerate(as_completed(futures),1):
   entry=future.result()
   with open(CACHE,'a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
   if i%25==0 or i==len(futures):
    hits=sum(bool(x['candidates']) for x in [])
    print(f'  OpenAIRE query {i}/{len(futures)} (latest {entry["split"]} id {entry["id"]}: {len(entry["candidates"])} title matches).',flush=True)
 print('OpenAIRE title lookups complete; candidates and any repository PDF URLs are cached.',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Harvest Crossref metadata by conference/year, then validate against local titles/authors."""
import csv, html, io, json, os, re, time, unicodedata, zipfile
from collections import defaultdict
from difflib import SequenceMatcher
import requests
from urllib.parse import quote

ROOT=os.path.dirname(os.path.abspath(__file__))
ZIP_PATH=os.path.join(ROOT,'assignment-data-mining-261.zip')
ARTIFACT=os.path.join(ROOT,'abstracts_multisource.csv')
CACHE=os.path.join(ROOT,'abstract_retrieval_cache.jsonl')
UA='Academic paper abstract retrieval for a data-mining assignment'
VENUE_QUERY={
 'cav':'International Conference on Computer Aided Verification CAV',
 'kr':'International Conference on Principles of Knowledge Representation and Reasoning KR',
 'iclp':'International Conference on Logic Programming ICLP',
 'lpnmr':'International Conference on Logic Programming and Nonmonotonic Reasoning LPNMR',
 'lics':'Symposium on Logic in Computer Science LICS',
}

def parse_test_rtf(raw):
 s=raw.decode('latin1'); start=s.find('\\cf0 ')
 if start<0: raise ValueError('RTF CSV marker missing')
 s=s[start+len('\\cf0 '):]; s=re.sub(r'\\\r?\n','\n',s)
 s=re.sub(r'\\u(-?\d+) ?(?:\?)?',lambda m:chr(int(m.group(1))%65536),s).replace('\\uc0','')
 s=re.sub(r"\\'([0-9a-fA-F]{2})",lambda m:bytes([int(m.group(1),16)]).decode('cp1252'),s)
 s=s.replace('\\{','{').replace('\\}','}').replace('\\\\','\\')
 s='\n'.join(x[:-1] if x.endswith('\\') else x for x in s.splitlines()).rsplit('}',1)[0]
 return list(csv.DictReader(io.StringIO(s)))

def norm(s):
 s=html.unescape(s or '').lower()
 s=''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
 return re.sub(r'[^a-z0-9]+',' ',s).strip()

def title_base(s):
 s=html.unescape(s or '').strip()
 s=re.sub(r'\s*\((?:short paper|invited talk|abstract|extended abstract|system description|poster|demo)\)\s*\.?$','',s,flags=re.I)
 s=re.sub(r'\s*(?:[-:]\s*)?(?:extended\s+)?abstract\s*\.?$','',s,flags=re.I)
 return s.rstrip(' .')

def sim(a,b):return SequenceMatcher(None,norm(title_base(a)),norm(title_base(b))).ratio()

def clean_abstract(s):
 if not s:return ''
 s=html.unescape(re.sub(r'<[^>]+>',' ',str(s)))
 return re.sub(r'\s+',' ',s).strip()

def candidate(item):
 return {'doi':item.get('DOI',''),'title':(item.get('title')or[''])[0],
  'year':str(((item.get('published') or item.get('issued') or {}).get('date-parts') or [['']])[0][0]),
  'abstract':clean_abstract(item.get('abstract','')),
  'authors':[a.get('family','') for a in item.get('author',[]) if a.get('family')],
  'container_title':item.get('container-title',[])}

def author_overlap(src,cand):
 left={norm(x).split()[-1] for x in (src or '').split(',') if norm(x)}
 right={norm(x).split()[-1] for x in (cand or []) if norm(x)}
 return len(left&right)/max(1,min(len(left),len(right))) if left and right else 0.0

def load_done():
 done=set()
 if os.path.exists(CACHE):
  for line in open(CACHE,encoding='utf-8'):
   try:
    x=json.loads(line)
    if x.get('method')=='crossref_venue_batch':done.add((x['split'],x['id']))
   except Exception:pass
 return done

def append_cache(x):
 with open(CACHE,'a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')

def fetch(query,year):
 params={'query':query,'filter':f'from-pub-date:{year-1}-01-01,until-pub-date:{year+1}-12-31',
         'rows':300,'select':'DOI,title,author,issued,published,abstract,container-title'}
 for n in range(5):
  try:
   r=requests.get('https://api.crossref.org/works',params=params,headers={'User-Agent':UA},timeout=60)
   if r.status_code==429:
    time.sleep(max(3,float(r.headers.get('Retry-After','3'))));continue
   r.raise_for_status(); return r.json()['message'],None
  except Exception as e:
   if n==4:return None,str(e)
   time.sleep(2+n)
 return None,'retry limit exceeded'

def main():
 with zipfile.ZipFile(ZIP_PATH) as z:
  tr=list(csv.DictReader(io.TextIOWrapper(z.open('train.csv'),encoding='utf-8-sig',newline='')))
  te=parse_test_rtf(z.read('Test.csv'))
 source={('train',r['id']):r for r in tr}; source.update({('test',r['id']):r for r in te})
 with open(ARTIFACT,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 missing=[r for r in rows if not r['abstract'].strip()]
 groups=defaultdict(list)
 for r in missing:
  src=source[(r['split'],r['id'])]
  venue=(src.get('venue') or '').lower(); year=int(src.get('year') or 0)
  if venue in VENUE_QUERY and year:groups[(venue,year)].append(r)
 done=load_done(); batches=sorted(groups.items())
 print(f'{len(missing)} missing rows grouped into {len(batches)} venue/year batches.',flush=True)
 last=0
 for i,((venue,year),targets) in enumerate(batches,1):
  todo=[r for r in targets if (r['split'],r['id']) not in done]
  if not todo:continue
  query=f'{VENUE_QUERY[venue]} {year}'
  wait=1.15-(time.monotonic()-last)
  if wait>0:time.sleep(wait)
  last=time.monotonic()
  result,error=fetch(query,year)
  matches=0; abstract_matches=0
  candidates=[candidate(x) for x in (result or {}).get('items',[])]
  for r in todo:
   src=source[(r['split'],r['id'])]
   ranked=[]
   for c in candidates:
    score=sim(src.get('title',''),c['title'])
    authors=author_overlap(src.get('authors',''),c['authors'])
    confidence='high' if score>=.96 else ('medium' if score>=.92 and authors>=.5 else '')
    if confidence:ranked.append((score,authors,confidence,c))
   ranked.sort(key=lambda x:(x[0],x[1]),reverse=True)
   entry={'split':r['split'],'id':r['id'],'method':'crossref_venue_batch','query':query,
          'error':error or '', 'scanned':len(candidates), 'candidate':None}
   if ranked:
    score,authors,confidence,c=ranked[0]
    entry.update(candidate=c,title_similarity=round(score,4),author_overlap=round(authors,4),confidence=confidence)
    matches+=1;abstract_matches+=bool(c['abstract'])
   append_cache(entry)
  if i%1==0:
   print(f'  {i}/{len(batches)} {venue} {year}: {len(todo)} rows, {matches} title matches, {abstract_matches} with deposited abstracts; scanned {len(candidates)}.',flush=True)
 print('Crossref venue/year harvesting complete.',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Match unresolved KR papers against the official KR proceedings and capture their abstracts."""
import csv, html, json, os, re, time, unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from difflib import SequenceMatcher
from html.parser import HTMLParser
import requests

ROOT=os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.join(ROOT,'abstracts_collected.csv')
CACHE=os.path.join(ROOT,'kr_proceedings_cache.jsonl')
UA='Academic paper metadata retrieval (KR proceedings matching)'
YEARS=range(2020,2026)

def clean_title(s):
 s=html.unescape(s or '').strip()
 s=re.sub(r'\s*\((?:short paper|invited talk|abstract|extended abstract|system description|poster|demo)\)\s*\.?$','',s,flags=re.I)
 s=re.sub(r'\s*(?:[-:]\s*)?(?:extended\s+)?abstract\s*\.?$','',s,flags=re.I)
 return s.rstrip(' .')
def norm(s):
 s=clean_title(s).lower();s=''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c));return re.sub(r'[^a-z0-9]+',' ',s).strip()
def sim(a,b):return SequenceMatcher(None,norm(a),norm(b)).ratio()
def author_overlap(src,authors):
 a={norm(x).split()[-1] for x in (src or '').split(',') if norm(x)}
 b={norm(x).split()[-1] for x in (authors or []) if norm(x)}
 return len(a&b)/max(1,min(len(a),len(b))) if a and b else 0.0

class IndexParser(HTMLParser):
 def __init__(self):super().__init__();self.href=None;self.buf=[];self.rows=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   d=dict(attrs);u=d.get('href','')
   if re.match(r'^/20\d\d/[^/]+/$',u):self.href=u;self.buf=[]
 def handle_data(self,data):
  if self.href is not None:self.buf.append(data)
 def handle_endtag(self,tag):
  if tag=='a' and self.href is not None:
   t=re.sub(r'\s+',' ',''.join(self.buf)).strip()
   if t and t.lower() not in {'details','pdf','bibtex'}:self.rows.append((self.href,t))
   self.href=None;self.buf=[]

class DetailParser(HTMLParser):
 def __init__(self):super().__init__();self.meta={};self.in_h2=False;self.in_abs=False;self.h2='';self.parts=[];self.authors=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='meta':
   name=(d.get('name') or '').lower()
   if name:self.meta[name]=d.get('content','')
  if tag=='h2':self.in_h2=True;self.h2=''
  if tag=='meta' and (d.get('name') or '').lower()=='citation_author' and d.get('content'):self.authors.append(d['content'])
 def handle_data(self,data):
  if self.in_h2:self.h2+=data
  if self.in_abs:self.parts.append(data)
 def handle_endtag(self,tag):
  if tag=='h2':
   if self.in_abs:self.in_abs=False
   if self.h2.strip().lower()=='abstract':self.in_abs=True
   self.in_h2=False

def fetch(url):
 r=requests.get(url,headers={'User-Agent':UA},timeout=30)
 if r.status_code==429:
  time.sleep(max(2,float(r.headers.get('Retry-After','2'))));r=requests.get(url,headers={'User-Agent':UA},timeout=30)
 r.raise_for_status();return r

def load_latest():
 out={}
 if os.path.exists(CACHE):
  for l in open(CACHE,encoding='utf-8'):
   try:
    x=json.loads(l);out[(x['split'],x['id'])]=x
   except:pass
 return out

def main():
 with open(INPUT,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 unresolved=[r for r in rows if not r.get('abstract','').strip() and r.get('venue')=='kr']
 done=load_latest()
 targets=[r for r in unresolved if (r['split'],r['id']) not in done]
 print(f'{len(unresolved)} KR rows remain; checking official 2020–2025 proceedings for {len(targets)}.',flush=True)
 index={}
 for y in YEARS:
  try:
   r=fetch(f'https://proceedings.kr.org/{y}/')
   p=IndexParser();p.feed(r.text)
   index[y]=[(route,title) for route,title in p.rows]
   print(f'  KR {y}: {len(index[y])} paper/detail links.',flush=True)
  except Exception as e:print(f'  KR {y}: index error {e}',flush=True)
 jobs=[]
 for row in targets:
  best=[]
  for y,items in index.items():
   for route,title in items:
    score=sim(row.get('title',''),title)
    ov=0.0
    if score>=.92:best.append((score,ov,y,route,title))
  if best:
   best.sort(reverse=True);score,ov,y,route,title=best[0]
   jobs.append((row,y,route,title,score))
  else:
   entry={'split':row['split'],'id':row['id'],'method':'kr_official_proceedings','abstract':'','title':'','url':'','error':'No title match in official proceedings pages 2020-2025'}
   with open(CACHE,'a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
 print(f'  Matched {len(jobs)} remaining KR rows by title.',flush=True)
 def work(job):
  row,y,route,title,score=job
  try:
   url='https://proceedings.kr.org'+route
   r=fetch(url);p=DetailParser();p.feed(r.text)
   abstract=re.sub(r'\s+',' ',' '.join(p.parts)).strip()
   if len(abstract)<100:
    abstract=html.unescape(p.meta.get('description','')).strip()
   if len(abstract)<100:raise ValueError('Official detail page contains no Abstract section or usable description')
   return {'split':row['split'],'id':row['id'],'method':'kr_official_proceedings','abstract':abstract,'title':title,'url':url,'title_similarity':round(score,4),'error':''}
  except Exception as e:
   return {'split':row['split'],'id':row['id'],'method':'kr_official_proceedings','abstract':'','title':title,'url':'https://proceedings.kr.org'+route,'title_similarity':round(score,4),'error':str(e)}
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(work,j) for j in jobs]
  for i,f in enumerate(as_completed(futures),1):
   entry=f.result()
   with open(CACHE,'a',encoding='utf-8') as out:out.write(json.dumps(entry,ensure_ascii=False)+'\n')
   if i%10==0 or i==len(futures):print(f'  KR detail {i}/{len(futures)}: {"abstract" if entry["abstract"] else "no abstract"}.',flush=True)
 print('KR official proceedings lookup complete.',flush=True)
if __name__=='__main__':main()

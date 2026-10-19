#!/usr/bin/env python3
"""Extract only explicitly labeled abstracts from OpenAIRE-linked open full texts."""
import csv, html, json, os, re, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import requests

ROOT=os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.join(ROOT,'abstracts_completed.csv')
OPENAIRE_CACHE=os.path.join(ROOT,'openaire_abstract_cache.jsonl')
CACHE=os.path.join(ROOT,'openaire_pdf_abstract_cache.jsonl')
UA='Academic paper abstract retrieval (public repository full text)'
STOP={'the','and','for','with','from','into','using','based','over','under','via','on','in','of','to','a','an','is','are','by','as','at','or'}


def load_latest(path):
 out={}
 if not os.path.exists(path):return out
 with open(path,encoding='utf-8') as f:
  for line in f:
   try:
    x=json.loads(line); key=(x['split'],x['id']); out[key]=x
   except Exception:continue
 return out

class MetaParser(HTMLParser):
 def __init__(self):super().__init__();self.meta=[];self.links=[];self.page_title='';self._title=False
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='meta':
   name=(d.get('name') or d.get('property') or '').lower()
   if name:self.meta.append((name,d.get('content','')))
  if tag=='link' and d.get('href') and '.pdf' in d['href'].lower():self.links.append(d['href'])
  if tag=='a' and d.get('href') and '.pdf' in d['href'].lower():self.links.append(d['href'])
  if tag=='title':self._title=True
 def handle_endtag(self,tag):
  if tag=='title':self._title=False
 def handle_data(self,data):
  if self._title:self.page_title+=data

def normal(s):return re.sub(r'[^a-z0-9]+',' ',html.unescape(s or '').lower()).strip()

def title_ok(source_title,pdf_text):
 src={w for w in normal(source_title).split() if len(w)>=3 and w not in STOP}
 body=set(normal(pdf_text[:1800]).split())
 if not src:return True
 overlap=len(src & body)
 return overlap>=max(1,(3*len(src)+3)//4)

def extract_abstract(text):
 # Accept only a labeled Abstract section; never substitute the introduction.
 m=re.search(r'(?im)^\s*abstract\s*(?:[.:\-–—]\s*|\r?\n)',text)
 if not m:return ''
 body=text[m.end():]
 stops=[]
 patterns=[r'(?im)^\s*(?:keywords?|key\s+words|index\s+terms|ccs\s+concepts)\b',
          r'(?im)^\s*(?:1\s*\.?\s*)?introduction\b',
          r'(?im)^\s*(?:bug report|background|preliminaries|problem statement|references|bibliography)\s*$',
          r'(?im)^\s*\d+\s*\.?\s+[A-Z][A-Za-z ]{2,35}\s*$']
 for pat in patterns:
  q=re.search(pat,body)
  if q:stops.append(q.start())
 if stops:body=body[:min(stops)]
 body=re.sub(r'(?m)^\s*\d+\s*$', ' ',body)
 body=re.sub(r'\s+',' ',body).strip(' .\t\r\n')
 if len(body)<80:return ''
 return body[:6000]

def fetch_url(url,session):
 r=session.get(url,headers={'User-Agent':UA},timeout=(12,35),allow_redirects=True)
 if r.status_code==429:
  time.sleep(max(2,float(r.headers.get('Retry-After','2'))))
  r=session.get(url,headers={'User-Agent':UA},timeout=(12,35),allow_redirects=True)
 r.raise_for_status()
 if len(r.content)>35_000_000:raise ValueError('response larger than 35 MB')
 return r

def parse_pdf(content,title):
 if not content.startswith(b'%PDF'):return ''
 fd,path=tempfile.mkstemp(suffix='.pdf')
 try:
  with os.fdopen(fd,'wb') as f:f.write(content)
  p=subprocess.run(['pdftotext','-f','1','-l','2','-layout',path,'-'],capture_output=True,text=True,timeout=35)
  text=p.stdout or ''
  if not title_ok(title,text):return ''
  return extract_abstract(text)
 finally:
  try:os.unlink(path)
  except OSError:pass

def run_one(row,open_entry):
 session=requests.Session()
 candidate_list=open_entry.get('candidates',[])
 urls=[]
 for c in candidate_list:
  for u in c.get('fulltext_urls',[]):
   u=html.unescape(u)
   if u and u not in urls:urls.append(u)
 # Direct PDF URLs first; use repository links before publisher landing pages.
 urls.sort(key=lambda u:(0 if '.pdf' in u.lower() else 1,0 if any(x in u for x in ('ceur-ws.org','drops.dagstuhl.de','arxiv.org')) else 1))
 for url in urls[:6]:
  try:
   response=fetch_url(url,session)
   ctype=response.headers.get('Content-Type','').lower()
   if 'pdf' in ctype or response.content.startswith(b'%PDF'):
    for c in candidate_list:
     abstract=parse_pdf(response.content,row.get('title',''))
     if abstract:return {'split':row['split'],'id':row['id'],'method':'openaire_pdf','abstract':abstract,'url':response.url,'title':c.get('title',''),'doi':(c.get('dois') or [''])[0],'error':''}
   elif 'html' in ctype or '<html' in response.text[:500].lower():
    page=MetaParser();page.feed(response.text)
    meta={k:v for k,v in page.meta}
    for key in ('citation_abstract','dc.description','description'):
     value=meta.get(key,'')
     value=html.unescape(re.sub(r'<[^>]+>',' ',value));value=re.sub(r'\s+',' ',value).strip()
     if key in ('citation_abstract','dc.description') and len(value)>=100:
      return {'split':row['split'],'id':row['id'],'method':'openaire_landing_page','abstract':value,'url':response.url,'title':page.page_title,'doi':'','error':''}
    for link in page.links[:3]:
     nested=requests.compat.urljoin(response.url,link)
     try:
      pdf=fetch_url(nested,session)
      for c in candidate_list:
       abstract=parse_pdf(pdf.content,row.get('title',''))
       if abstract:return {'split':row['split'],'id':row['id'],'method':'openaire_pdf','abstract':abstract,'url':pdf.url,'title':c.get('title',''),'doi':(c.get('dois') or [''])[0],'error':''}
     except Exception:pass
  except Exception as exc:
   last_error=str(exc)
 else:
  return {'split':row['split'],'id':row['id'],'method':'openaire_pdf','abstract':'','url':'','title':'','doi':'','error':locals().get('last_error','No labeled abstract found in available full-text URLs')}
 return {'split':row['split'],'id':row['id'],'method':'openaire_pdf','abstract':'','url':'','title':'','doi':'','error':'No labeled abstract found in available full-text URLs'}

def main():
 with open(INPUT,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 openaire=load_latest(OPENAIRE_CACHE); done=load_latest(CACHE)
 targets=[]
 for r in rows:
  key=(r['split'],r['id'])
  if r.get('abstract','').strip() or key in done:continue
  entry=openaire.get(key)
  if not entry or not entry.get('candidates'):continue
  # A repository URL is useful only if the matched record didn't already provide an abstract description.
  if any(c.get('abstract') for c in entry.get('candidates',[])):continue
  if any(c.get('fulltext_urls') for c in entry.get('candidates',[])):targets.append((r,entry))
 print(f'{len(targets)} still-missing rows have OpenAIRE full-text links to check.',flush=True)
 if not targets:return
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(run_one,r,e) for r,e in targets]
  for i,future in enumerate(as_completed(futures),1):
   entry=future.result()
   with open(CACHE,'a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
   if i%10==0 or i==len(futures):
    print(f'  PDF check {i}/{len(futures)}; latest {entry["split"]} id {entry["id"]}: {"abstract" if entry.get("abstract") else "no labeled abstract"}.',flush=True)
 print('Full-text abstract extraction complete.',flush=True)
if __name__=='__main__':main()

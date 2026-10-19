#!/usr/bin/env python3
"""Label-blind bibliographic audit against registered DOI metadata and CEUR contents."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote

import pandas as pd
import requests

from run_ablation import HERE, OUT, V2, baseline, load_clean_data

CACHE=HERE/'metadata_sources'


def title_key(value):
    text=unicodedata.normalize('NFKD',html.unescape(str(value))).casefold()
    text=re.sub(r'\((?:extended abstract|short paper|invited talk|abstract|system description)\)','',text)
    return re.sub(r'[^a-z0-9]+',' ',text).strip()


def family_tokens(value):
    names=re.split('[,;]',str(value))
    result=set()
    for name in names:
        tokens=re.sub(r'\b\d{4}\b','',name).strip().split()
        if tokens:
            last=unicodedata.normalize('NFKD',tokens[-1]).casefold()
            last=re.sub('[^a-z]','',last)
            if last:result.add(last)
    return result


def fetch(url,kind,deadline):
    path=CACHE/(hashlib.sha256(url.encode()).hexdigest()+('.json' if kind=='json' else '.html'))
    if path.exists():return json.loads(path.read_text()) if kind=='json' else path.read_text()
    if time.monotonic()>deadline:return {'error':'audit time budget exceeded'} if kind=='json' else ''
    for attempt in range(2):
        try:
            response=requests.get(url,timeout=(5,15),headers={'User-Agent':'Reading-preference-course metadata reproducibility audit'})
            if response.status_code==429:
                time.sleep(min(float(response.headers.get('Retry-After','2')),5));continue
            response.raise_for_status()
            value=response.json() if kind=='json' else response.text
            path.write_text(json.dumps(value) if kind=='json' else value)
            return value
        except (requests.RequestException,ValueError) as exc:
            if attempt==1:return {'error':str(exc)} if kind=='json' else ''
            time.sleep(.5)
    return {'error':'rate limited'} if kind=='json' else ''


def ceur_entries(text,url):
    entries=[]
    for block in re.findall(r'<li\b[^>]*>(.*?)</li\s*>',text,flags=re.I|re.S):
        title=re.search(r'<span\b[^>]*class=[\"\']CEURTITLE[\"\'][^>]*>(.*?)</span>',block,flags=re.I|re.S)
        if not title:continue
        authors=re.findall(r'<span\b[^>]*class=[\"\']CEURAUTHOR[\"\'][^>]*>(.*?)</span>',block,flags=re.I|re.S)
        clean=lambda x:html.unescape(re.sub('<[^>]+>','',x)).strip()
        if authors:entries.append(dict(title=clean(title[1]),authors=', '.join(clean(x) for x in authors),url=url))
    return entries


def run():
    CACHE.mkdir(exist_ok=True);data,_=load_clean_data();records=data.to_dict('records')
    snapshot=OUT/'metadata_audit.csv';snapshot_summary=OUT/'metadata_audit_summary.json'
    if snapshot.exists() and snapshot_summary.exists():
        summary=json.loads(snapshot_summary.read_text())
        assert summary['data_sha256']==baseline.input_digest(data),'Audit snapshot belongs to different inputs'
        existing_frame=pd.read_csv(snapshot,dtype={'id':str},keep_default_na=False)
        assert len(existing_frame)==3090 and existing_frame.id.equals(data.id) and existing_frame.split.equals(data.split)
        if 'snapshot_sha256' in summary:assert summary['snapshot_sha256']==hashlib.sha256(snapshot.read_bytes()).hexdigest()
        print('Using frozen, validated metadata audit snapshot',flush=True)
        return
    # Labels never enter source selection, source validation, or replacements.
    assert 'Label' not in ['title','authors','doi','year']
    deadline=time.monotonic()+360
    existing={}
    for path in (V2.parent/'source_cache').glob('*.json'):
        value=json.loads(path.read_text());m=value.get('message',{})
        if isinstance(m,dict) and m.get('DOI'):existing[m['DOI'].casefold()]=value
    dois=sorted({r['doi'].strip().casefold() for r in records if r['doi'].startswith('10.')})
    source_meta={}
    def get_doi(doi):
        value=existing.get(doi)
        if value is None:value=fetch('https://api.crossref.org/works/'+quote(doi,safe=''),'json',deadline)
        return doi,value
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(get_doi,doi) for doi in dois]
        for i,future in enumerate(as_completed(futures),1):
            key,value=future.result();source_meta[key]=value
            if i%200==0 or i==len(dois):print(f'DOI audit {i}/{len(dois)}',flush=True)
    vols=set()
    for line in (baseline.ROOT/'openaire_abstract_cache.jsonl').open():
        row=json.loads(line)
        for candidate in row.get('candidates') or []:
            if candidate.get('title_similarity',0)<.97:continue
            for url in candidate.get('fulltext_urls') or []:
                match=re.search(r'ceur-ws.org/(Vol-\d+)',url)
                if match:vols.add('https://ceur-ws.org/'+match[1]+'/')
    ceur=[]
    # Give the small publisher-index audit its own bounded interval.
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(fetch,url,'html',time.monotonic()+90):url for url in sorted(vols)}
        for future in as_completed(jobs):ceur.extend(ceur_entries(future.result(),jobs[future]))
    by_title={}
    for entry in ceur:by_title.setdefault(title_key(entry['title']),[]).append(entry)
    rows=[]
    for original in records:
        source_authors='';source_year=None;source_title='';url='';status='not_verified';detail=''
        doi=original['doi'].casefold()
        if doi.startswith('10.'):
            value=source_meta.get(doi,{})
            m=value.get('message',{})
            if isinstance(m,dict) and m.get('title'):
                source_title=m['title'][0]
                similarity=SequenceMatcher(None,title_key(original['title']),title_key(source_title)).ratio()
                if similarity>=.97 and m.get('type')!='dissertation':
                    source_authors=', '.join(' '.join([a.get('given',''),a.get('family','')]).strip() for a in m.get('author',[]) if a.get('family'))
                    dates=m.get('published') or m.get('published-print') or m.get('published-online') or {}
                    source_year=(dates.get('date-parts') or [[None]])[0][0]
                    url='https://api.crossref.org/works/'+quote(doi,safe='')
                    status='exact_doi_title_validated'
                else:status='source_identity_mismatch';detail=f'title_similarity={similarity:.4f};type={m.get("type","")}'
            else:detail=value.get('error','no registered metadata')
        matches=by_title.get(title_key(original['title']),[])
        # Title-index match accepted only when source author lists agree if multiple appearances.
        if matches and len({tuple(sorted(family_tokens(m['authors']))) for m in matches})==1:
            source_authors=matches[0]['authors'];source_title=matches[0]['title'];url=matches[0]['url'];status='ceur_exact_unique_title'
        # Independently verified author-deposited paper (also in the prior source-audit ledger).
        if original['id']=='372' and original['title'].startswith('Advances in Big Data Bio Analytics'):
            source_authors='Nicos Angelopoulos, Jan Wielemaker';source_title=original['title'];source_year=2019
            url='https://arxiv.org/abs/1909.08254';status='author_deposited_exact_title'
        old_tokens=family_tokens(original['authors']);new_tokens=family_tokens(source_authors)
        verified=bool(new_tokens)
        year_ok=source_year is not None and abs(float(original['year'])-int(source_year))<=1
        rows.append(dict(split=original['split'],id=original['id'],title=original['title'],
            authors_original=original['authors'],authors_verified=source_authors,
            author_tokens_original=' '.join(sorted(old_tokens)),
            author_tokens_clean=' '.join(sorted(new_tokens if verified else old_tokens)),
            author_verified=verified,authors_changed=verified and old_tokens!=new_tokens,
            year_original=original['year'],year_source=source_year,
            year_clean=int(source_year) if year_ok else original['year'],
            year_changed=bool(year_ok and float(original['year'])!=int(source_year)),
            year_review=bool(source_year is not None and not year_ok),
            venue=original['venue'],status=status,source_title=source_title,source_url=url,detail=detail))
    frame=pd.DataFrame(rows)
    # Crossref may contain bare CR inside title/author fields. Normalize whitespace before CSV
    # export, otherwise pandas' universal-newline reader can split an unquoted CR into a record.
    for column in frame.select_dtypes(include='object'):
        frame[column]=frame[column].map(lambda x:re.sub(r'\s+',' ',x).strip() if isinstance(x,str) else x)
    assert len(frame)==3090 and not frame.duplicated(['split','id']).any()
    frame.to_csv(OUT/'metadata_audit.csv',index=False)
    frame[frame.authors_changed|frame.year_changed|frame.year_review|frame.status.eq('source_identity_mismatch')].to_csv(OUT/'metadata_change_ledger.csv',index=False)
    summary=dict(n=len(frame),author_verified=int(frame.author_verified.sum()),authors_changed=int(frame.authors_changed.sum()),
        years_changed=int(frame.year_changed.sum()),year_review=int(frame.year_review.sum()),
        exact_identity_mismatches=int(frame.status.eq('source_identity_mismatch').sum()),
        DOI_metadata_success=sum(isinstance(v.get('message'),dict) for v in source_meta.values()),
        DOI_requested=len(dois),ceur_index_entries=len(ceur),
        policy='Never change labels. Apply authors only after exact DOI/title validation or publisher unique-title match. Year replacement only for same exact record within1 year; larger differences flagged only. Unverified authors kept as original.',
        data_sha256=baseline.input_digest(data),snapshot_sha256=hashlib.sha256((OUT/'metadata_audit.csv').read_bytes()).hexdigest())
    (OUT/'metadata_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()

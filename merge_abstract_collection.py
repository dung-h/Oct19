#!/usr/bin/env python3
"""Merge validated Crossref/OpenAIRE metadata and repository PDF abstracts."""
import csv, html, json, os, re
from collections import Counter,defaultdict
from urllib.parse import quote

ROOT=os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.join(ROOT,'abstracts_completed.csv')
OPENAIRE_CACHE=os.path.join(ROOT,'openaire_abstract_cache.jsonl')
PDF_CACHE=os.path.join(ROOT,'openaire_pdf_abstract_cache.jsonl')
VENUE_BATCH_CACHE=os.path.join(ROOT,'abstract_retrieval_cache.jsonl')
SPRINGER_CACHE=os.path.join(ROOT,'springer_official_cache.jsonl')
VERIFIED_CACHE=os.path.join(ROOT,'verified_external_abstracts.jsonl')
AAAI_CACHE=os.path.join(ROOT,'publisher_pdf_abstracts.jsonl')
OUTPUT=os.path.join(ROOT,'abstracts_collected.csv')
MISSING=os.path.join(ROOT,'abstracts_still_missing.csv')
REPORT=os.path.join(ROOT,'abstract_collection_report.md')

def latest(path):
 out={}
 if not os.path.exists(path):return out
 with open(path,encoding='utf-8') as f:
  for line in f:
   try:
    x=json.loads(line);out[(x['split'],x['id'])]=x
   except Exception:pass
 return out

def write_csv(path,rows,fields):
 with open(path,'w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def classify_title(title):
 """Title-only triage; candidates require review and are not definitive types."""
 t=html.unescape(title or '').strip().lower()
 if re.search(r'\b(proceedings?|front matter|table of contents|list of authors)\b',t):
  return 'proceedings_or_collection_candidate'
 if re.match(r'^\s*\d+(?:st|nd|rd|th)?\.?\s*krypto[- ]tag\b',t):
  return 'event_or_program_record_candidate'
 if ('workshop' in t or 'conference' in t or 'symposium' in t) and re.match(
    r'^\s*(?:\d+(?:st|nd|rd|th)?\.?\s+)?(?:international|joint|european|acm|ieee|aaai|computer aided verification|logic programming and nonmonotonic reasoning|software verification|numerical software verification)',t):
  return 'event_or_program_record_candidate'
 return 'paper_or_short_contribution_candidate'

def main():
 with open(INPUT,newline='',encoding='utf-8-sig') as f:
  rows=list(csv.DictReader(f));fields=list(rows[0])
 oa=latest(OPENAIRE_CACHE);pdf=latest(PDF_CACHE);batch=latest(VENUE_BATCH_CACHE);springer=latest(SPRINGER_CACHE);verified=latest(VERIFIED_CACHE);aaai=latest(AAAI_CACHE)
 add_oa=add_pdf=add_springer=add_aaai=add_verified=add_identity_review=add_this_pass=0
 for row in rows:
  key=(row['split'],row['id'])
  was_missing=not row.get('abstract','').strip()
  src=row.get('abstract_source','')
  if not row.get('abstract','').strip():
   oe=oa.get(key,{})
   candidates=oe.get('candidates',[])
   with_abs=[c for c in candidates if c.get('abstract','').strip()]
   if with_abs:
    c=max(with_abs,key=lambda z:(z.get('title_similarity',0),z.get('author_overlap',0),len(z.get('abstract',''))))
    row['abstract']=c['abstract'];row['abstract_source']='OpenAIRE description';row['retrieval_method']='OpenAIRE publication metadata'
    row['retrieved_doi']=(c.get('dois') or [''])[0];row['retrieval_confidence']='high';row['abstract_url']=(c.get('fulltext_urls') or [''])[0]
    add_oa+=1
   else:
    pe=pdf.get(key,{})
    if pe.get('abstract','').strip():
     row['abstract']=pe['abstract'];row['abstract_source']='OpenAIRE-linked full text (PDF)';row['retrieval_method']='PDF first-page Abstract section'
     row['retrieved_doi']=pe.get('doi','');row['retrieval_confidence']='high';row['abstract_url']=pe.get('url','')
     add_pdf+=1
    else:
     sp=springer.get(key,{})
     if sp.get('abstract','').strip():
      row['abstract']=sp['abstract'];row['abstract_source']='Springer Nature publisher page';row['retrieval_method']='Springer Nature chapter Abstract section'
      row['retrieved_doi']=sp.get('doi','');row['retrieved_title']=sp.get('title','');row['retrieval_confidence']='high';row['abstract_url']=sp.get('url','')
      add_springer+=1
     else:
      aa=aaai.get(key,{})
      if aa.get('abstract','').strip():
       row['abstract']=aa['abstract'];row['abstract_source']=aa.get('source_label','AAAI Proceedings PDF')
       row['retrieval_method']=aa.get('method','AAAI official proceedings PDF, labeled Abstract section')
       row['retrieval_confidence']='high';row['abstract_url']=aa.get('source_url') or aa.get('url','')
       row['retrieved_title']=aa.get('source_title') or aa.get('title','')
       add_aaai+=1
  if row.get('abstract','').strip():
   row['has_abstract']='1'
   row['collection_status']='abstract_found_identity_review' if row.get('identity_review_note') else 'abstract_found'
   if not row.get('retrieval_method'):
    s=row.get('abstract_source','')
    row['retrieval_method']=s or 'Previously collected'
   if not row.get('retrieval_confidence'):row['retrieval_confidence']='existing dataset/API match'
   if not row.get('abstract_url'):
    doi=row.get('retrieved_doi') or row.get('dataset_doi','')
    if doi and doi.lower().startswith('10.'):
     row['abstract_url']='https://doi.org/'+doi
  else:
   row['has_abstract']='0';row['collection_status']='abstract_not_retrieved'
   pe=pdf.get(key,{})
   oe=oa.get(key,{})
   candidates=oe.get('candidates',[])
   if pe:
    err=(pe.get('error') or '').lower()
    row['collection_status']='open_fulltext_unavailable' if any(x in err for x in ('403','forbidden','timeout')) else 'no_labeled_abstract_in_open_pdf'
   elif oe.get('error'):
    row['collection_status']='openaire_query_error'
   elif candidates and any(c.get('fulltext_urls') for c in candidates):
    row['collection_status']='open_fulltext_not_yet_extracted'
   elif candidates:
    row['collection_status']='metadata_match_without_abstract_or_fulltext'
   else:
    row['collection_status']='no_validated_openaire_record'
   b=batch.get(key,{})
   if b.get('candidate') and not row.get('crossref_title'):
    row['crossref_title']=b['candidate'].get('title','')
    row['crossref_year']=b['candidate'].get('year','')
    row['crossref_title_similarity']=str(b.get('title_similarity',''))
  if row.get('abstract_source') and not row.get('retrieval_method'):
   row['retrieval_method']=row['abstract_source']
  # A later, explicitly title-validated primary author/institution page may
  # replace a bad or truncated publisher-page capture for the same record.
  correction=verified.get(key,{})
  if correction.get('abstract','').strip():
   row['abstract']=correction['abstract']
   row['abstract_source']=correction.get('source_label','Verified author/institution page')
   row['retrieval_method']=correction.get('method','Author/institution publication page')
   row['retrieved_doi']=correction.get('doi',row.get('retrieved_doi',''))
   row['retrieval_confidence']=correction.get('confidence','high')
   row['identity_review_note']=correction.get('identity_note','')
   row['abstract_url']=correction.get('url','')
   row['retrieved_title']=correction.get('retrieval_title') or correction.get('title','')
   row['has_abstract']='1';row['collection_status']='abstract_found'
   if correction.get('identity_note'):
    row['collection_status']='abstract_found_identity_review'
   if was_missing:add_verified+=1
   if was_missing and correction.get('retrieval_batch')=='residual_primary_source_pass_2026-10':add_this_pass+=1
   if was_missing and correction.get('identity_note'):add_identity_review+=1
  if row.get('abstract','').strip():
   row['has_abstract']='1'
   row['collection_status']='abstract_found_identity_review' if row.get('identity_review_note') else 'abstract_found'
  row['record_type_candidate']=classify_title(row.get('title',''))
  # append source dataset columns when not already present
  if not row.get('venue') or not row.get('authors'):
   pass
  if not row.get('retrieval_confidence'):row['retrieval_confidence']=''
  if not row.get('retrieved_doi'):row['retrieved_doi']=''
  if not row.get('abstract_url'):row['abstract_url']=''
  if not row.get('retrieval_method'):row['retrieval_method']=''

 fields += [x for x in ['retrieval_method','abstract_url','collection_status','retrieved_title','record_type_candidate','identity_review_note'] if x not in fields]
 # Include dataset columns already appended by the first collection pass.
 write_csv(OUTPUT,rows,fields)
 missing=[r for r in rows if not r.get('abstract','').strip()]
 write_csv(MISSING,missing,fields)
 by_split=Counter(r['split'] for r in rows if r.get('abstract','').strip())
 source_count=Counter(r.get('abstract_source','') for r in rows if r.get('abstract','').strip())
 by_venue=Counter(r.get('venue','') for r in missing)
 by_status=Counter(r.get('collection_status','') for r in missing)
 by_record_type=Counter(r.get('record_type_candidate','') for r in missing)
 total=sum(bool(r.get('abstract','').strip()) for r in rows)
 tr=sum(r['split']=='train' and bool(r.get('abstract','').strip()) for r in rows)
 te=sum(r['split']=='test' and bool(r.get('abstract','').strip()) for r in rows)
 report=[]
 report.append('# Abstract collection report\n')
 report.append('Collection used direct DOI resolution, title and metadata searches in Crossref/OpenAlex/Semantic Scholar/OpenAIRE, official publisher and proceedings records, and author/institution repositories. Matches were checked against title, venue/year, DOI, and source authors where available. PDF extraction uses a labeled Abstract section; for GI Krypto-Tag, the venue defines contributions as submitted abstracts published as citable technical reports, so the report body is extracted with metadata, references, and repository footer removed. Explicitly titled extended abstracts are retained as full extended-abstract text and labeled as such. Publisher-provided invited-talk summaries are labeled separately. Introductions are not substituted for abstracts. Author or publication-version mismatches are flagged for review.\n')
 report.append('## Coverage\n')
 report.append(f'- Total: **{total}/{len(rows)} ({total/len(rows):.2%})**\n- Train: **{tr}/2494 ({tr/2494:.2%})**\n- Test: **{te}/596 ({te/596:.2%})**\n- Recovered in the focused source-review pass: **{add_this_pass}** ({add_this_pass-add_identity_review} without a review note; {add_identity_review} flagged for identity/version review).\n- Manually verified overrides applied from the source cache: **{add_verified}**.\n')
 report.append('## Abstract source counts\n')
 for s,n in source_count.most_common():report.append(f'- {s or "unspecified"}: {n}\n')
 report.append('\n## Remaining rows\n')
 report.append(f'**{len(missing)}** rows remain in [abstracts_still_missing.csv](abstracts_still_missing.csv). This means the abstract was not returned by checked metadata sources or could not be extracted from an accessible full text; it does not establish that the paper has no abstract. Proceedings-volume records are identified separately in that file because they describe a whole collection rather than an individual paper.\n')
 report.append('The `record_type_candidate` field is title-only triage, not a verified publication-type label:\n')
 for k,n in by_record_type.most_common():report.append(f'- {k}: {n}\n')
 report.append('\nA manual source-title audit of the 20 paper/short-contribution candidates found 12 additional records that are actually proceedings or edited-volume titles (three ICLP technical-communications volumes, the KR4HC volume, the FMBC workshop volume, and seven LICS proceedings records). Eight remaining rows appear to be individual papers or talks: `train:898`, `train:1028`, `train:1029`, `train:1030`, `train:1039`, `train:3052`, `test:899`, and `test:1042`. For `train:898` and `test:899`, the CEUR publisher PDFs contain the full contributions but no separately labeled Abstract section, and the dataset author fields conflict with publisher metadata; introductions were not substituted. The four Krypto-Tag records have no attached full-text bundle in the checked GI Digital Library records. `train:3052` has no abstract in the checked Crossref or Semantic Scholar metadata, and the ACM page did not provide retrievable content. The similarly titled VU thesis for `test:1042` has a different author and publication type from the dataset row, so it was not treated as the same work. These are retrieval/identity limits, not evidence that the works have no abstracts.\n')
 for s,n in by_status.most_common():report.append(f'- {s}: {n}\n')
 review_rows=[r for r in rows if r.get('identity_review_note')]
 report.append('\n## Identity review\n')
 report.append(f'{len(review_rows)} collected abstract(s) carry an identity or source-version review note, most often because dataset authors conflict with the matched publication record. See `identity_review_note` in [abstracts_collected.csv](abstracts_collected.csv). No validated source candidates remain withheld from coverage; the audit schema is retained in [abstract_identity_review.csv](abstract_identity_review.csv).\n')
 report.append('\n## Remaining rows by venue\n')
 for v,n in by_venue.most_common():report.append(f'- {v or "(blank)"}: {n}\n')
 report.append('\nSources used in these passes include the [GI Krypto-Tag description](https://fg-krypto.gi.de/krypto-tag), [Dagstuhl LIPIcs TIME 2023 record](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2023.12), [AAAI KR 2018 proceedings paper](https://cdn.aaai.org/ocs/17987/17987-78707-1-PB.pdf), [Rutgers KR 2018 extended-abstract record](https://www.researchwithnj.com/en/publications/on-limited-conjunctions-in-polynomial-feature-logics-with-applica-2), [CEUR KR4L plenary-talk PDF](https://ceur-ws.org/Vol-3020/KR4L_paper_2.pdf), [Springer Nature 2019 journal version](https://link.springer.com/article/10.1007/s10472-019-09656-4), [Springer LPNMR 2024 volume](https://link.springer.com/book/10.1007/978-3-031-74209-5), [Springer LPNMR 2022 chapter](https://link.springer.com/chapter/10.1007/978-3-031-15707-3_18), [Springer CAV workshop chapter](https://link.springer.com/chapter/10.1007/978-3-031-21222-2_11), [ICLP 2025 EPTCS proceedings](https://cgi.cse.unsw.edu.au/~eptcs/Published/ICLP2025/Proceedings.pdf), [Dagstuhl ICLP 2018 proceedings](https://drops.dagstuhl.de/entities/volume/OASIcs-volume-64), and source links recorded with each recovered row. A missing metadata-index abstract does not establish that the publication itself lacks an abstract.\n')
 with open(REPORT,'w',encoding='utf-8') as f:f.writelines(report)
 print(f'Added {add_oa} OpenAIRE descriptions, {add_pdf} PDF abstracts, {add_springer} Springer abstracts, {add_aaai} publisher/author PDF abstracts, and {add_verified} verified institutional/author records ({add_identity_review} flagged for identity review).')
 print(f'Coverage: {total}/{len(rows)} = {total/len(rows):.2%}; train {tr}/2494 = {tr/2494:.2%}; test {te}/596 = {te/596:.2%}; remaining {len(missing)}.')
 print('source counts:',dict(source_count))
 print('remaining by status:',dict(by_status))
 print('remaining by venue:',dict(by_venue))
 print('remaining by title-only record-type candidate:',dict(by_record_type))
 print('wrote',OUTPUT,MISSING,REPORT)
if __name__=='__main__':main()

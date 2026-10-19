#!/usr/bin/env python3
"""Continue abstract retrieval with DOI/title validation and persistent checkpoints."""
import csv, html, io, json, os, re, threading, time, unicodedata, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from urllib.parse import quote
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(ROOT, 'assignment-data-mining-261.zip')
INPUT_PATH = os.path.join(ROOT, 'abstracts_multisource.csv')
CACHE_PATH = os.path.join(ROOT, 'abstract_retrieval_cache.jsonl')
OUTPUT_PATH = os.path.join(ROOT, 'abstracts_completed.csv')
UA = 'Academic paper abstract retrieval for a data-mining assignment'


def parse_test_rtf(raw):
    s = raw.decode('latin1')
    start = s.find('\\cf0 ')
    if start < 0:
        raise ValueError('RTF CSV payload marker \\cf0 not found')
    s = s[start + len('\\cf0 '):]
    s = re.sub(r'\\\r?\n', '\n', s)
    s = re.sub(r'\\u(-?\d+) ?(?:\?)?', lambda m: chr(int(m.group(1)) % 65536), s)
    s = s.replace('\\uc0', '')
    s = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]).decode('cp1252'), s)
    s = s.replace('\\{', '{').replace('\\}', '}').replace('\\\\', '\\')
    s = '\n'.join(line[:-1] if line.endswith('\\') else line for line in s.splitlines())
    s = s.rsplit('}', 1)[0]
    return list(csv.DictReader(io.StringIO(s)))


def norm(value):
    value = html.unescape(value or '').lower().replace('&amp;', ' and ')
    value = ''.join(ch for ch in unicodedata.normalize('NFKD', value) if not unicodedata.combining(ch))
    value = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', value)).strip()
    return value


def base_title(value):
    value = html.unescape(value or '').strip()
    value = re.sub(r'\s*\((?:short paper|invited talk|abstract|extended abstract|system description|poster|demo)\)\s*\.?$', '', value, flags=re.I)
    value = re.sub(r'\s*(?:[-:]\s*)?(?:extended\s+)?abstract\s*\.?$', '', value, flags=re.I)
    return value.rstrip(' .')


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def clean_abstract(value):
    if not value:
        return ''
    value = re.sub(r'<[^>]+>', ' ', str(value))
    value = html.unescape(value)
    return re.sub(r'\s+', ' ', value).strip()


def parse_crossref_item(item):
    return {
        'doi': item.get('DOI', ''),
        'title': (item.get('title') or [''])[0],
        'year': str(((item.get('published') or item.get('published-print') or item.get('published-online') or {}).get('date-parts') or [['']])[0][0]),
        'abstract': clean_abstract(item.get('abstract', '')),
        'authors': [a.get('family', '') for a in item.get('author', []) if a.get('family')],
        'score': item.get('score'),
    }


def reconstruct_openalex(inv):
    if not inv:
        return ''
    positions = {}
    for token, idxs in inv.items():
        for idx in idxs:
            positions.setdefault(int(idx), token)
    if not positions:
        return ''
    return ' '.join(positions.get(i, '') for i in range(max(positions) + 1)).strip()


def clean_doi(value):
    value = (value or '').strip()
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value, flags=re.I)
    value = re.sub(r'^doi\s*:\s*', '', value, flags=re.I)
    value = value.rstrip(' .')
    return value if re.match(r'^10\.\d{4,9}/\S+$', value, re.I) else ''


def load_sources():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        train = list(csv.DictReader(io.TextIOWrapper(zf.open('train.csv'), encoding='utf-8-sig', newline='')))
        test = parse_test_rtf(zf.read('Test.csv'))
    if len(train) != 2494 or len(test) != 596:
        raise ValueError(f'Unexpected source row count: train={len(train)} test={len(test)}')
    sources = {('train', r['id']): r for r in train}
    sources.update({('test', r['id']): r for r in test})
    return sources


def load_cache():
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            for line in f:
                try:
                    v = json.loads(line)
                    cache[(v['split'], v['id'], v['method'])] = v
                except Exception:
                    continue
    return cache


def save_cache(entry):
    with open(CACHE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def get_json(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers={'User-Agent': UA}, timeout=30)
            if response.status_code == 429:
                delay = float(response.headers.get('Retry-After', '5'))
                time.sleep(max(2.0, delay))
                continue
            response.raise_for_status()
            return response.json(), None
        except Exception as exc:
            if attempt + 1 == retries:
                return None, str(exc)
            time.sleep(1.5 * (attempt + 1))
    return None, 'retry limit reached'


def openalex_lookup(doi):
    # The shared unauthenticated OpenAlex budget was exhausted in this session.
    # Preserve this route for a later rerun after reset, without retrying 429s now.
    return None, 'OpenAlex shared unauthenticated quota exhausted; skipped'
    url = 'https://api.openalex.org/works/https://doi.org/' + doi
    data, error = get_json(url, {'select': 'doi,title,publication_year,authorships,abstract_inverted_index'})
    if error or not data:
        return None, error or 'empty response'
    return {
        'doi': (data.get('doi') or '').removeprefix('https://doi.org/'),
        'title': data.get('title', ''),
        'year': str(data.get('publication_year') or ''),
        'abstract': reconstruct_openalex(data.get('abstract_inverted_index')),
        'authors': [((a.get('author') or {}).get('display_name') or '') for a in data.get('authorships', [])],
    }, None


def crossref_lookup(record, method, title_query=None, doi_query=None):
    headers = {'User-Agent': UA}
    if doi_query:
        url = 'https://api.crossref.org/works/' + quote(doi_query, safe='/')
        params = None
    else:
        url = 'https://api.crossref.org/works'
        params = {'query.title': title_query or record['title'], 'rows': 10,
                  'select': 'DOI,title,author,published,abstract,score'}
    data, error = get_json(url, params)
    if error:
        return {'split': record['split'], 'id': record['id'], 'method': method, 'error': error, 'candidates': []}
    message = data.get('message', {})
    items = [message] if doi_query and isinstance(message, dict) else message.get('items', [])
    candidates = [parse_crossref_item(item) for item in items]
    for cand in candidates:
        cand['title_similarity'] = similarity(base_title(record['title']), base_title(cand.get('title', '')))
    return {'split': record['split'], 'id': record['id'], 'method': method, 'error': '', 'candidates': candidates}


def author_overlap(source_authors, candidate_authors):
    src = {norm(x).split()[-1] for x in (source_authors or '').split(',') if norm(x)}
    cand = {norm(x).split()[-1] for x in (candidate_authors or []) if norm(x)}
    if not src or not cand:
        return 0.0
    return len(src & cand) / max(1, min(len(src), len(cand)))


def select_candidate(record, candidates, direct_doi=''):
    expected_doi = clean_doi(direct_doi).lower()
    ranked = []
    for cand in candidates:
        sim = max(cand.get('title_similarity', 0), similarity(base_title(record['title']), base_title(cand.get('title', ''))))
        doi_match = bool(expected_doi and clean_doi(cand.get('doi', '')).lower() == expected_doi)
        author_score = author_overlap(record.get('authors', ''), cand.get('authors', []))
        confidence = 'high' if doi_match and sim >= .90 or sim >= .97 else ('medium' if sim >= .92 and author_score >= .5 else '')
        if confidence:
            ranked.append((sim, author_score, doi_match, confidence, cand))
    ranked.sort(key=lambda t: (t[2], t[0], t[1]), reverse=True)
    return ranked[0] if ranked else None


def main():
    sources = load_sources()
    with open(INPUT_PATH, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
        old_fields = list(rows[0])
    missing = [r for r in rows if not r.get('abstract', '').strip()]
    for r in rows:
        r['_source'] = sources[(r['split'], r['id'])]
        r['_doi'] = clean_doi(r['_source'].get('doi', ''))
        r['_source_title'] = r['_source'].get('title', '')
    print(f'Loaded {len(rows)} records; {len(missing)} currently lack abstracts.', flush=True)
    print(f'Among missing: {sum(bool(r["_doi"]) for r in missing)} have a direct DOI; {sum(not r["_doi"] for r in missing)} need title lookup.', flush=True)

    cache = load_cache()
    # Pass 1: DOI lookups in OpenAlex, parallelized because these are direct record reads.
    doi_rows = [r for r in missing if r['_doi'] and ('train', r['id'], 'openalex_doi') not in cache and ('test', r['id'], 'openalex_doi') not in cache]
    print(f'OpenAlex DOI pass: {len(doi_rows)} requests.', flush=True)
    def oa_task(r):
        result, error = openalex_lookup(r['_doi'])
        return {'split': r['split'], 'id': r['id'], 'method': 'openalex_doi', 'error': error or '', 'candidate': result}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(oa_task, r) for r in doi_rows]
        for i, fut in enumerate(as_completed(futures), 1):
            entry = fut.result()
            save_cache(entry); cache[(entry['split'], entry['id'], entry['method'])] = entry
            if i % 25 == 0 or i == len(futures):
                print(f'  OpenAlex DOI {i}/{len(futures)}', flush=True)

    # Pass 2: Crossref direct DOI lookups when OpenAlex did not return an abstract.
    doi_fallback = []
    for r in missing:
        if not r['_doi']:
            continue
        oa = cache.get((r['split'], r['id'], 'openalex_doi'), {})
        if not ((oa.get('candidate') or {}).get('abstract') or '').strip() and (r['split'], r['id'], 'crossref_doi') not in cache:
            doi_fallback.append(r)
    print(f'Crossref direct DOI fallback: {len(doi_fallback)} requests (1 request/second per Crossref limit).', flush=True)
    last_start = 0.0
    for i, r in enumerate(doi_fallback, 1):
        delay = 1.05 - (time.monotonic() - last_start)
        if delay > 0:
            time.sleep(delay)
        last_start = time.monotonic()
        entry = crossref_lookup(r, 'crossref_doi', doi_query=r['_doi'])
        save_cache(entry); cache[(r['split'], r['id'], entry['method'])] = entry
        if i % 25 == 0 or i == len(doi_fallback):
            print(f'  Crossref DOI {i}/{len(doi_fallback)}', flush=True)

    # Pass 3: Crossref exact-title search only for rows without a DOI.
    title_rows = []
    for r in missing:
        if r['_doi']:
            continue
        key = (r['split'], r['id'])
        batch = cache.get((key[0], key[1], 'crossref_venue_batch'))
        if batch and batch.get('candidate'):
            continue
        cached_title = cache.get((key[0], key[1], 'crossref_title'))
        if not cached_title or cached_title.get('error'):
            title_rows.append(r)
    print(f'Crossref title pass: {len(title_rows)} requests (1 request/second per Crossref limit).', flush=True)
    last_start = 0.0
    for i, r in enumerate(title_rows, 1):
        delay = 1.05 - (time.monotonic() - last_start)
        if delay > 0:
            time.sleep(delay)
        last_start = time.monotonic()
        query = base_title(r['_source_title'])
        entry = crossref_lookup(r, 'crossref_title', title_query=query)
        save_cache(entry); cache[(r['split'], r['id'], entry['method'])] = entry
        if i % 25 == 0 or i == len(title_rows):
            print(f'  Crossref title {i}/{len(title_rows)}', flush=True)

    # Candidate DOIs from exact Crossref title hits can be checked against OpenAlex.
    candidate_rows = {}
    for r in missing:
        for method in ('crossref_doi', 'crossref_title'):
            entry = cache.get((r['split'], r['id'], method))
            if not entry:
                continue
            direct = r['_doi'] if method == 'crossref_doi' else ''
            selected = select_candidate(r['_source'], entry.get('candidates', []), direct)
            if selected and not selected[4].get('abstract') and selected[4].get('doi'):
                candidate_rows[(r['split'], r['id'])] = (r, selected[4])
    oa_candidate_rows = [(r, c) for key, (r, c) in candidate_rows.items() if (key[0], key[1], 'openalex_crossref_doi') not in cache]
    print(f'OpenAlex follow-up for title-validated Crossref DOIs: {len(oa_candidate_rows)} requests.', flush=True)
    def oa_cand_task(pair):
        r, c = pair
        cand, error = openalex_lookup(clean_doi(c['doi']))
        return {'split': r['split'], 'id': r['id'], 'method': 'openalex_crossref_doi', 'error': error or '', 'candidate': cand}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(oa_cand_task, pair) for pair in oa_candidate_rows]
        for i, fut in enumerate(as_completed(futures), 1):
            entry = fut.result()
            save_cache(entry); cache[(entry['split'], entry['id'], entry['method'])] = entry
            if i % 25 == 0 or i == len(futures):
                print(f'  OpenAlex follow-up {i}/{len(futures)}', flush=True)

    # Merge high-confidence abstracts; title and author checks are required for title-only matches.
    additions = 0
    for r in missing:
        accepted = None
        # OpenAlex direct DOI response is identity-safe when the displayed title agrees.
        oa_entry = cache.get((r['split'], r['id'], 'openalex_doi'), {})
        oa = oa_entry.get('candidate') or {}
        if oa.get('abstract') and similarity(base_title(r['_source_title']), base_title(oa.get('title', ''))) >= .90:
            accepted = (oa, 'OpenAlex via dataset DOI', 'high')
        if not accepted:
            for method in ('crossref_doi', 'crossref_title'):
                entry = cache.get((r['split'], r['id'], method))
                if entry:
                    selected = select_candidate(r['_source'], entry.get('candidates', []), r['_source'].get('doi', '') if method == 'crossref_doi' else '')
                    if selected and selected[4].get('abstract'):
                        accepted = (selected[4], 'Crossref', selected[3]); break
        if not accepted:
            batch = cache.get((r['split'], r['id'], 'crossref_venue_batch'), {})
            candidate = batch.get('candidate') or {}
            if candidate.get('abstract'):
                confidence = batch.get('confidence', 'high')
                accepted = (candidate, 'Crossref', confidence)
        if not accepted:
            entry = cache.get((r['split'], r['id'], 'openalex_crossref_doi'), {})
            oa = entry.get('candidate') or {}
            if oa.get('abstract') and similarity(base_title(r['_source_title']), base_title(oa.get('title', ''))) >= .92:
                accepted = (oa, 'OpenAlex via verified Crossref DOI', 'high')
        if accepted:
            cand, source, confidence = accepted
            r['abstract'] = clean_abstract(cand.get('abstract', ''))
            r['has_abstract'] = '1' if r['abstract'] else '0'
            r['abstract_source'] = source
            r['matched'] = '1'
            r['api_title'] = cand.get('title', '')
            r['api_year'] = cand.get('year', '')
            r['title_similarity'] = f'{similarity(base_title(r["_source_title"]), base_title(cand.get("title", ""))):.4f}'
            r['retrieval_confidence'] = confidence
            r['retrieved_doi'] = cand.get('doi', '')
            additions += 1

    extra_fields = ['venue', 'authors', 'dataset_doi', 'retrieval_confidence', 'retrieved_doi']
    fields = old_fields + [f for f in extra_fields if f not in old_fields]
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            src = r['_source']
            r['venue'] = src.get('venue', '')
            r['authors'] = src.get('authors', '')
            r['dataset_doi'] = src.get('doi', '')
            if 'retrieval_confidence' not in r:
                r['retrieval_confidence'] = ''
            if 'retrieved_doi' not in r:
                r['retrieved_doi'] = ''
            writer.writerow(r)
    total = sum(bool(r.get('abstract', '').strip()) for r in rows)
    print(f'Done: added {additions}; coverage {total}/{len(rows)} ({total/len(rows):.2%}); remaining {len(rows)-total}.', flush=True)
    print(f'Output: {OUTPUT_PATH}', flush=True)

if __name__ == '__main__':
    main()

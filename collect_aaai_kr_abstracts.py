#!/usr/bin/env python3
"""Extract title-validated abstracts from official AAAI KR 2016 PDFs."""
import html, json, os, re, subprocess, tempfile, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(ROOT, 'aaai_kr2016_pdf_index.jsonl')
EXTRA_INDEX = os.path.join(ROOT, 'aaai_kr2016_extra_urls.jsonl')
KR2018_INDEX = os.path.join(ROOT, 'aaai_kr2018_pdf_index.jsonl')
KR2018_INDEX2 = os.path.join(ROOT, 'aaai_kr2018_pdf_index_batch2.jsonl')
KR2018_INDEX3 = os.path.join(ROOT, 'aaai_kr2018_pdf_index_batch3.jsonl')
OLD_CACHE = os.path.join(ROOT, 'aaai_kr2016_abstracts.jsonl')
CACHE = os.path.join(ROOT, 'publisher_pdf_abstracts.jsonl')
UA = 'Academic paper abstract retrieval; official AAAI proceedings PDF'


def normalize(value):
    value = html.unescape(value or '').lower()
    value = ''.join(ch for ch in unicodedata.normalize('NFKD', value)
                    if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+', ' ', value).strip()


def extract(record, session):
    response = session.get(record['source_url'], headers={'User-Agent': UA}, timeout=(8, 18))
    response.raise_for_status()
    if not response.content.startswith(b'%PDF'):
        raise ValueError('Official URL did not return a PDF')
    fd, path = tempfile.mkstemp(suffix='.pdf')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(response.content)
        completed = subprocess.run(
            ['pdftotext', '-f', '1', '-l', '1', '-raw', path, '-'],
            capture_output=True, text=True, timeout=30, check=True)
        page = completed.stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    aliases = [record['title']] + record.get('title_aliases', [])
    overlaps = []
    page_tokens = set(normalize(page).split())
    for title in aliases:
        title_tokens = [word for word in normalize(title).split() if len(word) > 2]
        overlaps.append(sum(word in page_tokens for word in title_tokens) / max(1, len(title_tokens)))
    overlap = max(overlaps, default=0.0)
    if overlap < .75:
        raise ValueError(f'PDF title mismatch ({overlap:.0%} token overlap)')

    match = re.search(r'(?im)^\s*abstract\s*\n', page)
    if not match:
        raise ValueError('No labeled Abstract section on page 1')
    body = page[match.end():]
    stop = re.search(
        r'(?im)^\s*(?:keywords?\b|(?:\d+\s*\.?\s*)?introduction\b|copyright\b)',
        body)
    if stop:
        body = body[:stop.start()]
    body = re.sub(r'(?<=[A-Za-z])[-\u2010\u2011]\s*\n\s*(?=[a-z])', '', body)
    body = re.sub(r'\s+', ' ', body).strip(' \t\r\n.-')
    if not 80 <= len(body) <= 5000:
        raise ValueError(f'Abstract section length outside expected range ({len(body)})')
    return body, overlap


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    records = {}
    for record in (read_jsonl(INDEX) + read_jsonl(EXTRA_INDEX) +
                   read_jsonl(KR2018_INDEX) + read_jsonl(KR2018_INDEX2) +
                   read_jsonl(KR2018_INDEX3) +
                   read_jsonl(OLD_CACHE) + read_jsonl(CACHE)):
        if record.get('split') and record.get('id'):
            records[(record['split'], str(record['id']))] = record
    todo = []
    for key, record in records.items():
        url = record.get('source_url') or record.get('url')
        if not url:
            continue
        if record.get('abstract') and not record.get('error'):
            continue
        if record.get('error') and not record.get('abstract'):
            continue
        record['source_url'] = url
        record['url'] = url
        todo.append((key, record))

    def work(record):
        session = requests.Session()
        try:
            record['abstract'], record['title_validation'] = extract(record, session)
            record['method'] = 'Official/author PDF, labeled Abstract section'
            record['source_label'] = record.get('source_label') or 'AAAI Proceedings PDF'
            record['error'] = ''
        except Exception as exc:
            record['abstract'] = ''
            record['error'] = str(exc)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(work, record): key for key, record in todo}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            records[key] = future.result()
            record = records[key]
            if record.get('abstract'):
                print(f"OK {key[0]}:{key[1]} ({len(record['abstract'])} chars): {record['title']}")
            else:
                print(f"SKIP {key[0]}:{key[1]}: {record['error']}: {record['title']}")
            if i % 8 == 0 or i == len(futures):
                with open(CACHE, 'w', encoding='utf-8') as stream:
                    for item in records.values():
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f'Checkpoint {i}/{len(futures)}.')
    with open(CACHE, 'w', encoding='utf-8') as stream:
        for record in records.values():
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Wrote {CACHE} ({sum(bool(x.get("abstract")) for x in records.values())} validated abstracts).')


if __name__ == '__main__':
    main()

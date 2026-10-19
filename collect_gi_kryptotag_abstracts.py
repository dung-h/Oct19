#!/usr/bin/env python3
"""Extract Krypto-Tag submitted-abstract text from official GI Digital Library reports.

The GI Krypto-Tag explains that contributions are submitted abstracts published
as citable technical reports. These reports do not consistently contain a
separate "Abstract" heading, so this script extracts their contribution text,
omitting the author/venue header, references, and repository footer.
"""
import argparse
import csv
import html
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.abspath(__file__))
MISSING = os.path.join(ROOT, "abstracts_still_missing.csv")
CACHE = os.path.join(ROOT, "gi_kryptotag_text_cache.jsonl")
OUTPUT = os.path.join(ROOT, "verified_external_abstracts.jsonl")
BATCH = "residual_primary_source_pass_2026-10"


def norm(text):
    text = html.unescape(text or "").casefold().replace("ø", "o").replace("ω", "omega")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # Treat German transliterations as equivalent when comparing author surnames.
    return re.sub(r"\s+", " ", text.replace("ae", "a").replace("oe", "o").replace("ue", "u")).strip()


def dataset_surnames(authors):
    out = set()
    for author in (authors or "").split(","):
        words = norm(author).split()
        if words:
            out.add(words[-1])
    return out


def source_surnames(authors):
    out = set()
    for author in authors or []:
        surname = norm(author.split(",", 1)[0]).split()
        if surname:
            out.add(surname[-1])
    return out


def is_header(line):
    s = norm(line)
    if not s:
        return True
    if len(s.split()) < 5:
        return True
    return bool(re.search(
        r"\b(department|university|universitat|institute|institut|school|faculty|"
        r"laboratory|lab|gmbh|company|research group|germany|frankfurt|munich|"
        r"darmstadt|berlin|dresden|hagen|renningen|technology|technische|"
        r"website|http|www|email|mail|\.com|\.de|\.edu)\b", s
    ))


def body_start(lines, source_authors):
    # The report's venue/date line gives the cleanest boundary when present.
    event = [i for i, line in enumerate(lines[:16]) if re.search(
        r"\b(?:crypto\s+day|krypto[- ]?tag)\b", line, re.I
    )]
    if event:
        return event[0] + 1
    # A small number of reports use a dated header instead of naming the event.
    dated = [i for i, line in enumerate(lines[:12]) if re.match(
        r"(?i)^\s*(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|"
        r"august|september|october|november|december)\s+\d{1,2},?\s+20\d{2}\s*$", line
    )]
    if dated:
        return dated[0] + 1

    # Some reports omit the venue line. Skip the title/author/affiliation block
    # and start at the first long prose line after those metadata lines.
    surnames = source_surnames(source_authors)
    for i, line in enumerate(lines[:30]):
        s = norm(line)
        words = set(s.split())
        if words & surnames:
            continue
        if i < 2 or is_header(line):
            continue
        if len(s.split()) >= 9 and len(s) >= 55:
            return i
    return 0


def extract_report_body(text, source_authors):
    text = (text or "").replace("\ufeff", "").replace("\xad", "")
    # Join words split at line endings in the repository's PDF-to-text bundle.
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\r?\n[ \t]*(?=[a-z])", "", text)
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    start = body_start(lines, source_authors)
    body = lines[start:]

    # Submitted contribution text may have references or a repository footer;
    # neither is part of the abstract body.
    stop = len(body)
    for i, line in enumerate(body):
        if re.match(r"(?i)^\s*(?:references|bibliography|literature cited)\s*:?\s*$", line):
            stop = min(stop, i)
        if re.search(r"(?i)^\s*(?:crypto day matters|beitr[aä]ge vom|doi\s*:)\b", line):
            stop = min(stop, i)
    body = body[:stop]
    body = [line for line in body if not re.fullmatch(r"\d+", line)]
    value = re.sub(r"\s+", " ", " ".join(body)).strip()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="append verified rows to the override file")
    args = parser.parse_args()

    with open(MISSING, newline="", encoding="utf-8-sig") as f:
        missing = {(r["split"], r["id"]): r for r in csv.DictReader(f)}
    cache = [json.loads(line) for line in open(CACHE, encoding="utf-8") if line.strip()]
    existing = set()
    if os.path.exists(OUTPUT):
        for line in open(OUTPUT, encoding="utf-8"):
            try:
                x = json.loads(line)
                existing.add((x["split"], str(x["id"])))
            except Exception:
                pass

    additions = []
    for source in cache:
        key = (source["split"], str(source["id"]))
        target = missing.get(key)
        if not target or key in existing:
            continue
        if str(target.get("year", "")) != str(source.get("source_year", "")):
            continue
        if SequenceMatcher(None, norm(target.get("title", "")), norm(source.get("source_title", ""))).ratio() < 0.97:
            continue
        text = next((f.get("content", "") for f in source.get("files", []) if f.get("kind") == "text"), "")
        abstract = extract_report_body(text, source.get("source_authors", []))
        if len(abstract) < 80 or not source.get("doi"):
            continue

        ds = dataset_surnames(target.get("authors", ""))
        ss = source_surnames(source.get("source_authors", []))
        conflict = bool(ds and ss and not (ds & ss))
        identity_note = ""
        confidence = "high"
        if conflict:
            confidence = "medium"
            identity_note = (
                "Exact title, year and official GI Digital Library DOI match this Krypto-Tag report, "
                "but the dataset author field does not overlap the report authors; retained with identity review."
            )

        additions.append({
            "split": source["split"],
            "id": str(source["id"]),
            "method": "GI Digital Library Krypto-Tag technical-report text; submitted contribution body extracted, with header, references and repository footer removed",
            "source_label": "GI Digital Library Krypto-Tag technical report (submitted abstract)",
            "doi": source["doi"],
            "retrieval_title": source["source_title"],
            "url": "https://doi.org/" + source["doi"],
            "abstract": abstract,
            "confidence": confidence,
            "identity_note": identity_note,
            "retrieval_batch": BATCH,
        })

    print(f"Validated {len(additions)} source-text candidates; {sum(bool(x['identity_note']) for x in additions)} have author-field conflicts.")
    for item in additions:
        print(f"{item['split']}:{item['id']} {item['confidence']} {len(item['abstract'])} chars — {item['retrieval_title']}")
        print("  ", item["abstract"][:220])
    if args.apply:
        with open(OUTPUT, "a", encoding="utf-8") as f:
            for item in additions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Appended {len(additions)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()

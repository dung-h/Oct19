#!/usr/bin/env python3
"""Resumable, rate-respectful Crossref direct-DOI reference collector."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
CACHE = OUT / "crossref_raw.jsonl"
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "reference_graph_audit")]
from collect_remaining_abstracts import load_sources
import run as audit

UA = "Personalized-reading-reference-ablation/1.0"


def records() -> pd.DataFrame:
    sources = load_sources()
    frame = pd.DataFrame([dict(split=split, **row) for (split, _), row in sources.items()])
    frame.id = frame.id.astype(str)
    frame["doi_canonical"] = frame.doi.map(audit.canonical_doi)
    return frame[frame.doi_canonical.ne("")].copy()


def cache() -> dict[str, dict]:
    values = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf8").splitlines():
            item = json.loads(line)
            values[item["doi"]] = item
    return values


def fetch(doi: str, identities: list[dict]) -> dict:
    item = dict(doi=doi, retrieved_at=datetime.now(timezone.utc).isoformat(),
                requested_records=[dict(split=x["split"], id=str(x["id"]), title=x["title"]) for x in identities])
    try:
        response = requests.get("https://api.crossref.org/works/" + quote(doi, safe="/"),
            headers={"User-Agent": UA}, timeout=30)
        item["http_status"] = response.status_code
        if response.ok:
            item["message"] = response.json().get("message", {})
        else:
            item["error"] = response.text[:300]
    except requests.RequestException as exc:
        item.update(http_status=0, error=f"{type(exc).__name__}: {str(exc)[:180]}")
    return item


def collect(limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = records(); saved = cache()
    groups = [(doi, part.to_dict("records")) for doi, part in frame.groupby("doi_canonical", sort=True)]
    pending = [(doi, rows) for doi, rows in groups if doi not in saved or saved[doi].get("http_status") in (0, 429, 500, 502, 503, 504)]
    if limit is not None:
        pending = pending[:limit]
    print(f"distinct_dois={len(groups)} cached={len(saved)} pending={len(pending)}", flush=True)
    previous = 0.0
    with CACHE.open("a", encoding="utf8") as sink:
        for number, (doi, identities) in enumerate(pending, 1):
            wait = 1.05 - (time.monotonic() - previous)
            if wait > 0:
                time.sleep(wait)
            previous = time.monotonic()
            item = fetch(doi, identities)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n"); sink.flush()
            if number % 25 == 0 or number == len(pending) or item["http_status"] != 200:
                print(f"{number}/{len(pending)} status={item['http_status']} doi={doi}", flush=True)
            if item["http_status"] == 429:
                print("stopping on Crossref rate limit; rerun resumes from cache", flush=True)
                return


def norm(value: str) -> str:
    return audit.norm(value)


def materialize() -> None:
    frame = records(); saved = cache(); rows = []
    for record in frame.to_dict("records"):
        item = saved.get(record["doi_canonical"])
        message = (item or {}).get("message", {})
        returned = audit.canonical_doi(message.get("DOI", ""))
        title = (message.get("title") or [""])[0]
        similarity = __import__("difflib").SequenceMatcher(None, norm(record["title"]), norm(title)).ratio() if title else 0.0
        accepted = bool((item or {}).get("http_status") == 200 and returned == record["doi_canonical"] and similarity >= .90)
        references = sorted({audit.canonical_doi(ref.get("DOI", "")) for ref in message.get("reference", []) if audit.canonical_doi(ref.get("DOI", ""))}) if accepted else []
        status = "ACCEPTED" if accepted else ("NOT_REQUESTED" if item is None else "REQUEST_FAILED" if item.get("http_status") != 200 else "IDENTITY_CONFLICT")
        rows.append(dict(split=record["split"], id=record["id"], doi=record["doi_canonical"], identity_status=status,
            title_similarity=similarity, references_available=bool(references), references_count=len(references), references=references))
    ledger = pd.DataFrame(rows)
    ledger.to_json(OUT / "doi_reference_ledger.jsonl", orient="records", lines=True, force_ascii=False)
    summary = ledger.groupby("split").agg(n=("id", "size"), accepted=("identity_status", lambda x: int((x == "ACCEPTED").sum())),
        with_references=("references_available", "sum"), median_references=("references_count", "median"))
    summary.to_csv(OUT / "coverage_summary.csv")
    value = dict(records=len(ledger), cached_dois=len(saved), accepted=int((ledger.identity_status == "ACCEPTED").sum()),
        with_references=int(ledger.references_available.sum()), status_counts=ledger.identity_status.value_counts().to_dict(),
        source_sha256=hashlib.sha256((OUT / "doi_reference_ledger.jsonl").read_bytes()).hexdigest())
    (OUT / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2)); print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["collect", "materialize"]); parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    collect(args.limit) if args.stage == "collect" else materialize()

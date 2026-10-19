#!/usr/bin/env python3
"""Audit whether public full-text sources are sufficiently available to justify a new feature pipeline.

This script does not download PDFs, create labels, or train a model.  It asks
Semantic Scholar only for stable bibliographic fields and an open-access PDF
pointer for a deterministic, stratified DOI sample.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "abstract_preference_margin"))
import apm_context as ac

SEED = 20261102
FIELDS = "title,year,venue,openAccessPdf,publicationTypes"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def label_value(value) -> str | int:
    return "" if pd.isna(value) or str(value).strip() == "" else int(value)


def select_sample(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["doi"] = frame.doi.fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame.doi.ne("")].copy()
    rng = np.random.default_rng(SEED)
    selected = []
    # Training is stratified by venue and known class solely to measure source
    # coverage; labels are not sent to the API nor used to infer preferences.
    for _, group in frame[frame.split.eq("train")].groupby(["venue", "Label"], sort=True, dropna=False):
        n = min(8, len(group))
        selected.extend(rng.choice(group.index.to_numpy(), size=n, replace=False).tolist())
    # Test has no labels, so it is stratified only by venue.
    for _, group in frame[frame.split.eq("test")].groupby("venue", sort=True, dropna=False):
        n = min(12, len(group))
        selected.extend(rng.choice(group.index.to_numpy(), size=n, replace=False).tolist())
    # DOI is the API identity.  Retain the first deterministic representative
    # when duplicate dataset rows resolve to the same DOI.
    result = frame.loc[sorted(set(selected))].copy().drop_duplicates("doi", keep="first")
    assert result.doi.is_unique
    return result.sort_values(["split", "venue", "id"], kind="stable")


def request_batch(dois: list[str]) -> list[dict | None]:
    response = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params={"fields": FIELDS}, json={"ids": ["DOI:" + value for value in dois]}, timeout=45,
        headers={"User-Agent": "assignment-fulltext-feasibility-audit/1.0"},
    )
    response.raise_for_status()
    value = response.json()
    assert isinstance(value, list) and len(value) == len(dois)
    return value


def crossref_cache() -> dict[str, dict]:
    """Publisher-deposited metadata already collected for reference-graph work."""
    path = ROOT / "experiments" / "reference_graph_crossref" / "outputs" / "crossref_raw.jsonl"
    result = {}
    for line in path.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row.get("http_status") == 200 and isinstance(row.get("message"), dict):
            result[str(row["doi"]).lower()] = row["message"]
    return result


def publisher_row(record, message: dict | None) -> dict:
    api_title = "" if message is None else str((message.get("title") or [""])[0])
    original = title_tokens(record.title); returned = title_tokens(api_title)
    similarity = len(original & returned) / len(original | returned) if original or returned else 0.0
    accepted = message is not None and similarity >= 0.60
    links = [] if not accepted else message.get("link") or []
    pdf_links = [str(x.get("URL")) for x in links if isinstance(x, dict) and str(x.get("URL") or "") and
                 ("pdf" in str(x.get("URL")).lower() or str(x.get("content-type") or "").lower() == "application/pdf")]
    resource = message.get("resource") if accepted else None
    primary = resource.get("primary", {}).get("URL", "") if isinstance(resource, dict) else ""
    return dict(
        source="crossref_cached_metadata", split=record.split, id=str(record.id), venue=record.venue, year=int(record.year), doi=record.doi,
        Label=label_value(record.Label), api_found=message is not None,
        title_jaccard=similarity, identity_accepted=accepted,
        open_access_pdf=False, open_access_pdf_url="", publication_types="" if not accepted else str(message.get("type") or ""),
        api_year="" if not accepted else str(message.get("published", {}).get("date-parts", [[""]])[0][0]),
        api_venue="" if not accepted else str((message.get("container-title") or [""])[0]),
        publisher_pdf_pointer=bool(pdf_links), publisher_pdf_url="|".join(pdf_links), publisher_page_pointer=bool(primary), publisher_page_url=str(primary),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.context(); data = ctx["data"].copy()
    sample = select_sample(data)
    sample_path = OUT / "sample.csv"
    sample[["split", "id", "title", "venue", "year", "doi", "Label"]].to_csv(sample_path, index=False)
    # Semantic Scholar batch endpoint returned HTTP 429 on the initial audit
    # attempt.  Keep this run offline against the existing Crossref cache rather
    # than retrying a rate-limited service.
    use_semantic_scholar = True
    try:
        if not use_semantic_scholar:
            raise requests.HTTPError(response=type("RateLimited", (), {"status_code": 429})())
        rows = request_batch(sample.doi.tolist())
        source_name = "semantic_scholar_api"
        ledger = []
        for record, source in zip(sample.itertuples(index=False), rows):
            api_title = "" if source is None else str(source.get("title") or "")
            original = title_tokens(record.title); returned = title_tokens(api_title)
            similarity = len(original & returned) / len(original | returned) if original or returned else 0.0
            open_pdf = None if source is None else source.get("openAccessPdf")
            accepted = source is not None and similarity >= 0.60
            ledger.append(dict(source=source_name, split=record.split, id=str(record.id), venue=record.venue, year=int(record.year), doi=record.doi,
                Label=label_value(record.Label), api_found=source is not None, title_jaccard=similarity,
                identity_accepted=accepted, open_access_pdf=bool(accepted and isinstance(open_pdf, dict) and open_pdf.get("url")),
                open_access_pdf_url="" if not accepted or not isinstance(open_pdf, dict) else str(open_pdf.get("url") or ""),
                publication_types="" if not accepted else json.dumps(source.get("publicationTypes") or []),
                api_year="" if not accepted else str(source.get("year") or ""), api_venue="" if not accepted else str(source.get("venue") or ""),
                publisher_pdf_pointer=False, publisher_pdf_url="", publisher_page_pointer=False, publisher_page_url=""))
    except requests.HTTPError as error:
        # Do not retry rate-limited services.  A cached, already verified source
        # is enough for this feasibility audit, while keeping its weaker meaning explicit.
        if error.response.status_code != 429:
            raise
        source_name = "crossref_cached_metadata_after_semantic_scholar_429"
        cache = crossref_cache()
        ledger = [publisher_row(record, cache.get(record.doi)) for record in sample.itertuples(index=False)]
    result = pd.DataFrame(ledger)
    result.to_csv(OUT / "ledger.csv", index=False)
    summary = result.groupby(["split", "venue"], dropna=False).agg(
        n=("id", "size"), found=("api_found", "mean"), identity_accepted=("identity_accepted", "mean"),
        publisher_pdf_pointer=("publisher_pdf_pointer", "mean"), publisher_page_pointer=("publisher_page_pointer", "mean"),
        typed=("publication_types", lambda x: float(x.ne("").mean())),
    )
    summary.to_csv(OUT / "coverage_by_split_venue.csv")
    train_labels = result[result.split.eq("train")].groupby("Label", dropna=False).agg(
        n=("id", "size"), publisher_pdf_pointer=("publisher_pdf_pointer", "mean"), typed=("publication_types", lambda x: float(x.ne("").mean())),
    )
    train_labels.to_csv(OUT / "coverage_by_train_label.csv")
    overall = dict(
        sample_rows=int(len(result)), train_rows=int(result.split.eq("train").sum()), test_rows=int(result.split.eq("test").sum()),
        found_rate=float(result.api_found.mean()), identity_accepted_rate=float(result.identity_accepted.mean()),
        publisher_pdf_pointer_rate=float(result.publisher_pdf_pointer.mean()), publisher_page_pointer_rate=float(result.publisher_page_pointer.mean()),
        publication_type_rate=float(result.publication_types.ne("").mean()), source=source_name,
        limitations="A publisher link is a feasibility signal, not proof of open access or a model feature. No PDF body, citation count, test label, or manual preference judgement was collected.",
        sample_sha256=digest(sample_path),
    )
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    report = "# Public full-text feasibility audit\n\n" + pd.DataFrame([overall]).to_markdown(index=False) + "\n\n## Coverage by split and venue\n\n" + summary.reset_index().to_markdown(index=False, floatfmt=".3f") + "\n\n## Training coverage by rating\n\n" + train_labels.reset_index().to_markdown(index=False, floatfmt=".3f") + "\n"
    (OUT / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

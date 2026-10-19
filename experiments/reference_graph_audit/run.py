#!/usr/bin/env python3
"""Prepare, collect and audit a public outgoing-reference graph pilot."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
CACHE = OUT / "openalex_raw.jsonl"
sys.path.insert(0, str(ROOT))
from collect_remaining_abstracts import load_sources

FIELDS = "id,doi,title,publication_year,type,referenced_works,authorships,primary_topic,biblio"


def norm(value):
    value = html.unescape(str(value or "")).lower()
    value = "".join(x for x in unicodedata.normalize("NFKD", value) if not unicodedata.combining(x))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def surname_tokens(value):
    # Dataset authors are semicolon-separated; normalize suffix accents and punctuation.
    return {norm(x).split()[-1] for x in re.split(r",|;|\band\b", value) if norm(x)}


def canonical_doi(value):
    value = unquote(str(value or "")).strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value if re.match(r"^10\.\d{4,9}/\S+$", value) else ""


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    records = pd.DataFrame([dict(split=s, **row) for (s, _), row in sources.items()])
    records.id = records.id.astype(str)
    records.year = pd.to_numeric(records.year, errors="raise").astype(int)
    records["Label"] = pd.to_numeric(records.Label, errors="coerce")
    consensus = pd.read_csv(ROOT / "experiments/cross_model_error_audit/outputs/incumbent_fp_consensus.csv", dtype={"id": str})
    fp = consensus[consensus.models_rejecting.eq(0)]
    assert len(fp) == 137
    oof = pd.read_csv(ROOT / "experiments/tabpfn_preference/outputs/outer_predictions.csv", dtype={"id": str})
    oof = oof[oof.arm.eq("C0_deployed")].drop(columns=["Label"])
    train = records[records.split.eq("train")].merge(oof, on="id", validate="one_to_one")
    candidates = train[train.selected & train.Label.gt(1)].copy()
    matched = []; used = set(); cohort_rows = []
    for row in train[train.id.isin(fp.id)].sort_values(["outer_fold", "venue", "year", "id"]).itertuples():
        pool = candidates[candidates.outer_fold.eq(row.outer_fold) & candidates.venue.eq(row.venue) & ~candidates.id.isin(used)].copy()
        pool["year_gap"] = (pool.year - row.year).abs()
        pool["read_gap"] = (pool.read_probability - row.read_probability).abs()
        control = pool.sort_values(["year_gap", "read_gap", "id"]).iloc[0]
        used.add(control.id)
        pair_id = f"P{len(matched)+1:03d}"
        matched.append(dict(pair_id=pair_id, fp_id=row.id, tp_id=control.id, outer_fold=int(row.outer_fold),
                            venue=row.venue, fp_year=int(row.year), tp_year=int(control.year), year_gap=int(control.year_gap)))
        for identifier, group in [(row.id, "consensus_fp"), (control.id, "matched_tp")]:
            item = records[records.split.eq("train") & records.id.eq(identifier)].iloc[0].to_dict()
            item.update(cohort=group, pair_id=pair_id, outer_fold=int(row.outer_fold))
            cohort_rows.append(item)
    cohort = pd.DataFrame(cohort_rows)
    rng = np.random.default_rng(20261018)
    for split in ["train", "test"]:
        pool = records[records.split.eq(split) & ~records.id.isin(cohort.id)].sort_values("id")
        sample = pool.iloc[rng.choice(len(pool), 20, replace=False)].copy()
        sample["cohort"] = f"random_{split}"; sample["pair_id"] = ""; sample["outer_fold"] = -1
        cohort = pd.concat([cohort, sample], ignore_index=True)
    cohort.to_csv(OUT / "cohort.csv", index=False)
    pd.DataFrame(matched).to_csv(OUT / "matched_pairs.csv", index=False)
    print("prepared", len(cohort), "papers; year gaps", pd.Series([x["year_gap"] for x in matched]).value_counts().to_dict(), flush=True)


def cache_entries():
    if not CACHE.exists():
        return {}
    result = {}
    for line in CACHE.read_text().splitlines():
        item = json.loads(line); result[(item["split"], str(item["id"]))] = item
    return result


def fetch(row):
    item = dict(split=row["split"], id=str(row["id"]), requested_title=row["title"],
                retrieved_at=datetime.now(timezone.utc).isoformat())
    query = row["title"] if row["split"] == "batch" else norm(row["title"])
    item["query_filter"] = row.get("query_filter") or "title.search:" + query
    for attempt in range(2):
        try:
            response = requests.get("https://api.openalex.org/works", params={"filter": item["query_filter"],
                "per_page": 100 if row["split"] == "batch" else 5, "select": FIELDS}, headers={"User-Agent": "Personalized-reading-metadata-audit/1.0"}, timeout=25)
            item.update(http_status=response.status_code,
                        rate_headers={k:v for k,v in response.headers.items() if "ratelimit" in k.lower()})
            if response.ok:
                item["response"] = response.json(); return item
            item["error"] = response.text[:300]
            if response.status_code in (401, 403, 429):
                return item
        except requests.RequestException as error:
            item.update(http_status=0, error=type(error).__name__ + ": " + str(error)[:180])
        if attempt == 0:
            time.sleep(1)
    return item


def collect():
    rows = pd.read_csv(OUT / "cohort.csv", dtype={"id": str}, keep_default_na=False).to_dict("records")
    cache = cache_entries()
    remaining = [r for r in rows if (r["split"], r["id"]) not in cache
                 or cache[(r["split"], r["id"])].get("http_status") == 0
                 or (cache[(r["split"], r["id"])].get("request_titles")
                     and ("|" not in cache[(r["split"], r["id"])].get("query_filter", "")
                          or len(cache[(r["split"], r["id"])].get("response", {}).get("results", []))
                          < min(100, cache[(r["split"], r["id"])].get("response", {}).get("meta", {}).get("count", 0))))]
    print("cached", len(cache), "remaining", len(remaining), flush=True)
    failures = 0
    # OR on a title filter has the same query price as a singleton search. A batch
    # of 20 titles fits the anonymous budget without adding credentials or accounts.
    with CACHE.open("a", encoding="utf8") as sink:
        for start in range(0, len(remaining), 20):
            pending = remaining[start:start+20]
            request = dict(split="batch", id="batch", title="|".join(norm(r["title"]) for r in pending))
            response = fetch(request)
            batch = [dict(response, split=r["split"], id=str(r["id"]), requested_title=r["title"],
                          request_titles=[x["title"] for x in pending]) for r in pending]
            for item in batch:
                sink.write(json.dumps(item, ensure_ascii=False) + "\n"); sink.flush()
            failures = failures + len(batch) if all(x["http_status"] == 0 for x in batch) else 0
            if any(x["http_status"] in (401, 403, 429) for x in batch) or failures >= 6:
                print("stopping on access/network block", [x["http_status"] for x in batch], flush=True); break
            print("collected", start+len(batch), "of", len(remaining), flush=True)
            time.sleep(1.1)


def identify(row, response):
    if not response:
        return {"identity_status": "NOT_REQUESTED"}, None
    if response.get("http_status") != 200:
        return {"identity_status": "REQUEST_FAILED", "http_status": response.get("http_status")}, None
    candidates = response.get("response", {}).get("results", [])
    accepted = []; evaluated = []
    source_authors = surname_tokens(str(row["authors"]))
    for candidate in candidates:
        a, b = norm(row["title"]), norm(candidate.get("title"))
        similarity = SequenceMatcher(None, a, b).ratio()
        aa, bb = set(a.split()), set(b.split())
        jaccard = len(aa & bb) / max(1, len(aa | bb))
        year_gap = abs(int(row["year"])-int(candidate.get("publication_year") or 0))
        names = {norm(x.get("author", {}).get("display_name", "").split(",")[0]).split()[-1]
                 for x in candidate.get("authorships", []) if norm(x.get("author", {}).get("display_name", ""))}
        author_ok = bool(source_authors & names) if source_authors and names else True
        evidence = dict(title_similarity=similarity, token_jaccard=jaccard, matched_year=candidate.get("publication_year"),
                        year_gap=year_gap, author_overlap=bool(source_authors & names), author_check_available=bool(source_authors and names),
                        matched_title=candidate.get("title"), work_id=candidate.get("id"))
        evaluated.append(evidence)
        if similarity >= .96 and jaccard >= .90 and year_gap <= 1 and author_ok:
            accepted.append((evidence, candidate))
    if len({x[1]["id"] for x in accepted}) == 1:
        evidence, candidate = accepted[0]
        return dict(identity_status="ACCEPTED", identity_method="unique_title_year_author", **evidence), candidate
    direct_doi = canonical_doi(row.get("doi"))
    doi_matches = [(e, c) for e, c in accepted if direct_doi and canonical_doi(c.get("doi")) == direct_doi]
    if len(doi_matches) == 1:
        evidence, candidate = doi_matches[0]
        return dict(identity_status="ACCEPTED", identity_method="dataset_doi_disambiguated", **evidence), candidate
    best = max(evaluated, key=lambda x:x["title_similarity"]) if evaluated else {}
    status = "AMBIGUOUS" if accepted else "NO_VERIFIED_MATCH"
    if not accepted and best.get("title_similarity", 0) >= .96 and best.get("token_jaccard", 0) >= .90:
        if best["year_gap"] > 1:
            status = "YEAR_CONFLICT"
        elif best["author_check_available"] and not best["author_overlap"]:
            status = "AUTHOR_CONFLICT"
    return dict(identity_status=status, **best), None


def doi_fallback():
    ledger = pd.read_csv(OUT / "identity_ledger.csv", dtype={"id": str}, keep_default_na=False)
    pending = ledger[~ledger.verified & ledger.doi.map(canonical_doi).ne("")].to_dict("records")
    cache = cache_entries()
    with CACHE.open("a", encoding="utf8") as sink:
        for start in range(0, len(pending), 50):
            batch = pending[start:start+50]
            query = "doi:" + "|".join("https://doi.org/"+canonical_doi(r["doi"]) for r in batch)
            response = fetch(dict(split="batch", id="doi_batch", title="DOI lookup", query_filter=query))
            if response["http_status"] != 200:
                print("DOI fallback blocked", response["http_status"], flush=True); break
            for row in batch:
                prior = cache[(row["split"], row["id"])]
                combined = {w["id"]: w for w in prior.get("response", {}).get("results", [])}
                combined.update({w["id"]: w for w in response["response"].get("results", [])
                                 if canonical_doi(w.get("doi")) == canonical_doi(row["doi"])})
                item = dict(response, split=row["split"], id=row["id"], requested_title=row["title"],
                            response={"results": list(combined.values())}, method="title_plus_dataset_doi", dataset_doi=row["doi"])
                sink.write(json.dumps(item, ensure_ascii=False)+"\n")
            sink.flush()
            print("DOI fallback attempted", start+len(batch), "of", len(pending), flush=True)


def refine_titles():
    rows = pd.read_csv(OUT / "cohort.csv", dtype={"id": str}, keep_default_na=False).to_dict("records")
    cache = cache_entries()
    pending = []
    for row in rows:
        response = cache[(row["split"], row["id"])].get("response", {})
        if response.get("meta", {}).get("count", 0) > len(response.get("results", [])):
            pending.append(row)
    with CACHE.open("a", encoding="utf8") as sink:
        for start in range(0, len(pending)):
            batch = pending[start:start+1]
            year = int(batch[0]["year"])
            query = 'title.search:"'+norm(batch[0]["title"])+'",publication_year:'+"|".join(map(str,[year-1,year,year+1]))
            response = fetch(dict(split="batch", id="phrase_batch", title="Exact title phrases", query_filter=query))
            if response["http_status"] != 200:
                print("title refinement blocked", response["http_status"]); break
            for row in batch:
                item = dict(response, split=row["split"], id=row["id"], requested_title=row["title"], method="exact_title_phrase_refinement")
                sink.write(json.dumps(item, ensure_ascii=False)+"\n")
            sink.flush()
            print("refined truncated searches", start+len(batch), "of", len(pending), flush=True)
            time.sleep(1.1)


def report():
    data = pd.read_csv(OUT / "cohort.csv", dtype={"id": str}, keep_default_na=False)
    author_audit = pd.read_csv(ROOT / "experiments/input_ablation/outputs/metadata_audit.csv", dtype={"id": str}, keep_default_na=False)
    clean_authors = {(r.split, str(r.id)): r.authors_verified for r in author_audit.itertuples()
                     if str(r.author_verified).lower() == "true" and r.authors_verified}
    cache = cache_entries(); rows = []; works = {}
    for row in data.to_dict("records"):
        author_value = clean_authors.get((row["split"], row["id"]), row["authors"])
        evidence, work = identify(dict(row, authors=author_value), cache.get((row["split"], row["id"])))
        evidence["authors_for_identity"] = author_value
        evidence["authors_from_verified_metadata"] = (row["split"], row["id"]) in clean_authors
        if work:
            works[row["id"]] = work
        topic = (work or {}).get("primary_topic") or {}
        refs = (work or {}).get("referenced_works") or []
        rows.append(dict(**row, **evidence, verified=work is not None,
            references_count=len(refs) if work else np.nan, references_available=bool(refs),
            retrieved_doi=(work or {}).get("doi") or "", work_type=(work or {}).get("type") or "",
            primary_topic=topic.get("display_name", ""),
            authors_with_ids=sum(bool(x.get("author", {}).get("id")) for x in (work or {}).get("authorships", []))))
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "identity_ledger.csv", index=False)
    (OUT / "verified_works.json").write_text(json.dumps(works, ensure_ascii=False, indent=2)+"\n")
    edges = []; support = Counter()
    ordered = sorted(works)
    for j, first in enumerate(ordered):
        refs_first = set(works[first].get("referenced_works") or [])
        for second in ordered[j+1:]:
            if works[first]["id"] == works[second]["id"]:
                continue
            refs_second = set(works[second].get("referenced_works") or [])
            shared = len(refs_first & refs_second)
            if not shared:
                continue
            support[first] += 1; support[second] += 1
            edges.append(dict(first_id=first, second_id=second, shared_references=shared,
                              jaccard=shared/max(1,len(refs_first|refs_second))))
    pd.DataFrame(edges, columns=["first_id", "second_id", "shared_references", "jaccard"]).to_csv(OUT / "local_graph_edges.csv", index=False)
    ledger["local_shared_reference_neighbors"] = ledger.id.map(support).fillna(0).astype(int)
    ledger.to_csv(OUT / "identity_ledger.csv", index=False)
    summary = ledger.groupby("cohort").agg(n=("id", "size"), verified=("verified", "sum"),
        with_references=("references_available", "sum"), median_reference_count=("references_count", "median"),
        local_neighbor_mean=("local_shared_reference_neighbors", "mean"))
    summary["verified_fraction"] = summary.verified / summary.n
    summary["reference_fraction"] = summary.with_references / summary.n
    summary.to_csv(OUT / "coverage_summary.csv")
    pairs = pd.read_csv(OUT / "matched_pairs.csv", dtype={"fp_id": str, "tp_id": str})
    lookup = ledger.set_index("id")
    for prefix in ["fp", "tp"]:
        for field in ["verified", "references_available", "references_count", "local_shared_reference_neighbors", "identity_status"]:
            pairs[prefix+"_"+field] = pairs[prefix+"_id"].map(lookup[field])
    pairs.to_csv(OUT / "paired_coverage.csv", index=False)
    pair_summaries = []
    for subset, subset_rows in [("all_pairs", pairs), ("exact_year", pairs[pairs.year_gap.eq(0)])]:
        both = subset_rows[subset_rows.fp_references_available & subset_rows.tp_references_available]
        diff = both.tp_references_count - both.fp_references_count
        pair_summaries.append(dict(subset=subset, n_pairs=len(subset_rows), both_have_references=len(both),
            only_fp_has_references=int((subset_rows.fp_references_available & ~subset_rows.tp_references_available).sum()),
            only_tp_has_references=int((~subset_rows.fp_references_available & subset_rows.tp_references_available).sum()),
            median_tp_minus_fp_reference_count=float(diff.median()) if len(diff) else None,
            tp_more_references=int((diff > 0).sum()), fp_more_references=int((diff < 0).sum()), equal_reference_count=int((diff == 0).sum())))
    pair_summary = pd.DataFrame(pair_summaries)
    pair_summary.to_csv(OUT / "matched_pair_summary.csv", index=False)
    status = "COVERAGE_AUDIT_COMPLETE" if ledger.identity_status.ne("NOT_REQUESTED").all() and not ledger.identity_status.eq("REQUEST_FAILED").any() else "PARTIAL_COLLECTION"
    decision = dict(status=status, requested_papers=len(ledger), responses=len(cache), verified=int(ledger.verified.sum()),
        papers_with_references=int(ledger.references_available.sum()), local_shared_reference_edges=len(edges),
        identity_status_counts=ledger.identity_status.value_counts().to_dict(), models_trained=0,
        note="Coverage and local graph support do not establish added predictive value. The local audit pool includes no supervised label aggregation.")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2)+"\n")
    text = ["# Public outgoing-reference graph pilot", "", json.dumps(decision, indent=2), "",
        summary.reset_index().to_markdown(index=False, floatfmt=".4f"), "",
        "The diagnostic pairs were selected by original training labels and incumbent outcomes; they are not an independent benchmark. Test rows are used only for identity/metadata coverage. Request failures are kept separate from absent references.", "",
        "Sources: https://help.openalex.org/api/authentication/ ; https://help.openalex.org/how-to/api-recipes/ . Raw timestamps and candidate records are in openalex_raw.jsonl. No incoming citation count was collected as a feature.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(text), encoding="utf8")
    print(json.dumps(decision, indent=2)); print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["prepare", "collect", "report", "doi_fallback", "refine_titles"])
    args = parser.parse_args()
    globals()[args.stage]()

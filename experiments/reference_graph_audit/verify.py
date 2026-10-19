#!/usr/bin/env python3
"""Verify identity guards, paired sampling, source labels and graph edge counts."""
import hashlib
import json

import numpy as np
import pandas as pd

import run as audit


def main():
    row = dict(title="Branching Bisimulation Learning.", year=2025, authors="Jane Smith, John Doe", doi="10.1234/published")
    work = dict(id="https://openalex.org/W1", title="Branching Bisimulation Learning", publication_year=2025,
                doi="https://doi.org/10.1234/published", authorships=[dict(author=dict(display_name="Jane Smith"))])
    def identify(works, query=row):
        return audit.identify(query, dict(http_status=200, response=dict(results=works)))[0]["identity_status"]
    assert identify([work]) == "ACCEPTED"
    assert identify([dict(work, publication_year=2021)]) == "YEAR_CONFLICT"
    assert identify([dict(work, title="Other research")]) == "NO_VERIFIED_MATCH"
    assert identify([dict(work, authorships=[dict(author=dict(display_name="Another Person"))])]) == "AUTHOR_CONFLICT"
    assert identify([dict(work, authorships=[dict(author=dict(display_name="Smith, Jane"))])]) == "ACCEPTED"
    second = dict(work, id="https://openalex.org/W2", doi="https://doi.org/10.48550/arxiv.1234")
    assert identify([work, second], dict(row, doi="")) == "AMBIGUOUS"
    assert identify([work, second]) == "ACCEPTED"
    assert audit.identify(row, dict(http_status=429))[0]["identity_status"] == "REQUEST_FAILED"

    out = audit.OUT
    cohort = pd.read_csv(out / "cohort.csv", dtype={"id": str}, keep_default_na=False)
    pairs = pd.read_csv(out / "matched_pairs.csv", dtype={"fp_id": str, "tp_id": str})
    ledger = pd.read_csv(out / "identity_ledger.csv", dtype={"id": str}, keep_default_na=False)
    assert len(cohort) == len(ledger) == 314 and len(pairs) == 137
    assert not cohort.duplicated(["split", "id"]).any()
    assert set(pairs.fp_id).isdisjoint(pairs.tp_id)
    assert pairs.tp_id.nunique() == 137 and pairs.year_gap.le(1).all()
    records = audit.load_sources()
    for r in cohort.itertuples():
        source = records[(r.split, r.id)]
        assert r.title == source["title"] and r.authors == source["authors"]
        if r.split == "train":
            assert float(r.Label) == float(source["Label"])
        else:
            assert r.Label == ""
    assert (cohort.loc[cohort.cohort.eq("consensus_fp"), "Label"].astype(float) == 1).all()
    assert (cohort.loc[cohort.cohort.eq("matched_tp"), "Label"].astype(float) > 1).all()
    cache = audit.cache_entries()
    assert all(cache[(r.split, r.id)]["http_status"] == 200 for r in cohort.itertuples())
    truncated = []
    for key, response in cache.items():
        body = response.get("response", {})
        if body.get("meta", {}).get("count", 0) > len(body.get("results", [])):
            truncated.append(key)
    assert not truncated, f"Incomplete candidate pools: {truncated}"
    works = json.loads((out / "verified_works.json").read_text())
    assert set(works) == set(ledger.loc[ledger.verified, "id"])
    edges = pd.read_csv(out / "local_graph_edges.csv", dtype={"first_id": str, "second_id": str})
    assert not edges.duplicated(["first_id", "second_id"]).any()
    for edge in edges.itertuples():
        a, b = works[edge.first_id], works[edge.second_id]
        assert a["id"] != b["id"]
        first, second = set(a.get("referenced_works") or []), set(b.get("referenced_works") or [])
        assert len(first & second) == edge.shared_references > 0
        np.testing.assert_allclose(edge.jaccard, len(first & second)/len(first | second), atol=1e-12)
    paths = [audit.ROOT / "assignment-data-mining-261.zip", audit.HERE / "PLAN.md", audit.HERE / "run.py",
             audit.HERE / "verify.py", out / "cohort.csv", out / "openalex_raw.jsonl", out / "identity_ledger.csv"]
    result = dict(status="PASSED", identity_fixture_checks=8, source_labels_unchanged=True,
                  cohort_rows=len(cohort), matched_pairs=len(pairs), verified_works=len(works), graph_edges=len(edges),
                  incomplete_candidate_pools=len(truncated),
                  sha256={str(p.relative_to(audit.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()

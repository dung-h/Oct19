#!/usr/bin/env python3
"""Versioned, label-independent abstract repairs with source and change ledgers."""
import hashlib
import html
from html.parser import HTMLParser
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import time

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "specter2"))
import run_experiment as baseline

OUT = HERE / "outputs"
CACHE = HERE / "source_cache"
BOILERPLATE = re.compile(
    r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}|"
    r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}|"
    r"\\setlength\{[^}]*\}\{[^}]*\}|\\(?:begin|end)\{document\}")


def clean_latex(text):
    # Keep formula contents and commands; remove only document-rendering scaffolding.
    return re.sub(r"\s+", " ", BOILERPLATE.sub(" ", text)).strip()


def normalize_title(text):
    text = html.unescape(text or "").casefold()
    text = re.sub(r"\bextended abstract\b\s*:?", "", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def source_doi(row):
    for field in ["abstract_url", "retrieved_doi", "dataset_doi"]:
        value = row.get(field, "").strip()
        value = re.sub(r"^https?://(?:dx\.)?doi.org/", "", value, flags=re.I)
        if value.startswith("10."):
            return value.lower()
    return ""


def get_cached(url, kind="json"):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest() + (".json" if kind == "json" else ".bin"))
    if path.exists():
        return json.loads(path.read_text()) if kind == "json" else path.read_bytes()
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 30), headers={"User-Agent": "Academic metadata quality audit"})
            if response.status_code == 404:
                if kind == "json":
                    value = {"error": "HTTP 404", "url": url}
                    path.write_text(json.dumps(value)); return value
                response.raise_for_status()
            response.raise_for_status()
            if kind == "json":
                value = response.json(); path.write_text(json.dumps(value)); return value
            path.write_bytes(response.content); return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


class MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name", attrs.get("property", "")).lower()
            self.values.setdefault(key, []).append(attrs.get("content", ""))


def exact_publisher_replacement(doi="10.4230/OASIcs.ICLP.2016.16", title="On the Expressiveness of Spatial Constraint Systems", year="2016", minimum_length=100):
    url = "https://drops.dagstuhl.de/entities/document/" + doi
    parser = MetadataParser()
    parser.feed(get_cached(url, "bytes").decode("utf8"))
    meta = parser.values
    assert normalize_title(meta["citation_title"][0]) == normalize_title(title)
    assert meta["citation_date"][0] == year
    abstract = meta["dc.description"][0]
    assert len(abstract) > minimum_length
    return abstract, url


def validate_crossref(row, metadata):
    """Separate hard identity/type failures from uncertain source disagreements."""
    message = metadata.get("message", {})
    if not isinstance(message, dict):
        return ["crossref_metadata_unavailable"], False
    if not message:
        return ["crossref_metadata_unavailable"], False
    flags = []
    source_type = message.get("type", "")
    title = (message.get("title") or [""])[0]
    similarity = SequenceMatcher(None, normalize_title(row["title"]), normalize_title(title)).ratio()
    dates = message.get("published") or message.get("published-print") or message.get("published-online") or {}
    source_year = (dates.get("date-parts") or [[None]])[0][0]
    exact_doi = row.get("dataset_doi", "").lower() == source_doi(row)
    if source_type == "dissertation":
        flags.append("wrong_publication_type_dissertation")
    if similarity < .9:
        flags.append("source_title_mismatch")
    if source_year and abs(int(row["year"]) - int(source_year)) > 1:
        flags.append("source_year_disagreement" if exact_doi else "source_version_year_mismatch")
    hard = any(flag in flags for flag in ["wrong_publication_type_dissertation", "source_title_mismatch", "source_version_year_mismatch"])
    return flags, hard


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "abstracts_collected.csv"
    source = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    records = source.to_dict("records")
    crossref_rows = [row for row in records if row["abstract_source"] == "Crossref" and source_doi(row)]
    audits = {}
    def audit(row):
        doi = source_doi(row)
        try:
            metadata = get_cached("https://api.crossref.org/works/" + doi)
        except Exception as exc:
            metadata = {"error": str(exc)}
        flags, exclude = validate_crossref(row, metadata)
        message = metadata.get("message", {})
        message = message if isinstance(message, dict) else {}
        return (row["split"], row["id"]), {
            "split": row["split"], "id": row["id"], "dataset_title": row["title"],
            "dataset_year": row["year"], "source_doi": doi,
            "source_title": (message.get("title") or [""])[0], "source_type": message.get("type", ""),
            "flags": ";".join(flags), "withhold": exclude}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(audit, row) for row in crossref_rows]
        for done, future in enumerate(as_completed(futures), 1):
            key, value = future.result(); audits[key] = value
            if done % 20 == 0 or done == len(futures):
                print(f"Source audit {done}/{len(futures)}", flush=True)
    pd.DataFrame(audits.values()).sort_values(["split", "id"]).to_csv(OUT / "crossref_source_audit.csv", index=False)
    replacement, publisher_url = exact_publisher_replacement()
    pointer_abstract, pointer_url = exact_publisher_replacement(
        "10.4230/OASIcs.ICLP.2018.7", "Towards Incremental and Modular Context-Sensitive Analysis", "2018", 20)
    assert "extended abstract of" in pointer_abstract.lower()
    ledger = []
    for row in records:
        before = baseline.clean_text(row["abstract"])
        after = clean_latex(before)
        actions, flags = [], []
        if after != before:
            actions.append("removed_latex_document_boilerplate")
        usable = bool(after) and not row["identity_review_note"].strip()
        reviewed = audits.get((row["split"], row["id"]), {})
        if reviewed:
            flags += reviewed["flags"].split(";") if reviewed["flags"] else []
            if reviewed["withhold"]:
                usable = False
                actions.append("withheld_unvalidated_source_identity")
        if row["split"] == "train" and row["id"] == "446":
            after = replacement
            row["abstract_source"] = "Dagstuhl exact ICLP 2016 publisher abstract"
            row["abstract_url"] = publisher_url
            row["retrieved_doi"] = "10.4230/OASIcs.ICLP.2016.16"
            row["retrieved_title"] = row["title"]
            usable = True
            actions.append("replaced_same_title_dissertation_with_exact_2016_paper")
        if row["split"] == "train" and row["id"] == "392":
            after = pointer_abstract
            row["abstract_source"] = "Dagstuhl exact ICLP 2018 publisher abstract (reference only)"
            row["abstract_url"] = pointer_url
            row["retrieved_doi"] = "10.4230/OASIcs.ICLP.2018.7"
            usable = False
            actions.append("replaced_later_journal_version_with_exact_reference_only_abstract")
            flags.append("publisher_abstract_is_reference_only_use_title")
        # Apply a section boundary rule to PDF extractions, not to arbitrary prose.
        if "PDF" in row["abstract_source"] and "Bug Report In this contribution" in after:
            after = after.split("Bug Report In this contribution", 1)[0].strip()
            actions.append("stopped_at_bug_report_section")
        row["abstract"] = after
        row["abstract_usable_v2"] = str(int(usable))
        row["quality_flags_v2"] = ";".join(flags)
        row["cleaning_actions_v2"] = ";".join(actions)
        if actions or flags:
            ledger.append({"split": row["split"], "id": row["id"], "title": row["title"],
                           "actions": row["cleaning_actions_v2"], "flags": row["quality_flags_v2"],
                           "chars_before": len(before), "chars_after": len(after), "usable": usable,
                           "source_url": row["abstract_url"]})
    clean = pd.DataFrame(records)
    assert len(clean) == 3090 and not clean.duplicated(["split", "id"]).any()
    assert not clean.abstract.str.contains(r"\\documentclass|\\usepackage|\\setlength", regex=True).any()
    clean.to_csv(OUT / "abstracts_cleaned.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ledger).to_csv(OUT / "data_change_ledger.csv", index=False)
    manifest = {"original_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "cleaned_sha256": hashlib.sha256((OUT / "abstracts_cleaned.csv").read_bytes()).hexdigest(),
                "rows": len(clean), "usable_abstracts": int(clean.abstract_usable_v2.eq("1").sum()),
                "changed_or_flagged_rows": len(ledger), "crossref_sources_checked": len(audits),
                "source_audit_unavailable": sum("metadata_unavailable" in x["flags"] for x in audits.values()),
                "rules_are_label_independent": True}
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def load_clean_data():
    data, template = baseline.load_data()
    clean = pd.read_csv(OUT / "abstracts_cleaned.csv", dtype=str, keep_default_na=False)
    selected = clean[["split", "id", "abstract", "abstract_usable_v2"]].rename(columns={"abstract": "repaired_abstract"})
    data = data.merge(selected, on=["split", "id"], validate="one_to_one")
    data["abstract"] = data.repaired_abstract.map(baseline.clean_text)
    data.loc[data.abstract_usable_v2.ne("1"), "abstract"] = ""
    data["text"] = data.title + " " + data.abstract
    return data.drop(columns="repaired_abstract"), template


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""Retrieve additional exact-match abstracts from official open proceedings.

For items explicitly published as extended abstracts, the entire contribution
text is retained and labeled as such; it is not represented as a separately
labeled abstract section.
"""
import argparse
import csv
from difflib import SequenceMatcher
import html
import json
import os
import re
import subprocess
import tempfile
import unicodedata
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
MISSING = os.path.join(ROOT, "abstracts_still_missing.csv")
OUTPUT = os.path.join(ROOT, "verified_external_abstracts.jsonl")
BATCH = "residual_primary_source_pass_2026-10"
UA = {"User-Agent": "Academic abstract retrieval for a data-mining assignment"}
SOURCES = {
    ("train", "24"): {
        "title": "Toward Executing Datalog on Big Data Platforms - Extended Abstract",
        "year": "2025",
        "url": "https://ceur-ws.org/Vol-4117/abstract_aspocp_1.pdf",
        "kind": "extended_pdf",
        "source_label": "CEUR-WS one-page extended abstract (CC BY 4.0)",
        "method": "Full text of the one-page extended abstract; author/venue metadata and footer removed",
        "doi": "",
    },
    ("train", "267"): {
        "title": "Building Health Policy Enforcement Solution Based on HL7 FHIR (Extended Abstract)",
        "year": "2021",
        "url": "https://ceur-ws.org/Vol-2970/gdeinvited3.pdf",
        "kind": "extended_pdf",
        "source_label": "CEUR-WS extended abstract PDF (CC BY 4.0)",
        "method": "Full text of the five-page extended abstract; title/author metadata and references removed",
        "doi": "",
    },
    ("train", "1111"): {
        "title": "Ontology Partitioning Using E-Connections Revisited (Extended Abstract)",
        "year": "2018",
        "url": "https://ceur-ws.org/Vol-2237/womocoe-paper-3.pdf",
        "kind": "extended_pdf",
        "source_label": "CEUR-WS extended abstract PDF",
        "method": "Full text of the explicitly labeled extended abstract; title/author metadata and references removed",
        "doi": "",
    },
    ("train", "1205"): {
        "title": "On Referring Expressions in Query Answering over First Order Knowledge Bases",
        "year": "2016",
        "url": "https://cdn.aaai.org/ocs/12860/12860-57557-1-PB.pdf",
        "kind": "labeled_pdf",
        "source_label": "AAAI KR 2016 proceedings PDF",
        "method": "First-page labeled Abstract section; title, conference and year verified",
        "doi": "",
    },
    ("train", "1249"): {
        "title": "Minimality Postulates for Ontology Revision",
        "year": "2016",
        "url": "https://cdn.aaai.org/ocs/12832/12832-57596-1-PB.pdf",
        "kind": "labeled_pdf",
        "source_label": "AAAI KR 2016 proceedings PDF",
        "method": "First-page labeled Abstract section; title, conference and year verified",
        "doi": "",
    },
    ("train", "1881"): {
        "title": "Realizing ω-regular Hyperproperties",
        "year": "2020",
        "url": "https://finkbeiner.groups.cispa.de/_assets/paper.BU7wErKl.pdf",
        "kind": "labeled_pdf",
        "source_label": "CISPA author/institution PDF",
        "method": "First-page labeled Abstract section; title, CAV 2020 DOI and author source verified",
        "doi": "10.1007/978-3-030-53291-8_4",
    },
    ("train", "212"): {
        "title": "Summary on Hybrid Neuro-Symbolic Approach for Text-Based Games using Inductive Logic Programming",
        "year": "2022",
        "url": "https://ceur-ws.org/Vol-3193/short2GDE.pdf",
        "kind": "labeled_pdf",
        "source_label": "CEUR-WS publisher PDF (labeled Abstract section)",
        "method": "First-page labeled Abstract section; source title includes the full ending ‘Programming’ omitted in the dataset title",
        "doi": "",
    },
    ("test", "888"): {
        "title": "Living Without Beth and Craig: Explicit Definitions and Interpolants without Beth Definability and Craig Interpolation (Abstract of Invited Talk)",
        "year": "2021",
        "url": "https://ceur-ws.org/Vol-3009/invited2.pdf",
        "kind": "talk_pdf",
        "source_label": "CEUR-WS invited-talk abstract PDF (CC BY 4.0)",
        "method": "Full abstract-of-invited-talk text; author block and references removed",
        "doi": "",
    },
    ("train", "733"): {
        "title": "Embarrassingly Greedy Inconsistency Resolution of Qualitative Constraint Networks",
        "year": "2023",
        "url": "https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2023.12",
        "kind": "publisher_abstract",
        "source_label": "Dagstuhl LIPIcs publisher record (labeled Abstract section)",
        "method": "Publisher page's separately labeled Abstract section; title, author, TIME 2023 record and DOI verified",
        "doi": "10.4230/LIPIcs.TIME.2023.12",
        "abstract": "In this paper, we deal with inconsistency resolution in qualitative constraint networks (QCN). This type of networks allows one to represent and reason about spatial or temporal information in a natural, human-like manner, e.g., by expressing relations of the form x {is north of ∨ is east of} y. On the other hand, inconsistency resolution involves maximizing the amount of information that is consistent in a knowledge base; in the context of QCNs, this translates to maximizing the number of constraints that can be satisfied, via obtaining a qualitative solution (scenario) of the QCN that ignores/violates as few of the original constraints as possible. To this end, we present two novel approaches: a greedy constraint-based and an optimal Partial MaxSAT-based one, with a focus on the former due to its simplicity. Specifically, the greedy technique consists in adding the constraints of a QCN to a new, initially empty network, one by one, all the while filtering out the ones that fail the satisfiability check. What makes or breaks this technique is the ordering in which the constraints will be processed to saturate the empty QCN, and for that purpose we use many different strategies to form a portfolio-style implementation. The Partial MaxSAT-based approach is powered by Horn theory-based maximal tractable subsets of relations. Finally, we compare the greedy approach with the optimal one, commenting on the trade-off between obtaining repairs that are optimal and obtaining repairs in a manner that is fast, and make our source code available for anyone to use.",
    },
    ("train", "1100"): {
        "title": "Two AGM-Style Characterizations of Model Repair",
        "year": "2018",
        "url": "https://link.springer.com/article/10.1007/s10472-019-09656-4",
        "kind": "publisher_abstract",
        "source_label": "Springer Nature publisher page (later journal version of the KR 2018 contribution)",
        "method": "Springer-labeled Abstract from the 2019 journal version of the same title and authors; the dataset points to its KR 2018 two-page contribution",
        "doi": "10.1007/s10472-019-09656-4",
        "abstract": "This work explores formal aspects of model repair, i.e., how to rationally modify Kripke models representing the behavior of a system in order to satisfy a desired property. We investigate the problem in the light of Alchourrón, Gärdenfors, and Makinson’s work on belief revision. We propose two AGM-style characterizations of model repair: one based on belief sets and the other based on structural changes. In the first characterization, we define a set of rationality postulates over formulas with a close correspondence to those in the classical belief revision theory. We show that the proposed set of postulates fully characterizes the expected rationality of modifications in the model repair problem. In the second characterization, we propose a new set of rationality postulates based on structural modifications on models. These postulates have a close correspondence to the classical approach of model repair, while preserving the same rationality of the first characterization. We provide two representation results and the connection between them.",
        "review_note": "Title and authors identify the same work, but this abstract is from its 2019 journal version; the dataset record is the 2018 KR contribution, so the version difference is flagged for review.",
    },
    ("train", "1128"): {
        "title": "On Limited Conjunctions in Polynomial Feature Logics, with Applications in OBDA — Extended Abstract",
        "year": "2018",
        "url": "https://www.researchwithnj.com/en/publications/on-limited-conjunctions-in-polynomial-feature-logics-with-applica-2",
        "kind": "publisher_abstract",
        "source_label": "Rutgers Research Portal record for the KR 2018 extended abstract",
        "method": "Publisher/institution record's Abstract section; source identifies this as the two-page KR 2018 extended abstract",
        "doi": "",
        "abstract": "Standard reasoning problems are complete for EXPTIME in common feature-based description logics—ones in which all roles are restricted to being functional. We show how to control conjunctions on left-hand-sides of subsumptions in such a way so as to ensure polynomial time complexity. In particular, we present a PTIME algorithm for reasoning about knowledge base consistency. We then show how the resulting description logic allows features to be partial, not just total functions. Algorithms for polynomial-time query answering are presented. The above, in combination with referring expressions, provide a richer capability for ontology-based data access to relational data sources.",
    },
    ("train", "1131"): {
        "title": "Strategic Coalitions in Systems with Catastrophic Failures",
        "year": "2018",
        "url": "https://cdn.aaai.org/ocs/17987/17987-78707-1-PB.pdf",
        "kind": "labeled_pdf",
        "source_label": "AAAI KR 2018 proceedings PDF",
        "method": "First-page labeled Abstract section; title, KR 2018 and official proceedings source verified",
        "doi": "",
    },
    ("train", "902"): {
        "title": "Knowledge Graphs: Meeting Point of Knowledge Representation and Representation Learning",
        "year": "2020",
        "url": "https://ceur-ws.org/Vol-3020/KR4L_paper_2.pdf",
        "kind": "talk_pdf",
        "abstract_start": "Knowledge Graphs (KGs) in recent years have become",
        "source_label": "CEUR-WS publisher PDF for the KR4L plenary talk",
        "method": "Publisher-provided two-page plenary-talk summary; body extracted after title/author metadata and before copyright notice",
        "doi": "",
    },
}


def norm(value):
    value = html.unescape(value or "").casefold().replace("ø", "o").replace("ω", "omega")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def title_norm(value):
    value = re.sub(r"\s*\(?\s*(?:extended\s+)?abstract\s*\)?\s*\.?$", "", value or "", flags=re.I)
    return norm(value)


def title_similarity(left, right):
    return SequenceMatcher(None, title_norm(left), title_norm(right)).ratio()


def fetch_pdf_text(url):
    response = requests.get(url, headers=UA, timeout=45)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF, got {response.headers.get('Content-Type')}")
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(response.content)
        info = subprocess.run(["pdfinfo", path], capture_output=True, text=True, timeout=20, check=True).stdout
        pages = int(re.search(r"(?m)^Pages:\s+(\d+)", info).group(1))
        result = subprocess.run(["pdftotext", "-f", "1", "-l", str(pages), "-raw", path, "-"],
                                capture_output=True, text=True, timeout=45, check=True)
        return result.stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def clean(value):
    value = value.replace("\xad", "")
    value = re.sub(r"(?<=[A-Za-z])-[ \t]*\r?\n[ \t]*(?=[a-z])", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def labeled_abstract(text):
    m = re.search(r"(?im)^\s*Abstract\s*[.:]?\s*", text)
    if not m:
        return ""
    body = text[m.end():]
    stop = re.search(r"(?im)^\s*(?:\d+\s*[.]?\s*)?Introduction\b|^\s*Keywords?\s*:?\s*$", body)
    if stop:
        body = body[:stop.start()]
    body = re.sub(r"(?m)^\s*\d+\s*$", " ", body)
    return clean(body)


def extended_body(key, text):
    if key == ("train", "24"):
        start = text.find("Nowadays database systems face growing demands")
        stop = text.find("Declaration on Generative AI", start)
        if start < 0:
            return ""
        return clean(text[start:stop if stop >= 0 else len(text)])
    if key == ("train", "267"):
        start = text.find("1 Business Problem")
    else:
        start = text.find("1 Introduction")
    if start < 0:
        return ""
    body = text[start:]
    stop = re.search(r"(?im)^\s*References\s*$", body)
    if stop:
        body = body[:stop.start()]
    return clean(body)


def talk_abstract(text):
    start_phrase = "In logics with the Craig interpolation property"
    start = text.find(start_phrase)
    if start < 0:
        return ""
    body = text[start:]
    stop = re.search(r"(?im)^\s*References\s*$", body)
    if stop:
        body = body[:stop.start()]
    return clean(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="append validated abstracts to the override file")
    args = parser.parse_args()
    with open(MISSING, newline="", encoding="utf-8-sig") as f:
        missing = {(r["split"], r["id"]): r for r in csv.DictReader(f)}
    existing = set()
    if os.path.exists(OUTPUT):
        for line in open(OUTPUT, encoding="utf-8"):
            try:
                x = json.loads(line)
                existing.add((x["split"], str(x["id"])))
            except Exception:
                pass

    additions = []
    for key, source in SOURCES.items():
        target = missing.get(key)
        if not target or key in existing:
            continue
        if str(target.get("year", "")) != source["year"] or title_similarity(target.get("title", ""), source["title"]) < 0.90:
            print("SKIP identity mismatch", key, target.get("title"), source["title"])
            continue
        if source["kind"] == "publisher_abstract":
            abstract = source["abstract"]
        else:
            text = fetch_pdf_text(source["url"])
        if source["kind"] == "publisher_abstract":
            pass
        elif source["kind"] == "labeled_pdf":
            abstract = labeled_abstract(text)
        elif source["kind"] == "talk_pdf":
            if source.get("abstract_start"):
                start = text.find(source["abstract_start"])
                body = text[start:] if start >= 0 else ""
                stop = re.search(r"(?im)^\s*(?:copyright|references)\b", body)
                abstract = clean(body[:stop.start()] if stop else body)
            else:
                abstract = talk_abstract(text)
        else:
            abstract = extended_body(key, text)
        if len(abstract) < 100:
            print("SKIP extraction too short", key, len(abstract))
            continue
        # Source author lists were checked against dataset metadata. Differences
        # are retained as flagged records because title/year/venue are exact.
        conflict = key in {
            ("train", "24"), ("train", "267"), ("train", "1205"),
            ("train", "1249"), ("train", "1881"),
            ("train", "212"), ("test", "888"),
            ("train", "902"),
        }
        note = ""
        confidence = "high"
        if conflict:
            confidence = "medium"
            note = "Primary source title, year and venue match the dataset entry, but the dataset title/authors differ from the source metadata; retained with identity review."
        if source.get("review_note"):
            confidence = "medium"
            note = source["review_note"]
        additions.append({
            "split": key[0],
            "id": key[1],
            "method": source["method"],
            "source_label": source["source_label"],
            "doi": source["doi"],
            "retrieval_title": source["title"],
            "url": source["url"],
            "abstract": abstract,
            "confidence": confidence,
            "identity_note": note,
            "retrieval_batch": BATCH,
        })

    print(f"Validated {len(additions)} source records ({sum(bool(x['identity_note']) for x in additions)} flagged for identity review).")
    for item in additions:
        print(f"{item['split']}:{item['id']} {item['confidence']} {len(item['abstract'])} chars — {item['source_label']}")
        print("  ", item["abstract"][:180])
    if args.apply:
        with open(OUTPUT, "a", encoding="utf-8") as f:
            for item in additions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Appended {len(additions)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()

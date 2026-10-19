#!/usr/bin/env python3
"""Create a row-level audit of the 80 records with no collected abstract."""
import csv
import os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(ROOT, "abstracts_still_missing.csv")
OUTPUT = os.path.join(ROOT, "abstract_missing_audit.csv")

# These 12 titles were manually confirmed to be collections/volumes even though
# title-only triage originally put them in the paper/short-contribution bucket.
MANUAL_COLLECTIONS = {
    ("train", "413"), ("train", "414"), ("train", "455"),
    ("train", "1184"), ("train", "1960"), ("train", "2368"),
    ("train", "2509"), ("train", "2574"), ("train", "2670"),
    ("train", "2744"), ("train", "2811"), ("train", "3003"),
}

INDIVIDUALS = {
    ("train", "898"), ("train", "1028"), ("train", "1029"),
    ("train", "1030"), ("train", "1039"), ("train", "3052"),
    ("test", "899"), ("test", "1042"),
}

GI_EVENT_COLLECTIONS = {
    ("train", "1047"), ("train", "1048"),
    ("train", "1183"), ("test", "1182"),
}
FMBC_VOLUMES = {("train", "1755"), ("train", "1849")}
CAVS_PROCEEDINGS = {("train", "1961"), ("train", "2067")}
SPRINGER_NSV_VOLUME = ("train", "2216")

CEUR_MISLINKED = {
    ("train", x) for x in ("641", "894", "1181", "2219", "2299", "2300", "2441", "2908", "3093")
}

REUSED_S2 = {
    ("train", "895"): "Shared S2 URL with test:896, a differently titled KR 2021 workshop-proceedings record.",
    ("test", "896"): "Shared S2 URL with train:895, the KR 2021 main-proceedings record.",
    ("train", "1420"): "Shared S2 URL with train:1421, a separately titled ASPOCP@LPNMR 2017 workshop record.",
    ("train", "1421"): "Shared S2 URL with train:1420, the LPNMR 2017 main-proceedings record.",
    ("train", "2217"): "Shared S2 URL with train:2218; the dataset lists CAV 2017 Part I and Part II separately, so this identifier alone is ambiguous.",
    ("train", "2218"): "Shared S2 URL with train:2217; the dataset lists CAV 2017 Part II and Part I separately, so this identifier alone is ambiguous.",
}

GI = {
    ("train", "1028"): ("10.18420/cdm-2019-30-01", "Franziskus Kiefer", "Dataset author field is blank; GI record names Franziskus Kiefer."),
    ("train", "1029"): ("10.18420/cdm-2019-30-03", "Franziskus Kiefer", "Dataset author field is blank; GI record names Franziskus Kiefer."),
    ("train", "1030"): ("10.18420/cdm-2019-30-09", "Klaus Kursawe", "Dataset lists J. Horner; GI record names Klaus Kursawe."),
    ("train", "1039"): ("10.18420/cdm-2019-31-02", "Marcel Selhorst", "Dataset author field is blank; GI record names Marcel Selhorst."),
    ("test", "1042"): ("10.18420/cdm-2019-31-03", "Victor van der Veen", "Dataset lists Gert Storms; GI/Dblp record names Victor van der Veen. A same-title VU thesis exists, but its thesis summary is not substituted for the Krypto-Tag technical abstract."),
}

EVIDENCE = {
    ("train", "898"): "https://ceur-ws.org/Vol-3020/KR4L_paper_4.pdf",
    ("test", "899"): "https://ceur-ws.org/Vol-3020/KR4L_paper_9.pdf",
    ("train", "3052"): "https://lics.siglog.org/lics16/toc.html; https://lics.siglog.org/archive/invited.html; https://doi.org/10.1145/2933575.2934579",
    ("train", "1028"): "https://doi.org/10.18420/cdm-2019-30-01; https://dl.gi.de/handle/20.500.12116/30271",
    ("train", "1029"): "https://doi.org/10.18420/cdm-2019-30-03; https://dl.gi.de/handle/20.500.12116/30273",
    ("train", "1030"): "https://doi.org/10.18420/cdm-2019-30-09; https://dl.gi.de/handle/20.500.12116/30263",
    ("train", "1039"): "https://doi.org/10.18420/cdm-2019-31-02; https://dl.gi.de/handle/20.500.12116/30623",
    ("test", "1042"): "https://doi.org/10.18420/cdm-2019-31-03; https://dl.gi.de/handle/20.500.12116/30624; https://dblp.org/rec/conf/kryptotag/Veen19",
    ("train", "1047"): "https://fg-krypto.gi.de/krypto-tag; https://dblp.org/rec/conf/kryptotag/2019-1",
    ("train", "1048"): "https://fg-krypto.gi.de/krypto-tag; https://dblp.org/rec/conf/kryptotag/2019-2",
    ("train", "1183"): "https://fg-krypto.gi.de/krypto-tag",
    ("test", "1182"): "https://fg-krypto.gi.de/krypto-tag",
    ("train", "1755"): "https://drops.dagstuhl.de/entities/volume/OASIcs-volume-105",
    ("train", "1849"): "https://drops.dagstuhl.de/entities/volume/OASIcs-volume-95",
    ("train", "1961"): "https://dblp.org/rec/conf/cavs/2020",
    ("train", "2067"): "https://vtsociety.org/event/symposium/2nd-ieee-connected-and-automated-vehicles-symposium",
    ("train", "2216"): "https://link.springer.com/book/10.1007/978-3-319-54292-8",
}


def audit(row):
    key = (row["split"], str(row["id"]))
    original = row.get("dataset_doi", "")
    type_candidate = row.get("record_type_candidate", "")
    if key in INDIVIDUALS:
        group = "individual_paper_or_talk"
        if key == ("train", "898"):
            finding = "Exact CEUR paper located; source authors differ from dataset authors. PDF starts with Introduction and has no separately labeled Abstract section."
            action = "Do not substitute the Introduction. Keep missing unless a publisher/author abstract record is located."
            source_author = "Paolo Atzeni; Luigi Bellomarini; Davide Benedetto; Emanuel Sallinger"
            identity = "Dataset author list (Arthur Benjamin, Gary Chartrand, Ping Zhang) conflicts with the exact publisher PDF author list."
        elif key == ("test", "899"):
            finding = "Exact CEUR paper located; source authors differ from dataset authors. PDF begins with 'The Industrial Setting' and has no separately labeled Abstract section."
            action = "Do not use the related EDBT paper's description as this paper's abstract. Keep missing pending an exact abstract source."
            source_author = "Paolo Atzeni; Luigi Bellomarini; Michela Iezzi; Emanuel Sallinger; Adriano Vlad"
            identity = "Dataset author list (Aditya Malusare) conflicts with the exact publisher PDF author list."
        elif key in GI:
            doi, source_authors, identity = GI[key]
            finding = f"Exact GI Krypto-Tag technical-abstract record located (source author: {source_authors}); GI metadata has no abstract text and its item has no downloadable file bundle. {identity}"
            action = "Resolve the source-author/record mismatch and obtain the submitted abstract text from GI or the author."
            source_author = source_authors
        elif key == ("train", "3052"):
            finding = "Exact DOI and LICS invited-talk listing confirmed. Checked Crossref/Semantic Scholar metadata have no abstract; official LICS pages list title/speakers but no summary; ACM page content was not retrievable."
            action = "Keep missing unless ACM/full text or author-held keynote abstract becomes accessible. Do not use the related but separately titled Herlihy–Moir paper."
            source_author = "Maurice Herlihy; Mark Moir"
            identity = "Dataset and publisher bibliographic author lists match."
        else:
            finding = "Individual contribution identified; no exact abstract found in checked metadata or accessible full text."
            action = "Keep missing pending an exact publisher/author abstract source."
            source_author = ""
            identity = ""
        return {
            "audit_group": group,
            "record_subtype": "individual paper or talk",
            "audit_basis": "Manually checked title, publisher/source record, authors and accessible full text where available.",
            "source_authors_verified": source_author,
            "abstract_finding": finding,
            "identity_or_version_note": identity,
            "identifier_finding": "Exact DOI/official record located." if key in GI or key == ("train", "3052") else "",
            "recommended_action": action,
            "evidence_urls": EVIDENCE.get(key, ""),
        }

    group = "container_or_collection_record"
    if key in GI_EVENT_COLLECTIONS:
        subtype = "GI Krypto-Tag technical-report collection"
        finding = "Record represents an event-level Krypto-Tag report collection. GI states that submitted abstracts are collected and published as citable technical reports; the event record itself is not one individual contribution."
        action = "If the assignment requires paper abstracts, expand this event/report collection into its individually DOI-linked contributions."
    elif key in FMBC_VOLUMES:
        subtype = "OASIcs FMBC proceedings volume"
        finding = "Record is the complete OASIcs proceedings volume for an FMBC workshop, not an individual paper. The publisher lists component documents separately."
        action = "If the assignment requires paper abstracts, expand the volume into its individual OASIcs contributions."
    elif key in CAVS_PROCEEDINGS:
        subtype = "IEEE CAVS proceedings"
        finding = "Record is the IEEE Connected and Automated Vehicles Symposium proceedings container, not an individual paper."
        action = "If the assignment requires paper abstracts, expand the proceedings into individual IEEE papers."
    elif key == SPRINGER_NSV_VOLUME:
        subtype = "Springer LNCS proceedings volume"
        finding = "Record is a Springer LNCS revised-selected-papers book. Springer identifies it as refereed workshop proceedings with separately listed chapters."
        action = "If the assignment requires paper abstracts, expand the book into its individual chapters."
    elif key in MANUAL_COLLECTIONS or type_candidate == "proceedings_or_collection_candidate":
        subtype = "proceedings_or_edited_volume"
        finding = "Record names a proceedings volume, workshop collection, front matter, or edited-book container rather than one paper. A volume-level description, if present, is not an individual paper abstract."
        action = "If the assignment requires paper abstracts, expand the container into its component papers; otherwise retain as a container record and do not invent a paper abstract."
    else:
        subtype = "conference-level collection record"
        finding = "Source check identifies a conference-level collection/proceedings record rather than an individual contribution. No single paper abstract belongs to the record as represented."
        action = "If the assignment requires paper abstracts, expand this record into its component contributions."

    if key in CEUR_MISLINKED:
        identifier_finding = "Mislinked source: dataset_doi repeats https://ceur-ws.org/Vol-3799/abstract2GDE24.pdf. That URL is the fCASP extended abstract in the 2024 ICLP Workshops volume and does not match this row."
        evidence = "https://ceur-ws.org/Vol-3799/; https://ceur-ws.org/Vol-3799/abstract2GDE24.pdf"
    elif key in REUSED_S2:
        identifier_finding = REUSED_S2[key]
        evidence = original
    else:
        identifier_finding = "No repeated identifier detected within this 80-row missing set; original identifier retained without claiming it is a DOI." if original else "No source identifier present."
        evidence = original
    return {
        "audit_group": group,
        "record_subtype": subtype,
        "audit_basis": "Manual title/source check; record is a proceedings, edited volume, or event-level report collection, not an individual paper record.",
        "source_authors_verified": "",
        "abstract_finding": finding,
        "identity_or_version_note": "",
        "identifier_finding": identifier_finding,
        "recommended_action": action,
        "evidence_urls": evidence,
    }


def main():
    with open(SOURCE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    audited = []
    for row in rows:
        key = (row["split"], str(row["id"]))
        item = {
            "split": row["split"],
            "id": row["id"],
            "year": row["year"],
            "venue": row["venue"],
            "title": row["title"],
            "dataset_authors": row["authors"],
            "original_identifier": row["dataset_doi"],
            **audit(row),
        }
        audited.append(item)

    fields = list(audited[0])
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(audited)

    counts = Counter((x["audit_group"], x["record_subtype"]) for x in audited)
    print(f"Audited {len(audited)} rows -> {OUTPUT}")
    print("By audit group:", dict(counts))
    print("Rows with suspicious/reused identifiers:", sum(bool(x["identifier_finding"]) and ("mislink" in x["identifier_finding"].lower() or "shared" in x["identifier_finding"].lower()) for x in audited))


if __name__ == "__main__":
    main()

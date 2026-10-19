# Abstract collection report
Collection used direct DOI resolution, title and metadata searches in Crossref/OpenAlex/Semantic Scholar/OpenAIRE, official publisher and proceedings records, and author/institution repositories. Matches were checked against title, venue/year, DOI, and source authors where available. PDF extraction uses a labeled Abstract section; for GI Krypto-Tag, the venue defines contributions as submitted abstracts published as citable technical reports, so the report body is extracted with metadata, references, and repository footer removed. Explicitly titled extended abstracts are retained as full extended-abstract text and labeled as such. Publisher-provided invited-talk summaries are labeled separately. Introductions are not substituted for abstracts. Author or publication-version mismatches are flagged for review.
## Coverage
- Total: **3010/3090 (97.41%)**
- Train: **2424/2494 (97.19%)**
- Test: **586/596 (98.32%)**
- Recovered in the focused source-review pass: **70** (35 without a review note; 35 flagged for identity/version review).
- Manually verified overrides applied from the source cache: **89**.

Subsequent [SPECTER2 failure analysis](experiments/specter2/FAILURE_ANALYSIS.md) found additional quality issues among already-collected texts: 13 abstracts contain repeated LaTeX rendering preambles; `train:175` extends past the labeled Abstract into the Bug Report section; `train:446` uses a same-title dissertation abstract rather than the ICLP 2016 contribution. These examples were not covered by the earlier missing-row audit. The coverage above counts nonempty retrieved text, not a complete validation of publication identity, abstract boundaries, or cleanliness. See the [source-issue ledger](experiments/specter2/outputs/failure_analysis/verified_source_issues.csv). The original collection and experiment snapshots are retained for reproducibility.

The confirmed repairs have now been applied in the separate [v2 cleaned dataset](experiments/specter2_v2/outputs/abstracts_cleaned.csv), with a [change ledger](experiments/specter2_v2/outputs/data_change_ledger.csv) and [experiment report](experiments/specter2_v2/README.md). The original CSV reported above remains the baseline snapshot. V2 audits the 100 Crossref sources and uses 2,974 abstracts after conservative review/version handling; 116 model inputs use title only. This does not claim exhaustive validation of every previously collected abstract.

## Abstract source counts
- Semantic Scholar: 1866
- OpenAIRE description: 453
- OpenAlex: 243
- OpenAIRE-linked full text (PDF): 166
- Crossref: 100
- AAAI Proceedings PDF: 64
- GI Digital Library Krypto-Tag technical report (submitted abstract): 42
- Springer Nature publisher page: 32
- Author/institution PDF: 15
- Author institutional PDF: 5
- arXiv author preprint: 3
- Author institutional publication page: 3
- EPTCS publisher proceedings PDF: 2
- AAAI KR 2016 proceedings PDF: 2
- CEUR-WS one-page extended abstract (CC BY 4.0): 1
- CEUR-WS publisher PDF (labeled Abstract section): 1
- CEUR-WS extended abstract PDF (CC BY 4.0): 1
- Dagstuhl OASIcs proceedings PDF: 1
- Dagstuhl LIPIcs publisher record (labeled Abstract section): 1
- CEUR-WS publisher PDF for the KR4L plenary talk: 1
- Springer Nature publisher page (later journal version of the KR 2018 contribution): 1
- CEUR-WS extended abstract PDF: 1
- Rutgers Research Portal record for the KR 2018 extended abstract: 1
- AAAI KR 2018 proceedings PDF: 1
- PUCRS institutional repository: 1
- CISPA author/institution PDF: 1
- CEUR-WS invited-talk abstract PDF (CC BY 4.0): 1
- Author publication page: 1

## Remaining rows
**80** rows remain in [abstracts_still_missing.csv](abstracts_still_missing.csv). This means the abstract was not returned by checked metadata sources or could not be extracted from an accessible full text; it does not establish that the paper has no abstract. Proceedings-volume records are identified separately in that file because they describe a whole collection rather than an individual paper.
The `record_type_candidate` field is title-only triage, not a verified publication-type label:
- proceedings_or_collection_candidate: 51
- paper_or_short_contribution_candidate: 20
- event_or_program_record_candidate: 9

A row-by-row audit is in [abstract_missing_audit.csv](abstract_missing_audit.csv). Source checks show that 72 of the 80 rows are collection-level records: 64 proceedings/edited-book records, four Krypto-Tag technical-report collections, two FMBC OASIcs volumes, and two IEEE CAVS proceedings. The other eight are individual contributions or talks. Twelve rows initially classified as possible papers by title-only triage are actually volumes: three ICLP technical-communications volumes, KR4HC, FMBC 2020, and seven LICS proceedings records. The nine title-only event/program candidates are also collection-level records after source checks (the Springer NSV book, four Krypto-Tag collections, two FMBC OASIcs volumes, and two IEEE CAVS proceedings). A collection record has no one paper-level abstract; if the assignment needs paper abstracts, these records should be expanded into their component contributions.

The eight individual contributions/talks are `train:898`, `train:1028`, `train:1029`, `train:1030`, `train:1039`, `train:3052`, `test:899`, and `test:1042`. The official CEUR PDFs for `train:898` and `test:899` contain the full contributions but begin with an Introduction/section 1 and have no separately labeled Abstract section; both dataset author lists conflict with the PDF authors, so introductions and summaries for related works were not substituted. Five Krypto-Tag records have exact GI Digital Library DOI records. The GI item metadata exposes no abstract text and its current DSpace item has no downloadable file bundle. Their source-author checks identify `Franziskus Kiefer` for `train:1028` and `train:1029`, `Klaus Kursawe` for `train:1030` (the dataset says `J. Horner`), `Marcel Selhorst` for `train:1039`, and `Victor van der Veen` for `test:1042` (the dataset says `Gert Storms`). The official GI description says Krypto-Tag submissions are collected as citable technical reports; the missing text here is a retrieval problem, not evidence that no abstract was submitted. A same-title VU thesis by van der Veen is available, but its thesis summary was not substituted for the distinct Krypto-Tag technical abstract. `train:3052` is confirmed as the LICS 2016 keynote; checked Crossref/Semantic Scholar metadata and the official LICS program expose no abstract text, while the ACM page was not retrievable. A separate Herlihy–Moir preprint on a related topic was not substituted.

The identifier audit also found 15 rows that need care: nine rows reuse `https://ceur-ws.org/Vol-3799/abstract2GDE24.pdf`, which is the fCASP extended abstract in the 2024 ICLP Workshops volume and does not match those rows; three pairs (six rows) reuse one Semantic Scholar URL across differently titled records. The original identifier fields are preserved in the input/output CSVs, and the row-level audit flags these rather than treating them as validated abstract sources. These retrieval, record-type, and identity findings do not establish that all underlying research papers lack abstracts.
- metadata_match_without_abstract_or_fulltext: 68
- no_validated_openaire_record: 9
- no_labeled_abstract_in_open_pdf: 3

## Identity review
35 collected abstract(s) carry an identity or source-version review note, most often because dataset authors conflict with the matched publication record. See `identity_review_note` in [abstracts_collected.csv](abstracts_collected.csv). No validated source candidates remain withheld from coverage; the audit schema is retained in [abstract_identity_review.csv](abstract_identity_review.csv).

## Remaining rows by venue
- kr: 27
- cav: 18
- iclp: 17
- lics: 12
- lpnmr: 6

Sources used in these passes include the [GI Krypto-Tag description](https://fg-krypto.gi.de/krypto-tag), [official LICS '16 table of contents](https://lics.siglog.org/lics16/toc.html), the [CEUR-WS Vol. 3799 table of contents](https://ceur-ws.org/Vol-3799/) and [fCASP extended-abstract PDF](https://ceur-ws.org/Vol-3799/abstract2GDE24.pdf), [FMBC 2021 OASIcs proceedings](https://drops.dagstuhl.de/entities/volume/OASIcs-volume-95), [FMBC 2022 OASIcs proceedings](https://drops.dagstuhl.de/entities/volume/OASIcs-volume-105), [IEEE CAVS 2019 event/proceedings page](https://vtsociety.org/event/symposium/2nd-ieee-connected-and-automated-vehicles-symposium), [Springer NSV 2016 proceedings book](https://link.springer.com/book/10.1007/978-3-319-54292-8), [Dagstuhl LIPIcs TIME 2023 record](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2023.12), [AAAI KR 2018 proceedings paper](https://cdn.aaai.org/ocs/17987/17987-78707-1-PB.pdf), [Rutgers KR 2018 extended-abstract record](https://www.researchwithnj.com/en/publications/on-limited-conjunctions-in-polynomial-feature-logics-with-applica-2), [CEUR KR4L plenary-talk PDF](https://ceur-ws.org/Vol-3020/KR4L_paper_2.pdf), [Springer Nature 2019 journal version](https://link.springer.com/article/10.1007/s10472-019-09656-4), [Springer LPNMR 2024 volume](https://link.springer.com/book/10.1007/978-3-031-74209-5), [Springer LPNMR 2022 chapter](https://link.springer.com/chapter/10.1007/978-3-031-15707-3_18), [Springer CAV workshop chapter](https://link.springer.com/chapter/10.1007/978-3-031-21222-2_11), [ICLP 2025 EPTCS proceedings](https://cgi.cse.unsw.edu.au/~eptcs/Published/ICLP2025/Proceedings.pdf), [Dagstuhl ICLP 2018 proceedings](https://drops.dagstuhl.de/entities/volume/OASIcs-volume-64), and source links recorded with each recovered row. A missing metadata-index abstract does not establish that the publication itself lacks an abstract.

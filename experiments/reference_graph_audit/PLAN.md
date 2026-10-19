# Reference graph feasibility audit

Question: can publicly recorded outgoing references add a usable relation between
papers beyond title/abstract scores? This audit measures access, identity precision,
coverage and graph support. It does not train a preference model or infer hidden labels.

The cohort contains all 137 incumbent Label-1 papers selected by all 11 previously
audited models, one distinct incumbent TP control per paper, and 20 additional training
and 20 test papers selected with seed 20261018. Controls share outer fold and venue;
choose the closest publication year, then closest incumbent P(read), then string ID.
Save the year gap; report exact-year and within-one-year subsets separately.

Collect OpenAlex metadata using only paper titles. Store raw responses and timestamps.
Accept identity only with normalized title similarity >= .96, token Jaccard >= .90,
year difference <= 1, and no competing qualifying distinct work. If both records provide
authors, require overlapping surname evidence; use the existing verified author ledger
where available. If multiple qualifying versions exist, an exact dataset DOI may resolve
the ambiguity. Otherwise reject ambiguous matches. Do not convert
HTTP failures into evidence of missing metadata. Stop on an access/budget block.

Measure outgoing-reference availability, DOI/type/topic/author-ID coverage and local
shared-reference edges. Contrast FP/control coverage, reference counts and local edge
support descriptively; count differences alone do not establish predictive value.
Every sample has its cohort and original source identity. External requests never include
labels, preference scores, cohort names or user information.

No present-day incoming citation count or future citation relations are features.
Any later supervised graph feature must aggregate ratings only from its current fit
partition, excluding the query publication group, and be compared with the incumbent
under nested grouped CV. References point to publications that existed when the citing
paper was written, though metadata corrections are retrieved today. OpenAlex inferred
topics and disambiguated authors are collected for coverage only in this audit.

Deliverables: cohort ledger, matched-pair ledger, resumable raw cache, identity ledger,
coverage tables, graph-edge audit and a feasibility decision. No classifier or submission.

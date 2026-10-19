# Crossref DOI reference collection

This is a second, independent source for outgoing references while the OpenAlex
anonymous quota is exhausted. It queries each distinct supplied DOI at Crossref's
direct `/works/{doi}` endpoint, at one request per second, and stores every response
append-only. The only transmitted field is the DOI; labels, scores and test outcomes
are never sent.

Accept a record only when returned DOI equals the supplied canonical DOI and the
normalised title similarity to the supplied title is at least .90. Its graph node is
the canonical source DOI. Its outgoing links are only canonical DOI values present in
Crossref's deposited `reference` field. A missing or empty `reference` field is not a
network failure. Request failures remain distinct and are retried on a later run.

After collection, use the same grouped cross-fitting and A/B/C controls as the
OpenAlex experiment: A is the incumbent; B gets identity/reference-availability and
reference-count metadata; C additionally gets shared-reference similarity. Do not
mix Crossref DOI nodes and OpenAlex work-ID nodes in one graph feature.

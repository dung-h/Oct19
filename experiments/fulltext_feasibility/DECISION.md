# Decision after the public-full-text feasibility audit

The current Crossref cache cannot support a fair full-text feature experiment.
Among the deterministic DOI sample, only 35.6% has a publisher PDF pointer and
coverage is strongly venue-dependent: the train sample is 80.0% for LICS but
0.0% for ICLP and KR.  A model feature built from these pointers would mostly
encode which publisher deposited a link, rather than paper content.

Semantic Scholar was attempted through its documented batch endpoint on 18 and
19 October 2026 and returned HTTP 429 on both single-batch attempts.  The OA
field therefore remains unavailable to this audit.  This cannot establish that
only 35.6% of all papers are publicly accessible; it establishes only that the
existing Crossref source is inadequate for the next pipeline stage.

Do not download or parse publisher PDFs, train a full-text model, or produce a
test prediction from this source.  A future feasibility audit may proceed only
with a lawful, rate-available OA index whose records can be matched by DOI and
whose coverage is measured across every venue before any feature extraction.

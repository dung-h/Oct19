# Outgoing-reference preference ablation

This is a controlled test of bibliographic coupling: papers are related when their
verified public outgoing-reference lists overlap. It is deliberately narrower than
the earlier coverage pilot.

## Collection and identity

The collector starts from every one of the 3,090 supplied records. It first issues
batched exact-DOI queries, then title-search batches only while the documented
anonymous OpenAlex quota permits. Every raw response, response status and rate-limit
header is append-only. A failed request is not evidence that a work or references do
not exist.

The identity rule is the pilot's rule: normalised title similarity >= .96, token
Jaccard >= .90, year gap <= 1, surname overlap when both sides have author evidence,
and one qualifying work only (or a dataset DOI resolving competing versions). The
existing verified-author ledger may replace a known-bad dataset author field only for
this identity check. No label, score, cohort or user data is transmitted to OpenAlex.

No incoming citation count, present-day citation total, inferred topic or author
centrality is used as a predictive feature.

## Features and leakage controls

For a query paper, calculate Jaccard overlap between its outgoing references and each
eligible fit paper's outgoing references. Aggregate maximum and top-5 mean overlap
against Label 1 (N), Label 2--5 (P), and Label 4--5 (G), together with P-N/G-N
margins and non-label coverage indicators. Exclude the query's duplicate-publication
group from every reference bank.

Training feature rows are made by the existing grouped inner cross-fit views: each
query's graph statistics see labels only in the corresponding reference fold. A
validation/test prediction uses the same three fit-reference views and averages the
three model probabilities. Thus it cannot read its own label, an outer-validation
label, or a test label.

If the query has no verified nonempty outgoing-reference list, its final probability
is replaced exactly with the immutable incumbent probability. The graph model is
therefore evaluated only where the public relation exists, while the whole-set result
cannot lose merely because coverage is incomplete.

## Comparison and decision

Use the incumbent 277-column feature pack and exactly its CatBoost recipe. The only
changed input is the graph feature family. Select the read threshold from the four
inner folds only, using official H. Report H/Q/R, AUC, selected/TP/FP/FN, Label 4--5
misses, fold effects, selection transitions and the graph-covered subgroup.

The result is exploratory because outer folds were previously inspected. Do not
promote a submission unless the mean H gain is at least .005, at least three of five
outer folds improve, and missed Label 4--5 papers do not increase. If coverage is
too low or no criterion is met, retain the incumbent and record the negative result.

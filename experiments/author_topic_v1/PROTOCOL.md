# Author × topic feasibility and pilot

This package implements branch B of `experiments/qwen_head_v1/MASTER_PLAN.md`.
All work is CPU-only. B0 reads documents and group memberships but never
parses target labels. B1 may run only if the metadata identity check is
credible and both partitions pass the pre-registered author×topic support
gate. Prediction labels are joined only after both B1 partition outputs are
complete.

## B0 lock

- Split author strings on commas, normalize with Unicode NFC, casefold and
  whitespace collapse. Keep full names as distinct identities; never merge
  surnames, initials, or uncertain strings. Missing and uncertain names remain
  separate and are excluded from identity histories.
- Before inspecting labels, choose 30 records with NumPy seed 20261102. Query
  Crossref by title and preserve raw response, query, DOI, publisher and author
  evidence. Use one public-pool query every 1.2 seconds and retry HTTP 429 with
  backoff. This sample documents source reliability; it does not verify the
  whole author bundle.
- For each partition fit TF-IDF on capped title/abstract text from fit IDs
  only (1–2 grams, min_df 2, max 20,000), then randomized SVD 32 and KMeans 8
  (seed 20261102, n_init 10). Assign validation topics with those fit-only
  transformations.
- Author and author×topic support counts distinct fit publication groups and
  excludes the query group. The practical gate requires at least 20% of
  prediction documents to have one author×topic history supported by two or
  more reference groups in each partition. “Known author” coverage means a
  parseable full name with at least one reference group; cold named authors
  are reported separately.
- The metadata gate is set before execution: at least 80% of sampled rows with
  nonmissing authors must have a high title match to Crossref and every
  parseable raw full name represented in Crossref authors. A missing or
  unresponsive source does not count as verified.

If either gate fails, record the evidence and stop B before quality fitting.
Do not add more author stats, change topic count, or read prediction labels to
try to pass.

## B1 if B0 passes

Use three CPU arms with the same TF-IDF text and multinomial five-class
LogisticRegression (max_iter 2000): B-text, B-author, and primary B-topic.
Choose one shared C from `[.01,.1,1,10]` using pooled group-CV cross-entropy
of B-text only; ties within 1e-6 choose smaller C.

For each training set, build three group-crossfit rows for training target-
derived author features; every query row may use labels from reference rows
only, excluding its publication group. During inner CV, refit topic/text
transforms inside inner-train and rebuild author history within that scope.
Prediction histories use only the applicable fit set. Persist reference IDs,
groups, support, priors, and source lineage. Shrink author probabilities to
the fit prior with alpha 20; author×topic probabilities to author history
with alpha 10. Missing author falls back to prior; cold topic falls back to
author history with flags. No adapted Qwen in-sample scores enter these arms.

Evaluate only after both partitions are scored. Use all five probabilities,
read cutoff .55, expected-gain ranking and the same incumbent baseline. The
B-topic gate is the master-plan gate: mean CE improves over B-author by at
least .01, read AUC does not fall, mean H exceeds incumbent by .003, each
partition H is nonnegative, missed labels 4 and 5 do not rise separately, and
lineage/technical checks pass. B-author cannot be promoted if B-topic fails.

# Preference augmentation: protocol before evaluation

## Scope and assignment

Predict selection and reading order for the 596 test IDs, using the original 2,494 training labels. Classification (CatBoost multiclass) and regression (Ridge control) are substantive course methods. Optimize the detailed Evaluation metric H=2QR/(Q+R), not accuracy or title diversity. Produce reproducible source, report, a candidate CSV, and a decision about replacing the existing submission. Never edit labels or hand-predict validation/test preferences.

## Questions and fixed comparisons

1. Does explicit, evidence-backed content help beyond the same base learner?
2. Does local preference history help beyond the same base learner?
3. Does their combination help? Does the whole inner-selection procedure beat the existing workflow?

Four matched CatBoost arms: base, content, neighbors, both. Base uses title TF-IDF SVD32, cross-fitted Ridge title/venue score, venue and year. Content adds fixed lexical indicators for document type, problem, method, contribution and domain, separately for title/usable abstract, plus evidence coverage. Unknown means no explicit lexical evidence, not a negative assertion about a paper. No LLM, synthetic papers, paraphrases or new preference labels. These features are a limited lexical extraction baseline, not a complete semantic understanding of contribution.

Neighbors use cached frozen SPECTER2 **retrieval** vectors plus separately fit title/abstract TF-IDF. Use top5/top15 globally and within venue. Include smoothed five-label distributions, weighted gain, disagreement, similarity and support counts. Same duplicate group is always excluded. Retrieval cache is target-independent; vocabularies and any target-derived features are fitted within the relevant reference training partition.

## Leakage controls

Reuse exactly the existing 5 outer / 4 inner grouped splits. For every fit F -> prediction V, construct training target-derived features by THREE further grouped cross-fitting splits within F. Each query's history and Ridge score come exclusively from other groups. Build V features from F only, excluding any query duplicate groups. The validation/test labels are never accessed by feature construction. Full-reference prediction has more available history than cross-fitted training; expose reference counts and record this limitation.

Content dictionary is fixed before evaluation and uses general scientific categories. It was designed after earlier exploratory failures, so it is not a pristine independent hypothesis test. Store exact evidence spans with offsets and source fields; verify every span. Original raw input and previous artifacts stay intact.

## Fixed models and selection

CatBoost 1.2.8, CPU, multiclass labels1..5; depths4/6, 300 iterations, learning_rate0.05, l2_leaf_reg8, seed20261018, no outer early stopping. Both configurations are used in all four arms. Train only on original labels. Selection threshold is P(y>1)>t, t=0,.05,...,.90; order by expected gain sum P(y=k)*(k-1)/4. Inner mean official H chooses configuration/threshold per arm and across arms.

Fixed Ridge control uses L2[normalized SPECTER2 title, venue onehot], alpha0.01; cutoff gain0..0.625 step0.025 chosen on the same inner folds. Also report the existing complete workflow's cached H0.6289581993614293 as a historical paired reference. Do not confuse the score of a retrospectively best arm with the full selection procedure.

## Execution and stopping

1. Export evidence-backed content features and coverage; verify inputs/caches and labels.
2. Implement and test group exclusion, cross-fitting, label isolation and official metric.
3. Run the four matched arms plus Ridge, with all choices made in inner folds. Cache predictions and lineage for resumability.
4. Recompute scores/selection independently; report paired fold changes, repaired/new FP/FN, high-label misses, venues and known failure cases. Known cases are diagnostic only.
5. Repeat inner selection on all train, fit selected candidate, export 596 test priorities. Reload and independently rebuild features to verify identical predictions/ranking.

Do not expand hyperparameters or extraction rules after seeing outer results. A practical success target (not an assignment requirement or promised result) is full-procedure delta H>=0.02 vs historical workflow, improvement on >=4/5 paired folds, with recall/high-label tradeoffs disclosed. If unmet, retain the existing deliverable and label the new CSV experimental; still finish analysis and final verification. No Kaggle upload.

## Deliverables

`run_augmentation.py`, `features.py`, `test_features.py`, `verify.py`, locked protocol/hash metadata, content evidence/coverage, cached fold features/predictions, metrics/OOF/failure transitions, final model and candidate CSV, `RESULTS.md`, reproduction instructions.

Administrative correction before any augmentation outer score was inspected: rename the entry module to avoid Python's `run_experiment` import collision with the existing baseline. No feature, model, split, threshold or selection setting changed. The prior protocol is archived for provenance.

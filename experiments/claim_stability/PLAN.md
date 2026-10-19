# Stability and verbatim contribution features

## Purpose

Act on the completed plateau diagnosis: assess a small regularization change and add actual candidate problem/method/result sentences rather than more topic flags. Original preferences remain unchanged. This is classification for the assignment; evaluation uses the detailed official harmonic-mean metric. No new labels, synthetic papers, ID-specific fixes, architecture search or Kaggle upload.

## Five fixed arms

1. `reference`: existing277 features, CatBoost depth4/l2_leaf_reg8.
2. `regularized`: same277 features, depth3/l2_leaf_reg32.
3. `repaired`: regularized head, education indicators require explicit educational wording rather than standalone student; add tutorial document-type flags and update unknown indicators.
4. `prefix`: repaired arm plus frozen embeddings of the first six abstract sentences, arranged as three blocks of at most two sentences each.
5. `claims`: repaired arm plus frozen embeddings of up to two candidate problem, method and result sentences per aspect, using fixed generic discourse cues and exact evidence spans.

Every head uses300 iterations, learning_rate0.05, multiclass original labels1..5, CPU4 threads. Regularization is a package change (depth and L2); do not attribute its effect to either setting alone. No early stopping or extra parameter grid.

## Text representation and audit

All new text is verbatim from existing usable abstracts. Fixed generic verbs/cues identify candidates, not verified semantic triples or the reason for personal preferences. No filler abstract when an aspect is missing. Preserve unknown masks and offsets. Do not add phrases for individual paper IDs or use labels to extract sentences.

Both text arms use the SAME frozen cached MiniLM checkpoint/revision, attention-mask mean pooling, max128 tokens per block, up to three blocks per paper, and L2-normalized vectors. Fit separate SVD16 per block inside each model training partition, plus three presence indicators (51 extra features in either arm). Prefix is a position-based matched-encoder/control with the same maximum block/token budget, not an exactly equal word-count control. Report actual coverage and truncation. Differences cannot be attributed exclusively to semantic role extraction if the amount of text differs.

Record repaired-feature ledger, all candidate sentences/evidence, role coverage and previously reviewed20-pair probes. Those pairs helped motivate the design and are diagnostic probes, not a held-out extractor-accuracy set. Show whether earlier indistinguishable category pairs now have different verbatim text; never infer corrected preference labels.

## Evaluation and stability

Reuse exactly five outer/four inner grouped folds and the verified existing277-feature caches. Their target-derived histories/Ridge scores were cross-fitted within the relevant fit partition. Verify data/source hashes and fit/query indices before reuse. Repairs are label-independent; every SVD is trained on fit vectors only. No validation/test labels enter features or optimizers.

Primary model seed20261018. For each arm choose P(read)>threshold in0,.05,...,.90 by mean four-inner-fold H, ranking by expected gain. Also choose an overall arm in inner CV; report its complete outer score rather than the retrospectively best arm. Existing primary reference probabilities can be reused verbatim; primary reference training probabilities are available in the verified learning-curve full-size caches.

Seeds20261019/20261020 refit **outer training partitions only**, with primary inner choices/cutoffs held fixed. This isolates head-seed sensitivity. They are not independent new CV splits or separately tuned procedures. Report all seeds, their mean/range and decision disagreement; do not pick the best seed. Claim/projection inputs and crossfit memberships stay fixed across seeds. The final model uses the primary seed, not a seed selected from outer scores.

Report H/Q/R, FP/FN and missed labels4/5, read AUC, multiclass log-loss, train/validation AUC gap, and effects regularized-reference, repaired-regularized, prefix-repaired, claims-repaired, claims-prefix. The reused outer data was repeatedly inspected before; scores are exploratory and not an independent leaderboard guarantee. Increased model options can worsen selection.

## Final and verification

Repeat arm/cutoff selection on the SAME four all-train inner folds already saved in preference_augmentation. Fit selected arm on all2494 labels, export an experimental596-ID candidate without overwriting prior models/submissions. Rebuild features independently after loading the final model and compare exact selection/full order. No arbitrary+0.02 automatic-rejection rule: recommend based on paired effects, seed sensitivity, complexity and limitations, and clearly distinguish evidence from uncertainty.

Before declaring completion, verify original labels, source/plan hashes, all new evidence offsets, fit-only SVD indices, inherited grouped crossfit lineage, inner decisions, outer metrics for all arms/seeds, and reloaded final inference. Produce a Vietnamese report, failure transitions and a single reproducible runner. Do not expand the experiment after inspecting its outer results.

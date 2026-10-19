# Diagnose the plateau before changing models

## Fixed scope

Two diagnostics, no hyperparameter search and no new submission: (1) learning curves for the existing Ridge and CatBoost-both, (2) a content audit of nearby training papers with contrasting original preferences. Preserve original labels and all previous artifacts. This supports the assignment's regression/classification, evaluation and error-analysis requirements. The detailed harmonic-mean Evaluation remains the official metric used locally; its conflict with the copied short Overview is unresolved.

## Learning curves

- Reuse all five saved outer grouped folds; outer validation membership stays fixed at every sample size.
- Within each outer fit, use StratifiedGroupKFold(4, shuffle=True) with subset seeds20261019/20261020. Take the first1/2/3/4 groups of folds as nested approximately25/50/75/100 percent subsets. The fractions are by fold, so actual row counts can differ slightly. Entire duplicate groups remain intact. The 100-percent dataset is identical for both subset seeds and is fit/aggregated once.
- Ridge: normalized SPECTER2 title plus venue, alpha0.01, lsqr tol1e-6. CatBoost: existing both features, depth4, iterations300, learning_rate0.05, l2_leaf_reg8, seed20261018, multiclass, CPU. No tuning or early stopping on outer labels.
- Rebuild all history features, reference vocabularies, Ridge scores and SVD in each subset; training target-derived features use the existing three grouped cross-fitting splits. Outer features only reference the selected subset. The full-size feature cache may be reused after checking indices and code/data hashes.
- Record training and validation rating MAE/RMSE, read AUC and official H. CatBoost also records multiclass/binary log-loss. Training CatBoost predictions are in-sample for the head, evaluated on its cross-fitted input features; full-reference validation inputs have more history references. This difference is a possible source of the gap, not proof of overfitting alone.
- Use **fixed diagnostic cutoffs**: Ridge expected gain>0.25; CatBoost P(read)>0.50 and expected-gain order. Do not optimize H cutoffs for these curves or reuse all-data thresholds. Curves diagnose representation/sample efficiency and are not scores for a newly tuned submission.
- Primary summaries average the two subset seeds within each outer fold, then average the five folds. Do not count repeated subsets as independent validation sets. Report fold variation descriptively, without an independence-based confidence claim. Compare25->100 and75->100 changes and train/validation gaps. The existing architecture was selected after previous exploration on these folds: this is exploratory diagnosis, not a new untouched benchmark.

## Content audit

- Select20 disjoint pairs, four per venue, from train only. Each pair has one original label1 and one original label4/5, distinct duplicate groups, and is greedily selected by highest cached retrieval cosine within venue, using text IDs for deterministic ties. No selection by model error or case ID. Add previously discussed cases separately if useful; never include them in the sample counts unless selected by the rule.
- Export titles, usable abstracts, similarity, original labels, lexical category differences and word-overlap diagnostics. This is a deliberately difficult, selected sample: it does not estimate population label noise or the prevalence of missing information.
- Extract candidate problem/method/result sentences by fixed discourse cues, keeping exact offsets/evidence. These spans are diagnostic and are **not new model features in this run**; a cue is not a verified semantic interpretation.
- Read all20 pairs. Record observable differences in problem/method/result/application, available evidence and whether the coarse lexical categories represent those differences. Never infer a new reading-preference label, claim the rater's actual rationale, or hand-correct validation/test decisions.
- Distinguish observable content differences from explained preference differences. Even when text differences are clear, the reason behind the original preferences can remain unknown.

## Completion and decision

Verify subset nesting/group isolation, original labels, history lineage, all metrics from saved predictions, and full-size cache/model prediction agreement. Produce standalone PNG/PDF learning-curve plots, per-fit metrics, a reviewed pair table with original evidence, and a Vietnamese conclusion. State which hypotheses the evidence supports and which remain unresolved. No new model family or extractor is introduced based on outer outcomes. No Kaggle upload.

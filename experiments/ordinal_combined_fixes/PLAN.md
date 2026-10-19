# Combined fixes and reference-count intervention

Locked before scoring this experiment, 2026-10-16. This is an exploratory follow-up on repeatedly inspected outer data, not a new independent holdout.

Use detailed assignment Evaluation H (harmonic mean of rank-weighted relevance and positive-label recall). The short Overview's 60/40 description remains unresolved. Preserve original labels, source data, five grouped outer folds, four inner folds, final inner memberships, feature-cache lineage and seed 20261018. No Kaggle upload.

Reuse four fixed controls: multiclass full 277; ordinal full with explicit Bayesian bootstrap, temperature 1, one leaf-estimation update; multiclass without 150 neighbor features; ordinal without those features with its original MVS/10-update defaults.

Fit exactly four new configurations:

1. ORD_combined: explicit Bayesian/1-update settings plus removal of all neighbor features (127 columns), depth 4, L2 8.
2. ORD_combined_regularized: same but depth 3 and L2 32.
3. ORD_count_fraction: full features with each of six support counts replaced by support/references and six raw reference-count columns removed (271 columns), explicit Bayesian/1-update settings, depth 4/L2 8.
4. MC_count_fraction: identical count transformation, multiclass, explicit Bayesian/1-update settings, depth 4/L2 8.

Counts are divided row by row within the same channel/scope. Zero references produce zero support fraction. No labels or fitted statistics are used in this transformation. Keep all other neighbor features and cross-fitted ridge unchanged. This tests the representation of raw sample-size features; it does not equalize the actual reference corpora, class evidence, nearest neighbors, or all distribution shifts caused by reference size.

All new models use 300 iterations, learning rate .05, CPU 4 threads. Four ordinal targets y>=2,3,4,5 use the existing label-independent equal-weight decreasing PAV. Read probability is q2; ranking gain is mean(q2..q5). Each family's cutoff is selected from 0 to .9 by .05 using mean H over four inner folds. A separate selection policy chooses among all eight families using only inner mean H; outer scores never choose cutoffs or families. Ties follow listed family order and ascending cutoff. Report both individual-family means and the actual selection policy.

Record per-fold H, Q, recall, AUC, training AUC, FP/FN and missed labels 4/5. Predeclared contrasts: combined versus each single fix; regularized versus combined; count fraction versus full within each formulation. Do not add isolated improvements arithmetically. Refit final candidates using the same four final inner folds. Save models, recipes and CSVs. Independently recompute thresholds/metrics/PAV, verify source hashes and labels, rebuild final features and reload final models before reporting. Keep the existing incumbent available regardless of outcome.

# Evidence-backed preference augmentation: completed experiment

## Result and decision

retain previous deliverable; candidate experimental. Full inner-selected procedure delta H vs previous workflow: **+0.00838**, improving **4/5** paired folds. The target +0.02 was a practical research stopping criterion, not an assignment requirement. Recall/high-label tradeoffs remain visible below. No Kaggle upload or previous artifact replacement.

| family              |       H |   H_std |       Q |       R |   FP |   FN |   missed_high |     AUC |
|:--------------------|--------:|--------:|--------:|--------:|-----:|-----:|--------------:|--------:|
| base                | 0.63423 | 0.01863 | 0.53144 | 0.78738 |  328 |  338 |            45 | 0.79788 |
| content             | 0.63698 | 0.01651 | 0.52573 | 0.81000 |  354 |  302 |            39 | 0.80059 |
| neighbors           | 0.63777 | 0.01562 | 0.51502 | 0.83782 |  398 |  258 |            30 | 0.80101 |
| both                | 0.64028 | 0.01120 | 0.53250 | 0.80324 |  338 |  313 |            38 | 0.79946 |
| ridge               | 0.63483 | 0.01819 | 0.53134 | 0.79076 |  325 |  334 |            42 | 0.79291 |
| selection_procedure | 0.63734 | 0.01667 | 0.52776 | 0.80541 |  346 |  309 |            38 | 0.80272 |

Each family above independently chooses its depth and threshold using four inner folds. `selection_procedure` chooses among all arms and Ridge in those inner folds. Its outer score estimates the complete selection procedure; the retrospectively best family score is not the final candidate score. Existing outer labels were repeatedly inspected in earlier research: this is not a pristine independent benchmark or a claim about hidden Kaggle performance.

## Matched augmentation effects

| arm                 | reference         |   delta_H |   folds_improved |   delta_R |   delta_FP |   delta_FN |   delta_high_FN |
|:--------------------|:------------------|----------:|-----------------:|----------:|-----------:|-----------:|----------------:|
| content             | base              |   0.00275 |                3 |   0.02262 |         26 |        -36 |              -6 |
| neighbors           | base              |   0.00355 |                2 |   0.05044 |         70 |        -80 |             -15 |
| both                | base              |   0.00605 |                3 |   0.01586 |         10 |        -25 |              -7 |
| both                | content           |   0.00330 |                4 |  -0.00676 |        -16 |         11 |              -1 |
| both                | neighbors         |   0.00250 |                3 |  -0.03458 |        -60 |         55 |               8 |
| selection_procedure | ridge             |   0.00251 |                4 |   0.01465 |         21 |        -25 |              -4 |
| selection_procedure | previous_workflow |   0.00838 |                4 |  -0.00215 |        -25 |          2 |              -5 |

All four CatBoost arms share the same learner/grid, base features and target. Content effects are measured against `base`, not only against a different Ridge learner. Neighbor effects likewise keep the classifier unchanged. Interactions are examined by both-versus-single comparisons; these are paired descriptive comparisons, not independent significance tests.

## What was augmented

No synthetic paper or preference label was created. The original 2,494 labels were preserved. Content rules record literal lexical evidence for document type, problem, method, contribution and domain, separately from title and usable abstract. Every indicator has a source field, exact text and offsets in `content_evidence.csv`; lack of a match becomes unknown, not a confirmed semantic negative. The fixed dictionary cannot resolve all subtle contribution differences between same-topic papers.

| field    | aspect       |    n |   with_evidence |   unknown |
|:---------|:-------------|-----:|----------------:|----------:|
| title    | type         | 2494 |              89 |      2405 |
| title    | problem      | 2494 |             825 |      1669 |
| title    | method       | 2494 |             853 |      1641 |
| title    | contribution | 2494 |             249 |      2245 |
| title    | domain       | 2494 |             350 |      2144 |
| abstract | type         | 2494 |              78 |      2416 |
| abstract | problem      | 2494 |            1809 |       685 |
| abstract | method       | 2494 |            1473 |      1021 |
| abstract | contribution | 2494 |            1701 |       793 |
| abstract | domain       | 2494 |             909 |      1585 |

History features use SPECTER2 retrieval (not the regression adapter for nearest neighbors), title TF-IDF and abstract TF-IDF; top5/top15 globally and within venue. Features include five-label distributions, weighted gain, entropy/disagreement, support and nearest per-class similarity. Three extra grouped cross-fitting folds create every training history and Ridge score. Prediction references contain only the fit set, with the query's entire duplicate group excluded. Two groups overlap train and test; their label histories are excluded for affected test queries. The ordinary Ridge control is trained on all original fit labels.

Base features are venue, year, 32 title TF-IDF SVD coordinates and a cross-fitted title/venue Ridge rating. Target-free SVD/vocabulary are fitted inside each model training partition. History vocabularies are fitted separately on each reference subset. Training crossfit references are smaller than full-fit prediction references, a remaining distribution shift; reference counts are exposed to the learner.

## Failures and limitations

Old binary errors **678**, new errors **655**; fixed **188**, introduced **165**. Binary error counts supplement, rather than replace, official H.

|   id | title                                                                                                                                                                                                                           |   Label | old_status   | new_status   |
|-----:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|:-------------|:-------------|
|  206 | Explainable and Trustworthy Traffic Sign Detection for Safe Autonomous Driving: An Inductive Logic Programming Approach.                                                                                                        |       1 | FP           | FP           |
|  217 | Epistemic Logic Programs: a Novel Perspective and Some Extensions.                                                                                                                                                              |       1 | FP           | FP           |
|  355 | Epistemic Logic Programs: A Different World View.                                                                                                                                                                               |       5 | TP           | TP           |
| 1000 | Exploring Antimicrobial Resistance Prediction Using Post-hoc Interpretable Methods.                                                                                                                                             |       5 | FN           | FN           |
| 1001 | Towards Understanding ICU Treatments Using Patient Health Trajectories.                                                                                                                                                         |       4 | TP           | TP           |
| 1045 | Artificial Intelligence in Medicine: Knowledge Representation and Transparent and Explainable Systems - AIME 2019 International Workshops, KR4HC/ProHealth and TEAAM, Poznan, Poland, June 26-29, 2019, Revised Selected Papers |       5 | FN           | FN           |
| 1184 | Knowledge Representation for Health Care - HEC 2016 International Joint Workshop, KR4HC/ProHealth 2016, Munich, Germany, September 2, 2016, Revised Selected Papers                                                             |       1 | FP           | TN           |
| 2383 | Soundness of reset workflow nets.                                                                                                                                                                                               |       5 | FN           | FN           |

Known examples were already inspected and are diagnostic only; they did not receive hand labels, special weights or ID-specific rules. Label disagreement between similar texts does not prove label corruption. A negative result for this limited lexical augmentation or history summary does not prove that full semantic extraction or every local preference model will fail. Conversely, positive aggregate effects cannot establish the researcher's true rationale.

Per-arm priorities are preserved in `known_cases.csv`. In this run, local history rescues ID1000 (antimicrobial prediction) and ID2383 (workflow-net soundness) in the neighbors arm, but the globally inner-selected arms still exclude them. ID217 remains a false positive across the four CatBoost arms; ID1045 remains excluded across them. This separates two issues: some useful local signals are present but not reliably retained by model selection, while other hard cases still lack useful discrimination in these augmented representations. No manual per-ID blending is used.

An extraction audit exposes a concrete limitation: IDs217/355 have **zero different content-category flags**, despite their different preference labels; the model can still distinguish their base title SVD/Ridge values, but this augmentation adds no category distinction for that pair. IDs1045/1184 differ in 3 content flags. This prevents claiming that the extractor captures the exact contribution differences needed for every hard case. See `content_pair_audit.csv`.

## Final candidate

Selected on four grouped inner folds of all training data (seed20261118), without selecting directly by outer results:

```json
{
  "key": "both_d4",
  "family": "both",
  "depth": 4,
  "threshold": 0.55,
  "inner_score": 0.6386873823729493
}
```

`outputs/submission_candidate.csv`: 596 template IDs, 356 positive priorities. Model/recipe and features are retained. The candidate is experimental; retain the prior submission under the predeclared decision criterion.

## Verification and reproduction

Verified 13994 evidence spans, 90 crossfit partitions, 30 outer metrics, all inner choices, label perturbation invariance, original labels, independently rebuilt final features and identical full ranking after model reload. Six leakage/evidence/metric unit tests also passed.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/preference_augmentation/run_plan.py
```

See [PLAN.md](PLAN.md) for the pre-evaluation protocol, and [outputs/verification.json](outputs/verification.json), [outputs/paired_effects.csv](outputs/paired_effects.csv), [outputs/failure_transitions.csv](outputs/failure_transitions.csv). Classification/regression satisfy the assignment's substantive-method requirement. Report both successful and unsuccessful comparisons; use the detailed harmonic-mean Evaluation because the copied short Overview has a conflicting metric description.

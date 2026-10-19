# Semantic roles: controlled classification experiment

Primary giảm tổng lỗi nhưng chưa đạt đầy đủ rule FP/FN/high-label/folds; giữ incumbent.

## Counts and assignment metric

| branch      | accepted   | dropped   | FP   | FN   | total_error   | missed_high   | H       | Q       | R       |
|:------------|:-----------|:----------|:-----|:-----|:--------------|:--------------|:--------|:--------|:--------|
| incumbent   | 1616       | 878       | 338  | 313  | 651           | 38            | 0.64028 | 0.53250 | 0.80324 |
| text_raw    | 1651       | 843       | 362  | 302  | 664           | 38            | 0.63608 | 0.52371 | 0.80998 |
| roles_raw   | 1655       | 839       | 373  | 309  | 682           | 40            | 0.63328 | 0.52181 | 0.80560 |
| text_blend  | 1625       | 869       | 337  | 303  | 640           | 33            | 0.64303 | 0.53353 | 0.80935 |
| roles_blend | 1652       | 842       | 352  | 291  | 643           | 34            | 0.64198 | 0.52882 | 0.81704 |

Counts sum2494 OOF rows; H/Q/R are means of five fold metrics. Ranking always uses incumbent expected gain. Cutoffs selected using four-inner OOF only, subject to incumbent FN/high-FN budgets; constraints are not guaranteed on outer/test.

## Error transitions vs incumbent

| branch      |   FN_rescued |   FP_removed |   new_FP |   lost_TP |   delta_error |
|:------------|-------------:|-------------:|---------:|----------:|--------------:|
| text_raw    |          117 |           68 |       92 |       106 |            13 |
| roles_raw   |          121 |           66 |      101 |       117 |            31 |
| text_blend  |           58 |           45 |       44 |        48 |           -11 |
| roles_blend |           62 |           34 |       48 |        40 |            -8 |

Primary reduces total error on 2/5 outer folds. All four branches were declared in advance; primary is roles_blend, not the best outer branch.

## Matched representation contrasts

| arm         | reference   |   delta_FP |   delta_FN |   delta_error |     delta_H |
|:------------|:------------|-----------:|-----------:|--------------:|------------:|
| roles_raw   | text_raw    |         11 |          7 |            18 | -0.00280317 |
| roles_blend | text_blend  |         15 |        -12 |             3 | -0.00105088 |
| roles_blend | incumbent   |         14 |        -22 |            -8 |  0.00169939 |

## Evidence extraction

| split   | role   |   total |   present |
|:--------|:-------|--------:|----------:|
| test    | object |     596 |       587 |
| test    | method |     596 |       577 |
| test    | result |     596 |       573 |
| train   | object |    2494 |      2451 |
| train   | method |    2494 |      2385 |
| train   | result |    2494 |      2363 |

Invalid extractions: 1. Each present role is an exact source sentence, not a generated paraphrase. This guarantees provenance, not semantic role correctness. Qwen receives title/abstract only; no ratings/IDs/metadata/predictions. Original full text retained in both feature arms. No SVD; all384 dimensions retained per dense block.

## Final artifacts

| branch      |   accepted |   dropped |   newly_selected |   newly_dropped |
|:------------|-----------:|----------:|-----------------:|----------------:|
| text_raw    |        377 |       219 |               73 |              52 |
| roles_raw   |        373 |       223 |               66 |              49 |
| text_blend  |        352 |       244 |               28 |              32 |
| roles_blend |        349 |       247 |               23 |              30 |

Test has no known ratings: only selected/dropped/changed-ID counts are reported. All596 IDs exported for every branch; no Kaggle upload or incumbent replacement.

Verifier PASSED: independent threshold and metric replay, exact evidence spans, final vocab refit,12 classifier reloads, and submission-order checks. Existing outer folds were repeatedly inspected; results are exploratory. Selected diagnostic pairs are not an unseen semantic benchmark.

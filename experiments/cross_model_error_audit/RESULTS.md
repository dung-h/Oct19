# Cross-model false-positive audit

This is a diagnostic of saved out-of-fold predictions from already examined folds. It trains no model, searches no ensemble weight, selects no final candidate, and does not use test labels.

## Baseline

The incumbent has 338 selected Label1 papers (FP), 1,278 selected worthwhile papers (TP), and 313 dropped worthwhile papers (FN). Every candidate was required to have exactly the same 2,494 IDs, labels and outer-fold assignment before comparison.

## Can another model act as a veto?

`incumbent_FP_removed` is the number of incumbent FP that the candidate rejects. `incumbent_TP_lost` is the cost: incumbent TP that it also rejects. `veto_*` scores retain the incumbent order and select only papers selected by both models; this is an explanatory counterfactual, not a tuned ensemble.

| model             |        H |        Q |        R |   accepted |   TP |   FP |   FN |   total_error |   missed_high |   delta_H |   incumbent_FP_removed |   incumbent_TP_lost |   incumbent_FN_rescued |   new_FP |   decision_changes |   veto_H |   veto_Q |   veto_R |   veto_FP_removed |   veto_TP_lost |   veto_net_correct |   veto_precision |
|:------------------|---------:|---------:|---------:|-----------:|-----:|-----:|-----:|--------------:|--------------:|----------:|-----------------------:|--------------------:|-----------------------:|---------:|-------------------:|---------:|---------:|---------:|------------------:|---------------:|-------------------:|-----------------:|
| history_neighbors | 0.637774 | 0.515024 | 0.837821 |       1731 | 1333 |  398 |  258 |           656 |            30 | -0.002502 |                      7 |                  11 |                     66 |       67 |                151 | 0.639775 | 0.534792 | 0.796463 |                 7 |             11 |                 -4 |         0.388889 |
| abstract_margin   | 0.640929 | 0.534186 | 0.801339 |       1606 | 1275 |  331 |  316 |           647 |            38 |  0.000653 |                     22 |                  31 |                     28 |       15 |                 96 | 0.638866 | 0.539392 | 0.783914 |                22 |             31 |                 -9 |         0.415094 |
| content_flags     | 0.636978 | 0.525732 | 0.809998 |       1643 | 1289 |  354 |  302 |           656 |            39 | -0.003298 |                     47 |                  59 |                     70 |       63 |                239 | 0.638658 | 0.548001 | 0.766096 |                47 |             59 |                -12 |         0.443396 |
| title_margin      | 0.642623 | 0.534873 | 0.805184 |       1612 | 1281 |  331 |  310 |           641 |            37 |  0.002347 |                     35 |                  50 |                     53 |       28 |                166 | 0.638869 | 0.545147 | 0.771844 |                35 |             50 |                -15 |         0.411765 |
| specter_title     | 0.630099 | 0.521043 | 0.802029 |       1638 | 1274 |  364 |  317 |           681 |            38 | -0.010177 |                     72 |                  88 |                     84 |       98 |                342 | 0.638197 | 0.557539 | 0.748619 |                72 |             88 |                -16 |         0.450000 |
| claims            | 0.638848 | 0.527449 | 0.812773 |       1648 | 1292 |  356 |  299 |           655 |            36 | -0.001428 |                     15 |                  32 |                     46 |       33 |                126 | 0.637514 | 0.537954 | 0.783472 |                15 |             32 |                -17 |         0.319149 |
| venue_metadata    | 0.632157 | 0.530446 | 0.783722 |       1572 | 1246 |  326 |  345 |           671 |            43 | -0.008119 |                     78 |                 100 |                     68 |       66 |                312 | 0.636422 | 0.558468 | 0.740842 |                78 |            100 |                -22 |         0.438202 |
| ordinal           | 0.635347 | 0.532601 | 0.788864 |       1594 | 1254 |  340 |  337 |           677 |            49 | -0.004929 |                     32 |                  64 |                     40 |       34 |                170 | 0.634439 | 0.543305 | 0.763506 |                32 |             64 |                -32 |         0.333333 |
| adapter_random    | 0.620891 | 0.498081 | 0.841175 |       1799 | 1336 |  463 |  255 |           718 |            43 | -0.019385 |                     84 |                 127 |                    185 |      209 |                605 | 0.631149 | 0.561738 | 0.724215 |                84 |            127 |                -43 |         0.398104 |
| retrieval_rbf     | 0.617967 | 0.494990 | 0.826207 |       1767 | 1313 |  454 |  278 |           732 |            36 | -0.022309 |                     52 |                  98 |                    133 |      168 |                451 | 0.632069 | 0.551297 | 0.742377 |                52 |             98 |                -46 |         0.346667 |
| neural_adapted    | 0.633184 | 0.566215 | 0.728052 |       1404 | 1155 |  249 |  436 |           685 |            67 | -0.007092 |                    135 |                 203 |                     80 |       46 |                464 | 0.625054 | 0.584601 | 0.677144 |               135 |            203 |                -68 |         0.399408 |

## Persistent incumbent false positives

The histogram below counts, for each incumbent FP, how many of the 11 distinct candidate streams reject it. A high rejection count indicates cross-model disagreement with the incumbent; zero means every tested stream still selects the paper. It does not prove a paper should be rejected, because candidates can share blind spots or use different thresholds.

|   models_rejecting |   incumbent_FP_count |
|-------------------:|---------------------:|
|                  0 |                  137 |
|                  1 |                   67 |
|                  2 |                   49 |
|                  3 |                   21 |
|                  4 |                   18 |
|                  5 |                   19 |
|                  6 |                   16 |
|                  7 |                    4 |
|                  8 |                    4 |
|                  9 |                    2 |
|                 10 |                    1 |

The shared errors are uneven by venue. This is descriptive: venue is already a model feature and the table does not establish a causal venue effect.

| venue   |   incumbent_FP |   selected_by_all_11 |   rejected_by_at_least_5 |   mean_models_rejecting |   persistent_share |
|:--------|---------------:|---------------------:|-------------------------:|------------------------:|-------------------:|
| kr      |            103 |                   52 |                       18 |                  1.6311 |             0.5049 |
| cav     |             79 |                   20 |                       12 |                  2.0633 |             0.2532 |
| iclp    |             79 |                   42 |                        3 |                  1.0633 |             0.5316 |
| lics    |             63 |                   16 |                       12 |                  2.3651 |             0.2540 |
| lpnmr   |             14 |                    7 |                        1 |                  1.0714 |             0.5000 |

## Untuned multi-model vote diagnostic

For transparency, the table reports every possible vote threshold. A paper is removed from the incumbent list only if at least `rejections_needed_to_veto` of the 11 streams reject it; incumbent ranking stays unchanged. The table is diagnostic only: choosing a threshold by its highest row would tune on the same outer labels and is therefore invalid for promotion.

|   rejections_needed_to_veto |        H |        Q |        R |    accepted |          TP |         FP |         FN |   total_error |   missed_high |   delta_H |   incumbent_FP_removed |   incumbent_TP_lost |   net_correct_decisions |   veto_precision |
|----------------------------:|---------:|---------:|---------:|------------:|------------:|-----------:|-----------:|--------------:|--------------:|----------:|-----------------------:|--------------------:|------------------------:|-----------------:|
|                    1.000000 | 0.590033 | 0.621820 | 0.569421 | 1040.000000 |  903.000000 | 137.000000 | 688.000000 |    825.000000 |    131.000000 | -0.050243 |             201.000000 |          375.000000 |             -174.000000 |         0.348958 |
|                    2.000000 | 0.626782 | 0.584639 | 0.680014 | 1284.000000 | 1080.000000 | 204.000000 | 511.000000 |    715.000000 |     80.000000 | -0.013494 |             134.000000 |          198.000000 |              -64.000000 |         0.403614 |
|                    3.000000 | 0.635888 | 0.563501 | 0.731743 | 1416.000000 | 1163.000000 | 253.000000 | 428.000000 |    681.000000 |     58.000000 | -0.004388 |              85.000000 |          115.000000 |              -30.000000 |         0.425000 |
|                    4.000000 | 0.641762 | 0.555218 | 0.761616 | 1485.000000 | 1211.000000 | 274.000000 | 380.000000 |    654.000000 |     46.000000 |  0.001486 |              64.000000 |           67.000000 |               -3.000000 |         0.488550 |
|                    5.000000 | 0.641378 | 0.547303 | 0.775233 | 1525.000000 | 1233.000000 | 292.000000 | 358.000000 |    650.000000 |     44.000000 |  0.001102 |              46.000000 |           45.000000 |                1.000000 |         0.505495 |
|                    6.000000 | 0.640008 | 0.541273 | 0.783390 | 1557.000000 | 1246.000000 | 311.000000 | 345.000000 |    656.000000 |     41.000000 | -0.000268 |              27.000000 |           32.000000 |               -5.000000 |         0.457627 |
|                    7.000000 | 0.639836 | 0.536325 | 0.793335 | 1589.000000 | 1262.000000 | 327.000000 | 329.000000 |    656.000000 |     39.000000 | -0.000440 |              11.000000 |           16.000000 |               -5.000000 |         0.407407 |
|                    8.000000 | 0.640334 | 0.534801 | 0.798293 | 1601.000000 | 1270.000000 | 331.000000 | 321.000000 |    652.000000 |     38.000000 |  0.000058 |               7.000000 |            8.000000 |               -1.000000 |         0.466667 |
|                    9.000000 | 0.640255 | 0.533576 | 0.800779 | 1609.000000 | 1274.000000 | 335.000000 | 317.000000 |    652.000000 |     38.000000 | -0.000021 |               3.000000 |            4.000000 |               -1.000000 |         0.428571 |
|                   10.000000 | 0.640183 | 0.532895 | 0.802033 | 1613.000000 | 1276.000000 | 337.000000 | 315.000000 |    652.000000 |     38.000000 | -0.000093 |               1.000000 |            2.000000 |               -1.000000 |         0.333333 |
|                   11.000000 | 0.640180 | 0.532635 | 0.802635 | 1615.000000 | 1277.000000 | 338.000000 | 314.000000 |    652.000000 |     38.000000 | -0.000096 |               0.000000 |            1.000000 |               -1.000000 |         0.000000 |

Case-level IDs/titles are in `incumbent_fp_consensus.csv`; per-model outcomes for all incumbent FP/FN are in `incumbent_error_by_model.csv`. Pairwise agreement is in `candidate_pair_agreement.csv`.

## Interpretation boundary

The result can justify a pre-registered, nested-CV ensemble experiment only when a candidate removes materially more incumbent FP than incumbent TP and the pattern is not merely one threshold tradeoff. It cannot justify choosing the best row above or writing ID-specific rules: these OOF labels and folds have already been inspected.

Verification passed: all sources align exactly on ID/label/fold and their hashes did not change during analysis. No Kaggle upload.

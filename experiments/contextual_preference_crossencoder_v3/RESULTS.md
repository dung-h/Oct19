# V3 full nested-CV result

All 5 outer folds and both arms completed: 50/50 fits passed the predeclared
finite-gradient and output-variation guards. The V3 protocol differs from V2 only by
replacing each prediction head's ReLU with `LeakyReLU(0.01)`.

| Arm | Mean H | Mean Q | Mean R | Read AUC | TP | FP | FN | Missed L4 | Missed L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent CatBoost | 0.640276 | 0.532500 | 0.803238 | 0.799457 | 1278 | 338 | 313 | 33 | 5 |
| V3 query-only | 0.616254 | 0.507642 | 0.836670 | 0.792189 | 1331 | 455 | 260 | 32 | 10 |
| V3 contextual | 0.625397 | 0.507375 | 0.852184 | 0.798878 | 1356 | 462 | 235 | 21 | 8 |

Contextual V3 beats query-only in mean H by 0.009143 and rescues 25 more positive
papers, so the reference context carries signal. It nonetheless loses to the incumbent
by 0.014879 H and loses in every outer fold. Relative to the incumbent, contextual V3
rescues 78 positives but accepts 124 additional Label-1 papers. The recall improvement
(+0.048946) therefore cannot offset the quality loss (-0.025125).

The predeclared conclusion is to retain the incumbent. V3 establishes that the prior
ReLU collapse was a real implementation failure and is fixed, but it does not justify a
final refit, submission, or further activation tuning. See `outputs/nested_cv/` for the
protocol, 50 guarded fit artifacts, thresholds, and outer-fold metrics.

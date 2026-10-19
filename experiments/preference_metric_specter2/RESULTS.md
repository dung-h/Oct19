# SPECTER2 preference metric-learning pilot

The pilot changes the geometry of hard positive/negative pairs, but harms the
held-out read gate.  It must not advance to nested CV.

| Held-out diagnostic | Frozen retrieval / control | Adapted metric model | Difference |
|:--|--:|--:|--:|
| Mean hard-pair cosine | 0.912108 | 0.490326 | -0.421782 |
| Fraction of pairs with lower cosine | – | 0.972222 | – |
| Read AUC | 0.718667 | 0.669139 | -0.049527 |

The adapter trained on 1,503 rows and 959 training-only hard pairs for two
fixed epochs.  The probe contains 495 held-out inner-fold rows and 324 hard
pairs.  No test rows or outer-validation labels entered pair mining or the
optimizer.

The predeclared continuation rule required both lower held-out hard-pair
cosine and higher held-out read AUC than the frozen logistic control.  The
first condition passed and the second failed, so `continuation_allowed=false`.
The plausible mechanism is over-separation: a direct cosine margin moves
nearby positive/negative papers apart, but it also distorts the broader
scientific neighbourhood on which a small read classifier relies.

No nested CV, final refit, test prediction, or Kaggle submission was run.

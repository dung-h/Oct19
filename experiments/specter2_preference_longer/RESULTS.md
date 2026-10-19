# SPECTER2 title-only: epoch 8 vs 16

The same title-only input, pretrained adapter, head, loss, optimizer, folds and seed were used. Each partition trained once to epoch16; epoch8 was read from that run. The archived epoch8 predictions were checked to match. Thresholds and the global family were selected by four inner folds. The outer folds are previously studied and this remains exploratory.

| Policy | H | Q | Recall | AUC | FP | FN | Missed labels 4/5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy              |       H |       Q |       R |     AUC |        FP |        FN |   missed_high |
|:--------------------|--------:|--------:|--------:|--------:|----------:|----------:|--------------:|
| adapted_8           | 0.63318 | 0.56622 | 0.72805 | 0.80838 | 249.00000 | 436.00000 |      67.00000 |
| adapted_16          | 0.63170 | 0.52851 | 0.78672 | 0.78297 | 349.00000 | 340.00000 |      51.00000 |
| incumbent           | 0.64028 | 0.53250 | 0.80324 | 0.79946 | 338.00000 | 313.00000 |      38.00000 |
| neural_selection    | 0.63318 | 0.56622 | 0.72805 | 0.80838 | 249.00000 | 436.00000 |      67.00000 |
| selection_procedure | 0.63277 | 0.56000 | 0.73871 | 0.80709 | 269.00000 | 419.00000 |      60.00000 |

## Epoch 16 versus epoch 8

- Mean outer H change: **-0.00148**; improved in **2/5** folds.
- Mean recall change: **+0.05866**; mean Q change: **-0.03770**.
- Total FN change: **-96**; missed labels4/5 change: **-16**.

## Next step from the locked rule

**STOP this extension; retain the existing incumbent.** The gate was predeclared as higher mean H, improvement on at least3/5 folds, no increase in missed high-rated papers, and mean recall loss no worse than.01. This is a conservative research stop/go rule, not part of the assignment metric. It does not claim the epoch is converged or guarantee a Kaggle gain.

Final inner choices:

```json
{
  "adapted_8": {
    "family": "adapted_8",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  },
  "adapted_16": {
    "family": "adapted_16",
    "epoch": 16,
    "threshold": 0.6000000000000001,
    "inner_score": 0.6284368893346014,
    "inner_recall": 0.792548159556038
  },
  "incumbent": {
    "family": "incumbent",
    "epoch": 0,
    "threshold": 0.55,
    "inner_score": 0.6386873823729493,
    "inner_recall": 0.8090378105564809
  },
  "neural_selection": {
    "family": "adapted_8",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  },
  "selection_procedure": {
    "family": "adapted_8",
    "epoch": 8,
    "threshold": 0.45,
    "inner_score": 0.6399621136029068,
    "inner_recall": 0.8296722384534565
  }
}
```

Global final family: `adapted_8`, checkpoint `8`. `outputs/submission_candidate.csv` is the selected candidate; the existing incumbent is preserved. No upload was made.

Independent verification: `PASSED`; 25 outer policy rows and 5 final choices recomputed; archived checkpoint8 replay, final-test epoch8 replay (matched independent epoch-8 replay; prior cache absent), labels, candidate format, and selected model reload checked. The original run had no saved final-test cache, so that one comparison used a fresh replay of the identical epoch8 recipe. Full per-fold deltas are in `outputs/epoch16_vs8_by_fold.csv`.

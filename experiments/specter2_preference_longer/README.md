# Checkpoint 8 vs 16

This follow-up reuses the pinned model code and exact cached title embeddings from `specter2_preference`; it stores all new predictions and models in this folder. It trains one adapted run to epoch16 per partition and compares snapshots at8 and16. See PLAN.md for the locked controls and stop/go rule.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/specter2_preference_longer/run_longer.py all
```

The original run and incumbent submission remain untouched. This is exploratory nested evaluation on previously inspected folds; the script does not upload to Kaggle.

# Combined ordinal fixes

See PLAN.md for the configuration set locked before scoring. This experiment preserves the existing incumbent and never uploads to Kaggle.

Read [DECISION.md](DECISION.md) for the interpretation and [RESULTS.md](RESULTS.md) for all controls and diagnostics. Verification is recorded in `outputs/verification.json`.

From the assignment directory, with the existing environment:

```bash
/tmp/assignment-specter2-venv/bin/python -m unittest discover -s experiments/ordinal_combined_fixes -p 'test_*.py'
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_combined_fixes/run_combined.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_combined_fixes/verify_experiment.py
```

Requires the parent augmentation, ordinal and diagnosis caches. `outputs/protocol.json` binds source, plan, parent hashes and feature names; saved predictions are reused only under that fixed protocol. `final_recipe.joblib` records the parent feature recipe and transformation source hash; prediction requires the matching `transform` function and the corresponding saved CatBoost models. `submission_candidate.csv` follows the predeclared inner-selection policy, not retrospective outer-score selection. All outer data have been explored previously.

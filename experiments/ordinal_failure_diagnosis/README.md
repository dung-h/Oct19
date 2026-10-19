# Ordinal failure diagnosis

`PLAN.md`: gate/ranking accounting, PAV, preprocessing audit, fixed regularization
and neighbor-group interventions. `DEFAULTS_PLAN.md`: additional intervention after
discovering CatBoost loss-dependent defaults in the saved model parameters.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_failure_diagnosis/run_controls.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_failure_diagnosis/defaults_control.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_failure_diagnosis/analyze_mechanisms.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_failure_diagnosis/verify_diagnosis.py
/tmp/assignment-specter2-venv/bin/python experiments/ordinal_failure_diagnosis/write_report.py
```

Reuses dependencies, data and immutable feature/split caches of `ordinal_preference`.
Each new control has fixed hyperparameters, inner cutoff selection and separate
outputs. No outer label tuning, external messaging or Kaggle upload. Oracle
cutoffs are diagnostic only. `RESULTS.md` distinguishes observed effects from
unproven mechanisms, and reports whole selection policies separately.

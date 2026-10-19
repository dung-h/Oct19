# Diagnose the plateau

Read [PLAN.md](PLAN.md) and the completed [RESULTS.md](RESULTS.md). This experiment diagnoses sample efficiency and information extraction, without new hyperparameter search or submission.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/plateau_diagnosis/run_plan.py
```

Dependencies inherit `requirements.txt`; source/data/plan hashes are recorded before evaluation. The runner reuses cached predictions and rebuilds all target-derived features in each selected subset. Full-size feature reuse is checked against the previous verified cache. Repeated subset seeds are averaged inside outer folds; they are not independent evaluation sets.

`reviews.json` contains content-only observations from reading all20 deliberately difficult training pairs. These are not preference labels, label corrections or an estimate of population label noise. Every evidence quote is matched to the existing usable abstract with exact offsets. `materialize_reviews.py` makes CSV/Markdown review artifacts; `verify_diagnosis.py` checks isolation, metrics and evidence; `write_summary.py` creates plots and the Vietnamese report. Candidate discourse spans are diagnostic only and do not alter the model being measured.

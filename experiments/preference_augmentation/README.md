# Preference augmentation

Read [PLAN.md](PLAN.md) for the frozen experiment and [RESULTS.md](RESULTS.md) after completion. Original labels and previous submissions remain intact.

Install `requirements.txt` in the existing experiment environment, then:

```bash
/tmp/assignment-specter2-venv/bin/python experiments/preference_augmentation/run_plan.py
```

The script runs six leakage/evidence/metric tests, creates evidence-backed lexical content features, computes nested cross-fitted history features, evaluates four matched CatBoost arms and a Ridge control, selects/refits on all training data, verifies scores/inference, and writes the report. It reuses local cleaned abstracts and pinned embedding caches from previous experiments; no new source collection or private API key is required.

`outputs/protocol.json` binds the plan, data, split, model parameters and feature code hashes. Do not silently modify feature rules or hyperparameters and reuse the cache; start a new experiment directory. CatBoost is CPU/deterministic seed, no outer early stopping. Exact evidence is in `outputs/content_evidence.csv`, alongside coverage and per-partition crossfit lineage. Failed research is reported, not hidden by replacing the baseline. The candidate CSV is not uploaded to Kaggle.

Upstream documentation: [CatBoost text/data feature support](https://catboost.ai/docs/en/features/text-features). This experiment explicitly vectorizes title text; it does not rely on CatBoost's target-derived native text calcers.

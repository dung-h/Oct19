# Stability and verbatim candidate claims

Read [PLAN.md](PLAN.md) for the fixed design and [RESULTS.md](RESULTS.md) after completion.

```bash
/tmp/assignment-specter2-venv/bin/python experiments/claim_stability/run_plan.py
```

Dependencies inherit `requirements.txt`. Frozen MiniLM weights, cleaned abstracts, verified original features and splits are reused from previous local experiments; no API key or new collection is required. Protocol hashes bind the plan, source, data and encoder. Each field stores exact sentence offsets and unknown masks; extracted text is not generated content or a new preference annotation.

Five matched arms isolate a regularization package, two conservative flag repairs, generic abstract context, and candidate problem/method/result sentences. Primary inner CV chooses arm/cutoff; two extra head seeds refit outer training partitions only with those choices held fixed. The final all-train candidate uses the primary seed. Do not silently change extraction rules or hyperparameters after seeing outer results and reuse cached outputs; create a new experiment directory. No previous model/submission is overwritten and no Kaggle upload occurs.

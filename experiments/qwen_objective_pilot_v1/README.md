# Qwen objective screening

Current terminal scope: fit-only CPU screening of CE, CE+ordinal RPS, and
lightly weighted CE. The screening uses the frozen hidden cache from the
completed Qwen ranking pilot. GPU quality fitting is conditional on both
partition-local selectors finding an eligible challenger.

Run with the repository's `.venv-qwen` environment:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./.venv-qwen/bin/python experiments/qwen_objective_pilot_v1/screen.py
./.venv-qwen/bin/python experiments/qwen_objective_pilot_v1/verify.py
```

The scorer reads only each partition's committed fit-label cache. It never
loads prediction labels. The default output namespace is already occupied by
this run. For a source change, set `QWEN_OBJECTIVE_OUTPUT` to a fresh path; do
not overwrite or resume an existing screen namespace.

Protocol and locked thresholds are in `../qwen_head_v1/OBJECTIVE_AUGMENTATION_PLAN.md`.

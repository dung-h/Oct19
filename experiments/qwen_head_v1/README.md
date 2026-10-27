# Qwen five-class learned head diagnostic

Follows [PLAN.md](PLAN.md). This diagnostic uses a previously seen, group-
disjoint 1,196/298 split from `qwen_learning_check_v1`; it is exploratory.
It does not fit on test documents or promote a model.

The CPU stage selects one L2 penalty using three-fold publication-group CV on
the 1,196 training documents. Arm B fits a regularized affine five-class head
on frozen Qwen hidden states. Arm C initializes that same head and learns it
jointly with fresh q/v LoRA for three epochs. Arm A reuses the verified epoch-3
digit-head prediction cache from learning check v1 when its source hashes match.

The previous feature extraction, memberships, token audits and predictions
must already exist at `experiments/qwen_learning_check_v1/outputs/check_v1/`.
The preparation stage checks hashes and copies only the small artifacts it
needs. It does not export data, scrape, or re-extract features.

Run CPU contracts, then the pipeline from repository root:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-qwen/bin/python experiments/qwen_head_v1/verify.py
export HF_HOME=/mnt/windows/Datamining/.cache-qwen-hf
export QWEN_PYTHON="$PWD/.venv-qwen/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export QWEN_HEAD_OUTPUT=experiments/qwen_head_v1/outputs/head_v1_torch212_cu130_r3
tmux new-session -d -s qwenhead "bash experiments/qwen_head_v1/run_head.sh > $QWEN_HEAD_OUTPUT.log 2>&1"
```

The pipeline stages are `prepare`, `cpu-head`, `smoke`, `train`, `evaluate`.
For a partial technical recovery, inspect the failure record and preserve the
old namespace; source/runtime changes need a new output root and a new smoke.
The training process does not resume optimizer state.

Quality evaluation reads validation labels only after all three epochs finish.
The report compares A/B/C on the same 298 documents and includes the previous
prior and TF-IDF controls. It reports H/Q/R, class metrics, confusion,
selection/ranking diagnostics, group bootstrap intervals and a fixed gate.
Its H values must not be mixed with the incumbent pilot folds or public score.

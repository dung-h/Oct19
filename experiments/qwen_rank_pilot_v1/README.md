# Qwen ranking pilot

This package implements the locked ranking pilot in `PROTOCOL.md`, which is
branch A of `experiments/qwen_head_v1/MASTER_PLAN.md`. It reuses the generic
regularized-head, smoke, and three-epoch trainer from `qwen_head_v1/run.py`
without modifying that previously locked source. It uses the committed
`qwen_remote_v1/inputs` data and does not export or scrape inputs.

Run CPU contracts first, then run the two fits sequentially inside tmux:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-qwen/bin/python experiments/qwen_rank_pilot_v1/verify.py
tmux new-session -d -s qwenrank 'bash experiments/qwen_rank_pilot_v1/run_pilot.sh'
```

By default outputs go to the gitignored
`experiments/qwen_rank_pilot_v1/outputs/ranking_pilot_v1_r2_torch212_cu130`.
Set `QWEN_RANK_OUTPUT` before starting to choose a fresh namespace. Never
reuse a namespace after a failed or interrupted job; optimizer resume is not
implemented. Preserve all failed artifacts. Check `run.log`, per-step
`training.jsonl`, and `stage_status.json` while the job runs.

The pipeline verifies and reuses only compatible frozen features, extracts any
missing label-free features, fits each partition's frozen head on fit labels,
smokes, and trains each adapter sequentially. Quality evaluation waits for
both partitions to finish training and scoring. The report is exploratory and
does not authorize final/test inference, submission, or incumbent replacement.

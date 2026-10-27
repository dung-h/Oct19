# CPU author×topic preference pilot

Run B0 while the separate Qwen GPU pilot runs:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-qwen/bin/python experiments/author_topic_v1/run.py feasibility --output experiments/author_topic_v1/outputs/author_topic_v1_b0_r3
```

B0 never reads rating labels. Its report decides whether B1 may run under the
locked protocol. Keep outputs in this gitignored directory and preserve them
if a technical or feasibility gate fails.

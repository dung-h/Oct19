# Qwen learning check v1

This fit-internal diagnostic tests training length, digit-prior correction,
and frozen representation quality. Read PROTOCOL.md and config.json before
running. The original pilot and its held-out labels are not used to fit or
select these arms. Existing pilot adapters cannot be reused because they
have seen this experiment's internal validation labels.

The locked split uses only outer1_inner0 fit: 1,196 inner-train documents and
298 group-disjoint validation documents. A fixed 128 inner-train probe set
provides comparable fitting metrics for every arm.

| Arm | Purpose |
|---|---|
| Fit prior | No-content reference |
| Frozen digit logits | Original pretrained output geometry |
| Bias-only frozen digit logits | Global prior correction control |
| Frozen hidden state +5-class linear head | Test representation without fixed digit projection |
| Same head, shuffled train labels | Association/leakage sanity control |
| TF-IDF +5-class linear head | Cheap content-only model reference |
| Fresh q/v LoRA epochs1/2/3 | Fixed learning curve and additional fitting |

Use the existing .venv-qwen and pinned checkpoint cache. Run from repo root:

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-qwen/bin/python experiments/qwen_learning_check_v1/verify.py
    export HF_HOME=/mnt/windows/Datamining/.cache-qwen-hf
    export QWEN_PYTHON="$PWD/.venv-qwen/bin/python"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export QWEN_OUTPUT=experiments/qwen_learning_check_v1/outputs/check_replay
    bash experiments/qwen_learning_check_v1/run_check.sh

Use tmux for long sessions. The pipeline is prepare → new technical smoke →
one frozen feature extraction → CPU controls → fixed3-epoch LoRA → evaluation.
Source/config/stack/allocator hashes are checked at every stage. Only the final
stage joins validation labels to predictions. No early stopping or threshold
search is implemented. The controls are trained directly on frozen Qwen
features, not on in-sample predictions from a rating-trained adapter.

The three-epoch linear schedule is locked as a single trajectory. Its first
epoch is not an exact replay of the original one-epoch schedule. Compare
epochs within this run; do not mix these metrics with old pilot folds.

Keep failed artifacts. There is no mid-training optimizer resume and existing
training/evaluation paths are not overwritten. A source/runtime change requires
a new output root and another smoke. Checkpoints and fitted CPU models remain
under ignored outputs.

Artifacts: membership.json, runtime_lock.json, features.npz, capped feature
texts and token hashes, fitted CPU heads, per-epoch adapter hashes/probabilities,
training.jsonl, train/val metrics.csv, paired group bootstrap intervals, and
decision.json. The bootstrap is descriptive exploratory uncertainty, not new
independent validation or a competition promotion gate.

CPU tests verify validation-label mutation cannot change any fitted controls,
all epochs are required before evaluation, prediction/target order is respected,
and a completed evaluation cannot be silently overwritten. Synthetic fixtures
are never model-quality evidence.

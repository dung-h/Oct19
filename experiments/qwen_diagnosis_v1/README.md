# Reproduce the Qwen pilot diagnosis

Read DIAGNOSIS.md for findings, PROTOCOL.md for the fit-probe contract, and
REFERENCES.md for the explicitly post-hoc missingness reference.

Use the same .venv-qwen, pinned checkpoint cache and completed source pilot.
The original qwen_remote_v1 code/config/outputs must remain unchanged. All
new artifacts go under this directory's gitignored outputs. CPU programs
and the GPU probe refuse completed output paths; choose a fresh root.

Run from repository root:

    qwen_source=experiments/qwen_remote_v1/outputs/pilot_torch212_cu130
    qwen_diagnosis=experiments/qwen_diagnosis_v1/outputs/diagnosis_replay
    .venv-qwen/bin/python experiments/qwen_diagnosis_v1/analyze.py --source-output "$qwen_source" --output "$qwen_diagnosis"
    .venv-qwen/bin/python experiments/qwen_diagnosis_v1/references.py --source-output "$qwen_source" --output "$qwen_diagnosis/references_v2"

Set the cache and runtime before GPU work; use tmux for a remote session:

    export HF_HOME=/mnt/windows/Datamining/.cache-qwen-hf
    export QWEN_PYTHON="$PWD/.venv-qwen/bin/python"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    bash experiments/qwen_diagnosis_v1/run_probe.sh --smoke "$qwen_diagnosis/runtime_expandable"
    bash experiments/qwen_diagnosis_v1/run_probe.sh --source-output "$qwen_source" --runtime-smoke-output "$qwen_diagnosis/runtime_expandable" --output "$qwen_diagnosis/fit_probe_v2_expandable"
    .venv-qwen/bin/python experiments/qwen_diagnosis_v1/summarize.py --diagnosis-output "$qwen_diagnosis"

The technical smoke creates its own tiny adapter using fit-only examples.
The following probe loads each completed pilot adapter, checks saved-fit
parity, performs inference and one backward without an optimizer step, and
does not save modified adapter weights. Its fit samples never enter that
adapter's held-out set. Permutations and ablations are diagnostics, not clean
OOF estimates and not new submissions.

Verified on RTX5070Ti, Python3.11.16, torch2.12.0+cu130, bnb0.48.0 with the
remaining dependencies recorded in the source runtime locks. The first
default-allocator probe failure is retained under outputs/diagnosis_v1.

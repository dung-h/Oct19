#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
qwen_python="${QWEN_PYTHON:-python}"
qwen_output="${QWEN_HEAD_OUTPUT:-experiments/qwen_head_v1/outputs/head_v1_torch212_cu130}"
qwen_lib_root="$("$qwen_python" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia")')"
qwen_library_path="${LD_LIBRARY_PATH:-}"
for directory in "$qwen_lib_root"/*/lib; do
  if [[ -d "$directory" ]]; then
    qwen_library_path="${directory}${qwen_library_path:+:${qwen_library_path}}"
  fi
done
export LD_LIBRARY_PATH="$qwen_library_path"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
for stage in prepare cpu-head smoke train evaluate; do
  "$qwen_python" experiments/qwen_head_v1/run.py "$stage" --output "$qwen_output"
done

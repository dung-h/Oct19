#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/mnt/windows/Datamining/.cache-qwen-hf}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
qwen_python="$PWD/.venv-qwen/bin/python"
if [[ ! -x "$qwen_python" ]]; then
  echo "Missing Qwen environment: $qwen_python" >&2
  exit 2
fi
qwen_lib_root="$("$qwen_python" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia")')"
for directory in "$qwen_lib_root"/*/lib; do
  if [[ -d "$directory" ]]; then
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
output="${QWEN_RANK_OUTPUT:-experiments/qwen_rank_pilot_v1/outputs/ranking_pilot_v1_r2_torch212_cu130}"
mkdir -p "$(dirname "$output")"
exec > >(tee -a "$output.log") 2>&1
echo "output=$output"
"$qwen_python" experiments/qwen_rank_pilot_v1/run.py prepare --output "$output"
for partition in outer1_inner0 outer1_inner1; do
  part_output="$output/$partition"
  "$qwen_python" experiments/qwen_rank_pilot_v1/run.py cpu-head --output "$output" --partition "$partition"
  "$qwen_python" experiments/qwen_rank_pilot_v1/run.py smoke --output "$output" --partition "$partition"
  "$qwen_python" experiments/qwen_rank_pilot_v1/run.py train --output "$output" --partition "$partition"
done
"$qwen_python" experiments/qwen_rank_pilot_v1/run.py evaluate --output "$output"

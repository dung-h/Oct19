#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
qwen_python="${QWEN_PYTHON:-python}"
qwen_output="${QWEN_OUTPUT:-experiments/qwen_remote_v1/outputs/pilot_v1}"

# Include CUDA shared libraries shipped with the selected Python's torch wheels.
qwen_lib_root="$("${qwen_python}" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia")')"
qwen_library_path="${LD_LIBRARY_PATH:-}"
for directory in "${qwen_lib_root}"/*/lib; do
  if [[ -d "${directory}" ]]; then qwen_library_path="${directory}${qwen_library_path:+:${qwen_library_path}}"; fi
done
export LD_LIBRARY_PATH="${qwen_library_path}"

"${qwen_python}" experiments/qwen_remote_v1/verify.py --tokenizer
"${qwen_python}" experiments/qwen_remote_v1/run.py smoke --output "${qwen_output}"
"${qwen_python}" experiments/qwen_remote_v1/run.py train --partition outer1_inner0 --output "${qwen_output}"
"${qwen_python}" experiments/qwen_remote_v1/run.py train --partition outer1_inner1 --output "${qwen_output}"
"${qwen_python}" experiments/qwen_remote_v1/evaluate.py --output "${qwen_output}"

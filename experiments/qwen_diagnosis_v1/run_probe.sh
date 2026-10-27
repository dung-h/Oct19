#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
qwen_python="${QWEN_PYTHON:-python}"
qwen_lib_root="$("$qwen_python" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia")')"
qwen_library_path="${LD_LIBRARY_PATH:-}"
for directory in "$qwen_lib_root"/*/lib; do
  if [[ -d "$directory" ]]; then qwen_library_path="${directory}${qwen_library_path:+:${qwen_library_path}}"; fi
done
export LD_LIBRARY_PATH="$qwen_library_path"
if [[ "${1:-}" == "--smoke" ]]; then
  exec "$qwen_python" experiments/qwen_remote_v1/run.py smoke --output "$2"
fi
exec "$qwen_python" experiments/qwen_diagnosis_v1/probe_fit.py "$@"

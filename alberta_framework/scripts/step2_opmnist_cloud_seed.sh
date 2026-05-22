#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed="${SEED:?set SEED to the split seed index, for example SEED=1}"
python_bin="${PYTHON:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-outputs/step2_opmnist_solution_full}"
split_dir="$output_root/seed_splits"
prefix="step2_opmnist_solution_800task_3seed_seed${seed}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  echo "Create the project venv first, for example with: uv venv --python 3.13 .venv" >&2
  exit 2
fi

mkdir -p "$split_dir"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"

command=(
  "$python_bin"
  "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py"
  --mnist-published-scale
  --allow-openml-download
  --n-seeds 1
  --seed "$seed"
  --final-window 5000
  --chunk-size 60000
  --include-sharpened-mlp
  --include-adaptive-primary-sharpened
  --evaluate-all-permutation-views
  --max-test-permutation-views 800
  --only-methods "step2_hybrid_memory_trace,step2_hybrid_memory_trace_adaptive_sharp,mlp_h64,mlp_h128,mlp_h64_sharp,mlp_h128_sharp"
  --output-dir "$split_dir"
  --result-prefix "$prefix"
  --note-path "$split_dir/${prefix}.md"
  --status-path "$split_dir/${prefix}_status.json"
)

if [[ -n "${STOP_AFTER_CHUNKS:-}" ]]; then
  command+=(--stop-after-chunks "$STOP_AFTER_CHUNKS")
fi

printf 'Running split seed %s:\n' "$seed"
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}"

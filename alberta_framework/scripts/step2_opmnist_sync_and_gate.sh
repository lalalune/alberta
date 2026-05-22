#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-outputs/step2_opmnist_solution_full}"
split_dir="$output_root/seed_splits"
seed1_host="${SEED1_HOST:-ubuntu@10.0.0.126}"
seed1_jump="${SEED1_JUMP:-ubuntu@89.169.115.239}"
seed2_host="${SEED2_HOST:-ubuntu@89.169.115.239}"
remote_root="${REMOTE_ROOT:-~/alberta-step2-opmnist}"
loop_seconds="${LOOP_SECONDS:-}"

sync_once() {
  mkdir -p "$split_dir"
  rsync -az -e "ssh -J $seed1_jump" \
    "$seed1_host:$remote_root/outputs/step2_opmnist_solution_full/seed_splits/*seed1*" \
    "$split_dir/"
  rsync -az \
    "$seed2_host:$remote_root/outputs/step2_opmnist_solution_full/seed_splits/*seed2*" \
    "$split_dir/"

  "$python_bin" benchmarks/step2_opmnist_solution_pipeline.py \
    --write-plan "$output_root/plan.json" \
    --write-status "$output_root/pipeline_status.json" \
    --no-dry-run >/tmp/step2_opmnist_pipeline_status_refresh.log

  "$python_bin" - <<'PY'
import json
from pathlib import Path

status = json.loads(
    Path("outputs/step2_opmnist_solution_full/pipeline_status.json").read_text()
).get("status")
for seed in status["seeds"]:
    print(
        "seed",
        seed["seed"],
        "result",
        seed["result_exists"],
        "steps",
        seed["completed_steps"],
        seed["completed_steps_source"],
    )
print("ready_to_merge", status["ready_to_merge"])
print("ready_to_audit", status["ready_to_audit"])
PY

  if "$python_bin" - <<'PY'
import json
from pathlib import Path

status = json.loads(
    Path("outputs/step2_opmnist_solution_full/pipeline_status.json").read_text()
).get("status")
raise SystemExit(0 if status["ready_to_merge"] else 1)
PY
  then
    "$python_bin" benchmarks/step2_opmnist_solution_pipeline.py \
      --write-plan "$output_root/plan.json" \
      --write-status "$output_root/pipeline_status.json" \
      --merge-ready \
      --audit \
      --no-dry-run
  fi
}

if [[ -n "$loop_seconds" ]]; then
  while true; do
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    if ! sync_once; then
      echo "sync_once failed; will retry after ${loop_seconds}s" >&2
    fi
    sleep "$loop_seconds"
  done
else
  sync_once
fi

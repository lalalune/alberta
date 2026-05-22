#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-outputs/step2_opmnist_solution_full}"
seed2_result="$output_root/seed_splits/step2_opmnist_solution_800task_3seed_seed2_results.json"
poll_seconds="${POLL_SECONDS:-600}"

while [[ ! -f "$seed2_result" ]]; do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  "$python_bin" benchmarks/step2_opmnist_solution_pipeline.py \
    --write-plan "$output_root/plan.json" \
    --write-status "$output_root/pipeline_status.json" \
    --no-dry-run >/tmp/step2_opmnist_remote_pipeline_status.log || true
  "$python_bin" - <<'PY' || true
import json
from pathlib import Path

status_path = Path("outputs/step2_opmnist_solution_full/pipeline_status.json")
if status_path.exists():
    status = json.loads(status_path.read_text()).get("status")
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
PY
  sleep "$poll_seconds"
done

"$python_bin" benchmarks/step2_opmnist_solution_pipeline.py \
  --write-plan "$output_root/plan.json" \
  --write-status "$output_root/pipeline_status.json" \
  --merge-ready \
  --audit \
  --no-dry-run

#!/usr/bin/env python3
"""Coordinate the canonical split-seed Step 2 OPMNIST solution run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_PATH = REPO_ROOT / "benchmarks" / "step2_opmnist_full_run_plan.py"


def load_planner_module() -> Any:
    """Load the full-run planner module."""
    spec = importlib.util.spec_from_file_location(
        "step2_opmnist_full_run_plan_for_pipeline",
        PLANNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OPMNIST planner at {PLANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if the file exists."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(dict[str, Any], payload)


def command_value(command: Sequence[str], flag: str) -> str | None:
    """Return the value following a CLI flag in a planned command."""
    parts = list(command)
    try:
        index = parts.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        raise ValueError(f"{flag} in planned command has no value")
    return parts[index + 1]


def seed_index_for_command(command: Sequence[str]) -> int:
    """Return the seed configured for one split command."""
    value = command_value(command, "--seed")
    if value is None:
        raise ValueError("split command is missing --seed")
    return int(value)


def result_path_for_command(command: Sequence[str]) -> Path:
    """Return the result path produced by one runner command."""
    output_dir = command_value(command, "--output-dir")
    result_prefix = command_value(command, "--result-prefix")
    if output_dir is None or result_prefix is None:
        raise ValueError("runner command is missing output path flags")
    return Path(output_dir) / f"{result_prefix}_results.json"


def status_path_for_command(command: Sequence[str]) -> Path | None:
    """Return the optional status path produced by one runner command."""
    value = command_value(command, "--status-path")
    return Path(value) if value is not None else None


def completed_steps_from_result(payload: dict[str, Any]) -> int | None:
    """Extract completed step count from a seed result payload."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None
    dataset = records[0].get("dataset") if isinstance(records[0], dict) else None
    if not isinstance(dataset, dict):
        return None
    steps = dataset.get("steps")
    return int(steps) if isinstance(steps, int) and not isinstance(steps, bool) else None


def completed_steps_from_status(payload: dict[str, Any]) -> int | None:
    """Extract completed step count from a runner status payload."""
    steps = payload.get("completed_steps")
    return int(steps) if isinstance(steps, int) and not isinstance(steps, bool) else None


def summarize_seed_command(command: Sequence[str]) -> dict[str, Any]:
    """Summarize one split seed's current artifact status."""
    result_path = result_path_for_command(command)
    status_path = status_path_for_command(command)
    result_payload = load_json_if_present(result_path)
    status_payload = load_json_if_present(status_path) if status_path is not None else None
    completed_steps = None
    source = "missing"
    if result_payload is not None:
        completed_steps = completed_steps_from_result(result_payload)
        source = "result"
    elif status_payload is not None:
        completed_steps = completed_steps_from_status(status_payload)
        source = "status"
    return {
        "seed": seed_index_for_command(command),
        "result_path": str(result_path),
        "result_exists": result_payload is not None,
        "status_path": str(status_path) if status_path is not None else None,
        "status_exists": status_payload is not None,
        "completed_steps": completed_steps,
        "completed_steps_source": source,
        "command": [str(part) for part in command],
    }


def build_pipeline_status(plan: dict[str, Any]) -> dict[str, Any]:
    """Return artifact readiness for the planned split-run pipeline."""
    split_commands = [
        [str(part) for part in command]
        for command in plan["split_seed_runner_commands"]
    ]
    seeds = [summarize_seed_command(command) for command in split_commands]
    expected_result = Path(str(plan["expected_result"]))
    expected_solution_status = Path(str(plan["expected_solution_status"]))
    ready_to_merge = bool(seeds and all(seed["result_exists"] for seed in seeds))
    merged_result_exists = expected_result.exists()
    ready_to_audit = merged_result_exists
    return {
        "schema": "alberta.step2.opmnist_solution_pipeline.status.v1",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "plan_schema": plan["schema"],
        "expected_result": str(expected_result),
        "expected_result_exists": merged_result_exists,
        "expected_solution_status": str(expected_solution_status),
        "expected_solution_status_exists": expected_solution_status.exists(),
        "ready_to_merge": ready_to_merge,
        "ready_to_audit": ready_to_audit,
        "all_seed_results_exist": ready_to_merge,
        "seeds": seeds,
    }


def first_incomplete_seed(status: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first seed without a completed result."""
    for seed in status["seeds"]:
        if not bool(seed["result_exists"]):
            return cast(dict[str, Any], seed)
    return None


def run_command(command: Sequence[str], *, dry_run: bool) -> dict[str, Any]:
    """Run or report one command."""
    command_list = [str(part) for part in command]
    if dry_run:
        return {"command": command_list, "returncode": None, "dry_run": True}
    completed = subprocess.run(command_list, check=False)
    return {
        "command": command_list,
        "returncode": int(completed.returncode),
        "dry_run": False,
    }


def command_with_stop_after_chunks(
    command: Sequence[str],
    *,
    stop_after_chunks: int | None,
) -> list[str]:
    """Return a runner command optionally bounded to a chunk count."""
    command_list = [str(part) for part in command]
    if stop_after_chunks is None:
        return command_list
    if stop_after_chunks <= 0:
        raise ValueError("--run-next-chunks must be positive")
    return [*command_list, "--stop-after-chunks", str(stop_after_chunks)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    planner = load_planner_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=planner.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=planner.DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=planner.DEFAULT_NOTE_PATH)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=5_000)
    parser.add_argument("--chunk-size", type=int, default=60_000)
    parser.add_argument(
        "--only-methods",
        nargs="+",
        default=list(planner.DEFAULT_METHODS),
    )
    parser.add_argument("--write-plan", type=Path, default=None)
    parser.add_argument("--write-status", type=Path, default=None)
    parser.add_argument(
        "--run-next",
        action="store_true",
        help="Run the first split seed whose result JSON is missing.",
    )
    parser.add_argument(
        "--run-next-chunks",
        type=int,
        default=None,
        help=(
            "When used with --run-next, checkpoint and return after this many "
            "runner chunks instead of running the whole seed."
        ),
    )
    parser.add_argument(
        "--merge-ready",
        action="store_true",
        help="Run the merge command when all expected split seed results exist.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the solution gate when the merged result exists.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print planned actions without running subprocesses.",
    )
    return parser.parse_args(argv)


def build_plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the canonical planner payload from pipeline CLI args."""
    planner = load_planner_module()
    planner.validate_args(args)
    return cast(dict[str, Any], planner.build_plan(args))


def main(argv: Sequence[str] | None = None) -> int:
    """Coordinate status, optional seed execution, merge, and audit."""
    args = parse_args(argv)
    plan = build_plan_from_args(args)
    rendered_plan = json.dumps(plan, indent=2, sort_keys=True)
    if args.write_plan is not None:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(rendered_plan + "\n", encoding="utf-8")
    status = build_pipeline_status(plan)
    actions: list[dict[str, Any]] = []

    if args.run_next:
        seed = first_incomplete_seed(status)
        if seed is None:
            actions.append({"action": "run_next", "skipped": "all_seed_results_exist"})
        else:
            command = command_with_stop_after_chunks(
                seed["command"],
                stop_after_chunks=args.run_next_chunks,
            )
            actions.append(
                {
                    "action": "run_next",
                    "seed": seed["seed"],
                    "bounded_chunks": args.run_next_chunks,
                    **run_command(command, dry_run=args.dry_run),
                }
            )

    if args.merge_ready:
        if not status["ready_to_merge"]:
            actions.append({"action": "merge", "skipped": "missing_seed_results"})
        else:
            actions.append(
                {
                    "action": "merge",
                    **run_command(plan["merge_command"], dry_run=args.dry_run),
                }
            )

    if args.audit:
        if not Path(str(plan["expected_result"])).exists():
            actions.append({"action": "audit", "skipped": "missing_merged_result"})
        else:
            actions.append(
                {
                    "action": "audit",
                    **run_command(plan["audit_command"], dry_run=args.dry_run),
                }
            )

    status = build_pipeline_status(plan)
    output = {"plan": plan, "status": status, "actions": actions}
    rendered_status = json.dumps(output, indent=2, sort_keys=True)
    print(rendered_status)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered_status + "\n", encoding="utf-8")
    for action in actions:
        returncode = action.get("returncode")
        if isinstance(returncode, int) and returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the canonical Step 2 full-scale OPMNIST run plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "examples"
    / "The Alberta Plan"
    / "Step2"
    / "step2_upgd_memory_opmnist.py"
)
DEFAULT_OUTPUT_DIR = Path("outputs/step2_opmnist_solution_full")
DEFAULT_RESULT_PREFIX = "step2_opmnist_solution_800task_3seed"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_opmnist_solution_800task_3seed.md"
)
DEFAULT_METHODS = (
    "step2_hybrid_memory_trace",
    "step2_hybrid_memory_trace_adaptive_sharp",
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_sharp",
    "mlp_h128_sharp",
)
SINGLE_UPGD_PREFIXES = (
    "upgd_structure_linear_",
    "upgd_structure_softmax_",
)
SHARPENED_MLP_METHODS = ("mlp_h64_sharp", "mlp_h128_sharp")
CENTROID_METHOD = "centroid_hysteretic64_center_c030"
ADAPTIVE_PRIMARY_METHOD = "step2_hybrid_memory_trace_adaptive_sharp"
PRIMARY_SHARP_METHOD = "step2_hybrid_memory_trace_sharp"
PRIMARY_RLS_CALIBRATED_METHOD = "step2_hybrid_memory_trace_rls_cal"
RLS_CALIBRATED_METHOD = "upgd_structure_softmax_h64_rls_cal"
PRIMARY_DREAM_METHOD = "step2_hybrid_memory_trace_dream_surprise"
PROTO_MEMORY_PREFIX = "proto_mem_"
BRIER_SINGLE_UPGD_PREFIX = "upgd_structure_brier_"
TEMPERATURE_SINGLE_UPGD_PREFIX = "upgd_structure_softmax_h256_temp"
DREAM_SINGLE_UPGD_PREFIX = "upgd_structure_softmax_h64_dream_"
DELIGHT_SUFFIX = "_delight_gate30"
HYBRID_DELIGHT_PREFIX = "step2_hybrid_memory_trace_delight_gate"


def load_runner_module() -> Any:
    """Load the OPMNIST runner module from its path with spaces."""
    spec = importlib.util.spec_from_file_location(
        "step2_upgd_memory_opmnist_for_full_run_plan",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load OPMNIST runner at {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shell_join(parts: Sequence[str | Path]) -> str:
    """Return a shell-safe command string."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def runner_command(args: argparse.Namespace) -> list[str | Path]:
    """Return the canonical runner command."""
    return runner_command_for_seed(args, seed=None)


def runner_command_for_seed(
    args: argparse.Namespace,
    *,
    seed: int | None,
) -> list[str | Path]:
    """Return a runner command for either all seeds or one split seed."""
    result_methods = ",".join(args.only_methods)
    method_flags = runner_method_flags(args.only_methods)
    if seed is None:
        result_prefix = args.result_prefix
        output_dir = args.output_dir
        note_path = args.note_path
        n_seeds = args.n_seeds
        seed_start = args.seed
    else:
        result_prefix = f"{args.result_prefix}_seed{seed}"
        output_dir = args.output_dir / "seed_splits"
        note_path = args.output_dir / "seed_splits" / f"{result_prefix}.md"
        n_seeds = 1
        seed_start = seed
    status_path = output_dir / f"{result_prefix}_status.json"
    return [
        sys.executable,
        "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py",
        "--mnist-published-scale",
        "--allow-openml-download",
        "--n-seeds",
        str(n_seeds),
        "--seed",
        str(seed_start),
        "--final-window",
        str(args.final_window),
        "--chunk-size",
        str(args.chunk_size),
        *method_flags,
        "--evaluate-all-permutation-views",
        "--max-test-permutation-views",
        "800",
        "--only-methods",
        result_methods,
        "--output-dir",
        output_dir,
        "--result-prefix",
        result_prefix,
        "--note-path",
        note_path,
        "--status-path",
        status_path,
    ]


def runner_method_flags(methods: Sequence[str]) -> list[str]:
    """Return runner include flags needed before applying --only-methods."""
    flags: list[str] = []
    if any(method == CENTROID_METHOD for method in methods):
        flags.append("--include-centroid-candidates")
    if any(method in SHARPENED_MLP_METHODS for method in methods):
        flags.append("--include-sharpened-mlp")
    if PRIMARY_SHARP_METHOD in methods:
        flags.append("--include-primary-sharpened")
    if ADAPTIVE_PRIMARY_METHOD in methods:
        flags.append("--include-adaptive-primary-sharpened")
    if any(method.startswith(PROTO_MEMORY_PREFIX) for method in methods):
        flags.append("--include-prototype-memory")
    if any(method.startswith(SINGLE_UPGD_PREFIXES) for method in methods):
        flags.append("--include-single-upgd")
    if any(
        "smooth" in method and method.startswith("upgd_structure_softmax_")
        for method in methods
    ):
        flags.append("--include-smoothed-single-upgd")
    if any(method.startswith(BRIER_SINGLE_UPGD_PREFIX) for method in methods):
        flags.append("--include-brier-single-upgd")
    if any(method.startswith(TEMPERATURE_SINGLE_UPGD_PREFIX) for method in methods):
        flags.append("--include-temperature-single-upgd")
    if PRIMARY_RLS_CALIBRATED_METHOD in methods or RLS_CALIBRATED_METHOD in methods:
        flags.append("--include-rls-calibrated")
    if PRIMARY_DREAM_METHOD in methods or any(
        method.startswith(DREAM_SINGLE_UPGD_PREFIX) for method in methods
    ):
        flags.append("--include-dreaming-candidates")
    if any(
        method.startswith(HYBRID_DELIGHT_PREFIX) or method.endswith(DELIGHT_SUFFIX)
        for method in methods
    ):
        flags.append("--include-delight-candidates")
    return flags


def method_include_kwargs(methods: Sequence[str]) -> dict[str, bool]:
    """Return make_methods kwargs implied by the planned method names."""
    flags = set(runner_method_flags(methods))
    return {
        "include_centroid_candidates": "--include-centroid-candidates" in flags,
        "include_sharpened_mlp": "--include-sharpened-mlp" in flags,
        "include_primary_sharpened": "--include-primary-sharpened" in flags,
        "include_adaptive_primary_sharpened": (
            "--include-adaptive-primary-sharpened" in flags
        ),
        "include_prototype_memory": "--include-prototype-memory" in flags,
        "include_single_upgd": "--include-single-upgd" in flags,
        "include_smoothed_single_upgd": "--include-smoothed-single-upgd" in flags,
        "include_brier_single_upgd": "--include-brier-single-upgd" in flags,
        "include_temperature_single_upgd": "--include-temperature-single-upgd" in flags,
        "include_rls_calibrated": "--include-rls-calibrated" in flags,
        "include_dreaming_candidates": "--include-dreaming-candidates" in flags,
        "include_delight_candidates": "--include-delight-candidates" in flags,
    }


def validate_method_availability(methods: Sequence[str]) -> None:
    """Validate the planned method list against the runner's actual method set."""
    runner = load_runner_module()
    available = runner.make_methods(784, **method_include_kwargs(methods))
    runner.filter_methods(available, ",".join(methods))


def audit_command(args: argparse.Namespace) -> list[str | Path]:
    """Return the solution-gate command for the planned result."""
    return [
        sys.executable,
        "benchmarks/step2_opmnist_solution_gate.py",
        args.output_dir / f"{args.result_prefix}_results.json",
        "--min-seeds",
        str(args.n_seeds),
        "--write-status",
        args.output_dir / f"{args.result_prefix}_solution_gate.json",
    ]


def merge_command(args: argparse.Namespace) -> list[str | Path]:
    """Return the command that merges one-result-per-seed split outputs."""
    split_paths = [
        args.output_dir
        / "seed_splits"
        / f"{args.result_prefix}_seed{seed}_results.json"
        for seed in range(args.seed, args.seed + args.n_seeds)
    ]
    return [
        sys.executable,
        "benchmarks/step2_opmnist_merge_seed_results.py",
        *split_paths,
        "--output",
        args.output_dir / f"{args.result_prefix}_results.json",
        "--write-summary",
        args.output_dir / f"{args.result_prefix}_SUMMARY.md",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-serializable full-run plan."""
    run = runner_command(args)
    split_runs = [
        runner_command_for_seed(args, seed=seed)
        for seed in range(args.seed, args.seed + args.n_seeds)
    ]
    merge = merge_command(args)
    audit = audit_command(args)
    run_json = [str(part) for part in run]
    split_runs_json = [[str(part) for part in command] for command in split_runs]
    merge_json = [str(part) for part in merge]
    audit_json = [str(part) for part in audit]
    return {
        "schema": "alberta.step2.opmnist_full_run_plan.v1",
        "claim": "Step 2 OPMNIST solution candidate run",
        "n_seeds": int(args.n_seeds),
        "seed_start": int(args.seed),
        "protocol": {
            "mnist_source": "openml",
            "mnist_split": "canonical_60000_10000",
            "n_permutations": 800,
            "task_block_size": 60_000,
            "updates_per_seed": 48_000_000,
            "test_views": "all_800_permutation_views",
            "task_id_provided_to_learner": False,
            "prediction_before_update": True,
        },
        "methods": list(args.only_methods),
        "runner_command": run_json,
        "runner_command_shell": shell_join(run),
        "split_seed_runner_commands": split_runs_json,
        "split_seed_runner_command_shells": [
            shell_join(command) for command in split_runs
        ],
        "merge_command": merge_json,
        "merge_command_shell": shell_join(merge),
        "audit_command": audit_json,
        "audit_command_shell": shell_join(audit),
        "expected_result": str(args.output_dir / f"{args.result_prefix}_results.json"),
        "expected_solution_status": str(
            args.output_dir / f"{args.result_prefix}_solution_gate.json"
        ),
        "promotion_rule": (
            "The audit command must exit 0 without --allow-unsolved and report "
            "status.solved_opmnist_step2=true."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=5_000)
    parser.add_argument("--chunk-size", type=int, default=60_000)
    parser.add_argument(
        "--only-methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Methods to include; must include at least one fair MLP comparator.",
    )
    parser.add_argument("--write-plan", type=Path, default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate planner arguments."""
    if args.n_seeds < 3:
        raise ValueError("--n-seeds must be at least 3 for the solution gate")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not any(str(method).startswith("mlp_") for method in args.only_methods):
        raise ValueError("--only-methods must include at least one mlp_* comparator")
    if not any(not str(method).startswith("mlp_") for method in args.only_methods):
        raise ValueError("--only-methods must include at least one candidate")
    validate_method_availability(args.only_methods)


def main(argv: Sequence[str] | None = None) -> int:
    """Print or write the canonical plan."""
    args = parse_args(argv)
    validate_args(args)
    plan = build_plan(args)
    rendered = json.dumps(plan, indent=2, sort_keys=True)
    print(rendered)
    if args.write_plan is not None:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

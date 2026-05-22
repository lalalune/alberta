"""Audit Step 10 STOMP evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_step10_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 10 STOMP audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    stomp_benchmark_path = project_root / "outputs/step10_stomp/results.json"
    autodiscovery_path = project_root / "outputs/step10_feature_autodiscovery/results.json"
    stomp_benchmark = _read_json(stomp_benchmark_path)
    autodiscovery = _read_json(autodiscovery_path)

    implementation_files = {
        "src/alberta_framework/core/options.py": (
            project_root / "src/alberta_framework/core/options.py"
        ).exists(),
        "src/alberta_framework/steps/step10.py": (
            project_root / "src/alberta_framework/steps/step10.py"
        ).exists(),
        "tests/test_step10_production.py": (
            project_root / "tests/test_step10_production.py"
        ).exists(),
        "benchmarks/step10_stomp_options.py": (
            project_root / "benchmarks/step10_stomp_options.py"
        ).exists(),
        "benchmarks/step10_feature_autodiscovery.py": (
            project_root / "benchmarks/step10_feature_autodiscovery.py"
        ).exists(),
    }
    options_source = (project_root / "src/alberta_framework/core/options.py").read_text(
        encoding="utf-8"
    )
    step10_source = (project_root / "src/alberta_framework/steps/step10.py").read_text(
        encoding="utf-8"
    )
    tests_source = (project_root / "tests/test_step10_production.py").read_text(
        encoding="utf-8"
    )

    stomp_surface = all(
        marker in options_source
        for marker in (
            "class SubtaskSpec",
            "class STOMPAgent",
            "class STOMPState",
            "class OptionModelsState",
            "_differential_semidp_q_update",
            "subtasks_from_feature_scores",
        )
    ) and all(
        marker in step10_source
        for marker in (
            "Step10STOMPConfig",
            "make_step10_stomp_agent",
            "run_step10_scan",
            "run_step10_smoke",
        )
    )
    off_policy_surface = all(
        marker in options_source
        for marker in (
            "option_target_epsilon",
            "option_importance_clip",
            "_clipped_epsilon_greedy_importance_ratio",
            "option_importance_ratio",
            "option_importance_ratios",
        )
    ) and all(
        marker in tests_source
        for marker in (
            "test_step10_off_policy_intra_option_importance_ratio_is_clipped",
            "test_config_invalid_option_target_epsilon_raises",
            "test_config_invalid_option_importance_clip_raises",
        )
    )
    tests_present = all(
        marker in tests_source
        for marker in (
            "test_run_step10_scan_output_shapes",
            "test_step10_state_stays_finite_over_many_steps",
            "test_subtasks_from_feature_scores_integrates_with_stomp",
            "test_step10_off_policy_intra_option_importance_ratio_is_clipped",
        )
    )

    stomp_benchmark_passed = bool(
        stomp_benchmark is not None
        and stomp_benchmark.get("schema") == "alberta.step10.stomp_benchmark.v1"
        and stomp_benchmark.get("passed") is True
        and stomp_benchmark.get("n_seeds", 0) >= 10
        and stomp_benchmark.get("mean_diff_stomp_minus_sarsa", 0.0) > 0.0
        and stomp_benchmark.get("convergence_speedup_steps", 0.0) >= 200.0
    )
    autodiscovery_passed = bool(
        autodiscovery is not None
        and autodiscovery.get("schema") == "alberta.step10.feature_autodiscovery.v1"
        and autodiscovery.get("passed") is True
        and autodiscovery.get("aggregate", {}).get("n_seeds", 0) >= 10
        and autodiscovery.get("aggregate", {}).get("discovery_hit_rate", 0.0) == 1.0
        and autodiscovery.get("aggregate", {}).get("discovered_wins", 0) >= 10
    )

    evidence = {
        "implementation_files": implementation_files,
        "stomp_mechanics_surface": stomp_surface,
        "off_policy_intra_option_surface": off_policy_surface,
        "tests_present": tests_present,
        "stomp_accelerates_control_benchmark": {
            "path": str(stomp_benchmark_path),
            "exists": stomp_benchmark is not None,
            "passed": stomp_benchmark_passed,
            "summary": {
                "n_seeds": None
                if stomp_benchmark is None
                else stomp_benchmark.get("n_seeds"),
                "mean_diff_stomp_minus_sarsa": None
                if stomp_benchmark is None
                else stomp_benchmark.get("mean_diff_stomp_minus_sarsa"),
                "convergence_speedup_steps": None
                if stomp_benchmark is None
                else stomp_benchmark.get("convergence_speedup_steps"),
            },
        },
        "feature_autodiscovery_benchmark": {
            "path": str(autodiscovery_path),
            "exists": autodiscovery is not None,
            "passed": autodiscovery_passed,
            "aggregate": None if autodiscovery is None else autodiscovery.get("aggregate"),
        },
    }
    accepted = bool(
        all(implementation_files.values())
        and stomp_surface
        and off_policy_surface
        and tests_present
        and stomp_benchmark_passed
        and autodiscovery_passed
    )
    return {
        "schema": "alberta.step10.solution_gate.v6",
        "accepted_step10_stomp_progression": accepted,
        "claim_scope": (
            "stomp_auto_discovery_semidp_planning_off_policy_completion"
            if accepted
            else "step10_stomp_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [] if accepted else [
            "Step 10 STOMP evidence incomplete",
        ],
    }


def main() -> None:
    """Print the Step 10 solution-gate report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path where the JSON report should be written.",
    )
    args = parser.parse_args()
    report = run_step10_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

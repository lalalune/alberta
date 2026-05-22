"""Audit Step 9 guarded-dreaming evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_step9_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 9 guarded-dreaming audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    benchmark_path = (
        project_root / "outputs/step9_dreaming/guarded_dreaming_results.json"
    )
    benchmark = _read_json(benchmark_path)

    implementation_files = {
        "src/alberta_framework/steps/step9.py": (
            project_root / "src/alberta_framework/steps/step9.py"
        ).exists(),
        "src/alberta_framework/core/dreaming.py": (
            project_root / "src/alberta_framework/core/dreaming.py"
        ).exists(),
        "src/alberta_framework/core/behavior_model.py": (
            project_root / "src/alberta_framework/core/behavior_model.py"
        ).exists(),
        "tests/test_step9_production.py": (
            project_root / "tests/test_step9_production.py"
        ).exists(),
        "tests/test_dreaming.py": (
            project_root / "tests/test_dreaming.py"
        ).exists(),
        "tests/test_behavior_model.py": (
            project_root / "tests/test_behavior_model.py"
        ).exists(),
        "benchmarks/step9_guarded_dreaming.py": (
            project_root / "benchmarks/step9_guarded_dreaming.py"
        ).exists(),
    }

    step9_source = (project_root / "src/alberta_framework/steps/step9.py").read_text(
        encoding="utf-8"
    )
    dreaming_source = (
        project_root / "src/alberta_framework/core/dreaming.py"
    ).read_text(encoding="utf-8")
    behavior_source = (
        project_root / "src/alberta_framework/core/behavior_model.py"
    ).read_text(encoding="utf-8")
    step9_tests = (project_root / "tests/test_step9_production.py").read_text(
        encoding="utf-8"
    )

    guarded_surface = all(
        marker in step9_source
        for marker in (
            "Step9DreamingConfig",
            "Step9DreamingState",
            "step9_update",
            "dreaming_max_model_error",
            "RecentObservationBuffer",
            "dream_gate",
        )
    )
    behavior_rollout_surface = all(
        marker in step9_source
        for marker in (
            "behavior_model_step_size",
            "behavior_model_state",
            "BehaviorModel(",
            "sample_action",
            "dream_rollout_horizon",
            "rollout_td_signal",
        )
    ) and all(
        marker in behavior_source
        for marker in (
            "class BehaviorModel",
            "class BehaviorModelState",
            "def sample_action",
        )
    )
    prioritized_selection_surface = all(
        marker in step9_source
        for marker in (
            "dream_candidate_count",
            "dream_surprise_weight",
            "dream_utility_weight",
            "score_dream_candidates",
            "selected_indices",
            "selected_accepted",
        )
    ) and all(
        marker in dreaming_source
        for marker in (
            "class DreamSelectionConfig",
            "class DreamSelectionResult",
            "def score_dream_candidates",
        )
    )
    tests_present = all(
        marker in step9_tests
        for marker in (
            "test_step9_config_roundtrip",
            "test_step9_single_update_increments_counters",
            "test_step9_multi_step_behavior_model_dreaming_path",
            "test_step9_prioritized_candidate_selection_path",
            "test_step9_state_stays_finite_over_many_steps",
        )
    )

    benchmark_passed = bool(
        benchmark is not None
        and benchmark.get("schema") == "alberta.step9.guarded_dreaming_benchmark.v1"
        and benchmark.get("passed") is True
        and benchmark.get("n_seeds", 0) >= 10
        and benchmark.get("guard_wins", 0) >= 6
        and benchmark.get("guard_vs_naive_mean_diff", 0.0) > 0.0
        and benchmark.get("guarded_dyna", {}).get("phase2_mean", 0.0)
        >= benchmark.get("naive_dyna", {}).get("phase2_mean", 1.0)
    )

    evidence = {
        "implementation_files": implementation_files,
        "guarded_dreaming_surface": guarded_surface,
        "learned_behavior_multi_step_rollout_surface": behavior_rollout_surface,
        "prioritized_dream_selection_surface": prioritized_selection_surface,
        "tests_present": tests_present,
        "seeded_guarded_dreaming_benchmark": {
            "path": str(benchmark_path),
            "exists": benchmark is not None,
            "passed": benchmark_passed,
            "summary": {
                "n_seeds": None if benchmark is None else benchmark.get("n_seeds"),
                "guard_wins": None if benchmark is None else benchmark.get("guard_wins"),
                "guard_vs_naive_mean_diff": (
                    None if benchmark is None else benchmark.get("guard_vs_naive_mean_diff")
                ),
                "guarded_phase2": (
                    None
                    if benchmark is None
                    else benchmark.get("guarded_dyna", {}).get("phase2_mean")
                ),
                "naive_phase2": (
                    None
                    if benchmark is None
                    else benchmark.get("naive_dyna", {}).get("phase2_mean")
                ),
            },
        },
    }
    accepted = bool(
        all(implementation_files.values())
        and guarded_surface
        and behavior_rollout_surface
        and prioritized_selection_surface
        and tests_present
        and benchmark_passed
    )
    return {
        "schema": "alberta.step9.solution_gate.v3",
        "accepted_step9_guarded_dreaming": accepted,
        "claim_scope": (
            "guarded_multi_step_behavior_model_dreaming_completion"
            if accepted
            else "step9_guarded_dreaming_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [] if accepted else [
            "Step 9 guarded-dreaming evidence incomplete",
        ],
    }


def main() -> None:
    """Print the Step 9 solution-gate report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path where the JSON report should be written.",
    )
    args = parser.parse_args()
    report = run_step9_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

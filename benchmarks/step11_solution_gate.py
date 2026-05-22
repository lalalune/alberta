"""Audit Step 11 OaK evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_step11_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 11 OaK audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    benchmark_path = project_root / "outputs/step11_oak/results.json"
    benchmark = _read_json(benchmark_path)

    implementation_files = {
        "src/alberta_framework/core/oak.py": (
            project_root / "src/alberta_framework/core/oak.py"
        ).exists(),
        "src/alberta_framework/steps/step11.py": (
            project_root / "src/alberta_framework/steps/step11.py"
        ).exists(),
        "tests/test_step11_production.py": (
            project_root / "tests/test_step11_production.py"
        ).exists(),
        "benchmarks/step11_oak_curation.py": (
            project_root / "benchmarks/step11_oak_curation.py"
        ).exists(),
    }
    oak_source = (project_root / "src/alberta_framework/core/oak.py").read_text(
        encoding="utf-8"
    )
    step11_source = (project_root / "src/alberta_framework/steps/step11.py").read_text(
        encoding="utf-8"
    )
    tests_source = (project_root / "tests/test_step11_production.py").read_text(
        encoding="utf-8"
    )

    oak_surface = all(
        marker in oak_source
        for marker in (
            "class OaKAgent",
            "class OaKState",
            "def curate",
            "def keyboard_q_values",
            "def keyboard_action",
        )
    ) and all(
        marker in step11_source
        for marker in (
            "Step11OaKConfig",
            "make_step11_oak_agent",
            "run_step11_scan",
            "run_step11_smoke",
        )
    )
    learned_feature_surface = all(
        marker in oak_source
        for marker in (
            "learned_feature_subtask_specs",
            "subtasks_from_feature_scores",
            "base_scores",
            "option_scores",
        )
    )
    keyboard_learning_surface = all(
        marker in oak_source
        for marker in (
            "KeyboardChordLearnerConfig",
            "KeyboardChordLearnerState",
            "init_keyboard_chord_learner",
            "update_keyboard_chord_learner",
        )
    )
    tests_present = all(
        marker in tests_source
        for marker in (
            "test_learned_feature_subtask_specs_ranks_weighted_features",
            "test_keyboard_chord_learner_positive_reward_moves_toward_chord",
            "test_keyboard_chord_learner_max_norm_bounds_vector",
            "test_step11_state_stays_finite_200_steps",
        )
    )

    benchmark_passed = bool(
        benchmark is not None
        and benchmark.get("schema") == "alberta.step11.oak_curation.v1"
        and benchmark.get("accepted_step11_oak_curation") is True
        and benchmark.get("evidence", {})
        .get("valid_oak_final_avg_reward", {})
        .get("mean_final_avg_reward", 0.0)
        >= 0.70
        and benchmark.get("evidence", {})
        .get("post_curation_recovery", {})
        .get("mean_final_avg_reward", 0.0)
        >= 0.70
    )

    evidence = {
        "implementation_files": implementation_files,
        "oak_curation_keyboard_surface": oak_surface,
        "learned_feature_construction_surface": learned_feature_surface,
        "keyboard_chord_learning_surface": keyboard_learning_surface,
        "tests_present": tests_present,
        "seeded_oak_curation_benchmark": {
            "path": str(benchmark_path),
            "exists": benchmark is not None,
            "passed": benchmark_passed,
            "summary": None if benchmark is None else benchmark.get("evidence"),
        },
    }
    accepted = bool(
        all(implementation_files.values())
        and oak_surface
        and learned_feature_surface
        and keyboard_learning_surface
        and tests_present
        and benchmark_passed
    )
    return {
        "schema": "alberta.step11.solution_gate.v1",
        "accepted_step11_oak_fc_stomp": accepted,
        "claim_scope": (
            "oak_curation_feature_construction_keyboard_learning_completion"
            if accepted
            else "step11_oak_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [] if accepted else [
            "Step 11 OaK evidence incomplete",
        ],
    }


def main() -> None:
    """Print the Step 11 solution-gate report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path where the JSON report should be written.",
    )
    args = parser.parse_args()
    report = run_step11_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

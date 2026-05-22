"""Audit Step 12 IA evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_step12_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 12 IA audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    benchmark_path = project_root / "outputs/step12_ia/results.json"
    benchmark = _read_json(benchmark_path)

    implementation_files = {
        "src/alberta_framework/core/intelligence_amplification.py": (
            project_root / "src/alberta_framework/core/intelligence_amplification.py"
        ).exists(),
        "src/alberta_framework/steps/step12.py": (
            project_root / "src/alberta_framework/steps/step12.py"
        ).exists(),
        "tests/test_step12_production.py": (
            project_root / "tests/test_step12_production.py"
        ).exists(),
        "benchmarks/step12_ia_augmentation.py": (
            project_root / "benchmarks/step12_ia_augmentation.py"
        ).exists(),
    }
    ia_source = (
        project_root / "src/alberta_framework/core/intelligence_amplification.py"
    ).read_text(encoding="utf-8")
    step12_source = (project_root / "src/alberta_framework/steps/step12.py").read_text(
        encoding="utf-8"
    )
    tests_source = (project_root / "tests/test_step12_production.py").read_text(
        encoding="utf-8"
    )

    ia_surface = all(
        marker in ia_source
        for marker in (
            "class ExoCerebellumAgent",
            "class ExoCortexAgent",
            "class IAAgent",
            "class IAUpdateResult",
            "augmented_obs",
            "recommendation",
        )
    ) and all(
        marker in step12_source
        for marker in (
            "Step12IAConfig",
            "make_step12_ia_agent",
            "run_step12_scan",
            "run_step12_smoke",
        )
    )
    communication_protocol_surface = all(
        marker in ia_source
        for marker in (
            "RecommendationProtocolConfig",
            "RecommendationProtocolState",
            "RecommendationProtocolResult",
            "init_recommendation_protocol_state",
            "update_recommendation_protocol",
            "accepted_count",
            "rejected_count",
            "effective_action",
        )
    )
    tests_present = all(
        marker in tests_source
        for marker in (
            "test_recommendation_protocol_accepts_matching_action",
            "test_recommendation_protocol_rejects_different_action",
            "test_run_step12_scan_output_shapes",
            "test_step12_state_stays_finite_200_steps",
        )
    )

    benchmark_passed = bool(
        benchmark is not None
        and benchmark.get("schema") == "alberta.step12.ia_augmentation.v1"
        and benchmark.get("accepted_step12_ia_augmentation") is True
        and benchmark.get("accepted_step12_cerebellum_beats_baseline") is True
        and benchmark.get("accepted_step12_cortex_accuracy") is True
    )

    evidence = {
        "implementation_files": implementation_files,
        "ia_surface": ia_surface,
        "communication_protocol_surface": communication_protocol_surface,
        "tests_present": tests_present,
        "seeded_ia_augmentation_benchmark": {
            "path": str(benchmark_path),
            "exists": benchmark is not None,
            "passed": benchmark_passed,
            "summary": None if benchmark is None else benchmark.get("evidence"),
        },
    }
    accepted = bool(
        all(implementation_files.values())
        and ia_surface
        and communication_protocol_surface
        and tests_present
        and benchmark_passed
    )
    return {
        "schema": "alberta.step12.solution_gate.v1",
        "accepted_step12_prototype_ia": accepted,
        "claim_scope": (
            "prototype_ia_recommendation_protocol_completion"
            if accepted
            else "step12_prototype_ia_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [] if accepted else [
            "Step 12 IA evidence incomplete",
        ],
    }


def main() -> None:
    """Print the Step 12 solution-gate report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path where the JSON report should be written.",
    )
    args = parser.parse_args()
    report = run_step12_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

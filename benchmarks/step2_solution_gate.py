"""Audit Step 2 feature-discovery evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def run_step2_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 2 feature-discovery audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    gate_path = project_root / "outputs/step2_solution_gate.json"
    gate = _read_json(gate_path)

    implementation_files = {
        "src/alberta_framework/core/optimizers.py": (
            project_root / "src/alberta_framework/core/optimizers.py"
        ).exists(),
        "tests/test_optimizers.py": (
            project_root / "tests/test_optimizers.py"
        ).exists(),
    }

    upgd_accepted = bool(gate and gate.get("accepted_step2_upgd_feature_finding"))
    lifecycle_accepted = bool(gate and gate.get("accepted_step2_feature_lifecycle_implementation"))
    external_accepted = bool(gate and gate.get("accepted_step2_external_benchmark"))
    full_scope_solved = bool(gate and gate.get("solved_step2_full_research_scope"))

    accepted = upgd_accepted and lifecycle_accepted and external_accepted

    return {
        "schema": "alberta.step2.solution_gate.v2",
        "accepted_step2_feature_finding": accepted,
        "solved_step2_full_research_scope": full_scope_solved,
        "claim_scope": (
            gate.get("claim_scope", "unknown") if gate else "no_gate_artifact"
        ),
        "evidence": {
            "implementation_files": implementation_files,
            "gate_artifact": {
                "path": str(gate_path),
                "exists": gate is not None,
                "accepted_upgd": upgd_accepted,
                "accepted_lifecycle": lifecycle_accepted,
                "accepted_external": external_accepted,
            },
            "out_of_class_results": {
                "path": str(
                    project_root / "outputs/step2_canonical/out_of_class_results.json"
                ),
                "exists": (
                    project_root / "outputs/step2_canonical/out_of_class_results.json"
                ).exists(),
            },
            "digits_results": {
                "path": str(
                    project_root
                    / "outputs/step2_canonical/universal_portfolio_strict_results.json"
                ),
                "exists": (
                    project_root
                    / "outputs/step2_canonical/universal_portfolio_strict_results.json"
                ).exists(),
            },
        },
        "remaining_research_boundaries": [
            "Full multi-seed OPMNIST benchmark (requires external bsuite infra)",
            "Lifecycle wins limited to polynomial-approximable streams"
            " (sinusoidal/compositional open)",
        ] if not full_scope_solved else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-status", type=Path, default=None)
    args = parser.parse_args()
    report = run_step2_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

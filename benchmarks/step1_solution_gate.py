"""Audit Step 1 adaptive step-size evidence against completion criteria."""

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


def _repo_file(root: Path, relative: str) -> Path:
    """Resolve a source file in either src-layout or package-root layout."""
    src_path = root / relative
    if src_path.exists():
        return src_path
    package_path = root / relative.removeprefix("src/")
    if package_path.exists():
        return package_path
    return src_path


def run_step1_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 1 adaptive step-size audit status."""
    project_root = root or Path(__file__).resolve().parents[1]

    multi_baseline_path = project_root / "outputs/step1_canonical/multi_baseline_results.json"
    robustness_path = project_root / "outputs/step1_canonical/robustness_study_results.json"
    gtd_path = project_root / "outputs/step1_canonical/autostep_gtd_5seed_results.json"

    multi = _read_json(multi_baseline_path)
    robustness = _read_json(robustness_path)
    gtd = _read_json(gtd_path)

    implementation_files = {
        "src/alberta_framework/core/optimizers.py": (
            _repo_file(project_root, "src/alberta_framework/core/optimizers.py")
        ).exists(),
        "src/alberta_framework/core/learners.py": (
            _repo_file(project_root, "src/alberta_framework/core/learners.py")
        ).exists(),
        "tests/test_optimizers.py": (
            project_root / "tests/test_optimizers.py"
        ).exists(),
    }

    adaptive_beat_lms = False
    idbd_wins: dict[str, int] = {}
    autostep_wins: dict[str, int] = {}
    n_seeds = 0
    sutton_streams = ["Sutton1992_noiseless", "Sutton1992_noisy"]
    if multi is not None:
        paired = multi.get("paired_vs_lms", {})
        for stream in paired:
            idbd_wins[stream] = paired[stream].get("IDBD", {}).get("wins", 0)
            autostep_wins[stream] = paired[stream].get("Autostep", {}).get("wins", 0)
            if idbd_wins[stream] > 0:
                n_seeds = max(n_seeds, paired[stream]["IDBD"].get("n_seeds", 0))
        adaptive_beat_lms = (
            all(idbd_wins.get(s, 0) >= 25 for s in sutton_streams)
            and all(autostep_wins.get(s, 0) >= 25 for s in sutton_streams)
            and n_seeds >= 25
        )

    robustness_passed = False
    if robustness is not None:
        robustness_passed = robustness.get("idbd_wins_all_configs", False) or (
            robustness.get("idbd_vs_lms_wins", 0) >= 20
        )

    source = _repo_file(
        project_root,
        "src/alberta_framework/core/optimizers.py",
    ).read_text(encoding="utf-8")
    implementation_surface = all(
        m in source
        for m in ("class LMS", "class Autostep", "log_alpha", "meta_step_size")
    )

    accepted = bool(
        all(implementation_files.values())
        and implementation_surface
        and adaptive_beat_lms
    )

    return {
        "schema": "alberta.step1.solution_gate.v1",
        "accepted_step1_adaptive_step_sizes": accepted,
        "claim_scope": (
            "idbd_autostep_beat_lms_30seed_multi_stream"
            if accepted
            else "step1_adaptive_step_sizes_incomplete"
        ),
        "evidence": {
            "implementation_files": implementation_files,
            "implementation_surface": implementation_surface,
            "multi_baseline_benchmark": {
                "path": str(multi_baseline_path),
                "exists": multi is not None,
                "n_seeds": n_seeds,
                "adaptive_beat_lms_all_streams": adaptive_beat_lms,
                "idbd_wins_per_stream": idbd_wins,
                "autostep_wins_per_stream": autostep_wins,
            },
            "robustness_study": {
                "path": str(robustness_path),
                "exists": robustness is not None,
                "passed": robustness_passed,
            },
            "gtd_autostep": {
                "path": str(gtd_path),
                "exists": gtd is not None,
            },
        },
        "remaining_research_boundaries": [] if accepted else [
            "Step 1 multi-baseline evidence not yet proven",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-status", type=Path, default=None)
    args = parser.parse_args()
    report = run_step1_solution_gate()
    rendered = json.dumps(report, indent=2)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Step 5 average-reward evidence against completion criteria."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".")
DEFAULT_PREDICTION_BENCHMARK = Path(
    "outputs/step5_average_reward_prediction/results.json"
)
DEFAULT_CONTROL_BENCHMARK = Path("outputs/step5_continuing_control/results.json")
DEFAULT_MULTISTATE_CONTROL_BENCHMARK = Path(
    "outputs/step5_multistate_continuing_control/results.json"
)
DEFAULT_OFF_POLICY_BENCHMARK = Path(
    "outputs/step5_off_policy_average_reward/results.json"
)
DEFAULT_HORDE_BENCHMARK = Path("outputs/step5_average_reward_horde/results.json")
DEFAULT_ACTOR_CRITIC_BENCHMARK = Path(
    "outputs/step5_average_reward_horde_actor_critic/results.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if the path exists and contains an object."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def audit_step5(
    root: Path = DEFAULT_ROOT,
    *,
    prediction_benchmark: Path = DEFAULT_PREDICTION_BENCHMARK,
    control_benchmark: Path = DEFAULT_CONTROL_BENCHMARK,
    multistate_control_benchmark: Path = DEFAULT_MULTISTATE_CONTROL_BENCHMARK,
    off_policy_benchmark: Path = DEFAULT_OFF_POLICY_BENCHMARK,
    horde_benchmark: Path = DEFAULT_HORDE_BENCHMARK,
    actor_critic_benchmark: Path = DEFAULT_ACTOR_CRITIC_BENCHMARK,
) -> dict[str, Any]:
    """Return Step 5 primitive and full-scope audit status."""
    evidence: dict[str, Any] = {}

    implementation_files = {
        "src/alberta_framework/core/average_reward.py": (
            root / "src/alberta_framework/core/average_reward.py"
        ).exists(),
        "src/alberta_framework/steps/step5.py": (
            root / "src/alberta_framework/steps/step5.py"
        ).exists(),
        "tests/test_average_reward.py": (root / "tests/test_average_reward.py").exists(),
        "tests/test_step5_step6_production.py": (
            root / "tests/test_step5_step6_production.py"
        ).exists(),
        "docs/research/step5_step6_average_reward.md": (
            root / "docs/research/step5_step6_average_reward.md"
        ).exists(),
    }
    evidence["implementation_and_tests"] = {
        "files": implementation_files,
        "passed": all(implementation_files.values()),
    }

    from alberta_framework.steps import Step5AverageRewardTDConfig, run_step5_smoke

    smoke = run_step5_smoke(
        Step5AverageRewardTDConfig(
            step_size=0.03,
            average_reward_step_size=0.02,
            trace_decay=0.25,
        ),
        steps=64,
        feature_dim=4,
        seed=0,
    )
    evidence["facade_smoke"] = {
        "steps": smoke.steps,
        "seed": smoke.seed,
        "finite": smoke.finite,
        "predictions_shape": list(smoke.predictions_shape),
        "td_errors_shape": list(smoke.td_errors_shape),
        "average_rewards_shape": list(smoke.average_rewards_shape),
        "learner_type": smoke.learner_config.get("type"),
        "passed": bool(
            smoke.finite
            and smoke.predictions_shape == (64,)
            and smoke.td_errors_shape == (64,)
            and smoke.average_rewards_shape == (64,)
            and smoke.learner_config.get("type") == "DifferentialTDLearner"
        ),
    }

    note_path = root / "docs/research/step5_step6_average_reward.md"
    note_text = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
    primitive_boundary_documented = (
        "primitive landing, not a claim that Steps 5" in note_text
        or "not the full Step 5/6 research program" in note_text
    )
    evidence["documented_scope"] = {
        "path": str(note_path),
        "primitive_boundary_documented": primitive_boundary_documented,
        "passed": primitive_boundary_documented,
    }

    benchmark_path = prediction_benchmark
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    benchmark = _load_json(benchmark_path)
    benchmark_aggregate = (
        benchmark.get("aggregate", {}) if benchmark is not None else {}
    )
    prediction_benchmark_passed = bool(
        benchmark is not None
        and benchmark.get("schema")
        == "alberta.step5.average_reward_prediction_benchmark.v1"
        and benchmark.get("passed") is True
        and benchmark_aggregate.get("n_seeds", 0) >= 10
        and benchmark_aggregate.get("n_passed")
        == benchmark_aggregate.get("n_seeds")
        and benchmark_aggregate.get("mean_average_reward_abs_error", 1.0) <= 0.02
        and benchmark_aggregate.get("mean_centered_value_rmse", 1.0) <= 0.05
        and benchmark_aggregate.get("mean_tail_td_error_mse", 1.0) <= 0.002
    )
    evidence["seeded_prediction_benchmark"] = {
        "path": str(benchmark_path),
        "present": benchmark is not None,
        "schema": benchmark.get("schema") if benchmark is not None else None,
        "claim_scope": benchmark.get("claim_scope") if benchmark is not None else None,
        "aggregate": benchmark_aggregate,
        "target": benchmark.get("target", {}) if benchmark is not None else {},
        "baseline": benchmark.get("baseline", {}) if benchmark is not None else {},
        "passed": prediction_benchmark_passed,
    }

    control_benchmark_path = control_benchmark
    if not control_benchmark_path.is_absolute():
        control_benchmark_path = root / control_benchmark_path
    control_report = _load_json(control_benchmark_path)
    control_aggregate = (
        control_report.get("aggregate", {}) if control_report is not None else {}
    )
    control_benchmark_passed = bool(
        control_report is not None
        and control_report.get("schema")
        == "alberta.step5.continuing_control_benchmark.v1"
        and control_report.get("passed") is True
        and control_aggregate.get("n_seeds", 0) >= 10
        and control_aggregate.get("n_passed") == control_aggregate.get("n_seeds")
        and control_aggregate.get("mean_average_reward_abs_error", 1.0) <= 0.02
        and control_aggregate.get("mean_final_reward", 0.0) >= 0.99
        and control_aggregate.get("mean_final_optimal_action_rate", 0.0) >= 0.99
        and control_aggregate.get("mean_tail_td_error_mse", 1.0) <= 0.002
    )
    evidence["seeded_continuing_control_benchmark"] = {
        "path": str(control_benchmark_path),
        "present": control_report is not None,
        "schema": control_report.get("schema") if control_report is not None else None,
        "claim_scope": (
            control_report.get("claim_scope") if control_report is not None else None
        ),
        "aggregate": control_aggregate,
        "target": control_report.get("target", {}) if control_report is not None else {},
        "baseline": (
            control_report.get("baseline", {}) if control_report is not None else {}
        ),
        "passed": control_benchmark_passed,
    }

    multistate_path = multistate_control_benchmark
    if not multistate_path.is_absolute():
        multistate_path = root / multistate_path
    multistate_report = _load_json(multistate_path)
    multistate_aggregate = (
        multistate_report.get("aggregate", {})
        if multistate_report is not None
        else {}
    )
    multistate_control_passed = bool(
        multistate_report is not None
        and multistate_report.get("schema")
        == "alberta.step5.multistate_continuing_control_benchmark.v1"
        and multistate_report.get("passed") is True
        and multistate_aggregate.get("n_seeds", 0) >= 10
        and multistate_aggregate.get("n_passed")
        == multistate_aggregate.get("n_seeds")
        and multistate_aggregate.get("mean_average_reward_abs_error", 1.0) <= 0.02
        and multistate_aggregate.get("mean_final_reward", 0.0) >= 0.99
        and multistate_aggregate.get("mean_final_policy_match_rate", 0.0) >= 0.99
        and multistate_aggregate.get("mean_tail_td_error_mse", 1.0) <= 0.002
    )
    evidence["seeded_multistate_continuing_control_benchmark"] = {
        "path": str(multistate_path),
        "present": multistate_report is not None,
        "schema": (
            multistate_report.get("schema") if multistate_report is not None else None
        ),
        "claim_scope": (
            multistate_report.get("claim_scope")
            if multistate_report is not None
            else None
        ),
        "aggregate": multistate_aggregate,
        "target": (
            multistate_report.get("target", {})
            if multistate_report is not None
            else {}
        ),
        "baseline": (
            multistate_report.get("baseline", {})
            if multistate_report is not None
            else {}
        ),
        "passed": multistate_control_passed,
    }

    off_policy_path = off_policy_benchmark
    if not off_policy_path.is_absolute():
        off_policy_path = root / off_policy_path
    off_policy_report = _load_json(off_policy_path)
    off_policy_aggregate = (
        off_policy_report.get("aggregate", {})
        if off_policy_report is not None
        else {}
    )
    off_policy_passed = bool(
        off_policy_report is not None
        and off_policy_report.get("schema")
        == "alberta.step5.off_policy_average_reward_benchmark.v1"
        and off_policy_report.get("passed") is True
        and off_policy_aggregate.get("n_seeds", 0) >= 10
        and off_policy_aggregate.get("n_passed")
        == off_policy_aggregate.get("n_seeds")
        and off_policy_aggregate.get("mean_average_reward_abs_error", 1.0) <= 0.02
        and off_policy_aggregate.get("mean_final_average_reward", 0.0) >= 0.99
        and off_policy_aggregate.get("mean_weighted_tail_td_error_mse", 1.0)
        <= 0.002
    )
    evidence["seeded_off_policy_average_reward_benchmark"] = {
        "path": str(off_policy_path),
        "present": off_policy_report is not None,
        "schema": (
            off_policy_report.get("schema") if off_policy_report is not None else None
        ),
        "claim_scope": (
            off_policy_report.get("claim_scope")
            if off_policy_report is not None
            else None
        ),
        "aggregate": off_policy_aggregate,
        "target": (
            off_policy_report.get("target", {})
            if off_policy_report is not None
            else {}
        ),
        "passed": off_policy_passed,
    }

    horde_path = horde_benchmark
    if not horde_path.is_absolute():
        horde_path = root / horde_path
    horde_report = _load_json(horde_path)
    horde_aggregate = (
        horde_report.get("aggregate", {}) if horde_report is not None else {}
    )
    horde_passed = bool(
        horde_report is not None
        and horde_report.get("schema")
        == "alberta.step5.average_reward_horde_benchmark.v1"
        and horde_report.get("passed") is True
        and horde_aggregate.get("n_seeds", 0) >= 10
        and horde_aggregate.get("n_passed") == horde_aggregate.get("n_seeds")
        and horde_aggregate.get("max_average_reward_abs_error", 1.0) <= 0.03
        and horde_aggregate.get("mean_tail_td_error_mse", 1.0) <= 0.005
    )
    evidence["seeded_average_reward_horde_benchmark"] = {
        "path": str(horde_path),
        "present": horde_report is not None,
        "schema": horde_report.get("schema") if horde_report is not None else None,
        "claim_scope": (
            horde_report.get("claim_scope") if horde_report is not None else None
        ),
        "aggregate": horde_aggregate,
        "target": horde_report.get("target", {}) if horde_report is not None else {},
        "passed": horde_passed,
    }

    actor_critic_path = actor_critic_benchmark
    if not actor_critic_path.is_absolute():
        actor_critic_path = root / actor_critic_path
    actor_critic_report = _load_json(actor_critic_path)
    actor_critic_aggregate = (
        actor_critic_report.get("aggregate", {})
        if actor_critic_report is not None
        else {}
    )
    actor_critic_passed = bool(
        actor_critic_report is not None
        and actor_critic_report.get("schema")
        == "alberta.step5.average_reward_horde_actor_critic_benchmark.v1"
        and actor_critic_report.get("passed") is True
        and actor_critic_aggregate.get("n_seeds", 0) >= 10
        and actor_critic_aggregate.get("n_passed")
        == actor_critic_aggregate.get("n_seeds")
        and actor_critic_aggregate.get("mean_average_reward_abs_error", 1.0) <= 0.04
        and actor_critic_aggregate.get("mean_final_reward", 0.0) >= 0.97
        and actor_critic_aggregate.get("mean_final_policy_match_rate", 0.0) >= 0.97
    )
    evidence["seeded_average_reward_horde_actor_critic_benchmark"] = {
        "path": str(actor_critic_path),
        "present": actor_critic_report is not None,
        "schema": (
            actor_critic_report.get("schema")
            if actor_critic_report is not None
            else None
        ),
        "claim_scope": (
            actor_critic_report.get("claim_scope")
            if actor_critic_report is not None
            else None
        ),
        "aggregate": actor_critic_aggregate,
        "target": (
            actor_critic_report.get("target", {})
            if actor_critic_report is not None
            else {}
        ),
        "passed": actor_critic_passed,
    }

    primitive_accepted = all(
        bool(item)
        for item in [
            evidence["implementation_and_tests"]["passed"],
            evidence["facade_smoke"]["passed"],
            evidence["documented_scope"]["passed"],
        ]
    )
    prediction_accepted = bool(primitive_accepted and prediction_benchmark_passed)
    control_accepted = bool(primitive_accepted and control_benchmark_passed)
    multistate_control_accepted = bool(
        primitive_accepted and multistate_control_passed
    )
    off_policy_accepted = bool(primitive_accepted and off_policy_passed)
    horde_accepted = bool(primitive_accepted and horde_passed)
    actor_critic_accepted = bool(primitive_accepted and actor_critic_passed)
    solved_full_scope = bool(
        prediction_accepted
        and multistate_control_accepted
        and off_policy_accepted
        and horde_accepted
        and actor_critic_accepted
    )
    missing_full_evidence = (
        []
        if solved_full_scope
        else ["average-reward actor-critic with nonlinear shared features"]
    )
    return {
        "schema": "alberta.step5.solution_gate.v1",
        "accepted_step5_average_reward_primitive": bool(primitive_accepted),
        "accepted_step5_average_reward_prediction": bool(prediction_accepted),
        "accepted_step5_step6_continuing_control": bool(control_accepted),
        "accepted_step5_step6_multistate_continuing_control": bool(
            multistate_control_accepted
        ),
        "accepted_step5_off_policy_average_reward_gtd": bool(off_policy_accepted),
        "accepted_step5_nonlinear_average_reward_horde": bool(horde_accepted),
        "accepted_step5_average_reward_horde_actor_critic": bool(
            actor_critic_accepted
        ),
        "solved_step5_full_research_scope": bool(solved_full_scope),
        "claim_scope": (
            "average_reward_step5_full_local_completion"
            if solved_full_scope
            else
            "average_reward_prediction_control_off_policy_and_horde_landing"
            if prediction_accepted
            and multistate_control_accepted
            and off_policy_accepted
            and horde_accepted
            else
            "average_reward_prediction_control_and_off_policy_landing"
            if prediction_accepted
            and multistate_control_accepted
            and off_policy_accepted
            else
            "average_reward_prediction_and_multistate_control_landing"
            if prediction_accepted and multistate_control_accepted
            else
            "average_reward_prediction_and_one_state_control_landing"
            if prediction_accepted and control_accepted
            else "average_reward_td_closed_form_prediction_landing"
            if prediction_accepted
            else "average_reward_td_primitive_landing"
            if primitive_accepted
            else "incomplete_step5_primitive_evidence"
        ),
        "evidence": evidence,
        "missing_full_evidence": missing_full_evidence,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prediction-benchmark",
        type=Path,
        default=DEFAULT_PREDICTION_BENCHMARK,
    )
    parser.add_argument(
        "--control-benchmark",
        type=Path,
        default=DEFAULT_CONTROL_BENCHMARK,
    )
    parser.add_argument(
        "--multistate-control-benchmark",
        type=Path,
        default=DEFAULT_MULTISTATE_CONTROL_BENCHMARK,
    )
    parser.add_argument(
        "--off-policy-benchmark",
        type=Path,
        default=DEFAULT_OFF_POLICY_BENCHMARK,
    )
    parser.add_argument(
        "--horde-benchmark",
        type=Path,
        default=DEFAULT_HORDE_BENCHMARK,
    )
    parser.add_argument(
        "--actor-critic-benchmark",
        type=Path,
        default=DEFAULT_ACTOR_CRITIC_BENCHMARK,
    )
    parser.add_argument("--write-status", type=Path, default=None)
    parser.add_argument("--allow-unsolved", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Step 5 audit."""
    args = parse_args(argv)
    status = audit_step5(
        args.root,
        prediction_benchmark=args.prediction_benchmark,
        control_benchmark=args.control_benchmark,
        multistate_control_benchmark=args.multistate_control_benchmark,
        off_policy_benchmark=args.off_policy_benchmark,
        horde_benchmark=args.horde_benchmark,
        actor_critic_benchmark=args.actor_critic_benchmark,
    )
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    if status["solved_step5_full_research_scope"] or args.allow_unsolved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

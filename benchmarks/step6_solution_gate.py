#!/usr/bin/env python3
"""Audit Step 6 continuing-control evidence against completion criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object when the artifact exists."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def run_step6_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return the Step 6 continuing-control audit report."""
    project_root = root or Path(__file__).resolve().parents[1]
    deterministic_path = project_root / "outputs/step6_riverswim/results.json"
    stochastic_path = (
        project_root / "outputs/step6_riverswim/riverswim_stochastic_results.json"
    )
    multistate_path = (
        project_root / "outputs/step5_multistate_continuing_control/results.json"
    )
    nonlinear_actor_critic_path = (
        project_root
        / "outputs/step5_average_reward_horde_actor_critic/results.json"
    )
    security_gym_path = project_root / "outputs/step6_security_gym/results.json"
    deterministic = _read_json(deterministic_path)
    stochastic = _read_json(stochastic_path)
    multistate = _read_json(multistate_path)
    nonlinear_actor_critic = _read_json(nonlinear_actor_critic_path)
    security_gym = _read_json(security_gym_path)

    implementation_files = {
        "src/alberta_framework/core/average_reward.py": (
            project_root / "src/alberta_framework/core/average_reward.py"
        ).exists(),
        "src/alberta_framework/steps/step6.py": (
            project_root / "src/alberta_framework/steps/step6.py"
        ).exists(),
        "tests/test_average_reward.py": (
            project_root / "tests/test_average_reward.py"
        ).exists(),
        "tests/test_step5_step6_production.py": (
            project_root / "tests/test_step5_step6_production.py"
        ).exists(),
        "benchmarks/step6_continuing_control.py": (
            project_root / "benchmarks/step6_continuing_control.py"
        ).exists(),
        "benchmarks/step6_riverswim_stochastic.py": (
            project_root / "benchmarks/step6_riverswim_stochastic.py"
        ).exists(),
        "benchmarks/step6_security_gym_integration.py": (
            project_root / "benchmarks/step6_security_gym_integration.py"
        ).exists(),
    }

    step6_source = (project_root / "src/alberta_framework/steps/step6.py").read_text(
        encoding="utf-8"
    )
    average_reward_source = (
        project_root / "src/alberta_framework/core/average_reward.py"
    ).read_text(encoding="utf-8")
    production_tests = (
        project_root / "tests/test_step5_step6_production.py"
    ).read_text(encoding="utf-8")
    average_reward_tests = (
        project_root / "tests/test_average_reward.py"
    ).read_text(encoding="utf-8")

    differential_sarsa_surface = all(
        marker in average_reward_source
        for marker in (
            "class DifferentialSARSAAgent",
            "class DifferentialSARSAConfig",
            "run_differential_sarsa_from_arrays",
            "average_reward_step_size",
            "trace_decay",
        )
    ) and all(
        marker in step6_source
        for marker in (
            "Step6DifferentialSARSAConfig",
            "make_step6_differential_sarsa_agent",
            "init_step6_state",
            "step6_update",
            "run_step6_scan",
        )
    )
    test_surface = all(
        marker in production_tests + average_reward_tests
        for marker in (
            "test_step6_facade_config_roundtrip_one_step_and_smoke",
            "test_differential_sarsa_config_roundtrip_and_exact_td_error",
            "test_differential_sarsa_update_and_scan_are_finite",
            "test_differential_sarsa_learns_better_action_on_continuing_bandit",
        )
    )

    deterministic_aggregate = (
        {} if deterministic is None else deterministic.get("aggregate", {})
    )
    deterministic_passed = bool(
        deterministic is not None
        and deterministic.get("schema") == "alberta.step6.chain_continuing_control.v1"
        and deterministic.get("passed") is True
        and deterministic_aggregate.get("n_seeds", 0) >= 10
        and deterministic_aggregate.get("n_passed")
        == deterministic_aggregate.get("n_seeds")
        and deterministic_aggregate.get("mean_final_window_reward", 0.0) >= 0.95
        and deterministic_aggregate.get("mean_right_action_rate", 0.0) >= 0.95
    )

    stochastic_passed = bool(
        stochastic is not None
        and stochastic.get("schema")
        == "alberta.step6.riverswim_stochastic_benchmark.v1"
        and stochastic.get("passed") is True
        and stochastic.get("n_seeds", 0) >= 10
        and stochastic.get("right_wins") == stochastic.get("n_seeds")
        and stochastic.get("reward_wins") == stochastic.get("n_seeds")
        and stochastic.get("mean_final_reward", 0.0) >= 0.30
        and stochastic.get("mean_final_right_rate", 0.0) >= 0.80
    )

    multistate_aggregate = {} if multistate is None else multistate.get("aggregate", {})
    multistate_passed = bool(
        multistate is not None
        and multistate.get("schema")
        == "alberta.step5.multistate_continuing_control_benchmark.v1"
        and multistate.get("passed") is True
        and multistate_aggregate.get("n_seeds", 0) >= 10
        and multistate_aggregate.get("n_passed")
        == multistate_aggregate.get("n_seeds")
        and multistate_aggregate.get("mean_final_reward", 0.0) >= 0.99
        and multistate_aggregate.get("mean_final_policy_match_rate", 0.0) >= 0.99
    )

    nonlinear_aggregate = (
        {}
        if nonlinear_actor_critic is None
        else nonlinear_actor_critic.get("aggregate", {})
    )
    nonlinear_actor_critic_passed = bool(
        nonlinear_actor_critic is not None
        and nonlinear_actor_critic.get("schema")
        == "alberta.step5.average_reward_horde_actor_critic_benchmark.v1"
        and nonlinear_actor_critic.get("passed") is True
        and nonlinear_aggregate.get("n_seeds", 0) >= 10
        and nonlinear_aggregate.get("n_passed") == nonlinear_aggregate.get("n_seeds")
        and nonlinear_aggregate.get("mean_final_reward", 0.0) >= 0.97
        and nonlinear_aggregate.get("mean_final_policy_match_rate", 0.0) >= 0.97
    )
    security_gym_aggregate = (
        {} if security_gym is None else security_gym.get("aggregate", {})
    )
    security_gym_passed = bool(
        security_gym is not None
        and security_gym.get("schema") == "alberta.step6.security_gym_integration.v1"
        and security_gym_aggregate.get("passed") is True
        and security_gym_aggregate.get("n_seeds", 0) >= 10
        and security_gym_aggregate.get("reward_win_count", 0)
        == security_gym_aggregate.get("n_seeds")
        and security_gym_aggregate.get("mean_reward_improvement_vs_pass", 0.0)
        >= 1.0
        and security_gym_aggregate.get("mean_attack_alert_rate", 0.0) >= 0.85
        and security_gym_aggregate.get("mean_benign_pass_rate", 0.0) >= 0.85
    )

    evidence = {
        "implementation_files": {
            "files": implementation_files,
            "passed": all(implementation_files.values()),
        },
        "differential_sarsa_surface": {
            "implemented": differential_sarsa_surface,
            "tests_present": test_surface,
        },
        "seeded_deterministic_chain_control": {
            "path": str(deterministic_path),
            "exists": deterministic is not None,
            "passed": deterministic_passed,
            "aggregate": deterministic_aggregate,
            "baselines": deterministic.get("baselines", {})
            if deterministic is not None
            else {},
        },
        "seeded_stochastic_riverswim_control": {
            "path": str(stochastic_path),
            "exists": stochastic is not None,
            "passed": stochastic_passed,
            "aggregate": {
                key: stochastic.get(key)
                for key in (
                    "mean_final_reward",
                    "stderr_final_reward",
                    "mean_final_right_rate",
                    "stderr_final_right_rate",
                    "right_wins",
                    "reward_wins",
                    "n_seeds",
                    "random_baseline_avg_reward",
                    "optimal_avg_reward_approx",
                )
            }
            if stochastic is not None
            else {},
        },
        "seeded_multistate_policy_control": {
            "path": str(multistate_path),
            "exists": multistate is not None,
            "passed": multistate_passed,
            "aggregate": multistate_aggregate,
        },
        "seeded_nonlinear_average_reward_actor_critic": {
            "path": str(nonlinear_actor_critic_path),
            "exists": nonlinear_actor_critic is not None,
            "passed": nonlinear_actor_critic_passed,
            "aggregate": nonlinear_aggregate,
        },
        "downstream_security_gym_integration": {
            "path": str(security_gym_path),
            "exists": security_gym is not None,
            "passed": security_gym_passed,
            "aggregate": security_gym_aggregate,
        },
    }
    accepted = bool(
        evidence["implementation_files"]["passed"]
        and differential_sarsa_surface
        and test_surface
        and deterministic_passed
        and stochastic_passed
        and multistate_passed
        and nonlinear_actor_critic_passed
        and security_gym_passed
    )
    return {
        "schema": "alberta.step6.solution_gate.v1",
        "accepted_step6_continuing_control": accepted,
        "claim_scope": (
            "average_reward_continuing_control_completion"
            if accepted
            else "step6_continuing_control_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [] if accepted else [
            "Step 6 continuing-control completion evidence incomplete",
        ],
    }


def main() -> None:
    """Print the Step 6 solution-gate report."""
    print(json.dumps(run_step6_solution_gate(), indent=2))


if __name__ == "__main__":
    main()

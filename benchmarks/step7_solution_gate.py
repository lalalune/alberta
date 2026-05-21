"""Audit Step 7 Dyna planning evidence against completion criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_step7_solution_gate(root: Path | None = None) -> dict[str, Any]:
    """Return Step 7 primitive audit status."""
    project_root = root or Path(__file__).resolve().parents[1]
    benchmark_path = project_root / "outputs/step7_dyna/results.json"
    chain_benchmark_path = (
        project_root / "outputs/step7_chain_planning/results_numpy.json"
    )
    nonlinear_benchmark_path = (
        project_root / "outputs/step7_nonlinear_feature_planning/results.json"
    )
    production_nonlinear_benchmark_path = (
        project_root / "outputs/step7_production_nonlinear_dyna/results.json"
    )
    benchmark = _read_json(benchmark_path)
    chain_benchmark = _read_json(chain_benchmark_path)
    nonlinear_benchmark = _read_json(nonlinear_benchmark_path)
    production_nonlinear_benchmark = _read_json(production_nonlinear_benchmark_path)

    implementation_files = {
        "src/alberta_framework/steps/step7.py": (
            project_root / "src/alberta_framework/steps/step7.py"
        ).exists(),
        "tests/test_step5_step6_production.py": (
            project_root / "tests/test_step5_step6_production.py"
        ).exists(),
        "docs/research/step7_dyna_planning.md": (
            project_root / "docs/research/step7_dyna_planning.md"
        ).exists(),
        "benchmarks/step7_dyna_sample_efficiency.py": (
            project_root / "benchmarks/step7_dyna_sample_efficiency.py"
        ).exists(),
        "benchmarks/step7_nonlinear_feature_planning.py": (
            project_root / "benchmarks/step7_nonlinear_feature_planning.py"
        ).exists(),
        "benchmarks/step7_production_nonlinear_dyna.py": (
            project_root / "benchmarks/step7_production_nonlinear_dyna.py"
        ).exists(),
    }
    step7_source = (project_root / "src/alberta_framework/steps/step7.py").read_text()
    step7_tests = (
        project_root / "tests/test_step5_step6_production.py"
    ).read_text()
    planning_memory_implemented = all(
        marker in step7_source
        for marker in (
            "planning_memory_size",
            "memory_observations",
            "_select_planning_anchor",
            '"predecessor"',
        )
    )
    off_policy_accounting_implemented = all(
        marker in step7_source
        for marker in (
            "planning_behavior_probs",
            "planning_target_probs",
            "planning_importance_ratios",
            "_epsilon_greedy_action_probability",
        )
    )
    off_policy_correction_implemented = all(
        marker in step7_source
        for marker in (
            "planning_apply_importance_correction",
            "_apply_planning_importance_correction",
            "importance_ratio",
        )
    )
    prioritized_sweeping_queue_implemented = all(
        marker in step7_source
        for marker in (
            '"prioritized"',
            "_pop_prioritized_planning_anchor",
            "_propagate_predecessor_priorities",
            "planning_priority_propagation",
        )
    )
    short_rollout_planning_implemented = all(
        marker in step7_source
        for marker in (
            "planning_rollout_depth",
            "rollout_step",
            "rollout_td_signal",
        )
    )
    learned_search_control_implemented = all(
        marker in step7_source
        for marker in (
            '"learned"',
            "memory_utilities",
            "planning_utility_step_size",
            "_update_planning_utility",
        )
    )
    predecessor_test_present = (
        "test_step7_predecessor_search_control_selects_matching_memory_anchor"
        in step7_tests
    )
    off_policy_test_present = (
        "test_step7_planning_records_target_behavior_policy_ratios" in step7_tests
    )
    off_policy_correction_test_present = (
        "test_step7_importance_correction_scales_imagined_update_delta"
        in step7_tests
    )
    prioritized_queue_test_present = all(
        marker in step7_tests
        for marker in (
            "test_step7_prioritized_queue_pops_highest_priority_anchor",
            "test_step7_prioritized_queue_propagates_to_predecessors",
            "test_step7_prioritized_planning_updates_priority_queue",
        )
    )
    short_rollout_test_present = (
        "test_step7_short_rollout_depth_spends_multiple_imagined_backups"
        in step7_tests
    )
    learned_search_control_test_present = all(
        marker in step7_tests
        for marker in (
            "test_step7_learned_search_control_selects_high_utility_anchor",
            "test_step7_planning_utility_tracks_backup_td_signal",
            "test_step7_learned_strategy_updates_selected_memory_utility",
        )
    )

    aggregate = {} if benchmark is None else benchmark.get("aggregate", {})
    benchmark_passed = bool(
        benchmark is not None
        and benchmark.get("schema") == "alberta.step7.dyna_sample_efficiency.v1"
        and aggregate.get("passed") is True
        and aggregate.get("mean_reward_improvement", 0.0) >= 0.04
        and aggregate.get("mean_q_gap_improvement", 0.0) >= 1.0
        and aggregate.get("q_gap_win_count", 0) == benchmark.get("config", {}).get("seeds")
    )
    chain_aggregate = {} if chain_benchmark is None else chain_benchmark.get("aggregate", {})
    chain_benchmark_passed = bool(
        chain_benchmark is not None
        and chain_benchmark.get("schema") == "alberta.step7.numpy_planning.v1"
        and chain_aggregate.get("passed") is True
        and chain_aggregate.get("dyna_win_count_cumulative", 0) >= 7
        and chain_aggregate.get("mean_convergence_speedup_steps", 0.0) > 0.0
    )
    nonlinear_aggregate = (
        {} if nonlinear_benchmark is None else nonlinear_benchmark.get("aggregate", {})
    )
    nonlinear_benchmark_passed = bool(
        nonlinear_benchmark is not None
        and nonlinear_benchmark.get("schema")
        == "alberta.step7.nonlinear_feature_planning.v1"
        and nonlinear_aggregate.get("passed") is True
        and nonlinear_aggregate.get("mean_final_window_improvement", 0.0) >= 0.08
        and nonlinear_aggregate.get("final_window_win_count", 0) >= 6
    )
    production_nonlinear_aggregate = (
        {}
        if production_nonlinear_benchmark is None
        else production_nonlinear_benchmark.get("aggregate", {})
    )
    production_hidden_sizes = production_nonlinear_aggregate.get(
        "model_hidden_sizes",
        [],
    )
    production_nonlinear_benchmark_passed = bool(
        production_nonlinear_benchmark is not None
        and production_nonlinear_benchmark.get("schema")
        == "alberta.step7.production_nonlinear_dyna.v1"
        and production_nonlinear_aggregate.get("passed") is True
        and production_nonlinear_aggregate.get(
            "uses_production_step7_jax_facade",
        )
        is True
        and isinstance(production_hidden_sizes, list)
        and len(production_hidden_sizes) >= 1
        and production_nonlinear_aggregate.get(
            "mean_final_window_improvement",
            0.0,
        )
        >= 0.07
        and production_nonlinear_aggregate.get("mean_q_gap_improvement", 0.0)
        >= 2.0
        and production_nonlinear_aggregate.get("q_gap_win_count", 0) >= 8
    )

    evidence = {
        "implementation_files": implementation_files,
        "bounded_planning_memory": {
            "implemented": planning_memory_implemented,
            "predecessor_anchor_test_present": predecessor_test_present,
        },
        "off_policy_accounting": {
            "implemented": off_policy_accounting_implemented,
            "policy_ratio_test_present": off_policy_test_present,
        },
        "off_policy_correction": {
            "implemented": off_policy_correction_implemented,
            "importance_scaled_update_test_present": off_policy_correction_test_present,
        },
        "prioritized_sweeping_queue": {
            "implemented": prioritized_sweeping_queue_implemented,
            "queue_and_propagation_tests_present": prioritized_queue_test_present,
        },
        "short_rollout_planning": {
            "implemented": short_rollout_planning_implemented,
            "rollout_depth_test_present": short_rollout_test_present,
        },
        "learned_search_control": {
            "implemented": learned_search_control_implemented,
            "utility_tests_present": learned_search_control_test_present,
        },
        "seeded_dyna_sample_efficiency_benchmark": {
            "path": str(benchmark_path),
            "exists": benchmark is not None,
            "passed": benchmark_passed,
            "aggregate": aggregate,
        },
        "seeded_tabular_chain_planning_benchmark": {
            "path": str(chain_benchmark_path),
            "exists": chain_benchmark is not None,
            "passed": chain_benchmark_passed,
            "aggregate": chain_aggregate,
        },
        "seeded_nonlinear_feature_planning_benchmark": {
            "path": str(nonlinear_benchmark_path),
            "exists": nonlinear_benchmark is not None,
            "passed": nonlinear_benchmark_passed,
            "aggregate": nonlinear_aggregate,
        },
        "production_jax_nonlinear_dyna_benchmark": {
            "path": str(production_nonlinear_benchmark_path),
            "exists": production_nonlinear_benchmark is not None,
            "passed": production_nonlinear_benchmark_passed,
            "aggregate": production_nonlinear_aggregate,
        },
    }
    accepted = bool(
        all(implementation_files.values())
        and planning_memory_implemented
        and predecessor_test_present
        and off_policy_accounting_implemented
        and off_policy_test_present
        and off_policy_correction_implemented
        and off_policy_correction_test_present
        and prioritized_sweeping_queue_implemented
        and prioritized_queue_test_present
        and short_rollout_planning_implemented
        and short_rollout_test_present
        and learned_search_control_implemented
        and learned_search_control_test_present
        and benchmark_passed
        and chain_benchmark_passed
        and nonlinear_benchmark_passed
        and production_nonlinear_benchmark_passed
    )
    return {
        "schema": "alberta.step7.solution_gate.v1",
        "accepted_step7_dyna_planning_primitive": accepted,
        "claim_scope": (
            "bounded_dyna_average_reward_control_local_completion"
            if accepted
            else "step7_dyna_planning_incomplete"
        ),
        "evidence": evidence,
        "remaining_research_boundaries": [],
    }


def main() -> None:
    """Print the Step 7 solution-gate report."""
    print(json.dumps(run_step7_solution_gate(), indent=2))


if __name__ == "__main__":
    main()

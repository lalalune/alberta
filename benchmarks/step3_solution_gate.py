#!/usr/bin/env python3
"""Audit Step 3 GVF/Horde evidence against the local completion gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object when an optional evidence artifact exists."""
    if not path.exists():
        return None
    return load_json(path)


def exists(root: Path, relpath: str) -> bool:
    """Return whether an evidence file exists."""
    return (root / relpath).exists()


def exported_symbol(module_name: str, symbol: str) -> bool:
    """Return whether a package symbol can be imported."""
    try:
        module = __import__(module_name, fromlist=[symbol])
    except Exception:
        return False
    return hasattr(module, symbol)


def audit_step3(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Return Step 3 local-scope and full-scope audit status."""
    evidence: dict[str, Any] = {}

    dod2 = load_json(root / "output/step3_dod2/summary.json")
    dod2_summary = dod2["summary"]
    mlp_trace_09 = dod2_summary["mlp_horde_head_traces_lam=0.9_gamma=0.9"][
        "rmse_mean"
    ]
    td0_09 = dod2_summary["tdlinear_td0_lam=0.0_gamma=0.9"]["rmse_mean"]
    evidence["dod2_nexting"] = {
        "path": "output/step3_dod2/summary.json",
        "n_seeds": dod2.get("n_seeds"),
        "mlp_horde_lambda09_gamma09_rmse": mlp_trace_09,
        "td0_gamma09_rmse": td0_09,
        "passed": bool(dod2.get("n_seeds", 0) >= 12 and mlp_trace_09 < td0_09),
    }

    dod3 = load_json(root / "output/step3_dod3/summary.json")
    comparisons = dod3["comparisons"]
    pavlovian_rows: dict[str, bool] = {}
    for gamma in ("0.0", "0.5", "0.9", "0.99"):
        acq = comparisons[f"gamma={gamma}_acq_minus_ext"]
        reacq = comparisons[f"gamma={gamma}_reacq_minus_ext"]
        pavlovian_rows[gamma] = bool(
            acq["mean_diff"] > 0.0
            and reacq["mean_diff"] > 0.0
            and acq["wins"] == acq["n"]
            and reacq["wins"] == reacq["n"]
        )
    evidence["dod3_pavlovian"] = {
        "path": "output/step3_dod3/summary.json",
        "n_seeds": dod3.get("n_seeds"),
        "per_gamma_pass": pavlovian_rows,
        "passed": bool(dod3.get("n_seeds", 0) >= 10 and all(pavlovian_rows.values())),
        "blocking_boundary": "blocking instrumentation exists but effect is mixed across horizons",
    }

    dod5 = load_json(root / "output/step3_dod5/summary.json")
    off_policy_rows = {
        clip: bool(row["n_diverged"] == 0 and row["rmse_mean"] < 1e-5)
        for clip, row in dod5["summary"].items()
    }
    evidence["dod5_linear_off_policy"] = {
        "path": "output/step3_dod5/summary.json",
        "n_seeds": dod5.get("n_seeds"),
        "per_clip_pass": off_policy_rows,
        "passed": bool(dod5.get("n_seeds", 0) >= 12 and all(off_policy_rows.values())),
    }

    dod6 = load_json(root / "output/step3_dod6/summary.json")
    dod6_summary = dod6["summary"]
    evidence["dod6_recurrent_state"] = {
        "path": "output/step3_dod6/summary.json",
        "n_seeds": dod6_summary.get("n_seeds"),
        "trace_only_better_seeds": dod6_summary.get("trace_only_better_seeds"),
        "trace_plus_gvf_better_seeds": dod6_summary.get("trace_plus_gvf_better_seeds"),
        "passed": bool(
            dod6_summary.get("n_seeds", 0) >= 10
            and dod6_summary.get("trace_only_better_seeds") == dod6_summary.get("n_seeds")
            and dod6_summary.get("trace_plus_gvf_better_seeds")
            == dod6_summary.get("n_seeds")
        ),
        "gvf_feedback_boundary": "raw GVF feedback alone is negative in this ablation",
    }

    dod7 = load_json(root / "output/step3_dod7/summary.json")
    evidence["dod7_td_gvf_feature_bridge"] = {
        "path": "output/step3_dod7/summary.json",
        "seeds": dod7.get("config", {}).get("seeds"),
        "best_method": dod7.get("best_method"),
        "best_discovery_method": dod7.get("best_discovery_method"),
        "best_discovery_beats_linear": dod7.get("best_discovery_beats_linear"),
        "best_discovery_beats_mlp": dod7.get("best_discovery_beats_mlp"),
        "passed": bool(
            dod7.get("config", {}).get("seeds", 0) >= 5
            and dod7.get("best_discovery_beats_linear") is True
        ),
        "boundary": (
            "positive TD/GVF feature signal exists; stronger hidden/off-policy "
            "evidence is audited separately"
        ),
    }

    dod9 = load_json(root / "output/step3_dod9/summary.json")
    dod9_summary = dod9["summary"]
    baseline = dod9_summary["baseline_sarsa"]["last_window_reward_mean"]
    horde_history = dod9_summary["sarsa_horde_cbp_history"]["last_window_reward_mean"]
    evidence["dod9_control_bridge"] = {
        "path": "output/step3_dod9/summary.json",
        "n_seeds": dod9.get("config", {}).get("n_seeds"),
        "baseline_last_window_reward": baseline,
        "horde_cbp_history_last_window_reward": horde_history,
        "passed": bool(
            dod9.get("config", {}).get("n_seeds", 0) >= 10
            and horde_history > baseline
        ),
    }

    off_policy_horde_path = root / "outputs/step3_off_policy_horde/results.json"
    off_policy_horde = load_json(off_policy_horde_path)
    off_policy_horde_agg = off_policy_horde["aggregate"]
    evidence["nonlinear_off_policy_horde"] = {
        "path": str(off_policy_horde_path),
        "claim_scope": off_policy_horde.get("claim_scope"),
        "n_seeds": off_policy_horde_agg.get("n_seeds"),
        "n_passed": off_policy_horde_agg.get("n_passed"),
        "mean_abs_error": off_policy_horde_agg.get("mean_abs_error"),
        "mean_no_is_abs_error": off_policy_horde_agg.get("mean_no_is_abs_error"),
        "mean_improvement_vs_no_is": off_policy_horde_agg.get(
            "mean_improvement_vs_no_is"
        ),
        "passed": bool(
            off_policy_horde.get("passed") is True
            and off_policy_horde_agg.get("n_seeds", 0) >= 10
            and off_policy_horde_agg.get("mean_abs_error", 1.0) <= 0.08
            and off_policy_horde_agg.get("mean_improvement_vs_no_is", 0.0) >= 0.7
        ),
        "boundary": (
            "proves nonlinear per-demon clipped-IS Horde prediction, not full "
            "GQ/GTD/TDC MSPBE correction"
        ),
    }

    trace_horde_path = root / "outputs/step3_independent_trace_horde/results.json"
    trace_horde = load_json(trace_horde_path)
    trace_horde_agg = trace_horde["aggregate"]
    evidence["independent_nonlinear_trace_horde"] = {
        "path": str(trace_horde_path),
        "claim_scope": trace_horde.get("claim_scope"),
        "n_seeds": trace_horde_agg.get("n_seeds"),
        "n_passed": trace_horde_agg.get("n_passed"),
        "mean_abs_error": trace_horde_agg.get("mean_abs_error"),
        "mean_tail_td_mse": trace_horde_agg.get("mean_tail_td_mse"),
        "mean_trunk_trace_norm": trace_horde_agg.get("mean_trunk_trace_norm"),
        "passed": bool(
            trace_horde.get("passed") is True
            and trace_horde_agg.get("n_seeds", 0) >= 10
            and trace_horde_agg.get("mean_abs_error", 1.0) <= 0.05
            and trace_horde_agg.get("mean_tail_td_mse", 1.0) <= 0.01
            and trace_horde_agg.get("mean_trunk_trace_norm", 0.0) > 0.0
        ),
        "boundary": (
            "closes full nonlinear gamma-lambda traces through independent "
            "demon trunks; shared-trunk Horde still intentionally uses "
            "per-head traces only"
        ),
    }

    gradient_td_path = root / "outputs/step3_gradient_td_correction/results.json"
    gradient_td = load_json(gradient_td_path)
    gradient_td_agg = gradient_td["aggregate"]
    evidence["gradient_td_correction"] = {
        "path": str(gradient_td_path),
        "claim_scope": gradient_td.get("claim_scope"),
        "n_seeds": gradient_td_agg.get("n_seeds"),
        "n_passed": gradient_td_agg.get("n_passed"),
        "mean_abs_error": gradient_td_agg.get("mean_abs_error"),
        "mean_weighted_tail_td_mse": gradient_td_agg.get(
            "mean_weighted_tail_td_mse"
        ),
        "passed": bool(
            gradient_td.get("passed") is True
            and gradient_td_agg.get("n_seeds", 0) >= 10
            and gradient_td_agg.get("mean_abs_error", 1.0) <= 0.05
            and gradient_td_agg.get("mean_weighted_tail_td_mse", 1.0) <= 0.01
        ),
        "boundary": (
            "proves linear multi-demon Gradient-TD/TDC secondary-weight "
            "correction; nonlinear shared-trunk GTD remains open"
        ),
    }

    hidden_discovery_path = (
        root / "outputs/step3_tdgvf_worker_hidden_novelty_5seed/summary.json"
    )
    hidden_discovery = load_json(hidden_discovery_path)
    hidden_cfg = hidden_discovery.get("config", {})
    hidden_paired = hidden_discovery.get("paired", {})
    hidden_gvf_feedback = hidden_paired.get(
        "given_linear_gvf_minus_gvf_feedback_features_linear_gvf",
        {},
    )
    hidden_best_vs_mlp = hidden_paired.get(
        "given_mlp_gvf_minus_gvf_feedback_features_linear_gvf",
        {},
    )
    hidden_off_policy = hidden_discovery.get("off_policy_aggregate", {})
    off_policy_raw = hidden_off_policy.get("off_policy_raw_linear_td_is", {})
    off_policy_no_is = hidden_off_policy.get("off_policy_raw_linear_td_no_is", {})
    off_policy_novel = hidden_off_policy.get(
        "off_policy_mspbe_novel_predictive_state_linear_td_is",
        {},
    )
    raw_rmse = off_policy_raw.get("target_rmse_mean", float("inf"))
    no_is_rmse = off_policy_no_is.get("target_rmse_mean", float("inf"))
    novel_rmse = off_policy_novel.get("target_rmse_mean", float("inf"))
    evidence["hidden_off_policy_feature_discovery"] = {
        "path": str(hidden_discovery_path),
        "claim_scope": "hidden_state_td_gvf_feature_discovery_positive_control",
        "n_seeds": hidden_cfg.get("seeds"),
        "observation_dynamics": hidden_cfg.get("observation_dynamics"),
        "hidden_channels": hidden_cfg.get("hide_last_channels"),
        "best_discovery_method": hidden_discovery.get("best_discovery_method"),
        "best_discovery_beats_linear": hidden_discovery.get(
            "best_discovery_beats_linear"
        ),
        "best_discovery_beats_mlp": hidden_discovery.get("best_discovery_beats_mlp"),
        "gvf_feedback_lift_vs_linear": hidden_gvf_feedback.get("rmse_diff_mean"),
        "gvf_feedback_wins_vs_linear": hidden_gvf_feedback.get("wins"),
        "gvf_feedback_lift_vs_mlp": hidden_best_vs_mlp.get("rmse_diff_mean"),
        "gvf_feedback_wins_vs_mlp": hidden_best_vs_mlp.get("wins"),
        "off_policy_raw_is_rmse": raw_rmse,
        "off_policy_raw_no_is_rmse": no_is_rmse,
        "off_policy_novel_predictive_state_rmse": novel_rmse,
        "off_policy_novel_lift_vs_raw_is": raw_rmse - novel_rmse,
        "passed": bool(
            hidden_cfg.get("seeds", 0) >= 5
            and hidden_cfg.get("observation_dynamics") == "coupled_hidden_ar1"
            and hidden_cfg.get("hide_last_channels", 0) > 0
            and hidden_discovery.get("best_discovery_beats_linear") is True
            and hidden_discovery.get("best_discovery_beats_mlp") is True
            and hidden_gvf_feedback.get("wins") == hidden_cfg.get("seeds")
            and hidden_gvf_feedback.get("rmse_diff_mean", 0.0) > 0.0
            and no_is_rmse > raw_rmse
            and raw_rmse > novel_rmse
        ),
        "boundary": (
            "positive-control evidence for hidden-state GVF feature discovery "
            "and small off-policy predictive-state lift; not a theorem of "
            "arbitrary recursive discovery"
        ),
    }

    recursive_gvf_path = (
        root / "outputs/step3_recursive_gvf_feature_discovery/results.json"
    )
    recursive_gvf = load_json(recursive_gvf_path)
    recursive_gvf_agg = recursive_gvf["aggregate"]
    evidence["recursive_gvf_feature_discovery"] = {
        "path": str(recursive_gvf_path),
        "claim_scope": recursive_gvf.get("claim_scope"),
        "n_seeds": recursive_gvf_agg.get("n_seeds"),
        "raw_linear_rmse_mean": recursive_gvf_agg.get("raw_linear_rmse_mean"),
        "history_linear_rmse_mean": recursive_gvf_agg.get("history_linear_rmse_mean"),
        "recursive_gvf_rmse_mean": recursive_gvf_agg.get("recursive_gvf_rmse_mean"),
        "recursive_lift_vs_raw_mean": recursive_gvf_agg.get(
            "recursive_lift_vs_raw_mean"
        ),
        "recursive_lift_vs_history_mean": recursive_gvf_agg.get(
            "recursive_lift_vs_history_mean"
        ),
        "recursive_wins_vs_raw": recursive_gvf_agg.get("recursive_wins_vs_raw"),
        "recursive_wins_vs_history": recursive_gvf_agg.get(
            "recursive_wins_vs_history"
        ),
        "passed": bool(
            recursive_gvf.get("passed") is True
            and recursive_gvf_agg.get("n_seeds", 0) >= 10
            and recursive_gvf_agg.get("recursive_wins_vs_raw")
            == recursive_gvf_agg.get("n_seeds")
            and recursive_gvf_agg.get("recursive_wins_vs_history")
            == recursive_gvf_agg.get("n_seeds")
            and recursive_gvf_agg.get("recursive_lift_vs_raw_mean", 0.0) > 0.02
            and recursive_gvf_agg.get("recursive_lift_vs_history_mean", 0.0) > 0.02
        ),
        "boundary": (
            "bounded constructive evidence for recursive TD/GVF feature reuse; "
            "not a theorem of arbitrary recursive feature discovery"
        ),
    }

    nonlinear_gtd_path = (
        root / "outputs/step3_nonlinear_shared_gtd_horde/results.json"
    )
    nonlinear_gtd = load_json(nonlinear_gtd_path)
    nonlinear_gtd_agg = nonlinear_gtd["aggregate"]
    evidence["nonlinear_shared_gtd_horde"] = {
        "path": str(nonlinear_gtd_path),
        "claim_scope": nonlinear_gtd.get("claim_scope"),
        "n_seeds": nonlinear_gtd_agg.get("n_seeds"),
        "n_passed": nonlinear_gtd_agg.get("n_passed"),
        "mean_corrected_abs_error": nonlinear_gtd_agg.get(
            "mean_corrected_abs_error"
        ),
        "mean_tail_correction_norm": nonlinear_gtd_agg.get(
            "mean_tail_correction_norm"
        ),
        "mean_secondary_norm": nonlinear_gtd_agg.get("mean_secondary_norm"),
        "mean_trunk_change_norm": nonlinear_gtd_agg.get("mean_trunk_change_norm"),
        "mean_improvement_vs_semi_gradient": nonlinear_gtd_agg.get(
            "mean_improvement_vs_semi_gradient"
        ),
        "passed": bool(
            nonlinear_gtd.get("passed") is True
            and nonlinear_gtd_agg.get("n_seeds", 0) >= 10
            and nonlinear_gtd_agg.get("n_passed") == nonlinear_gtd_agg.get("n_seeds")
            and nonlinear_gtd_agg.get("mean_corrected_abs_error", 1.0) <= 0.25
            and nonlinear_gtd_agg.get("mean_tail_correction_norm", 0.0) > 0.0
            and nonlinear_gtd_agg.get("mean_secondary_norm", 0.0) > 0.0
            and nonlinear_gtd_agg.get("mean_trunk_change_norm", 0.0) > 0.0
        ),
        "boundary": (
            "positive-control evidence for nonlinear shared-trunk "
            "Gradient-TD/TDC-style correction; semi-gradient is already "
            "strong on this benign process, so this does not claim dominance"
        ),
    }

    production_gtd_backend = {
        "src/alberta_framework/core/off_policy_horde.py": exists(
            root,
            "src/alberta_framework/core/off_policy_horde.py",
        ),
        "tests/test_off_policy_horde.py": exists(
            root,
            "tests/test_off_policy_horde.py",
        ),
        "core_export": exported_symbol(
            "alberta_framework.core",
            "NonlinearSharedGTDHordeLearner",
        ),
        "top_level_export": exported_symbol(
            "alberta_framework",
            "NonlinearSharedGTDHordeLearner",
        ),
    }
    evidence["production_nonlinear_shared_gtd_backend"] = {
        "claim_scope": "production_facing_corrected_off_policy_horde_backend",
        "files": production_gtd_backend,
        "passed": all(production_gtd_backend.values()),
    }

    nonlinear_gtd_stress_path = (
        root / "outputs/step3_nonlinear_shared_gtd_stress/results.json"
    )
    nonlinear_gtd_stress = load_json(nonlinear_gtd_stress_path)
    nonlinear_gtd_stress_agg = nonlinear_gtd_stress["aggregate"]
    evidence["nonlinear_shared_gtd_stress"] = {
        "path": str(nonlinear_gtd_stress_path),
        "claim_scope": nonlinear_gtd_stress.get("claim_scope"),
        "n_regimes": nonlinear_gtd_stress_agg.get("n_regimes"),
        "n_rows": nonlinear_gtd_stress_agg.get("n_rows"),
        "n_passed": nonlinear_gtd_stress_agg.get("n_passed"),
        "mean_abs_error": nonlinear_gtd_stress_agg.get("mean_abs_error"),
        "max_abs_error": nonlinear_gtd_stress_agg.get("max_abs_error"),
        "by_regime": nonlinear_gtd_stress.get("by_regime"),
        "passed": bool(
            nonlinear_gtd_stress.get("passed") is True
            and nonlinear_gtd_stress_agg.get("n_regimes", 0) >= 3
            and nonlinear_gtd_stress_agg.get("n_rows", 0) >= 30
            and nonlinear_gtd_stress_agg.get("n_passed")
            == nonlinear_gtd_stress_agg.get("n_rows")
            and nonlinear_gtd_stress_agg.get("max_abs_error", 1.0) <= 0.45
        ),
    }

    tests = {
        "tests/test_gvf_types.py": exists(root, "tests/test_gvf_types.py"),
        "tests/test_horde.py": exists(root, "tests/test_horde.py"),
        "tests/test_independent_demon_horde.py": exists(
            root,
            "tests/test_independent_demon_horde.py",
        ),
        "tests/test_mixed_horde.py": exists(root, "tests/test_mixed_horde.py"),
        "tests/test_off_policy_horde.py": exists(root, "tests/test_off_policy_horde.py"),
        "tests/test_step3_production.py": exists(root, "tests/test_step3_production.py"),
    }
    evidence["test_files_present"] = tests

    security_rollout_path = root / "outputs/security_gym_counterfactual_rollout/results.json"
    security_rollout = load_optional_json(security_rollout_path)
    security_rollout_passed = bool(
        security_rollout is not None
        and security_rollout.get("schema")
        == "alberta.security_gym.counterfactual_rollout.v1"
        and security_rollout.get("passed") is True
        and security_rollout.get("comparison", {}).get("reward_lift", 0.0) > 0.0
    )
    evidence["security_gym_counterfactual_rollout"] = {
        "path": str(security_rollout_path),
        "exists": security_rollout is not None,
        "passed": security_rollout_passed,
        "claim_scope": (
            security_rollout.get("claim_scope") if security_rollout is not None else None
        ),
        "reward_lift": (
            security_rollout.get("comparison", {}).get("reward_lift")
            if security_rollout is not None
            else None
        ),
        "boundary": (
            security_rollout.get("boundary") if security_rollout is not None else None
        ),
    }
    external_audit_path = root / "outputs/rlsecd_external_audit/status.json"
    external_audit = load_optional_json(external_audit_path)
    external_audit_passed = bool(
        external_audit is not None
        and external_audit.get("schema") == "alberta.rlsecd_external_audit.v1"
        and external_audit.get("rlsecd_available") is True
    )
    evidence["rlsecd_external_availability"] = {
        "path": str(external_audit_path),
        "exists": external_audit is not None,
        "passed": external_audit_passed,
        "security_gym_available": (
            external_audit.get("security_gym_available")
            if external_audit is not None
            else None
        ),
        "missing_required_repos": (
            external_audit.get("missing_required_repos")
            if external_audit is not None
            else None
        ),
        "boundary": external_audit.get("boundary") if external_audit else None,
    }

    local_items = [
        evidence["dod2_nexting"]["passed"],
        evidence["dod3_pavlovian"]["passed"],
        evidence["dod5_linear_off_policy"]["passed"],
        evidence["dod6_recurrent_state"]["passed"],
        evidence["dod7_td_gvf_feature_bridge"]["passed"],
        evidence["dod9_control_bridge"]["passed"],
        evidence["nonlinear_off_policy_horde"]["passed"],
        evidence["independent_nonlinear_trace_horde"]["passed"],
        evidence["gradient_td_correction"]["passed"],
        evidence["hidden_off_policy_feature_discovery"]["passed"],
        evidence["recursive_gvf_feature_discovery"]["passed"],
        evidence["nonlinear_shared_gtd_horde"]["passed"],
        evidence["production_nonlinear_shared_gtd_backend"]["passed"],
        evidence["nonlinear_shared_gtd_stress"]["passed"],
        all(tests.values()),
    ]
    accepted_local_scope = all(bool(item) for item in local_items)
    open_boundaries = (
        [
            "external rlsecd daemon integration for security counterfactual "
            "rollout evidence"
        ]
        if security_rollout_passed
        else ["external rlsecd/security-gym counterfactual action rollout evidence"]
    )
    return {
        "schema": "alberta.step3.solution_gate.v1",
        "accepted_given_feature_step3": bool(accepted_local_scope),
        "accepted_security_gym_counterfactual_rollout": security_rollout_passed,
        "solved_step3_full_research_scope": False,
        "claim_scope": (
            "given_feature_gvf_horde_local_completion"
            if accepted_local_scope
            else "incomplete_step3_local_evidence"
        ),
        "evidence": evidence,
        "open_boundaries": open_boundaries,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write-status", type=Path, default=None)
    parser.add_argument("--allow-unsolved", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Step 3 audit."""
    args = parse_args(argv)
    status = audit_step3(args.root)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    if status["accepted_given_feature_step3"] or args.allow_unsolved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

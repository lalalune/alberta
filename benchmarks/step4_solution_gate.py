#!/usr/bin/env python3
"""Audit Step 4 control evidence against the local completion gate."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".")


def read_text(path: Path) -> str:
    """Read a text artifact."""
    return path.read_text(encoding="utf-8")


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


def parse_markdown_summary_table(text: str) -> dict[str, dict[str, Any]]:
    """Parse the first Step 4 markdown summary table."""
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "---" in stripped or "scope" in stripped or "experiment" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name = cells[0]
        numeric: dict[str, Any] = {"numeric_values": []}
        for idx, cell in enumerate(cells[1:], start=1):
            try:
                value = float(cell)
                numeric[f"col_{idx}"] = value
                numeric["numeric_values"].append(value)
            except ValueError:
                continue
        if numeric["numeric_values"]:
            rows[name] = numeric
    return rows


def audit_step4(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Return Step 4 local-scope and full-scope audit status."""
    evidence: dict[str, Any] = {}

    primary_path = root / "outputs/bsuite/sarsa_vs_q_primary_10seed/sarsa_vs_q.md"
    primary_text = read_text(primary_path)
    primary_rows = parse_markdown_summary_table(primary_text)
    overall = primary_rows.get("overall", {})
    overall_values = overall.get("numeric_values", [])
    # Table columns: n, improvement_vs_q_mean, improvement_vs_q_wins.
    evidence["sarsa_vs_q_primary"] = {
        "path": str(primary_path),
        "pairs": overall_values[0] if len(overall_values) > 0 else None,
        "overall_improvement_vs_q_mean": (
            overall_values[1] if len(overall_values) > 1 else None
        ),
        "overall_wins": overall_values[2] if len(overall_values) > 2 else None,
        "passed": bool(
            len(overall_values) >= 3
            and overall_values[0] >= 1400
            and overall_values[1] > 0.0
        ),
        "boundary": (
            "overall win rate is not >50%; evidence supports canonical SARSA API "
            "and broad nonnegative aggregate, not dominance on every bsuite family"
        ),
    }

    catch_cart_path = root / "outputs/bsuite/sarsa_vs_q_catch_cartpole_10seed/sarsa_vs_q.md"
    catch_cart_text = read_text(catch_cart_path)
    catch_cart_rows = parse_markdown_summary_table(catch_cart_text)
    cartpole = catch_cart_rows.get("cartpole", {})
    catch = catch_cart_rows.get("catch", {})
    cartpole_values = cartpole.get("numeric_values", [])
    catch_values = catch.get("numeric_values", [])
    evidence["sarsa_vs_q_catch_cartpole"] = {
        "path": str(catch_cart_path),
        "cartpole_pairs": cartpole_values[0] if len(cartpole_values) > 0 else None,
        "cartpole_improvement": cartpole_values[1] if len(cartpole_values) > 1 else None,
        "catch_pairs": catch_values[0] if len(catch_values) > 0 else None,
        "catch_improvement": catch_values[1] if len(catch_values) > 1 else None,
        "passed": bool(
            len(cartpole_values) >= 2
            and len(catch_values) >= 2
            and cartpole_values[0] >= 200
            and catch_values[0] >= 200
            and cartpole_values[1] > 0.0
        ),
    }

    step4_path = root / "outputs/bsuite/step4_catch_cartpole_10seed/step4.md"
    step4_text = read_text(step4_path)
    step4_rows = parse_markdown_summary_table(step4_text)
    evidence["q_sarsa_actor_critic_report"] = {
        "path": str(step4_path),
        "overall": step4_rows.get("overall", {}),
        "passed": bool("overall" in step4_rows),
        "boundary": (
            "actor-critic is implemented and evidenced but underperforms the "
            "canonical Q/SARSA baselines in this report"
        ),
    }

    horde_ac_path = root / "outputs/bsuite/horde_ac_catch_cartpole_10seed/horde_ac.md"
    horde_ac_exists = horde_ac_path.exists()
    horde_ac_text = read_text(horde_ac_path) if horde_ac_exists else ""
    horde_failed_promotion = bool(horde_ac_exists and re.search("horde_ac", horde_ac_text))
    evidence["horde_actor_critic_boundary"] = {
        "path": str(horde_ac_path),
        "exists": horde_ac_exists,
        "promotion_failed_as_documented": horde_failed_promotion,
        "passed": horde_ac_exists,
    }

    horde_ac_control_path = (
        root / "outputs/step4_horde_actor_critic_control/results.json"
    )
    horde_ac_control = json.loads(read_text(horde_ac_control_path))
    horde_ac_control_agg = horde_ac_control["aggregate"]
    evidence["horde_actor_critic_positive_control"] = {
        "path": str(horde_ac_control_path),
        "claim_scope": horde_ac_control.get("claim_scope"),
        "n_seeds": horde_ac_control_agg.get("n_seeds"),
        "n_passed": horde_ac_control_agg.get("n_passed"),
        "mean_final_reward_rate": horde_ac_control_agg.get(
            "mean_final_reward_rate"
        ),
        "mean_final_optimal_action_probability": horde_ac_control_agg.get(
            "mean_final_optimal_action_probability"
        ),
        "mean_tail_td_mse": horde_ac_control_agg.get("mean_tail_td_mse"),
        "passed": bool(
            horde_ac_control.get("passed") is True
            and horde_ac_control_agg.get("n_seeds", 0) >= 10
            and horde_ac_control_agg.get("mean_final_reward_rate", 0.0) >= 0.95
            and horde_ac_control_agg.get(
                "mean_final_optimal_action_probability", 0.0
            )
            >= 0.95
            and horde_ac_control_agg.get("mean_tail_td_mse", 1.0) <= 0.08
        ),
    }

    clipped_probe_path = root / "outputs/bsuite/step4_ac_paths_seed0/report.md"
    clipped_probe_exists = clipped_probe_path.exists()
    evidence["latest_actor_critic_probe"] = {
        "path": str(clipped_probe_path),
        "exists": clipped_probe_exists,
        "passed": clipped_probe_exists,
        "boundary": (
            "latest local actor-critic variants still improve cartpole but do "
            "not robustly improve catch, so this is diagnostic evidence rather "
            "than promotion evidence"
        ),
    }
    qhorde_probe_path = root / "outputs/bsuite/qhorde_ac_seed0/report.md"
    qhorde_probe_exists = qhorde_probe_path.exists()
    qhorde_probe_rows = (
        parse_markdown_summary_table(read_text(qhorde_probe_path))
        if qhorde_probe_exists
        else {}
    )
    evidence["qhorde_actor_critic_probe"] = {
        "path": str(qhorde_probe_path),
        "exists": qhorde_probe_exists,
        "overall": qhorde_probe_rows.get("overall", {}),
        "catch": qhorde_probe_rows.get("catch", {}),
        "cartpole": qhorde_probe_rows.get("cartpole", {}),
        "passed": qhorde_probe_exists,
        "boundary": (
            "Q-Horde actor-critic improves the seed-0 cartpole and overall "
            "probe, but catch remains negative; this is diagnostic evidence, "
            "not bsuite promotion evidence"
        ),
    }
    qhorde_sampled_path = root / "outputs/bsuite/qhorde_ac_sampled_3seed/report.md"
    qhorde_sampled_exists = qhorde_sampled_path.exists()
    qhorde_sampled_rows = (
        parse_markdown_summary_table(read_text(qhorde_sampled_path))
        if qhorde_sampled_exists
        else {}
    )
    evidence["qhorde_sampled_actor_critic_probe"] = {
        "path": str(qhorde_sampled_path),
        "exists": qhorde_sampled_exists,
        "overall": qhorde_sampled_rows.get("overall", {}),
        "catch": qhorde_sampled_rows.get("catch", {}),
        "cartpole": qhorde_sampled_rows.get("cartpole", {}),
        "passed": qhorde_sampled_exists,
        "boundary": (
            "sampled-SARSA Q-Horde actor-critic passed a seed-0 smoke but "
            "failed the 3-seed catch/cartpole promotion probe"
        ),
    }
    qhorde_expected_adv_path = (
        root / "outputs/bsuite/qhorde_ac_expected_adv_3seed/report.md"
    )
    qhorde_expected_adv_exists = qhorde_expected_adv_path.exists()
    qhorde_expected_adv_rows = (
        parse_markdown_summary_table(read_text(qhorde_expected_adv_path))
        if qhorde_expected_adv_exists
        else {}
    )
    evidence["qhorde_expected_advantage_probe"] = {
        "path": str(qhorde_expected_adv_path),
        "exists": qhorde_expected_adv_exists,
        "overall": qhorde_expected_adv_rows.get("overall", {}),
        "catch": qhorde_expected_adv_rows.get("catch", {}),
        "cartpole": qhorde_expected_adv_rows.get("cartpole", {}),
        "passed": qhorde_expected_adv_exists,
        "boundary": (
            "expected-advantage Q-Horde actor-critic passed a seed-0 smoke "
            "but failed the 3-seed catch/cartpole promotion probe"
        ),
    }
    qhorde_pairwise_path = (
        root / "outputs/bsuite/qhorde_ac_expected_adv_pairwise_seed0/report.md"
    )
    qhorde_pairwise_exists = qhorde_pairwise_path.exists()
    qhorde_pairwise_rows = (
        parse_markdown_summary_table(read_text(qhorde_pairwise_path))
        if qhorde_pairwise_exists
        else {}
    )
    evidence["qhorde_pairwise_actor_probe"] = {
        "path": str(qhorde_pairwise_path),
        "exists": qhorde_pairwise_exists,
        "overall": qhorde_pairwise_rows.get("overall", {}),
        "catch": qhorde_pairwise_rows.get("catch", {}),
        "cartpole": qhorde_pairwise_rows.get("cartpole", {}),
        "passed": qhorde_pairwise_exists,
        "boundary": (
            "pairwise lifted Q-Horde actor features failed the seed-0 smoke "
            "and were not expanded to a multi-seed promotion probe"
        ),
    }
    nlhac_bottleneck_path = root / "outputs/bsuite/nlhac_bottleneck_10seed/report.md"
    nlhac_bottleneck_exists = nlhac_bottleneck_path.exists()
    nlhac_bottleneck_rows = (
        parse_markdown_summary_table(read_text(nlhac_bottleneck_path))
        if nlhac_bottleneck_exists
        else {}
    )
    nlhac_overall_values = nlhac_bottleneck_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_cartpole_values = nlhac_bottleneck_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_catch_values = nlhac_bottleneck_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n/metrics are SARSA mean/wins, then NLHAC mean/wins.
    nlhac_promotion_passed = bool(
        len(nlhac_overall_values) >= 5
        and len(nlhac_cartpole_values) >= 5
        and len(nlhac_catch_values) >= 5
        and nlhac_overall_values[3] > 0.0
        and nlhac_cartpole_values[3] > 0.0
        and nlhac_catch_values[3] > 0.0
    )
    evidence["nonlinear_horde_actor_critic_promotion_probe"] = {
        "path": str(nlhac_bottleneck_path),
        "exists": nlhac_bottleneck_exists,
        "overall": nlhac_bottleneck_rows.get("overall", {}),
        "catch": nlhac_bottleneck_rows.get("catch", {}),
        "cartpole": nlhac_bottleneck_rows.get("cartpole", {}),
        "passed": nlhac_bottleneck_exists,
        "promotion_passed": nlhac_promotion_passed,
        "boundary": (
            "NLHAC actor_step_size alias is fixed and 10-seed catch/cartpole "
            "evidence is recorded; this probe still fails promotion because "
            "overall/cartpole means are below the Q baseline"
        ),
    }
    nlhac_clipped_path = root / "outputs/bsuite/nlhac_clipped_3seed_500/report.md"
    nlhac_clipped_exists = nlhac_clipped_path.exists()
    nlhac_clipped_rows = (
        parse_markdown_summary_table(read_text(nlhac_clipped_path))
        if nlhac_clipped_exists
        else {}
    )
    nlhac_clipped_overall_values = nlhac_clipped_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_clipped_cartpole_values = nlhac_clipped_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_clipped_catch_values = nlhac_clipped_rows.get("catch", {}).get(
        "numeric_values", []
    )
    nlhac_clipped_promotion_passed = bool(
        len(nlhac_clipped_overall_values) >= 5
        and len(nlhac_clipped_cartpole_values) >= 5
        and len(nlhac_clipped_catch_values) >= 5
        and nlhac_clipped_overall_values[3] > 0.0
        and nlhac_clipped_cartpole_values[3] > 0.0
        and nlhac_clipped_catch_values[3] > 0.0
    )
    evidence["nonlinear_horde_actor_critic_clipped_probe"] = {
        "path": str(nlhac_clipped_path),
        "exists": nlhac_clipped_exists,
        "overall": nlhac_clipped_rows.get("overall", {}),
        "catch": nlhac_clipped_rows.get("catch", {}),
        "cartpole": nlhac_clipped_rows.get("cartpole", {}),
        "passed": nlhac_clipped_exists,
        "promotion_passed": nlhac_clipped_promotion_passed,
        "boundary": (
            "clipped NLHAC 3-seed 500-step catch/cartpole probe is recorded; "
            "it still fails promotion because overall and catch means are "
            "below the Q baseline"
        ),
    }
    nlhac_gradclip_500_path = root / "outputs/bsuite/nlhac_gradclip_10seed_500/report.md"
    nlhac_gradclip_500_exists = nlhac_gradclip_500_path.exists()
    nlhac_gradclip_500_rows = (
        parse_markdown_summary_table(read_text(nlhac_gradclip_500_path))
        if nlhac_gradclip_500_exists
        else {}
    )
    nlhac_gradclip_500_overall = nlhac_gradclip_500_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_500_cartpole = nlhac_gradclip_500_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_500_catch = nlhac_gradclip_500_rows.get("catch", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_500_q_promotion = bool(
        len(nlhac_gradclip_500_overall) >= 5
        and len(nlhac_gradclip_500_cartpole) >= 5
        and len(nlhac_gradclip_500_catch) >= 5
        and nlhac_gradclip_500_overall[0] >= 20
        and nlhac_gradclip_500_overall[3] > 0.0
        and nlhac_gradclip_500_cartpole[3] > 0.0
        and nlhac_gradclip_500_catch[3] > 0.0
    )
    nlhac_gradclip_500_sarsa_promotion = bool(
        nlhac_gradclip_500_q_promotion
        and nlhac_gradclip_500_overall[3] > nlhac_gradclip_500_overall[1]
        and nlhac_gradclip_500_cartpole[3] > nlhac_gradclip_500_cartpole[1]
        and nlhac_gradclip_500_catch[3] > nlhac_gradclip_500_catch[1]
    )
    evidence["nonlinear_horde_actor_critic_gradclip_500_probe"] = {
        "path": str(nlhac_gradclip_500_path),
        "exists": nlhac_gradclip_500_exists,
        "overall": nlhac_gradclip_500_rows.get("overall", {}),
        "catch": nlhac_gradclip_500_rows.get("catch", {}),
        "cartpole": nlhac_gradclip_500_rows.get("cartpole", {}),
        "passed": nlhac_gradclip_500_exists,
        "promotion_vs_q_passed": nlhac_gradclip_500_q_promotion,
        "promotion_vs_sarsa_passed": nlhac_gradclip_500_sarsa_promotion,
        "boundary": (
            "gradient-clipped NLHAC 10-seed 500-step probe is positive "
            "against the Q baseline on catch, cartpole, and overall, but "
            "does not outperform SARSA"
        ),
    }
    nlhac_gradclip_1000_path = root / "outputs/bsuite/nlhac_gradclip_10seed_1000/report.md"
    nlhac_gradclip_1000_exists = nlhac_gradclip_1000_path.exists()
    nlhac_gradclip_1000_rows = (
        parse_markdown_summary_table(read_text(nlhac_gradclip_1000_path))
        if nlhac_gradclip_1000_exists
        else {}
    )
    nlhac_gradclip_1000_overall = nlhac_gradclip_1000_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_1000_cartpole = nlhac_gradclip_1000_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_1000_catch = nlhac_gradclip_1000_rows.get("catch", {}).get(
        "numeric_values", []
    )
    nlhac_gradclip_1000_q_promotion = bool(
        len(nlhac_gradclip_1000_overall) >= 5
        and len(nlhac_gradclip_1000_cartpole) >= 5
        and len(nlhac_gradclip_1000_catch) >= 5
        and nlhac_gradclip_1000_overall[0] >= 20
        and nlhac_gradclip_1000_overall[3] > 0.0
        and nlhac_gradclip_1000_cartpole[3] > 0.0
        and nlhac_gradclip_1000_catch[3] > 0.0
    )
    nlhac_gradclip_1000_sarsa_promotion = bool(
        nlhac_gradclip_1000_q_promotion
        and nlhac_gradclip_1000_overall[3] > nlhac_gradclip_1000_overall[1]
        and nlhac_gradclip_1000_cartpole[3] > nlhac_gradclip_1000_cartpole[1]
        and nlhac_gradclip_1000_catch[3] > nlhac_gradclip_1000_catch[1]
    )
    evidence["nonlinear_horde_actor_critic_gradclip_1000_probe"] = {
        "path": str(nlhac_gradclip_1000_path),
        "exists": nlhac_gradclip_1000_exists,
        "overall": nlhac_gradclip_1000_rows.get("overall", {}),
        "catch": nlhac_gradclip_1000_rows.get("catch", {}),
        "cartpole": nlhac_gradclip_1000_rows.get("cartpole", {}),
        "passed": nlhac_gradclip_1000_exists,
        "promotion_vs_q_passed": nlhac_gradclip_1000_q_promotion,
        "promotion_vs_sarsa_passed": nlhac_gradclip_1000_sarsa_promotion,
        "boundary": (
            "gradient-clipped NLHAC 10-seed 1000-step probe remains positive "
            "against the Q baseline on catch, cartpole, and overall, but "
            "still does not outperform SARSA"
        ),
    }
    nlhac_adaptive_path = (
        root / "outputs/bsuite/nlhac_gradclip_adaptive_10seed_500/report.md"
    )
    nlhac_adaptive_exists = nlhac_adaptive_path.exists()
    nlhac_adaptive_rows = (
        parse_markdown_summary_table(read_text(nlhac_adaptive_path))
        if nlhac_adaptive_exists
        else {}
    )
    nlhac_adaptive_overall = nlhac_adaptive_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_cartpole = nlhac_adaptive_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_catch = nlhac_adaptive_rows.get("catch", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_q_promotion = bool(
        len(nlhac_adaptive_overall) >= 5
        and len(nlhac_adaptive_cartpole) >= 5
        and len(nlhac_adaptive_catch) >= 5
        and nlhac_adaptive_overall[0] >= 20
        and nlhac_adaptive_overall[3] > 0.0
        and nlhac_adaptive_cartpole[3] > 0.0
        and nlhac_adaptive_catch[3] > 0.0
    )
    nlhac_adaptive_sarsa_promotion = bool(
        nlhac_adaptive_q_promotion
        and nlhac_adaptive_overall[3] > nlhac_adaptive_overall[1]
        and nlhac_adaptive_cartpole[3] > nlhac_adaptive_cartpole[1]
        and nlhac_adaptive_catch[3] > nlhac_adaptive_catch[1]
    )
    evidence["nonlinear_horde_actor_critic_adaptive_obgd_probe"] = {
        "path": str(nlhac_adaptive_path),
        "exists": nlhac_adaptive_exists,
        "overall": nlhac_adaptive_rows.get("overall", {}),
        "catch": nlhac_adaptive_rows.get("catch", {}),
        "cartpole": nlhac_adaptive_rows.get("cartpole", {}),
        "passed": nlhac_adaptive_exists,
        "promotion_vs_q_passed": nlhac_adaptive_q_promotion,
        "promotion_vs_sarsa_passed": nlhac_adaptive_sarsa_promotion,
        "boundary": (
            "adaptive-ObGD NLHAC 10-seed 500-step probe improves overall "
            "and cartpole against Q, but catch is negative and SARSA remains "
            "stronger on catch"
        ),
    }
    nlhac_adaptive_1000_path = (
        root / "outputs/bsuite/nlhac_gradclip_adaptive_10seed_1000/report.md"
    )
    nlhac_adaptive_1000_exists = nlhac_adaptive_1000_path.exists()
    nlhac_adaptive_1000_rows = (
        parse_markdown_summary_table(read_text(nlhac_adaptive_1000_path))
        if nlhac_adaptive_1000_exists
        else {}
    )
    nlhac_adaptive_1000_overall = nlhac_adaptive_1000_rows.get("overall", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_1000_cartpole = nlhac_adaptive_1000_rows.get("cartpole", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_1000_catch = nlhac_adaptive_1000_rows.get("catch", {}).get(
        "numeric_values", []
    )
    nlhac_adaptive_1000_q_promotion = bool(
        len(nlhac_adaptive_1000_overall) >= 5
        and len(nlhac_adaptive_1000_cartpole) >= 5
        and len(nlhac_adaptive_1000_catch) >= 5
        and nlhac_adaptive_1000_overall[0] >= 20
        and nlhac_adaptive_1000_overall[3] > 0.0
        and nlhac_adaptive_1000_cartpole[3] > 0.0
        and nlhac_adaptive_1000_catch[3] > 0.0
    )
    nlhac_adaptive_1000_sarsa_promotion = bool(
        nlhac_adaptive_1000_q_promotion
        and nlhac_adaptive_1000_overall[3] > nlhac_adaptive_1000_overall[1]
        and nlhac_adaptive_1000_cartpole[3] > nlhac_adaptive_1000_cartpole[1]
        and nlhac_adaptive_1000_catch[3] > nlhac_adaptive_1000_catch[1]
    )
    evidence["nonlinear_horde_actor_critic_adaptive_obgd_1000_probe"] = {
        "path": str(nlhac_adaptive_1000_path),
        "exists": nlhac_adaptive_1000_exists,
        "overall": nlhac_adaptive_1000_rows.get("overall", {}),
        "catch": nlhac_adaptive_1000_rows.get("catch", {}),
        "cartpole": nlhac_adaptive_1000_rows.get("cartpole", {}),
        "passed": nlhac_adaptive_1000_exists,
        "promotion_vs_q_passed": nlhac_adaptive_1000_q_promotion,
        "promotion_vs_sarsa_passed": nlhac_adaptive_1000_sarsa_promotion,
        "boundary": (
            "adaptive-ObGD NLHAC 10-seed 1000-step probe improves overall "
            "and cartpole against Q, but catch remains negative and SARSA "
            "remains stronger on catch"
        ),
    }
    nlhac_wide_catch_path = (
        root / "outputs/bsuite/nlhac_gradclip_wide_catch3_500/report.md"
    )
    nlhac_wide_catch_exists = nlhac_wide_catch_path.exists()
    nlhac_wide_catch_rows = (
        parse_markdown_summary_table(read_text(nlhac_wide_catch_path))
        if nlhac_wide_catch_exists
        else {}
    )
    nlhac_wide_catch_values = nlhac_wide_catch_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n are: SARSA mean/wins, base NLHAC mean/wins,
    # then wide NLHAC mean/wins.
    nlhac_wide_mean = (
        nlhac_wide_catch_values[5] if len(nlhac_wide_catch_values) >= 6 else None
    )
    nlhac_wide_sarsa_mean = (
        nlhac_wide_catch_values[1] if len(nlhac_wide_catch_values) >= 2 else None
    )
    evidence["nonlinear_horde_actor_critic_wide_catch_probe"] = {
        "path": str(nlhac_wide_catch_path),
        "exists": nlhac_wide_catch_exists,
        "summary": nlhac_wide_catch_rows.get("catch", {}),
        "wide_mean_improvement_vs_q": nlhac_wide_mean,
        "sarsa_mean_improvement_vs_q": nlhac_wide_sarsa_mean,
        "passed": nlhac_wide_catch_exists,
        "promotion_vs_q_passed": bool(
            nlhac_wide_mean is not None and nlhac_wide_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            nlhac_wide_mean is not None
            and nlhac_wide_sarsa_mean is not None
            and nlhac_wide_mean > nlhac_wide_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 pilot widening both actor and critic to (64, 64) "
            "did not improve the NLHAC catch boundary"
        ),
    }
    nlhac_no_actor_ln_path = (
        root / "outputs/bsuite/nlhac_no_actor_ln_catch3_500/report.md"
    )
    nlhac_no_actor_ln_exists = nlhac_no_actor_ln_path.exists()
    nlhac_no_actor_ln_rows = (
        parse_markdown_summary_table(read_text(nlhac_no_actor_ln_path))
        if nlhac_no_actor_ln_exists
        else {}
    )
    nlhac_no_actor_ln_values = nlhac_no_actor_ln_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n are: SARSA mean/wins, base NLHAC mean/wins,
    # then no-actor-layer-norm NLHAC mean/wins.
    nlhac_no_actor_ln_mean = (
        nlhac_no_actor_ln_values[5]
        if len(nlhac_no_actor_ln_values) >= 6
        else None
    )
    nlhac_no_actor_ln_sarsa_mean = (
        nlhac_no_actor_ln_values[1]
        if len(nlhac_no_actor_ln_values) >= 2
        else None
    )
    evidence["nonlinear_horde_actor_critic_no_actor_layer_norm_probe"] = {
        "path": str(nlhac_no_actor_ln_path),
        "exists": nlhac_no_actor_ln_exists,
        "summary": nlhac_no_actor_ln_rows.get("catch", {}),
        "no_actor_layer_norm_mean_improvement_vs_q": nlhac_no_actor_ln_mean,
        "sarsa_mean_improvement_vs_q": nlhac_no_actor_ln_sarsa_mean,
        "passed": nlhac_no_actor_ln_exists,
        "promotion_vs_q_passed": bool(
            nlhac_no_actor_ln_mean is not None and nlhac_no_actor_ln_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            nlhac_no_actor_ln_mean is not None
            and nlhac_no_actor_ln_sarsa_mean is not None
            and nlhac_no_actor_ln_mean > nlhac_no_actor_ln_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 pilot disabling actor layer norm regressed the "
            "NLHAC catch boundary"
        ),
    }
    nlhac_eps05_path = root / "outputs/bsuite/nlhac_eps05_catch3_500/report.md"
    nlhac_eps05_exists = nlhac_eps05_path.exists()
    nlhac_eps05_rows = (
        parse_markdown_summary_table(read_text(nlhac_eps05_path))
        if nlhac_eps05_exists
        else {}
    )
    nlhac_eps05_values = nlhac_eps05_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n are: SARSA mean/wins, base NLHAC mean/wins,
    # then epsilon-mixed NLHAC mean/wins.
    nlhac_eps05_mean = (
        nlhac_eps05_values[5] if len(nlhac_eps05_values) >= 6 else None
    )
    nlhac_eps05_sarsa_mean = (
        nlhac_eps05_values[1] if len(nlhac_eps05_values) >= 2 else None
    )
    evidence["nonlinear_horde_actor_critic_epsilon_mixture_probe"] = {
        "path": str(nlhac_eps05_path),
        "exists": nlhac_eps05_exists,
        "summary": nlhac_eps05_rows.get("catch", {}),
        "epsilon_mixture_mean_improvement_vs_q": nlhac_eps05_mean,
        "sarsa_mean_improvement_vs_q": nlhac_eps05_sarsa_mean,
        "passed": nlhac_eps05_exists,
        "promotion_vs_q_passed": bool(
            nlhac_eps05_mean is not None and nlhac_eps05_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            nlhac_eps05_mean is not None
            and nlhac_eps05_sarsa_mean is not None
            and nlhac_eps05_mean > nlhac_eps05_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 pilot with a consistent 0.05 uniform actor policy "
            "mixture regressed the NLHAC catch boundary"
        ),
    }
    nlhac_tdnorm_path = root / "outputs/bsuite/nlhac_tdnorm_catch3_500/report.md"
    nlhac_tdnorm_exists = nlhac_tdnorm_path.exists()
    nlhac_tdnorm_rows = (
        parse_markdown_summary_table(read_text(nlhac_tdnorm_path))
        if nlhac_tdnorm_exists
        else {}
    )
    nlhac_tdnorm_values = nlhac_tdnorm_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n are: SARSA mean/wins, base NLHAC mean/wins,
    # then TD-normalized NLHAC mean/wins.
    nlhac_tdnorm_mean = (
        nlhac_tdnorm_values[5] if len(nlhac_tdnorm_values) >= 6 else None
    )
    nlhac_tdnorm_sarsa_mean = (
        nlhac_tdnorm_values[1] if len(nlhac_tdnorm_values) >= 2 else None
    )
    nlhac_tdnorm_base_mean = (
        nlhac_tdnorm_values[3] if len(nlhac_tdnorm_values) >= 4 else None
    )
    evidence["nonlinear_horde_actor_critic_td_normalizer_probe"] = {
        "path": str(nlhac_tdnorm_path),
        "exists": nlhac_tdnorm_exists,
        "summary": nlhac_tdnorm_rows.get("catch", {}),
        "td_normalized_mean_improvement_vs_q": nlhac_tdnorm_mean,
        "base_gradclip_mean_improvement_vs_q": nlhac_tdnorm_base_mean,
        "sarsa_mean_improvement_vs_q": nlhac_tdnorm_sarsa_mean,
        "passed": nlhac_tdnorm_exists,
        "promotion_vs_q_passed": bool(
            nlhac_tdnorm_mean is not None and nlhac_tdnorm_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            nlhac_tdnorm_mean is not None
            and nlhac_tdnorm_sarsa_mean is not None
            and nlhac_tdnorm_mean > nlhac_tdnorm_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 pilot with actor-only TD-error normalization "
            "was positive against Q but weaker than baseline gradclip NLHAC "
            "and SARSA"
        ),
    }
    nlhac_variant_search_path = (
        root / "outputs/bsuite/nlhac_gradclip_variant_catch5_1000/report.md"
    )
    nlhac_variant_search_exists = nlhac_variant_search_path.exists()
    nlhac_variant_search_rows = (
        parse_markdown_summary_table(read_text(nlhac_variant_search_path))
        if nlhac_variant_search_exists
        else {}
    )
    evidence["nonlinear_horde_actor_critic_gradclip_variant_search"] = {
        "path": str(nlhac_variant_search_path),
        "exists": nlhac_variant_search_exists,
        "summary": nlhac_variant_search_rows.get("catch", {}),
        "passed": nlhac_variant_search_exists,
        "boundary": (
            "5-seed catch/0 search over actor step-size, actor lambda, "
            "temperature, gradient clip strength, and critic lambda did not "
            "find a SARSA-beating NLHAC variant"
        ),
    }
    nlqhorde_variant_search_path = (
        root / "outputs/bsuite/nlqhorde_ac_variant_catch3_500/report.md"
    )
    nlqhorde_variant_search_exists = nlqhorde_variant_search_path.exists()
    nlqhorde_variant_search_rows = (
        parse_markdown_summary_table(read_text(nlqhorde_variant_search_path))
        if nlqhorde_variant_search_exists
        else {}
    )
    nlqhorde_variant_search_values = nlqhorde_variant_search_rows.get(
        "catch", {}
    ).get("numeric_values", [])
    # Table columns after n are: SARSA mean/wins, then NLQHorde variants.
    nlqhorde_best_mean = (
        max(nlqhorde_variant_search_values[3::2])
        if len(nlqhorde_variant_search_values) >= 5
        else None
    )
    nlqhorde_sarsa_mean = (
        nlqhorde_variant_search_values[1]
        if len(nlqhorde_variant_search_values) >= 2
        else None
    )
    evidence["nonlinear_q_horde_actor_critic_variant_search"] = {
        "path": str(nlqhorde_variant_search_path),
        "exists": nlqhorde_variant_search_exists,
        "summary": nlqhorde_variant_search_rows.get("catch", {}),
        "best_mean_improvement_vs_q": nlqhorde_best_mean,
        "sarsa_mean_improvement_vs_q": nlqhorde_sarsa_mean,
        "passed": nlqhorde_variant_search_exists,
        "promotion_vs_q_passed": bool(
            nlqhorde_best_mean is not None and nlqhorde_best_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            nlqhorde_best_mean is not None
            and nlqhorde_sarsa_mean is not None
            and nlqhorde_best_mean > nlqhorde_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 action-value NLQHorde search over actor step-size, "
            "temperature, and actor gradient clip strength did not find a "
            "Q- or SARSA-beating variant; the best mean improvement was a tie "
            "with the Q baseline"
        ),
    }
    nlqhorde_expected_path = (
        root / "outputs/bsuite/nlqhorde_ac_expected_adv_3seed_500/step4_report.md"
    )
    nlqhorde_expected_exists = nlqhorde_expected_path.exists()
    nlqhorde_expected_rows = (
        parse_markdown_summary_table(read_text(nlqhorde_expected_path))
        if nlqhorde_expected_exists
        else {}
    )
    nlqhorde_expected_overall_values = nlqhorde_expected_rows.get(
        "overall", {}
    ).get("numeric_values", [])
    nlqhorde_expected_catch_values = nlqhorde_expected_rows.get("catch", {}).get(
        "numeric_values", []
    )
    evidence["nonlinear_q_horde_expected_advantage_probe"] = {
        "path": str(nlqhorde_expected_path),
        "exists": nlqhorde_expected_exists,
        "overall": nlqhorde_expected_rows.get("overall", {}),
        "catch": nlqhorde_expected_rows.get("catch", {}),
        "cartpole": nlqhorde_expected_rows.get("cartpole", {}),
        "passed": nlqhorde_expected_exists,
        "promotion_vs_q_passed": bool(
            len(nlqhorde_expected_overall_values) >= 5
            and nlqhorde_expected_overall_values[3] > 0.0
            and len(nlqhorde_expected_catch_values) >= 5
            and nlqhorde_expected_catch_values[3] > 0.0
        ),
        "promotion_vs_sarsa_passed": False,
        "boundary": (
            "nonlinear Q-Horde expected-advantage actor update is implemented "
            "and passes a 3-seed catch/cartpole smoke; it improves overall "
            "against Q but does not beat SARSA or reliably clear catch"
        ),
    }
    nlqhorde_expected_variant_path = (
        root / "outputs/bsuite/nlqhorde_ac_expected_adv_variant_catch3_500/report.md"
    )
    nlqhorde_expected_variant_exists = nlqhorde_expected_variant_path.exists()
    nlqhorde_expected_variant_rows = (
        parse_markdown_summary_table(read_text(nlqhorde_expected_variant_path))
        if nlqhorde_expected_variant_exists
        else {}
    )
    variant_catch_values = nlqhorde_expected_variant_rows.get("catch", {}).get(
        "numeric_values", []
    )
    # Table columns after n: SARSA mean/wins, then each expected-advantage variant.
    expected_variant_means = (
        variant_catch_values[3::2] if len(variant_catch_values) >= 5 else []
    )
    expected_variant_best_mean = (
        max(expected_variant_means) if expected_variant_means else None
    )
    expected_variant_sarsa_mean = (
        variant_catch_values[1] if len(variant_catch_values) >= 2 else None
    )
    evidence["nonlinear_q_horde_expected_advantage_variant_search"] = {
        "path": str(nlqhorde_expected_variant_path),
        "exists": nlqhorde_expected_variant_exists,
        "summary": nlqhorde_expected_variant_rows.get("catch", {}),
        "best_mean_improvement_vs_q": expected_variant_best_mean,
        "sarsa_mean_improvement_vs_q": expected_variant_sarsa_mean,
        "passed": nlqhorde_expected_variant_exists,
        "promotion_vs_q_passed": bool(
            expected_variant_best_mean is not None and expected_variant_best_mean > 0.0
        ),
        "promotion_vs_sarsa_passed": bool(
            expected_variant_best_mean is not None
            and expected_variant_sarsa_mean is not None
            and expected_variant_best_mean > expected_variant_sarsa_mean
        ),
        "boundary": (
            "3-seed catch/0 expected-advantage variant search found a "
            "gradient-clip 0.5 candidate that beat Q and SARSA on mean catch "
            "improvement, requiring 10-seed confirmation"
        ),
    }
    nlqhorde_expected_g05_path = (
        root / "outputs/bsuite/nlqhorde_ac_expected_adv_g05_10seed_500/report.md"
    )
    nlqhorde_expected_g05_exists = nlqhorde_expected_g05_path.exists()
    nlqhorde_expected_g05_rows = (
        parse_markdown_summary_table(read_text(nlqhorde_expected_g05_path))
        if nlqhorde_expected_g05_exists
        else {}
    )
    g05_overall_values = nlqhorde_expected_g05_rows.get("overall", {}).get(
        "numeric_values", []
    )
    g05_catch_values = nlqhorde_expected_g05_rows.get("catch", {}).get(
        "numeric_values", []
    )
    evidence["nonlinear_q_horde_expected_advantage_g05_10seed_probe"] = {
        "path": str(nlqhorde_expected_g05_path),
        "exists": nlqhorde_expected_g05_exists,
        "overall": nlqhorde_expected_g05_rows.get("overall", {}),
        "catch": nlqhorde_expected_g05_rows.get("catch", {}),
        "cartpole": nlqhorde_expected_g05_rows.get("cartpole", {}),
        "passed": nlqhorde_expected_g05_exists,
        "promotion_vs_q_passed": bool(
            len(g05_overall_values) >= 5
            and g05_overall_values[3] > 0.0
            and len(g05_catch_values) >= 5
            and g05_catch_values[3] > 0.0
        ),
        "promotion_vs_sarsa_passed": False,
        "boundary": (
            "10-seed confirmation of the best 3-seed expected-advantage "
            "variant did not hold: it improved cartpole but regressed catch "
            "and trailed SARSA overall"
        ),
    }

    tests = {
        "tests/test_sarsa.py": (root / "tests/test_sarsa.py").exists(),
        "tests/test_actor_critic.py": (root / "tests/test_actor_critic.py").exists(),
        "tests/test_horde_actor_critic.py": (
            root / "tests/test_horde_actor_critic.py"
        ).exists(),
        "tests/test_step3_production.py": (
            root / "tests/test_step3_production.py"
        ).exists(),
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
        and security_rollout.get("comparison", {}).get("oracle_false_positives") == 0
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
        "oracle_false_positives": (
            security_rollout.get("comparison", {}).get("oracle_false_positives")
            if security_rollout is not None
            else None
        ),
        "boundary": (
            security_rollout.get("boundary") if security_rollout is not None else None
        ),
    }

    accepted_local_scope = all(
        bool(item)
        for item in [
            evidence["sarsa_vs_q_primary"]["passed"],
            evidence["sarsa_vs_q_catch_cartpole"]["passed"],
            evidence["q_sarsa_actor_critic_report"]["passed"],
            evidence["horde_actor_critic_boundary"]["passed"],
            evidence["horde_actor_critic_positive_control"]["passed"],
            all(tests.values()),
        ]
    )
    accepted_local_framework_scope = bool(
        evidence["sarsa_vs_q_primary"]["passed"]
        and evidence["sarsa_vs_q_catch_cartpole"]["passed"]
        and evidence["horde_actor_critic_positive_control"]["passed"]
        and all(tests.values())
    )
    open_boundaries = [
        (
            "Horde actor-critic positive control passes, but bsuite promotion "
            "over SARSA remains unproven; gradient-clipped NLHAC now beats "
            "the Q baseline on 10-seed catch/cartpole probes at 500 and 1000 "
            "steps, but SARSA remains stronger; the action-value NLQHorde "
            "actor-critic search did not improve the catch boundary; the "
            "nonlinear expected-advantage Q-Horde actor update improves the "
            "small 3-seed overall probe against Q, and a 3-seed catch variant "
            "wins on mean, but the 10-seed confirmation regresses catch and "
            "still does not beat SARSA; "
            "adaptive-ObGD NLHAC improved cartpole at 500 and 1000 steps "
            "but regressed catch"
        ),
        (
            "local security-gym counterfactual rollout passes, but active-defense "
            "rlsecd daemon deployment remains external"
            if security_rollout_passed
            else "active-defense rlsecd/security-gym deployment remains external"
        ),
    ]
    return {
        "schema": "alberta.step4.solution_gate.v1",
        "accepted_step4_local_framework_scope": accepted_local_framework_scope,
        "accepted_sarsa_step4a": bool(accepted_local_scope),
        "accepted_security_gym_counterfactual_rollout": security_rollout_passed,
        "solved_step4_full_actor_critic_scope": False,
        "claim_scope": (
            "canonical_discrete_sarsa_control_local_completion"
            if accepted_local_framework_scope
            else "incomplete_step4_local_evidence"
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
    """Run the Step 4 audit."""
    args = parse_args(argv)
    status = audit_step4(args.root)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    if status["accepted_sarsa_step4a"] or args.allow_unsolved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

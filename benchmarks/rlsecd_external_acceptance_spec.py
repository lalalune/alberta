#!/usr/bin/env python3
"""Generate acceptance criteria for unavailable rlsecd external work.

The remaining Alberta Plan TODOs in this checkout are deliberately external:
they require the ``rlsecd`` and ``chronos-sec`` repositories plus daemon logs.
This script makes those blockers executable by writing a machine-readable
contract for the evidence that must be produced in the sibling repositories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("outputs/rlsecd_external_acceptance/spec.json")
DEFAULT_MARKDOWN = Path("outputs/rlsecd_external_acceptance/spec.md")


@dataclass(frozen=True)
class ExternalAcceptanceItem:
    """Acceptance contract for one external TODO item."""

    todo_line_hint: int
    todo_text: str
    claim_scope: str
    required_repositories: tuple[str, ...]
    command_template: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    required_metrics: tuple[str, ...]
    pass_conditions: tuple[str, ...]


ACCEPTANCE_ITEMS: tuple[ExternalAcceptanceItem, ...] = (
    ExternalAcceptanceItem(
        todo_line_hint=131,
        todo_text=(
            "External: rlsecd `--gym-control` mode: existing 5 prediction "
            "demons + SARSA control demon"
        ),
        claim_scope="rlsecd_gym_control_horde_sarsa_daemon",
        required_repositories=("rlsecd", "chronos-sec", "security-gym"),
        command_template=(
            "python",
            "-m",
            "rlsecd.daemon",
            "--gym-control",
            "--prediction-demons",
            "5",
            "--control-agent",
            "sarsa",
            "--action-heads",
            "6",
            "--rollout-log",
            "outputs/rlsecd_gym_control/rollouts.jsonl",
            "--metrics-out",
            "outputs/rlsecd_gym_control/metrics.json",
        ),
        required_artifacts=(
            "outputs/rlsecd_gym_control/rollouts.jsonl",
            "outputs/rlsecd_gym_control/metrics.json",
            "outputs/rlsecd_gym_control/config.json",
        ),
        required_metrics=(
            "n_prediction_demons",
            "n_control_actions",
            "n_transitions",
            "mean_reward",
            "sarsa_td_error_final_window",
            "uses_framework_sarsa_agent",
            "uses_framework_horde_learner",
        ),
        pass_conditions=(
            "n_prediction_demons == 5",
            "n_control_actions == 6",
            "n_transitions > 0 and equals the rollout JSONL row count",
            "config.json declares SARSA control, 5 prediction demons, 6 actions, "
            "temporal uniformity, framework SARSAAgent/HordeLearner usage, and "
            "the security-gym action vocabulary",
            "all rollout records preserve temporal order with unique step ids",
            "all rollout records include state, action, reward, next_state, "
            "termination, and policy metadata with finite numeric state/reward values",
            "all action ids are valid security-gym action heads",
            "mean_reward and sarsa_td_error_final_window are finite scalar metrics",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=134,
        todo_text=(
            "External: rlsecd end-to-end throughput must include parsing, "
            "feature extraction, learner update, checkpoint/reporting, and "
            "action dispatch"
        ),
        claim_scope="rlsecd_end_to_end_daemon_throughput",
        required_repositories=("rlsecd", "chronos-sec"),
        command_template=(
            "python",
            "-m",
            "rlsecd.benchmarks.throughput",
            "--include-parsing",
            "--include-feature-extraction",
            "--include-learner-update",
            "--include-checkpoint-reporting",
            "--include-action-dispatch",
            "--metrics-out",
            "outputs/rlsecd_throughput/metrics.json",
        ),
        required_artifacts=("outputs/rlsecd_throughput/metrics.json",),
        required_metrics=(
            "n_events",
            "events_per_second",
            "wall_clock_s",
            "parse_ms_p50",
            "feature_ms_p50",
            "learner_update_ms_p50",
            "checkpoint_reporting_ms_p50",
            "action_dispatch_ms_p50",
            "stage_event_counts",
            "measured_real_daemon_path",
        ),
        pass_conditions=(
            "n_events > 0",
            "events_per_second > 0",
            "wall_clock_s > 0",
            "events_per_second agrees with n_events / wall_clock_s within 5%",
            "all five pipeline stage timings are present",
            "stage_event_counts records all measured events for parsing, "
            "feature extraction, learner update, checkpoint/reporting, and "
            "action dispatch",
            "measurement wraps the real daemon event path",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=137,
        todo_text=(
            "External: generate `(state, action, reward, outcome)` experience "
            "for autoresearch LLM oracle pipeline from rlsecd/security-gym "
            "rollouts"
        ),
        claim_scope="rlsecd_oracle_experience_export",
        required_repositories=("rlsecd", "security-gym"),
        command_template=(
            "python",
            "-m",
            "rlsecd.exports.oracle_experience",
            "--rollout-log",
            "outputs/rlsecd_gym_control/rollouts.jsonl",
            "--records-out",
            "outputs/rlsecd_oracle_experience/records.jsonl",
            "--manifest-out",
            "outputs/rlsecd_oracle_experience/manifest.json",
        ),
        required_artifacts=(
            "outputs/rlsecd_oracle_experience/records.jsonl",
            "outputs/rlsecd_oracle_experience/manifest.json",
        ),
        required_metrics=(
            "n_records",
            "schema",
            "source_rollout_log",
            "source_rollout_record_count",
            "exported_from_production_rollout",
        ),
        pass_conditions=(
            "n_records > 0",
            "schema == rlsecd.oracle_experience.v1",
            "each record includes state, action, reward, outcome, "
            "source_rollout_step, and policy metadata",
            "oracle records are ordered by unique source rollout step ids",
            "manifest names an existing source rlsecd/security-gym rollout log",
            "manifest proves records were exported from a production rollout",
            "manifest source rollout record count matches the source JSONL",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=287,
        todo_text="External: AF-2 IDBD-MLP 100k-event replay test in rlsecd",
        claim_scope="rlsecd_idbd_mlp_100k_replay",
        required_repositories=("rlsecd",),
        command_template=(
            "python",
            "-m",
            "rlsecd.replay",
            "--learner",
            "idbd_mlp",
            "--events",
            "data/replay_100k.jsonl",
            "--checkpoint-out",
            "outputs/idbd_mlp_100k/checkpoint",
            "--metrics-out",
            "outputs/idbd_mlp_100k/metrics.json",
        ),
        required_artifacts=(
            "outputs/idbd_mlp_100k/checkpoint",
            "outputs/idbd_mlp_100k/metrics.json",
        ),
        required_metrics=(
            "n_events",
            "final_window_loss",
            "all_finite",
            "finite_components",
            "mean_step_size",
            "validation_batch_size",
            "checkpoint_roundtrip_max_abs_diff",
        ),
        pass_conditions=(
            "n_events >= 100000",
            "all_finite is true",
            "finite_components marks predictions, parameters, traces, and step "
            "sizes as finite",
            "validation_batch_size > 0",
            "checkpoint roundtrip preserves predictions on a fixed validation batch",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=288,
        todo_text="External: AF-2 IDBD-MLP full 1.6M log stability test",
        claim_scope="rlsecd_idbd_mlp_full_log_stability",
        required_repositories=("rlsecd",),
        command_template=(
            "python",
            "-m",
            "rlsecd.replay",
            "--learner",
            "idbd_mlp",
            "--events",
            "data/full_1_6m.jsonl",
            "--resume-dir",
            "outputs/idbd_mlp_1_6m/checkpoints",
            "--metrics-out",
            "outputs/idbd_mlp_1_6m/metrics.json",
        ),
        required_artifacts=(
            "outputs/idbd_mlp_1_6m/checkpoints",
            "outputs/idbd_mlp_1_6m/metrics.json",
        ),
        required_metrics=(
            "n_events",
            "resumed_from_midpoint",
            "all_finite",
            "finite_components",
            "events_per_second",
            "max_rss_mb",
            "final_window_loss",
            "checkpoint_count",
            "resume_final_loss_abs_diff",
        ),
        pass_conditions=(
            "n_events >= 1600000",
            "all_finite is true through the final event",
            "finite_components marks predictions, parameters, traces, and step "
            "sizes as finite",
            "checkpoint_count >= 2",
            "midpoint checkpoint resume completes with equivalent final metrics",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=289,
        todo_text=(
            "External: simplify rlsecd SecurityAgent to use Orbax checkpoint "
            "utilities (format v2)"
        ),
        claim_scope="rlsecd_security_agent_orbax_checkpoint_v2",
        required_repositories=("rlsecd",),
        command_template=(
            "python",
            "-m",
            "rlsecd.tests.checkpoint_roundtrip",
            "--format",
            "v2",
            "--checkpoint-dir",
            "outputs/rlsecd_checkpoint_v2",
            "--metrics-out",
            "outputs/rlsecd_checkpoint_v2/metrics.json",
        ),
        required_artifacts=(
            "outputs/rlsecd_checkpoint_v2",
            "outputs/rlsecd_checkpoint_v2/metrics.json",
            "outputs/rlsecd_checkpoint_v2/metadata.json",
        ),
        required_metrics=(
            "format_version",
            "metadata_present",
            "learner_state_present",
            "optimizer_state_present",
            "normalizer_state_present",
            "restored_step_count_matches",
            "prediction_roundtrip_max_abs_diff",
        ),
        pass_conditions=(
            "format_version == 2",
            "metadata.json declares alberta.rlsecd.security_agent_checkpoint.v2",
            "metadata includes framework checkpoint schema fields",
            "restored learner, optimizer, normalizer, and step_count match the "
            "saved SecurityAgent state",
            "prediction_roundtrip_max_abs_diff <= 1e-6",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=290,
        todo_text=(
            "External: simplify rlsecd SecurityAgent to use framework config "
            "serialization"
        ),
        claim_scope="rlsecd_security_agent_framework_config_serialization",
        required_repositories=("rlsecd",),
        command_template=(
            "python",
            "-m",
            "rlsecd.tests.config_roundtrip",
            "--metrics-out",
            "outputs/rlsecd_config_roundtrip/metrics.json",
        ),
        required_artifacts=("outputs/rlsecd_config_roundtrip/metrics.json",),
        required_metrics=(
            "learner_config_roundtrip",
            "optimizer_config_roundtrip",
            "normalizer_config_roundtrip",
            "feature_schema_roundtrip",
            "security_agent_config_roundtrip",
            "serialized_component_types",
            "unknown_config_keys",
            "dropped_config_keys",
            "restored_schema_version_matches",
            "prediction_roundtrip_max_abs_diff",
        ),
        pass_conditions=(
            "all framework config dictionaries roundtrip without dropped fields",
            "serialized component type names are present for learner, optimizer, "
            "normalizer, and feature schema",
            "unknown_config_keys and dropped_config_keys are empty",
            "restored config schema version matches the saved schema version",
            "SecurityAgent reconstructed from config produces equivalent predictions",
        ),
    ),
    ExternalAcceptanceItem(
        todo_line_hint=291,
        todo_text=(
            "External: integrate `compute_feature_relevance` into rlsecd "
            "periodic reporting (60s interval)"
        ),
        claim_scope="rlsecd_feature_relevance_periodic_reporting",
        required_repositories=("rlsecd",),
        command_template=(
            "python",
            "-m",
            "rlsecd.daemon",
            "--feature-relevance-interval-s",
            "60",
            "--metrics-out",
            "outputs/rlsecd_feature_relevance/metrics.jsonl",
        ),
        required_artifacts=("outputs/rlsecd_feature_relevance/metrics.jsonl",),
        required_metrics=(
            "feature_relevance_report_count",
            "feature_relevance_interval_s",
            "report_timestamp_s",
            "top_feature_names",
            "top_feature_relevance_values",
            "report_nonblocking",
            "learner_updates_skipped_for_reporting",
            "uses_framework_compute_feature_relevance",
            "latest_report_latency_ms",
        ),
        pass_conditions=(
            "feature_relevance_interval_s == 60",
            "at least two timestamped reports are emitted",
            "report timestamps are ordered with approximately 60 seconds between reports",
            "every report contains non-empty feature names and matching finite relevance values",
            "reporting uses alberta_framework.core.diagnostics.compute_feature_relevance",
            "reporting does not block or skip learner updates",
        ),
    ),
)


def build_spec() -> dict[str, Any]:
    """Return the full external acceptance specification."""
    return {
        "schema": "alberta.rlsecd_external_acceptance_spec.v1",
        "claim_scope": "remaining_external_alberta_plan_work",
        "status": "pending_external_repositories",
        "items": [asdict(item) for item in ACCEPTANCE_ITEMS],
    }


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a compact human-readable acceptance checklist."""
    lines = [
        "# rlsecd External Acceptance Spec",
        "",
        "Status: pending external `rlsecd` / `chronos-sec` repositories.",
        "",
    ]
    for item in spec["items"]:
        lines.extend(
            [
                f"## {item['claim_scope']}",
                "",
                f"TODO: {item['todo_text']}",
                "",
                "Required repositories: "
                + ", ".join(f"`{repo}`" for repo in item["required_repositories"]),
                "",
                "Command:",
                "",
                "```bash",
                " ".join(item["command_template"]),
                "```",
                "",
                "Required artifacts:",
            ]
        )
        lines.extend(f"- `{artifact}`" for artifact in item["required_artifacts"])
        lines.extend(["", "Pass conditions:"])
        lines.extend(f"- {condition}" for condition in item["pass_conditions"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    """Write the acceptance specification artifacts."""
    args = parse_args()
    spec = build_spec()
    rendered = json.dumps(spec, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(spec), encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate rlsecd external acceptance artifacts against the local spec."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_SPEC = Path("outputs/rlsecd_external_acceptance/spec.json")
DEFAULT_ARTIFACT_ROOT = Path(".")
DEFAULT_STATUS = Path("outputs/rlsecd_external_acceptance/status.json")
SECURITY_GYM_ACTION_NAMES = (
    "pass",
    "alert",
    "throttle",
    "block_source",
    "unblock",
    "isolate",
)
ORACLE_EXPERIENCE_SCHEMA = "rlsecd.oracle_experience.v1"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.is_dir():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_jsonl(path: Path, limit: int | None = 100) -> list[dict[str, Any]]:
    if not path.exists() or path.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contained a non-object JSONL row")
        rows.append(payload)
    return rows


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _numeric_gt(metrics: dict[str, Any], key: str, threshold: float) -> bool:
    value = metrics.get(key)
    return _finite_number(value) and value > threshold


def _numeric_gte(metrics: dict[str, Any], key: str, threshold: float) -> bool:
    value = metrics.get(key)
    return _finite_number(value) and value >= threshold


def _numeric_lte(metrics: dict[str, Any], key: str, threshold: float) -> bool:
    value = metrics.get(key)
    return _finite_number(value) and value <= threshold


def _all_finite_flag(metrics: dict[str, Any]) -> bool:
    return metrics.get("all_finite") is True


def _finite_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(_finite_number(item) for item in value)


def _validate_rollout_rows(root: Path) -> tuple[bool, str]:
    rows = _load_jsonl(root / "outputs/rlsecd_gym_control/rollouts.jsonl")
    if not rows:
        return False, "rollouts.jsonl missing or empty"
    required = {"state", "action", "reward", "next_state", "terminated", "policy_metadata"}
    missing_required = [
        idx for idx, row in enumerate(rows) if not required.issubset(row)
    ]
    if missing_required:
        return False, "rollout rows are missing transition fields"
    malformed_transition = [
        idx
        for idx, row in enumerate(rows)
        if (
            not _finite_number_list(row.get("state"))
            or not _finite_number_list(row.get("next_state"))
            or len(row["state"]) != len(row["next_state"])
            or not _finite_number(row.get("reward"))
            or not isinstance(row.get("terminated"), bool)
            or not isinstance(row.get("policy_metadata"), dict)
        )
    ]
    if malformed_transition:
        return False, "rollout rows have malformed transition payloads"
    invalid_actions = [
        row.get("action")
        for row in rows
        if (
            not isinstance(row.get("action"), int)
            or isinstance(row.get("action"), bool)
            or not 0 <= row["action"] < 6
        )
    ]
    if invalid_actions:
        return False, "rollout rows contain invalid security-gym action ids"
    steps = [row.get("step") for row in rows if "step" in row]
    if steps:
        if any(not isinstance(step, int) or isinstance(step, bool) for step in steps):
            return False, "rollout step ids are not integers"
        if steps != sorted(steps):
            return False, "rollout rows are not temporally ordered by step"
        if len(steps) != len(set(steps)):
            return False, "rollout rows contain duplicate step ids"
    return True, "rollout rows preserve order and valid actions"


def _validate_gym_control_config(root: Path) -> tuple[bool, str]:
    config = _load_json(root / "outputs/rlsecd_gym_control/config.json")
    if not config:
        return False, "gym-control config.json missing or empty"
    if config.get("control_agent") != "sarsa":
        return False, "gym-control config does not declare SARSA control"
    if config.get("n_prediction_demons") != 5:
        return False, "gym-control config does not declare five prediction demons"
    if config.get("n_control_actions") != 6:
        return False, "gym-control config does not declare six control actions"
    if tuple(config.get("action_names", ())) != SECURITY_GYM_ACTION_NAMES:
        return False, "gym-control config action_names do not match security-gym"
    if config.get("temporal_uniformity") is not True:
        return False, "gym-control config does not assert temporal uniformity"
    if config.get("uses_framework_sarsa_agent") is not True:
        return False, "gym-control config does not prove framework SARSAAgent usage"
    if config.get("uses_framework_horde_learner") is not True:
        return False, "gym-control config does not prove framework HordeLearner usage"
    return True, "gym-control config declares SARSA and security-gym action contract"


def _validate_oracle_rows(root: Path) -> tuple[bool, str]:
    records_path = root / "outputs/rlsecd_oracle_experience/records.jsonl"
    rows = _load_jsonl(records_path)
    if not rows:
        return False, "oracle experience JSONL missing or empty"
    required = {
        "state",
        "action",
        "reward",
        "outcome",
        "source_rollout_step",
        "policy_metadata",
    }
    if any(not required.issubset(row) for row in rows):
        return False, "oracle experience rows lack required rollout fields"
    malformed_rows = [
        idx
        for idx, row in enumerate(rows)
        if (
            not isinstance(row.get("state"), list)
            or not (
                isinstance(row.get("action"), int)
                and not isinstance(row.get("action"), bool)
                and 0 <= row["action"] < len(SECURITY_GYM_ACTION_NAMES)
            )
            or not _finite_number(row.get("reward"))
            or not isinstance(row.get("outcome"), str)
            or not row.get("outcome")
            or not (
                isinstance(row.get("source_rollout_step"), int)
                and not isinstance(row.get("source_rollout_step"), bool)
            )
            or not isinstance(row.get("policy_metadata"), dict)
        )
    ]
    if malformed_rows:
        return False, "oracle experience rows have malformed payloads"
    source_steps = [row["source_rollout_step"] for row in rows]
    if source_steps != sorted(source_steps):
        return False, "oracle experience rows are not ordered by source rollout step"
    if len(source_steps) != len(set(source_steps)):
        return False, "oracle experience rows duplicate source rollout steps"
    manifest = _load_json(root / "outputs/rlsecd_oracle_experience/manifest.json")
    if not manifest or not manifest.get("source_rollout_log"):
        return False, "oracle manifest lacks source_rollout_log"
    if manifest.get("schema") != ORACLE_EXPERIENCE_SCHEMA:
        return False, "oracle manifest schema is not rlsecd.oracle_experience.v1"
    if manifest.get("exported_from_production_rollout") is not True:
        return False, "oracle manifest does not prove production rollout provenance"
    source_rollout_log = root / str(manifest["source_rollout_log"])
    if not source_rollout_log.exists():
        return False, "oracle manifest source_rollout_log does not exist"
    source_rows = _load_jsonl(source_rollout_log, limit=None)
    if manifest.get("source_rollout_record_count") != len(source_rows):
        return False, "oracle source rollout record count does not match"
    full_records = _load_jsonl(records_path, limit=None)
    if manifest.get("n_records") != len(full_records):
        return False, "oracle manifest n_records does not match records JSONL"
    if not _numeric_gt(manifest, "n_records", 0):
        return False, "oracle manifest n_records is not positive"
    return True, "oracle records and manifest match required schema"


def _checkpoint_path_exists(root: Path, relative_path: str) -> bool:
    path = root / relative_path
    return path.exists()


def _validate_feature_relevance(root: Path, metrics: dict[str, Any]) -> tuple[bool, str]:
    rows = _load_jsonl(root / "outputs/rlsecd_feature_relevance/metrics.jsonl")
    if len(rows) < 2:
        return False, "fewer than two feature relevance reports were emitted"
    if metrics.get("feature_relevance_report_count") != len(rows):
        return False, "feature relevance report count does not match JSONL rows"
    timestamps: list[float] = []
    for row in rows:
        if row.get("feature_relevance_interval_s") != 60:
            return False, "feature relevance interval is not 60 seconds"
        if row.get("report_nonblocking") is not True:
            return False, "feature relevance reporting was not marked nonblocking"
        if row.get("learner_updates_skipped_for_reporting") != 0:
            return False, "feature relevance reporting skipped learner updates"
        if row.get("uses_framework_compute_feature_relevance") is not True:
            return False, "feature relevance did not use framework diagnostics"
        if not _numeric_gte(row, "latest_report_latency_ms", 0):
            return False, "feature relevance report latency is missing or non-finite"
        if not _numeric_gte(row, "report_timestamp_s", 0):
            return False, "feature relevance report timestamp is missing or non-finite"
        names = row.get("top_feature_names")
        values = row.get("top_feature_relevance_values")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or not isinstance(values, list)
            or len(values) != len(names)
            or any(not _finite_number(value) for value in values)
        ):
            return False, "feature relevance rows lack finite named values"
        timestamps.append(float(row["report_timestamp_s"]))
    if timestamps != sorted(timestamps):
        return False, "feature relevance reports are not timestamp ordered"
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if any(interval < 59.0 or interval > 61.0 for interval in intervals):
        return False, "feature relevance report cadence is not approximately 60 seconds"
    return True, "feature relevance reports prove cadence and finite named values"


def _validate_daemon_throughput(metrics: dict[str, Any]) -> tuple[bool, str]:
    stage_keys = (
        "parse_ms_p50",
        "feature_ms_p50",
        "learner_update_ms_p50",
        "checkpoint_reporting_ms_p50",
        "action_dispatch_ms_p50",
    )
    stage_count_keys = (
        "parse",
        "feature",
        "learner_update",
        "checkpoint_reporting",
        "action_dispatch",
    )
    if not _numeric_gt(metrics, "events_per_second", 0):
        return False, "events_per_second is not positive and finite"
    if (
        not isinstance(metrics.get("n_events"), int)
        or isinstance(metrics.get("n_events"), bool)
        or metrics["n_events"] <= 0
    ):
        return False, "n_events is not a positive integer"
    if not _numeric_gt(metrics, "wall_clock_s", 0):
        return False, "wall_clock_s is not positive and finite"
    measured_eps = metrics["n_events"] / metrics["wall_clock_s"]
    relative_eps_error = abs(metrics["events_per_second"] - measured_eps) / max(
        measured_eps, 1e-12
    )
    if relative_eps_error > 0.05:
        return False, "events_per_second is inconsistent with n_events/wall_clock_s"
    if any(not _numeric_gte(metrics, key, 0) for key in stage_keys):
        return False, "one or more daemon stage timings are negative or non-finite"
    stage_event_counts = metrics.get("stage_event_counts")
    if not isinstance(stage_event_counts, dict):
        return False, "stage_event_counts is missing or malformed"
    if any(stage_event_counts.get(key) != metrics["n_events"] for key in stage_count_keys):
        return False, "stage_event_counts do not cover every measured event"
    if metrics.get("measured_real_daemon_path") is not True:
        return False, "throughput metrics do not mark the real daemon path"
    return True, "throughput metrics include finite real daemon path timings"


def _validate_finite_components(
    metrics: dict[str, Any], required_components: tuple[str, ...]
) -> tuple[bool, str]:
    components = metrics.get("finite_components")
    if not isinstance(components, dict):
        return False, "finite_components is missing or malformed"
    missing = [key for key in required_components if components.get(key) is not True]
    if missing:
        return False, "finite_components does not prove every required component"
    return True, "finite_components proves all required components"


def _validate_idbd_100k_replay(metrics: dict[str, Any]) -> tuple[bool, str]:
    finite_ok, finite_message = _validate_finite_components(
        metrics,
        ("predictions", "parameters", "traces", "step_sizes"),
    )
    if not finite_ok:
        return False, finite_message
    if not _numeric_gte(metrics, "n_events", 100000):
        return False, "100k replay did not process enough events"
    if not _all_finite_flag(metrics):
        return False, "100k replay all_finite flag is not true"
    if not _validate_finite_metrics_gte(metrics, ("final_window_loss",), 0):
        return False, "100k replay final_window_loss is negative or non-finite"
    if not _validate_finite_metrics_gt(metrics, ("mean_step_size",)):
        return False, "100k replay mean_step_size is not positive and finite"
    if not _numeric_gt(metrics, "validation_batch_size", 0):
        return False, "100k replay validation_batch_size is not positive"
    if not _numeric_lte(metrics, "checkpoint_roundtrip_max_abs_diff", 1e-6):
        return False, "100k replay checkpoint roundtrip exceeded tolerance"
    return True, "100k replay metrics meet event, finite, and checkpoint bounds"


def _validate_idbd_full_log(metrics: dict[str, Any]) -> tuple[bool, str]:
    finite_ok, finite_message = _validate_finite_components(
        metrics,
        ("predictions", "parameters", "traces", "step_sizes"),
    )
    if not finite_ok:
        return False, finite_message
    if not _numeric_gte(metrics, "n_events", 1600000):
        return False, "full-log replay did not process enough events"
    if not _all_finite_flag(metrics):
        return False, "full-log replay all_finite flag is not true"
    if metrics.get("resumed_from_midpoint") is not True:
        return False, "full-log replay did not resume from midpoint"
    if not _validate_finite_metrics_gt(metrics, ("events_per_second", "max_rss_mb")):
        return False, "full-log replay throughput or memory is not positive"
    if not _validate_finite_metrics_gte(metrics, ("final_window_loss",), 0):
        return False, "full-log replay final_window_loss is negative or non-finite"
    if not _numeric_gte(metrics, "checkpoint_count", 2):
        return False, "full-log replay did not produce at least two checkpoints"
    if not _numeric_lte(metrics, "resume_final_loss_abs_diff", 1e-6):
        return False, "full-log replay resume equivalence exceeded tolerance"
    return True, "full-log replay metrics meet event, finite, and resume conditions"


def _validate_checkpoint_v2(root: Path, metrics: dict[str, Any]) -> tuple[bool, str]:
    metadata = _load_json(root / "outputs/rlsecd_checkpoint_v2/metadata.json")
    if metrics.get("format_version") != 2:
        return False, "checkpoint format_version is not 2"
    if metrics.get("metadata_present") is not True or not metadata:
        return False, "checkpoint metadata is missing"
    if metadata.get("schema") != "alberta.rlsecd.security_agent_checkpoint.v2":
        return False, "checkpoint metadata schema is not v2"
    if metadata.get("framework_checkpoint_schema") != "alberta.framework.checkpoint.v1":
        return False, "checkpoint metadata lacks framework checkpoint schema"
    if metrics.get("learner_state_present") is not True:
        return False, "checkpoint learner state was not restored"
    if metrics.get("optimizer_state_present") is not True:
        return False, "checkpoint optimizer state was not restored"
    if metrics.get("normalizer_state_present") is not True:
        return False, "checkpoint normalizer state was not restored"
    if metrics.get("restored_step_count_matches") is not True:
        return False, "checkpoint restored step_count does not match"
    if not _numeric_lte(metrics, "prediction_roundtrip_max_abs_diff", 1e-6):
        return False, "checkpoint prediction roundtrip exceeded tolerance"
    return True, "checkpoint v2 restores metadata, state, and predictions"


def _validate_config_roundtrip(metrics: dict[str, Any]) -> tuple[bool, str]:
    required_flags = (
        "learner_config_roundtrip",
        "optimizer_config_roundtrip",
        "normalizer_config_roundtrip",
        "feature_schema_roundtrip",
        "security_agent_config_roundtrip",
    )
    if any(metrics.get(key) is not True for key in required_flags):
        return False, "one or more framework config roundtrips failed"
    component_types = metrics.get("serialized_component_types")
    if not isinstance(component_types, dict):
        return False, "serialized_component_types is missing or malformed"
    required_components = ("learner", "optimizer", "normalizer", "feature_schema")
    if any(not isinstance(component_types.get(key), str) for key in required_components):
        return False, "serialized_component_types lacks required component names"
    if metrics.get("unknown_config_keys") not in ([], ()):
        return False, "config roundtrip reported unknown config keys"
    if metrics.get("dropped_config_keys") not in ([], ()):
        return False, "config roundtrip reported dropped config keys"
    if metrics.get("restored_schema_version_matches") is not True:
        return False, "restored config schema version does not match"
    if not _numeric_lte(metrics, "prediction_roundtrip_max_abs_diff", 1e-6):
        return False, "config prediction roundtrip exceeded tolerance"
    return True, "framework config roundtrip preserves fields and predictions"


def _validate_finite_metrics_gte(
    metrics: dict[str, Any], keys: tuple[str, ...], threshold: float
) -> bool:
    return all(_numeric_gte(metrics, key, threshold) for key in keys)


def _validate_finite_metrics_gt(
    metrics: dict[str, Any], keys: tuple[str, ...]
) -> bool:
    return all(_numeric_gt(metrics, key, 0) for key in keys)


def _claim_metrics_path(claim_scope: str) -> str:
    paths = {
        "rlsecd_gym_control_horde_sarsa_daemon": (
            "outputs/rlsecd_gym_control/metrics.json"
        ),
        "rlsecd_end_to_end_daemon_throughput": "outputs/rlsecd_throughput/metrics.json",
        "rlsecd_oracle_experience_export": (
            "outputs/rlsecd_oracle_experience/manifest.json"
        ),
        "rlsecd_idbd_mlp_100k_replay": "outputs/idbd_mlp_100k/metrics.json",
        "rlsecd_idbd_mlp_full_log_stability": "outputs/idbd_mlp_1_6m/metrics.json",
        "rlsecd_security_agent_orbax_checkpoint_v2": (
            "outputs/rlsecd_checkpoint_v2/metrics.json"
        ),
        "rlsecd_security_agent_framework_config_serialization": (
            "outputs/rlsecd_config_roundtrip/metrics.json"
        ),
        "rlsecd_feature_relevance_periodic_reporting": (
            "outputs/rlsecd_feature_relevance/metrics.jsonl"
        ),
    }
    return paths[claim_scope]


def _load_metrics(root: Path, claim_scope: str) -> dict[str, Any] | None:
    metrics_path = root / _claim_metrics_path(claim_scope)
    if metrics_path.suffix == ".jsonl":
        rows = _load_jsonl(metrics_path, limit=1)
        return rows[0] if rows else None
    return _load_json(metrics_path)


def _claim_condition(
    root: Path, item: dict[str, Any], metrics: dict[str, Any]
) -> tuple[bool, str]:
    claim_scope = item["claim_scope"]

    def validate_gym_control() -> tuple[bool, str]:
        rollout_ok, rollout_message = _validate_rollout_rows(root)
        if not rollout_ok:
            return False, rollout_message
        config_ok, config_message = _validate_gym_control_config(root)
        if not config_ok:
            return False, config_message
        rollout_rows = _load_jsonl(
            root / "outputs/rlsecd_gym_control/rollouts.jsonl", limit=None
        )
        if metrics.get("n_prediction_demons") != 5 or metrics.get("n_control_actions") != 6:
            return False, "gym-control metrics do not match demon/action counts"
        if not (
            isinstance(metrics.get("n_transitions"), int)
            and not isinstance(metrics.get("n_transitions"), bool)
            and metrics["n_transitions"] == len(rollout_rows)
            and metrics["n_transitions"] > 0
        ):
            return False, "gym-control n_transitions does not match rollout rows"
        if not _finite_number(metrics.get("sarsa_td_error_final_window")) or not _finite_number(
            metrics.get("mean_reward")
        ):
            return False, "gym-control scalar metrics are missing or non-finite"
        if metrics.get("uses_framework_sarsa_agent") is not True:
            return False, "gym-control metrics do not prove framework SARSAAgent usage"
        if metrics.get("uses_framework_horde_learner") is not True:
            return False, "gym-control metrics do not prove framework HordeLearner usage"
        return True, "gym-control rollout, config, and metrics match contract"

    validators: dict[str, Callable[[], tuple[bool, str]]] = {
        "rlsecd_gym_control_horde_sarsa_daemon": validate_gym_control,
        "rlsecd_end_to_end_daemon_throughput": lambda: (
            _validate_daemon_throughput(metrics)
        ),
        "rlsecd_oracle_experience_export": lambda: _validate_oracle_rows(root),
        "rlsecd_idbd_mlp_100k_replay": lambda: _validate_idbd_100k_replay(metrics),
        "rlsecd_idbd_mlp_full_log_stability": lambda: _validate_idbd_full_log(
            metrics
        ),
        "rlsecd_security_agent_orbax_checkpoint_v2": lambda: _validate_checkpoint_v2(
            root, metrics
        ),
        "rlsecd_security_agent_framework_config_serialization": lambda: (
            _validate_config_roundtrip(metrics)
        ),
        "rlsecd_feature_relevance_periodic_reporting": lambda: (
            _validate_feature_relevance(root, metrics)
        ),
    }
    return validators[claim_scope]()


def validate_item(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    """Validate one external acceptance item against produced artifacts."""
    missing_artifacts = [
        artifact for artifact in item["required_artifacts"] if not (root / artifact).exists()
    ]
    metrics = _load_metrics(root, item["claim_scope"])
    missing_metrics = list(item["required_metrics"])
    if metrics is not None:
        missing_metrics = [key for key in item["required_metrics"] if key not in metrics]
    condition_passed = False
    condition_message = "metrics missing"
    if not missing_artifacts and metrics is not None and not missing_metrics:
        condition_passed, condition_message = _claim_condition(root, item, metrics)
    passed = not missing_artifacts and not missing_metrics and condition_passed
    return {
        "claim_scope": item["claim_scope"],
        "todo_text": item["todo_text"],
        "passed": passed,
        "missing_artifacts": missing_artifacts,
        "missing_metrics": missing_metrics,
        "condition_passed": condition_passed,
        "condition_message": condition_message,
    }


def validate(
    spec_path: Path = DEFAULT_SPEC,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Validate all external acceptance items."""
    spec = _load_json(spec_path)
    if spec is None:
        return {
            "schema": "alberta.rlsecd_external_acceptance_status.v1",
            "accepted": False,
            "spec_exists": False,
            "items": [],
            "boundary": f"acceptance spec not found: {spec_path}",
        }
    item_statuses = [validate_item(artifact_root, item) for item in spec["items"]]
    accepted = all(item["passed"] for item in item_statuses)
    return {
        "schema": "alberta.rlsecd_external_acceptance_status.v1",
        "accepted": accepted,
        "spec_exists": True,
        "artifact_root": str(artifact_root),
        "n_items": len(item_statuses),
        "n_passed": sum(1 for item in item_statuses if item["passed"]),
        "items": item_statuses,
        "boundary": (
            "all external rlsecd acceptance artifacts passed"
            if accepted
            else "external rlsecd acceptance artifacts are missing or incomplete"
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--write-status", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def main() -> int:
    """Validate artifacts and write status JSON."""
    args = parse_args()
    status = validate(args.spec, args.artifact_root)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    return 0 if status["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

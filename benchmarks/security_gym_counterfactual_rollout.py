#!/usr/bin/env python3
"""Generate local security-gym counterfactual action rollout evidence.

This benchmark closes the framework-side part of the active-defense evidence
boundary by exercising the real ``security-gym`` environment when it is present
as a sibling checkout. It does not require or claim the missing ``rlsecd``
daemon loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRolloutStep,
    ThroughputMeter,
    to_security_gym_action,
    validate_security_rollout,
)

DEFAULT_SECURITY_GYM = Path("/Users/shawwalters/Desktop/nca_fun/security-gym")
DEFAULT_OUTPUT = Path("outputs/security_gym_counterfactual_rollout/results.json")


@dataclass(frozen=True)
class PolicyResult:
    """Summary metrics for one security-gym policy rollout."""

    name: str
    steps: int
    total_reward: float
    mean_reward: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    elapsed_s: float
    events_per_second: float
    n_rollout_records: int


def _import_security_gym(security_gym_root: Path) -> tuple[object, object, object]:
    """Import sibling security-gym modules after adding its ``src`` path."""
    src = security_gym_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"security-gym src path not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import gymnasium as gym  # noqa: PLC0415
    import security_gym  # noqa: F401, PLC0415
    from security_gym.data.event_store import EventStore  # noqa: PLC0415
    from security_gym.parsers.base import ParsedEvent  # noqa: PLC0415

    return gym, EventStore, ParsedEvent


def _make_event_stream(
    db_path: Path,
    event_store_cls: object,
    parsed_event_cls: object,
) -> None:
    """Create a deterministic labeled auth-log stream in security-gym format."""
    base = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
    campaign_id = str(uuid4())
    rows: list[tuple[object, dict[str, object]]] = []
    benign_ip = "10.0.0.5"
    attack_ip = "192.168.1.50"
    for idx in range(48):
        is_attack = (idx % 4) in (2, 3)
        src_ip = attack_ip if is_attack else benign_ip
        event_type = "auth_failure" if is_attack else "auth_success"
        verb = "Failed password for root" if is_attack else "Accepted password for admin"
        rows.append(
            (
                parsed_event_cls(
                    timestamp=base + timedelta(seconds=idx),
                    source="auth_log",
                    raw_line=(
                        f"May 21 12:00:{idx:02d} host sshd[{1000 + idx}]: "
                        f"{verb} from {src_ip} port {50000 + idx} ssh2"
                    ),
                    event_type=event_type,
                    fields={
                        "event_type": event_type,
                        "auth_method": "password",
                        "port": 50000 + idx,
                    },
                    src_ip=src_ip,
                    username="root" if is_attack else "admin",
                    service="sshd",
                    session_id=f"{src_ip}:{50000 + idx}",
                ),
                {
                    "is_malicious": int(is_attack),
                    "campaign_id": campaign_id if is_attack else None,
                    "attack_type": "brute_force" if is_attack else None,
                    "attack_stage": "initial_access" if is_attack else None,
                    "severity": 2 if is_attack else 0,
                },
            )
        )

    store = event_store_cls(db_path, mode="w")
    try:
        for event, ground_truth in rows:
            store.insert_event(event, ground_truth=ground_truth)
    finally:
        store.close()


def _state_vector(obs: Mapping[str, object], info: Mapping[str, object]) -> tuple[float, ...]:
    """Convert a security-gym observation/info pair to compact rollout features."""
    system_stats = obs.get("system_stats", np.zeros(3, dtype=np.float32))
    stats = np.asarray(system_stats, dtype=np.float32).reshape(-1)
    if stats.size < 3:
        stats = np.pad(stats, (0, 3 - stats.size))
    return (
        min(len(str(obs.get("auth_log", ""))) / 1000.0, 1.0),
        min(len(str(obs.get("syslog", ""))) / 1000.0, 1.0),
        min(len(str(obs.get("web_log", ""))) / 1000.0, 1.0),
        min(len(str(obs.get("process_events", ""))) / 1000.0, 1.0),
        min(len(str(obs.get("network_events", ""))) / 1000.0, 1.0),
        min(len(str(obs.get("file_events", ""))) / 1000.0, 1.0),
        float(stats[0]),
        float(stats[1]),
        float(stats[2]),
        float(len(info.get("blocked_ips", []))),
        float(len(info.get("throttled_ips", []))),
        float(bool(info.get("is_isolated", False))),
    )


def _run_policy(
    *,
    gym: object,
    db_path: Path,
    name: str,
    choose_action: Callable[
        [Mapping[str, object], Mapping[str, object]],
        tuple[SecurityAction, float],
    ],
    schema: SecurityFeatureSchema,
    max_steps: int,
) -> tuple[PolicyResult, list[SecurityRolloutStep]]:
    """Run one policy and return metrics plus validated rollout records."""
    env = gym.make("SecurityLogStream-Text-v0", db_path=str(db_path))
    obs, info = env.reset(seed=0)
    steps: list[SecurityRolloutStep] = []
    total_reward = 0.0
    tp = fp = tn = fn = 0
    meter = ThroughputMeter()
    start = time.perf_counter()
    for _ in range(max_steps):
        state = _state_vector(obs, info)
        action, risk = choose_action(obs, info)
        ground_truth = info.get("ground_truth", {})
        is_malicious = bool(
            ground_truth.get("is_malicious", False)
            if isinstance(ground_truth, Mapping)
            else False
        )
        acted = action in {
            SecurityAction.ALERT,
            SecurityAction.THROTTLE,
            SecurityAction.BLOCK,
            SecurityAction.ISOLATE,
        }
        next_obs, reward, terminated, truncated, next_info = env.step(
            to_security_gym_action(action, risk)
        )
        next_state = _state_vector(next_obs, next_info)
        total_reward += float(reward)
        steps.append(
            SecurityRolloutStep(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=bool(terminated),
                truncated=bool(truncated),
                policy_metadata={
                    "policy": name,
                    "risk_score": risk,
                    "is_malicious": is_malicious,
                    "src_ip": info.get("src_ip"),
                },
            )
        )
        if is_malicious and acted:
            tp += 1
        elif is_malicious and not acted:
            fn += 1
        elif not is_malicious and acted:
            fp += 1
        else:
            tn += 1
        obs, info = next_obs, next_info
        meter.tick()
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - start
    throughput = meter.measure()
    env.close()
    validate_security_rollout(steps, schema)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = PolicyResult(
        name=name,
        steps=len(steps),
        total_reward=total_reward,
        mean_reward=total_reward / len(steps) if steps else 0.0,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        elapsed_s=elapsed,
        events_per_second=throughput.events_per_second,
        n_rollout_records=len(steps),
    )
    return result, steps


def run_benchmark(
    security_gym_root: Path = DEFAULT_SECURITY_GYM,
    max_steps: int = 48,
    *,
    include_rollout_records: bool = False,
) -> dict[str, object]:
    """Run pass-only and oracle-block counterfactual rollouts."""
    gym, event_store_cls, parsed_event_cls = _import_security_gym(security_gym_root)
    schema = SecurityFeatureSchema(
        names=(
            "auth_chars_norm",
            "syslog_chars_norm",
            "web_chars_norm",
            "process_chars_norm",
            "network_chars_norm",
            "file_chars_norm",
            "system_load",
            "system_mem_used",
            "system_disk_used",
            "n_blocked_ips",
            "n_throttled_ips",
            "is_isolated",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "security_gym_counterfactual.db"
        _make_event_stream(db_path, event_store_cls, parsed_event_cls)

        def pass_policy(
            _obs: Mapping[str, object],
            _info: Mapping[str, object],
        ) -> tuple[SecurityAction, float]:
            return SecurityAction.PASS, 0.0

        def oracle_block_policy(
            _obs: Mapping[str, object],
            info: Mapping[str, object],
        ) -> tuple[SecurityAction, float]:
            ground_truth = info.get("ground_truth", {})
            is_malicious = bool(
                ground_truth.get("is_malicious", False)
                if isinstance(ground_truth, Mapping)
                else False
            )
            true_risk = float(
                ground_truth.get("true_risk", 0.0)
                if isinstance(ground_truth, Mapping)
                else 0.0
            )
            return (
                SecurityAction.BLOCK if is_malicious else SecurityAction.PASS,
                true_risk,
            )

        pass_result, pass_steps = _run_policy(
            gym=gym,
            db_path=db_path,
            name="pass_only",
            choose_action=pass_policy,
            schema=schema,
            max_steps=max_steps,
        )
        oracle_result, oracle_steps = _run_policy(
            gym=gym,
            db_path=db_path,
            name="oracle_block_malicious",
            choose_action=oracle_block_policy,
            schema=schema,
            max_steps=max_steps,
        )

    reward_lift = oracle_result.total_reward - pass_result.total_reward
    recall_lift = oracle_result.recall - pass_result.recall
    passed = bool(
        pass_result.steps >= 20
        and oracle_result.steps >= 20
        and reward_lift > 5.0
        and recall_lift > 0.5
        and oracle_result.false_positives == 0
    )
    result: dict[str, object] = {
        "schema": "alberta.security_gym.counterfactual_rollout.v1",
        "claim_scope": "local_security_gym_counterfactual_action_rollout",
        "security_gym_root": str(security_gym_root),
        "feature_schema": schema.to_dict(),
        "policies": {
            "pass_only": asdict(pass_result),
            "oracle_block_malicious": asdict(oracle_result),
        },
        "comparison": {
            "reward_lift": reward_lift,
            "recall_lift": recall_lift,
            "oracle_precision": oracle_result.precision,
            "oracle_false_positives": oracle_result.false_positives,
        },
        "sample_rollout_records": {
            "pass_only": [step.to_dict() for step in pass_steps[:3]],
            "oracle_block_malicious": [step.to_dict() for step in oracle_steps[:3]],
        },
        "passed": passed,
        "boundary": (
            "uses real local security-gym environment and framework rollout "
            "records; does not prove unavailable rlsecd daemon integration"
        ),
    }
    if include_rollout_records:
        result["rollout_records"] = {
            "pass_only": [step.to_dict() for step in pass_steps],
            "oracle_block_malicious": [step.to_dict() for step in oracle_steps],
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security-gym-root", type=Path, default=DEFAULT_SECURITY_GYM)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write JSON evidence."""
    args = parse_args(argv)
    result = run_benchmark(args.security_gym_root, args.max_steps)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 6 downstream integration benchmark on security-gym.

This benchmark exercises the sibling ``security-gym`` API with the promoted
Step 6 differential SARSA facade.  It builds a small synthetic security-event
SQLite stream, observes the real ``SecurityLogStreamEnv`` dictionary
observations, maps Step 6 actions to defensive actions, and verifies that
average-reward control learns to alert on attack-like log windows while passing
benign windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.steps import (
    Step6DifferentialSARSAConfig,
    init_step6_state,
    make_step6_differential_sarsa_agent,
    step6_update,
)

SECURITY_GYM_ROOT = Path("/Users/shawwalters/Desktop/nca_fun/security-gym")


@dataclass(frozen=True)
class Step6SecurityGymConfig:
    """Configuration for the security-gym integration benchmark."""

    seeds: int = 10
    cycles: int = 24
    benign_per_cycle: int = 8
    attack_per_cycle: int = 8
    passes: int = 6
    eval_cycles: int = 6
    q_step_size: float = 0.08
    average_reward_step_size: float = 0.01
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02

    def __post_init__(self) -> None:
        """Validate scalar settings."""
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.cycles < 1 or self.eval_cycles < 1:
            raise ValueError("cycles and eval_cycles must be positive")
        if self.benign_per_cycle < 1 or self.attack_per_cycle < 1:
            raise ValueError("events per cycle must be positive")
        if self.passes < 1:
            raise ValueError("passes must be positive")


@dataclass(frozen=True)
class Step6SecurityGymSummary:
    """JSON-serializable benchmark summary."""

    schema: str
    claim_scope: str
    config: dict[str, Any]
    elapsed_s: float
    aggregate: dict[str, float | int | bool | str]
    per_seed: list[dict[str, float | int]]


def _ensure_security_gym() -> None:
    src = SECURITY_GYM_ROOT / "src"
    if not src.exists():
        raise FileNotFoundError(f"security-gym source not found at {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _make_security_db(path: Path, config: Step6SecurityGymConfig) -> None:
    _ensure_security_gym()
    from security_gym.data.event_store import EventStore
    from security_gym.parsers.base import ParsedEvent

    base = datetime(2026, 2, 17, 10, 0, tzinfo=UTC)
    attack_campaign = str(uuid4())
    events: list[ParsedEvent] = []
    labels: list[dict[str, Any]] = []
    idx = 0
    for cycle in range(config.cycles):
        for item in range(config.benign_per_cycle):
            timestamp = base + timedelta(seconds=idx)
            port = 50_000 + cycle * config.benign_per_cycle + item
            events.append(
                ParsedEvent(
                    timestamp=timestamp,
                    source="auth_log",
                    raw_line=(
                        f"{timestamp.isoformat()} host sshd[100]: Accepted "
                        f"password for admin from 10.0.0.1 port {port} ssh2"
                    ),
                    event_type="auth_success",
                    fields={"event_type": "auth_success", "port": port},
                    src_ip="10.0.0.1",
                    username="admin",
                    service="sshd",
                    session_id=f"10.0.0.1:{port}",
                )
            )
            labels.append({"is_malicious": 0, "severity": 0})
            idx += 1
        for item in range(config.attack_per_cycle):
            timestamp = base + timedelta(seconds=idx)
            port = 60_000 + cycle * config.attack_per_cycle + item
            events.append(
                ParsedEvent(
                    timestamp=timestamp,
                    source="auth_log",
                    raw_line=(
                        f"{timestamp.isoformat()} host sshd[200]: Failed "
                        f"password for root from 192.168.1.50 port {port} ssh2"
                    ),
                    event_type="auth_failure",
                    fields={"event_type": "auth_failure", "port": port},
                    src_ip="192.168.1.50",
                    username="root",
                    service="sshd",
                    session_id=f"192.168.1.50:{port}",
                )
            )
            labels.append(
                {
                    "is_malicious": 1,
                    "campaign_id": attack_campaign,
                    "attack_type": "brute_force",
                    "attack_stage": "initial_access",
                    "severity": 2,
                }
            )
            idx += 1
    with EventStore(path, mode="w") as store:
        store.bulk_insert(events, labels)


def _features(observation: dict[str, Any]) -> jnp.ndarray:
    auth_log = str(observation.get("auth_log", "")).lower()
    stats = np.asarray(
        observation.get("system_stats", np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    )
    failed = auth_log.count("failed")
    accepted = auth_log.count("accepted")
    root = auth_log.count("root")
    attack_ip = auth_log.count("192.168.1.50")
    benign_ip = auth_log.count("10.0.0.1")
    line_count = len([line for line in auth_log.splitlines() if line.strip()])
    feature_values = np.array(
        [
            1.0,
            min(failed / 4.0, 1.0),
            min(accepted / 4.0, 1.0),
            min(root / 4.0, 1.0),
            min(attack_ip / 4.0, 1.0),
            min(benign_ip / 4.0, 1.0),
            min(line_count / 8.0, 1.0),
            *stats.tolist(),
        ],
        dtype=np.float32,
    )
    return jnp.asarray(feature_values, dtype=jnp.float32)


def _env_action(step6_action: int) -> dict[str, Any]:
    if step6_action == 1:
        return {
            "action": 1,  # ACTION_ALERT
            "risk_score": np.array([5.0], dtype=np.float32),
        }
    return {
        "action": 0,  # ACTION_PASS
        "risk_score": np.array([0.0], dtype=np.float32),
    }


def _policy_action(agent: Any, state: Any, features: jnp.ndarray) -> int:
    q_values = agent.q_values(state, features)
    return int(jnp.argmax(q_values))


def _run_pass_policy(db_path: Path, config: Step6SecurityGymConfig) -> float:
    _ensure_security_gym()
    from security_gym.envs.log_stream_env import SecurityLogStreamEnv

    env = SecurityLogStreamEnv(
        db_path=db_path,
        tail_lines=1,
        reward_config={"include_risk_reward": True},
    )
    try:
        env.reset(seed=0)
        rewards: list[float] = []
        for _ in range(config.eval_cycles * (config.benign_per_cycle + config.attack_per_cycle)):
            _, reward, _, truncated, _ = env.step(_env_action(0))
            rewards.append(float(reward))
            if truncated:
                break
        return sum(rewards) / max(len(rewards), 1)
    finally:
        env.close()


def _evaluate_policy(
    db_path: Path,
    config: Step6SecurityGymConfig,
    agent: Any,
    state: Any,
) -> tuple[float, float, float]:
    _ensure_security_gym()
    from security_gym.envs.log_stream_env import SecurityLogStreamEnv

    env = SecurityLogStreamEnv(
        db_path=db_path,
        tail_lines=1,
        reward_config={"include_risk_reward": True},
    )
    try:
        obs, _ = env.reset(seed=0)
        rewards: list[float] = []
        attack_correct = 0
        benign_correct = 0
        attack_total = 0
        benign_total = 0
        for _ in range(config.eval_cycles * (config.benign_per_cycle + config.attack_per_cycle)):
            features = _features(obs)
            action = _policy_action(agent, state, features)
            obs, reward, _, truncated, info = env.step(_env_action(action))
            rewards.append(float(reward))
            is_attack = bool(info.get("ground_truth", {}).get("is_malicious"))
            if is_attack:
                attack_total += 1
                attack_correct += int(action == 1)
            else:
                benign_total += 1
                benign_correct += int(action == 0)
            if truncated:
                break
        return (
            sum(rewards) / max(len(rewards), 1),
            attack_correct / max(attack_total, 1),
            benign_correct / max(benign_total, 1),
        )
    finally:
        env.close()


def run_step6_security_gym_benchmark(
    config: Step6SecurityGymConfig,
) -> Step6SecurityGymSummary:
    """Run the downstream security-gym integration benchmark."""
    _ensure_security_gym()
    from security_gym.envs.log_stream_env import SecurityLogStreamEnv

    start = time.time()
    control_cfg = Step6DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=config.q_step_size,
        average_reward_step_size=config.average_reward_step_size,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_steps=config.passes
        * config.cycles
        * (config.benign_per_cycle + config.attack_per_cycle),
    )
    agent = make_step6_differential_sarsa_agent(control_cfg)

    per_seed: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "security_gym_step6.db"
        _make_security_db(db_path, config)
        pass_policy_reward = _run_pass_policy(db_path, config)
        for seed in range(config.seeds):
            env = SecurityLogStreamEnv(
                db_path=db_path,
                tail_lines=1,
                reward_config={"include_risk_reward": True},
            )
            try:
                obs, _ = env.reset(seed=seed)
                features = _features(obs)
                state = init_step6_state(
                    agent,
                    feature_dim=int(features.shape[0]),
                    key=jr.key(seed),
                    initial_features=features,
                )
                train_rewards: list[float] = []
                for _ in range(
                    config.passes
                    * config.cycles
                    * (config.benign_per_cycle + config.attack_per_cycle)
                ):
                    action = int(state.last_action)
                    next_obs, reward, _, truncated, _ = env.step(_env_action(action))
                    next_features = _features(next_obs)
                    result = step6_update(
                        agent,
                        state,
                        jnp.asarray(reward, dtype=jnp.float32),
                        next_features,
                    )
                    state = result.state
                    train_rewards.append(float(reward))
                    if truncated:
                        next_obs, _ = env.reset(seed=seed)
                        state = state.replace(  # type: ignore[attr-defined]
                            last_observation=_features(next_obs)
                        )
                eval_reward, attack_rate, benign_rate = _evaluate_policy(
                    db_path,
                    config,
                    agent,
                    state,
                )
                per_seed.append(
                    {
                        "seed": seed,
                        "train_mean_reward": sum(train_rewards)
                        / max(len(train_rewards), 1),
                        "eval_mean_reward": eval_reward,
                        "pass_policy_eval_mean_reward": pass_policy_reward,
                        "reward_improvement_vs_pass": eval_reward
                        - pass_policy_reward,
                        "attack_alert_rate": attack_rate,
                        "benign_pass_rate": benign_rate,
                        "average_reward_estimate": float(state.average_reward),
                    }
                )
            finally:
                env.close()

    improvements = [row["reward_improvement_vs_pass"] for row in per_seed]
    attack_rates = [row["attack_alert_rate"] for row in per_seed]
    benign_rates = [row["benign_pass_rate"] for row in per_seed]
    aggregate: dict[str, float | int | bool | str] = {
        "security_gym_path": str(SECURITY_GYM_ROOT),
        "n_seeds": len(per_seed),
        "mean_eval_reward": sum(row["eval_mean_reward"] for row in per_seed)
        / len(per_seed),
        "mean_pass_policy_eval_reward": sum(
            row["pass_policy_eval_mean_reward"] for row in per_seed
        )
        / len(per_seed),
        "mean_reward_improvement_vs_pass": sum(improvements) / len(per_seed),
        "stderr_reward_improvement_vs_pass": _stderr(improvements),
        "reward_win_count": sum(value > 0.0 for value in improvements),
        "mean_attack_alert_rate": sum(attack_rates) / len(per_seed),
        "mean_benign_pass_rate": sum(benign_rates) / len(per_seed),
    }
    aggregate["passed"] = bool(
        aggregate["n_seeds"] >= 10
        and aggregate["mean_reward_improvement_vs_pass"] >= 1.0
        and aggregate["reward_win_count"] == aggregate["n_seeds"]
        and aggregate["mean_attack_alert_rate"] >= 0.85
        and aggregate["mean_benign_pass_rate"] >= 0.85
    )
    return Step6SecurityGymSummary(
        schema="alberta.step6.security_gym_integration.v1",
        claim_scope="downstream_security_gym_average_reward_control_integration",
        config=asdict(config),
        elapsed_s=time.time() - start,
        aggregate=aggregate,
        per_seed=per_seed,
    )


def main() -> None:
    """Run the benchmark and write a JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=24)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step6_security_gym/results.json"),
    )
    args = parser.parse_args()
    config = Step6SecurityGymConfig(
        seeds=args.seeds,
        cycles=args.cycles,
        passes=args.passes,
    )
    summary = run_step6_security_gym_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

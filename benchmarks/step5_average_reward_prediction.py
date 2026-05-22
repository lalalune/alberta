#!/usr/bin/env python3
"""Seeded Step 5 differential TD prediction benchmark.

The benchmark uses a deterministic continuing cycle with a known average reward
and known centered differential value function.  It is intentionally simple: the
point is to prove that the Step 5 average-reward learner solves a continuing
prediction problem with a closed-form target, not to claim broader nonlinear
Horde or control coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.average_reward import (  # noqa: E402
    DifferentialTDConfig,
    DifferentialTDLearner,
    run_differential_td_from_arrays,
)

DEFAULT_REWARDS = (0.0, 1.0, 2.0)


def cycle_transitions(
    rewards: Sequence[float],
    *,
    steps: int,
    start_state: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one-hot observations and rewards for a deterministic cycle."""
    if steps < 1:
        raise ValueError("steps must be positive")
    n_states = len(rewards)
    if n_states < 2:
        raise ValueError("at least two rewards/states are required")

    states = (jnp.arange(steps, dtype=jnp.int32) + start_state) % n_states
    next_states = (states + 1) % n_states
    observations = jnp.eye(n_states, dtype=jnp.float32)[states]
    next_observations = jnp.eye(n_states, dtype=jnp.float32)[next_states]
    reward_array = jnp.asarray(rewards, dtype=jnp.float32)[states]
    return observations, reward_array, next_observations


def centered_differential_values(rewards: Sequence[float]) -> jnp.ndarray:
    """Solve `rho + h(s) = r(s) + h(s')` for a deterministic cycle."""
    rewards_array = jnp.asarray(rewards, dtype=jnp.float32)
    rho = jnp.mean(rewards_array)
    centered_rewards = rewards_array - rho

    values = [0.0]
    running = 0.0
    for reward in centered_rewards[:-1]:
        running -= float(reward)
        values.append(running)
    raw = jnp.asarray(values, dtype=jnp.float32)
    return raw - jnp.mean(raw)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def run_seed(
    seed: int,
    *,
    rewards: Sequence[float],
    steps: int,
    step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    tail_window: int,
) -> dict[str, Any]:
    """Run one deterministic cycle seed and return scalar metrics."""
    n_states = len(rewards)
    start_state = seed % n_states
    learner = DifferentialTDLearner(
        DifferentialTDConfig(
            step_size=step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
        )
    )
    state = learner.init(n_states)
    observations, reward_array, next_observations = cycle_transitions(
        rewards,
        steps=steps,
        start_state=start_state,
    )
    result = run_differential_td_from_arrays(
        learner,
        state,
        observations,
        reward_array,
        next_observations,
    )
    result.td_errors.block_until_ready()

    true_average_reward = float(jnp.mean(jnp.asarray(rewards, dtype=jnp.float32)))
    true_values = centered_differential_values(rewards)
    predictions = learner.predict(result.state, jnp.eye(n_states, dtype=jnp.float32))
    centered_predictions = predictions - jnp.mean(predictions)
    tail = result.td_errors[-tail_window:]

    average_reward_abs_error = float(
        jnp.abs(result.state.average_reward - true_average_reward)
    )
    centered_value_rmse = float(
        jnp.sqrt(jnp.mean((centered_predictions - true_values) ** 2))
    )
    tail_td_error_mse = float(jnp.mean(tail**2))
    finite = bool(
        jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(jnp.isfinite(centered_predictions))
    )

    return {
        "seed": seed,
        "start_state": start_state,
        "final_average_reward": float(result.state.average_reward),
        "average_reward_abs_error": average_reward_abs_error,
        "centered_value_rmse": centered_value_rmse,
        "tail_td_error_mse": tail_td_error_mse,
        "centered_predictions": [float(value) for value in centered_predictions],
        "finite": finite,
        "passed": bool(
            finite
            and average_reward_abs_error <= 0.02
            and centered_value_rmse <= 0.05
            and tail_td_error_mse <= 0.002
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    rewards: Sequence[float],
    step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    tail_window: int,
) -> dict[str, Any]:
    """Run the multi-seed benchmark and return a JSON-serializable report."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if tail_window < 1 or tail_window > steps:
        raise ValueError("tail_window must be in [1, steps]")

    per_seed = [
        run_seed(
            seed,
            rewards=rewards,
            steps=steps,
            step_size=step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            tail_window=tail_window,
        )
        for seed in range(seeds)
    ]
    average_reward_errors = [
        float(row["average_reward_abs_error"]) for row in per_seed
    ]
    value_rmses = [float(row["centered_value_rmse"]) for row in per_seed]
    tail_mses = [float(row["tail_td_error_mse"]) for row in per_seed]
    true_average_reward = float(jnp.mean(jnp.asarray(rewards, dtype=jnp.float32)))
    true_values = centered_differential_values(rewards)
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.average_reward_prediction_benchmark.v1",
        "claim_scope": "closed_form_continuing_differential_td_prediction",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "rewards": [float(reward) for reward in rewards],
            "step_size": step_size,
            "average_reward_step_size": average_reward_step_size,
            "trace_decay": trace_decay,
            "tail_window": tail_window,
        },
        "target": {
            "true_average_reward": true_average_reward,
            "centered_differential_values": [float(value) for value in true_values],
        },
        "baseline": {
            "zero_average_reward_abs_error": abs(true_average_reward),
            "zero_value_centered_rmse": float(jnp.sqrt(jnp.mean(true_values**2))),
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(average_reward_errors)
            / len(average_reward_errors),
            "stderr_average_reward_abs_error": _stderr(average_reward_errors),
            "mean_centered_value_rmse": sum(value_rmses) / len(value_rmses),
            "stderr_centered_value_rmse": _stderr(value_rmses),
            "mean_tail_td_error_mse": sum(tail_mses) / len(tail_mses),
            "stderr_tail_td_error_mse": _stderr(tail_mses),
            "n_passed": sum(1 for row in per_seed if row["passed"]),
            "n_seeds": seeds,
        },
        "per_seed": per_seed,
        "passed": bool(passed),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--tail-window", type=int, default=1_000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--average-reward-step-size", type=float, default=0.01)
    parser.add_argument("--trace-decay", type=float, default=0.0)
    parser.add_argument("--rewards", type=float, nargs="+", default=DEFAULT_REWARDS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        rewards=args.rewards,
        step_size=args.step_size,
        average_reward_step_size=args.average_reward_step_size,
        trace_decay=args.trace_decay,
        tail_window=args.tail_window,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

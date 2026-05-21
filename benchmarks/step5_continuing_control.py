#!/usr/bin/env python3
"""Seeded average-reward continuing-control benchmark.

This benchmark exercises the Step 6 differential SARSA primitive that Step 5/6
average-reward work depends on.  The environment is a one-state continuing task
with two actions and a known optimal average reward.  It is deliberately small
so the result is a direct control sanity check, not a broad bsuite claim.
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
import jax.random as jr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.average_reward import (  # noqa: E402
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)

DEFAULT_ACTION_REWARDS = (0.0, 1.0)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def run_seed(
    seed: int,
    *,
    steps: int,
    final_window: int,
    action_rewards: Sequence[float],
    q_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> dict[str, Any]:
    """Run one differential SARSA seed on the one-state control task."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    rewards_by_action = jnp.asarray(action_rewards, dtype=jnp.float32)
    optimal_action = int(jnp.argmax(rewards_by_action))
    optimal_reward = float(jnp.max(rewards_by_action))
    observation = jnp.ones((1,), dtype=jnp.float32)

    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=len(action_rewards),
            q_step_size=q_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
    )
    state = agent.init(1, jr.key(seed))
    state, first_action = agent.start(state, observation)

    rewards: list[float] = []
    actions = [int(first_action)]
    td_errors: list[float] = []
    average_rewards: list[float] = []
    for _step in range(steps):
        reward = rewards_by_action[state.last_action]
        result = agent.update(state, reward, observation)
        state = result.state
        rewards.append(float(result.reward))
        actions.append(int(result.action))
        td_errors.append(float(result.td_error))
        average_rewards.append(float(result.average_reward))

    final_actions = actions[-final_window:]
    final_rewards = rewards[-final_window:]
    final_td_errors = td_errors[-final_window:]
    q_values = agent.q_values(state, observation)
    greedy_action = int(jnp.argmax(q_values))
    final_reward_mean = sum(final_rewards) / len(final_rewards)
    final_optimal_action_rate = sum(
        1 for action in final_actions if action == optimal_action
    ) / len(final_actions)
    average_reward_abs_error = abs(float(state.average_reward) - optimal_reward)
    tail_td_error_mse = sum(error * error for error in final_td_errors) / len(
        final_td_errors
    )
    finite = bool(
        jnp.all(jnp.isfinite(q_values))
        & jnp.isfinite(state.average_reward)
        & jnp.asarray(math.isfinite(tail_td_error_mse))
    )

    return {
        "seed": seed,
        "optimal_action": optimal_action,
        "first_action": int(first_action),
        "greedy_action": greedy_action,
        "final_average_reward": float(state.average_reward),
        "average_reward_abs_error": average_reward_abs_error,
        "final_reward_mean": final_reward_mean,
        "final_optimal_action_rate": final_optimal_action_rate,
        "tail_td_error_mse": tail_td_error_mse,
        "q_values": [float(value) for value in q_values],
        "finite": finite,
        "passed": bool(
            finite
            and greedy_action == optimal_action
            and average_reward_abs_error <= 0.02
            and final_reward_mean >= optimal_reward - 0.01
            and final_optimal_action_rate >= 0.99
            and tail_td_error_mse <= 0.002
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    action_rewards: Sequence[float],
    q_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> dict[str, Any]:
    """Run the multi-seed continuing-control benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if len(action_rewards) < 2:
        raise ValueError("at least two actions are required")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            action_rewards=action_rewards,
            q_step_size=q_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
        for seed in range(seeds)
    ]
    average_reward_errors = [
        float(row["average_reward_abs_error"]) for row in per_seed
    ]
    final_reward_means = [float(row["final_reward_mean"]) for row in per_seed]
    optimal_action_rates = [
        float(row["final_optimal_action_rate"]) for row in per_seed
    ]
    tail_mses = [float(row["tail_td_error_mse"]) for row in per_seed]
    optimal_reward = max(float(reward) for reward in action_rewards)
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.continuing_control_benchmark.v1",
        "claim_scope": "one_state_average_reward_differential_sarsa_control",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "action_rewards": [float(reward) for reward in action_rewards],
            "q_step_size": q_step_size,
            "average_reward_step_size": average_reward_step_size,
            "trace_decay": trace_decay,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
        },
        "target": {
            "optimal_action": int(jnp.argmax(jnp.asarray(action_rewards))),
            "optimal_average_reward": optimal_reward,
        },
        "baseline": {
            "uniform_random_expected_reward": sum(float(r) for r in action_rewards)
            / len(action_rewards),
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(average_reward_errors)
            / len(average_reward_errors),
            "stderr_average_reward_abs_error": _stderr(average_reward_errors),
            "mean_final_reward": sum(final_reward_means) / len(final_reward_means),
            "stderr_final_reward": _stderr(final_reward_means),
            "mean_final_optimal_action_rate": sum(optimal_action_rates)
            / len(optimal_action_rates),
            "stderr_final_optimal_action_rate": _stderr(optimal_action_rates),
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
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--action-rewards", type=float, nargs="+", default=DEFAULT_ACTION_REWARDS)
    parser.add_argument("--q-step-size", type=float, default=0.05)
    parser.add_argument("--average-reward-step-size", type=float, default=0.01)
    parser.add_argument("--trace-decay", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.5)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        action_rewards=args.action_rewards,
        q_step_size=args.q_step_size,
        average_reward_step_size=args.average_reward_step_size,
        trace_decay=args.trace_decay,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

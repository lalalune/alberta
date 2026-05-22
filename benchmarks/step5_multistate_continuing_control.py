#!/usr/bin/env python3
"""Seeded multi-state average-reward continuing-control benchmark."""

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


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _one_hot(state: int, n_states: int) -> jnp.ndarray:
    return jnp.eye(n_states, dtype=jnp.float32)[state]


def _transition(state: int, action: int) -> tuple[float, int]:
    """Two-state task: the optimal policy alternates states for reward 1."""
    if state == 0 and action == 1:
        return 1.0, 1
    if state == 1 and action == 0:
        return 1.0, 0
    return 0.0, state


def run_seed(
    seed: int,
    *,
    steps: int,
    final_window: int,
    q_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> dict[str, Any]:
    """Run one differential SARSA seed on the two-state task."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    n_states = 2
    n_actions = 2
    optimal_policy = [1, 0]
    optimal_reward = 1.0
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=n_actions,
            q_step_size=q_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
    )
    env_state = seed % n_states
    agent_state = agent.init(n_states, jr.key(seed))
    agent_state, first_action = agent.start(agent_state, _one_hot(env_state, n_states))

    rewards: list[float] = []
    policy_matches: list[float] = []
    td_errors: list[float] = []
    actions = [int(first_action)]
    visited_states = [env_state]
    for _step in range(steps):
        action = int(agent_state.last_action)
        reward, next_env_state = _transition(env_state, action)
        next_observation = _one_hot(next_env_state, n_states)
        result = agent.update(agent_state, jnp.array(reward, dtype=jnp.float32), next_observation)
        agent_state = result.state
        rewards.append(reward)
        policy_matches.append(float(action == optimal_policy[env_state]))
        td_errors.append(float(result.td_error))
        actions.append(int(result.action))
        visited_states.append(next_env_state)
        env_state = next_env_state

    q_values_by_state = [
        agent.q_values(agent_state, _one_hot(state, n_states))
        for state in range(n_states)
    ]
    greedy_policy = [int(jnp.argmax(values)) for values in q_values_by_state]
    final_rewards = rewards[-final_window:]
    final_policy_matches = policy_matches[-final_window:]
    final_td_errors = td_errors[-final_window:]
    final_reward_mean = sum(final_rewards) / len(final_rewards)
    final_policy_match_rate = sum(final_policy_matches) / len(final_policy_matches)
    average_reward_abs_error = abs(float(agent_state.average_reward) - optimal_reward)
    tail_td_error_mse = sum(error * error for error in final_td_errors) / len(
        final_td_errors
    )
    finite = bool(
        all(math.isfinite(value) for value in final_td_errors)
        and math.isfinite(float(agent_state.average_reward))
        and all(bool(jnp.all(jnp.isfinite(values))) for values in q_values_by_state)
    )

    return {
        "seed": seed,
        "start_state": seed % n_states,
        "first_action": int(first_action),
        "greedy_policy": greedy_policy,
        "optimal_policy": optimal_policy,
        "final_average_reward": float(agent_state.average_reward),
        "average_reward_abs_error": average_reward_abs_error,
        "final_reward_mean": final_reward_mean,
        "final_policy_match_rate": final_policy_match_rate,
        "tail_td_error_mse": tail_td_error_mse,
        "q_values_by_state": [
            [float(value) for value in values] for values in q_values_by_state
        ],
        "visited_state_counts": [
            sum(1 for state in visited_states[-final_window:] if state == idx)
            for idx in range(n_states)
        ],
        "finite": finite,
        "passed": bool(
            finite
            and greedy_policy == optimal_policy
            and average_reward_abs_error <= 0.02
            and final_reward_mean >= 0.99
            and final_policy_match_rate >= 0.99
            and tail_td_error_mse <= 0.002
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    q_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> dict[str, Any]:
    """Run the multi-seed two-state continuing-control benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
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
    final_rewards = [float(row["final_reward_mean"]) for row in per_seed]
    policy_rates = [float(row["final_policy_match_rate"]) for row in per_seed]
    tail_mses = [float(row["tail_td_error_mse"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.multistate_continuing_control_benchmark.v1",
        "claim_scope": "two_state_average_reward_differential_sarsa_control",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "q_step_size": q_step_size,
            "average_reward_step_size": average_reward_step_size,
            "trace_decay": trace_decay,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
        },
        "target": {
            "optimal_policy": [1, 0],
            "optimal_average_reward": 1.0,
        },
        "baseline": {
            "uniform_random_expected_reward": 0.5,
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(average_reward_errors)
            / len(average_reward_errors),
            "stderr_average_reward_abs_error": _stderr(average_reward_errors),
            "mean_final_reward": sum(final_rewards) / len(final_rewards),
            "stderr_final_reward": _stderr(final_rewards),
            "mean_final_policy_match_rate": sum(policy_rates) / len(policy_rates),
            "stderr_final_policy_match_rate": _stderr(policy_rates),
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
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--q-step-size", type=float, default=0.05)
    parser.add_argument("--average-reward-step-size", type=float, default=0.01)
    parser.add_argument("--trace-decay", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.5)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--epsilon-decay-steps", type=int, default=4_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
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

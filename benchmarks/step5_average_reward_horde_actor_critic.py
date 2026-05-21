#!/usr/bin/env python3
"""Seeded nonlinear average-reward Horde actor-critic benchmark."""

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
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _one_hot(state: int) -> jnp.ndarray:
    return jnp.eye(2, dtype=jnp.float32)[state]


def _transition(state: int, action: int) -> tuple[float, int]:
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
    actor_step_size: float,
    critic_step_size: float,
    average_reward_step_size: float,
    hidden_size: int,
    epsilon: float,
) -> dict[str, Any]:
    """Run one nonlinear average-reward Horde actor-critic seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    optimal_policy = [1, 0]
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(hidden_size,),
            actor_step_size=actor_step_size,
            critic_step_size=critic_step_size,
            average_reward_step_size=average_reward_step_size,
            epsilon=epsilon,
        )
    )
    env_state = seed % 2
    state = agent.init(2, jr.key(seed))
    state, first_action = agent.start(state, _one_hot(env_state))

    rewards: list[float] = []
    policy_matches: list[float] = []
    td_errors: list[float] = []
    for _step in range(steps):
        action = int(state.last_action)
        reward, next_env_state = _transition(env_state, action)
        result = agent.update(
            state,
            jnp.array(reward, dtype=jnp.float32),
            _one_hot(next_env_state),
        )
        state = result.state
        rewards.append(reward)
        policy_matches.append(float(action == optimal_policy[env_state]))
        td_errors.append(float(result.td_error))
        env_state = next_env_state

    policy0 = agent.policy(state, _one_hot(0))
    policy1 = agent.policy(state, _one_hot(1))
    greedy_policy = [int(jnp.argmax(policy0)), int(jnp.argmax(policy1))]
    final_rewards = rewards[-final_window:]
    final_matches = policy_matches[-final_window:]
    final_td_errors = td_errors[-final_window:]
    final_reward_mean = sum(final_rewards) / len(final_rewards)
    final_policy_match_rate = sum(final_matches) / len(final_matches)
    average_reward = float(state.critic_state.average_rewards[0])
    average_reward_abs_error = abs(average_reward - 1.0)
    tail_td_error_mse = sum(error * error for error in final_td_errors) / len(
        final_td_errors
    )
    finite = bool(
        math.isfinite(average_reward)
        and all(math.isfinite(error) for error in final_td_errors)
        and bool(jnp.all(jnp.isfinite(policy0)))
        and bool(jnp.all(jnp.isfinite(policy1)))
    )

    return {
        "seed": seed,
        "start_state": seed % 2,
        "first_action": int(first_action),
        "greedy_policy": greedy_policy,
        "optimal_policy": optimal_policy,
        "policy_state0": [float(value) for value in policy0],
        "policy_state1": [float(value) for value in policy1],
        "final_average_reward": average_reward,
        "average_reward_abs_error": average_reward_abs_error,
        "final_reward_mean": final_reward_mean,
        "final_policy_match_rate": final_policy_match_rate,
        "tail_td_error_mse": tail_td_error_mse,
        "finite": finite,
        "passed": bool(
            finite
            and greedy_policy == optimal_policy
            and average_reward_abs_error <= 0.04
            and final_reward_mean >= 0.97
            and final_policy_match_rate >= 0.97
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    actor_step_size: float,
    critic_step_size: float,
    average_reward_step_size: float,
    hidden_size: int,
    epsilon: float,
) -> dict[str, Any]:
    """Run the multi-seed nonlinear average-reward actor-critic benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            actor_step_size=actor_step_size,
            critic_step_size=critic_step_size,
            average_reward_step_size=average_reward_step_size,
            hidden_size=hidden_size,
            epsilon=epsilon,
        )
        for seed in range(seeds)
    ]
    average_reward_errors = [
        float(row["average_reward_abs_error"]) for row in per_seed
    ]
    final_rewards = [float(row["final_reward_mean"]) for row in per_seed]
    policy_rates = [float(row["final_policy_match_rate"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.average_reward_horde_actor_critic_benchmark.v1",
        "claim_scope": "nonlinear_shared_feature_average_reward_actor_critic",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "actor_step_size": actor_step_size,
            "critic_step_size": critic_step_size,
            "average_reward_step_size": average_reward_step_size,
            "hidden_size": hidden_size,
            "epsilon": epsilon,
        },
        "target": {
            "optimal_policy": [1, 0],
            "optimal_average_reward": 1.0,
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(average_reward_errors)
            / len(average_reward_errors),
            "stderr_average_reward_abs_error": _stderr(average_reward_errors),
            "mean_final_reward": sum(final_rewards) / len(final_rewards),
            "stderr_final_reward": _stderr(final_rewards),
            "mean_final_policy_match_rate": sum(policy_rates) / len(policy_rates),
            "stderr_final_policy_match_rate": _stderr(policy_rates),
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
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--actor-step-size", type=float, default=0.03)
    parser.add_argument("--critic-step-size", type=float, default=0.005)
    parser.add_argument("--average-reward-step-size", type=float, default=0.005)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        actor_step_size=args.actor_step_size,
        critic_step_size=args.critic_step_size,
        average_reward_step_size=args.average_reward_step_size,
        hidden_size=args.hidden_size,
        epsilon=args.epsilon,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Seeded positive-control benchmark for Horde-backed actor-critic control."""

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
from jax import lax

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.horde import HordeLearner  # noqa: E402
from alberta_framework.core.horde_actor_critic import (  # noqa: E402
    HordeActorCriticAgent,
    HordeActorCriticConfig,
)
from alberta_framework.core.optimizers import LMS, ObGDBounding  # noqa: E402
from alberta_framework.core.types import (  # noqa: E402
    DemonType,
    GVFSpec,
    create_horde_spec,
)


def _stderr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _make_agent(
    *,
    actor_step_size: float,
    critic_step_size: float,
    actor_lamda: float,
    hidden_size: int,
    temperature: float,
) -> HordeActorCriticAgent:
    demons = (
        GVFSpec(
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        ),
    )
    critic = HordeLearner(
        create_horde_spec(demons),
        hidden_sizes=(hidden_size,),
        optimizer=LMS(step_size=critic_step_size),
        sparsity=0.0,
        use_layer_norm=False,
    )
    return HordeActorCriticAgent(
        HordeActorCriticConfig(
            n_actions=2,
            actor_step_size=actor_step_size,
            actor_lamda=actor_lamda,
            temperature=temperature,
        ),
        critic=critic,
        actor_bounder=ObGDBounding(kappa=2.0),
    )


def run_seed(
    seed: int,
    *,
    steps: int,
    final_window: int,
    actor_step_size: float,
    critic_step_size: float,
    actor_lamda: float,
    hidden_size: int,
    temperature: float,
) -> dict[str, Any]:
    """Run one one-state control seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")

    agent = _make_agent(
        actor_step_size=actor_step_size,
        critic_step_size=critic_step_size,
        actor_lamda=actor_lamda,
        hidden_size=hidden_size,
        temperature=temperature,
    )
    observation = jnp.ones(1, dtype=jnp.float32)
    state = agent.init(1, jr.key(seed))
    state, action, _policy = agent.start(state, observation)

    def _step(
        carry: tuple[Any, jnp.ndarray],
        _idx: jnp.ndarray,
    ) -> tuple[tuple[Any, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        state_t, action_t = carry
        reward_t = jnp.asarray(action_t == 1, dtype=jnp.float32)
        result = agent.update(
            state_t,
            reward=reward_t,
            observation=observation,
            discount=jnp.array(0.0, dtype=jnp.float32),
        )
        return (
            (result.state, result.action),
            (
                reward_t,
                result.policy[1],
                result.td_error,
                result.bound_metric,
            ),
        )

    (state, _action), (rewards, optimal_probs, td_errors, bound_metrics) = lax.scan(
        _step,
        (state, action),
        jnp.arange(steps, dtype=jnp.int32),
    )
    td_errors.block_until_ready()

    final_policy = agent.policy(state, observation)
    final_reward_rate = float(jnp.mean(rewards[-final_window:]))
    final_optimal_prob = float(jnp.mean(optimal_probs[-final_window:]))
    tail_td_mse = float(jnp.mean(td_errors[-final_window:] ** 2))
    finite = bool(
        jnp.all(jnp.isfinite(final_policy))
        & jnp.all(jnp.isfinite(state.actor_weights))
        & jnp.all(jnp.isfinite(state.actor_bias))
        & jnp.all(jnp.isfinite(state.critic_state.head_params.weights[0]))
    )

    return {
        "seed": seed,
        "final_policy": [float(value) for value in final_policy],
        "final_window_reward_rate": final_reward_rate,
        "final_window_optimal_action_probability": final_optimal_prob,
        "tail_td_mse": tail_td_mse,
        "min_bound_metric": float(jnp.min(bound_metrics)),
        "finite": finite,
        "passed": bool(
            finite
            and final_reward_rate >= 0.95
            and final_optimal_prob >= 0.95
            and float(final_policy[1]) >= 0.95
            and tail_td_mse <= 0.08
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    actor_step_size: float,
    critic_step_size: float,
    actor_lamda: float,
    hidden_size: int,
    temperature: float,
) -> dict[str, Any]:
    """Run the multi-seed benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            actor_step_size=actor_step_size,
            critic_step_size=critic_step_size,
            actor_lamda=actor_lamda,
            hidden_size=hidden_size,
            temperature=temperature,
        )
        for seed in range(seeds)
    ]
    reward_rates = [float(row["final_window_reward_rate"]) for row in per_seed]
    optimal_probs = [
        float(row["final_window_optimal_action_probability"]) for row in per_seed
    ]
    tail_mses = [float(row["tail_td_mse"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step4.horde_actor_critic_control_benchmark.v1",
        "claim_scope": "horde_actor_critic_positive_control",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "actor_step_size": actor_step_size,
            "critic_step_size": critic_step_size,
            "actor_lamda": actor_lamda,
            "hidden_size": hidden_size,
            "temperature": temperature,
        },
        "aggregate": {
            "mean_final_reward_rate": sum(reward_rates) / len(reward_rates),
            "stderr_final_reward_rate": _stderr(reward_rates),
            "mean_final_optimal_action_probability": sum(optimal_probs)
            / len(optimal_probs),
            "stderr_final_optimal_action_probability": _stderr(optimal_probs),
            "mean_tail_td_mse": sum(tail_mses) / len(tail_mses),
            "stderr_tail_td_mse": _stderr(tail_mses),
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
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--actor-step-size", type=float, default=0.03)
    parser.add_argument("--critic-step-size", type=float, default=0.05)
    parser.add_argument("--actor-lamda", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
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
        actor_lamda=args.actor_lamda,
        hidden_size=args.hidden_size,
        temperature=args.temperature,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

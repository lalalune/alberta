"""Seeded Step 7 Dyna sample-efficiency benchmark.

This benchmark uses the smallest continuing-control problem that can expose a
planning benefit: one state, two actions, average reward, and reward 1 only for
action 1.  Step 6 differential SARSA learns only from real actions.  Step 7
learns the same real stream plus bounded model-generated backups from its
one-step world model.

The benchmark is intentionally tiny but not a unit test.  Its first JAX compile
can be slow because it scans agent, model, and planning updates together.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.steps import (
    Step6DifferentialSARSAConfig,
    Step7DynaConfig,
    Step8WorldModelConfig,
    init_step6_state,
    init_step7_state,
    make_step6_differential_sarsa_agent,
    make_step7_components,
    step6_update,
    step7_update,
)


@dataclass(frozen=True)
class Step7SampleEfficiencyConfig:
    """Benchmark configuration."""

    seeds: int = 5
    steps: int = 30
    final_window: int = 5
    planning_steps: int = 8
    planning_warmup_steps: int = 1
    q_step_size: float = 0.04
    average_reward_step_size: float = 0.01
    model_step_size: float = 0.08
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30

    def __post_init__(self) -> None:
        """Validate scalar settings."""
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if not 1 <= self.final_window <= self.steps:
            raise ValueError("final_window must be in [1, steps]")
        if self.planning_steps < 0:
            raise ValueError("planning_steps must be non-negative")


@dataclass(frozen=True)
class Step7SampleEfficiencySummary:
    """JSON-serializable benchmark summary."""

    schema: str
    claim_scope: str
    config: dict[str, Any]
    elapsed_s: float
    aggregate: dict[str, float | int | bool]
    per_seed: list[dict[str, float | int]]


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _make_configs(
    config: Step7SampleEfficiencyConfig,
) -> tuple[Step6DifferentialSARSAConfig, Step7DynaConfig]:
    control = Step6DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=config.q_step_size,
        average_reward_step_size=config.average_reward_step_size,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_steps=config.epsilon_decay_steps,
    )
    dyna = Step7DynaConfig(
        control=control,
        world_model=Step8WorldModelConfig(
            observation_dim=1,
            n_actions=2,
            hidden_sizes=(),
            step_size=config.model_step_size,
            sparsity=0.0,
            predict_delta=False,
        ),
        planning_steps=config.planning_steps,
        planning_warmup_steps=config.planning_warmup_steps,
        planning_strategy="reward",
    )
    return control, dyna


def run_step7_sample_efficiency_benchmark(
    config: Step7SampleEfficiencyConfig,
) -> Step7SampleEfficiencySummary:
    """Run the seeded real-only versus Step 7 Dyna benchmark."""
    start = time.time()
    control_cfg, dyna_cfg = _make_configs(config)
    real_agent = make_step6_differential_sarsa_agent(control_cfg)
    dyna_agent, world_model = make_step7_components(dyna_cfg)
    observation = jnp.array([1.0], dtype=jnp.float32)

    def run_real_only(seed: Array) -> tuple[Array, Array, Array]:
        state = init_step6_state(
            real_agent,
            feature_dim=1,
            key=jr.key(seed),
            initial_features=observation,
        )

        def scan_step(carry: Any, _: Array) -> tuple[Any, Array]:
            reward = jnp.asarray(carry.last_action == 1, dtype=jnp.float32)
            result = step6_update(real_agent, carry, reward, observation)
            return result.state, reward

        final_state, rewards = jax.lax.scan(
            scan_step,
            state,
            jnp.arange(config.steps, dtype=jnp.int32),
        )
        q_values = real_agent.q_values(final_state, observation)
        return (
            jnp.mean(rewards[-config.final_window :]),
            q_values[1] - q_values[0],
            final_state.average_reward,
        )

    def run_dyna(seed: Array) -> tuple[Array, Array, Array, Array]:
        state = init_step7_state(
            dyna_agent,
            world_model,
            key=jr.key(seed),
            initial_observation=observation,
        )

        def scan_step(carry: Any, _: Array) -> tuple[Any, tuple[Array, Array]]:
            reward = jnp.asarray(carry.control_state.last_action == 1, dtype=jnp.float32)
            result = step7_update(dyna_cfg, dyna_agent, world_model, carry, reward, observation)
            return result.state, (reward, jnp.sum(result.planning_accepted))

        final_state, (rewards, accepted) = jax.lax.scan(
            scan_step,
            state,
            jnp.arange(config.steps, dtype=jnp.int32),
        )
        q_values = dyna_agent.q_values(final_state.control_state, observation)
        return (
            jnp.mean(rewards[-config.final_window :]),
            q_values[1] - q_values[0],
            final_state.control_state.average_reward,
            jnp.sum(accepted),
        )

    seeds = jnp.arange(config.seeds, dtype=jnp.int32)
    real_rewards, real_q_gaps, real_average_rewards = jax.vmap(run_real_only)(seeds)
    dyna_rewards, dyna_q_gaps, dyna_average_rewards, dyna_planning_accepted = jax.vmap(
        run_dyna
    )(seeds)
    real_rewards.block_until_ready()

    per_seed = [
        {
            "seed": int(seed),
            "real_only_final_window_reward": float(real_rewards[idx]),
            "dyna_final_window_reward": float(dyna_rewards[idx]),
            "real_only_q_gap": float(real_q_gaps[idx]),
            "dyna_q_gap": float(dyna_q_gaps[idx]),
            "real_only_average_reward": float(real_average_rewards[idx]),
            "dyna_average_reward": float(dyna_average_rewards[idx]),
            "dyna_planning_accepted": int(dyna_planning_accepted[idx]),
        }
        for idx, seed in enumerate(range(config.seeds))
    ]

    reward_improvements = [
        row["dyna_final_window_reward"] - row["real_only_final_window_reward"]
        for row in per_seed
    ]
    q_gap_improvements = [
        row["dyna_q_gap"] - row["real_only_q_gap"] for row in per_seed
    ]
    dyna_rewards_list = [row["dyna_final_window_reward"] for row in per_seed]
    real_rewards_list = [row["real_only_final_window_reward"] for row in per_seed]

    aggregate: dict[str, float | int | bool] = {
        "mean_real_only_final_window_reward": sum(real_rewards_list) / len(per_seed),
        "mean_dyna_final_window_reward": sum(dyna_rewards_list) / len(per_seed),
        "mean_reward_improvement": sum(reward_improvements) / len(per_seed),
        "stderr_reward_improvement": _stderr(reward_improvements),
        "mean_q_gap_improvement": sum(q_gap_improvements) / len(per_seed),
        "reward_win_count": sum(value > 0.0 for value in reward_improvements),
        "q_gap_win_count": sum(value > 0.0 for value in q_gap_improvements),
        "mean_dyna_planning_accepted": sum(
            row["dyna_planning_accepted"] for row in per_seed
        )
        / len(per_seed),
    }
    aggregate["passed"] = bool(
        aggregate["mean_reward_improvement"] >= 0.04
        and aggregate["mean_q_gap_improvement"] >= 1.0
        and aggregate["q_gap_win_count"] == config.seeds
    )

    return Step7SampleEfficiencySummary(
        schema="alberta.step7.dyna_sample_efficiency.v1",
        claim_scope="one_state_average_reward_dyna_sample_efficiency",
        config=asdict(config),
        elapsed_s=time.time() - start,
        aggregate=aggregate,
        per_seed=per_seed,
    )


def main() -> None:
    """Run the benchmark and write a JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--planning-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/step7_dyna/results.json"))
    args = parser.parse_args()
    config = Step7SampleEfficiencyConfig(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        planning_steps=args.planning_steps,
        epsilon_decay_steps=args.steps,
    )
    summary = run_step7_sample_efficiency_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

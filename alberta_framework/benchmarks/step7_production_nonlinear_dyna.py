"""Production JAX Step 7 Dyna benchmark with a nonlinear world model.

This benchmark complements the fast NumPy nonlinear-feature diagnostic by
exercising the actual promoted Step 7 JAX facade with a non-empty hidden-layer
world model.  The control problem is intentionally tiny so that the benchmark
can run in CI-like local verification: one continuing state, two actions, and
average reward 1 for action 1.
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
class Step7ProductionNonlinearConfig:
    """Configuration for the production nonlinear Step 7 benchmark."""

    seeds: int = 10
    steps: int = 30
    final_window: int = 5
    planning_steps: int = 8
    planning_warmup_steps: int = 1
    q_step_size: float = 0.04
    average_reward_step_size: float = 0.01
    model_step_size: float = 0.08
    model_hidden_size: int = 8
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05

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
        if self.model_hidden_size < 1:
            raise ValueError("model_hidden_size must be positive")


@dataclass(frozen=True)
class Step7ProductionNonlinearSummary:
    """JSON-serializable benchmark summary."""

    schema: str
    claim_scope: str
    config: dict[str, Any]
    elapsed_s: float
    aggregate: dict[str, float | int | bool | list[int]]
    per_seed: list[dict[str, float | int]]


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _make_configs(
    config: Step7ProductionNonlinearConfig,
) -> tuple[Step6DifferentialSARSAConfig, Step7DynaConfig]:
    control = Step6DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=config.q_step_size,
        average_reward_step_size=config.average_reward_step_size,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_steps=config.steps,
    )
    dyna = Step7DynaConfig(
        control=control,
        world_model=Step8WorldModelConfig(
            observation_dim=1,
            n_actions=2,
            hidden_sizes=(config.model_hidden_size,),
            step_size=config.model_step_size,
            sparsity=0.0,
            predict_delta=False,
            use_layer_norm=False,
        ),
        planning_steps=config.planning_steps,
        planning_warmup_steps=config.planning_warmup_steps,
        planning_strategy="reward",
    )
    return control, dyna


def run_step7_production_nonlinear_benchmark(
    config: Step7ProductionNonlinearConfig,
) -> Step7ProductionNonlinearSummary:
    """Run the production nonlinear Step 7 benchmark."""
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
            jnp.sum(rewards),
            q_values[1] - q_values[0],
        )

    def run_dyna(seed: Array) -> tuple[Array, Array, Array, Array, Array]:
        state = init_step7_state(
            dyna_agent,
            world_model,
            key=jr.key(seed),
            initial_observation=observation,
        )

        def scan_step(carry: Any, _: Array) -> tuple[Any, tuple[Array, Array]]:
            reward = jnp.asarray(
                carry.control_state.last_action == 1,
                dtype=jnp.float32,
            )
            result = step7_update(
                dyna_cfg,
                dyna_agent,
                world_model,
                carry,
                reward,
                observation,
            )
            return result.state, (reward, jnp.sum(result.planning_accepted))

        final_state, (rewards, accepted) = jax.lax.scan(
            scan_step,
            state,
            jnp.arange(config.steps, dtype=jnp.int32),
        )
        q_values = dyna_agent.q_values(final_state.control_state, observation)
        return (
            jnp.mean(rewards[-config.final_window :]),
            jnp.sum(rewards),
            q_values[1] - q_values[0],
            jnp.sum(accepted),
            jnp.max(jnp.abs(final_state.memory_utilities)),
        )

    seeds = jnp.arange(config.seeds, dtype=jnp.int32)
    real_rewards, real_cumulative_rewards, real_q_gaps = jax.vmap(run_real_only)(
        seeds
    )
    (
        dyna_rewards,
        dyna_cumulative_rewards,
        dyna_q_gaps,
        dyna_planning_accepted,
        dyna_max_memory_utilities,
    ) = jax.vmap(run_dyna)(seeds)
    dyna_rewards.block_until_ready()

    per_seed = [
        {
            "seed": seed,
            "real_only_final_window_reward": float(real_rewards[idx]),
            "dyna_final_window_reward": float(dyna_rewards[idx]),
            "real_only_cumulative_reward": float(real_cumulative_rewards[idx]),
            "dyna_cumulative_reward": float(dyna_cumulative_rewards[idx]),
            "real_only_q_gap": float(real_q_gaps[idx]),
            "dyna_q_gap": float(dyna_q_gaps[idx]),
            "dyna_planning_accepted": int(dyna_planning_accepted[idx]),
            "dyna_max_memory_utility": float(dyna_max_memory_utilities[idx]),
        }
        for idx, seed in enumerate(range(config.seeds))
    ]

    final_reward_improvements = [
        row["dyna_final_window_reward"] - row["real_only_final_window_reward"]
        for row in per_seed
    ]
    cumulative_reward_improvements = [
        row["dyna_cumulative_reward"] - row["real_only_cumulative_reward"]
        for row in per_seed
    ]
    q_gap_improvements = [
        row["dyna_q_gap"] - row["real_only_q_gap"] for row in per_seed
    ]
    hidden_sizes = list(dyna_cfg.world_model.hidden_sizes)
    aggregate: dict[str, float | int | bool | list[int]] = {
        "model_hidden_sizes": hidden_sizes,
        "uses_production_step7_jax_facade": True,
        "mean_real_only_final_window_reward": sum(
            row["real_only_final_window_reward"] for row in per_seed
        )
        / len(per_seed),
        "mean_dyna_final_window_reward": sum(
            row["dyna_final_window_reward"] for row in per_seed
        )
        / len(per_seed),
        "mean_final_window_improvement": sum(final_reward_improvements)
        / len(per_seed),
        "stderr_final_window_improvement": _stderr(final_reward_improvements),
        "final_window_win_count": sum(
            value > 0.0 for value in final_reward_improvements
        ),
        "mean_cumulative_reward_improvement": sum(cumulative_reward_improvements)
        / len(per_seed),
        "cumulative_reward_win_count": sum(
            value > 0.0 for value in cumulative_reward_improvements
        ),
        "mean_q_gap_improvement": sum(q_gap_improvements) / len(per_seed),
        "q_gap_win_count": sum(value > 0.0 for value in q_gap_improvements),
        "mean_dyna_planning_accepted": sum(
            row["dyna_planning_accepted"] for row in per_seed
        )
        / len(per_seed),
        "mean_dyna_max_memory_utility": sum(
            row["dyna_max_memory_utility"] for row in per_seed
        )
        / len(per_seed),
    }
    aggregate["passed"] = bool(
        hidden_sizes
        and aggregate["mean_final_window_improvement"] >= 0.07
        and aggregate["mean_q_gap_improvement"] >= 2.0
        and aggregate["q_gap_win_count"] >= 8
        and aggregate["mean_dyna_planning_accepted"] > 0
    )

    return Step7ProductionNonlinearSummary(
        schema="alberta.step7.production_nonlinear_dyna.v1",
        claim_scope="production_jax_hidden_world_model_dyna_sample_efficiency",
        config=asdict(config),
        elapsed_s=time.time() - start,
        aggregate=aggregate,
        per_seed=per_seed,
    )


def main() -> None:
    """Run the benchmark and write a JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--final-window", type=int, default=5)
    parser.add_argument("--planning-steps", type=int, default=8)
    parser.add_argument("--model-hidden-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step7_production_nonlinear_dyna/results.json"),
    )
    args = parser.parse_args()
    config = Step7ProductionNonlinearConfig(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        planning_steps=args.planning_steps,
        model_hidden_size=args.model_hidden_size,
    )
    summary = run_step7_production_nonlinear_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

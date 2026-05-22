"""Seeded Step 8 world-model prediction benchmark.

The benchmark uses a continuing two-dimensional action-conditioned process with
known linear dynamics.  It verifies that the promoted Step 8 world-model facade
learns reward and next-observation predictions online, and that an explicit
ensemble disagreement surface collapses as independently seeded models see the
same stream.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps import (
    Step8WorldModelConfig,
    init_step8_state,
    make_step8_world_model,
    run_step8_scan,
    step8_ensemble_predict,
)


@dataclass(frozen=True)
class Step8PredictionBenchmarkConfig:
    """Configuration for the Step 8 prediction benchmark."""

    seeds: int = 10
    steps: int = 800
    final_window: int = 100
    step_size: float = 0.03

    def __post_init__(self) -> None:
        """Validate scalar settings."""
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if not 1 <= self.final_window <= self.steps:
            raise ValueError("final_window must be in [1, steps]")


@dataclass(frozen=True)
class Step8PredictionBenchmarkSummary:
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


def _transition_stream(
    steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    time_index = jnp.arange(steps, dtype=jnp.float32)
    actions = jnp.arange(steps, dtype=jnp.int32) % 2
    observations = jnp.stack(
        [
            jnp.sin(0.03 * time_index),
            jnp.cos(0.05 * time_index),
        ],
        axis=1,
    )
    action_sign = 2.0 * actions.astype(jnp.float32) - 1.0
    next_observations = jnp.stack(
        [
            0.80 * observations[:, 0] + 0.20 * action_sign,
            0.75 * observations[:, 1] - 0.10 * action_sign,
        ],
        axis=1,
    )
    rewards = (
        0.50 * next_observations[:, 0]
        - 0.25 * next_observations[:, 1]
        + 0.10 * action_sign
    )
    return observations, actions, rewards, next_observations


def run_step8_prediction_benchmark(
    config: Step8PredictionBenchmarkConfig,
) -> Step8PredictionBenchmarkSummary:
    """Run the seeded Step 8 world-model prediction benchmark."""
    start = time.time()
    model_config = Step8WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=config.step_size,
        sparsity=0.0,
        predict_delta=False,
    )
    model = make_step8_world_model(model_config)
    observations, actions, rewards, next_observations = _transition_stream(
        config.steps,
    )

    per_seed: list[dict[str, float | int]] = []
    initial_states = []
    final_states = []
    for seed in range(config.seeds):
        state = init_step8_state(model, key=jr.key(seed))
        initial_states.append(state)
        result = run_step8_scan(
            model,
            state,
            observations,
            actions,
            rewards,
            next_observations,
        )
        result.reward_errors.block_until_ready()
        final_states.append(result.state)

        model_error = result.reward_errors**2 + jnp.mean(
            result.next_observation_errors**2,
            axis=1,
        )
        zero_baseline_error = rewards**2 + jnp.mean(next_observations**2, axis=1)
        final_model_mse = float(jnp.mean(model_error[-config.final_window :]))
        final_baseline_mse = float(
            jnp.mean(zero_baseline_error[-config.final_window :])
        )
        per_seed.append(
            {
                "seed": seed,
                "final_window_model_mse": final_model_mse,
                "final_window_zero_baseline_mse": final_baseline_mse,
                "final_window_mse_reduction": final_baseline_mse - final_model_mse,
                "final_window_relative_reduction": 1.0
                - final_model_mse / final_baseline_mse,
            }
        )

    initial_disagreement = step8_ensemble_predict(
        model,
        initial_states,
        observations[0],
        actions[0],
    )
    final_disagreement = step8_ensemble_predict(
        model,
        final_states,
        observations[-1],
        actions[-1],
    )
    reductions = [row["final_window_relative_reduction"] for row in per_seed]
    model_mses = [row["final_window_model_mse"] for row in per_seed]
    baseline_mses = [row["final_window_zero_baseline_mse"] for row in per_seed]
    aggregate: dict[str, float | int | bool] = {
        "n_seeds": config.seeds,
        "mean_final_window_model_mse": sum(model_mses) / len(model_mses),
        "mean_final_window_zero_baseline_mse": sum(baseline_mses)
        / len(baseline_mses),
        "mean_final_window_relative_reduction": sum(reductions) / len(reductions),
        "stderr_final_window_relative_reduction": _stderr(reductions),
        "model_beats_baseline_count": sum(
            row["final_window_mse_reduction"] > 0.0 for row in per_seed
        ),
        "initial_ensemble_disagreement": float(
            initial_disagreement.total_disagreement
        ),
        "final_ensemble_disagreement": float(final_disagreement.total_disagreement),
    }
    aggregate["ensemble_disagreement_reduction"] = (
        aggregate["initial_ensemble_disagreement"]
        - aggregate["final_ensemble_disagreement"]
    )
    aggregate["passed"] = bool(
        aggregate["n_seeds"] >= 10
        and aggregate["model_beats_baseline_count"] == aggregate["n_seeds"]
        and aggregate["mean_final_window_relative_reduction"] >= 0.99
        and aggregate["ensemble_disagreement_reduction"] > 0.1
        and aggregate["final_ensemble_disagreement"] < 1.0e-3
    )

    return Step8PredictionBenchmarkSummary(
        schema="alberta.step8.world_model_prediction.v1",
        claim_scope="one_step_world_model_prediction_and_ensemble_uncertainty",
        config=asdict(config),
        elapsed_s=time.time() - start,
        aggregate=aggregate,
        per_seed=per_seed,
    )


def main() -> None:
    """Run the benchmark and write a JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step8_world_model_prediction/results.json"),
    )
    args = parser.parse_args()
    config = Step8PredictionBenchmarkConfig(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
    )
    summary = run_step8_prediction_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seeded nonlinear shared-trunk average-reward Horde benchmark."""

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
    AverageRewardHordeLearner,
    run_average_reward_horde_from_arrays,
)


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
    tail_window: int,
    step_size: float,
    average_reward_step_size: float,
    hidden_size: int,
) -> dict[str, Any]:
    """Run one shared-trunk average-reward Horde seed."""
    if tail_window < 1 or tail_window > steps:
        raise ValueError("tail_window must be in [1, steps]")
    states = (jnp.arange(steps, dtype=jnp.int32) + seed) % 3
    next_states = (states + 1) % 3
    observations = jnp.eye(3, dtype=jnp.float32)[states]
    next_observations = jnp.eye(3, dtype=jnp.float32)[next_states]
    cumulants = jnp.stack(
        [
            jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)[states],
            jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)[states],
        ],
        axis=1,
    )
    learner = AverageRewardHordeLearner(
        n_demons=2,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        average_reward_step_size=average_reward_step_size,
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(3, jr.key(seed))
    result = run_average_reward_horde_from_arrays(
        learner,
        state,
        observations,
        cumulants,
        next_observations,
    )
    result.td_errors.block_until_ready()
    average_reward_errors = jnp.abs(
        result.state.average_rewards - jnp.ones((2,), dtype=jnp.float32)
    )
    tail_td_error_mse = float(jnp.nanmean(result.td_errors[-tail_window:] ** 2))
    finite = bool(
        jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(jnp.isfinite(result.state.average_rewards))
        & jnp.all(jnp.isfinite(result.td_errors))
    )

    return {
        "seed": seed,
        "final_average_rewards": [
            float(value) for value in result.state.average_rewards
        ],
        "mean_average_reward_abs_error": float(jnp.mean(average_reward_errors)),
        "max_average_reward_abs_error": float(jnp.max(average_reward_errors)),
        "tail_td_error_mse": tail_td_error_mse,
        "finite": finite,
        "passed": bool(
            finite
            and float(jnp.max(average_reward_errors)) <= 0.03
            and tail_td_error_mse <= 0.005
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    tail_window: int,
    step_size: float,
    average_reward_step_size: float,
    hidden_size: int,
) -> dict[str, Any]:
    """Run the multi-seed nonlinear average-reward Horde benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            tail_window=tail_window,
            step_size=step_size,
            average_reward_step_size=average_reward_step_size,
            hidden_size=hidden_size,
        )
        for seed in range(seeds)
    ]
    mean_errors = [float(row["mean_average_reward_abs_error"]) for row in per_seed]
    max_errors = [float(row["max_average_reward_abs_error"]) for row in per_seed]
    tail_mses = [float(row["tail_td_error_mse"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.average_reward_horde_benchmark.v1",
        "claim_scope": "nonlinear_shared_trunk_average_reward_horde_prediction",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "tail_window": tail_window,
            "step_size": step_size,
            "average_reward_step_size": average_reward_step_size,
            "hidden_size": hidden_size,
            "n_demons": 2,
        },
        "target": {
            "true_average_rewards": [1.0, 1.0],
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(mean_errors) / len(mean_errors),
            "stderr_average_reward_abs_error": _stderr(mean_errors),
            "max_average_reward_abs_error": max(max_errors),
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
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--average-reward-step-size", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        tail_window=args.tail_window,
        step_size=args.step_size,
        average_reward_step_size=args.average_reward_step_size,
        hidden_size=args.hidden_size,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

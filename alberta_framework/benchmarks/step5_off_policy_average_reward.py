#!/usr/bin/env python3
"""Seeded off-policy average-reward GTD benchmark."""

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
    DifferentialGTDConfig,
    DifferentialGTDLearner,
    run_differential_gtd_from_arrays,
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
    final_window: int,
    behavior_target_action_prob: float,
    value_step_size: float,
    secondary_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run one off-policy average-reward prediction seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    if not 0.0 < behavior_target_action_prob <= 1.0:
        raise ValueError("behavior_target_action_prob must be in (0, 1]")

    key = jr.key(seed)
    target_action = jr.bernoulli(key, behavior_target_action_prob, (steps,))
    rewards = target_action.astype(jnp.float32)
    rhos = jnp.where(
        target_action,
        1.0 / behavior_target_action_prob,
        0.0,
    ).astype(jnp.float32)
    observations = jnp.ones((steps, 1), dtype=jnp.float32)
    next_observations = jnp.ones((steps, 1), dtype=jnp.float32)
    learner = DifferentialGTDLearner(
        DifferentialGTDConfig(
            value_step_size=value_step_size,
            secondary_step_size=secondary_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            ratio_clip=ratio_clip,
        )
    )
    state = learner.init(1)
    result = run_differential_gtd_from_arrays(
        learner,
        state,
        observations,
        rewards,
        next_observations,
        rhos,
    )
    result.td_errors.block_until_ready()

    final_average_reward = float(result.state.average_reward)
    target_average_reward = 1.0
    behavior_average_reward = behavior_target_action_prob
    average_reward_abs_error = abs(final_average_reward - target_average_reward)
    weighted_tail_td_error_mse = float(
        jnp.sum((result.rho_clipped[-final_window:] * result.td_errors[-final_window:]) ** 2)
        / jnp.maximum(1.0, jnp.sum(result.rho_clipped[-final_window:] > 0.0))
    )
    final_target_action_rate = float(jnp.mean(rewards[-final_window:]))
    finite = bool(
        jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.isfinite(result.state.average_reward)
    )

    return {
        "seed": seed,
        "behavior_target_action_prob": behavior_target_action_prob,
        "final_average_reward": final_average_reward,
        "target_average_reward": target_average_reward,
        "behavior_average_reward": behavior_average_reward,
        "average_reward_abs_error": average_reward_abs_error,
        "final_target_action_rate": final_target_action_rate,
        "weighted_tail_td_error_mse": weighted_tail_td_error_mse,
        "finite": finite,
        "passed": bool(
            finite
            and average_reward_abs_error <= 0.02
            and final_average_reward >= behavior_average_reward + 0.4
            and weighted_tail_td_error_mse <= 0.002
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    behavior_target_action_prob: float,
    value_step_size: float,
    secondary_step_size: float,
    average_reward_step_size: float,
    trace_decay: float,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run the multi-seed off-policy benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            behavior_target_action_prob=behavior_target_action_prob,
            value_step_size=value_step_size,
            secondary_step_size=secondary_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=trace_decay,
            ratio_clip=ratio_clip,
        )
        for seed in range(seeds)
    ]
    average_reward_errors = [
        float(row["average_reward_abs_error"]) for row in per_seed
    ]
    final_average_rewards = [float(row["final_average_reward"]) for row in per_seed]
    weighted_tail_mses = [
        float(row["weighted_tail_td_error_mse"]) for row in per_seed
    ]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step5.off_policy_average_reward_benchmark.v1",
        "claim_scope": "off_policy_average_reward_differential_gtd_prediction",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "behavior_target_action_prob": behavior_target_action_prob,
            "value_step_size": value_step_size,
            "secondary_step_size": secondary_step_size,
            "average_reward_step_size": average_reward_step_size,
            "trace_decay": trace_decay,
            "ratio_clip": ratio_clip,
        },
        "target": {
            "target_policy_average_reward": 1.0,
            "behavior_policy_average_reward": behavior_target_action_prob,
        },
        "aggregate": {
            "mean_average_reward_abs_error": sum(average_reward_errors)
            / len(average_reward_errors),
            "stderr_average_reward_abs_error": _stderr(average_reward_errors),
            "mean_final_average_reward": sum(final_average_rewards)
            / len(final_average_rewards),
            "stderr_final_average_reward": _stderr(final_average_rewards),
            "mean_weighted_tail_td_error_mse": sum(weighted_tail_mses)
            / len(weighted_tail_mses),
            "stderr_weighted_tail_td_error_mse": _stderr(weighted_tail_mses),
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
    parser.add_argument("--behavior-target-action-prob", type=float, default=0.5)
    parser.add_argument("--value-step-size", type=float, default=0.02)
    parser.add_argument("--secondary-step-size", type=float, default=0.005)
    parser.add_argument("--average-reward-step-size", type=float, default=0.005)
    parser.add_argument("--trace-decay", type=float, default=0.0)
    parser.add_argument("--ratio-clip", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        behavior_target_action_prob=args.behavior_target_action_prob,
        value_step_size=args.value_step_size,
        secondary_step_size=args.secondary_step_size,
        average_reward_step_size=args.average_reward_step_size,
        trace_decay=args.trace_decay,
        ratio_clip=args.ratio_clip,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

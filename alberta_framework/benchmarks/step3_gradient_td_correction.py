#!/usr/bin/env python3
"""Seeded off-policy Gradient-TD correction benchmark for Step 3."""

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

from alberta_framework.core.off_policy_td import (  # noqa: E402
    GradientTDLinearLearner,
    run_gradient_td_learning_loop,
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
    behavior_action_1_prob: float,
    gamma: float,
    step_size: float,
    secondary_step_size: float,
    trace_decay: float,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run one two-demon off-policy Gradient-TD seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    if not 0.0 < behavior_action_1_prob < 1.0:
        raise ValueError("behavior_action_1_prob must be in (0, 1)")

    key = jr.key(seed)
    action_1 = jr.bernoulli(key, behavior_action_1_prob, (steps,))
    rewards = jnp.where(action_1, 1.0, -1.0).astype(jnp.float32)
    observations = jnp.ones((steps, 1), dtype=jnp.float32)
    next_observations = jnp.ones((steps, 1), dtype=jnp.float32)
    gammas = jnp.full((steps,), gamma, dtype=jnp.float32)
    rhos = [
        jnp.where(action_1, 0.0, 1.0 / (1.0 - behavior_action_1_prob)).astype(
            jnp.float32
        ),
        jnp.where(action_1, 1.0 / behavior_action_1_prob, 0.0).astype(jnp.float32),
    ]

    learner = GradientTDLinearLearner(
        step_size=step_size,
        secondary_step_size=secondary_step_size,
        trace_decay=trace_decay,
        ratio_clip=ratio_clip,
    )
    results = [
        run_gradient_td_learning_loop(
            learner,
            learner.init(1),
            observations,
            rewards,
            next_observations,
            gammas,
            rho,
        )
        for rho in rhos
    ]
    results[0].td_errors.block_until_ready()
    results[1].td_errors.block_until_ready()

    predictions = jnp.array(
        [learner.predict(result.state, jnp.ones(1))[0] for result in results],
        dtype=jnp.float32,
    )
    targets = jnp.array([-1.0, 1.0], dtype=jnp.float32) / (1.0 - gamma)
    abs_errors = jnp.abs(predictions - targets)
    weighted_tail_mses = []
    secondary_norms = []
    finite = True
    for result in results:
        weighted_tail_mses.append(
            float(
                jnp.sum(
                    result.rho_clipped[-final_window:]
                    * result.td_errors[-final_window:] ** 2
                )
                / jnp.maximum(
                    1.0,
                    jnp.sum(result.rho_clipped[-final_window:] > 0.0),
                )
            )
        )
        secondary_norms.append(float(jnp.linalg.norm(result.state.secondary_weights)))
        finite = finite and bool(
            jnp.all(jnp.isfinite(result.td_errors))
            & jnp.all(jnp.isfinite(result.state.weights))
            & jnp.all(jnp.isfinite(result.state.secondary_weights))
        )

    mean_abs_error = float(jnp.mean(abs_errors))
    max_weighted_tail_mse = max(weighted_tail_mses)
    min_secondary_norm = min(secondary_norms)

    return {
        "seed": seed,
        "predictions": [float(value) for value in predictions],
        "target_values": [float(value) for value in targets],
        "mean_abs_error": mean_abs_error,
        "weighted_tail_td_mse": weighted_tail_mses,
        "secondary_weight_norms": secondary_norms,
        "finite": finite,
        "passed": bool(
            finite
            and mean_abs_error <= 0.05
            and max_weighted_tail_mse <= 0.01
            and min_secondary_norm >= 0.0
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    behavior_action_1_prob: float,
    gamma: float,
    step_size: float,
    secondary_step_size: float,
    trace_decay: float,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run all seeds."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            behavior_action_1_prob=behavior_action_1_prob,
            gamma=gamma,
            step_size=step_size,
            secondary_step_size=secondary_step_size,
            trace_decay=trace_decay,
            ratio_clip=ratio_clip,
        )
        for seed in range(seeds)
    ]
    errors = [float(row["mean_abs_error"]) for row in per_seed]
    tail_mses = [
        max(float(value) for value in row["weighted_tail_td_mse"])
        for row in per_seed
    ]
    passed = all(bool(row["passed"]) for row in per_seed)
    return {
        "schema": "alberta.step3.gradient_td_correction_benchmark.v1",
        "claim_scope": "linear_multi_demon_off_policy_gradient_td_correction",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "behavior_action_1_prob": behavior_action_1_prob,
            "gamma": gamma,
            "step_size": step_size,
            "secondary_step_size": secondary_step_size,
            "trace_decay": trace_decay,
            "ratio_clip": ratio_clip,
        },
        "aggregate": {
            "mean_abs_error": sum(errors) / len(errors),
            "stderr_abs_error": _stderr(errors),
            "mean_weighted_tail_td_mse": sum(tail_mses) / len(tail_mses),
            "stderr_weighted_tail_td_mse": _stderr(tail_mses),
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
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--behavior-action-1-prob", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--secondary-step-size", type=float, default=0.05)
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
        behavior_action_1_prob=args.behavior_action_1_prob,
        gamma=args.gamma,
        step_size=args.step_size,
        secondary_step_size=args.secondary_step_size,
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

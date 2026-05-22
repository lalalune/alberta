#!/usr/bin/env python3
"""Seeded nonlinear off-policy Horde benchmark for Step 3."""

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

from alberta_framework.core.off_policy_horde import (  # noqa: E402
    OffPolicyHordeLearner,
    run_off_policy_horde_learning_loop,
)
from alberta_framework.core.optimizers import LMS  # noqa: E402
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


def _make_spec(gamma: float) -> Any:
    demons = (
        GVFSpec(
            name="target_action_0",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=0.0,
            cumulant_index=0,
        ),
        GVFSpec(
            name="target_action_1",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=0.0,
            cumulant_index=1,
        ),
    )
    return create_horde_spec(demons)


def run_seed(
    seed: int,
    *,
    steps: int,
    final_window: int,
    behavior_action_1_prob: float,
    gamma: float,
    step_size: float,
    hidden_size: int,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run one off-policy Horde positive-control seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")
    if not 0.0 < behavior_action_1_prob < 1.0:
        raise ValueError("behavior_action_1_prob must be in (0, 1)")

    key = jr.key(seed)
    action_1 = jr.bernoulli(key, behavior_action_1_prob, (steps,))
    rewards = jnp.where(action_1, 1.0, -1.0).astype(jnp.float32)

    observations = jnp.ones((steps, 1), dtype=jnp.float32)
    next_observations = jnp.ones((steps, 1), dtype=jnp.float32)
    cumulants = jnp.stack([rewards, rewards], axis=1)
    rhos = jnp.stack(
        [
            jnp.where(action_1, 0.0, 1.0 / (1.0 - behavior_action_1_prob)),
            jnp.where(action_1, 1.0 / behavior_action_1_prob, 0.0),
        ],
        axis=1,
    ).astype(jnp.float32)
    no_is_rhos = jnp.ones_like(rhos)

    learner = OffPolicyHordeLearner(
        _make_spec(gamma),
        hidden_sizes=(hidden_size,),
        optimizer=LMS(step_size=step_size),
        ratio_clip=ratio_clip,
        trace_ratio_clip=ratio_clip,
        sparsity=0.0,
        use_layer_norm=False,
    )
    initial_state = learner.init(1, jr.key(seed + 10_000))
    off_policy = run_off_policy_horde_learning_loop(
        learner,
        initial_state,
        observations,
        cumulants,
        next_observations,
        rhos,
    )
    no_is = run_off_policy_horde_learning_loop(
        learner,
        initial_state,
        observations,
        cumulants,
        next_observations,
        no_is_rhos,
    )
    off_policy.td_errors.block_until_ready()
    no_is.td_errors.block_until_ready()

    query = jnp.ones(1, dtype=jnp.float32)
    predictions = learner.predict(off_policy.state, query)
    no_is_predictions = learner.predict(no_is.state, query)
    targets = jnp.array([-1.0, 1.0], dtype=jnp.float32) / (1.0 - gamma)
    abs_errors = jnp.abs(predictions - targets)
    no_is_abs_errors = jnp.abs(no_is_predictions - targets)
    weighted_tail_mse = jnp.sum(
        off_policy.clipped_rhos[-final_window:] * off_policy.td_errors[-final_window:] ** 2,
        axis=0,
    ) / jnp.maximum(
        1.0,
        jnp.sum(off_policy.clipped_rhos[-final_window:] > 0.0, axis=0),
    )
    finite = bool(
        jnp.all(jnp.isfinite(predictions))
        & jnp.all(jnp.isfinite(no_is_predictions))
        & jnp.all(jnp.isfinite(off_policy.td_errors))
        & jnp.all(jnp.isfinite(no_is.td_errors))
    )
    mean_abs_error = float(jnp.mean(abs_errors))
    mean_no_is_abs_error = float(jnp.mean(no_is_abs_errors))

    return {
        "seed": seed,
        "target_values": [float(value) for value in targets],
        "off_policy_predictions": [float(value) for value in predictions],
        "no_is_predictions": [float(value) for value in no_is_predictions],
        "mean_abs_error": mean_abs_error,
        "mean_no_is_abs_error": mean_no_is_abs_error,
        "weighted_tail_td_mse": [float(value) for value in weighted_tail_mse],
        "finite": finite,
        "passed": bool(
            finite
            and mean_abs_error <= 0.08
            and mean_no_is_abs_error - mean_abs_error >= 0.7
            and float(jnp.max(weighted_tail_mse)) <= 0.02
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
    hidden_size: int,
    ratio_clip: float,
) -> dict[str, Any]:
    """Run the multi-seed benchmark."""
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
            hidden_size=hidden_size,
            ratio_clip=ratio_clip,
        )
        for seed in range(seeds)
    ]
    errors = [float(row["mean_abs_error"]) for row in per_seed]
    no_is_errors = [float(row["mean_no_is_abs_error"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step3.off_policy_horde_benchmark.v1",
        "claim_scope": "nonlinear_per_demon_importance_weighted_horde_prediction",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "behavior_action_1_prob": behavior_action_1_prob,
            "gamma": gamma,
            "step_size": step_size,
            "hidden_size": hidden_size,
            "ratio_clip": ratio_clip,
        },
        "aggregate": {
            "mean_abs_error": sum(errors) / len(errors),
            "stderr_abs_error": _stderr(errors),
            "mean_no_is_abs_error": sum(no_is_errors) / len(no_is_errors),
            "stderr_no_is_abs_error": _stderr(no_is_errors),
            "mean_improvement_vs_no_is": (
                sum(no_is_errors) / len(no_is_errors)
                - sum(errors) / len(errors)
            ),
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
    parser.add_argument("--hidden-size", type=int, default=8)
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
        hidden_size=args.hidden_size,
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

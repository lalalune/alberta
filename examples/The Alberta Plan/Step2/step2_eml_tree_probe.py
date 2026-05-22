#!/usr/bin/env python3
"""Probe fixed-depth differentiable EML trees on symbolic targets.

This script is intentionally small: it tests whether a soft leaf-selection EML
tree can reduce error on simple bounded symbolic-regression targets, and reports
the gap between the trained soft tree and its hard argmax-leaf tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import EMLTreeLearner, EMLTreeState


def make_dataset(name: str, n_points: int) -> tuple[jax.Array, jax.Array]:
    """Create one bounded symbolic-regression dataset."""
    if name == "x":
        x = jnp.linspace(-0.8, 0.8, n_points, dtype=jnp.float32)
        y = x
    elif name == "x_plus_one":
        x = jnp.linspace(-0.8, 0.8, n_points, dtype=jnp.float32)
        y = x + 1.0
    elif name == "x_squared":
        x = jnp.linspace(-0.8, 0.8, n_points, dtype=jnp.float32)
        y = x**2
    elif name == "exp":
        x = jnp.linspace(-0.8, 0.8, n_points, dtype=jnp.float32)
        y = jnp.exp(x)
    elif name == "log":
        x = jnp.linspace(0.2, 2.0, n_points, dtype=jnp.float32)
        y = jnp.log(x)
    else:
        raise ValueError(f"unknown target {name!r}")
    return x.reshape(-1, 1), y.reshape(-1, 1)


def mse(
    learner: EMLTreeLearner,
    state: EMLTreeState,
    observations: jax.Array,
    targets: jax.Array,
    *,
    hard: bool = False,
) -> float:
    """Compute scalar MSE for soft or hard tree predictions."""
    predict = learner.predict_hard if hard else learner.predict
    errors = [
        (predict(state, observation)[0] - target[0]) ** 2
        for observation, target in zip(observations, targets)
    ]
    return float(jnp.mean(jnp.array(errors)))


def run_target(
    target_name: str,
    *,
    seed: int,
    depth: int,
    n_constants: int,
    n_points: int,
    num_updates: int,
    step_size: float,
    temperature: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Train one EML tree on one target and return summary metrics."""
    observations, targets = make_dataset(target_name, n_points)
    learner = EMLTreeLearner(
        depth=depth,
        n_constants=n_constants,
        step_size=step_size,
        temperature=temperature,
        max_grad_norm=max_grad_norm,
        output_init_scale=0.1,
    )
    state = learner.init(feature_dim=1, key=jr.key(seed))

    initial_soft_mse = mse(learner, state, observations, targets)
    initial_hard_mse = mse(learner, state, observations, targets, hard=True)

    for step in range(num_updates):
        idx = step % n_points
        state = learner.update(state, observations[idx], targets[idx]).state

    final_soft_mse = mse(learner, state, observations, targets)
    final_hard_mse = mse(learner, state, observations, targets, hard=True)
    constants = learner._constant_values(state.params)  # noqa: SLF001 - experiment inspection

    return {
        "target": target_name,
        "initial_soft_mse": initial_soft_mse,
        "final_soft_mse": final_soft_mse,
        "initial_hard_mse": initial_hard_mse,
        "final_hard_mse": final_hard_mse,
        "soft_improvement": initial_soft_mse - final_soft_mse,
        "hard_expression": learner.hard_expression(state, feature_dim=1),
        "hard_leaf_indices": np.asarray(learner.hard_leaf_indices(state)).tolist(),
        "constants": np.asarray(constants).tolist(),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-constants", type=int, default=2)
    parser.add_argument("--n-points", type=int, default=33)
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["x", "x_plus_one", "x_squared", "exp", "log"],
    )
    return parser.parse_args()


def main() -> int:
    """Run the EML tree symbolic probe."""
    args = parse_args()
    records = [
        run_target(
            target,
            seed=args.seed,
            depth=args.depth,
            n_constants=args.n_constants,
            n_points=args.n_points,
            num_updates=args.num_updates,
            step_size=args.step_size,
            temperature=args.temperature,
            max_grad_norm=args.max_grad_norm,
        )
        for target in args.targets
    ]
    result = {
        "config": {
            "seed": args.seed,
            "depth": args.depth,
            "n_constants": args.n_constants,
            "n_points": args.n_points,
            "num_updates": args.num_updates,
            "step_size": args.step_size,
            "temperature": args.temperature,
            "max_grad_norm": args.max_grad_norm,
        },
        "records": records,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Seeded nonlinear trace benchmark for the independent-demon Horde."""

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

from alberta_framework.core.independent_demon_horde import (  # noqa: E402
    IndependentDemonHorde,
    run_independent_horde_learning_loop,
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


def _make_spec(gamma: float, lamda: float) -> Any:
    demons = (
        GVFSpec(
            name="nonlinear_trace_value",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamda,
            cumulant_index=0,
        ),
    )
    return create_horde_spec(demons)


def run_seed(
    seed: int,
    *,
    steps: int,
    final_window: int,
    gamma: float,
    lamda: float,
    step_size: float,
    hidden_size: int,
) -> dict[str, Any]:
    """Run one nonlinear independent-demon trace seed."""
    if final_window < 1 or final_window > steps:
        raise ValueError("final_window must be in [1, steps]")

    learner = IndependentDemonHorde(
        _make_spec(gamma, lamda),
        hidden_sizes=(hidden_size,),
        optimizer=LMS(step_size=step_size),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(1, jr.key(seed))
    observations = jnp.ones((steps, 1), dtype=jnp.float32)
    next_observations = jnp.ones((steps, 1), dtype=jnp.float32)
    cumulants = jnp.ones((steps, 1), dtype=jnp.float32)
    result = run_independent_horde_learning_loop(
        learner,
        state,
        observations,
        cumulants,
        next_observations,
    )
    result.td_errors.block_until_ready()

    prediction = float(learner.predict(result.state, jnp.ones(1, dtype=jnp.float32))[0])
    target = 1.0 / (1.0 - gamma)
    abs_error = abs(prediction - target)
    tail_td_mse = float(jnp.mean(result.td_errors[-final_window:, 0] ** 2))
    final_demon_state = result.state.demon_states[0]
    trunk_trace_norm = float(jnp.linalg.norm(final_demon_state.traces[0]))
    finite = bool(
        jnp.isfinite(jnp.asarray(prediction))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(final_demon_state.params.weights[0]))
        & jnp.all(jnp.isfinite(final_demon_state.traces[0]))
    )

    return {
        "seed": seed,
        "prediction": prediction,
        "target": target,
        "abs_error": abs_error,
        "tail_td_mse": tail_td_mse,
        "trunk_trace_norm": trunk_trace_norm,
        "finite": finite,
        "passed": bool(
            finite
            and trunk_trace_norm > 0.0
            and abs_error <= 0.05
            and tail_td_mse <= 0.01
        ),
    }


def run_benchmark(
    *,
    seeds: int,
    steps: int,
    final_window: int,
    gamma: float,
    lamda: float,
    step_size: float,
    hidden_size: int,
) -> dict[str, Any]:
    """Run the multi-seed benchmark."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    per_seed = [
        run_seed(
            seed,
            steps=steps,
            final_window=final_window,
            gamma=gamma,
            lamda=lamda,
            step_size=step_size,
            hidden_size=hidden_size,
        )
        for seed in range(seeds)
    ]
    errors = [float(row["abs_error"]) for row in per_seed]
    tail_mses = [float(row["tail_td_mse"]) for row in per_seed]
    trace_norms = [float(row["trunk_trace_norm"]) for row in per_seed]
    passed = all(bool(row["passed"]) for row in per_seed)

    return {
        "schema": "alberta.step3.independent_trace_horde_benchmark.v1",
        "claim_scope": "independent_nonlinear_demon_full_gamma_lambda_traces",
        "config": {
            "seeds": seeds,
            "steps": steps,
            "final_window": final_window,
            "gamma": gamma,
            "lamda": lamda,
            "step_size": step_size,
            "hidden_size": hidden_size,
        },
        "aggregate": {
            "mean_abs_error": sum(errors) / len(errors),
            "stderr_abs_error": _stderr(errors),
            "mean_tail_td_mse": sum(tail_mses) / len(tail_mses),
            "stderr_tail_td_mse": _stderr(tail_mses),
            "mean_trunk_trace_norm": sum(trace_norms) / len(trace_norms),
            "stderr_trunk_trace_norm": _stderr(trace_norms),
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
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark."""
    args = parse_args(argv)
    report = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        final_window=args.final_window,
        gamma=args.gamma,
        lamda=args.lamda,
        step_size=args.step_size,
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

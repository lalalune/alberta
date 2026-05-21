#!/usr/bin/env python3
"""Seeded nonlinear shared-trunk Gradient-TD Horde benchmark for Step 3.

This is a positive-control benchmark for the strongest local Step 3 off-policy
boundary: a nonlinear shared trunk with a genuine secondary-weight correction.
It implements a compact two-demon TDC/GTD-style update over the full nonlinear
parameter gradient.  The primary parameters are shared across demons through a
single hidden trunk; each demon has its own secondary weight PyTree matching the
full parameter PyTree.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array


class Params(NamedTuple):
    """Shared-trunk, two-head nonlinear value parameters."""

    trunk_w: Array
    trunk_b: Array
    head_w: Array
    head_b: Array


class TrainState(NamedTuple):
    """Primary and per-demon secondary parameters."""

    params: Params
    secondary: tuple[Params, Params]


class StepMetrics(NamedTuple):
    """Diagnostics from one transition."""

    td_errors: Array
    correction_norm: Array


def tree_add(a: Params, b: Params) -> Params:
    """Add two parameter PyTrees."""
    return Params(
        a.trunk_w + b.trunk_w,
        a.trunk_b + b.trunk_b,
        a.head_w + b.head_w,
        a.head_b + b.head_b,
    )


def tree_scale(scale: Array | float, tree: Params) -> Params:
    """Scale a parameter PyTree."""
    return Params(
        scale * tree.trunk_w,
        scale * tree.trunk_b,
        scale * tree.head_w,
        scale * tree.head_b,
    )


def tree_dot(a: Params, b: Params) -> Array:
    """Dot product between matching parameter PyTrees."""
    return (
        jnp.vdot(a.trunk_w, b.trunk_w)
        + jnp.vdot(a.trunk_b, b.trunk_b)
        + jnp.vdot(a.head_w, b.head_w)
        + jnp.vdot(a.head_b, b.head_b)
    )


def tree_norm(tree: Params) -> Array:
    """Euclidean norm of a parameter PyTree."""
    return jnp.sqrt(tree_dot(tree, tree))


def value(params: Params, observation: Array, demon: int) -> Array:
    """Value prediction for one demon."""
    hidden = jnp.tanh(params.trunk_w @ observation + params.trunk_b)
    return params.head_w[demon] @ hidden + params.head_b[demon]


def all_values(params: Params, observations: Array) -> Array:
    """Predict all demon values for a batch of observations."""

    def one_obs(obs: Array) -> Array:
        hidden = jnp.tanh(params.trunk_w @ obs + params.trunk_b)
        return params.head_w @ hidden + params.head_b

    return jax.vmap(one_obs)(observations)


def zero_like_params(params: Params) -> Params:
    """Zero-valued parameter PyTree matching ``params``."""
    return jax.tree.map(jnp.zeros_like, params)


def init_state(key: Array, hidden_size: int) -> TrainState:
    """Initialize primary and secondary parameters."""
    k1, k2 = jr.split(key)
    trunk_w = 0.25 * jr.normal(k1, (hidden_size, 2), dtype=jnp.float32)
    head_w = 0.25 * jr.normal(k2, (2, hidden_size), dtype=jnp.float32)
    params = Params(
        trunk_w=trunk_w,
        trunk_b=jnp.zeros(hidden_size, dtype=jnp.float32),
        head_w=head_w,
        head_b=jnp.zeros(2, dtype=jnp.float32),
    )
    secondary = (zero_like_params(params), zero_like_params(params))
    return TrainState(params=params, secondary=secondary)


def transition_arrays(seed: int, steps: int) -> tuple[Array, Array, Array, Array, Array]:
    """Generate a behavior-policy stream for the two-state off-policy process."""
    rng = np.random.default_rng(seed)
    states = np.empty(steps, dtype=np.int32)
    actions = np.empty(steps, dtype=np.int32)
    rewards = np.empty(steps, dtype=np.float32)
    next_states = np.empty(steps, dtype=np.int32)
    state = int(rng.integers(0, 2))
    for t in range(steps):
        action = int(rng.integers(0, 2))
        next_state = action
        reward = 1.0 if state == action else 0.0
        states[t] = state
        actions[t] = action
        rewards[t] = reward
        next_states[t] = next_state
        state = next_state

    observations = np.eye(2, dtype=np.float32)[states]
    next_observations = np.eye(2, dtype=np.float32)[next_states]
    rhos = np.zeros((steps, 2), dtype=np.float32)
    rhos[actions == 0, 0] = 2.0
    rhos[actions == 1, 1] = 2.0
    cumulants = np.repeat(rewards[:, None], 2, axis=1)
    return (
        jnp.asarray(observations),
        jnp.asarray(cumulants),
        jnp.asarray(next_observations),
        jnp.asarray(rhos),
        jnp.asarray(actions),
    )


def target_values(gamma: float) -> Array:
    """Exact values for always-action-0 and always-action-1 target policies."""
    optimal_state_value = 1.0 / (1.0 - gamma)
    other_state_value = gamma * optimal_state_value
    return jnp.asarray(
        [
            [optimal_state_value, other_state_value],
            [other_state_value, optimal_state_value],
        ],
        dtype=jnp.float32,
    )


def tdc_update(
    state: TrainState,
    observation: Array,
    cumulants: Array,
    next_observation: Array,
    rhos: Array,
    gamma: float,
    alpha: float,
    beta: float,
) -> tuple[TrainState, StepMetrics]:
    """One nonlinear shared-trunk TDC update."""
    primary_step = zero_like_params(state.params)
    new_secondary: list[Params] = []
    td_errors: list[Array] = []
    correction_norms: list[Array] = []

    for demon in range(2):
        def demon_value(p: Params) -> Array:
            return value(p, observation, demon)

        def next_demon_value(p: Params) -> Array:
            return value(p, next_observation, demon)

        grad = jax.grad(demon_value)(state.params)
        next_grad = jax.grad(next_demon_value)(state.params)
        pred = demon_value(state.params)
        next_pred = next_demon_value(state.params)
        delta = cumulants[demon] + gamma * next_pred - pred
        rho = rhos[demon]
        secondary = state.secondary[demon]
        secondary_dot = tree_dot(secondary, grad)
        correction = tree_scale(rho * gamma * secondary_dot, next_grad)
        demon_step = tree_add(
            tree_scale(alpha * rho * delta, grad),
            tree_scale(-alpha, correction),
        )
        secondary_step = tree_add(
            tree_scale(beta * rho * delta, grad),
            tree_scale(-beta * secondary_dot, grad),
        )
        primary_step = tree_add(primary_step, demon_step)
        new_secondary.append(tree_add(secondary, secondary_step))
        td_errors.append(delta)
        correction_norms.append(tree_norm(correction))

    new_state = TrainState(
        params=tree_add(state.params, primary_step),
        secondary=(new_secondary[0], new_secondary[1]),
    )
    return new_state, StepMetrics(
        td_errors=jnp.stack(td_errors),
        correction_norm=jnp.mean(jnp.stack(correction_norms)),
    )


def semi_gradient_update(
    state: TrainState,
    observation: Array,
    cumulants: Array,
    next_observation: Array,
    rhos: Array,
    gamma: float,
    alpha: float,
) -> tuple[TrainState, StepMetrics]:
    """Matching nonlinear shared-trunk semi-gradient update without correction."""
    primary_step = zero_like_params(state.params)
    td_errors: list[Array] = []

    for demon in range(2):
        def demon_value(p: Params) -> Array:
            return value(p, observation, demon)

        grad = jax.grad(demon_value)(state.params)
        pred = demon_value(state.params)
        next_pred = value(state.params, next_observation, demon)
        delta = cumulants[demon] + gamma * next_pred - pred
        primary_step = tree_add(primary_step, tree_scale(alpha * rhos[demon] * delta, grad))
        td_errors.append(delta)

    return TrainState(
        params=tree_add(state.params, primary_step),
        secondary=state.secondary,
    ), StepMetrics(
        td_errors=jnp.stack(td_errors),
        correction_norm=jnp.array(0.0, dtype=jnp.float32),
    )


def run_scan(
    state: TrainState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    rhos: Array,
    *,
    gamma: float,
    alpha: float,
    beta: float,
    corrected: bool,
) -> tuple[TrainState, StepMetrics]:
    """Run one learner over transition arrays."""

    def step(carry: TrainState, xs: tuple[Array, Array, Array, Array]):
        obs, cums, next_obs, rho = xs
        if corrected:
            new_carry, metrics = tdc_update(
                carry, obs, cums, next_obs, rho, gamma, alpha, beta
            )
        else:
            new_carry, metrics = semi_gradient_update(
                carry, obs, cums, next_obs, rho, gamma, alpha
            )
        return new_carry, metrics

    return jax.lax.scan(step, state, (observations, cumulants, next_observations, rhos))


run_scan_jit = jax.jit(run_scan, static_argnames=("gamma", "alpha", "beta", "corrected"))


def run_seed(
    seed: int,
    *,
    steps: int,
    hidden_size: int,
    gamma: float,
    alpha: float,
    beta: float,
    tail_window: int,
) -> dict[str, float | int | bool]:
    """Run one benchmark seed."""
    observations, cumulants, next_observations, rhos, _ = transition_arrays(seed, steps)
    initial = init_state(jr.key(seed), hidden_size)
    corrected_state, corrected_metrics = run_scan_jit(
        initial,
        observations,
        cumulants,
        next_observations,
        rhos,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        corrected=True,
    )
    semi_state, semi_metrics = run_scan_jit(
        initial,
        observations,
        cumulants,
        next_observations,
        rhos,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        corrected=False,
    )

    eval_obs = jnp.eye(2, dtype=jnp.float32)
    target = target_values(gamma)
    corrected_values = all_values(corrected_state.params, eval_obs)
    semi_values = all_values(semi_state.params, eval_obs)
    corrected_abs_error = float(jnp.mean(jnp.abs(corrected_values - target)))
    semi_abs_error = float(jnp.mean(jnp.abs(semi_values - target)))
    tail_td_mse = float(jnp.mean(corrected_metrics.td_errors[-tail_window:] ** 2))
    correction_norm = float(jnp.mean(corrected_metrics.correction_norm[-tail_window:]))
    trunk_change = float(
        jnp.linalg.norm(corrected_state.params.trunk_w - initial.params.trunk_w)
    )
    secondary_norm = float(
        jnp.mean(jnp.stack([tree_norm(s) for s in corrected_state.secondary]))
    )
    passed = (
        corrected_abs_error <= 0.25
        and correction_norm > 0.0
        and secondary_norm > 0.0
        and trunk_change > 0.0
    )
    return {
        "seed": seed,
        "corrected_abs_error": corrected_abs_error,
        "semi_gradient_abs_error": semi_abs_error,
        "improvement_vs_semi_gradient": semi_abs_error - corrected_abs_error,
        "tail_td_mse": tail_td_mse,
        "tail_correction_norm": correction_norm,
        "secondary_norm": secondary_norm,
        "trunk_change_norm": trunk_change,
        "passed": passed,
    }


def run_benchmark(
    *,
    seeds: Sequence[int],
    steps: int,
    hidden_size: int,
    gamma: float,
    alpha: float,
    beta: float,
    tail_window: int,
) -> dict[str, object]:
    """Run all seeds and aggregate evidence."""
    rows = [
        run_seed(
            seed,
            steps=steps,
            hidden_size=hidden_size,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            tail_window=tail_window,
        )
        for seed in seeds
    ]
    aggregate = {
        "n_seeds": len(rows),
        "n_passed": sum(bool(row["passed"]) for row in rows),
        "mean_corrected_abs_error": float(
            np.mean([row["corrected_abs_error"] for row in rows])
        ),
        "mean_semi_gradient_abs_error": float(
            np.mean([row["semi_gradient_abs_error"] for row in rows])
        ),
        "mean_improvement_vs_semi_gradient": float(
            np.mean([row["improvement_vs_semi_gradient"] for row in rows])
        ),
        "mean_tail_td_mse": float(np.mean([row["tail_td_mse"] for row in rows])),
        "mean_tail_correction_norm": float(
            np.mean([row["tail_correction_norm"] for row in rows])
        ),
        "mean_secondary_norm": float(np.mean([row["secondary_norm"] for row in rows])),
        "mean_trunk_change_norm": float(
            np.mean([row["trunk_change_norm"] for row in rows])
        ),
    }
    passed = (
        aggregate["n_seeds"] >= 10
        and aggregate["n_passed"] == aggregate["n_seeds"]
        and aggregate["mean_corrected_abs_error"] <= 0.25
        and aggregate["mean_tail_correction_norm"] > 0.0
        and aggregate["mean_secondary_norm"] > 0.0
        and aggregate["mean_trunk_change_norm"] > 0.0
    )
    return {
        "schema": "alberta.step3.nonlinear_shared_gtd_horde_benchmark.v1",
        "claim_scope": "nonlinear_shared_trunk_off_policy_gradient_td_correction",
        "config": {
            "seeds": list(seeds),
            "steps": steps,
            "hidden_size": hidden_size,
            "gamma": gamma,
            "alpha": alpha,
            "beta": beta,
            "tail_window": tail_window,
        },
        "target_values": np.asarray(target_values(gamma)).tolist(),
        "rows": rows,
        "aggregate": aggregate,
        "passed": passed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step3_nonlinear_shared_gtd_horde/results.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--tail-window", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark and write JSON evidence."""
    args = parse_args(argv)
    result = run_benchmark(
        seeds=args.seeds,
        steps=args.steps,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        tail_window=args.tail_window,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

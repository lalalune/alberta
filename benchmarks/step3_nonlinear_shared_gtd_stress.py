#!/usr/bin/env python3
"""Stress benchmark for production nonlinear shared-trunk GTD Horde."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    NonlinearSharedGTDHordeState,
    NonlinearSharedGTDHordeUpdateResult,
)
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec


class Regime(NamedTuple):
    """One off-policy stress regime."""

    name: str
    gamma: float
    behavior_action1_prob: float
    steps: int
    abs_error_threshold: float


REGIMES = (
    Regime("balanced_gamma05", 0.5, 0.5, 2500, 0.18),
    Regime("balanced_gamma08", 0.8, 0.5, 5000, 0.35),
    Regime("skewed_gamma08", 0.8, 0.7, 7000, 0.45),
)


def make_spec(gamma: float) -> HordeSpec:
    """Create a two-demon target-policy GVF spec."""
    return create_horde_spec(
        tuple(
            GVFSpec(
                name=f"target_action_{action}",
                demon_type=DemonType.PREDICTION,
                gamma=gamma,
                lamda=0.0,
                cumulant_index=action,
            )
            for action in range(2)
        )
    )


def target_values(gamma: float) -> Array:
    """Exact values for target policies that always choose action 0/1."""
    optimal = 1.0 / (1.0 - gamma)
    other = gamma * optimal
    return jnp.asarray([[optimal, other], [other, optimal]], dtype=jnp.float32)


def transition_arrays(
    seed: int,
    steps: int,
    behavior_action1_prob: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Generate two-state off-policy transitions and per-demon ratios."""
    rng = np.random.default_rng(seed)
    states = np.empty(steps, dtype=np.int32)
    actions = np.empty(steps, dtype=np.int32)
    rewards = np.empty(steps, dtype=np.float32)
    next_states = np.empty(steps, dtype=np.int32)
    state = int(rng.integers(0, 2))
    for t in range(steps):
        action = int(rng.random() < behavior_action1_prob)
        next_state = action
        reward = 1.0 if state == action else 0.0
        states[t] = state
        actions[t] = action
        rewards[t] = reward
        next_states[t] = next_state
        state = next_state

    behavior_probs = np.asarray(
        [1.0 - behavior_action1_prob, behavior_action1_prob],
        dtype=np.float32,
    )
    rhos = np.zeros((steps, 2), dtype=np.float32)
    rhos[actions == 0, 0] = 1.0 / behavior_probs[0]
    rhos[actions == 1, 1] = 1.0 / behavior_probs[1]
    return (
        jnp.asarray(np.eye(2, dtype=np.float32)[states]),
        jnp.repeat(jnp.asarray(rewards)[:, None], 2, axis=1),
        jnp.asarray(np.eye(2, dtype=np.float32)[next_states]),
        jnp.asarray(rhos),
        jnp.asarray(actions),
    )


def run_scan(
    learner: NonlinearSharedGTDHordeLearner,
    state: NonlinearSharedGTDHordeState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    rhos: Array,
    gamma: float,
) -> tuple[NonlinearSharedGTDHordeState, NonlinearSharedGTDHordeUpdateResult]:
    """Run production learner over arrays."""
    discounts = jnp.full_like(rhos, gamma)

    def step(carry, xs):  # type: ignore[no-untyped-def]
        obs, cums, next_obs, rho, discount = xs
        result = learner.update_with_ratios_and_discounts(
            carry,
            obs,
            cums,
            next_obs,
            rho,
            discount,
        )
        return result.state, result

    return jax.lax.scan(
        step,
        state,
        (observations, cumulants, next_observations, rhos, discounts),
    )


run_scan_jit = jax.jit(run_scan, static_argnums=(0, 6))


def run_seed(seed: int, regime: Regime) -> dict[str, float | int | str | bool]:
    """Run one seed in one regime."""
    learner = NonlinearSharedGTDHordeLearner(
        make_spec(regime.gamma),
        hidden_size=8,
        primary_step_size=0.002,
        secondary_step_size=1e-5,
        ratio_clip=10.0,
    )
    observations, cumulants, next_observations, rhos, _ = transition_arrays(
        seed,
        regime.steps,
        regime.behavior_action1_prob,
    )
    initial = learner.init(2, jax.random.key(seed))
    final_state, updates = run_scan_jit(
        learner,
        initial,
        observations,
        cumulants,
        next_observations,
        rhos,
        regime.gamma,
    )
    eval_obs = jnp.eye(2, dtype=jnp.float32)
    predictions = jax.vmap(lambda obs: learner.predict(final_state, obs))(eval_obs)
    abs_error = float(jnp.mean(jnp.abs(predictions - target_values(regime.gamma))))
    correction_norm = float(jnp.mean(updates.correction_norms[-250:]))
    secondary_norm = float(jnp.mean(updates.secondary_norms[-250:]))
    trunk_change = float(jnp.linalg.norm(final_state.trunk_w - initial.trunk_w))
    passed = (
        abs_error <= regime.abs_error_threshold
        and correction_norm > 0.0
        and secondary_norm > 0.0
        and trunk_change > 0.0
    )
    return {
        "seed": seed,
        "regime": regime.name,
        "gamma": regime.gamma,
        "behavior_action1_prob": regime.behavior_action1_prob,
        "abs_error": abs_error,
        "tail_correction_norm": correction_norm,
        "tail_secondary_norm": secondary_norm,
        "trunk_change_norm": trunk_change,
        "passed": passed,
    }


def run_benchmark(seeds: Sequence[int]) -> dict[str, object]:
    """Run all regimes and aggregate evidence."""
    rows = [run_seed(seed, regime) for regime in REGIMES for seed in seeds]
    by_regime: dict[str, dict[str, float | int | bool]] = {}
    for regime in REGIMES:
        regime_rows = [row for row in rows if row["regime"] == regime.name]
        by_regime[regime.name] = {
            "n_seeds": len(regime_rows),
            "n_passed": sum(bool(row["passed"]) for row in regime_rows),
            "mean_abs_error": float(np.mean([row["abs_error"] for row in regime_rows])),
            "max_abs_error": float(np.max([row["abs_error"] for row in regime_rows])),
            "mean_tail_correction_norm": float(
                np.mean([row["tail_correction_norm"] for row in regime_rows])
            ),
            "mean_tail_secondary_norm": float(
                np.mean([row["tail_secondary_norm"] for row in regime_rows])
            ),
            "passed": all(bool(row["passed"]) for row in regime_rows),
        }
    aggregate = {
        "n_regimes": len(REGIMES),
        "n_rows": len(rows),
        "n_passed": sum(bool(row["passed"]) for row in rows),
        "mean_abs_error": float(np.mean([row["abs_error"] for row in rows])),
        "max_abs_error": float(np.max([row["abs_error"] for row in rows])),
    }
    passed = (
        aggregate["n_passed"] == aggregate["n_rows"]
        and all(bool(row["passed"]) for row in by_regime.values())
    )
    return {
        "schema": "alberta.step3.nonlinear_shared_gtd_stress.v1",
        "claim_scope": "production_nonlinear_shared_gtd_multi_regime_stress",
        "regimes": [regime._asdict() for regime in REGIMES],
        "rows": rows,
        "by_regime": by_regime,
        "aggregate": aggregate,
        "passed": passed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/step3_nonlinear_shared_gtd_stress/results.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark and write JSON evidence."""
    args = parse_args(argv)
    result = run_benchmark(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

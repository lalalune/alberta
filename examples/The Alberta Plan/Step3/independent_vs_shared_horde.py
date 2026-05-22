"""Independent-demon vs shared-trunk Horde on a random-walk stream.

Demonstrates the architectural trade-off at the heart of Phase C:
- ``HordeLearner`` (shared trunk): one MLP with N heads. Forced to
  trunk ``gamma * lamda = 0`` because the VJP backward pass folds
  per-head error into the trunk cotangent before trace accumulation.
- :class:`IndependentDemonHorde` (separate trunks): N independent MLPs.
  Full per-parameter eligibility traces (trunk + head) with
  ``gamma * lamda > 0`` are forward-view-correct everywhere.

We run the same ``gamma=0.9`` temporal demons through each architecture
on the same random-walk stream and compare prediction MSE. The
independent architecture pays a dramatic compute cost for the privilege
of full trace decay; this script shows whether that cost buys better
predictions on a non-stationary temporal target.

Usage:
    python "examples/The Alberta Plan/Step3/independent_vs_shared_horde.py" \\
        --output-dir output/step3_independent
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeLearner,
    IndependentDemonHorde,
    ObGDBounding,
    Timer,
    create_horde_spec,
    run_horde_learning_loop,
    run_independent_horde_learning_loop,
)
from alberta_framework.streams.synthetic import RandomWalkStream


def _build_temporal_demons() -> list[GVFSpec]:
    """Three temporal demons: gamma in {0.0, 0.5, 0.9} all sharing cumulant 0.

    The shared HordeLearner can use gamma>0 for the per-head trace decay
    but is forced to trunk ``gamma * lamda = 0``. The IndependentDemonHorde
    can use full ``gamma * lamda = 0.45`` (=0.9 * 0.5) on its temporal
    demon, including in the trunk.
    """
    return [
        GVFSpec(
            name="instantaneous",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=0,
        ),
        GVFSpec(
            name="short_horizon",
            demon_type=DemonType.PREDICTION,
            gamma=0.5,
            lamda=0.5,  # gamma*lamda = 0.25 (shared horde uses for head-only)
            cumulant_index=0,
        ),
        GVFSpec(
            name="long_horizon",
            demon_type=DemonType.PREDICTION,
            gamma=0.9,
            lamda=0.5,  # gamma*lamda = 0.45 (independent uses for trunk + head)
            cumulant_index=0,
        ),
    ]


def _generate_stream(
    feature_dim: int, num_steps: int, key
) -> tuple:
    """Roll out a RandomWalkStream and return obs/cumulants/next_obs arrays."""
    stream = RandomWalkStream(feature_dim=feature_dim, drift_rate=0.001)
    state = stream.init(key)

    observations = []
    targets = []
    for i in range(num_steps):
        ts, state = stream.step(state, jnp.array(i))
        observations.append(ts.observation)
        targets.append(ts.target.squeeze())

    obs_arr = jnp.stack(observations)
    target_arr = jnp.stack(targets)
    # All demons predict the same scalar target (cumulant 0).
    cum_arr = jnp.broadcast_to(target_arr[:, None], (num_steps, 3))
    next_obs_arr = jnp.concatenate([obs_arr[1:], obs_arr[:1]], axis=0)
    return obs_arr, cum_arr, next_obs_arr


def main(
    output_dir: Path, num_steps: int = 5000, seed: int = 42
) -> None:
    """Run independent vs shared Horde on the same random walk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = 8
    key = jr.key(seed)
    k_stream, k_shared, k_independent = jr.split(key, 3)

    demons = _build_temporal_demons()
    spec = create_horde_spec(demons)

    obs, cumulants, next_obs = _generate_stream(
        feature_dim, num_steps, k_stream
    )

    # Shared-trunk Horde: trunk gamma*lamda forced to 0; per-head decay only.
    shared = HordeLearner(
        horde_spec=spec,
        hidden_sizes=(32, 32),
        step_size=0.05,
        sparsity=0.9,
        bounder=ObGDBounding(kappa=2.0),
    )
    shared_state = shared.init(feature_dim, k_shared)

    # Independent-demon Horde: every demon has its own MLP; full trace
    # decay applies to trunk + head.
    independent = IndependentDemonHorde(
        horde_spec=spec,
        hidden_sizes=(32, 32),
        step_size=0.05,
        sparsity=0.9,
        bounder=ObGDBounding(kappa=2.0),
    )
    independent_state = independent.init(feature_dim, k_independent)

    print(
        f"Running shared HordeLearner over {num_steps} steps "
        f"(trunk gamma*lamda forced to 0)..."
    )
    with Timer("shared horde"):
        shared_result = run_horde_learning_loop(
            shared, shared_state, obs, cumulants, next_obs
        )

    print(
        f"Running IndependentDemonHorde over {num_steps} steps "
        f"(full per-parameter traces)..."
    )
    with Timer("independent horde"):
        independent_result = run_independent_horde_learning_loop(
            independent, independent_state, obs, cumulants, next_obs
        )

    # Final mean squared error per demon (last 500 steps).
    last = min(500, num_steps)
    print(f"\nMean squared error over last {last} steps:")
    print(
        f"{'demon':<20}{'shared MSE':>14}{'independent MSE':>20}{'ratio (ind/shared)':>22}"
    )
    summary: list[dict[str, object]] = []
    for i, demon in enumerate(demons):
        shared_mse = float(
            jnp.nanmean(shared_result.per_demon_metrics[-last:, i, 0])
        )
        independent_mse = float(
            jnp.nanmean(independent_result.per_demon_metrics[-last:, i, 0])
        )
        ratio = independent_mse / shared_mse if shared_mse > 0 else float("nan")
        print(
            f"{demon.name:<20}{shared_mse:>14.5f}{independent_mse:>20.5f}{ratio:>22.3f}"
        )
        summary.append(
            {
                "demon": demon.name,
                "gamma": demon.gamma,
                "lamda": demon.lamda,
                "shared_mse_last": shared_mse,
                "independent_mse_last": independent_mse,
                "ratio_independent_over_shared": ratio,
            }
        )

    summary_path = output_dir / "independent_vs_shared_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IndependentDemonHorde vs HordeLearner on RandomWalk"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step3_independent"),
    )
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output_dir, args.num_steps, args.seed)

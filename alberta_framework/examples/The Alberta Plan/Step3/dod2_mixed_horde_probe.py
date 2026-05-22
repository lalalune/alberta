"""DoD-2: MixedHorde nexting probe on the 5-state cyclic chain.

Multi-seed sanity probe for :class:`alberta_framework.MixedHorde` on the
canonical 5-state cyclic deterministic chain used in the Alberta Plan
DoD-2 nexting setup. Each demon predicts the discounted forward-view
return ``1 / (1 - gamma)`` at its own gamma horizon.

Setup
-----
- 5-state cyclic chain emitting cumulant=1 every step, so the analytic
  steady-state forward-view return is ``1 / (1 - gamma)``.
- Four demons with gamma in ``{0.0, 0.5, 0.9, 0.99}``; lambda is set to
  ``0.0`` for the gamma=0 demon (forcing it onto the shared trunk path)
  and to ``0.9`` for the rest (forcing them onto the independent path).
  Routing is therefore mixed by construction.
- 5 seeds x 2000 steps each.
- Outputs go to ``output/step3_mixed_horde_probe/``.

The probe verifies that:
- the run finishes without trunk-trace assertion errors,
- TD errors are finite,
- per-gamma RMSE vs the forward-view return is reported per seed and
  averaged across seeds.

It is a small integration probe, not a research-scale claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    DemonType,
    GVFSpec,
    MixedHorde,
    ObGDBounding,
    create_horde_spec,
    forward_view_returns,
    run_mixed_horde_learning_loop,
)

OUTPUT_DIR = Path("output/step3_mixed_horde_probe")
N_SEEDS = 5
N_STATES = 5
N_STEPS = 2000
STEP_SIZE = 0.05
GAMMAS = (0.0, 0.5, 0.9, 0.99)
# lambda=0 for gamma=0 demon (forces shared-path), lambda>0 elsewhere
# (forces independent-path).
LAMDAS = (0.0, 0.9, 0.9, 0.9)


def generate_chain(seed: int, n_steps: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return ``(observations, cumulants, next_observations)`` for a 5-state chain."""
    eye = np.eye(N_STATES, dtype=np.float32)
    rng = np.random.default_rng(seed)
    state = int(rng.integers(0, N_STATES))
    obs_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    cum_list: list[float] = []
    for _ in range(n_steps):
        next_state = (state + 1) % N_STATES
        obs_list.append(eye[state])
        next_obs_list.append(eye[next_state])
        cum_list.append(1.0)
        state = next_state
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(cum_list, dtype=np.float32),
        np.asarray(next_obs_list, dtype=np.float32),
    )


def _score_predictions(
    predictions: np.ndarray, cumulants: np.ndarray, gamma: float
) -> dict[str, float]:
    """Score a prediction trace against the forward-view and analytic return."""
    fv = forward_view_returns(
        jnp.asarray(cumulants), gamma=gamma, terminal_value=0.0
    )
    fv_np = np.asarray(fv)
    burn_head = min(200, max(0, len(predictions) // 4))
    burn_tail = min(50, max(0, (len(predictions) - burn_head - 1) // 4))
    sl = slice(burn_head, len(predictions) - burn_tail)
    rmse = float(np.sqrt(np.mean((predictions[sl] - fv_np[sl]) ** 2)))

    analytic = 1.0 / (1.0 - gamma) if gamma < 1.0 else float("inf")
    final_pred = float(np.mean(predictions[-100:]))
    abs_err = (
        abs(final_pred - analytic) if np.isfinite(analytic) else float("nan")
    )
    return {
        "rmse": rmse,
        "final_pred": final_pred,
        "analytic": analytic,
        "abs_err": abs_err,
    }


def make_horde() -> MixedHorde:
    demons = [
        GVFSpec(
            name=f"demon_g{g:.2f}",
            demon_type=DemonType.PREDICTION,
            gamma=g,
            lamda=lam,
            cumulant_index=i,
        )
        for i, (g, lam) in enumerate(zip(GAMMAS, LAMDAS, strict=True))
    ]
    spec = create_horde_spec(demons)
    return MixedHorde(
        horde_spec=spec,
        hidden_sizes=(16,),
        step_size=STEP_SIZE,
        sparsity=0.0,
        bounder=ObGDBounding(kappa=2.0),
    )


def collect_predictions(
    horde: MixedHorde,
    state,
    observations: np.ndarray,
) -> np.ndarray:
    """Materialize per-step predictions over the trajectory."""
    obs_jax = jnp.asarray(observations)
    preds = jnp.stack(
        [horde.predict(state, obs_jax[t]) for t in range(observations.shape[0])]
    )
    return np.asarray(preds)


def run_seed(seed: int) -> dict[str, object]:
    obs_np, cums_np, next_obs_np = generate_chain(seed, N_STEPS)
    obs = jnp.asarray(obs_np)
    cums_per_demon = jnp.broadcast_to(
        jnp.asarray(cums_np)[:, None], (N_STEPS, len(GAMMAS))
    )
    next_obs = jnp.asarray(next_obs_np)

    horde = make_horde()
    init_state = horde.init(N_STATES, jr.key(seed))

    t0 = time.time()
    result = run_mixed_horde_learning_loop(
        horde, init_state, obs, cums_per_demon, next_obs
    )
    runtime = time.time() - t0

    finite_metrics = bool(jnp.all(jnp.isfinite(result.per_demon_metrics)))
    finite_errors = bool(jnp.all(jnp.isfinite(result.td_errors)))

    # Pull predictions step-by-step from the FINAL state. This exercises
    # the predict path on the trained network.
    predictions = collect_predictions(horde, result.state, obs_np)

    per_gamma: dict[str, dict[str, float]] = {}
    for i, gamma in enumerate(GAMMAS):
        per_gamma[f"gamma={gamma:.2f}"] = _score_predictions(
            predictions[:, i], cums_np, gamma
        )

    return {
        "seed": seed,
        "n_steps": N_STEPS,
        "runtime_s": runtime,
        "finite_metrics": finite_metrics,
        "finite_errors": finite_errors,
        "per_gamma": per_gamma,
        "shared_indices": list(horde.shared_indices),
        "independent_indices": list(horde.independent_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS,
        help="Number of seeds to run (default: %(default)s)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[dict[str, object]] = []
    for seed in range(args.n_seeds):
        per_seed_results.append(run_seed(seed))

    # Aggregate per-gamma RMSE across seeds.
    summary: dict[str, dict[str, float]] = {}
    for gamma_key in per_seed_results[0]["per_gamma"]:
        rmses = np.asarray(
            [
                seed_result["per_gamma"][gamma_key]["rmse"]
                for seed_result in per_seed_results
            ]
        )
        finals = np.asarray(
            [
                seed_result["per_gamma"][gamma_key]["final_pred"]
                for seed_result in per_seed_results
            ]
        )
        analytic = per_seed_results[0]["per_gamma"][gamma_key]["analytic"]
        summary[gamma_key] = {
            "rmse_mean": float(rmses.mean()),
            "rmse_std": float(rmses.std()),
            "final_pred_mean": float(finals.mean()),
            "final_pred_std": float(finals.std()),
            "analytic": float(analytic) if np.isfinite(analytic) else float("nan"),
        }

    payload = {
        "config": {
            "n_seeds": args.n_seeds,
            "n_states": N_STATES,
            "n_steps": N_STEPS,
            "step_size": STEP_SIZE,
            "gammas": list(GAMMAS),
            "lamdas": list(LAMDAS),
            "shared_indices": per_seed_results[0]["shared_indices"],
            "independent_indices": per_seed_results[0]["independent_indices"],
        },
        "summary": summary,
        "per_seed": per_seed_results,
    }

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=False))

    print(f"Wrote {out_file}")
    print(
        "Routing partition: "
        f"shared={per_seed_results[0]['shared_indices']}, "
        f"independent={per_seed_results[0]['independent_indices']}"
    )
    for gamma_key, stats in summary.items():
        print(
            f"  {gamma_key}: rmse={stats['rmse_mean']:.4f} "
            f"+- {stats['rmse_std']:.4f}, "
            f"final_pred={stats['final_pred_mean']:.4f} "
            f"+- {stats['final_pred_std']:.4f} "
            f"(analytic={stats['analytic']})"
        )


if __name__ == "__main__":
    main()

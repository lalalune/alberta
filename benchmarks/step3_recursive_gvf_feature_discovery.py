#!/usr/bin/env python3
r"""Bounded recursive GVF feature-discovery benchmark for Step 3.

The stream is partially observable.  A latent trace ``z_{t+1}`` is driven by
the current visible observation ``x_t``.  The downstream transition cumulant is
``x_t * z_{t+1}``, so a raw linear learner on ``x_t`` alone cannot represent it,
but a recursive learner can first learn the auxiliary GVF ``z_{t+1}`` and then
reuse ``x_t * \hat z_{t+1}`` as a discovered feature.

This is deliberately a bounded constructive benchmark: it proves that the
implemented TD/GVF recursion can discover and reuse useful predictive features
on a seeded hidden-state process.  It does not claim arbitrary representation
universality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    DemonType,
    GVFSpec,
    HordeLearner,
    ObGDBounding,
    create_horde_spec,
    multi_channel_horizon_returns,
)

DEFAULT_OUTPUT = Path("outputs/step3_recursive_gvf_feature_discovery/results.json")


def collect_hidden_trace_stream(
    *,
    seed: int,
    total_steps: int,
    ar_rho: float = 0.82,
    latent_rho: float = 0.0,
    latent_drive: float = 1.0,
    noise_std: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visible observations, auxiliary latent cumulants, and targets."""
    key = jr.key(seed)
    key, k_x0, k_z0 = jr.split(key, 3)
    x = float(jr.normal(k_x0, (), dtype=jnp.float32))
    z = float(jr.normal(k_z0, (), dtype=jnp.float32))
    x_prev = x
    obs: list[list[float]] = []
    aux: list[list[float]] = []
    targets: list[list[float]] = []
    x_innovation = float(np.sqrt(max(1.0 - ar_rho**2, 0.0)))

    for step in range(total_steps):
        if step > 0:
            key, kx, kz = jr.split(key, 3)
            eps_x = float(jr.normal(kx, (), dtype=jnp.float32))
            eps_z = float(noise_std * jr.normal(kz, (), dtype=jnp.float32))
            x_new = float(np.tanh(ar_rho * x + x_innovation * eps_x))
            z = float(np.tanh(latent_rho * z + latent_drive * x_prev + eps_z))
            x_prev = x
            x = x_new

        obs.append([x])
        aux.append([z])
        targets.append([x_prev * z])

    return (
        np.asarray(obs, dtype=np.float32),
        np.asarray(aux, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def history_features(observations: np.ndarray, rhos: tuple[float, ...]) -> np.ndarray:
    """Append causal EMA traces of visible observations."""
    traces = np.zeros((len(rhos), observations.shape[1]), dtype=np.float32)
    rows: list[np.ndarray] = []
    for obs in observations:
        pieces = [obs.astype(np.float32)]
        for idx, rho in enumerate(rhos):
            traces[idx] = rho * traces[idx] + (1.0 - rho) * obs
            pieces.append(traces[idx].copy())
        rows.append(np.concatenate(pieces))
    return np.asarray(rows, dtype=np.float32)


def make_prediction_horde(
    *,
    n_cumulants: int,
    gammas: tuple[float, ...],
    feature_dim: int,
    step_size: float,
    key: jnp.ndarray,
) -> tuple[HordeLearner, Any]:
    specs: list[GVFSpec] = []
    for cumulant_idx in range(n_cumulants):
        for gamma in gammas:
            specs.append(
                GVFSpec(  # type: ignore[call-arg]
                    name=f"c{cumulant_idx}_g{gamma:g}",
                    demon_type=DemonType.PREDICTION,
                    gamma=gamma,
                    lamda=0.0,
                    cumulant_index=cumulant_idx,
                )
            )
    learner = HordeLearner(
        create_horde_spec(specs),
        hidden_sizes=(),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        use_layer_norm=False,
    )
    return learner, learner.init(feature_dim, key)


def run_horde(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    cumulants: np.ndarray,
    gammas: tuple[float, ...],
    key: jnp.ndarray,
    step_size: float,
) -> np.ndarray:
    learner, state = make_prediction_horde(
        n_cumulants=cumulants.shape[1],
        gammas=gammas,
        feature_dim=observations.shape[1],
        step_size=step_size,
        key=key,
    )
    repeated_cumulants = np.repeat(cumulants, len(gammas), axis=1).astype(np.float32)
    predictions = np.zeros_like(repeated_cumulants)
    for t in range(observations.shape[0]):
        result = learner.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(repeated_cumulants[t]),
            jnp.asarray(next_observations[t]),
        )
        predictions[t] = np.asarray(result.predictions)
        state = result.state
    return predictions


def returns_for(cumulants: np.ndarray, gammas: tuple[float, ...]) -> np.ndarray:
    returns = multi_channel_horizon_returns(
        jnp.asarray(cumulants, dtype=jnp.float32),
        jnp.asarray(gammas, dtype=jnp.float32),
        terminal_value=0.0,
    )
    return np.asarray(returns, dtype=np.float32).reshape(cumulants.shape[0], -1)


def tail_rmse(
    predictions: np.ndarray,
    returns: np.ndarray,
    *,
    burn_in: int,
    burn_tail: int,
) -> float:
    end = predictions.shape[0] - burn_tail if burn_tail else predictions.shape[0]
    err = predictions[burn_in:end] - returns[burn_in:end]
    return float(np.sqrt(np.mean(err**2)))


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    total_steps = args.discovery_steps + args.eval_steps + 1
    obs, aux, targets = collect_hidden_trace_stream(
        seed=seed,
        total_steps=total_steps,
        ar_rho=args.ar_rho,
        latent_rho=args.latent_rho,
        latent_drive=args.latent_drive,
        noise_std=args.noise_std,
    )
    obs_t = obs[:-1]
    obs_tp1 = obs[1:]
    aux_tp1 = aux[1:]
    target_tp1 = targets[1:]

    disc = slice(0, args.discovery_steps)
    eval_slice = slice(args.discovery_steps, None)
    hist = history_features(obs, args.history_rhos)
    hist_t = hist[:-1]
    hist_tp1 = hist[1:]

    key = jr.key(seed + 50_000)
    k_raw, k_hist, k_aux, k_recursive = jr.split(key, 4)

    raw_preds = run_horde(
        observations=obs_t[eval_slice],
        next_observations=obs_tp1[eval_slice],
        cumulants=target_tp1[eval_slice],
        gammas=args.gammas,
        key=k_raw,
        step_size=args.downstream_step_size,
    )
    hist_preds = run_horde(
        observations=hist_t[eval_slice],
        next_observations=hist_tp1[eval_slice],
        cumulants=target_tp1[eval_slice],
        gammas=args.gammas,
        key=k_hist,
        step_size=args.downstream_step_size,
    )

    aux_all_preds = run_horde(
        observations=hist_t,
        next_observations=hist_tp1,
        cumulants=aux_tp1,
        gammas=args.aux_gammas,
        key=k_aux,
        step_size=args.aux_step_size,
    )
    aux_features = aux_all_preds[:, : len(args.aux_gammas)].astype(np.float32)
    recursive_obs = np.concatenate([obs_t, aux_features, obs_t * aux_features], axis=1)
    next_aux_features = np.roll(aux_features, shift=-1, axis=0)
    recursive_next = np.concatenate(
        [obs_tp1, next_aux_features, obs_tp1 * next_aux_features],
        axis=1,
    ).astype(np.float32)

    recursive_preds = run_horde(
        observations=recursive_obs[eval_slice],
        next_observations=recursive_next[eval_slice],
        cumulants=target_tp1[eval_slice],
        gammas=args.gammas,
        key=k_recursive,
        step_size=args.downstream_step_size,
    )
    eval_returns = returns_for(target_tp1[eval_slice], args.gammas)

    raw_rmse = tail_rmse(raw_preds, eval_returns, burn_in=args.burn_in, burn_tail=args.burn_tail)
    hist_rmse = tail_rmse(hist_preds, eval_returns, burn_in=args.burn_in, burn_tail=args.burn_tail)
    recursive_rmse = tail_rmse(
        recursive_preds,
        eval_returns,
        burn_in=args.burn_in,
        burn_tail=args.burn_tail,
    )
    aux_returns = returns_for(aux_tp1[disc], args.aux_gammas)
    aux_rmse = tail_rmse(
        aux_all_preds[disc],
        aux_returns,
        burn_in=min(args.burn_in, max(args.discovery_steps // 5, 1)),
        burn_tail=0,
    )
    return {
        "seed": seed,
        "raw_linear_rmse": raw_rmse,
        "history_linear_rmse": hist_rmse,
        "recursive_gvf_rmse": recursive_rmse,
        "auxiliary_gvf_discovery_rmse": aux_rmse,
        "recursive_lift_vs_raw": raw_rmse - recursive_rmse,
        "recursive_lift_vs_history": hist_rmse - recursive_rmse,
        "recursive_feature_dim": int(recursive_obs.shape[1]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lifts_raw = np.asarray([r["recursive_lift_vs_raw"] for r in rows], dtype=np.float64)
    lifts_hist = np.asarray([r["recursive_lift_vs_history"] for r in rows], dtype=np.float64)
    recursive_rmse = np.asarray([r["recursive_gvf_rmse"] for r in rows], dtype=np.float64)
    raw_rmse = np.asarray([r["raw_linear_rmse"] for r in rows], dtype=np.float64)
    history_rmse = np.asarray([r["history_linear_rmse"] for r in rows], dtype=np.float64)
    return {
        "n_seeds": len(rows),
        "raw_linear_rmse_mean": float(np.mean(raw_rmse)),
        "history_linear_rmse_mean": float(np.mean(history_rmse)),
        "recursive_gvf_rmse_mean": float(np.mean(recursive_rmse)),
        "recursive_lift_vs_raw_mean": float(np.mean(lifts_raw)),
        "recursive_lift_vs_history_mean": float(np.mean(lifts_hist)),
        "recursive_wins_vs_raw": int(np.sum(lifts_raw > 0.0)),
        "recursive_wins_vs_history": int(np.sum(lifts_hist > 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--discovery-steps", type=int, default=900)
    parser.add_argument("--eval-steps", type=int, default=1200)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--burn-tail", type=int, default=25)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--aux-gammas", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--history-rhos", type=float, nargs="+", default=[0.6, 0.85, 0.97])
    parser.add_argument("--ar-rho", type=float, default=0.82)
    parser.add_argument("--latent-rho", type=float, default=0.0)
    parser.add_argument("--latent-drive", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--aux-step-size", type=float, default=0.01)
    parser.add_argument("--downstream-step-size", type=float, default=0.025)
    args = parser.parse_args()
    args.gammas = tuple(float(v) for v in args.gammas)
    args.aux_gammas = tuple(float(v) for v in args.aux_gammas)
    args.history_rhos = tuple(float(v) for v in args.history_rhos)
    return args


def main() -> int:
    args = parse_args()
    rows = [run_seed(seed, args) for seed in range(args.seeds)]
    aggregate = summarize(rows)
    passed = bool(
        aggregate["n_seeds"] >= 10
        and aggregate["recursive_wins_vs_raw"] == aggregate["n_seeds"]
        and aggregate["recursive_wins_vs_history"] >= aggregate["n_seeds"] - 1
        and aggregate["recursive_lift_vs_raw_mean"] > 0.02
        and aggregate["recursive_lift_vs_history_mean"] >= 0.0
    )
    payload = {
        "schema": "alberta.step3.recursive_gvf_feature_discovery.v1",
        "claim_scope": "bounded_recursive_td_gvf_feature_reuse",
        "passed": passed,
        "aggregate": aggregate,
        "rows": rows,
        "config": {
            "seeds": args.seeds,
            "discovery_steps": args.discovery_steps,
            "eval_steps": args.eval_steps,
            "gammas": list(args.gammas),
            "aux_gammas": list(args.aux_gammas),
            "history_rhos": list(args.history_rhos),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"passed={passed} output={args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

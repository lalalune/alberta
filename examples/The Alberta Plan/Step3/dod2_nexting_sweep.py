"""DoD-2: Multi-timescale nexting multi-seed sweep.

Compares per-horizon prediction error of linear TD, TrueOnlineTD(lambda),
Horde head-only traces, and CBP-augmented Horde at several gamma values on
a deterministic chain with known forward-view returns.

Setup
-----
A 5-state cyclic chain emitting cumulant=1 at every step (so the
forward-view return at gamma=g is 1 / (1 - g) in steady state). We train
each demon (one per gamma) and measure how its predictions track the
analytic return as steps proceed.

Conditions:
- Methods: TDLinear TD(0), TDLinear TD(lambda), TrueOnlineTD(lambda),
  MLPHorde head-only traces, MLPHorde head-only traces + CBP.
- Lambda values: {0.0, 0.5, 0.9} for the lambda-sweep methods.
- Gamma horizons: {0.0, 0.5, 0.9, 0.99}
- 12 seeds each
- 4000 steps per seed

Output: ``output/step3_dod2/results.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    TDIDBD,
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    DemonType,
    GVFSpec,
    HordeLearner,
    TDLinearLearner,
    TrueOnlineTDLearner,
    create_horde_spec,
    forward_view_returns,
)

OUTPUT_DIR = Path("output/step3_dod2")
N_SEEDS = 12
N_STATES = 5
N_STEPS = 4000
ALPHA = 0.05
MLP_ALPHA = 0.01


def generate_chain(seed: int, n_steps: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """5-state cyclic chain: deterministic move forward, cumulant=1
    every step. Generates obs, cumulants, next_obs, gammas (all gamma=g
    for non-terminal continuing setting)."""
    eye = np.eye(N_STATES, dtype=np.float32)
    rng = np.random.default_rng(seed)
    state = int(rng.integers(0, N_STATES))
    obs_list, next_obs_list, cum_list = [], [], []
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
        np.full(n_steps, np.nan, dtype=np.float32),  # gammas filled below
    )


def _score_predictions(
    predictions: np.ndarray, cumulants: np.ndarray, gamma: float
) -> dict[str, float]:
    """Score a prediction trace against forward-view and analytic returns."""
    fv = forward_view_returns(jnp.asarray(cumulants), gamma=gamma, terminal_value=0.0)
    fv_np = np.asarray(fv)
    burn_head = min(200, max(0, len(predictions) // 4))
    burn_tail = min(50, max(0, (len(predictions) - burn_head - 1) // 4))
    sl = slice(burn_head, len(predictions) - burn_tail)
    rmse = float(np.sqrt(np.mean((predictions[sl] - fv_np[sl]) ** 2)))

    analytic = 1.0 / (1.0 - gamma) if gamma < 1.0 else float("inf")
    final_pred = float(np.mean(predictions[-100:]))
    abs_err = abs(final_pred - analytic) if np.isfinite(analytic) else float("nan")
    return {
        "rmse": rmse,
        "final_pred": final_pred,
        "analytic": analytic,
        "abs_err_vs_analytic": abs_err,
    }


def run_true_online(seed: int, lam: float, gamma: float, n_steps: int) -> dict[str, float]:
    obs, cum, nxt, _ = generate_chain(seed, n_steps)
    gammas = np.full(n_steps, gamma, dtype=np.float32)

    learner = TrueOnlineTDLearner(step_size=ALPHA, trace_decay=lam)
    state = learner.init(N_STATES)

    predictions = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        pred = float(jnp.squeeze(learner.predict(state, jnp.asarray(obs[t]))))
        predictions[t] = pred
        res = learner.update(
            state,
            jnp.asarray(obs[t]),
            jnp.asarray(cum[t]),
            jnp.asarray(nxt[t]),
            jnp.asarray(gammas[t]),
        )
        state = res.state

    return _score_predictions(predictions, cum, gamma)


def run_td_linear(seed: int, lam: float, gamma: float, n_steps: int) -> dict[str, float]:
    obs, cum, nxt, _ = generate_chain(seed, n_steps)
    gammas = np.full(n_steps, gamma, dtype=np.float32)

    learner = TDLinearLearner(
        optimizer=TDIDBD(
            initial_step_size=ALPHA,
            meta_step_size=0.0,
            trace_decay=lam,
        )
    )
    state = learner.init(N_STATES)

    predictions = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        pred = float(jnp.squeeze(learner.predict(state, jnp.asarray(obs[t]))))
        predictions[t] = pred
        res = learner.update(
            state,
            jnp.asarray(obs[t]),
            jnp.asarray(cum[t]),
            jnp.asarray(nxt[t]),
            jnp.asarray(gammas[t]),
        )
        state = res.state

    return _score_predictions(predictions, cum, gamma)


def run_horde(seed: int, lam: float, gamma: float, n_steps: int, cbp: bool) -> dict[str, float]:
    obs, cum, nxt, _ = generate_chain(seed, n_steps)
    demons = (
        GVFSpec(  # type: ignore[call-arg]
            name=f"d_g{gamma}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lam,
            cumulant_index=0,
        ),
    )
    spec = create_horde_spec(demons)

    if cbp:
        learner: Any
        learner = CBPMultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=(32,),
            cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
                enabled=True,
                replacement_rate=1e-4,
                maturity_threshold=200,
            ),
            step_size=MLP_ALPHA,
            gamma=0.0,
            lamda=0.0,
            per_head_gamma_lamda=(gamma * lam,),
            sparsity=0.5,
            use_layer_norm=True,
        )
        state = learner.init(N_STATES, jr.key(seed + 10_000))
    else:
        learner = HordeLearner(
            horde_spec=spec,
            hidden_sizes=(32,),
            step_size=MLP_ALPHA,
            sparsity=0.5,
            use_layer_norm=True,
        )
        state = learner.init(N_STATES, jr.key(seed + 10_000))

    predictions = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        obs_t = jnp.asarray(obs[t])
        pred = learner.predict(state, obs_t)
        predictions[t] = float(np.asarray(pred)[0])
        if cbp:
            next_pred = learner.predict(state, jnp.asarray(nxt[t]))
            target = jnp.asarray([cum[t] + gamma * float(np.asarray(next_pred)[0])])
            res = learner.update(state, obs_t, target)
        else:
            res = learner.update(state, obs_t, jnp.asarray([cum[t]]), jnp.asarray(nxt[t]))
        state = res.state

    return _score_predictions(predictions, cum, gamma)


def run_one(
    seed: int, method: str, lam: float, gamma: float, n_steps: int
) -> dict[str, float | str]:
    if method == "tdlinear_td0":
        metrics = run_td_linear(seed, 0.0, gamma, n_steps)
        lam = 0.0
    elif method == "tdlinear_tdlambda":
        metrics = run_td_linear(seed, lam, gamma, n_steps)
    elif method == "true_online_tdlambda":
        metrics = run_true_online(seed, lam, gamma, n_steps)
    elif method == "mlp_horde_head_traces":
        metrics = run_horde(seed, lam, gamma, n_steps, cbp=False)
    elif method == "mlp_horde_head_traces_cbp":
        metrics = run_horde(seed, lam, gamma, n_steps, cbp=True)
    else:
        raise ValueError(f"unknown method: {method}")

    return {"method": method, "seed": seed, "lambda": lam, "gamma": gamma, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "tdlinear_td0",
            "tdlinear_tdlambda",
            "true_online_tdlambda",
            "mlp_horde_head_traces",
            "mlp_horde_head_traces_cbp",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float | str]] = []
    lambdas = [0.0, 0.5, 0.9]
    gammas = [0.0, 0.5, 0.9, 0.99]

    for method in args.methods:
        method_lambdas = [0.0] if method == "tdlinear_td0" else lambdas
        for lam in method_lambdas:
            for gamma in gammas:
                for seed in range(args.n_seeds):
                    row = run_one(seed, method, lam, gamma, args.n_steps)
                    rows.append(row)
                    print(
                        f"{method} lam={row['lambda']:.1f} gamma={gamma:.2f} "
                        f"seed={seed:>2} rmse={row['rmse']:.4f} "
                        f"final={row['final_pred']:.3f} target={row['analytic']:.3f}"
                    )

    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float | str]] = {}
    for method in args.methods:
        method_lambdas = [0.0] if method == "tdlinear_td0" else lambdas
        for lam in method_lambdas:
            for gamma in gammas:
                sub = [
                    r
                    for r in rows
                    if r["method"] == method and r["lambda"] == lam and r["gamma"] == gamma
                ]
                if not sub:
                    continue
                rmses = [float(r["rmse"]) for r in sub]
                errs = [
                    float(r["abs_err_vs_analytic"])
                    for r in sub
                    if not np.isnan(float(r["abs_err_vs_analytic"]))
                ]
                summary[f"{method}_lam={lam}_gamma={gamma}"] = {
                    "method": method,
                    "lambda": lam,
                    "gamma": gamma,
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "abs_err_mean": float(np.mean(errs)) if errs else float("nan"),
                }

    t_total = time.perf_counter() - t0
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "total_seconds": t_total,
                "n_seeds": args.n_seeds,
                "n_steps": args.n_steps,
                "methods": args.methods,
            },
            f,
            indent=2,
        )

    print("\n=== DoD-2 summary (RMSE mean across seeds) ===")
    for method in args.methods:
        print(f"\n{method}")
        method_lambdas = [0.0] if method == "tdlinear_td0" else lambdas
        print(f"{'gamma':>8}  " + "  ".join(f"lam={lam:>3.1f}" for lam in method_lambdas))
        for gamma in gammas:
            print_row: list[str] = [f"{gamma:>8.2f}  "]
            for lam in method_lambdas:
                r = summary[f"{method}_lam={lam}_gamma={gamma}"]["rmse_mean"]
                print_row.append(f"{r:>7.4f}")
            print(" ".join(print_row))
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

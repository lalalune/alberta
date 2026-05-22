"""DoD-8: 200k-step plasticity sustained sweep.

Replicates the central claim of Dohare et al. (Nature 2024): standard
deep continual learning loses plasticity over time, and Continual
Backprop (CBP) preserves it.

Setup
-----
- Non-stationary supervised stream: cumulant pattern reverses every
  10000 steps (sign flip of half the regression weights). The learner
  must re-fit each context.
- 64-2 MLP shared-trunk MultiHeadMLPLearner (1 head).
- 200,000 total steps (20 contexts).
- Conditions:
  * cbp_off -- standard MultiHeadMLPLearner
  * cbp_on  -- CBPMultiHeadMLPLearner with replacement_rate=1e-4,
               maturity_threshold=200, decay_rate=0.99
- 5 seeds (this is heavy; we keep the seed count modest).

Metric
------
Final-context squared error and hidden-activation effective-rank
trajectory. Lower final-context squared error and sustained rank are
the intended CBP-positive signature.
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
    LMS,
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    MultiHeadMLPLearner,
    ObGDBounding,
)

OUTPUT_DIR = Path("output/step3_dod8")
N_SEEDS = 5
N_STEPS = 200_000
CONTEXT_LEN = 10_000
FEATURE_DIM = 8
HIDDEN_SIZES = (64,)
ALPHA = 0.05
RANK_INTERVAL = 10_000
RANK_BATCH_SIZE = 512


def generate_stream(
    seed: int, n_steps: int, context_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Non-stationary regression: target = w_t @ obs + noise. w_t flips
    sign on half its components every ``context_len`` steps.
    """
    rng = np.random.default_rng(seed)
    base_w = rng.normal(size=FEATURE_DIM).astype(np.float32)
    flip_mask = np.zeros(FEATURE_DIM, dtype=np.float32)
    flip_mask[: FEATURE_DIM // 2] = 1.0  # first half flips

    obs_l = []
    targets_l = []
    for t in range(n_steps):
        ctx = t // context_len
        # Flip sign of half the weights every other context
        if ctx % 2 == 1:
            w = base_w * (1.0 - 2.0 * flip_mask)  # flip first half
        else:
            w = base_w
        obs = rng.normal(size=FEATURE_DIM).astype(np.float32)
        target = float(np.dot(w, obs)) + 0.05 * float(rng.normal())
        obs_l.append(obs)
        targets_l.append(target)
    return (
        np.asarray(obs_l, dtype=np.float32),
        np.asarray(targets_l, dtype=np.float32),
    )


def _trunk_params_from_state(state: Any) -> Any:
    if hasattr(state, "mlp_state"):
        return state.mlp_state.trunk_params
    return state.trunk_params


def _hidden_activation_ranks(
    learner: Any,
    state: Any,
    observations: np.ndarray,
) -> list[dict[str, float | int]]:
    trunk_params = _trunk_params_from_state(state)
    base_learner = learner.learner if hasattr(learner, "learner") else learner
    x = jnp.asarray(observations)
    ranks: list[dict[str, float | int]] = []
    for layer_idx, (weights, biases) in enumerate(
        zip(trunk_params.weights, trunk_params.biases)
    ):
        x = x @ weights.T + biases
        if base_learner._use_layer_norm:
            mean = jnp.mean(x, axis=1, keepdims=True)
            var = jnp.var(x, axis=1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = jnp.where(x >= 0, x, base_learner._leaky_relu_slope * x)

        centered = x - jnp.mean(x, axis=0, keepdims=True)
        singular_values = jnp.linalg.svd(centered, compute_uv=False)
        squared = jnp.square(singular_values)
        fro_sq = jnp.sum(squared)
        spectral_sq = jnp.max(squared) if squared.size else jnp.array(0.0)
        stable_rank = jnp.where(spectral_sq > 0.0, fro_sq / spectral_sq, 0.0)
        mass = jnp.sum(singular_values)
        probs = jnp.where(mass > 0.0, singular_values / mass, 0.0)
        entropy = -jnp.sum(jnp.where(probs > 0.0, probs * jnp.log(probs), 0.0))
        ranks.append(
            {
                "layer": layer_idx,
                "batch_size": int(centered.shape[0]),
                "width": int(centered.shape[1]),
                "stable_rank": float(stable_rank),
                "effective_rank": float(jnp.exp(entropy)),
            }
        )
    return ranks


def run_one(
    seed: int,
    cbp_on: bool,
    *,
    n_steps: int,
    context_len: int,
    rank_interval: int,
    rank_batch_size: int,
    cbp_replacement_rate: float,
    cbp_maturity_threshold: int,
    cbp_decay_rate: float,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    obs, targets = generate_stream(seed, n_steps, context_len)

    bounder = ObGDBounding(kappa=2.0)
    if cbp_on:
        cbp_config = ContinualBackpropConfig(  # type: ignore[call-arg]
            decay_rate=cbp_decay_rate,
            replacement_rate=cbp_replacement_rate,
            maturity_threshold=cbp_maturity_threshold,
            enabled=True,
        )
        learner: Any = CBPMultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=HIDDEN_SIZES,
            optimizer=LMS(step_size=ALPHA),
            bounder=bounder,
            sparsity=0.0,
            cbp_config=cbp_config,
        )
    else:
        learner = MultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=HIDDEN_SIZES,
            optimizer=LMS(step_size=ALPHA),
            bounder=bounder,
            sparsity=0.0,
        )

    state = learner.init(FEATURE_DIM, jr.key(seed + 1000))

    sq_err_per_step = np.zeros(n_steps, dtype=np.float32)
    rank_rows: list[dict[str, float | int]] = []
    replacements_total = 0
    for t in range(n_steps):
        target_arr = jnp.atleast_1d(jnp.float32(targets[t]))
        res = learner.update(state, jnp.asarray(obs[t]), target_arr)
        state = res.state
        # error is shape (n_heads,) for multi-head; squared
        sq_err_per_step[t] = float(jnp.squeeze(res.errors[0]) ** 2)
        if cbp_on and hasattr(res, "replacements_made"):
            replacements_total += int(jnp.sum(res.replacements_made))
        if rank_interval > 0 and (t + 1) % rank_interval == 0:
            start = max(0, t + 1 - rank_batch_size)
            for rank in _hidden_activation_ranks(learner, state, obs[start : t + 1]):
                rank_rows.append(
                    {
                        "seed": seed,
                        "cbp_on": int(cbp_on),
                        "step": t + 1,
                        "layer": int(rank["layer"]),
                        "batch_size": int(rank["batch_size"]),
                        "width": int(rank["width"]),
                        "stable_rank": float(rank["stable_rank"]),
                        "effective_rank": float(rank["effective_rank"]),
                    }
                )

    # Per-context final-window MSE (last 500 steps of each context)
    n_ctx = n_steps // context_len
    per_ctx_mse = np.zeros(n_ctx, dtype=np.float32)
    for c in range(n_ctx):
        end = (c + 1) * context_len
        per_ctx_mse[c] = float(np.mean(sq_err_per_step[end - 500:end]))

    return {
        "seed": seed,
        "cbp_on": int(cbp_on),
        "final_window_mse": float(np.mean(sq_err_per_step[-1000:])),
        "first_context_mse": float(per_ctx_mse[0]),
        "last_context_mse": float(per_ctx_mse[-1]),
        "per_context_mse_max": float(np.max(per_ctx_mse[1:])),
        "per_context_mse_mean_after_first": float(np.mean(per_ctx_mse[1:])),
        "n_contexts": float(n_ctx),
        "replacements_total": float(replacements_total),
        "final_effective_rank": (
            float(rank_rows[-1]["effective_rank"]) if rank_rows else float("nan")
        ),
    }, rank_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    parser.add_argument(
        "--rank-interval",
        type=int,
        default=RANK_INTERVAL,
        help="Steps between effective-rank samples; set to 0 to disable.",
    )
    parser.add_argument("--rank-batch-size", type=int, default=RANK_BATCH_SIZE)
    parser.add_argument("--cbp-replacement-rate", type=float, default=1e-4)
    parser.add_argument("--cbp-maturity-threshold", type=int, default=200)
    parser.add_argument("--cbp-decay-rate", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, float]] = []
    rank_rows: list[dict[str, float | int]] = []
    for cbp_on in [False, True]:
        for seed in range(args.seeds):
            t_run = time.perf_counter()
            row, ranks = run_one(
                seed,
                cbp_on,
                n_steps=args.steps,
                context_len=args.context_len,
                rank_interval=args.rank_interval,
                rank_batch_size=args.rank_batch_size,
                cbp_replacement_rate=args.cbp_replacement_rate,
                cbp_maturity_threshold=args.cbp_maturity_threshold,
                cbp_decay_rate=args.cbp_decay_rate,
            )
            rt = time.perf_counter() - t_run
            rows.append(row)
            rank_rows.extend(ranks)
            print(
                f"cbp_on={str(bool(cbp_on)):>5} seed={seed:>2}  "
                f"first_ctx={row['first_context_mse']:.4f}  "
                f"last_ctx={row['last_context_mse']:.4f}  "
                f"mean_after_first={row['per_context_mse_mean_after_first']:.4f}  "
                f"rank={row['final_effective_rank']:.2f}  "
                f"repl={row['replacements_total']:.0f}  "
                f"({rt:.1f}s)"
            )

    csv_path = args.output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rank_path = args.output_dir / "effective_rank_trajectory.csv"
    with open(rank_path, "w", newline="") as f:
        if rank_rows:
            writer = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rank_rows)

    summary: dict[str, dict[str, float]] = {}
    for cbp_enabled in (False, True):
        sub = [r for r in rows if bool(r["cbp_on"]) == cbp_enabled]
        first = [r["first_context_mse"] for r in sub]
        last = [r["last_context_mse"] for r in sub]
        mean_after = [r["per_context_mse_mean_after_first"] for r in sub]
        final_window = [r["final_window_mse"] for r in sub]
        final_rank = [r["final_effective_rank"] for r in sub]
        replacements = [r["replacements_total"] for r in sub]
        summary[f"cbp_on={cbp_enabled}"] = {
            "first_ctx_mean": float(np.mean(first)),
            "last_ctx_mean": float(np.mean(last)),
            "mean_after_first_ctx": float(np.mean(mean_after)),
            "final_window_mse_mean": float(np.mean(final_window)),
            "final_window_mse_std": float(np.std(final_window)),
            "final_effective_rank_mean": float(np.mean(final_rank)),
            "replacements_total_mean": float(np.mean(replacements)),
            "plasticity_loss_factor": float(
                np.mean(last) / np.mean(first) if np.mean(first) > 0 else float("nan")
            ),
        }

    t_total = time.perf_counter() - t0
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": vars(args) | {"output_dir": str(args.output_dir)},
                "summary": summary,
                "total_seconds": t_total,
            },
            f,
            indent=2,
        )

    print("\n=== DoD-8 summary (plasticity sustained) ===")
    for k, s in summary.items():
        print(
            f"  {k:>14}  first_ctx={s['first_ctx_mean']:.4f}  "
            f"last_ctx={s['last_ctx_mean']:.4f}  "
            f"loss_factor={s['plasticity_loss_factor']:.2f}x  "
            f"final_window={s['final_window_mse_mean']:.4f}  "
            f"rank={s['final_effective_rank_mean']:.2f}  "
            f"repl={s['replacements_total_mean']:.1f}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path}, {rank_path}, and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

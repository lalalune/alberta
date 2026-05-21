"""DoD-8 v2: discriminating CBP sweep — ReLU + task-shift variants.

Resolves the DoD-8 ambiguity from the 200k sign-flip probe. The
original artifact showed CBP preserved effective rank (26 vs ~9) but
*did not* improve final-window MSE on the sign-flip stream. This v2
sweep tests whether CBP helps on streams where the original assumption
("the network can re-fit by reweighting existing units") breaks down.

Hypothesis
----------
CBP's home turf, per Dohare et al. (Nature 2024), is ReLU networks
where dead units accumulate. The sign-flip stream lets a leaky-ReLU
network re-fit by reweighting; CBP cannot show benefit there. This v2
varies the regime along two axes that should expose CBP-positive
signal:

* Variant A — ReLU activation on the same sign-flip stream.
  ``leaky_relu_slope=0.0`` makes negative pre-activations exactly zero
  and creates true "dead" units that cannot recover via gradient flow.
  CBP should now be the only mechanism that revives lost capacity.

* Variant B — Task-shift stream where every ``context_len`` steps the
  target switches to use a *different feature subspace* (a permuted
  contiguous slice of the input dims with previously inactive
  features), forcing the network to grow new representations that the
  prior context could not have learned. Uses leaky-ReLU like the
  original DoD-8 to isolate the stream change.

Conditions
----------
2 variants × 2 CBP toggles = 4 cells. 5 seeds × 50,000 steps (5
contexts × 10k). Total 4 × 5 = 20 runs.

Metrics
-------
* Final-window MSE (last 1000 steps).
* First-context MSE vs last-context MSE (plasticity-loss factor).
* Mean MSE across all but the first context.
* Final-step hidden-activation effective rank.
* Total replacements per seed.

Outputs
-------
* ``output/step3_dod8_v2/results.csv``
* ``output/step3_dod8_v2/effective_rank_trajectory.csv``
* ``output/step3_dod8_v2/summary.json``
* ``output/step3_dod8_v2/SUMMARY.md``
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

OUTPUT_DIR = Path("output/step3_dod8_v2")
N_SEEDS = 5
N_STEPS = 50_000
CONTEXT_LEN = 10_000
FEATURE_DIM = 8
HIDDEN_SIZES = (64,)
ALPHA = 0.05
RANK_INTERVAL = 10_000
RANK_BATCH_SIZE = 512


def generate_signflip_stream(
    seed: int, n_steps: int, context_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-flip stream from DoD-8 (variant A baseline target).

    Half the regression weights flip sign every ``context_len`` steps.
    """
    rng = np.random.default_rng(seed)
    base_w = rng.normal(size=FEATURE_DIM).astype(np.float32)
    flip_mask = np.zeros(FEATURE_DIM, dtype=np.float32)
    flip_mask[: FEATURE_DIM // 2] = 1.0

    obs_l = []
    targets_l = []
    for t in range(n_steps):
        ctx = t // context_len
        if ctx % 2 == 1:
            w = base_w * (1.0 - 2.0 * flip_mask)
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


def generate_taskshift_stream(
    seed: int, n_steps: int, context_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Task-shift stream — different feature subspaces per context.

    Each context uses a contiguous half of the feature vector with a
    fresh weight permutation. Context 0 uses dims [0..3] only,
    context 1 uses dims [4..7] only, context 2 uses dims [0..3]
    again with *different* weights, etc. The unused dims still appear
    in the observation (so the input distribution is fixed) but
    contribute zero to the target. This forces the network to learn
    new feature -> target maps each context, which the prior
    representation cannot satisfy by sign reweighting alone.
    """
    rng = np.random.default_rng(seed)
    half = FEATURE_DIM // 2
    n_ctx = (n_steps + context_len - 1) // context_len
    # Per-context weight vector: zeros outside the active half.
    ctx_weights = np.zeros((n_ctx, FEATURE_DIM), dtype=np.float32)
    for c in range(n_ctx):
        active = slice(0, half) if (c % 2 == 0) else slice(half, FEATURE_DIM)
        ctx_weights[c, active] = rng.normal(size=half).astype(np.float32)

    obs_l = []
    targets_l = []
    for t in range(n_steps):
        ctx = t // context_len
        w = ctx_weights[ctx]
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
        # Number of dead units (post-activation == 0 for the whole batch).
        dead = int(jnp.sum(jnp.all(x <= 0.0, axis=0)))
        ranks.append(
            {
                "layer": layer_idx,
                "batch_size": int(centered.shape[0]),
                "width": int(centered.shape[1]),
                "stable_rank": float(stable_rank),
                "effective_rank": float(jnp.exp(entropy)),
                "dead_units": dead,
            }
        )
    return ranks


def run_one(
    seed: int,
    cbp_on: bool,
    variant: str,
    *,
    n_steps: int,
    context_len: int,
    rank_interval: int,
    rank_batch_size: int,
    cbp_replacement_rate: float,
    cbp_maturity_threshold: int,
    cbp_decay_rate: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one (variant, cbp_on, seed) cell.

    Variant A: sign-flip stream + ReLU (``leaky_relu_slope=0.0``).
    Variant B: task-shift stream + leaky-ReLU (default 0.01).
    """
    if variant == "A_relu_signflip":
        obs, targets = generate_signflip_stream(seed, n_steps, context_len)
        leaky_slope = 0.0
    elif variant == "B_taskshift":
        obs, targets = generate_taskshift_stream(seed, n_steps, context_len)
        leaky_slope = 0.01
    else:
        raise ValueError(f"unknown variant {variant!r}")

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
            leaky_relu_slope=leaky_slope,
            cbp_config=cbp_config,
        )
    else:
        learner = MultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=HIDDEN_SIZES,
            optimizer=LMS(step_size=ALPHA),
            bounder=bounder,
            sparsity=0.0,
            leaky_relu_slope=leaky_slope,
        )

    state = learner.init(FEATURE_DIM, jr.key(seed + 1000))

    sq_err_per_step = np.zeros(n_steps, dtype=np.float32)
    rank_rows: list[dict[str, Any]] = []
    replacements_total = 0
    for t in range(n_steps):
        target_arr = jnp.atleast_1d(jnp.float32(targets[t]))
        res = learner.update(state, jnp.asarray(obs[t]), target_arr)
        state = res.state
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
                        "variant": variant,
                        "step": t + 1,
                        "layer": int(rank["layer"]),
                        "batch_size": int(rank["batch_size"]),
                        "width": int(rank["width"]),
                        "stable_rank": float(rank["stable_rank"]),
                        "effective_rank": float(rank["effective_rank"]),
                        "dead_units": int(rank["dead_units"]),
                    }
                )

    n_ctx = n_steps // context_len
    per_ctx_mse = np.zeros(n_ctx, dtype=np.float32)
    for c in range(n_ctx):
        end = (c + 1) * context_len
        per_ctx_mse[c] = float(np.mean(sq_err_per_step[end - 500:end]))

    return {
        "seed": seed,
        "cbp_on": int(cbp_on),
        "variant": variant,
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
        "final_dead_units": (
            float(rank_rows[-1]["dead_units"]) if rank_rows else float("nan")
        ),
    }, rank_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--rank-interval", type=int, default=RANK_INTERVAL)
    parser.add_argument("--rank-batch-size", type=int, default=RANK_BATCH_SIZE)
    parser.add_argument("--cbp-replacement-rate", type=float, default=1e-4)
    parser.add_argument("--cbp-maturity-threshold", type=int, default=200)
    parser.add_argument("--cbp-decay-rate", type=float, default=0.99)
    return parser.parse_args()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({r["variant"] for r in rows}):
        for cbp_enabled in (False, True):
            sub = [
                r
                for r in rows
                if r["variant"] == variant and bool(r["cbp_on"]) == cbp_enabled
            ]
            if not sub:
                continue
            first = [r["first_context_mse"] for r in sub]
            last = [r["last_context_mse"] for r in sub]
            mean_after = [r["per_context_mse_mean_after_first"] for r in sub]
            final_window = [r["final_window_mse"] for r in sub]
            final_rank = [r["final_effective_rank"] for r in sub]
            replacements = [r["replacements_total"] for r in sub]
            dead = [r["final_dead_units"] for r in sub]
            key = f"{variant}::cbp_on={cbp_enabled}"
            summary[key] = {
                "first_ctx_mean": float(np.mean(first)),
                "last_ctx_mean": float(np.mean(last)),
                "mean_after_first_ctx": float(np.mean(mean_after)),
                "final_window_mse_mean": float(np.mean(final_window)),
                "final_window_mse_std": float(np.std(final_window)),
                "final_effective_rank_mean": float(np.mean(final_rank)),
                "final_dead_units_mean": float(np.mean(dead)),
                "replacements_total_mean": float(np.mean(replacements)),
                "plasticity_loss_factor": float(
                    np.mean(last) / np.mean(first)
                    if np.mean(first) > 0
                    else float("nan")
                ),
            }
    return summary


def _write_summary_md(
    md_path: Path,
    summary: dict[str, dict[str, float]],
    args: argparse.Namespace,
    total_seconds: float,
) -> None:
    lines: list[str] = []
    lines.append("# DoD-8 v2: Plasticity Sweep — ReLU + Task-Shift Variants\n")
    lines.append(
        f"\n**Setup**: 5 seeds × {args.steps} steps × 2 variants × 2 CBP toggles. "
        f"Context length {args.context_len}; LMS step {ALPHA}; ObGD kappa=2; "
        f"hidden {HIDDEN_SIZES}; CBP rho={args.cbp_replacement_rate}, "
        f"maturity={args.cbp_maturity_threshold}.\n"
    )
    lines.append(
        "\nVariant A: same sign-flip stream as DoD-8 but with ReLU activation "
        "(`leaky_relu_slope=0.0`). Variant B: task-shift stream where every "
        f"{args.context_len} steps the target switches to a different feature "
        "subspace (alternating halves of the input vector), forcing new "
        "representations.\n"
    )

    def _row(key: str) -> str:
        s = summary[key]
        return (
            f"| {key} | {s['first_ctx_mean']:.4f} | {s['last_ctx_mean']:.4f} "
            f"| {s['final_window_mse_mean']:.4f} ± {s['final_window_mse_std']:.4f} "
            f"| {s['final_effective_rank_mean']:.2f} "
            f"| {s['final_dead_units_mean']:.1f} "
            f"| {s['replacements_total_mean']:.0f} |\n"
        )

    lines.append(
        "\n| Cell | first ctx MSE | last ctx MSE | final-window MSE (mean ± std) "
        "| eff. rank | dead units | replacements |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for key in sorted(summary.keys()):
        lines.append(_row(key))

    # Verdict logic computed against final-window MSE.
    def diff(variant: str) -> tuple[float, float, float]:
        on = summary[f"{variant}::cbp_on=True"]["final_window_mse_mean"]
        off = summary[f"{variant}::cbp_on=False"]["final_window_mse_mean"]
        return on, off, on - off

    a_on, a_off, a_delta = diff("A_relu_signflip")
    b_on, b_off, b_delta = diff("B_taskshift")
    lines.append("\n## Final-window MSE deltas (CBP on − CBP off)\n\n")
    lines.append(
        f"- Variant A (ReLU sign-flip): on={a_on:.4f}, off={a_off:.4f}, "
        f"delta={a_delta:+.4f} ({'CBP helps' if a_delta < 0 else 'CBP hurts/no effect'}).\n"
    )
    lines.append(
        f"- Variant B (task-shift):    on={b_on:.4f}, off={b_off:.4f}, "
        f"delta={b_delta:+.4f} ({'CBP helps' if b_delta < 0 else 'CBP hurts/no effect'}).\n"
    )
    lines.append(f"\n**Total wall time**: {total_seconds:.1f}s\n")
    md_path.write_text("".join(lines))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    variants = ["A_relu_signflip", "B_taskshift"]
    for variant in variants:
        for cbp_on in [False, True]:
            for seed in range(args.seeds):
                t_run = time.perf_counter()
                row, ranks = run_one(
                    seed,
                    cbp_on,
                    variant,
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
                    f"variant={variant:>17} cbp_on={str(bool(cbp_on)):>5} "
                    f"seed={seed:>2}  "
                    f"first_ctx={row['first_context_mse']:.4f}  "
                    f"last_ctx={row['last_context_mse']:.4f}  "
                    f"final_win={row['final_window_mse']:.4f}  "
                    f"rank={row['final_effective_rank']:.2f}  "
                    f"dead={row['final_dead_units']:.0f}  "
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

    summary = _summarize(rows)
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

    md_path = args.output_dir / "SUMMARY.md"
    _write_summary_md(md_path, summary, args, t_total)

    print("\n=== DoD-8 v2 summary (ReLU + task-shift) ===")
    for k, s in summary.items():
        print(
            f"  {k:>40}  first_ctx={s['first_ctx_mean']:.4f}  "
            f"last_ctx={s['last_ctx_mean']:.4f}  "
            f"final_window={s['final_window_mse_mean']:.4f}  "
            f"rank={s['final_effective_rank_mean']:.2f}  "
            f"dead={s['final_dead_units_mean']:.1f}  "
            f"repl={s['replacements_total_mean']:.1f}"
        )
    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Wrote {csv_path}, {rank_path}, {summary_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

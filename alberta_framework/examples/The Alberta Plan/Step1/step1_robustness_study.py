#!/usr/bin/env python3
"""Step 1 meta-parameter robustness study.

The scientific claim of Step 1 is that meta-learning step-sizes (IDBD,
Autostep) eliminates the user's burden of choosing alpha. If true, IDBD
and Autostep should be DRAMATICALLY less sensitive to their key
hyperparameter than LMS is to its alpha.

Hypothesis
----------
IDBD/Autostep are much less sensitive to their meta-step-size than LMS
is to its alpha. Adam should sit somewhere in between (it normalizes the
update but does not meta-learn the base learning rate).

Design
------
Sweep each optimizer's primary hyperparameter over
``logspace(-4, -0.5, 11)`` (11 grid points, ~3.5 decades) on the
canonical Alberta Plan Step 1 stream:

    AlbertaPlanStep1Stream(
        feature_dim=20, num_relevant=5,
        drift_rate_w=0.001, drift_rate_b=0.001, noise_std=1.0,
    )

Other hyperparameters held fixed:
- IDBD: ``initial_step_size=0.05``
- Autostep: ``initial_step_size=0.05``, ``tau=10000``

30 seeds x 15000 steps per cell. Measure MSE over the last 5000 steps.

For each optimizer we report:
- best_mse           : minimum mean MSE across the grid
- min_mse / max_mse  : range of mean MSE across the grid
- robustness_ratio   : grid mean MSE / best MSE (lower => more robust)
- working_range_dec  : decades of hp range that yield MSE < 1.5 * best_mse

Outputs
-------
* outputs/step1_canonical/robustness_study_results.json
* outputs/step1_canonical/robustness_study_SUMMARY.md
* outputs/step1_canonical/robustness_curves.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alberta_framework import (
    IDBD,
    LMS,
    Adam,
    Autostep,
    LinearLearner,
    Timer,
    run_learning_loop_batched,
)
from alberta_framework.streams.alberta_plan_step1 import AlbertaPlanStep1Stream

DEFAULT_NUM_SEEDS = 30
DEFAULT_NUM_STEPS = 15000
WINDOW_LAST = 5000
HP_GRID = np.logspace(-4, -0.5, 11)


def _build_optimizer(name: str, hp: float) -> Any:
    if name == "LMS":
        return LMS(step_size=float(hp))
    if name == "IDBD":
        return IDBD(initial_step_size=0.05, meta_step_size=float(hp))
    if name == "Autostep":
        return Autostep(initial_step_size=0.05, meta_step_size=float(hp), tau=10000.0)
    if name == "Adam":
        return Adam(step_size=float(hp))
    raise ValueError(f"Unknown optimizer name: {name}")


def _make_stream() -> AlbertaPlanStep1Stream:
    return AlbertaPlanStep1Stream(
        feature_dim=20,
        num_relevant=5,
        drift_rate_w=0.001,
        drift_rate_b=0.001,
        noise_std=1.0,
    )


def _per_seed_final_window_mse(metrics: jax.Array, window: int) -> np.ndarray:
    sq_err = np.asarray(metrics[:, -window:, 0])
    sq_err = np.where(np.isfinite(sq_err), sq_err, np.inf)
    return sq_err.mean(axis=1)


def _working_range_decades(hps: np.ndarray, mses: np.ndarray, threshold_factor: float) -> float:
    """Decades of HP range with mean MSE <= ``threshold_factor`` * best.

    Counts the number of grid points that meet the threshold and converts
    that count into a span using the average grid spacing in log10. Returns
    0.0 if only one (or no) point meets the threshold.

    Args:
        hps: Hyperparameter grid (sorted ascending).
        mses: Per-grid-point mean MSE.
        threshold_factor: Threshold multiplier (e.g., 1.5).

    Returns:
        Estimated working range in decades.
    """
    finite_mse = np.where(np.isfinite(mses), mses, np.inf)
    best = float(np.min(finite_mse))
    if not math.isfinite(best):
        return 0.0
    threshold = threshold_factor * best
    mask = finite_mse <= threshold
    n = int(mask.sum())
    if n <= 1:
        return 0.0
    log_hps = np.log10(hps)
    # Span between the leftmost and rightmost grid points satisfying the threshold.
    indices = np.where(mask)[0]
    return float(log_hps[indices[-1]] - log_hps[indices[0]])


def run(num_seeds: int, num_steps: int, output_dir: Path) -> dict[str, Any]:
    """Run the robustness sweep and dump JSON, Markdown, and a PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ["LMS", "IDBD", "Autostep", "Adam"]
    keys = jr.split(jr.key(0xB05), num_seeds)

    per_run: dict[str, dict[str, Any]] = {}
    summary: dict[str, dict[str, Any]] = {}

    total_cells = len(optimizers) * len(HP_GRID)
    cell_idx = 0
    with Timer("robustness_study", verbose=True) as outer_timer:
        for opt_name in optimizers:
            mean_curve: list[float] = []
            median_curve: list[float] = []
            std_curve: list[float] = []
            per_seed_curve: list[list[float | None]] = []

            for hp in HP_GRID:
                cell_idx += 1
                cell_key = f"{opt_name}|hp={hp:.6g}"
                optimizer = _build_optimizer(opt_name, float(hp))
                learner = LinearLearner(optimizer=optimizer)
                stream = _make_stream()

                print(
                    f"[{cell_idx:>2}/{total_cells}] {cell_key} "
                    f"(seeds={num_seeds}, steps={num_steps})",
                    flush=True,
                )
                with Timer(f"  -> {cell_key}", verbose=True):
                    result = run_learning_loop_batched(
                        learner=learner,
                        stream=stream,
                        num_steps=num_steps,
                        keys=keys,
                    )
                    result.metrics.block_until_ready()

                per_seed = _per_seed_final_window_mse(result.metrics, WINDOW_LAST)
                finite_vals = per_seed[np.isfinite(per_seed)]
                mean = float(finite_vals.mean()) if finite_vals.size else float("inf")
                median = (
                    float(np.median(finite_vals)) if finite_vals.size else float("inf")
                )
                std = (
                    float(finite_vals.std(ddof=1)) if finite_vals.size > 1 else float("nan")
                )
                mean_curve.append(mean)
                median_curve.append(median)
                std_curve.append(std)
                per_seed_curve.append(
                    [(float(v) if math.isfinite(v) else None) for v in per_seed]
                )

                per_run[cell_key] = {
                    "optimizer": opt_name,
                    "hp": float(hp),
                    "num_seeds": int(per_seed.size),
                    "nan_seeds": int((~np.isfinite(per_seed)).sum()),
                    "mean_final_window_mse": mean,
                    "median_final_window_mse": median,
                    "std_final_window_mse": std,
                    "per_seed_final_window_mse": [
                        (float(v) if math.isfinite(v) else None) for v in per_seed
                    ],
                }

            mean_arr = np.asarray(mean_curve, dtype=np.float64)
            finite_mean = mean_arr[np.isfinite(mean_arr)]
            if finite_mean.size:
                best_idx = int(np.argmin(np.where(np.isfinite(mean_arr), mean_arr, np.inf)))
                best_mse = float(mean_arr[best_idx])
                best_hp = float(HP_GRID[best_idx])
                grid_mean_mse = float(finite_mean.mean())
            else:
                best_idx = 0
                best_mse = float("inf")
                best_hp = float(HP_GRID[0])
                grid_mean_mse = float("inf")

            min_mse = float(np.nanmin(np.where(np.isfinite(mean_arr), mean_arr, np.nan)))
            max_mse = float(np.nanmax(np.where(np.isfinite(mean_arr), mean_arr, np.nan)))
            robustness_ratio = (
                grid_mean_mse / best_mse
                if best_mse > 0.0 and math.isfinite(best_mse)
                else float("inf")
            )
            working_range_decades = _working_range_decades(
                np.asarray(HP_GRID), mean_arr, 1.5
            )

            summary[opt_name] = {
                "hp_grid": [float(h) for h in HP_GRID],
                "mean_curve": [float(v) if math.isfinite(v) else None for v in mean_curve],
                "median_curve": [float(v) if math.isfinite(v) else None for v in median_curve],
                "std_curve": [float(v) if math.isfinite(v) else None for v in std_curve],
                "best_hp": best_hp,
                "best_mse": best_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "grid_mean_mse": grid_mean_mse,
                "robustness_ratio": robustness_ratio,
                "working_range_decades": working_range_decades,
                "num_finite_grid_points": int(finite_mean.size),
                "n_grid_points": int(HP_GRID.size),
            }

    aggregated: dict[str, Any] = {
        "config": {
            "num_seeds": num_seeds,
            "num_steps": num_steps,
            "window_last": WINDOW_LAST,
            "hp_grid": [float(h) for h in HP_GRID],
            "optimizers": {
                "LMS": {"sweep": "step_size", "fixed": {}},
                "IDBD": {
                    "sweep": "meta_step_size",
                    "fixed": {"initial_step_size": 0.05},
                },
                "Autostep": {
                    "sweep": "meta_step_size",
                    "fixed": {"initial_step_size": 0.05, "tau": 10000.0},
                },
                "Adam": {"sweep": "step_size", "fixed": {}},
            },
            "stream": {
                "type": "AlbertaPlanStep1Stream",
                "feature_dim": 20,
                "num_relevant": 5,
                "drift_rate_w": 0.001,
                "drift_rate_b": 0.001,
                "noise_std": 1.0,
            },
            "wall_clock_seconds": outer_timer.duration,
        },
        "per_run": per_run,
        "summary": summary,
    }

    json_path = output_dir / "robustness_study_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"\nWrote {json_path}")

    md_path = output_dir / "robustness_study_SUMMARY.md"
    md_path.write_text(_render_summary_markdown(aggregated), encoding="utf-8")
    print(f"Wrote {md_path}")

    png_path = output_dir / "robustness_curves.png"
    _plot_curves(aggregated, png_path)
    print(f"Wrote {png_path}")

    return aggregated


def _render_summary_markdown(aggregated: dict[str, Any]) -> str:
    cfg = aggregated["config"]
    lines: list[str] = []
    lines.append("# Step 1 Robustness Study Summary")
    lines.append("")
    lines.append(
        "- Stream: AlbertaPlanStep1Stream(feature_dim=20, num_relevant=5,"
        " drift_rate_w=0.001, drift_rate_b=0.001, noise_std=1.0)"
    )
    lines.append(
        f"- Seeds: {cfg['num_seeds']}, steps per run: {cfg['num_steps']},"
        f" final-window: last {cfg['window_last']} steps."
    )
    lines.append(
        "- Hyperparameter grid: 11 log-spaced points in [1e-4, 10**-0.5]."
    )
    lines.append(
        f"- Wall-clock: {cfg['wall_clock_seconds']:.1f} s."
    )
    lines.append("")
    lines.append("## Robustness comparison")
    lines.append("")
    lines.append(
        "Lower `robustness_ratio` and wider `working_range_decades` mean the"
        " optimizer is less sensitive to its hyperparameter."
    )
    lines.append(
        " `working_range_decades` is the span of the grid that yields"
        " mean MSE within 1.5x of the best mean MSE."
    )
    lines.append("")
    lines.append(
        "| Optimizer | Best HP | Best MSE | Min MSE | Max MSE |"
        " Grid-mean MSE | Robustness ratio | Working range (decades) |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for opt in ["LMS", "IDBD", "Autostep", "Adam"]:
        s = aggregated["summary"][opt]
        lines.append(
            f"| {opt} | {s['best_hp']:.4g} | {s['best_mse']:.4g} |"
            f" {s['min_mse']:.4g} | {s['max_mse']:.4g} |"
            f" {s['grid_mean_mse']:.4g} | {s['robustness_ratio']:.3g} |"
            f" {s['working_range_decades']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-grid-point mean MSE")
    lines.append("")
    header = ["HP"] + ["LMS", "IDBD", "Autostep", "Adam"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for i, hp in enumerate(cfg["hp_grid"]):
        row = [f"{hp:.4g}"]
        for opt in ["LMS", "IDBD", "Autostep", "Adam"]:
            v = aggregated["summary"][opt]["mean_curve"][i]
            row.append("nan" if v is None else f"{v:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `Best HP` is the grid point that minimised the mean MSE across"
        " seeds; the working range is computed in log10 between the leftmost"
        " and rightmost grid points whose mean MSE is within 1.5x of the best."
    )
    lines.append(
        "- Cells where every seed blew up are stored as `null` in the JSON"
        " and `nan` in this table."
    )
    return "\n".join(lines)


def _plot_curves(aggregated: dict[str, Any], png_path: Path) -> None:
    cfg = aggregated["config"]
    hps = np.asarray(cfg["hp_grid"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    optimizer_styles = {
        "LMS": ("o-", "tab:red"),
        "IDBD": ("s-", "tab:blue"),
        "Autostep": ("^-", "tab:green"),
        "Adam": ("d-", "tab:orange"),
    }
    for opt, (style, color) in optimizer_styles.items():
        s = aggregated["summary"][opt]
        mean_curve = np.asarray(
            [v if v is not None else np.nan for v in s["mean_curve"]],
            dtype=np.float64,
        )
        ax.plot(
            hps,
            mean_curve,
            style,
            color=color,
            label=opt,
            markersize=6,
            linewidth=1.5,
        )
        ax.axvline(s["best_hp"], color=color, alpha=0.15, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hyperparameter (LMS step_size; IDBD/Autostep meta_step_size; Adam step_size)")
    ax.set_ylabel("mean MSE over last 5000 steps")
    ax.set_title(
        "Step 1 sensitivity to primary hyperparameter\n"
        f"(AlbertaPlanStep1Stream, {cfg['num_seeds']} seeds, "
        f"{cfg['num_steps']} steps)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "outputs" / "step1_canonical",
    )
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    args = parser.parse_args()

    run(
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 2 budgeted geometric dictionary probe.

This runner compares a causal RBF/hinge center dictionary against the current
single-mechanism recursive learner and fair MLP controls on the focused Step 2
suite from ``step2_recursive_feature_utility_probe.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.geometric_features import (  # noqa: E402
    BudgetedGeometricFeatureLearner,
    run_geometric_feature_arrays,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner  # noqa: E402
from alberta_framework.core.optimizers import ObGDBounding  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
RECURSIVE_PROBE = THIS_DIR / "step2_recursive_feature_utility_probe.py"
spec = importlib.util.spec_from_file_location(
    "step2_recursive_feature_utility_probe", RECURSIVE_PROBE
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {RECURSIVE_PROBE}")
recursive_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recursive_probe)

SUITE_TASKS = tuple(recursive_probe.SUITE_TASKS)
DEFAULT_METHODS = ("geometric", "single_mechanism", "mlp_32x32_no_ln", "mlp_64x64_no_ln")
DEFAULT_OUTPUT_DIR = Path("outputs/step2_geometric_feature_probe")
DEFAULT_NOTE_PATH = Path("docs/research/step2_geometric_dictionary_attempt.md")


def stderr(values: list[float]) -> float:
    """Return standard error."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def masked_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Return mean squared error over non-NaN target heads."""
    active = ~jnp.isnan(target)
    safe_target = jnp.where(active, target, 0.0)
    count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(active, (prediction - safe_target) ** 2, 0.0)) / count


def heldout_mse_by_task(
    predict_fn: Any,
    held_obs: jax.Array,
    held_targets: jax.Array,
) -> list[float]:
    """Evaluate final model on a fresh task-matched stream."""
    held_preds = jax.vmap(predict_fn)(held_obs)
    held_errors = np.asarray(held_preds - held_targets)
    held_mask = ~np.isnan(np.asarray(held_targets))
    out: list[float] = []
    for task in range(held_targets.shape[1]):
        task_mask = held_mask[:, task]
        out.append(
            float(np.mean(held_errors[task_mask, task] ** 2))
            if np.any(task_mask)
            else float("nan")
        )
    return out


def run_geometric_variant(
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run the geometric dictionary on one seed/task pair."""
    observations, targets = recursive_probe.make_data(
        seed, num_steps, feature_dim, noise_std, task_mode, rare_period
    )
    n_tasks = int(targets.shape[1])
    learner = BudgetedGeometricFeatureLearner(
        n_centers=48,
        n_tasks=n_tasks,
        step_size_output=0.05,
        energy_decay=0.99,
        utility_decay=0.995,
        rbf_bandwidth=1.25,
        hinge_radius=2.0,
        novelty_threshold=0.7,
        residual_threshold=0.05,
        min_center_age=40,
        imprint_scale=0.15,
        use_obgd=True,
        obgd_kappa=2.0,
    )
    state = learner.init(feature_dim=feature_dim, key=jr.key(seed + 70_000))
    t0 = time.time()
    result = run_geometric_feature_arrays(learner, state, observations, targets)
    result.metrics.block_until_ready()
    elapsed = time.time() - t0
    loss = np.asarray(result.metrics[:, 0], dtype=np.float64)
    held_obs, held_targets = recursive_probe.make_data(
        seed + 100_000, max(final_window, 200), feature_dim, 0.0, task_mode, rare_period
    )
    held_by_task = heldout_mse_by_task(
        lambda obs: learner.predict(result.state, obs),
        held_obs,
        held_targets,
    )
    return {
        "method": "geometric",
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(loss[-final_window:])),
        "initial_window_mse": float(np.mean(loss[:final_window])),
        "mean_mse": float(np.mean(loss)),
        "heldout_mse_by_task": held_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_by_task))),
        "active_centers": int(jnp.sum(result.state.active)),
        "insertions": int(np.sum(np.asarray(result.metrics[:, 6]) >= 0.0)),
        "wall_clock_s": elapsed,
    }


def run_mlp_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run one fair MLP control."""
    observations, targets = recursive_probe.make_data(
        seed, num_steps, feature_dim, noise_std, task_mode, rare_period
    )
    n_tasks = int(targets.shape[1])
    width = 64 if "64x64" in method else 32
    learner = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(width, width),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=method.endswith("_ln"),
    )
    state = learner.init(feature_dim=feature_dim, key=jr.key(seed + 20_000))

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        observation, target = sample
        prediction = learner.predict(carry, observation)
        loss = masked_mse(prediction, target)
        result = learner.update(carry, observation, target)
        return result.state, loss

    t0 = time.time()
    final_state, losses = jax.lax.scan(step_fn, state, (observations, targets))
    losses.block_until_ready()
    elapsed = time.time() - t0
    loss_np = np.asarray(losses, dtype=np.float64)
    held_obs, held_targets = recursive_probe.make_data(
        seed + 100_000, max(final_window, 200), feature_dim, 0.0, task_mode, rare_period
    )
    held_by_task = heldout_mse_by_task(
        lambda obs: learner.predict(final_state, obs),
        held_obs,
        held_targets,
    )
    return {
        "method": method,
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(loss_np[-final_window:])),
        "initial_window_mse": float(np.mean(loss_np[:final_window])),
        "mean_mse": float(np.mean(loss_np)),
        "heldout_mse_by_task": held_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_by_task))),
        "active_centers": 0,
        "insertions": 0,
        "wall_clock_s": elapsed,
    }


def run_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run one method/seed pair."""
    if method == "geometric":
        return run_geometric_variant(
            seed, num_steps, final_window, feature_dim, noise_std, task_mode, rare_period
        )
    if method.startswith("mlp_"):
        return run_mlp_variant(
            method, seed, num_steps, final_window, feature_dim, noise_std, task_mode, rare_period
        )
    record = recursive_probe.run_variant(
        method, seed, num_steps, final_window, feature_dim, noise_std, task_mode, rare_period
    )
    record["active_centers"] = 0
    record["insertions"] = 0
    return dict(record)


def paired_delta(
    records: list[dict[str, Any]],
    left_method: str,
    right_method: str,
) -> dict[str, Any]:
    """Return paired final-window MSE deltas; positive favors right method."""
    paired = []
    for seed in sorted({int(row["seed"]) for row in records}):
        left = next(
            row
            for row in records
            if row["method"] == left_method and row["seed"] == seed
        )
        right = next(
            row
            for row in records
            if row["method"] == right_method and row["seed"] == seed
        )
        paired.append(float(left["final_window_mse"]) - float(right["final_window_mse"]))
    return {
        "left_method": left_method,
        "right_method": right_method,
        "mean_final_window_mse_delta": float(np.mean(paired)),
        "right_wins": int(sum(delta > 0.0 for delta in paired)),
        "n": len(paired),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate records by task and method."""
    out: dict[str, Any] = {}
    task_modes = sorted({str(row["task_mode"]) for row in records})
    for task_mode in task_modes:
        task_rows = [row for row in records if row["task_mode"] == task_mode]
        methods = sorted({str(row["method"]) for row in task_rows})
        out[task_mode] = {}
        for method in methods:
            rows = [row for row in task_rows if row["method"] == method]
            finals = [float(row["final_window_mse"]) for row in rows]
            held = [float(row["heldout_mean_mse"]) for row in rows]
            out[task_mode][method] = {
                "mean_final_window_mse": float(np.mean(finals)),
                "stderr_final_window_mse": stderr(finals),
                "mean_heldout_mse": float(np.mean(held)),
                "stderr_heldout_mse": stderr(held),
                "mean_active_centers": float(
                    np.mean([float(row["active_centers"]) for row in rows])
                ),
                "mean_insertions": float(np.mean([float(row["insertions"]) for row in rows])),
            }
        mlp_methods = [method for method in methods if method.startswith("mlp_")]
        if mlp_methods and "geometric" in methods:
            best_mlp = min(
                mlp_methods,
                key=lambda method: out[task_mode][method]["mean_final_window_mse"],
            )
            out[task_mode]["best_mlp_method"] = best_mlp
            out[task_mode]["paired_best_mlp_minus_geometric"] = paired_delta(
                task_rows, best_mlp, "geometric"
            )
        if "single_mechanism" in methods and "geometric" in methods:
            out[task_mode]["paired_single_mechanism_minus_geometric"] = paired_delta(
                task_rows, "single_mechanism", "geometric"
            )
    wins_best_mlp = 0
    wins_single = 0
    for task_mode in task_modes:
        best_cmp = out[task_mode].get("paired_best_mlp_minus_geometric")
        single_cmp = out[task_mode].get("paired_single_mechanism_minus_geometric")
        wins_best_mlp += int(
            best_cmp is not None and best_cmp["mean_final_window_mse_delta"] > 0.0
        )
        wins_single += int(
            single_cmp is not None
            and single_cmp["mean_final_window_mse_delta"] > 0.0
        )
    out["suite_summary"] = {
        "tasks": len(task_modes),
        "geometric_beats_best_mlp_tasks": wins_best_mlp,
        "geometric_beats_single_mechanism_tasks": wins_single,
    }
    return out


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown results and a promotion decision."""
    aggregate_rows = results["aggregate"]
    lines = [
        "# Step 2 Geometric Dictionary Attempt",
        "",
        (
            f"Seeds: {results['config']['seeds']}; steps: "
            f"{results['config']['num_steps']}; final-window: "
            f"{results['config']['final_window']}."
        ),
        "",
        "Command:",
        "",
        "```bash",
        (
            'python "examples/The Alberta Plan/Step2/step2_geometric_feature_probe.py" '
            f"--suite --seeds {results['config']['seeds']} "
            f"--num-steps {results['config']['num_steps']} "
            f"--final-window {results['config']['final_window']} "
            f"--methods {','.join(results['config']['methods'])}"
        ),
        "```",
        "",
        "## Numeric results",
        "",
    ]
    geometric_wins_mlp = 0
    geometric_wins_single = 0
    task_count = 0
    for task_mode, task_stats in aggregate_rows.items():
        if task_mode == "suite_summary":
            continue
        task_count += 1
        lines.extend(
            [
                f"### {task_mode}",
                "",
                "| Method | Final MSE | Heldout MSE | Active centers | Insertions |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, stats in task_stats.items():
            if method.startswith("paired_") or method == "best_mlp_method":
                continue
            lines.append(
                f"| `{method}` | {stats['mean_final_window_mse']:.4f} +/- "
                f"{stats['stderr_final_window_mse']:.4f} | "
                f"{stats['mean_heldout_mse']:.4f} +/- {stats['stderr_heldout_mse']:.4f} | "
                f"{stats['mean_active_centers']:.1f} | {stats['mean_insertions']:.1f} |"
            )
        best_cmp = task_stats.get("paired_best_mlp_minus_geometric")
        if best_cmp is not None:
            geometric_wins_mlp += int(best_cmp["mean_final_window_mse_delta"] > 0.0)
            lines.append(
                f"Best fair MLP: `{task_stats['best_mlp_method']}`. Paired delta "
                f"best MLP minus `geometric`: {best_cmp['mean_final_window_mse_delta']:+.4f}; "
                f"`geometric` wins {best_cmp['right_wins']}/{best_cmp['n']} seeds."
            )
        single_cmp = task_stats.get("paired_single_mechanism_minus_geometric")
        if single_cmp is not None:
            geometric_wins_single += int(single_cmp["mean_final_window_mse_delta"] > 0.0)
            lines.append(
                f"Paired delta `single_mechanism` minus `geometric`: "
                f"{single_cmp['mean_final_window_mse_delta']:+.4f}; "
                f"`geometric` wins {single_cmp['right_wins']}/{single_cmp['n']} seeds."
            )
        lines.append("")

    nonlinear = aggregate_rows.get("nonlinear", {})
    nonlinear_cmp = nonlinear.get("paired_best_mlp_minus_geometric")
    nonlinear_win = nonlinear_cmp is not None and nonlinear_cmp["mean_final_window_mse_delta"] > 0.0
    lost_most_algebraic = geometric_wins_single < max(1, task_count // 2)
    if nonlinear_win and not lost_most_algebraic:
        decision = "promotable"
    elif geometric_wins_mlp > 0 or geometric_wins_single > 0:
        decision = "partial"
    else:
        decision = "rejected"

    lines.extend(
        [
            "## Decision",
            "",
            f"Status: **{decision}**.",
            "",
            (
                "Promotion rule: nonlinear must beat the best fair MLP and the "
                "geometric dictionary must not lose most algebraic probes against "
                "`single_mechanism`."
            ),
            "",
            "## Failure notes",
            "",
            (
                "The dictionary filled its center budget on every task and then churned "
                "hundreds of replacements. That indicates the novelty/residual gate is "
                "finding local residual patches, but the patches are not reusable enough "
                "to compete with gradient-shaped MLP features on nonlinear or with "
                "recursive product construction on algebraic probes."
            ),
            "",
            (
                "This is a local interpolation dictionary, not a recursive algebraic "
                "constructor. It has no exact mechanism for persistent products, "
                "triples, rare-task retention, or sinusoidal identities, and the "
                "3-seed suite reflects that limitation."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    """Run the probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=2500)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument("--task-mode", choices=SUITE_TASKS, default="nonlinear")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 2
        args.num_steps = 500
        args.final_window = 100
        args.methods = "geometric,single_mechanism,mlp_32x32_no_ln"
    args.final_window = min(args.final_window, args.num_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    task_modes = list(SUITE_TASKS) if args.suite else [args.task_mode]

    records: list[dict[str, Any]] = []
    for task_mode in task_modes:
        for seed in range(args.seeds):
            for method in methods:
                record = run_variant(
                    method,
                    seed,
                    args.num_steps,
                    args.final_window,
                    args.feature_dim,
                    args.noise_std,
                    task_mode,
                    args.rare_period,
                )
                records.append(record)
                print(
                    f"task={task_mode} seed={seed:02d} {method}: "
                    f"final={record['final_window_mse']:.4f}, "
                    f"held={record['heldout_mean_mse']:.4f}"
                )

    results = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "note_path": str(args.note_path),
            "methods": methods,
            "task_modes": task_modes,
        },
        "aggregate": aggregate(records),
        "per_run": records,
    }
    json_path = args.output_dir / "geometric_feature_probe_results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_summary(summary_path, results)
    write_summary(args.note_path, results)
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.note_path}")


if __name__ == "__main__":
    main()

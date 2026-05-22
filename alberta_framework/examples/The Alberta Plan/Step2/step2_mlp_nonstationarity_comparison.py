"""Compare MLPLearner variants across synthetic non-stationarity types.

This runner closes the Step 2 comparison-study TODO for the built-in stream
family. It keeps the protocol deliberately small and reproducible: each method
is trained online with one update per observation, and the comparison metric is
final-window prequential MSE.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np

from alberta_framework import (
    LMS,
    AbruptChangeStream,
    Autostep,
    CyclicStream,
    MLPLearner,
    ObGDBounding,
    PeriodicChangeStream,
    RandomWalkStream,
    run_mlp_learning_loop,
)


@dataclass(frozen=True)
class MethodSpec:
    """Configuration for one learner variant."""

    name: str
    hidden_sizes: tuple[int, ...]
    optimizer: str
    use_bounder: bool = True
    head_autostep: bool = False
    use_layer_norm: bool = True


def _stream_factories(feature_dim: int) -> dict[str, Callable[[], Any]]:
    return {
        "random_walk_drift": lambda: RandomWalkStream(
            feature_dim=feature_dim,
            drift_rate=0.003,
            noise_std=0.05,
        ),
        "abrupt_change": lambda: AbruptChangeStream(
            feature_dim=feature_dim,
            change_interval=250,
            noise_std=0.05,
        ),
        "cyclic_contexts": lambda: CyclicStream(
            feature_dim=feature_dim,
            cycle_length=250,
            num_configurations=4,
            noise_std=0.05,
        ),
        "periodic_drift": lambda: PeriodicChangeStream(
            feature_dim=feature_dim,
            period=500,
            amplitude=1.0,
            noise_std=0.05,
        ),
    }


def _method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("linear_lms", (), "lms", use_bounder=True, use_layer_norm=False),
        MethodSpec("mlp_h32_lms", (32,), "lms", use_bounder=True),
        MethodSpec("mlp_h64_lms", (64,), "lms", use_bounder=True),
        MethodSpec("mlp_h32_autostep", (32,), "autostep", use_bounder=True),
        MethodSpec("mlp_h32_autostep_head", (32,), "lms", head_autostep=True),
    ]


def _build_learner(spec: MethodSpec) -> MLPLearner:
    if spec.optimizer == "lms":
        optimizer = LMS(step_size=0.03)
    elif spec.optimizer == "autostep":
        optimizer = Autostep(initial_step_size=0.01, meta_step_size=0.01)
    else:
        raise ValueError(f"unknown optimizer {spec.optimizer!r}")

    head_optimizer = (
        Autostep(initial_step_size=0.01, meta_step_size=0.01)
        if spec.head_autostep
        else None
    )
    bounder = ObGDBounding(kappa=2.0) if spec.use_bounder else None
    return MLPLearner(
        hidden_sizes=spec.hidden_sizes,
        optimizer=optimizer,
        head_optimizer=head_optimizer,
        bounder=bounder,
        sparsity=0.0,
        use_layer_norm=spec.use_layer_norm,
    )


def run_one(
    stream_name: str,
    stream_factory: Callable[[], Any],
    spec: MethodSpec,
    seed: int,
    num_steps: int,
    window: int,
) -> dict[str, Any]:
    """Run one stream/method/seed cell."""
    learner = _build_learner(spec)
    state, metrics = run_mlp_learning_loop(
        learner=learner,
        stream=stream_factory(),
        num_steps=num_steps,
        key=jr.key(seed),
    )
    metrics_np = np.asarray(metrics)
    final_window = metrics_np[-window:, 0]
    return {
        "stream": stream_name,
        "method": spec.name,
        "seed": seed,
        "num_steps": num_steps,
        "window": window,
        "final_window_mse": float(np.mean(final_window)),
        "online_mean_mse": float(np.mean(metrics_np[:, 0])),
        "last_error": float(metrics_np[-1, 1]),
        "step_count": int(state.step_count),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows by stream/method and compare against the best MLP."""
    summary: dict[str, Any] = {}
    stream_names = sorted({str(row["stream"]) for row in rows})
    for stream_name in stream_names:
        stream_rows = [row for row in rows if row["stream"] == stream_name]
        method_names = sorted({str(row["method"]) for row in stream_rows})
        method_summary: dict[str, Any] = {}
        for method_name in method_names:
            vals = np.asarray(
                [
                    row["final_window_mse"]
                    for row in stream_rows
                    if row["method"] == method_name
                ],
                dtype=np.float64,
            )
            method_summary[method_name] = {
                "mean_final_window_mse": float(np.mean(vals)),
                "std_final_window_mse": float(np.std(vals)),
                "n": int(vals.size),
            }

        mlp_methods = [
            name for name in method_names if name.startswith("mlp_") and "h" in name
        ]
        best_mlp = min(
            mlp_methods,
            key=lambda name: method_summary[name]["mean_final_window_mse"],
        )
        best_mlp_mean = method_summary[best_mlp]["mean_final_window_mse"]
        for method_name in method_names:
            mean = method_summary[method_name]["mean_final_window_mse"]
            method_summary[method_name]["diff_vs_best_mlp"] = float(
                best_mlp_mean - mean
            )
        summary[stream_name] = {
            "best_mlp_method": best_mlp,
            "methods": method_summary,
        }
    return summary


def write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    elapsed_s: float,
) -> None:
    """Write JSON and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "n_rows": len(rows),
            "elapsed_s": elapsed_s,
            "metric": (
                "final-window prequential MSE; positive diff_vs_best_mlp means "
                "lower MSE than the best MLP width in that stream"
            ),
        },
        "rows": rows,
        "summary": summary,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Step 2 MLPLearner Non-Stationarity Comparison",
        "",
        "Metric: final-window prequential MSE. In the table below, positive "
        "`diff_vs_best_mlp` means the method is lower-MSE than the best MLP "
        "width for that stream.",
        "",
    ]
    for stream_name, stream_summary in summary.items():
        lines.append(f"## {stream_name}")
        lines.append("")
        lines.append(f"Best MLP comparator: `{stream_summary['best_mlp_method']}`")
        lines.append("")
        lines.append("| Method | Mean final-window MSE | Std | Diff vs best MLP |")
        lines.append("|---|---:|---:|---:|")
        for method_name, data in sorted(stream_summary["methods"].items()):
            lines.append(
                f"| `{method_name}` | {data['mean_final_window_mse']:.6f} | "
                f"{data['std_final_window_mse']:.6f} | "
                f"{data['diff_vs_best_mlp']:+.6f} |"
            )
        lines.append("")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--feature-dim", type=int, default=12)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_mlp_nonstationarity_comparison"),
    )
    args = parser.parse_args()

    if args.window <= 0 or args.window > args.num_steps:
        raise ValueError("--window must be in (0, --num-steps]")

    t0 = time.perf_counter()
    factories = _stream_factories(args.feature_dim)
    methods = _method_specs()
    rows: list[dict[str, Any]] = []
    for stream_name, stream_factory in factories.items():
        for spec in methods:
            for seed in range(args.n_seeds):
                row = run_one(
                    stream_name,
                    stream_factory,
                    spec,
                    seed,
                    args.num_steps,
                    args.window,
                )
                rows.append(row)
                print(
                    f"{stream_name:>18} {spec.name:>22} seed={seed:>2} "
                    f"final_mse={row['final_window_mse']:.6f}"
                )

    summary = aggregate(rows)
    elapsed = time.perf_counter() - t0
    write_summary(args.output_dir, rows, summary, elapsed)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

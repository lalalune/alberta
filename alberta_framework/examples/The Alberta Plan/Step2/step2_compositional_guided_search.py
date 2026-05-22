#!/usr/bin/env python3
"""Quick guided-search ablation for ``CompositionalFeatureLearner``.

This script compares the current compositional baseline against two guided
candidate-generation variants under the same fixed feature and candidate
budgets:

1. ``compositional_current``: utility-biased parent selection, scaled candidate
   promotion, and the existing residual-aligned shadow-head imprint.
2. ``compositional_mutation_blend``: one high-score parent plus one shallow
   mutation partner, with blended promotion output weights.
3. ``compositional_residual_imprint_blend``: residual/credit-biased parent
   scoring, shallow mutation partner, stronger candidate imprint, and blended
   promotion output weights.

The control baseline is a fair ``MultiHeadMLPLearner(hidden_sizes=(64,))``.
The run is intentionally small by default: it is meant to find promising
directions, not to establish a final Step 2 claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CompositionalFeatureLearner,
    CompositionalStream,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    run_compositional_arrays,
    run_multi_head_learning_loop,
)

OBGD_KAPPA = 2.0
MLP_STEP_SIZE = 0.03
MLP_SPARSITY = 0.5

COMP_N_FEATURES = 16
COMP_CANDIDATE_COUNT = 16
COMP_MAX_DEPTH = 4
COMP_STEP_SIZE_OUTPUT = 0.03
COMP_STEP_SIZE_THETA = 0.003
COMP_UTILITY_DECAY = 0.995
COMP_REPLACEMENT_INTERVAL = 100
COMP_MIN_FEATURE_AGE = 100
COMP_CANDIDATE_MIN_AGE = 50


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize one scan-compatible stream into observation/target arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    return observations, targets


def mlp_curve(metrics: jax.Array) -> jax.Array:
    """Collapse per-head learner metrics to one MSE curve."""
    return jnp.nanmean(metrics[..., 0], axis=-1)


def run_mlp_64(
    feature_dim: int,
    n_tasks: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> np.ndarray:
    """Run the fair MLP(64) baseline."""
    learner = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(64,),
        step_size=MLP_STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=MLP_SPARSITY,
        use_layer_norm=True,
    )
    state = learner.init(feature_dim, key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    curve = mlp_curve(result.per_head_metrics)
    curve.block_until_ready()
    return np.asarray(curve, dtype=np.float64)


def make_compositional(method_name: str, n_tasks: int) -> CompositionalFeatureLearner:
    """Create one compositional guided-search variant."""
    common = {
        "n_features": COMP_N_FEATURES,
        "n_tasks": n_tasks,
        "candidate_count": COMP_CANDIDATE_COUNT,
        "step_size_output": COMP_STEP_SIZE_OUTPUT,
        "step_size_theta": COMP_STEP_SIZE_THETA,
        "utility_decay": COMP_UTILITY_DECAY,
        "replacement_interval": COMP_REPLACEMENT_INTERVAL,
        "min_feature_age": COMP_MIN_FEATURE_AGE,
        "candidate_min_age": COMP_CANDIDATE_MIN_AGE,
        "promotion_margin": 1.05,
        "promotion_blend": 0.5,
        "max_depth": COMP_MAX_DEPTH,
        "use_obgd": True,
        "obgd_kappa": OBGD_KAPPA,
        "generation_strategy": "utility",
        "parent_temperature": 1.0,
        "residual_guidance": 1.0,
        "candidate_imprint_scale": 0.1,
    }
    if method_name == "compositional_current":
        return CompositionalFeatureLearner(**common)
    if method_name == "compositional_mutation_blend":
        return CompositionalFeatureLearner(
            **{
                **common,
                "generation_strategy": "mutation",
                "promotion_output_mode": "blend",
                "promotion_blend": 0.25,
            }
        )
    if method_name == "compositional_residual_imprint_blend":
        return CompositionalFeatureLearner(
            **{
                **common,
                "generation_strategy": "residual_imprint",
                "promotion_output_mode": "blend",
                "promotion_blend": 0.25,
                "parent_temperature": 0.75,
                "residual_guidance": 0.5,
                "candidate_imprint_scale": 0.2,
            }
        )
    raise ValueError(f"unknown compositional method: {method_name}")


def run_compositional(
    method_name: str,
    feature_dim: int,
    n_tasks: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> np.ndarray:
    """Run one compositional variant."""
    learner = make_compositional(method_name, n_tasks)
    state = learner.init(feature_dim, key)
    result = run_compositional_arrays(learner, state, observations, targets)
    curve = result.metrics[:, 0]
    curve.block_until_ready()
    return np.asarray(curve, dtype=np.float64)


def stream_specs() -> list[dict[str, Any]]:
    """Return the three synthetic out-of-class probe streams."""
    return [
        {
            "name": "out_of_class_polynomial",
            "feature_dim": 8,
            "n_tasks": 3,
            "factory": lambda: OutOfClassPolynomialStream(
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=300,
                active_triples_per_context=2,
                noise_std=0.05,
            ),
        },
        {
            "name": "frequency_mismatch",
            "feature_dim": 4,
            "n_tasks": 2,
            "factory": lambda: FrequencyMismatchStream(
                feature_dim=4,
                n_tasks=2,
                n_components_per_task=3,
                n_contexts=4,
                context_length=300,
                noise_std=0.05,
            ),
        },
        {
            "name": "compositional",
            "feature_dim": 6,
            "n_tasks": 3,
            "factory": lambda: CompositionalStream(
                feature_dim=6,
                n_tasks=3,
                inner_hidden=4,
                outer_components=5,
                n_contexts=4,
                context_length=300,
                noise_std=0.05,
            ),
        },
    ]


METHODS = [
    "mlp_64",
    "compositional_current",
    "compositional_mutation_blend",
    "compositional_residual_imprint_blend",
]


def stderr(values: np.ndarray) -> float:
    """Return standard error for a 1D array."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def run_experiment(n_seeds: int, num_steps: int, final_window: int) -> dict[str, Any]:
    """Run all methods on all streams."""
    records: list[dict[str, Any]] = []
    curves_by_stream: dict[str, dict[str, list[np.ndarray]]] = {}

    for spec in stream_specs():
        stream_name = spec["name"]
        feature_dim = spec["feature_dim"]
        n_tasks = spec["n_tasks"]
        curves_by_stream[stream_name] = {method: [] for method in METHODS}
        print(f"\n=== {stream_name} ===")

        for seed in range(n_seeds):
            root = jr.key(seed)
            keys = jr.split(root, len(METHODS) + 1)
            stream = spec["factory"]()
            observations, targets = collect_stream_arrays(stream, num_steps, keys[0])
            observations.block_until_ready()
            targets.block_until_ready()

            for method_idx, method_name in enumerate(METHODS):
                t0 = time.time()
                if method_name == "mlp_64":
                    curve = run_mlp_64(
                        feature_dim,
                        n_tasks,
                        observations,
                        targets,
                        keys[method_idx + 1],
                    )
                else:
                    curve = run_compositional(
                        method_name,
                        feature_dim,
                        n_tasks,
                        observations,
                        targets,
                        keys[method_idx + 1],
                    )
                final_mse = float(np.mean(curve[-final_window:]))
                curves_by_stream[stream_name][method_name].append(curve)
                records.append(
                    {
                        "stream": stream_name,
                        "method": method_name,
                        "seed": seed,
                        "final_window_mse": final_mse,
                        "total_mean_mse": float(np.mean(curve)),
                        "wall_clock_s": time.time() - t0,
                    }
                )
                print(
                    f"  seed={seed:02d} {method_name}: "
                    f"final={final_mse:.4f}"
                )

    aggregate: dict[str, dict[str, Any]] = {}
    for stream_name, per_method in curves_by_stream.items():
        aggregate[stream_name] = {}
        for method_name, curves in per_method.items():
            arr = np.stack(curves)
            final_by_seed = np.mean(arr[:, -final_window:], axis=1)
            total_by_seed = np.mean(arr, axis=1)
            aggregate[stream_name][method_name] = {
                "mean_final": float(np.mean(final_by_seed)),
                "stderr_final": stderr(final_by_seed),
                "mean_total": float(np.mean(total_by_seed)),
                "stderr_total": stderr(total_by_seed),
                "wins_vs_mlp": int(
                    np.sum(
                        final_by_seed
                        < np.mean(
                            np.stack(per_method["mlp_64"])[:, -final_window:],
                            axis=1,
                        )
                    )
                ),
                "wins_vs_compositional_current": int(
                    np.sum(
                        final_by_seed
                        < np.mean(
                            np.stack(per_method["compositional_current"])[
                                :, -final_window:
                            ],
                            axis=1,
                        )
                    )
                ),
            }

    candidate_names = [m for m in METHODS if m.startswith("compositional_")]
    candidate_scores: dict[str, float] = {}
    for method_name in candidate_names:
        candidate_scores[method_name] = float(
            np.mean(
                [
                    aggregate[stream_name][method_name]["mean_final"]
                    for stream_name in aggregate
                ]
            )
        )
    best_candidate = min(candidate_scores, key=candidate_scores.get)

    return {
        "config": {
            "n_seeds": n_seeds,
            "num_steps": num_steps,
            "final_window": final_window,
            "methods": METHODS,
            "compositional_common": {
                "n_features": COMP_N_FEATURES,
                "candidate_count": COMP_CANDIDATE_COUNT,
                "max_depth": COMP_MAX_DEPTH,
                "replacement_interval": COMP_REPLACEMENT_INTERVAL,
                "min_feature_age": COMP_MIN_FEATURE_AGE,
                "candidate_min_age": COMP_CANDIDATE_MIN_AGE,
            },
        },
        "aggregate": aggregate,
        "per_run": records,
        "candidate_scores_mean_across_streams": candidate_scores,
        "best_candidate": best_candidate,
    }


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    aggregate = results["aggregate"]
    config = results["config"]
    best_candidate = results["best_candidate"]
    lines: list[str] = []
    lines.append("# Step 2 Compositional Guided Search")
    lines.append("")
    lines.append(
        f"Seeds: {config['n_seeds']}; steps: {config['num_steps']}; "
        f"final-window: {config['final_window']}."
    )
    lines.append("")
    lines.append(
        "This is a quick direction-finding run. It is not a canonical Step 2 "
        "claim unless rerun at larger seed/step counts."
    )
    lines.append("")
    for stream_name, per_method in aggregate.items():
        lines.append(f"## `{stream_name}`")
        lines.append("")
        lines.append("| Method | Final MSE | Total MSE | Wins vs MLP | Wins vs current |")
        lines.append("|---|---:|---:|---:|---:|")
        for method_name in sorted(
            per_method,
            key=lambda name: per_method[name]["mean_final"],
        ):
            stats = per_method[method_name]
            lines.append(
                f"| `{method_name}` | "
                f"{stats['mean_final']:.4f} +/- {stats['stderr_final']:.4f} | "
                f"{stats['mean_total']:.4f} +/- {stats['stderr_total']:.4f} | "
                f"{stats['wins_vs_mlp']}/{config['n_seeds']} | "
                f"{stats['wins_vs_compositional_current']}/{config['n_seeds']} |"
            )
        lines.append("")

    scores = results["candidate_scores_mean_across_streams"]
    lines.append("## Candidate Ranking")
    lines.append("")
    lines.append("| Candidate | Mean final MSE across streams |")
    lines.append("|---|---:|")
    for method_name in sorted(scores, key=scores.get):
        lines.append(f"| `{method_name}` | {scores[method_name]:.4f} |")
    lines.append("")
    lines.append(f"Best candidate in this quick run: `{best_candidate}`.")
    lines.append("")
    lines.append(
        "Interpretation should be conservative: a candidate that beats the "
        "current compositional baseline but still loses to MLP is a useful "
        "search-policy improvement, not a solved Step 2 result."
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/worker_s2b_compositional"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 2 seeds, 600 steps, and a 200-step final window.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 2
        args.num_steps = 600
        args.final_window = 200
    args.final_window = min(args.final_window, args.num_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_experiment(args.seeds, args.num_steps, args.final_window)
    results["wall_clock_s"] = time.time() - t0

    json_path = args.output_dir / "guided_search_results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_summary(summary_path, results)
    print(f"\nWrote {json_path}")
    print(f"Wrote {summary_path}")
    print(f"Best candidate: {results['best_candidate']}")


if __name__ == "__main__":
    main()

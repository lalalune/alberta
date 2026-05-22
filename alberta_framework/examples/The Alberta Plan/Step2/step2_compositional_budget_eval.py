#!/usr/bin/env python3
"""Step 2 compositional+budget+future-utility evaluation.

Tests whether ``CompositionalFeatureLearner`` configured with
``GeneratorMetaResourceManager`` (``learn_generator_resources=True``) and a
contribution-trace future-utility signal can beat a fair ``MultiHeadMLP(64)``
on at least one out-of-hypothesis-class stream.

The configuration extends the ``single_mechanism`` recipe from
``step2_recursive_feature_utility_probe.py`` with the learned generator-policy
resource manager.  That probe demonstrated recursive depth works on
triple-product targets when the right scoring signal is in place; this
evaluation asks whether the same mechanism transfers to the canonical Step 2
out-of-class streams.

Streams:
    * ``out_of_class_polynomial`` -- ``OutOfClassPolynomialStream`` (degree-3
      triples from the canonical Step 2 audit).
    * ``compositional`` -- ``CompositionalStream`` (2-layer tanh oracle).
    * ``triple_product`` -- inline triple-product target with a small linear
      head (parallels the recursive probe's ``triple`` task; the recursive
      probe already showed compositional + future-utility wins at 5 seeds, so
      this is the consistency check for a 10-seed/5000-step protocol).

Methods compared:
    * ``compositional_tuned`` -- the tuned compositional learner.
    * ``mlp_64`` -- fair single-hidden-layer baseline (``MultiHeadMLPLearner``
      with hidden_sizes=(64,), ObGD-bounded, layer-norm).
    * ``upgd`` -- ``UPGDLearner`` with the canonical Step 2 settings.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    run_compositional_arrays,
)
from alberta_framework.core.future_utility import trace_decay_from_half_life
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    run_multi_head_learning_loop,
)
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.upgd import UPGDLearner, run_upgd_arrays
from alberta_framework.streams.out_of_class import (
    CompositionalStream,
    OutOfClassPolynomialStream,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = 10
DEFAULT_NUM_STEPS = 5000
DEFAULT_FINAL_WINDOW = 1500

OBGD_KAPPA = 2.0
MLP_STEP_SIZE = 0.03
MLP_SPARSITY = 0.5
UPGD_STEP_SIZE = 0.03
UPGD_SPARSITY = 0.5

# Tuned compositional config: single_mechanism + GeneratorMetaResourceManager.
# The single_mechanism knobs are taken from the recursive probe's winning
# configuration on triple-product targets; ``learn_generator_resources=True``
# turns on the bounded resource budget called for in the Alberta Plan Step 2.
COMPOSITIONAL_KWARGS: dict[str, Any] = {
    "n_features": 36,
    "candidate_count": 36,
    "step_size_output": 0.05,
    "step_size_theta": 0.005,
    "utility_decay": 0.99,
    "replacement_interval": 15,
    "min_feature_age": 30,
    "candidate_min_age": 12,
    "promotion_margin": 1.0,
    "promotion_blend": 0.6,
    "promotion_output_mode": "blend",
    "max_depth": 3,
    "use_obgd": True,
    "obgd_kappa": OBGD_KAPPA,
    "generation_strategy": "robust_recursive",
    "parent_temperature": 0.75,
    "parent_novelty_weight": 0.01,
    "parent_depth_prior": 0.05,
    "retention_depth_bonus": 0.02,
    "residual_guidance": 0.75,
    "candidate_imprint_scale": 0.2,
    "train_candidate_theta": False,
    "signed_tanh_scaffold_count": 12,
    "future_utility_mix": 0.65,
    "future_utility_trace_decay": float(trace_decay_from_half_life(16.0)),
    "future_utility_trace_mode": "contribution",
    "future_utility_rare_task_power": 0.25,
    "learn_generator_resources": True,
    "generator_resource_contexts": 2,
    "generator_resource_learning_rate": 1.0,
    "generator_resource_discount": 0.995,
    "generator_resource_exploration": 0.01,
}


# ---------------------------------------------------------------------------
# Stream realisation
# ---------------------------------------------------------------------------


def _scan_stream(stream: Any, num_steps: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Materialise a Step 2 stream into ``(observations, targets)`` arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, stream_state, jnp.arange(num_steps)
    )
    return observations, targets


def make_triple_product_arrays(
    seed: int, num_steps: int, feature_dim: int = 4, noise_std: float = 0.05
) -> tuple[jax.Array, jax.Array]:
    """Generate the recursive-probe triple-product target as a pair of arrays.

    The target is ``x0 * x1 * x2`` with a small linear shadow head.  This
    matches the ``triple`` task from ``step2_recursive_feature_utility_probe``
    and is the cleanest stress test for compositional depth.
    """
    rng = np.random.default_rng(seed)
    observations = rng.standard_normal((num_steps, feature_dim)).astype(np.float32)
    triple = observations[:, 0] * observations[:, 1] * observations[:, 2]
    linear = 0.5 * observations[:, 0] - 0.25 * observations[:, 1]
    targets = np.stack([triple, linear], axis=1)
    targets += noise_std * rng.standard_normal(targets.shape).astype(np.float32)
    return jnp.asarray(observations), jnp.asarray(targets.astype(np.float32))


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def run_compositional_seed(
    n_tasks: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Run ``CompositionalFeatureLearner`` with the tuned config for one seed."""
    learner = CompositionalFeatureLearner(n_tasks=n_tasks, **COMPOSITIONAL_KWARGS)
    state = learner.init(feature_dim=feature_dim, key=key)
    result = run_compositional_arrays(learner, state, observations, targets)
    # Column 0 of the compositional metrics block is the per-step MSE.
    return result.metrics[:, 0]


def run_mlp_seed(
    n_tasks: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Run a fair ``MultiHeadMLPLearner(64,)`` baseline for one seed."""
    learner = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(64,),
        step_size=MLP_STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=MLP_SPARSITY,
        use_layer_norm=True,
    )
    state = learner.init(feature_dim=feature_dim, key=key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    # ``per_head_metrics`` shape is (T, n_heads, 3) with column 0 = squared
    # error.  Inactive heads emit NaN -- nanmean reduces correctly.
    return jnp.nanmean(result.per_head_metrics[..., 0], axis=-1)


def run_upgd_seed(
    n_tasks: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Run ``UPGDLearner`` with canonical Step 2 settings for one seed."""
    learner = UPGDLearner(
        n_heads=n_tasks,
        hidden_sizes=(64,),
        step_size=UPGD_STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=UPGD_SPARSITY,
        use_layer_norm=True,
        loss_normalization="mean",
    )
    state = learner.init(feature_dim, key)
    result = run_upgd_arrays(learner, state, observations, targets)
    # Column 0 of the UPGD metrics block is mean squared error already.
    return result.metrics[:, 0]


METHOD_RUNNERS: dict[str, Any] = {
    "compositional_tuned": run_compositional_seed,
    "mlp_64": run_mlp_seed,
    "upgd": run_upgd_seed,
}


# ---------------------------------------------------------------------------
# Stream specifications
# ---------------------------------------------------------------------------


def _stream_realiser(
    name: str, num_steps: int
) -> tuple[Callable[[int], tuple[jax.Array, jax.Array]], int, int]:
    """Return a callable mapping ``seed -> (observations, targets)``."""
    if name == "out_of_class_polynomial":
        feature_dim = 8
        n_tasks = 3

        def realise(seed: int) -> tuple[jax.Array, jax.Array]:
            stream = OutOfClassPolynomialStream(
                feature_dim=feature_dim,
                n_tasks=n_tasks,
                n_contexts=4,
                context_length=500,
                active_triples_per_context=2,
                noise_std=0.05,
            )
            return _scan_stream(stream, num_steps, jr.key(seed + 1_000))

        return realise, feature_dim, n_tasks

    if name == "compositional":
        feature_dim = 6
        n_tasks = 3

        def realise(seed: int) -> tuple[jax.Array, jax.Array]:
            stream = CompositionalStream(
                feature_dim=feature_dim,
                n_tasks=n_tasks,
                inner_hidden=4,
                outer_components=5,
                n_contexts=4,
                context_length=500,
                noise_std=0.05,
            )
            return _scan_stream(stream, num_steps, jr.key(seed + 2_000))

        return realise, feature_dim, n_tasks

    if name == "triple_product":
        feature_dim = 4
        n_tasks = 2

        def realise(seed: int) -> tuple[jax.Array, jax.Array]:
            return make_triple_product_arrays(seed + 3_000, num_steps, feature_dim)

        return realise, feature_dim, n_tasks

    raise ValueError(f"unknown stream: {name}")


STREAM_NAMES = ("out_of_class_polynomial", "compositional", "triple_product")


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def _cohens_d(diffs: list[float]) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(arr) / sd)


def _paired_summary(
    finals_left: list[float],
    finals_right: list[float],
) -> dict[str, Any]:
    """Return ``left - right`` paired-difference statistics."""
    diffs = [
        float(left - right)
        for left, right in zip(finals_left, finals_right, strict=True)
    ]
    return {
        "mean_diff": float(np.mean(diffs)),
        "stderr_diff": _stderr(diffs),
        "cohens_d": _cohens_d(diffs),
        "left_wins": int(sum(diff > 0.0 for diff in diffs)),
        "n": len(diffs),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument(
        "--streams",
        type=str,
        default=",".join(STREAM_NAMES),
        help="Comma-separated stream names.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHOD_RUNNERS),
        help="Comma-separated method names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_canonical"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="compositional_budget_10seed",
        help="Base name for the JSON and SUMMARY.md files.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.num_steps = 800
        args.final_window = 200
        args.output_name = f"{args.output_name}_smoke"

    args.final_window = min(args.final_window, args.num_steps)
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_summary: dict[str, Any] = {
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "final_window": args.final_window,
        "streams": streams,
        "methods": methods,
        "compositional_kwargs": COMPOSITIONAL_KWARGS,
        "mlp_step_size": MLP_STEP_SIZE,
        "mlp_sparsity": MLP_SPARSITY,
        "upgd_step_size": UPGD_STEP_SIZE,
        "upgd_sparsity": UPGD_SPARSITY,
        "obgd_kappa": OBGD_KAPPA,
    }

    # records is a flat list keyed by (stream, method, seed); curves stays in
    # python so we can compute paired final-window MSE and Cohen's d locally.
    per_run: list[dict[str, Any]] = []
    finals: dict[str, dict[str, list[float]]] = {}
    means: dict[str, dict[str, list[float]]] = {}

    for stream_name in streams:
        realise, feature_dim, n_tasks = _stream_realiser(stream_name, args.num_steps)
        finals[stream_name] = {method: [] for method in methods}
        means[stream_name] = {method: [] for method in methods}

        for seed in range(args.seeds):
            observations, targets = realise(seed)
            for method in methods:
                runner = METHOD_RUNNERS[method]
                t0 = time.time()
                key = jr.key(seed * 17 + hash(method) % 100_000)
                curve = runner(n_tasks, feature_dim, observations, targets, key)
                curve.block_until_ready()
                elapsed = time.time() - t0
                curve_np = np.asarray(curve, dtype=np.float64)
                final_mse = float(np.mean(curve_np[-args.final_window :]))
                mean_mse = float(np.mean(curve_np))
                finals[stream_name][method].append(final_mse)
                means[stream_name][method].append(mean_mse)
                per_run.append(
                    {
                        "stream": stream_name,
                        "method": method,
                        "seed": seed,
                        "final_window_mse": final_mse,
                        "mean_mse": mean_mse,
                        "wall_clock_s": elapsed,
                    }
                )
                print(
                    f"stream={stream_name} method={method} seed={seed:02d} "
                    f"final={final_mse:.4f} mean={mean_mse:.4f} "
                    f"({elapsed:.1f}s)"
                )

    aggregate: dict[str, Any] = {}
    for stream_name in streams:
        aggregate[stream_name] = {}
        for method in methods:
            method_finals = finals[stream_name][method]
            method_means = means[stream_name][method]
            aggregate[stream_name][method] = {
                "mean_final_window_mse": float(np.mean(method_finals)),
                "stderr_final_window_mse": _stderr(method_finals),
                "mean_total_mse": float(np.mean(method_means)),
                "stderr_total_mse": _stderr(method_means),
            }

        # Paired comparisons against the fair MLP baseline and against UPGD.
        if "mlp_64" in methods and "compositional_tuned" in methods:
            aggregate[stream_name]["paired_mlp_minus_compositional"] = _paired_summary(
                finals[stream_name]["mlp_64"],
                finals[stream_name]["compositional_tuned"],
            )
        if "upgd" in methods and "compositional_tuned" in methods:
            aggregate[stream_name]["paired_upgd_minus_compositional"] = _paired_summary(
                finals[stream_name]["upgd"],
                finals[stream_name]["compositional_tuned"],
            )
        if "mlp_64" in methods and "upgd" in methods:
            aggregate[stream_name]["paired_mlp_minus_upgd"] = _paired_summary(
                finals[stream_name]["mlp_64"],
                finals[stream_name]["upgd"],
            )

    results = {
        "config": config_summary,
        "aggregate": aggregate,
        "per_run": per_run,
    }
    json_path = args.output_dir / f"{args.output_name}_results.json"
    summary_path = args.output_dir / f"{args.output_name}_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_summary(summary_path, results)
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a human-readable summary in the canonical Step 2 format."""
    cfg = results["config"]
    lines = [
        "# Step 2 compositional + budget + future-utility evaluation",
        "",
        (
            f"Seeds: {cfg['seeds']}. Steps per run: {cfg['num_steps']}. "
            f"Final-window: last {cfg['final_window']} steps."
        ),
        "",
        (
            "Tuned compositional learner combines the ``single_mechanism`` "
            "recipe from ``step2_recursive_feature_utility_probe`` "
            "(contribution-trace future utility, residual imprint, "
            "product-biased priors, depth retention, signed-tanh scaffold) "
            "with ``learn_generator_resources=True`` so the "
            "GeneratorMetaResourceManager allocates a bounded budget across "
            "operation/parent-mode policies."
        ),
        "",
    ]
    for stream_name, stats in results["aggregate"].items():
        lines.extend(
            [
                f"## Stream: `{stream_name}`",
                "",
                "Final-window MSE (mean +/- stderr):",
                "",
                "| Method | Final-window MSE | Total mean MSE |",
                "|---|---|---|",
            ]
        )
        method_rows = sorted(
            (
                (name, body)
                for name, body in stats.items()
                if not name.startswith("paired_")
            ),
            key=lambda row: row[1]["mean_final_window_mse"],
        )
        for name, body in method_rows:
            lines.append(
                f"| `{name}` | "
                f"{body['mean_final_window_mse']:.4f} +/- "
                f"{body['stderr_final_window_mse']:.4f} | "
                f"{body['mean_total_mse']:.4f} +/- "
                f"{body['stderr_total_mse']:.4f} |"
            )
        lines.append("")
        for label in (
            "paired_mlp_minus_compositional",
            "paired_upgd_minus_compositional",
            "paired_mlp_minus_upgd",
        ):
            comp = stats.get(label)
            if not comp:
                continue
            lines.append(
                f"- `{label}`: mean diff "
                f"{comp['mean_diff']:+.4f} +/- {comp['stderr_diff']:.4f}, "
                f"Cohen d={comp['cohens_d']:+.3f}, "
                f"{comp['left_wins']}/{comp['n']} left-wins."
            )
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "A positive ``mlp_minus_compositional`` mean diff with "
            "Cohen d > 1.0 on any stream indicates the compositional path "
            "now beats a fair MLP at non-trivial effect size on that stream "
            "and can be promoted as a positive Step 2 result. Otherwise "
            "this run is a candid negative-result note: compositional "
            "construction with a learned resource budget did not close the "
            "gap to UPGD and stayed at-or-below MLP at this seed/step "
            "budget. UPGD is left as the canonical Step 2 promotion either "
            "way; this script is additive.",
        ]
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

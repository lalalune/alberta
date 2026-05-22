#!/usr/bin/env python3
"""D21: UPGD efficiency on the same compact OPMNIST bridge as D18/D20.

This runner measures UPGD variants on the local OPMNIST bridge protocol so they
can be compared against the recorded D18 and D20 artifacts.  It focuses on
UPGD's compute/performance tradeoff: online final-window loss, held-out
task-view retention, runtime, steps/second, trainable parameters, and float
state size.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import d18_opmnist_bridge as bridge  # noqa: E402
from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    aggregate_records,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)

from alberta_framework.core.upgd import UPGDLearner  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d21_upgd_opmnist_efficiency")
DEFAULT_RESULT_PREFIX = "d21_upgd_opmnist_efficiency"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d21_upgd_opmnist_efficiency.md"
)
DATASET_NAME = "opmnist_bridge"
N_CLASSES = bridge.N_CLASSES


@dataclass(frozen=True)
class UPGDVariant:
    """One UPGD variant to benchmark."""

    name: str
    hidden_size: int
    readout_mode: str


def default_variants() -> tuple[UPGDVariant, ...]:
    """Return the compact variant sweep."""
    return (
        UPGDVariant("upgd_h16_linear_mse", 16, "linear_mse"),
        UPGDVariant("upgd_h16_softmax_ce", 16, "softmax_ce"),
        UPGDVariant("upgd_h32_linear_mse", 32, "linear_mse"),
        UPGDVariant("upgd_h32_softmax_ce", 32, "softmax_ce"),
        UPGDVariant("upgd_h64_linear_mse", 64, "linear_mse"),
        UPGDVariant("upgd_h64_softmax_ce", 64, "softmax_ce"),
        UPGDVariant("upgd_h128_linear_mse", 128, "linear_mse"),
        UPGDVariant("upgd_h128_softmax_ce", 128, "softmax_ce"),
    )


def tree_float_size(tree: Any) -> int:
    """Count floating-point scalar slots in a JAX PyTree."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            total += int(leaf.size)
    return total


def trainable_param_count(state: Any) -> int:
    """Count trainable UPGD parameter scalars."""
    arrays = [
        *state.trunk_params.weights,
        *state.trunk_params.biases,
        *state.head_params.weights,
        *state.head_params.biases,
    ]
    return int(sum(array.size for array in arrays))


def make_upgd(variant: UPGDVariant) -> UPGDLearner:
    """Construct a promoted-family UPGD learner."""
    return UPGDLearner.step2_default(
        n_heads=N_CLASSES,
        hidden_sizes=(variant.hidden_size,),
        readout_mode=variant.readout_mode,
    )


def run_upgd_stream(
    learner: UPGDLearner,
    state: Any,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    """Run UPGD with one JIT scan and return predictions plus elapsed seconds."""
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)

    @jax.jit  # type: ignore[untyped-decorator]
    def run(initial_state: Any) -> tuple[Any, jax.Array, jax.Array]:
        def step_fn(
            carry: Any,
            batch: tuple[jax.Array, jax.Array],
        ) -> tuple[Any, tuple[jax.Array, jax.Array]]:
            observation, target = batch
            result = learner.update(carry, observation, target)
            return result.state, (result.predictions, result.metrics)

        final_state, (predictions, metrics) = jax.lax.scan(
            step_fn,
            initial_state,
            (observations_jax, targets_jax),
        )
        return final_state, predictions, metrics

    t0 = time.time()
    final_state, predictions, metrics = run(state)
    predictions.block_until_ready()
    elapsed = time.time() - t0
    return final_state, np.asarray(predictions), np.asarray(metrics), elapsed


def summarize_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    final_window: int,
) -> dict[str, float]:
    """Summarize prequential predictions."""
    losses = np.mean((predictions - targets) ** 2, axis=1)
    correct = np.argmax(predictions, axis=1) == labels
    window = min(final_window, predictions.shape[0])
    return {
        "online_mean_mse": float(np.mean(losses)),
        "final_window_mse": float(np.mean(losses[-window:])),
        "online_mean_accuracy": float(np.mean(correct)),
        "final_window_accuracy": float(np.mean(correct[-window:])),
    }


def evaluate_upgd_views(
    learner: UPGDLearner,
    state: Any,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate UPGD across held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        predictions = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        predictions.block_until_ready()
        preds = np.asarray(predictions)
        mse_values.append(float(np.mean((preds - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(preds, axis=1) == labels)))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
    }


def parse_optional_positive_int(value: str) -> int | None:
    """Parse bridge-compatible optional integer values."""
    parsed = bridge.parse_optional_positive_int(value)
    if parsed is None:
        return None
    return int(parsed)


def run_one_seed(
    seed: int,
    variants: tuple[UPGDVariant, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run MLP baselines and UPGD variants for one OPMNIST bridge seed."""
    dataset, stream = bridge.make_stream(seed, args)
    observations = np.asarray(stream.observations, dtype=np.float32)
    targets = np.asarray(stream.targets, dtype=np.float32)
    labels = np.asarray(stream.labels, dtype=np.int32)
    methods: dict[str, dict[str, float]] = {}

    for method in MLP_METHODS:
        print(f"opmnist seed={seed}: running {method}")
        mlp = make_mlp(
            method=method,
            n_heads=N_CLASSES,
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 60_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method].update(
            bridge.evaluate_mlp_classifier_views(
                learner=mlp,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.mlp_deployment_transform,
            )
        )
        runtime = float(time.time() - t0)
        methods[method]["runtime_s"] = runtime
        methods[method]["steps_per_second"] = float(args.steps / runtime)

    for variant in variants:
        print(f"opmnist seed={seed}: running {variant.name}")
        learner = make_upgd(variant)
        state = learner.init(observations.shape[1], jr.key(seed + 70_000 + variant.hidden_size))
        initial_params = trainable_param_count(state)
        initial_float_state = tree_float_size(state)
        final_state, predictions, _metrics, runtime = run_upgd_stream(
            learner,
            state,
            observations,
            targets,
        )
        entry = summarize_predictions(predictions, targets, labels, args.final_window)
        entry.update(
            evaluate_upgd_views(
                learner,
                final_state,
                stream.test_views,
                stream.test_labels,
            )
        )
        entry.update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "trainable_params": float(initial_params),
                "float_state_size": float(initial_float_state),
            }
        )
        methods[variant.name] = entry

    meta = dict(dataset.metadata)
    meta.update(stream.metadata)
    return (
        {
            "dataset_name": DATASET_NAME,
            "seed": int(seed),
            "dataset": meta,
            "methods": methods,
        },
        meta,
    )


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one metric cell."""
    if metric not in row:
        return ""
    value = row[metric]
    return f"{value['mean']:.6f} +/- {value['stderr']:.6f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    aggregate = results["aggregate"][DATASET_NAME]
    comparisons = aggregate["comparisons"]
    lines = [
        "# D21 UPGD OPMNIST Efficiency",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks."
        ),
        "",
        (
            "| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s | "
            "Steps/s | Params | Float State |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in aggregate.items():
        if method == "comparisons":
            continue
        lines.append(
            f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
            f"{metric_cell(row, 'final_window_accuracy')} | "
            f"{metric_cell(row, 'test_mse')} | "
            f"{metric_cell(row, 'test_accuracy')} | "
            f"{metric_cell(row, 'runtime_s')} | "
            f"{metric_cell(row, 'steps_per_second')} | "
            f"{metric_cell(row, 'trainable_params')} | "
            f"{metric_cell(row, 'float_state_size')} |"
        )
    lines.extend(["", "## UPGD vs Best MLP", ""])
    for metric in (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "runtime_s",
    ):
        if metric not in comparisons:
            continue
        comparison = comparisons[metric]["best_kernel_vs_best_mlp"]
        lines.append(f"- `{metric}`: {bridge.comparison_cell(comparison)}")
    lines.extend(
        [
            "",
            "Positive MSE/accuracy differences favor UPGD. Runtime comparisons use "
            "the raw `runtime_s` and `steps_per_second` table columns because the "
            "shared paired-difference helper is defined for losses and accuracies.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument(
        "--mnist-source",
        choices=(
            "auto",
            "openml",
            "torchvision",
            "sklearn_digits_28x28",
            "sklearn_digits_8x8",
        ),
        default="sklearn_digits_28x28",
    )
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--allow-torchvision-download", action="store_true")
    parser.add_argument(
        "--mnist-split",
        choices=("stratified", "canonical"),
        default="stratified",
    )
    parser.add_argument("--openml-data-home", type=Path, default=None)
    parser.add_argument("--torchvision-data-home", type=Path, default=None)
    parser.add_argument("--openml-n-retries", type=int, default=2)
    parser.add_argument("--openml-retry-delay", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=parse_optional_positive_int, default=1000)
    parser.add_argument("--max-test-examples", type=parse_optional_positive_int, default=400)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--task-block-size", type=int, default=200)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--include-identity-permutation", action="store_true")
    parser.add_argument(
        "--task-sampling",
        choices=("random", "sequential_epoch"),
        default="sequential_epoch",
    )
    parser.add_argument(
        "--max-test-permutation-views",
        type=parse_optional_positive_int,
        default=None,
    )
    parser.add_argument("--evaluate-all-permutation-views", action="store_true")
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--mlp-deployment-transform",
        choices=bridge.DEPLOYMENT_TRANSFORMS,
        default="raw",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.n_permutations < 2:
        raise ValueError("--n-permutations must be at least 2")
    if args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive or 'all'")
    if args.max_test_examples is not None and args.max_test_examples <= 0:
        raise ValueError("--max-test-examples must be positive or 'all'")
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")
    if not 0.0 <= args.mlp_sparsity < 1.0:
        raise ValueError("--mlp-sparsity must be in [0, 1)")


def main() -> None:
    """Run D21."""
    args = parse_args()
    validate_args(args)
    variants = default_variants()
    candidate_methods = tuple(variant.name for variant in variants)

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, variants, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta

    results = {
        "config": {
            "runner": "d21_upgd_opmnist_efficiency",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "mnist_source": args.mnist_source,
            "max_train_examples": args.max_train_examples,
            "max_test_examples": args.max_test_examples,
            "n_permutations": args.n_permutations,
            "task_block_size": args.task_block_size,
            "task_sampling": args.task_sampling,
            "sample_with_replacement": args.sample_with_replacement,
            "variants": [variant.__dict__ for variant in variants],
        },
        "datasets": datasets_meta,
        "records": records,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "aggregate": bridge.add_deployment_comparisons(
            aggregate_records(records, candidate_methods),
            records,
            candidate_methods,
        ),
        "wall_clock_s": float(time.time() - t0),
        "evidence_level": "upgd_compact_opmnist_efficiency",
    }
    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    bridge.atomic_write_json(json_path, results)
    write_summary(summary_path, results)
    if args.note_path is not None:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path is not None:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

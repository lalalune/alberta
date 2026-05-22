#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return"
"""D24: prototype universal-block sweep on compact OPMNIST.

This runner takes the strongest lesson from D22 and tests whether prototype
memory can become a reusable feature-construction block rather than only a
class-conditioned nearest-prototype classifier.

It compares:

* class-conditioned prototype memory budgets (`proto_mem_s*`);
* global prototype basis + online value readout (`basis_*`);
* adaptive-bandwidth basis variants; and
* a simple recursive basis (`raw -> basis activations -> basis readout`).
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
from d21_upgd_opmnist_efficiency import summarize_predictions, tree_float_size  # noqa: E402
from d22_upgd_prototype_hybrid_opmnist import (  # noqa: E402
    evaluate_prototype_views,
    make_memory,
    run_prototype_stream,
)

from alberta_framework.core.prototype_basis import (  # noqa: E402
    PrototypeBasisBlock,
    PrototypeBasisConfig,
    PrototypeBasisParams,
    PrototypeBasisState,
    run_prototype_basis_arrays,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d24_prototype_universal_blocks")
DEFAULT_RESULT_PREFIX = "d24_prototype_universal_blocks"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d24_prototype_universal_blocks.md"
)
DATASET_NAME = "opmnist_bridge"
N_CLASSES = bridge.N_CLASSES


@dataclass(frozen=True)
class BasisVariant:
    """One global prototype-basis readout variant."""

    name: str
    n_prototypes: int
    bandwidth: float = 0.01
    novelty_threshold: float = 0.08
    step_size: float = 0.05
    adaptive_bandwidth: bool = False
    recursive: bool = False


def memory_budgets() -> tuple[int, ...]:
    """Prototype memory budgets to sweep."""
    return (4, 8, 12, 16, 20, 32)


def basis_variants() -> tuple[BasisVariant, ...]:
    """Return compact universal-block variants."""
    return (
        BasisVariant("basis_p64_bw001", 64, bandwidth=0.01),
        BasisVariant("basis_p128_bw001", 128, bandwidth=0.01),
        BasisVariant("basis_p128_bw003", 128, bandwidth=0.03),
        BasisVariant("basis_p128_adaptbw", 128, bandwidth=0.03, adaptive_bandwidth=True),
        BasisVariant("recursive_basis_p64_p64", 64, bandwidth=0.03, recursive=True),
    )


def make_basis(variant: BasisVariant, input_dim: int, output_dim: int) -> PrototypeBasisBlock:
    """Create a prototype-basis block."""
    return PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            n_prototypes=variant.n_prototypes,
            step_size=variant.step_size,
            update_rate=0.3,
            novelty_threshold=variant.novelty_threshold,
            bandwidth=variant.bandwidth,
            adaptive_bandwidth=variant.adaptive_bandwidth,
            bandwidth_update_rate=0.1,
            min_bandwidth=1e-4,
            max_bandwidth=1.0,
            normalize_activations=True,
        )
    )


def logits_to_probs(predictions: np.ndarray) -> np.ndarray:
    """Convert raw vector predictions to probabilities."""
    shifted = predictions - np.max(predictions, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def summarize_prob_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    final_window: int,
) -> dict[str, float]:
    """Summarize raw predictions after softmax deployment transform."""
    return summarize_predictions(logits_to_probs(predictions), targets, labels, final_window)


def run_basis_stream(
    block: PrototypeBasisBlock,
    observations: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
) -> tuple[PrototypeBasisParams, PrototypeBasisState, np.ndarray, np.ndarray, float]:
    """Run a basis readout with one JIT scan."""
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)
    params, state = block.init(key)
    t0 = time.time()
    result = run_prototype_basis_arrays(
        block,
        observations_jax,
        targets_jax,
        params=params,
        state=state,
    )
    result.predictions.block_until_ready()
    return (
        result.params,
        result.state,
        np.asarray(result.predictions),
        np.asarray(result.metrics),
        time.time() - t0,
    )


def evaluate_basis_views(
    block: PrototypeBasisBlock,
    params: PrototypeBasisParams,
    state: PrototypeBasisState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a basis readout across held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        logits = jax.vmap(lambda obs: block.predict(params, state, obs))(observations)
        logits.block_until_ready()
        probs = logits_to_probs(np.asarray(logits))
        mse_values.append(float(np.mean((probs - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(probs, axis=1) == labels)))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
    }


def run_recursive_basis_stream(
    variant: BasisVariant,
    observations: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
) -> tuple[
    PrototypeBasisBlock,
    PrototypeBasisParams,
    PrototypeBasisState,
    PrototypeBasisBlock,
    PrototypeBasisParams,
    PrototypeBasisState,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Run a two-layer recursive basis."""
    first = make_basis(variant, observations.shape[1], variant.n_prototypes)
    second_variant = BasisVariant(
        name=f"{variant.name}_second",
        n_prototypes=variant.n_prototypes,
        bandwidth=0.03,
        novelty_threshold=0.08,
        step_size=variant.step_size,
        adaptive_bandwidth=variant.adaptive_bandwidth,
    )
    second = make_basis(second_variant, variant.n_prototypes, targets.shape[1])
    key1, key2 = jr.split(key)
    params1, state1 = first.init(key1)
    params2, state2 = second.init(key2)
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)

    @jax.jit  # type: ignore[untyped-decorator]
    def run(
        p1: PrototypeBasisParams,
        s1: PrototypeBasisState,
        p2: PrototypeBasisParams,
        s2: PrototypeBasisState,
    ) -> tuple[
        PrototypeBasisParams,
        PrototypeBasisState,
        PrototypeBasisParams,
        PrototypeBasisState,
        jax.Array,
        jax.Array,
    ]:
        def step_fn(
            carry: tuple[
                PrototypeBasisParams,
                PrototypeBasisState,
                PrototypeBasisParams,
                PrototypeBasisState,
            ],
            batch: tuple[jax.Array, jax.Array],
        ) -> tuple[
            tuple[
                PrototypeBasisParams,
                PrototypeBasisState,
                PrototypeBasisParams,
                PrototypeBasisState,
            ],
            tuple[jax.Array, jax.Array],
        ]:
            params1_inner, state1_inner, params2_inner, state2_inner = carry
            observation, target = batch
            features = first.activations(state1_inner, observation)
            second_result = second.update(params2_inner, state2_inner, features, target)
            state1_next, center_metrics = first.update_centers(state1_inner, observation)
            metrics = second_result.metrics.at[3].set(center_metrics[0])
            return (
                params1_inner,
                state1_next,
                second_result.params,
                second_result.state,
            ), (second_result.prediction, metrics)

        (fp1, fs1, fp2, fs2), (predictions, metrics) = jax.lax.scan(
            step_fn,
            (p1, s1, p2, s2),
            (observations_jax, targets_jax),
        )
        return fp1, fs1, fp2, fs2, predictions, metrics

    t0 = time.time()
    final_p1, final_s1, final_p2, final_s2, predictions, metrics = run(
        params1,
        state1,
        params2,
        state2,
    )
    predictions.block_until_ready()
    return (
        first,
        final_p1,
        final_s1,
        second,
        final_p2,
        final_s2,
        np.asarray(predictions),
        np.asarray(metrics),
        time.time() - t0,
    )


def evaluate_recursive_views(
    first: PrototypeBasisBlock,
    state1: PrototypeBasisState,
    second: PrototypeBasisBlock,
    params2: PrototypeBasisParams,
    state2: PrototypeBasisState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate recursive basis across held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))

        def predict_one(obs: jax.Array) -> jax.Array:
            features = first.activations(state1, obs)
            return second.predict(params2, state2, features)

        logits = jax.vmap(predict_one)(observations)
        logits.block_until_ready()
        probs = logits_to_probs(np.asarray(logits))
        mse_values.append(float(np.mean((probs - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(probs, axis=1) == labels)))
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
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one paired OPMNIST seed."""
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
            jr.key(seed + 100_000 + MLP_METHODS.index(method)),
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

    for budget in memory_budgets():
        method = f"proto_mem_s{budget}"
        print(f"opmnist seed={seed}: running {method}")
        memory = make_memory(budget, observations.shape[1])
        state, predictions, metrics, runtime = run_prototype_stream(
            memory,
            observations,
            targets,
        )
        methods[method] = summarize_predictions(
            predictions,
            targets,
            labels,
            args.final_window,
        )
        methods[method].update(
            evaluate_prototype_views(memory, state, stream.test_views, stream.test_labels)
        )
        methods[method].update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "active_prototypes": float(metrics[-1, 3]),
                "float_state_size": float(tree_float_size(state)),
            }
        )

    for variant in basis_variants():
        print(f"opmnist seed={seed}: running {variant.name}")
        if variant.recursive:
            (
                first,
                _params1,
                state1,
                second,
                params2,
                state2,
                predictions,
                metrics,
                runtime,
            ) = run_recursive_basis_stream(
                variant,
                observations,
                targets,
                jr.key(seed + 101_000 + variant.n_prototypes),
            )
            entry = summarize_prob_predictions(
                predictions,
                targets,
                labels,
                args.final_window,
            )
            entry.update(
                evaluate_recursive_views(
                    first,
                    state1,
                    second,
                    params2,
                    state2,
                    stream.test_views,
                    stream.test_labels,
                )
            )
            state_size = (
                tree_float_size(state1)
                + tree_float_size(params2)
                + tree_float_size(state2)
            )
        else:
            block = make_basis(variant, observations.shape[1], targets.shape[1])
            params, state, predictions, metrics, runtime = run_basis_stream(
                block,
                observations,
                targets,
                jr.key(seed + 101_000 + variant.n_prototypes),
            )
            entry = summarize_prob_predictions(
                predictions,
                targets,
                labels,
                args.final_window,
            )
            entry.update(
                evaluate_basis_views(
                    block,
                    params,
                    state,
                    stream.test_views,
                    stream.test_labels,
                )
            )
            state_size = tree_float_size(params) + tree_float_size(state)
        entry.update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "active_prototypes": float(metrics[-1, 3]),
                "float_state_size": float(state_size),
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
        "# D24 Prototype Universal Blocks",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks."
        ),
        "",
        "| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s | "
        "Steps/s | Protos | Float State |",
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
            f"{metric_cell(row, 'active_prototypes')} | "
            f"{metric_cell(row, 'float_state_size')} |"
        )
    lines.extend(["", "## Best Candidate vs Best MLP", ""])
    for metric in (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "deployment_test_mse",
        "deployment_test_accuracy",
    ):
        if metric not in comparisons:
            continue
        comparison = comparisons[metric]["best_kernel_vs_best_mlp"]
        lines.append(f"- `{metric}`: {bridge.comparison_cell(comparison)}")
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
    parser.add_argument("--mnist-split", choices=("stratified", "canonical"), default="stratified")
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
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def apply_smoke(args: argparse.Namespace) -> None:
    """Apply quick smoke settings."""
    if not args.smoke:
        return
    args.steps = 120
    args.n_seeds = 1
    args.final_window = 40
    args.max_train_examples = 200
    args.max_test_examples = 80
    args.n_permutations = 2
    args.task_block_size = 40


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
    """Run D24."""
    args = parse_args()
    apply_smoke(args)
    validate_args(args)
    candidate_methods = tuple(
        [*(f"proto_mem_s{budget}" for budget in memory_budgets())]
        + [variant.name for variant in basis_variants()]
    )

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta

    results = {
        "config": {
            "runner": "d24_prototype_universal_blocks",
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
            "memory_budgets": list(memory_budgets()),
            "basis_variants": [variant.__dict__ for variant in basis_variants()],
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
        "evidence_level": "prototype_universal_blocks_opmnist",
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

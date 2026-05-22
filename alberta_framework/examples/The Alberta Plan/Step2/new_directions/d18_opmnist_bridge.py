#!/usr/bin/env python3
"""Bridge D18 ``step2_canonical`` onto the OPMNIST stressor.

This runner is intentionally narrow: it reuses the Online Permuted MNIST style
stream and protocol metadata from ``step2_published_stressors.py`` while keeping
the learner side to D18 plus the same fair MLP baselines used by the Step 2
new-direction runners.  The default is local and no-network: sklearn digits
expanded to 28x28, pixel-permuted into sequential task blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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

import d18_simple_universal_resource_basis as d18  # noqa: E402
import step2_published_stressors as stressors  # noqa: E402
from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    aggregate_records,
    compare_to_group,
    is_higher_better,
    make_mlp,
    paired_diff,
    run_mlp_stream,
    stderr,
    summarize_prequential,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d18_opmnist_bridge")
DEFAULT_RESULT_PREFIX = "d18_opmnist_bridge"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d18_opmnist_bridge_smoke.md"
)
DATASET_NAME = "opmnist_bridge"
N_CLASSES = int(stressors.N_CLASSES)
PROTOCOL_GATE_KEYS = (
    "source_kind",
    "is_true_mnist",
    "is_full_mnist_split",
    "protocol",
    "steps",
    "n_permutations",
    "task_block_size",
    "sample_with_replacement",
    "task_sampling",
    "include_identity_permutation",
    "permutations_are_random_pixel_orders",
    "task_id_provided_to_learner",
    "prediction_before_update_every_step",
    "all_experts_update_every_step",
    "single_pass_examples_within_task",
    "test_permutation_views",
    "test_views_cover_observed_permutations",
    "test_views_cover_all_permutations",
    "full_mnist_task_blocks",
    "matches_dohare_opmnist_core_protocol",
    "matches_dohare_opmnist_published_task_count",
)
DEPLOYMENT_TRANSFORMS = ("raw", "clip01", "softmax", "hard")


def parse_optional_positive_int(value: str) -> int | None:
    """Parse a positive integer or an explicit all/none sentinel."""
    return cast(int | None, stressors.optional_positive_int(value))


def make_d18_default_args(config_names: str) -> argparse.Namespace:
    """Return D18 parser defaults without consuming this bridge's CLI."""
    old_argv = sys.argv[:]
    sys.argv = [str(Path(d18.__file__).name)]
    try:
        args = d18.parse_args()
    finally:
        sys.argv = old_argv
    args.configs = config_names
    return cast(argparse.Namespace, args)


def apply_d18_overrides(args: argparse.Namespace, d18_args: argparse.Namespace) -> None:
    """Apply bridge-level complexity overrides to the D18 config factory."""
    if args.d18_total_center_budget is not None:
        d18_args.total_center_budget = args.d18_total_center_budget
    if args.d18_tanh_width is not None:
        d18_args.tanh_width = args.d18_tanh_width
    if args.d18_poly_max_dim is not None:
        d18_args.poly_max_dim = args.d18_poly_max_dim
    if args.d18_fourier_max_dim is not None:
        d18_args.fourier_max_dim = args.d18_fourier_max_dim
    if args.d18_target_persistence_threshold is not None:
        d18_args.target_persistence_threshold = args.d18_target_persistence_threshold


def apply_resolved_config_overrides(
    configs: list[d18.UniversalConfig],
    args: argparse.Namespace,
) -> list[d18.UniversalConfig]:
    """Apply overrides to resolved configs that may hardcode promoted values."""
    resolved: list[d18.UniversalConfig] = []
    for config in configs:
        next_config = config
        if args.d18_tanh_width is not None:
            next_config = replace(
                next_config,
                basis_config=replace(
                    next_config.basis_config,
                    tanh_width=args.d18_tanh_width,
                ),
            )
        if args.d18_target_persistence_threshold is not None:
            next_config = replace(
                next_config,
                target_persistence_threshold=args.d18_target_persistence_threshold,
            )
        resolved.append(next_config)
    return resolved


def apply_deployment_transform(
    predictions: np.ndarray,
    transform: str,
) -> np.ndarray:
    """Apply a deterministic classifier deployment transform."""
    preds = np.asarray(predictions, dtype=np.float64)
    if transform == "raw":
        return cast(np.ndarray, preds)
    if transform == "clip01":
        return cast(np.ndarray, np.clip(preds, 0.0, 1.0))
    if transform == "softmax":
        shifted = preds - np.max(preds, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        softmax = exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)
        return cast(np.ndarray, np.asarray(softmax, dtype=np.float64))
    if transform == "hard":
        hard = np.zeros_like(preds)
        hard[np.arange(preds.shape[0]), np.argmax(preds, axis=1)] = 1.0
        return cast(np.ndarray, hard)
    raise ValueError(f"unknown deployment transform: {transform}")


def classifier_metrics_from_predictions(
    predictions: np.ndarray,
    y_test: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    """Return MSE and accuracy for prediction rows."""
    targets = np.eye(N_CLASSES, dtype=np.float64)[y_test]
    return {
        f"{prefix}mse": float(np.mean((predictions - targets) ** 2)),
        f"{prefix}accuracy": float(np.mean(np.argmax(predictions, axis=1) == y_test)),
    }


def evaluate_d18_classifier_views(
    learner: d18.SimpleUniversalResourceBasis,
    state: d18.UniversalState,
    test_views: np.ndarray,
    y_test: np.ndarray,
    *,
    deployment_transform: str,
) -> dict[str, float]:
    """Evaluate D18 averaged across held-out permutation views."""
    raw_mse_values: list[float] = []
    raw_accuracy_values: list[float] = []
    deployment_mse_values: list[float] = []
    deployment_accuracy_values: list[float] = []
    for view in test_views:
        preds = np.stack(
            [learner.predict(state, obs) for obs in view.astype(np.float64)]
        )
        raw_metrics = classifier_metrics_from_predictions(preds, y_test, prefix="test_")
        deployment_preds = apply_deployment_transform(preds, deployment_transform)
        deployment_metrics = classifier_metrics_from_predictions(
            deployment_preds,
            y_test,
            prefix="deployment_test_",
        )
        raw_mse_values.append(raw_metrics["test_mse"])
        raw_accuracy_values.append(raw_metrics["test_accuracy"])
        deployment_mse_values.append(deployment_metrics["deployment_test_mse"])
        deployment_accuracy_values.append(
            deployment_metrics["deployment_test_accuracy"]
        )
    return {
        "test_mse": float(np.mean(raw_mse_values)),
        "test_accuracy": float(np.mean(raw_accuracy_values)),
        "deployment_test_mse": float(np.mean(deployment_mse_values)),
        "deployment_test_accuracy": float(np.mean(deployment_accuracy_values)),
    }


def evaluate_mlp_classifier_views(
    learner: Any,
    state: Any,
    test_views: np.ndarray,
    y_test: np.ndarray,
    *,
    deployment_transform: str,
) -> dict[str, float]:
    """Evaluate an MLP averaged across held-out permutation views."""
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    raw_mse_values: list[float] = []
    raw_accuracy_values: list[float] = []
    deployment_mse_values: list[float] = []
    deployment_accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        mse = jnp.mean((preds - targets) ** 2)
        accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        raw_mse_values.append(float(mse))
        raw_accuracy_values.append(float(accuracy))
        preds_np = np.asarray(preds)
        deployment_preds = apply_deployment_transform(preds_np, deployment_transform)
        deployment_metrics = classifier_metrics_from_predictions(
            deployment_preds,
            y_test,
            prefix="deployment_test_",
        )
        deployment_mse_values.append(deployment_metrics["deployment_test_mse"])
        deployment_accuracy_values.append(
            deployment_metrics["deployment_test_accuracy"]
        )
    return {
        "test_mse": float(np.mean(raw_mse_values)),
        "test_accuracy": float(np.mean(raw_accuracy_values)),
        "deployment_test_mse": float(np.mean(deployment_mse_values)),
        "deployment_test_accuracy": float(np.mean(deployment_accuracy_values)),
    }


def make_stream(
    seed: int,
    args: argparse.Namespace,
) -> tuple[stressors.ClassificationDataset, stressors.ClassificationStream]:
    """Load a source dataset and materialize a compact OPMNIST-like stream."""
    loader_args = argparse.Namespace(
        mnist_source=args.mnist_source,
        allow_openml_download=args.allow_openml_download,
        allow_torchvision_download=args.allow_torchvision_download,
        mnist_split=args.mnist_split,
        openml_data_home=args.openml_data_home,
        torchvision_data_home=args.torchvision_data_home,
        openml_n_retries=args.openml_n_retries,
        openml_retry_delay=args.openml_retry_delay,
        train_fraction=args.train_fraction,
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
    )
    dataset = stressors.load_mnist_like_source(loader_args, seed)
    stream = stressors.make_permuted_classification_stream(
        dataset=dataset,
        steps=args.steps,
        seed=seed + 10_000,
        n_permutations=args.n_permutations,
        task_block_size=args.task_block_size,
        sample_with_replacement=args.sample_with_replacement,
        task_sampling=args.task_sampling,
        include_identity_permutation=args.include_identity_permutation,
        max_test_permutation_views=args.max_test_permutation_views,
        evaluate_all_permutation_views=args.evaluate_all_permutation_views,
    )
    return dataset, stream


def run_one_seed(
    seed: int,
    configs: list[d18.UniversalConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run fair MLP baselines and D18 on one OPMNIST bridge seed."""
    dataset, stream = make_stream(seed, args)
    labels_np = np.asarray(stream.labels)
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
            stream.observations,
            stream.targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(
            metrics,
            args.final_window,
            labels_np,
        )
        methods[method].update(
            evaluate_mlp_classifier_views(
                learner=mlp,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.mlp_deployment_transform,
            )
        )
        methods[method]["runtime_s"] = float(time.time() - t0)

    for offset, config in enumerate(configs):
        method = f"d18_{config.name}"
        print(f"opmnist seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = d18.run_universal_stream(
            stream.observations,
            stream.targets,
            config,
            seed=seed + 130_000 + offset,
        )
        methods[method] = d18.universal_summary(
            metrics,
            args.final_window,
            labels_np,
        )
        methods[method].update(
            evaluate_d18_classifier_views(
                learner=learner,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.d18_deployment_transform,
            )
        )
        methods[method]["runtime_s"] = float(time.time() - t0)

    meta = dict(dataset.metadata)
    meta.update(stream.metadata)
    record = {
        "dataset_name": DATASET_NAME,
        "seed": int(seed),
        "dataset": meta,
        "methods": methods,
    }
    return record, meta


def config_dict(
    args: argparse.Namespace,
    configs: list[d18.UniversalConfig],
) -> dict[str, Any]:
    """Return JSON-serializable run configuration."""
    return {
        "runner": "d18_opmnist_bridge",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "final_window": args.final_window,
        "mnist_source": args.mnist_source,
        "allow_openml_download": args.allow_openml_download,
        "allow_torchvision_download": args.allow_torchvision_download,
        "mnist_split": args.mnist_split,
        "train_fraction": args.train_fraction,
        "max_train_examples": args.max_train_examples,
        "max_test_examples": args.max_test_examples,
        "n_permutations": args.n_permutations,
        "task_block_size": args.task_block_size,
        "sample_with_replacement": args.sample_with_replacement,
        "task_sampling": args.task_sampling,
        "include_identity_permutation": args.include_identity_permutation,
        "max_test_permutation_views": args.max_test_permutation_views,
        "evaluate_all_permutation_views": args.evaluate_all_permutation_views,
        "mlp_methods": list(MLP_METHODS),
        "mlp_deployment_transform": args.mlp_deployment_transform,
        "mlp_step_size": args.mlp_step_size,
        "mlp_sparsity": args.mlp_sparsity,
        "d18_configs": [config.name for config in configs],
        "d18_deployment_transform": args.d18_deployment_transform,
        "d18_total_center_budget": configs[0].manager_config.total_center_budget,
        "d18_tanh_width": configs[0].basis_config.tanh_width,
        "d18_poly_max_dim": configs[0].poly_config.max_dim,
        "d18_fourier_max_dim": configs[0].basis_config.fourier_max_dim,
        "output_dir": str(args.output_dir),
        "result_prefix": args.result_prefix,
    }


def protocol_gate_rows(meta: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return stable protocol gate rows for reports."""
    return [(key, meta.get(key)) for key in PROTOCOL_GATE_KEYS if key in meta]


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format an aggregate metric cell."""
    if metric not in row:
        return ""
    value = row[metric]
    return f"{value['mean']:.6f} +/- {value['stderr']:.6f}"


def comparison_cell(comparison: dict[str, Any]) -> str:
    """Format a paired comparison row."""
    return (
        f"diff={comparison['paired_diff_mean_positive_favors_kernel']:+.6f}, "
        f"wins={comparison['wins_for_kernel']}/"
        f"{comparison['wins_for_mlp']}/{comparison['ties']}"
    )


def add_deployment_comparisons(
    aggregate: dict[str, Any],
    records: list[dict[str, Any]],
    candidate_methods: tuple[str, ...],
) -> dict[str, Any]:
    """Add bridge-specific comparisons for deployment-normalized metrics."""
    for dataset in sorted({record["dataset_name"] for record in records}):
        dataset_records = [record for record in records if record["dataset_name"] == dataset]
        if not dataset_records:
            continue
        comparisons = aggregate[dataset].setdefault("comparisons", {})
        method_names = list(dataset_records[0]["methods"])
        for metric in ("deployment_test_mse", "deployment_test_accuracy"):
            if metric not in dataset_records[0]["methods"][method_names[0]]:
                continue
            metric_comparisons = {
                method: compare_to_group(dataset_records, method, metric, MLP_METHODS)
                for method in candidate_methods
            }
            best_kernel_by_seed: list[str] = []
            diffs: list[float] = []
            for record in dataset_records:
                methods = record["methods"]
                kernel_values = {
                    method: methods[method][metric]
                    for method in candidate_methods
                    if metric in methods[method]
                }
                if is_higher_better(metric):
                    best_kernel = max(kernel_values, key=kernel_values.__getitem__)
                    best_mlp = max(MLP_METHODS, key=lambda name: methods[name][metric])
                else:
                    best_kernel = min(kernel_values, key=kernel_values.__getitem__)
                    best_mlp = min(MLP_METHODS, key=lambda name: methods[name][metric])
                best_kernel_by_seed.append(best_kernel)
                diffs.append(
                    paired_diff(
                        float(methods[best_kernel][metric]),
                        float(methods[best_mlp][metric]),
                        metric,
                    )
                )
            diff_arr = np.asarray(diffs, dtype=np.float64)
            metric_comparisons["best_kernel_vs_best_mlp"] = {
                "paired_diff_mean_positive_favors_kernel": float(np.mean(diff_arr)),
                "paired_diff_stderr": stderr(diff_arr),
                "wins_for_kernel": int(np.sum(diff_arr > 0.0)),
                "wins_for_mlp": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n": int(diff_arr.shape[0]),
                "diffs": diff_arr.tolist(),
                "best_kernel_counts": dict(
                    sorted(
                        (name, best_kernel_by_seed.count(name))
                        for name in set(best_kernel_by_seed)
                    )
                ),
            }
            comparisons[metric] = metric_comparisons
    return aggregate


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown report for the bridge run."""
    cfg = results["config"]
    aggregate = results["aggregate"][DATASET_NAME]
    candidate_methods = tuple(results["candidate_methods"])
    best = aggregate["comparisons"]
    protocol = results["datasets"][DATASET_NAME]
    lines = [
        "# D18 OPMNIST Bridge",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks, "
            f"block size {cfg['task_block_size']}."
        ),
        (
            f"Deployment transforms: D18 `{cfg['d18_deployment_transform']}`, "
            f"MLP `{cfg['mlp_deployment_transform']}`. Raw held-out metrics are "
            "reported separately from deployment-normalized held-out metrics."
        ),
        "",
        (
            "Learners: D18 configs "
            f"{', '.join(candidate_methods)} versus fair MLP baselines "
            f"{', '.join(MLP_METHODS)}."
        ),
        "",
        "## Protocol Gates",
        "",
        "| Gate | Value |",
        "| --- | --- |",
    ]
    for key, value in protocol_gate_rows(protocol):
        lines.append(f"| `{key}` | `{value}` |")

    metrics = (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "deployment_test_mse",
        "deployment_test_accuracy",
        "runtime_s",
    )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Method | Final MSE | Final Acc | Raw Test MSE | Raw Test Acc | "
            "Deploy Test MSE | Deploy Test Acc | Runtime s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, row in aggregate.items():
        if method == "comparisons":
            continue
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(metric_cell(row, metric) for metric in metrics)
            + " |"
        )

    lines.extend(["", "## D18 vs Best MLP", ""])
    for metric in (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "deployment_test_mse",
        "deployment_test_accuracy",
    ):
        if metric not in best:
            continue
        comparison = best[metric]["best_kernel_vs_best_mlp"]
        lines.append(f"- `{metric}`: {comparison_cell(comparison)}")

    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    if not protocol.get("is_true_mnist", False):
        lines.append(
            "- This run uses the local sklearn-digits 28x28 fallback; it is an "
            "OPMNIST analogue, not true MNIST."
        )
    if not protocol.get("matches_dohare_opmnist_core_protocol", False):
        lines.append(
            "- This run does not match the Dohare OPMNIST core protocol gates."
        )
    if not protocol.get("matches_dohare_opmnist_published_task_count", False):
        lines.append(
            "- This run does not match the 800-task/48M-step published OPMNIST "
            "task-count gate."
        )
    lines.append(
        "- This bridge is a materialized research runner; it is not yet the fused "
        "JAX/core production learner."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def apply_run_preset(args: argparse.Namespace) -> None:
    """Apply bridge run presets."""
    if args.mnist_published_scale:
        args.mnist_source = "openml"
        args.mnist_split = "canonical"
        args.max_train_examples = None
        args.max_test_examples = None
        args.task_block_size = stressors.DOHARE_OPMNIST_TASK_BLOCK_SIZE
        args.task_sampling = "sequential_epoch"
        args.sample_with_replacement = False

    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.n_permutations = 2
        args.task_block_size = 40
        args.max_train_examples = 200
        args.max_test_examples = 80
        args.task_sampling = "sequential_epoch"
        args.sample_with_replacement = False
        args.max_test_permutation_views = None


def validate_args(args: argparse.Namespace) -> None:
    """Validate bridge CLI arguments."""
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
    if args.d18_total_center_budget is not None and args.d18_total_center_budget <= 0:
        raise ValueError("--d18-total-center-budget must be positive")
    if args.d18_tanh_width is not None and args.d18_tanh_width <= 0:
        raise ValueError("--d18-tanh-width must be positive")
    if args.d18_poly_max_dim is not None and args.d18_poly_max_dim <= 0:
        raise ValueError("--d18-poly-max-dim must be positive")
    if args.d18_fourier_max_dim is not None and args.d18_fourier_max_dim <= 0:
        raise ValueError("--d18-fourier-max-dim must be positive")
    if (
        args.d18_target_persistence_threshold is not None
        and not 0.0 <= args.d18_target_persistence_threshold <= 1.0
    ):
        raise ValueError("--d18-target-persistence-threshold must be in [0, 1]")
    if args.d18_deployment_transform not in DEPLOYMENT_TRANSFORMS:
        raise ValueError("--d18-deployment-transform is invalid")
    if args.mlp_deployment_transform not in DEPLOYMENT_TRANSFORMS:
        raise ValueError("--mlp-deployment-transform is invalid")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=150)
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
    parser.add_argument("--max-train-examples", type=parse_optional_positive_int, default=4000)
    parser.add_argument("--max-test-examples", type=parse_optional_positive_int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--task-block-size", type=int, default=300)
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
    parser.add_argument(
        "--mnist-published-scale",
        action="store_true",
        help=(
            "Use OpenML canonical split and 60k task blocks. This bridge still "
            "materializes the requested step count and is not a 48M-step runner."
        ),
    )
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--mlp-deployment-transform",
        choices=DEPLOYMENT_TRANSFORMS,
        default="raw",
        help="Held-out deployment transform for MLP baselines.",
    )
    parser.add_argument("--d18-configs", default="step2_canonical")
    parser.add_argument(
        "--d18-deployment-transform",
        choices=DEPLOYMENT_TRANSFORMS,
        default="softmax",
        help="Held-out deployment transform for D18 candidate predictions.",
    )
    parser.add_argument("--d18-total-center-budget", type=int, default=None)
    parser.add_argument("--d18-tanh-width", type=int, default=None)
    parser.add_argument("--d18-poly-max-dim", type=int, default=None)
    parser.add_argument("--d18-fourier-max-dim", type=int, default=None)
    parser.add_argument("--d18-target-persistence-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Run the D18 OPMNIST bridge."""
    args = parse_args()
    apply_run_preset(args)
    validate_args(args)

    d18_args = make_d18_default_args(args.d18_configs)
    apply_d18_overrides(args, d18_args)
    d18.validate_args(d18_args)
    configs = apply_resolved_config_overrides(d18.make_configs(d18_args), args)
    candidate_methods = tuple(f"d18_{config.name}" for config in configs)

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, configs, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta
        comparison_values = {
            name: record["methods"][name]["final_window_mse"]
            for name in [*MLP_METHODS, *candidate_methods]
        }
        print(
            f"opmnist seed={seed}: final MSE "
            + ", ".join(f"{name}={value:.4f}" for name, value in comparison_values.items())
        )

    results = {
        "config": config_dict(args, configs),
        "datasets": datasets_meta,
        "records": records,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "aggregate": add_deployment_comparisons(
            aggregate_records(records, candidate_methods),
            records,
            candidate_methods,
        ),
        "wall_clock_s": float(time.time() - t0),
        "evidence_level": "compact_opmnist_bridge",
    }

    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    atomic_write_json(json_path, results)
    write_summary(summary_path, results)
    if args.note_path is not None:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path is not None:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

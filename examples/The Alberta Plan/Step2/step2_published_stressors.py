#!/usr/bin/env python3
"""Step 2 published-style external stressors.

This runner targets the remaining external-evidence gap after the strict
Step 2 portfolio result.  It keeps the learner side fixed by reusing
``step2_universal_portfolio.py`` and changes only the stream family:

* Online Permuted MNIST style classification.
* Slowly-Changing Regression style nonstationary supervised prediction.

The default is compact and local.  Real OpenML MNIST is only attempted when
``--mnist-source openml`` or ``--mnist-source auto --allow-openml-download`` is
set.  Otherwise the runner uses scikit-learn's bundled digits data, expanded to
28x28 pixels, so the stressor has MNIST-like input dimensionality and pixel
permutation while remaining reproducible without a large download.  That
fallback is deliberately labelled as a lightweight analogue, not as MNIST.
Use ``--mnist-published-scale`` with OpenML enabled to request the canonical
60,000/10,000 MNIST split and 60,000-example sequential task blocks.  Use
``--long-scr`` or ``--scr-preset dohare_paper`` for stronger
Slowly-Changing Regression variants.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_portfolio: Any = importlib.import_module("step2_universal_portfolio")
EXPERT_NAMES: tuple[str, ...] = tuple(_portfolio.EXPERT_NAMES)
METHOD_NAMES: tuple[str, ...] = tuple(_portfolio.METHOD_NAMES)
MLP_METHODS: tuple[str, ...] = tuple(_portfolio.MLP_METHODS)
aggregate_records: Any = _portfolio.aggregate_records
deployment_weights: Any = _portfolio.deployment_weights
portfolio_final_deployment_tracking_weights: Any = (
    _portfolio.final_deployment_tracking_weights
)
run_portfolio_stream: Any = _portfolio.run_portfolio_stream
summarize_prequential: Any = _portfolio.summarize_prequential
make_mlp: Any = _portfolio.make_mlp
make_upgd: Any = _portfolio.make_upgd
make_dynamic_sparse: Any = _portfolio.make_dynamic_sparse
LOSS_START: int = int(_portfolio.LOSS_START)
WEIGHT_START: int = int(_portfolio.WEIGHT_START)
MLP_WEIGHT_START: int = int(_portfolio.MLP_WEIGHT_START)
ACC_WEIGHT_START: int = int(_portfolio.ACC_WEIGHT_START)
ROUTER_START: int = int(_portfolio.ROUTER_START)
ALL_SELECTOR_START: int = int(_portfolio.ALL_SELECTOR_START)
MLP_SELECTOR_START: int = int(_portfolio.MLP_SELECTOR_START)
PRED_START: int = int(_portfolio.PRED_START)
ROUTER_NAMES: tuple[str, ...] = tuple(_portfolio.ROUTER_NAMES)
ONLINE_RETENTION_GUARD_ROUTE_ID: int = int(_portfolio.ONLINE_RETENTION_GUARD_ROUTE_ID)

N_CLASSES = 10
VALID_BENCHMARKS = ("permuted_mnist_like", "slowly_changing_regression")
DEFAULT_OUTPUT_DIR = Path("outputs/step2_published_stressors")
DEFAULT_RESULT_PREFIX = "published_stressors"
DOHARE_OPMNIST_TASKS = 800
DOHARE_OPMNIST_TASK_BLOCK_SIZE = 60_000
DOHARE_OPMNIST_TOTAL_STEPS = DOHARE_OPMNIST_TASKS * DOHARE_OPMNIST_TASK_BLOCK_SIZE
OPMNIST_CHECKPOINT_VERSION = 1
DOHARE_SCR_M_BITS = 20
DOHARE_SCR_SLOW_BITS = 15
DOHARE_SCR_FLIP_INTERVAL = 10_000
DOHARE_SCR_TARGET_HIDDEN = 100
DOHARE_SCR_BETA = 0.7
DOHARE_SCR_MIN_PUBLISHED_STEPS = 1_000_000
REFERENCES = (
    {
        "name": "Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning",
        "url": "https://www.nature.com/articles/s41586-024-07711-7",
        "relevance": "Online Permuted MNIST and Slowly-Changing Regression protocols.",
    },
    {
        "name": "shibhansh/loss-of-plasticity",
        "url": "https://github.com/shibhansh/loss-of-plasticity",
        "relevance": "Public reproduction repository for the loss-of-plasticity paper.",
    },
)


def final_deployment_tracking_weights(
    metrics: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return deployment weights, adding OPMNIST-only sparse expert deployment."""
    if args.digits_deployment_objective == "dynamic_sparse":
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
        weights[EXPERT_NAMES.index("dynamic_sparse")] = 1.0
        return weights
    return cast(np.ndarray, portfolio_final_deployment_tracking_weights(metrics, args))


@dataclass(frozen=True)
class ClassificationDataset:
    """Train/test arrays for an MNIST-like classification source."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ClassificationStream:
    """Materialized online classification stream plus held-out task views."""

    observations: jax.Array
    targets: jax.Array
    labels: jax.Array
    test_views: np.ndarray
    test_labels: np.ndarray
    feature_orders: tuple[np.ndarray, ...]
    metadata: dict[str, Any]


@dataclass
class PrequentialAccumulator:
    """Online metric accumulator for chunked portfolio runs."""

    n_steps: int
    loss_sum: np.ndarray
    correct_sum: np.ndarray
    final_losses: np.ndarray
    final_correct: np.ndarray
    weight_sum: np.ndarray
    mlp_weight_sum: np.ndarray
    acc_weight_sum: np.ndarray
    route_sum: float
    final_metric_row: np.ndarray | None


def parse_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated command-line value."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def optional_positive_int(value: str) -> int | None:
    """Parse a positive integer or an explicit all/none sentinel."""
    lowered = value.strip().lower()
    if lowered in {"all", "none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'all'")
    return parsed


def expand_benchmarks(value: str) -> tuple[str, ...]:
    """Expand benchmark aliases."""
    if value == "all":
        return VALID_BENCHMARKS
    names = parse_csv(value)
    unknown = sorted(set(names) - set(VALID_BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("--benchmarks cannot be empty")
    return names


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified train/test indices."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        n_train = min(max(n_train, 1), len(cls_idx) - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())
    train = np.asarray(train_idx, dtype=np.int32)
    test = np.asarray(test_idx, dtype=np.int32)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_subsample(
    y: np.ndarray,
    max_examples: int | None,
    seed: int,
) -> np.ndarray:
    """Return stratified indices into ``y`` capped at ``max_examples``."""
    if max_examples is None or max_examples <= 0 or max_examples >= y.shape[0]:
        return np.arange(y.shape[0], dtype=np.int32)

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    base = max_examples // classes.shape[0]
    remainder = max_examples % classes.shape[0]
    selected: list[int] = []
    for rank, cls in enumerate(classes):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        take = min(cls_idx.shape[0], base + int(rank < remainder))
        selected.extend(cls_idx[:take].tolist())
    selected_arr: np.ndarray = np.asarray(selected, dtype=np.int32)
    rng.shuffle(selected_arr)
    return selected_arr


def split_mnist_arrays(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_fraction: float,
    split: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return train/test indices for OpenML MNIST."""
    if split == "canonical":
        if x.shape[0] < 70_000:
            raise RuntimeError(
                "canonical MNIST split requires the 70,000-example OpenML array"
            )
        return (
            np.arange(60_000, dtype=np.int32),
            np.arange(60_000, 70_000, dtype=np.int32),
            "openml_canonical_60000_10000",
        )
    if split == "stratified":
        train_idx, test_idx = stratified_split(x, y, seed, train_fraction)
        return train_idx, test_idx, "stratified"
    raise ValueError(f"unknown MNIST split {split!r}")


def default_mnist_cache_dir(source: str) -> Path:
    """Return an output-scoped cache directory for optional MNIST downloads."""
    return REPO_ROOT / "outputs" / f"step2_published_mnist_{source}_cache"


def make_feature_orders(
    seed: int,
    feature_dim: int,
    n_permutations: int,
    include_identity_permutation: bool,
) -> tuple[np.ndarray, ...]:
    """Return reproducible pixel orders for OPMNIST-style tasks."""
    rng = np.random.default_rng(seed)
    feature_orders: list[np.ndarray] = []
    if include_identity_permutation:
        feature_orders.append(np.arange(feature_dim, dtype=np.int32))
    while len(feature_orders) < n_permutations:
        feature_orders.append(rng.permutation(feature_dim).astype(np.int32))
    return tuple(feature_orders)


def opmnist_task_ids_for_steps(
    steps: int,
    n_permutations: int,
    task_block_size: int,
) -> list[int]:
    """Return task ids touched by a sequential OPMNIST run."""
    if steps <= 0:
        return []
    n_blocks = int(math.ceil(steps / task_block_size))
    return sorted({block_idx % n_permutations for block_idx in range(n_blocks)})


def opmnist_test_task_ids(
    observed_task_ids: list[int],
    n_permutations: int,
    max_test_permutation_views: int | None,
    evaluate_all_permutation_views: bool,
) -> list[int]:
    """Return held-out permutation views selected for evaluation."""
    task_ids = (
        list(range(n_permutations))
        if evaluate_all_permutation_views
        else list(observed_task_ids)
    )
    if max_test_permutation_views is not None:
        task_ids = task_ids[:max_test_permutation_views]
    if not task_ids:
        raise ValueError("no held-out permutation views selected for evaluation")
    return task_ids


def opmnist_protocol_metadata(
    dataset: ClassificationDataset,
    *,
    steps: int,
    seed: int,
    n_permutations: int,
    task_block_size: int,
    sample_with_replacement: bool,
    task_sampling: str,
    include_identity_permutation: bool,
    max_test_permutation_views: int | None,
    evaluate_all_permutation_views: bool,
    observed_task_ids: list[int],
    test_task_ids: list[int],
    streaming_runner: bool,
    chunk_size: int | None = None,
    resume_checkpoint_path: str | None = None,
    checkpoint_loaded: bool = False,
) -> dict[str, Any]:
    """Return explicit OPMNIST protocol gates for materialized or chunked runs."""
    full_blocks = int(steps // task_block_size)
    partial_task_steps = int(steps % task_block_size)
    source_full_mnist = bool(dataset.metadata.get("is_full_mnist_split", False))
    true_mnist = bool(dataset.metadata.get("is_true_mnist", False))
    full_mnist_task_blocks = bool(
        source_full_mnist
        and task_sampling == "sequential_epoch"
        and not sample_with_replacement
        and task_block_size >= DOHARE_OPMNIST_TASK_BLOCK_SIZE
    )
    single_pass_task_order = bool(
        task_sampling == "sequential_epoch"
        and not sample_with_replacement
        and task_block_size <= dataset.x_train.shape[0]
    )
    random_pixel_permutations = bool(not include_identity_permutation)
    core_protocol = bool(
        true_mnist
        and source_full_mnist
        and task_sampling == "sequential_epoch"
        and not sample_with_replacement
        and random_pixel_permutations
        and task_block_size >= DOHARE_OPMNIST_TASK_BLOCK_SIZE
    )
    observed_task_blocks = int(math.ceil(steps / task_block_size))
    completed_published_blocks = int(full_blocks if full_mnist_task_blocks else 0)
    published_task_count = bool(
        core_protocol
        and n_permutations >= DOHARE_OPMNIST_TASKS
        and completed_published_blocks >= DOHARE_OPMNIST_TASKS
        and steps >= DOHARE_OPMNIST_TOTAL_STEPS
    )
    all_observed_views_evaluated = bool(set(observed_task_ids).issubset(test_task_ids))
    all_configured_views_evaluated = bool(
        len(test_task_ids) == n_permutations
        and set(test_task_ids) == set(range(n_permutations))
    )
    return {
        "benchmark": "permuted_mnist_like",
        "protocol": (
            "chunked_online_permuted_pixels"
            if streaming_runner
            else "compact_online_permuted_pixels"
        ),
        "steps": int(steps),
        "sequence_seed": int(seed),
        "n_permutations": int(n_permutations),
        "task_block_size": int(task_block_size),
        "observed_task_blocks": observed_task_blocks,
        "completed_full_task_blocks": full_blocks,
        "partial_task_steps": partial_task_steps,
        "opmnist_completed_full_60000_task_blocks": completed_published_blocks,
        "sample_with_replacement": bool(sample_with_replacement),
        "task_sampling": task_sampling,
        "include_identity_permutation": bool(include_identity_permutation),
        "permutations_are_random_pixel_orders": random_pixel_permutations,
        "random_pixel_permutation_seed": int(seed),
        "task_id_provided_to_learner": False,
        "prediction_before_update_every_step": True,
        "all_experts_update_every_step": True,
        "single_pass_examples_within_task": single_pass_task_order,
        "sequential_single_pass_task_epochs": single_pass_task_order,
        "evaluate_all_permutation_views": bool(evaluate_all_permutation_views),
        "max_test_permutation_views": max_test_permutation_views,
        "test_permutation_views": int(len(test_task_ids)),
        "test_task_ids_evaluated": test_task_ids,
        "test_views_cover_observed_permutations": all_observed_views_evaluated,
        "test_views_cover_all_permutations": all_configured_views_evaluated,
        "heldout_deployment_objective": "post-run selected weights; no task id",
        "full_mnist_task_blocks": full_mnist_task_blocks,
        "matches_dohare_opmnist_core_protocol": core_protocol,
        "matches_dohare_opmnist_published_task_count": published_task_count,
        "task_ids_observed": observed_task_ids,
        "streaming_runner": bool(streaming_runner),
        "stream_chunk_size": chunk_size,
        "resumable_runner": bool(resume_checkpoint_path),
        "resume_checkpoint_path": resume_checkpoint_path,
        "checkpoint_loaded": bool(checkpoint_loaded),
        "published_protocol_delta": (
            "Dohare et al. used true MNIST, randomized pixel permutations, "
            "60,000 examples per task, 800 tasks in the main OPMNIST protocol, "
            "a single pass through each task in random order, no mini-batches, "
            "and no task-switch indication. This runner is only a published-scale "
            "OPMNIST run when matches_dohare_opmnist_core_protocol and "
            "matches_dohare_opmnist_published_task_count are both true."
        ),
    }


def resize_digits_8x8_to_28x28(x: np.ndarray) -> np.ndarray:
    """Nearest-neighbor expand flattened sklearn 8x8 digits to 28x28."""
    if x.ndim != 2 or x.shape[1] != 64:
        raise ValueError("expected flattened 8x8 digit images with shape (n, 64)")
    row_idx = np.rint(np.linspace(0, 7, 28)).astype(np.int32)
    col_idx = np.rint(np.linspace(0, 7, 28)).astype(np.int32)
    images = x.reshape((-1, 8, 8))
    resized = images[:, row_idx, :]
    resized = resized[:, :, col_idx]
    return resized.reshape((x.shape[0], 28 * 28)).astype(np.float32)


def load_sklearn_digits_source(
    seed: int,
    train_fraction: float,
    max_train_examples: int | None,
    max_test_examples: int | None,
    expand_to_28x28: bool,
) -> ClassificationDataset:
    """Load the bundled sklearn digits fallback."""
    try:
        from sklearn.datasets import load_digits  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = "scikit-learn is required for the local MNIST-like fallback."
        raise RuntimeError(msg) from exc

    raw = load_digits()
    x = np.asarray(raw.data, dtype=np.float32) / 16.0
    y = np.asarray(raw.target, dtype=np.int32)
    if expand_to_28x28:
        x = resize_digits_8x8_to_28x28(x)

    train_idx, test_idx = stratified_split(x, y, seed, train_fraction)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    train_sub = stratified_subsample(y_train, max_train_examples, seed + 101)
    test_sub = stratified_subsample(y_test, max_test_examples, seed + 202)
    x_train = x_train[train_sub]
    y_train = y_train[train_sub]
    x_test = x_test[test_sub]
    y_test = y_test[test_sub]

    metadata = {
        "dataset": "sklearn.datasets.load_digits",
        "source_kind": "local_sklearn_digits_28x28"
        if expand_to_28x28
        else "local_sklearn_digits_8x8",
        "is_true_mnist": False,
        "description": (
            "Bundled sklearn 8x8 handwritten digits expanded to 28x28 by "
            "nearest-neighbor resizing."
            if expand_to_28x28
            else "Bundled sklearn 8x8 handwritten digits."
        ),
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_CLASSES,
        "train_fraction": float(train_fraction),
        "split_seed": int(seed),
        "max_train_examples": max_train_examples,
        "max_test_examples": max_test_examples,
        "limitations": (
            "This is not MNIST. It preserves the 10-class handwritten-digit "
            "classification form and, when expanded, the 784-dimensional pixel "
            "permutation stressor, but it has only 1,797 source examples and "
            "simpler images than MNIST."
        ),
    }
    return ClassificationDataset(x_train, y_train, x_test, y_test, metadata)


def load_openml_mnist_source(
    seed: int,
    train_fraction: float,
    max_train_examples: int | None,
    max_test_examples: int | None,
    split: str,
    data_home: Path | None,
    n_retries: int,
    retry_delay: float,
) -> ClassificationDataset:
    """Load real MNIST from OpenML and create a stratified compact split."""
    try:
        from sklearn.datasets import fetch_openml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = "scikit-learn is required for OpenML MNIST."
        raise RuntimeError(msg) from exc

    fetch_kwargs: dict[str, Any] = {
        "name": "mnist_784",
        "version": 1,
        "as_frame": False,
        "n_retries": n_retries,
        "delay": retry_delay,
    }
    cache_dir = data_home if data_home is not None else default_mnist_cache_dir("openml")
    fetch_kwargs["data_home"] = str(cache_dir)
    raw = fetch_openml(**fetch_kwargs)
    x = np.asarray(raw.data, dtype=np.float32) / 255.0
    y = np.asarray(raw.target, dtype=np.int32)

    train_idx, test_idx, split_kind = split_mnist_arrays(
        x=x,
        y=y,
        seed=seed,
        train_fraction=train_fraction,
        split=split,
    )
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    train_sub = stratified_subsample(y_train, max_train_examples, seed + 101)
    test_sub = stratified_subsample(y_test, max_test_examples, seed + 202)
    x_train = x_train[train_sub]
    y_train = y_train[train_sub]
    x_test = x_test[test_sub]
    y_test = y_test[test_sub]

    metadata = {
        "dataset": "sklearn.datasets.fetch_openml('mnist_784', version=1)",
        "source_kind": "openml_mnist_784",
        "is_true_mnist": True,
        "description": "OpenML MNIST 28x28 handwritten digits.",
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_CLASSES,
        "train_fraction": float(train_fraction),
        "split": split_kind,
        "split_seed": int(seed),
        "max_train_examples": max_train_examples,
        "max_test_examples": max_test_examples,
        "openml_data_home": str(cache_dir),
        "is_full_mnist_split": bool(
            split_kind == "openml_canonical_60000_10000"
            and max_train_examples is None
            and max_test_examples is None
            and x_train.shape[0] >= 60_000
            and x_test.shape[0] >= 10_000
        ),
        "limitations": (
            "True OpenML MNIST. This is a full source split only when "
            "is_full_mnist_split=true; the online protocol still depends on "
            "task_block_size, task_sampling, and n_permutations."
        ),
    }
    return ClassificationDataset(x_train, y_train, x_test, y_test, metadata)


def load_torchvision_mnist_source(
    seed: int,
    max_train_examples: int | None,
    max_test_examples: int | None,
    data_home: Path | None,
    download: bool,
) -> ClassificationDataset:
    """Load true MNIST from torchvision's canonical train/test datasets."""
    try:
        from torchvision.datasets import MNIST  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "torchvision is required for --mnist-source torchvision. Install it "
            "separately; it is intentionally optional for this runner."
        )
        raise RuntimeError(msg) from exc

    cache_dir = data_home if data_home is not None else default_mnist_cache_dir("torchvision")
    try:
        train = MNIST(root=str(cache_dir), train=True, download=download)
        test = MNIST(root=str(cache_dir), train=False, download=download)
    except RuntimeError as exc:  # pragma: no cover - cache/download dependent
        if not download:
            msg = (
                "torchvision MNIST is not available in the local cache. Re-run with "
                "--allow-torchvision-download or use --mnist-source openml "
                "--allow-openml-download."
            )
            raise RuntimeError(msg) from exc
        raise

    x_train = np.asarray(train.data.numpy(), dtype=np.float32).reshape((-1, 28 * 28)) / 255.0
    y_train = np.asarray(train.targets.numpy(), dtype=np.int32)
    x_test = np.asarray(test.data.numpy(), dtype=np.float32).reshape((-1, 28 * 28)) / 255.0
    y_test = np.asarray(test.targets.numpy(), dtype=np.int32)

    train_sub = stratified_subsample(y_train, max_train_examples, seed + 101)
    test_sub = stratified_subsample(y_test, max_test_examples, seed + 202)
    x_train = x_train[train_sub]
    y_train = y_train[train_sub]
    x_test = x_test[test_sub]
    y_test = y_test[test_sub]

    metadata = {
        "dataset": "torchvision.datasets.MNIST",
        "source_kind": "torchvision_mnist",
        "is_true_mnist": True,
        "description": "torchvision MNIST 28x28 handwritten digits.",
        "n_total": 70_000,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_CLASSES,
        "train_fraction": None,
        "split": "torchvision_canonical_60000_10000",
        "split_seed": int(seed),
        "max_train_examples": max_train_examples,
        "max_test_examples": max_test_examples,
        "torchvision_data_home": str(cache_dir),
        "torchvision_download": bool(download),
        "is_full_mnist_split": bool(
            max_train_examples is None
            and max_test_examples is None
            and x_train.shape[0] >= 60_000
            and x_test.shape[0] >= 10_000
        ),
        "limitations": (
            "True MNIST via torchvision's canonical train/test split. This is "
            "a full source split only when is_full_mnist_split=true; the online "
            "protocol still depends on task_block_size, task_sampling, and "
            "n_permutations."
        ),
    }
    return ClassificationDataset(x_train, y_train, x_test, y_test, metadata)


def load_mnist_like_source(args: argparse.Namespace, seed: int) -> ClassificationDataset:
    """Load the requested MNIST-like source with documented fallback behavior."""
    if args.mnist_source == "openml":
        if not args.allow_openml_download:
            raise RuntimeError(
                "--mnist-source openml requires --allow-openml-download because "
                "OpenML can download a large dataset."
            )
        return load_openml_mnist_source(
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
            split=args.mnist_split,
            data_home=args.openml_data_home,
            n_retries=args.openml_n_retries,
            retry_delay=args.openml_retry_delay,
        )

    if args.mnist_source == "torchvision":
        return load_torchvision_mnist_source(
            seed=seed,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
            data_home=args.torchvision_data_home,
            download=args.allow_torchvision_download,
        )

    if args.mnist_source == "auto" and args.allow_openml_download:
        try:
            return load_openml_mnist_source(
                seed=seed,
                train_fraction=args.train_fraction,
                max_train_examples=args.max_train_examples,
                max_test_examples=args.max_test_examples,
                split=args.mnist_split,
                data_home=args.openml_data_home,
                n_retries=args.openml_n_retries,
                retry_delay=args.openml_retry_delay,
            )
        except Exception as exc:  # pragma: no cover - network/cache dependent
            print(f"OpenML MNIST unavailable ({exc}); falling back to sklearn digits.")

    if args.mnist_source == "auto":
        try:
            return load_torchvision_mnist_source(
                seed=seed,
                max_train_examples=args.max_train_examples,
                max_test_examples=args.max_test_examples,
                data_home=args.torchvision_data_home,
                download=args.allow_torchvision_download,
            )
        except Exception as exc:  # pragma: no cover - environment/cache dependent
            print(f"torchvision MNIST unavailable ({exc}); falling back to sklearn digits.")

    if args.mnist_source in {"auto", "sklearn_digits_28x28"}:
        return load_sklearn_digits_source(
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
            expand_to_28x28=True,
        )

    if args.mnist_source == "sklearn_digits_8x8":
        return load_sklearn_digits_source(
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
            expand_to_28x28=False,
        )

    raise ValueError(f"unknown mnist source {args.mnist_source!r}")


def make_permuted_classification_stream(
    dataset: ClassificationDataset,
    steps: int,
    seed: int,
    n_permutations: int,
    task_block_size: int,
    sample_with_replacement: bool,
    task_sampling: str,
    include_identity_permutation: bool = False,
    max_test_permutation_views: int | None = None,
    evaluate_all_permutation_views: bool = False,
) -> ClassificationStream:
    """Create a compact Online Permuted MNIST-style stream."""
    if n_permutations < 2:
        raise ValueError("n_permutations must be at least 2")
    if task_block_size <= 0:
        raise ValueError("task_block_size must be positive")

    feature_dim = int(dataset.x_train.shape[1])
    feature_orders = make_feature_orders(
        seed=seed,
        feature_dim=feature_dim,
        n_permutations=n_permutations,
        include_identity_permutation=include_identity_permutation,
    )
    rng = np.random.default_rng(seed)

    if task_sampling not in {"random", "sequential_epoch"}:
        raise ValueError("task_sampling must be 'random' or 'sequential_epoch'")

    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    sample_orders = [
        rng.permutation(dataset.x_train.shape[0]).astype(np.int32)
        for _ in range(n_permutations)
    ]
    sample_cursors = np.zeros(n_permutations, dtype=np.int32)
    total = 0
    block_idx = 0
    while total < steps:
        task_id = block_idx % n_permutations
        block_len = min(task_block_size, steps - total)
        replace = sample_with_replacement or block_len > dataset.x_train.shape[0]
        if task_sampling == "random":
            idx = rng.choice(dataset.x_train.shape[0], size=block_len, replace=replace)
        else:
            pieces: list[np.ndarray] = []
            remaining = block_len
            while remaining > 0:
                cursor = int(sample_cursors[task_id])
                order = sample_orders[task_id]
                if cursor >= order.shape[0]:
                    order = rng.permutation(dataset.x_train.shape[0]).astype(np.int32)
                    sample_orders[task_id] = order
                    cursor = 0
                take = min(remaining, order.shape[0] - cursor)
                pieces.append(order[cursor : cursor + take])
                sample_cursors[task_id] = cursor + take
                remaining -= take
            idx = np.concatenate(pieces).astype(np.int32)
        order = feature_orders[task_id]
        obs_parts.append(dataset.x_train[idx][:, order])
        label_parts.append(dataset.y_train[idx])
        task_parts.append(np.full(block_len, task_id, dtype=np.int32))
        total += block_len
        block_idx += 1

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    task_ids = np.concatenate(task_parts, axis=0).astype(np.int32)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    observed_task_ids = sorted(int(value) for value in np.unique(task_ids))
    test_task_ids = opmnist_test_task_ids(
        observed_task_ids=observed_task_ids,
        n_permutations=n_permutations,
        max_test_permutation_views=max_test_permutation_views,
        evaluate_all_permutation_views=evaluate_all_permutation_views,
    )
    test_views = np.stack(
        [dataset.x_test[:, feature_orders[task_id]] for task_id in test_task_ids],
        axis=0,
    ).astype(np.float32)
    metadata = opmnist_protocol_metadata(
        dataset=dataset,
        steps=steps,
        seed=seed,
        n_permutations=n_permutations,
        task_block_size=task_block_size,
        sample_with_replacement=sample_with_replacement,
        task_sampling=task_sampling,
        include_identity_permutation=include_identity_permutation,
        max_test_permutation_views=max_test_permutation_views,
        evaluate_all_permutation_views=evaluate_all_permutation_views,
        observed_task_ids=observed_task_ids,
        test_task_ids=test_task_ids,
        streaming_runner=False,
    )
    return ClassificationStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        test_views=test_views,
        test_labels=dataset.y_test.astype(np.int32),
        feature_orders=tuple(feature_orders),
        metadata=metadata,
    )


def make_slowly_changing_regression_stream(
    steps: int,
    seed: int,
    m_bits: int,
    slow_bits: int,
    flip_interval: int,
    target_hidden: int,
    beta: float,
    noise_std: float,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Create a lightweight Slowly-Changing Regression analogue."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if m_bits <= 0:
        raise ValueError("m_bits must be positive")
    if not 0 < slow_bits <= m_bits:
        raise ValueError("slow_bits must be in [1, m_bits]")
    if flip_interval <= 0:
        raise ValueError("flip_interval must be positive")
    if target_hidden <= 0:
        raise ValueError("target_hidden must be positive")

    rng = np.random.default_rng(seed)
    feature_dim = m_bits + 1
    input_weights = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(target_hidden, feature_dim),
    ).astype(np.float32)
    output_weights = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(target_hidden,),
    ).astype(np.float32)
    negative_counts = np.sum(input_weights < 0.0, axis=1).astype(np.float32)
    thresholds = float(feature_dim) * float(beta) - negative_counts

    slow_state = rng.integers(0, 2, size=slow_bits, dtype=np.int32)
    observations: np.ndarray = np.zeros((steps, feature_dim), dtype=np.float32)
    targets: np.ndarray = np.zeros((steps, 1), dtype=np.float32)
    flips: list[dict[str, int]] = []
    for t in range(steps):
        if t > 0 and t % flip_interval == 0:
            bit = int(rng.integers(0, slow_bits))
            slow_state[bit] = 1 - slow_state[bit]
            flips.append({"step": int(t), "bit": bit})

        random_bits = rng.integers(0, 2, size=m_bits - slow_bits, dtype=np.int32)
        obs = np.concatenate(
            [
                slow_state.astype(np.float32),
                random_bits.astype(np.float32),
                np.ones(1, dtype=np.float32),
            ]
        )
        hidden = (obs @ input_weights.T > thresholds).astype(np.float32)
        target = float(hidden @ output_weights / math.sqrt(target_hidden))
        if noise_std > 0.0:
            target += float(rng.normal(0.0, noise_std))
        observations[t] = obs
        targets[t, 0] = target

    matches_public_config = bool(
        m_bits == DOHARE_SCR_M_BITS
        and slow_bits == DOHARE_SCR_SLOW_BITS
        and flip_interval == DOHARE_SCR_FLIP_INTERVAL
        and target_hidden == DOHARE_SCR_TARGET_HIDDEN
        and abs(beta - DOHARE_SCR_BETA) < 1e-12
    )
    reaches_published_steps = steps >= DOHARE_SCR_MIN_PUBLISHED_STEPS
    metadata = {
        "benchmark": "slowly_changing_regression",
        "protocol": "lightweight_slowly_changing_binary_regression",
        "steps": int(steps),
        "sequence_seed": int(seed),
        "feature_dim": int(feature_dim),
        "n_heads": 1,
        "m_bits": int(m_bits),
        "slow_bits": int(slow_bits),
        "random_bits": int(m_bits - slow_bits),
        "flip_interval": int(flip_interval),
        "target_hidden": int(target_hidden),
        "beta": float(beta),
        "noise_std": float(noise_std),
        "n_flips": int(len(flips)),
        "first_flips": flips[:10],
        "matches_dohare_public_config": matches_public_config,
        "matches_dohare_target_family": True,
        "task_id_provided_to_learner": False,
        "uses_online_stream_only": True,
        "uses_fixed_target_network": True,
        "dohare_public_scr_min_steps": DOHARE_SCR_MIN_PUBLISHED_STEPS,
        "meets_dohare_public_scr_step_count": bool(reaches_published_steps),
        "matches_dohare_public_scr_protocol": bool(
            matches_public_config and reaches_published_steps
        ),
        "published_protocol_delta": (
            "Matches the main structure described by Dohare et al.: binary "
            "inputs with slow-changing bits, iid random bits, a constant bias, "
            "and a fixed LTU target network. This local run is far shorter than "
            "the paper/reproduction repository's million-plus-example runs and "
            "uses the Step 2 portfolio architecture rather than the paper's "
            "learner architecture."
        ),
    }
    return jnp.asarray(observations), jnp.asarray(targets), metadata


def evaluate_classifier_views(
    learner: Any,
    state: Any,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate one expert averaged over all held-out permutation views."""
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        mse = jnp.mean((preds - targets) ** 2)
        accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        mse_values.append(float(mse))
        accuracy_values.append(float(accuracy))
    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def evaluate_mixture_classifier_views(
    final_states: dict[str, tuple[Any, Any]],
    weights: np.ndarray,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final convex portfolio averaged over held-out views."""
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    weights_jax = jnp.asarray(weights.astype(np.float32))
    mse_values: list[float] = []
    accuracy_values: list[float] = []

    def predict(obs: jax.Array) -> jax.Array:
        preds = jnp.stack(
            [
                final_states[expert][0].predict(final_states[expert][1], obs)
                for expert in EXPERT_NAMES
            ],
            axis=0,
        )
        return jnp.sum(weights_jax[:, None] * preds, axis=0)

    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        preds = jax.vmap(predict)(observations)
        mse = jnp.mean((preds - targets) ** 2)
        accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        mse_values.append(float(mse))
        accuracy_values.append(float(accuracy))
    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def init_prequential_accumulator() -> PrequentialAccumulator:
    """Create an empty online metric accumulator."""
    n_methods = len(METHOD_NAMES)
    return PrequentialAccumulator(
        n_steps=0,
        loss_sum=np.zeros(n_methods, dtype=np.float64),
        correct_sum=np.zeros(n_methods, dtype=np.float64),
        final_losses=np.zeros((0, n_methods), dtype=np.float32),
        final_correct=np.zeros((0, n_methods), dtype=np.float32),
        weight_sum=np.zeros(len(EXPERT_NAMES), dtype=np.float64),
        mlp_weight_sum=np.zeros(len(MLP_METHODS), dtype=np.float64),
        acc_weight_sum=np.zeros(len(EXPERT_NAMES), dtype=np.float64),
        route_sum=0.0,
        final_metric_row=None,
    )


def update_prequential_accumulator(
    accumulator: PrequentialAccumulator,
    metrics: np.ndarray,
    labels: np.ndarray,
    final_window: int,
) -> PrequentialAccumulator:
    """Fold one scanned chunk into online means and final-window buffers."""
    loss_cols = np.asarray(
        [0, *(LOSS_START + idx for idx in range(len(EXPERT_NAMES)))],
        dtype=np.int32,
    )
    pred_cols = np.asarray(
        [PRED_START + idx for idx in range(len(METHOD_NAMES))],
        dtype=np.int32,
    )
    losses = metrics[:, loss_cols].astype(np.float32)
    correct = (
        metrics[:, pred_cols].astype(np.int32)
        == labels[:, None].astype(np.int32)
    ).astype(np.float32)

    final_losses = np.concatenate([accumulator.final_losses, losses], axis=0)[
        -final_window:
    ]
    final_correct = np.concatenate([accumulator.final_correct, correct], axis=0)[
        -final_window:
    ]
    return PrequentialAccumulator(
        n_steps=accumulator.n_steps + int(metrics.shape[0]),
        loss_sum=accumulator.loss_sum + np.sum(losses, axis=0, dtype=np.float64),
        correct_sum=accumulator.correct_sum + np.sum(correct, axis=0, dtype=np.float64),
        final_losses=final_losses,
        final_correct=final_correct,
        weight_sum=(
            accumulator.weight_sum
            + np.sum(
                metrics[:, WEIGHT_START : WEIGHT_START + len(EXPERT_NAMES)],
                axis=0,
                dtype=np.float64,
            )
        ),
        mlp_weight_sum=(
            accumulator.mlp_weight_sum
            + np.sum(
                metrics[:, MLP_WEIGHT_START : MLP_WEIGHT_START + len(MLP_METHODS)],
                axis=0,
                dtype=np.float64,
            )
        ),
        acc_weight_sum=(
            accumulator.acc_weight_sum
            + np.sum(
                metrics[:, ACC_WEIGHT_START : ACC_WEIGHT_START + len(EXPERT_NAMES)],
                axis=0,
                dtype=np.float64,
            )
        ),
        route_sum=accumulator.route_sum + float(np.sum(metrics[:, ROUTER_START])),
        final_metric_row=metrics[-1].astype(np.float32),
    )


def summarize_prequential_accumulator(
    accumulator: PrequentialAccumulator,
) -> dict[str, dict[str, float]]:
    """Summarize online metrics without retaining all per-step rows."""
    if accumulator.n_steps <= 0:
        raise ValueError("cannot summarize an empty prequential accumulator")
    if accumulator.final_metric_row is None:
        raise ValueError("missing final metric row")

    summary: dict[str, dict[str, float]] = {}
    for method_idx, method in enumerate(METHOD_NAMES):
        entry = {
            "online_mean_mse": float(accumulator.loss_sum[method_idx] / accumulator.n_steps),
            "final_window_mse": float(np.mean(accumulator.final_losses[:, method_idx])),
            "online_mean_accuracy": float(
                accumulator.correct_sum[method_idx] / accumulator.n_steps
            ),
            "final_window_accuracy": float(np.mean(accumulator.final_correct[:, method_idx])),
        }
        if method == "mixture":
            for idx, expert in enumerate(EXPERT_NAMES):
                entry[f"mean_{expert}_weight"] = float(
                    accumulator.weight_sum[idx] / accumulator.n_steps
                )
                entry[f"final_{expert}_weight"] = float(
                    accumulator.final_metric_row[WEIGHT_START + idx]
                )
            for idx, method_name in enumerate(MLP_METHODS):
                entry[f"mean_mlp_guard_{method_name}_weight"] = float(
                    accumulator.mlp_weight_sum[idx] / accumulator.n_steps
                )
                entry[f"final_mlp_guard_{method_name}_weight"] = float(
                    accumulator.final_metric_row[MLP_WEIGHT_START + idx]
                )
            for idx, expert in enumerate(EXPERT_NAMES):
                entry[f"mean_accuracy_{expert}_weight"] = float(
                    accumulator.acc_weight_sum[idx] / accumulator.n_steps
                )
                entry[f"final_accuracy_{expert}_weight"] = float(
                    accumulator.final_metric_row[ACC_WEIGHT_START + idx]
                )
            entry["mean_meta_route_id"] = float(accumulator.route_sum / accumulator.n_steps)
            entry["final_meta_route_id"] = float(accumulator.final_metric_row[ROUTER_START])
        summary[method] = entry
    return summary


def make_portfolio_components_and_carry(
    feature_dim: int,
    n_heads: int,
    key: jax.Array,
    args: argparse.Namespace,
) -> tuple[tuple[Any, Any, Any, Any, Any], tuple[Any, ...]]:
    """Initialize portfolio learners and scan carry for chunked runs."""
    keys = jr.split(key, len(EXPERT_NAMES))
    mlp64 = make_mlp(n_heads, (64,), args.step_size, args.sparsity)
    mlp128 = make_mlp(n_heads, (128,), args.step_size, args.sparsity)
    mlp6464 = make_mlp(n_heads, (64, 64), args.step_size, args.sparsity)
    upgd = make_upgd(
        n_heads=n_heads,
        hidden_sizes=(64,),
        step_size=args.step_size,
        sparsity=args.sparsity,
        perturbation_sigma=args.perturbation_sigma,
        perturbation_warmup_steps=args.perturbation_warmup_steps,
        perturbation_ramp_steps=args.perturbation_ramp_steps,
    )
    dynamic = make_dynamic_sparse(
        n_heads=n_heads,
        hidden_size=args.dynamic_hidden_size,
        step_size=args.step_size,
        sparsity=args.sparsity,
        utility_decay=args.dynamic_utility_decay,
        rewire_interval=args.dynamic_rewire_interval,
        unit_replacement_rate=args.dynamic_unit_replacement_rate,
    )
    recent_buffer_size = int(args.final_window)
    carry = (
        mlp64.init(feature_dim, keys[0]),
        mlp128.init(feature_dim, keys[1]),
        mlp6464.init(feature_dim, keys[2]),
        upgd.init(feature_dim, keys[3]),
        dynamic.init(feature_dim, keys[4]),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(MLP_METHODS), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(ROUTER_NAMES), dtype=jnp.float32),
        jnp.zeros(n_heads, dtype=jnp.float32),
        jnp.zeros(n_heads, dtype=jnp.float32),
        jnp.zeros((recent_buffer_size, n_heads), dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    return (mlp64, mlp128, mlp6464, upgd, dynamic), carry


def scan_portfolio_chunk(
    learners: tuple[Any, Any, Any, Any, Any],
    carry: tuple[Any, ...],
    observations: jax.Array,
    targets: jax.Array,
    args: argparse.Namespace,
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Advance an initialized portfolio over one materialized chunk."""
    mlp64, mlp128, mlp6464, upgd, dynamic = learners
    n_heads = int(targets.shape[1])
    recent_buffer_size = int(args.final_window)
    eta = jnp.array(args.hedge_eta, dtype=jnp.float32)
    discount = jnp.array(args.hedge_discount, dtype=jnp.float32)
    router_decay = jnp.array(args.router_decay, dtype=jnp.float32)

    def step_fn(step_carry: tuple[Any, ...], inputs: tuple[jax.Array, jax.Array]) -> Any:
        (
            mlp64_s,
            mlp128_s,
            mlp6464_s,
            upgd_s,
            dynamic_s,
            log_w,
            mlp_log_w,
            acc_log_w,
            expert_ema,
            router_ema,
            lifetime_seen,
            recent_class_counts,
            recent_class_buffer,
            recent_class_buffer_idx,
        ) = step_carry
        obs, tgt = inputs

        preds = jnp.stack(
            [
                mlp64.predict(mlp64_s, obs),
                mlp128.predict(mlp128_s, obs),
                mlp6464.predict(mlp6464_s, obs),
                upgd.predict(upgd_s, obs),
                dynamic.predict(dynamic_s, obs),
            ],
            axis=0,
        )
        weights = jax.nn.softmax(log_w)
        mlp_weights = jax.nn.softmax(mlp_log_w)
        acc_weights = jax.nn.softmax(acc_log_w)
        all_convex_pred = jnp.sum(weights[:, None] * preds, axis=0)
        mlp_convex_pred = jnp.sum(
            mlp_weights[:, None] * preds[: len(MLP_METHODS)],
            axis=0,
        )
        all_selector_idx = jnp.argmin(expert_ema)
        mlp_selector_idx = jnp.argmin(expert_ema[: len(MLP_METHODS)])
        router_preds = jnp.stack(
            [
                all_convex_pred,
                preds[all_selector_idx],
                mlp_convex_pred,
                preds[mlp_selector_idx],
            ],
            axis=0,
        )
        if args.router_policy == "convex":
            router_idx = jnp.array(0, dtype=jnp.int32)
        elif args.router_policy == "all_selector":
            router_idx = jnp.array(1, dtype=jnp.int32)
        elif args.router_policy == "mlp_convex":
            router_idx = jnp.array(2, dtype=jnp.int32)
        elif args.router_policy == "mlp_selector":
            router_idx = jnp.array(3, dtype=jnp.int32)
        elif args.router_policy == "guarded_convex":
            router_idx = jnp.where(
                router_ema[0] <= router_ema[2] + args.guard_tolerance,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(2, dtype=jnp.int32),
            )
        elif args.router_policy == "guarded_best_mlp":
            best_mlp_route = jnp.where(
                router_ema[2] <= router_ema[3],
                jnp.array(2, dtype=jnp.int32),
                jnp.array(3, dtype=jnp.int32),
            )
            best_mlp_ema = jnp.minimum(router_ema[2], router_ema[3])
            router_idx = jnp.where(
                router_ema[0] <= best_mlp_ema + args.guard_tolerance,
                jnp.array(0, dtype=jnp.int32),
                best_mlp_route,
            )
        else:
            router_idx = jnp.argmin(router_ema)

        lifetime_class_count = jnp.sum((lifetime_seen > 0.0).astype(jnp.float32))
        recent_class_count = jnp.sum((recent_class_counts > 0.0).astype(jnp.float32))
        lifetime_class_fraction = lifetime_class_count / jnp.asarray(
            max(n_heads, 1),
            dtype=jnp.float32,
        )
        recent_fraction_of_lifetime = recent_class_count / jnp.maximum(
            lifetime_class_count,
            1.0,
        )
        online_retention_hazard = (
            (
                lifetime_class_fraction
                >= args.online_retention_min_lifetime_class_fraction
            )
            & (
                recent_fraction_of_lifetime
                <= args.online_retention_max_recent_class_fraction
            )
        )
        base_router_idx = router_idx
        if args.online_retention_mse_guard and n_heads == N_CLASSES:
            router_idx = jnp.where(
                online_retention_hazard,
                jnp.array(ONLINE_RETENTION_GUARD_ROUTE_ID, dtype=jnp.int32),
                router_idx,
            )
            mixture_pred = jnp.where(
                online_retention_hazard,
                preds[EXPERT_NAMES.index("mlp_h64_64")],
                router_preds[base_router_idx],
            )
        else:
            mixture_pred = router_preds[router_idx]

        expert_losses = jnp.mean((preds - tgt[None, :]) ** 2, axis=1)
        router_losses = jnp.mean((router_preds - tgt[None, :]) ** 2, axis=1)
        target_class = jnp.argmax(tgt)
        mixture_loss = jnp.mean((mixture_pred - tgt) ** 2)
        expert_pred_classes = jnp.argmax(preds, axis=1)
        expert_accuracy_losses = (
            expert_pred_classes != target_class
        ).astype(jnp.float32)
        current_class = jax.nn.one_hot(target_class, n_heads, dtype=jnp.float32)
        old_recent_class = recent_class_buffer[recent_class_buffer_idx]
        new_recent_class_counts = recent_class_counts - old_recent_class + current_class
        new_recent_class_buffer = recent_class_buffer.at[
            recent_class_buffer_idx
        ].set(current_class)
        new_recent_class_buffer_idx = (
            recent_class_buffer_idx + jnp.array(1, dtype=jnp.int32)
        ) % jnp.array(recent_buffer_size, dtype=jnp.int32)
        new_lifetime_seen = jnp.maximum(lifetime_seen, current_class)

        new_log_w = discount * log_w - eta * expert_losses
        new_log_w = new_log_w - jnp.max(new_log_w)
        new_mlp_log_w = discount * mlp_log_w - eta * expert_losses[: len(MLP_METHODS)]
        new_mlp_log_w = new_mlp_log_w - jnp.max(new_mlp_log_w)
        new_acc_log_w = discount * acc_log_w - eta * expert_accuracy_losses
        new_acc_log_w = new_acc_log_w - jnp.max(new_acc_log_w)
        new_expert_ema = (1.0 - router_decay) * expert_ema + router_decay * expert_losses
        new_router_ema = (1.0 - router_decay) * router_ema + router_decay * router_losses

        mlp64_result = mlp64.update(mlp64_s, obs, tgt)
        mlp128_result = mlp128.update(mlp128_s, obs, tgt)
        mlp6464_result = mlp6464.update(mlp6464_s, obs, tgt)
        upgd_result = upgd.update(upgd_s, obs, tgt)
        dynamic_result = dynamic.update(dynamic_s, obs, tgt)

        pred_classes = jnp.concatenate(
            [
                jnp.asarray([jnp.argmax(mixture_pred)], dtype=jnp.float32),
                jnp.argmax(preds, axis=1).astype(jnp.float32),
            ]
        )
        metric = jnp.concatenate(
            [
                jnp.asarray([mixture_loss], dtype=jnp.float32),
                expert_losses.astype(jnp.float32),
                weights.astype(jnp.float32),
                mlp_weights.astype(jnp.float32),
                acc_weights.astype(jnp.float32),
                jnp.asarray(
                    [
                        router_idx.astype(jnp.float32),
                        all_selector_idx.astype(jnp.float32),
                        mlp_selector_idx.astype(jnp.float32),
                    ],
                    dtype=jnp.float32,
                ),
                pred_classes,
            ]
        )
        return (
            mlp64_result.state,
            mlp128_result.state,
            mlp6464_result.state,
            upgd_result.state,
            dynamic_result.state,
            new_log_w,
            new_mlp_log_w,
            new_acc_log_w,
            new_expert_ema,
            new_router_ema,
            new_lifetime_seen,
            new_recent_class_counts,
            new_recent_class_buffer,
            new_recent_class_buffer_idx,
        ), metric

    final_carry, metrics = jax.lax.scan(step_fn, carry, (observations, targets))
    metrics.block_until_ready()
    return final_carry, np.asarray(metrics)


def final_states_from_carry(
    learners: tuple[Any, Any, Any, Any, Any],
    carry: tuple[Any, ...],
) -> dict[str, tuple[Any, Any]]:
    """Return the public final-states mapping used by held-out evaluators."""
    return {
        "mlp_h64": (learners[0], carry[0]),
        "mlp_h128": (learners[1], carry[1]),
        "mlp_h64_64": (learners[2], carry[2]),
        "upgd_low_noise": (learners[3], carry[3]),
        "dynamic_sparse": (learners[4], carry[4]),
    }


def opmnist_block_indices(
    dataset_size: int,
    seed: int,
    block_idx: int,
    task_id: int,
    block_len: int,
    sample_with_replacement: bool,
    task_sampling: str,
) -> np.ndarray:
    """Return reproducible source indices for one OPMNIST task block."""
    if task_sampling == "random":
        rng = np.random.default_rng(seed + 2_000_003 + block_idx)
        replace = sample_with_replacement or block_len > dataset_size
        return rng.choice(dataset_size, size=block_len, replace=replace).astype(np.int32)

    pieces: list[np.ndarray] = []
    remaining = block_len
    epoch = 0
    while remaining > 0:
        rng = np.random.default_rng(seed + 1_000_003 + task_id * 9_973 + epoch)
        order = rng.permutation(dataset_size).astype(np.int32)
        take = min(remaining, dataset_size)
        pieces.append(order[:take])
        remaining -= take
        epoch += 1
    return np.concatenate(pieces).astype(np.int32)


def make_permuted_classification_chunk(
    dataset: ClassificationDataset,
    *,
    start_step: int,
    chunk_steps: int,
    seed: int,
    n_permutations: int,
    task_block_size: int,
    sample_with_replacement: bool,
    task_sampling: str,
    feature_orders: tuple[np.ndarray, ...],
) -> tuple[jax.Array, jax.Array, np.ndarray]:
    """Materialize only one contiguous OPMNIST chunk."""
    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    total = 0
    dataset_size = int(dataset.x_train.shape[0])
    while total < chunk_steps:
        global_step = start_step + total
        block_idx = global_step // task_block_size
        block_offset = global_step % task_block_size
        task_id = int(block_idx % n_permutations)
        block_len = task_block_size
        take = min(chunk_steps - total, task_block_size - block_offset)
        indices = opmnist_block_indices(
            dataset_size=dataset_size,
            seed=seed,
            block_idx=int(block_idx),
            task_id=task_id,
            block_len=block_len,
            sample_with_replacement=sample_with_replacement,
            task_sampling=task_sampling,
        )[block_offset : block_offset + take]
        obs_parts.append(dataset.x_train[indices][:, feature_orders[task_id]])
        label_parts.append(dataset.y_train[indices])
        total += take

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return jnp.asarray(observations), jnp.asarray(targets), labels


def evaluate_classifier_feature_orders(
    learner: Any,
    state: Any,
    dataset: ClassificationDataset,
    feature_orders: tuple[np.ndarray, ...],
    test_task_ids: list[int],
) -> dict[str, float]:
    """Evaluate one expert over selected held-out OPMNIST task views."""
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[dataset.y_test])
    labels = jnp.asarray(dataset.y_test.astype(np.int32))
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for task_id in test_task_ids:
        observations = jnp.asarray(
            dataset.x_test[:, feature_orders[task_id]].astype(np.float32)
        )
        preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        mse = jnp.mean((preds - targets) ** 2)
        accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        mse_values.append(float(mse))
        accuracy_values.append(float(accuracy))
    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def evaluate_mixture_classifier_feature_orders(
    final_states: dict[str, tuple[Any, Any]],
    weights: np.ndarray,
    dataset: ClassificationDataset,
    feature_orders: tuple[np.ndarray, ...],
    test_task_ids: list[int],
) -> dict[str, float]:
    """Evaluate final convex portfolio over selected held-out OPMNIST views."""
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[dataset.y_test])
    labels = jnp.asarray(dataset.y_test.astype(np.int32))
    weights_jax = jnp.asarray(weights.astype(np.float32))
    mse_values: list[float] = []
    accuracy_values: list[float] = []

    def predict(obs: jax.Array) -> jax.Array:
        preds = jnp.stack(
            [
                final_states[expert][0].predict(final_states[expert][1], obs)
                for expert in EXPERT_NAMES
            ],
            axis=0,
        )
        return jnp.sum(weights_jax[:, None] * preds, axis=0)

    for task_id in test_task_ids:
        observations = jnp.asarray(
            dataset.x_test[:, feature_orders[task_id]].astype(np.float32)
        )
        preds = jax.vmap(predict)(observations)
        mse = jnp.mean((preds - targets) ** 2)
        accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        mse_values.append(float(mse))
        accuracy_values.append(float(accuracy))
    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def opmnist_resume_path(seed: int, args: argparse.Namespace) -> Path:
    """Return the per-seed checkpoint path for a chunked OPMNIST run."""
    if args.opmnist_resume_path is not None:
        path = cast(Path, args.opmnist_resume_path)
        if args.n_seeds > 1:
            return path.with_name(f"{path.stem}_seed{seed}{path.suffix or '.pkl'}")
        return path
    output_dir = cast(Path, args.output_dir)
    result_prefix = cast(str, args.result_prefix)
    return output_dir / f"{result_prefix}_seed{seed}_opmnist_resume.pkl"


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for machine-readable progress files."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes by fsyncing then replacing a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, text.encode("utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL progress row, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_git_command(args: list[str]) -> str | None:
    """Return a git command's stdout, or None when git metadata is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def git_metadata() -> dict[str, Any]:
    """Return lightweight git metadata for reproducibility manifests."""
    status = run_git_command(["status", "--short"])
    changed_files = [
        line[3:] for line in status.splitlines() if len(line) >= 4
    ] if status is not None else None
    return {
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "commit_short": run_git_command(["rev-parse", "--short", "HEAD"]),
        "branch": run_git_command(["branch", "--show-current"]),
        "status_short": status,
        "dirty": bool(status) if status is not None else None,
        "changed_files": changed_files,
    }


def shell_join(argv: list[str]) -> str:
    """Return a shell-escaped command string."""
    import shlex

    return " ".join(shlex.quote(part) for part in argv)


def run_manifest(
    args: argparse.Namespace,
    benchmarks: Iterable[str],
) -> dict[str, Any]:
    """Return command/config/git metadata for reproducibility."""
    command_argv = [sys.executable, *sys.argv]
    return {
        "schema": "alberta.opmnist.run_manifest.v1",
        "created_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "command_argv": command_argv,
        "command": shell_join(command_argv),
        "config": config_dict(args, benchmarks),
        "git": git_metadata(),
    }


def opmnist_status_artifact_path(args: argparse.Namespace) -> Path | None:
    """Return the periodic OPMNIST status artifact path for a run."""
    if not hasattr(args, "output_dir") or not hasattr(args, "result_prefix"):
        return None
    output_dir = cast(Path, args.output_dir)
    result_prefix = cast(str, args.result_prefix)
    return output_dir / f"{result_prefix}_opmnist_status.json"


def format_seconds(seconds: float | None) -> str:
    """Format an ETA duration for status output."""
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds_i = max(0, int(round(seconds)))
    days, rem = divmod(seconds_i, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def opmnist_progress_status(
    *,
    completed_steps: int,
    target_steps: int,
    task_block_size: int,
    elapsed_s: float | None = None,
    recent_steps_per_second: float | None = None,
) -> dict[str, Any]:
    """Return resumable OPMNIST progress and ETA fields."""
    completed_steps = int(completed_steps)
    target_steps = int(target_steps)
    remaining_steps = max(target_steps - completed_steps, 0)
    overall_sps = (
        float(completed_steps) / elapsed_s
        if elapsed_s is not None and elapsed_s > 0.0
        else None
    )
    eta_sps = recent_steps_per_second or overall_sps
    eta_seconds = (
        float(remaining_steps) / eta_sps
        if eta_sps is not None and eta_sps > 0.0
        else None
    )
    completed_blocks = completed_steps // task_block_size
    target_blocks = target_steps // task_block_size
    return {
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "remaining_steps": remaining_steps,
        "completed_full_task_blocks": int(completed_blocks),
        "target_full_task_blocks": int(target_blocks),
        "remaining_full_task_blocks": int(max(target_blocks - completed_blocks, 0)),
        "progress_fraction": (
            float(completed_steps / target_steps) if target_steps > 0 else 0.0
        ),
        "elapsed_s": elapsed_s,
        "overall_steps_per_second": overall_sps,
        "recent_steps_per_second": recent_steps_per_second,
        "eta_seconds": eta_seconds,
        "eta_human": format_seconds(eta_seconds),
    }


def opmnist_checkpoint_sidecar_path(path: Path) -> Path:
    """Return the JSON sidecar path for a binary OPMNIST checkpoint."""
    return path.with_suffix(f"{path.suffix}.json" if path.suffix else ".json")


def opmnist_progress_log_path(path: Path) -> Path:
    """Return the append-only JSONL progress log path for a checkpoint."""
    return path.with_suffix(f"{path.suffix}.progress.jsonl" if path.suffix else ".progress.jsonl")


def save_opmnist_checkpoint(
    path: Path,
    *,
    completed_steps: int,
    carry: tuple[Any, ...],
    accumulator: PrequentialAccumulator,
    feature_orders: tuple[np.ndarray, ...],
    config: dict[str, Any],
    elapsed_s: float | None = None,
    progress_history: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a chunked OPMNIST checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    latest_progress = progress_history[-1] if progress_history else None
    payload = {
        "version": OPMNIST_CHECKPOINT_VERSION,
        "completed_steps": int(completed_steps),
        "carry": carry,
        "accumulator": accumulator,
        "feature_orders": feature_orders,
        "config": config,
        "elapsed_s": elapsed_s,
        "progress_history": progress_history or [],
        "updated_at_utc": utc_now_iso(),
    }
    atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    sidecar = opmnist_checkpoint_sidecar_path(path)
    atomic_write_json(
        sidecar,
        {
            "version": OPMNIST_CHECKPOINT_VERSION,
            "completed_steps": int(completed_steps),
            "elapsed_s": elapsed_s,
            "latest_progress": latest_progress,
            "config": config,
            "updated_at_utc": payload["updated_at_utc"],
        },
    )


def load_opmnist_checkpoint(path: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    """Load and validate a chunked OPMNIST checkpoint."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != OPMNIST_CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported OPMNIST checkpoint version in {path}")
    saved_config = payload.get("config", {})
    legacy_defaults: dict[str, Any] = {
        "step_size": 0.03,
        "perturbation_sigma": 1e-4,
        "perturbation_warmup_steps": 0,
        "perturbation_ramp_steps": 0,
        "dynamic_hidden_size": 64,
        "dynamic_utility_decay": 0.99,
        "dynamic_rewire_interval": 240,
        "dynamic_unit_replacement_rate": 0.05,
        "online_retention_mse_guard": True,
        "online_retention_min_lifetime_class_fraction": 0.7,
        "online_retention_max_recent_class_fraction": 0.5,
    }
    if saved_config.get("source_kind") == "openml_mnist_784":
        legacy_defaults.update(
            {
                "dataset_split": "openml_canonical_60000_10000",
                "dataset_n_train": 60_000,
                "dataset_n_test": 10_000,
                "dataset_is_full_mnist_split": True,
                "max_train_examples": None,
                "max_test_examples": None,
            }
        )
    for key, expected_value in expected_config.items():
        saved_value = saved_config.get(key, legacy_defaults.get(key))
        if saved_value != expected_value:
            raise RuntimeError(
                f"OPMNIST checkpoint config mismatch for {key}: "
                f"saved={saved_value!r}, current={expected_value!r}"
            )
    validate_opmnist_checkpoint_feature_orders(payload, expected_config, path)
    payload["carry"], migrated_fields = migrate_opmnist_checkpoint_carry(payload["carry"])
    if migrated_fields:
        payload["migrated_fields"] = migrated_fields
    return cast(dict[str, Any], payload)


def migrate_opmnist_checkpoint_carry(carry: tuple[Any, ...]) -> tuple[tuple[Any, ...], list[str]]:
    """Add missing current state fields to older OPMNIST checkpoint carries."""
    if len(carry) <= 3:
        return carry, []

    upgd_state = carry[3]
    if not hasattr(upgd_state, "replace"):
        return carry, []

    updates: dict[str, Any] = {}
    migrated_fields: list[str] = []
    if not hasattr(upgd_state, "unit_long_utilities") and hasattr(
        upgd_state,
        "unit_utilities",
    ):
        updates["unit_long_utilities"] = tuple(upgd_state.unit_utilities)
        migrated_fields.append("upgd.unit_long_utilities")
    if not hasattr(upgd_state, "unit_gradient_emas") and hasattr(
        upgd_state,
        "unit_utilities",
    ):
        updates["unit_gradient_emas"] = tuple(
            jnp.zeros_like(unit_utility) for unit_utility in upgd_state.unit_utilities
        )
        migrated_fields.append("upgd.unit_gradient_emas")
    if not hasattr(upgd_state, "loss_fast_ema"):
        updates["loss_fast_ema"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.loss_fast_ema")
    if not hasattr(upgd_state, "loss_slow_ema"):
        updates["loss_slow_ema"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.loss_slow_ema")
    if not hasattr(upgd_state, "previous_targets"):
        n_heads = len(upgd_state.head_params.weights)
        updates["previous_targets"] = jnp.zeros(n_heads, dtype=jnp.float32)
        migrated_fields.append("upgd.previous_targets")
    if not hasattr(upgd_state, "target_repeat_ema"):
        updates["target_repeat_ema"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.target_repeat_ema")
    if not hasattr(upgd_state, "meta_trunk_log_scale"):
        updates["meta_trunk_log_scale"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.meta_trunk_log_scale")
    if not hasattr(upgd_state, "meta_head_weight_log_scale"):
        updates["meta_head_weight_log_scale"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.meta_head_weight_log_scale")
    if not hasattr(upgd_state, "meta_head_bias_log_scale"):
        updates["meta_head_bias_log_scale"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.meta_head_bias_log_scale")
    if not hasattr(upgd_state, "meta_repetition_log_scale"):
        updates["meta_repetition_log_scale"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.meta_repetition_log_scale")
    if not hasattr(upgd_state, "adaptive_kappa_log_scale"):
        updates["adaptive_kappa_log_scale"] = jnp.array(0.0, dtype=jnp.float32)
        migrated_fields.append("upgd.adaptive_kappa_log_scale")
    if not hasattr(upgd_state, "previous_trunk_weight_grads"):
        updates["previous_trunk_weight_grads"] = tuple(
            jnp.zeros_like(weight) for weight in upgd_state.trunk_params.weights
        )
        migrated_fields.append("upgd.previous_trunk_weight_grads")
    if not hasattr(upgd_state, "previous_trunk_bias_grads"):
        updates["previous_trunk_bias_grads"] = tuple(
            jnp.zeros_like(bias) for bias in upgd_state.trunk_params.biases
        )
        migrated_fields.append("upgd.previous_trunk_bias_grads")
    if not hasattr(upgd_state, "previous_head_weight_grads"):
        updates["previous_head_weight_grads"] = tuple(
            jnp.zeros_like(weight) for weight in upgd_state.head_params.weights
        )
        migrated_fields.append("upgd.previous_head_weight_grads")
    if not hasattr(upgd_state, "previous_head_bias_grads"):
        updates["previous_head_bias_grads"] = tuple(
            jnp.zeros_like(bias) for bias in upgd_state.head_params.biases
        )
        migrated_fields.append("upgd.previous_head_bias_grads")
    if not updates:
        return carry, []

    migrated_upgd_state = upgd_state.replace(**updates)
    migrated_carry = tuple(
        migrated_upgd_state if idx == 3 else value for idx, value in enumerate(carry)
    )
    return migrated_carry, migrated_fields


def validate_opmnist_checkpoint_feature_orders(
    payload: dict[str, Any],
    expected_config: dict[str, Any],
    path: Path,
) -> None:
    """Validate checkpointed OPMNIST pixel orders against deterministic config."""
    required = ("stream_seed", "feature_dim", "n_permutations", "include_identity_permutation")
    if any(key not in expected_config for key in required):
        return

    feature_orders = payload.get("feature_orders")
    if not isinstance(feature_orders, tuple):
        raise RuntimeError(f"OPMNIST checkpoint feature_orders missing in {path}")

    feature_dim = int(expected_config["feature_dim"])
    n_permutations = int(expected_config["n_permutations"])
    expected_orders = make_feature_orders(
        seed=int(expected_config["stream_seed"]),
        feature_dim=feature_dim,
        n_permutations=n_permutations,
        include_identity_permutation=bool(expected_config["include_identity_permutation"]),
    )
    if len(feature_orders) != n_permutations:
        raise RuntimeError(
            f"OPMNIST checkpoint feature_orders length mismatch in {path}: "
            f"saved={len(feature_orders)}, expected={n_permutations}"
        )

    expected_set = np.arange(feature_dim, dtype=np.int32)
    paired_orders = zip(feature_orders, expected_orders, strict=True)
    for idx, (saved_order, expected_order) in enumerate(paired_orders):
        saved_array = np.asarray(saved_order, dtype=np.int32)
        if saved_array.shape != (feature_dim,):
            raise RuntimeError(
                f"OPMNIST checkpoint feature order {idx} has shape "
                f"{saved_array.shape}, expected {(feature_dim,)}"
            )
        if not np.array_equal(np.sort(saved_array), expected_set):
            raise RuntimeError(f"OPMNIST checkpoint feature order {idx} is not a permutation")
        if not np.array_equal(saved_array, expected_order):
            raise RuntimeError(
                f"OPMNIST checkpoint feature order {idx} does not match "
                "the deterministic stream seed"
            )


def opmnist_status_from_checkpoint(
    path: Path,
    *,
    target_steps: int = DOHARE_OPMNIST_TOTAL_STEPS,
    task_block_size: int = DOHARE_OPMNIST_TASK_BLOCK_SIZE,
) -> dict[str, Any]:
    """Read lightweight OPMNIST resume status from a checkpoint or sidecar."""
    sidecar = opmnist_checkpoint_sidecar_path(path)
    payload: dict[str, Any]
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    latest = payload.get("latest_progress") or {}
    status = opmnist_progress_status(
        completed_steps=int(payload.get("completed_steps", 0) or 0),
        target_steps=target_steps,
        task_block_size=task_block_size,
        elapsed_s=cast(float | None, payload.get("elapsed_s")),
        recent_steps_per_second=cast(float | None, latest.get("steps_per_second")),
    )
    status.update(
        {
            "checkpoint_path": str(path),
            "sidecar_path": str(sidecar),
            "updated_at_utc": payload.get("updated_at_utc"),
            "latest_progress": latest,
        }
    )
    return status


def opmnist_eta_from_results(
    results_path: Path,
    *,
    target_steps: int = DOHARE_OPMNIST_TOTAL_STEPS,
    task_block_size: int = DOHARE_OPMNIST_TASK_BLOCK_SIZE,
) -> dict[str, Any]:
    """Estimate full-scale ETA from a completed result JSON."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    meta = results.get("datasets", {}).get("permuted_mnist_like", {})
    status = opmnist_progress_status(
        completed_steps=int(meta.get("steps", 0) or 0),
        target_steps=target_steps,
        task_block_size=task_block_size,
        elapsed_s=cast(float | None, results.get("wall_clock_s")),
    )
    status.update(
        {
            "results_path": str(results_path),
            "source_wall_clock_s": results.get("wall_clock_s"),
            "source_status": results.get("status", {}),
        }
    )
    return status


def run_streaming_permuted_mnist_like_seed(
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run OPMNIST from reproducible chunks without materializing the full stream."""
    dataset = load_mnist_like_source(args, seed)
    stream_seed = seed + 10_000
    feature_orders = make_feature_orders(
        seed=stream_seed,
        feature_dim=int(dataset.x_train.shape[1]),
        n_permutations=args.n_permutations,
        include_identity_permutation=args.include_identity_permutation,
    )
    observed_task_ids = opmnist_task_ids_for_steps(
        steps=args.steps,
        n_permutations=args.n_permutations,
        task_block_size=args.task_block_size,
    )
    test_task_ids = opmnist_test_task_ids(
        observed_task_ids=observed_task_ids,
        n_permutations=args.n_permutations,
        max_test_permutation_views=args.max_test_permutation_views,
        evaluate_all_permutation_views=args.evaluate_all_permutation_views,
    )
    checkpoint_path = opmnist_resume_path(seed, args) if args.opmnist_resume else None
    status_artifact_path = opmnist_status_artifact_path(args)
    meta = dict(dataset.metadata)
    protocol_meta = opmnist_protocol_metadata(
        dataset=dataset,
        steps=args.steps,
        seed=stream_seed,
        n_permutations=args.n_permutations,
        task_block_size=args.task_block_size,
        sample_with_replacement=args.sample_with_replacement,
        task_sampling=args.task_sampling,
        include_identity_permutation=args.include_identity_permutation,
        max_test_permutation_views=args.max_test_permutation_views,
        evaluate_all_permutation_views=args.evaluate_all_permutation_views,
        observed_task_ids=observed_task_ids,
        test_task_ids=test_task_ids,
        streaming_runner=True,
        chunk_size=args.opmnist_chunk_size,
        resume_checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )
    meta.update(protocol_meta)

    learners, carry = make_portfolio_components_and_carry(
        feature_dim=int(dataset.x_train.shape[1]),
        n_heads=N_CLASSES,
        key=jr.key(seed),
        args=args,
    )
    accumulator = init_prequential_accumulator()
    completed_steps = 0
    checkpoint_loaded = False
    elapsed_s = 0.0
    progress_history: list[dict[str, Any]] = []
    resume_config = {
        "seed": seed,
        "stream_seed": stream_seed,
        "source_kind": dataset.metadata.get("source_kind"),
        "dataset_split": dataset.metadata.get("split"),
        "dataset_n_train": dataset.metadata.get("n_train"),
        "dataset_n_test": dataset.metadata.get("n_test"),
        "dataset_is_full_mnist_split": dataset.metadata.get("is_full_mnist_split"),
        "max_train_examples": dataset.metadata.get("max_train_examples"),
        "max_test_examples": dataset.metadata.get("max_test_examples"),
        "feature_dim": int(dataset.x_train.shape[1]),
        "n_heads": N_CLASSES,
        "n_permutations": args.n_permutations,
        "task_block_size": args.task_block_size,
        "sample_with_replacement": args.sample_with_replacement,
        "task_sampling": args.task_sampling,
        "include_identity_permutation": args.include_identity_permutation,
        "final_window": args.final_window,
        "step_size": args.step_size,
        "sparsity": args.sparsity,
        "perturbation_sigma": args.perturbation_sigma,
        "perturbation_warmup_steps": args.perturbation_warmup_steps,
        "perturbation_ramp_steps": args.perturbation_ramp_steps,
        "dynamic_hidden_size": args.dynamic_hidden_size,
        "dynamic_utility_decay": args.dynamic_utility_decay,
        "dynamic_rewire_interval": args.dynamic_rewire_interval,
        "dynamic_unit_replacement_rate": args.dynamic_unit_replacement_rate,
        "router_policy": args.router_policy,
        "hedge_eta": args.hedge_eta,
        "hedge_discount": args.hedge_discount,
        "router_decay": args.router_decay,
        "guard_tolerance": args.guard_tolerance,
        "online_retention_mse_guard": args.online_retention_mse_guard,
        "online_retention_min_lifetime_class_fraction": (
            args.online_retention_min_lifetime_class_fraction
        ),
        "online_retention_max_recent_class_fraction": (
            args.online_retention_max_recent_class_fraction
        ),
    }
    if checkpoint_path and checkpoint_path.exists() and not args.opmnist_force_restart:
        payload = load_opmnist_checkpoint(checkpoint_path, resume_config)
        carry = payload["carry"]
        accumulator = payload["accumulator"]
        feature_orders = payload["feature_orders"]
        completed_steps = int(payload["completed_steps"])
        elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
        progress_history = list(payload.get("progress_history", []))
        checkpoint_loaded = True
        print(
            f"benchmark=permuted_mnist_like seed={seed}: resumed "
            f"{completed_steps}/{args.steps} steps from {checkpoint_path}"
        )
        if payload.get("migrated_fields"):
            print(
                "benchmark=permuted_mnist_like seed="
                f"{seed}: migrated checkpoint fields "
                f"{', '.join(payload['migrated_fields'])}"
            )
    elif checkpoint_path and args.opmnist_force_restart and checkpoint_path.exists():
        checkpoint_path.unlink()
        sidecar_path = opmnist_checkpoint_sidecar_path(checkpoint_path)
        if sidecar_path.exists():
            sidecar_path.unlink()
        progress_log_path = opmnist_progress_log_path(checkpoint_path)
        if progress_log_path.exists():
            progress_log_path.unlink()

    if status_artifact_path:
        atomic_write_json(
            status_artifact_path,
            {
                "schema": "alberta.opmnist.status.v1",
                "updated_at_utc": utc_now_iso(),
                "seed": int(seed),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "manifest_path": str(getattr(args, "run_manifest_path", "")) or None,
                "requested_steps": int(args.steps),
                "dohare_target_steps": DOHARE_OPMNIST_TOTAL_STEPS,
                "completed_steps": int(completed_steps),
                "status": opmnist_progress_status(
                    completed_steps=completed_steps,
                    target_steps=DOHARE_OPMNIST_TOTAL_STEPS,
                    task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
                    elapsed_s=elapsed_s,
                ),
                "latest_progress": progress_history[-1] if progress_history else None,
                "protocol": meta,
            },
        )

    while completed_steps < args.steps:
        chunk_steps = min(args.opmnist_chunk_size, args.steps - completed_steps)
        chunk_t0 = time.time()
        observations, targets, labels = make_permuted_classification_chunk(
            dataset=dataset,
            start_step=completed_steps,
            chunk_steps=chunk_steps,
            seed=stream_seed,
            n_permutations=args.n_permutations,
            task_block_size=args.task_block_size,
            sample_with_replacement=args.sample_with_replacement,
            task_sampling=args.task_sampling,
            feature_orders=feature_orders,
        )
        carry, metrics = scan_portfolio_chunk(
            learners=learners,
            carry=carry,
            observations=observations,
            targets=targets,
            args=args,
        )
        accumulator = update_prequential_accumulator(
            accumulator=accumulator,
            metrics=metrics,
            labels=labels,
            final_window=args.final_window,
        )
        completed_steps += chunk_steps
        chunk_elapsed_s = time.time() - chunk_t0
        elapsed_s += chunk_elapsed_s
        progress_row = {
            "timestamp_utc": utc_now_iso(),
            "seed": int(seed),
            "completed_steps": int(completed_steps),
            "requested_steps": int(args.steps),
            "dohare_target_steps": DOHARE_OPMNIST_TOTAL_STEPS,
            "chunk_steps": int(chunk_steps),
            "chunk_elapsed_s": float(chunk_elapsed_s),
            "elapsed_s": float(elapsed_s),
            "steps_per_second": float(chunk_steps / max(chunk_elapsed_s, 1e-12)),
            "completed_full_task_blocks": int(
                completed_steps // args.task_block_size
            ),
            "eta_to_requested_s": opmnist_progress_status(
                completed_steps=completed_steps,
                target_steps=args.steps,
                task_block_size=args.task_block_size,
                elapsed_s=elapsed_s,
                recent_steps_per_second=float(chunk_steps / max(chunk_elapsed_s, 1e-12)),
            )["eta_seconds"],
            "eta_to_dohare_800_s": opmnist_progress_status(
                completed_steps=completed_steps,
                target_steps=DOHARE_OPMNIST_TOTAL_STEPS,
                task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
                elapsed_s=elapsed_s,
                recent_steps_per_second=float(chunk_steps / max(chunk_elapsed_s, 1e-12)),
            )["eta_seconds"],
        }
        progress_history.append(progress_row)
        if checkpoint_path:
            save_opmnist_checkpoint(
                checkpoint_path,
                completed_steps=completed_steps,
                carry=carry,
                accumulator=accumulator,
                feature_orders=feature_orders,
                config=resume_config,
                elapsed_s=elapsed_s,
                progress_history=progress_history,
            )
            append_jsonl(opmnist_progress_log_path(checkpoint_path), progress_row)
        if status_artifact_path:
            atomic_write_json(
                status_artifact_path,
                {
                    "schema": "alberta.opmnist.status.v1",
                    "updated_at_utc": utc_now_iso(),
                    "seed": int(seed),
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                    "manifest_path": str(getattr(args, "run_manifest_path", "")) or None,
                    "requested_steps": int(args.steps),
                    "dohare_target_steps": DOHARE_OPMNIST_TOTAL_STEPS,
                    "completed_steps": int(completed_steps),
                    "status": opmnist_progress_status(
                        completed_steps=completed_steps,
                        target_steps=DOHARE_OPMNIST_TOTAL_STEPS,
                        task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
                        elapsed_s=elapsed_s,
                        recent_steps_per_second=float(
                            chunk_steps / max(chunk_elapsed_s, 1e-12)
                        ),
                    ),
                    "latest_progress": progress_row,
                    "protocol": meta,
                },
            )
        completed_blocks = completed_steps // args.task_block_size
        print(
            f"benchmark=permuted_mnist_like seed={seed}: "
            f"streamed {completed_steps}/{args.steps} steps "
            f"({completed_blocks} full task blocks, "
            f"{progress_row['steps_per_second']:.1f} steps/s, "
            f"800-task ETA {format_seconds(progress_row['eta_to_dohare_800_s'])})"
        )

    methods = summarize_prequential_accumulator(accumulator)
    final_states = final_states_from_carry(learners, carry)
    if accumulator.final_metric_row is None:
        raise RuntimeError("streaming OPMNIST run produced no final metric row")
    final_metric_tail = accumulator.final_metric_row[None, :]
    tracking_weights = final_deployment_tracking_weights(final_metric_tail, args)
    final_weights, retention_signal = deployment_weights(
        tracking_weights=tracking_weights,
        labels=None,
        n_heads=N_CLASSES,
        final_window=args.final_window,
        args=args,
    )

    for expert in EXPERT_NAMES:
        learner, state = final_states[expert]
        methods[expert].update(
            evaluate_classifier_feature_orders(
                learner=learner,
                state=state,
                dataset=dataset,
                feature_orders=feature_orders,
                test_task_ids=test_task_ids,
            )
        )
    methods["mixture"].update(
        evaluate_mixture_classifier_feature_orders(
            final_states=final_states,
            weights=final_weights,
            dataset=dataset,
            feature_orders=feature_orders,
            test_task_ids=test_task_ids,
        )
    )
    for idx, expert in enumerate(EXPERT_NAMES):
        methods["mixture"][f"deployment_{expert}_weight"] = float(final_weights[idx])
    methods["mixture"]["retention_hazard"] = float(bool(retention_signal["retention_hazard"]))

    meta["checkpoint_loaded"] = checkpoint_loaded
    meta["opmnist_elapsed_s"] = elapsed_s
    meta["opmnist_overall_steps_per_second"] = (
        float(completed_steps / elapsed_s) if elapsed_s > 0.0 else None
    )
    meta["opmnist_last_chunk_steps_per_second"] = (
        float(progress_history[-1]["steps_per_second"]) if progress_history else None
    )
    meta["opmnist_eta_to_800_tasks_s"] = opmnist_progress_status(
        completed_steps=completed_steps,
        target_steps=DOHARE_OPMNIST_TOTAL_STEPS,
        task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        elapsed_s=elapsed_s,
        recent_steps_per_second=(
            float(progress_history[-1]["steps_per_second"]) if progress_history else None
        ),
    )["eta_seconds"]
    record = {
        "dataset_name": "permuted_mnist_like",
        "seed": int(seed),
        "dataset": meta,
        "methods": methods,
        "retention_router": retention_signal,
    }
    return record, meta


def run_permuted_mnist_like_seed(
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one compact Online Permuted MNIST-style seed."""
    if args.opmnist_streaming or args.mnist_published_scale:
        return run_streaming_permuted_mnist_like_seed(seed, args)

    dataset = load_mnist_like_source(args, seed)
    stream = make_permuted_classification_stream(
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
    final_states, metrics = run_portfolio_stream(
        observations=stream.observations,
        targets=stream.targets,
        key=jr.key(seed),
        args=args,
    )
    labels_np = np.asarray(stream.labels)
    methods = summarize_prequential(metrics, args.final_window, labels_np)
    tracking_weights = final_deployment_tracking_weights(metrics, args)
    final_weights, retention_signal = deployment_weights(
        tracking_weights=tracking_weights,
        labels=labels_np,
        n_heads=N_CLASSES,
        final_window=args.final_window,
        args=args,
    )

    for expert in EXPERT_NAMES:
        learner, state = final_states[expert]
        methods[expert].update(
            evaluate_classifier_views(
                learner=learner,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
            )
        )
    methods["mixture"].update(
        evaluate_mixture_classifier_views(
            final_states=final_states,
            weights=final_weights,
            test_views=stream.test_views,
            y_test=stream.test_labels,
        )
    )
    for idx, expert in enumerate(EXPERT_NAMES):
        methods["mixture"][f"deployment_{expert}_weight"] = float(final_weights[idx])
    methods["mixture"]["retention_hazard"] = float(
        bool(retention_signal["retention_hazard"])
    )

    meta = dict(dataset.metadata)
    meta.update(stream.metadata)
    record = {
        "dataset_name": "permuted_mnist_like",
        "seed": int(seed),
        "dataset": meta,
        "methods": methods,
        "retention_router": retention_signal,
    }
    return record, meta


def run_slowly_changing_regression_seed(
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one Slowly-Changing Regression-style seed."""
    observations, targets, meta = make_slowly_changing_regression_stream(
        steps=args.steps,
        seed=seed + 20_000,
        m_bits=args.regression_bits,
        slow_bits=args.regression_slow_bits,
        flip_interval=args.regression_flip_interval,
        target_hidden=args.regression_target_hidden,
        beta=args.regression_beta,
        noise_std=args.regression_noise_std,
    )
    _, metrics = run_portfolio_stream(
        observations=observations,
        targets=targets,
        key=jr.key(seed),
        args=args,
    )
    methods = summarize_prequential(metrics, args.final_window, labels=None)
    record = {
        "dataset_name": "slowly_changing_regression",
        "seed": int(seed),
        "dataset": meta,
        "methods": methods,
    }
    return record, meta


def run_one_seed(
    benchmark: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispatch one benchmark/seed pair."""
    if benchmark == "permuted_mnist_like":
        return run_permuted_mnist_like_seed(seed, args)
    if benchmark == "slowly_changing_regression":
        return run_slowly_changing_regression_seed(seed, args)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format an aggregate cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def comparison_line(dataset_agg: dict[str, Any], metric: str) -> str | None:
    """Return one human-readable mixture-vs-MLP comparison."""
    if metric not in dataset_agg["comparisons"]:
        return None
    row = dataset_agg["comparisons"][metric]["mixture_vs_best_mlp"]
    return (
        f"`{metric}` portfolio-vs-best-MLP diff: "
        f"{row['paired_diff_mean_positive_favors_mixture']:+.4f} +/- "
        f"{row['paired_diff_stderr']:.4f}; wins/losses/ties "
        f"{row['wins_for_mixture']}/{row['wins_for_baseline']}/{row['ties']}."
    )


def benchmark_status(results: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the portfolio clears the fair-MLP comparator."""
    rows: dict[str, Any] = {}
    all_primary_nonnegative = True
    for benchmark, dataset_agg in results["aggregate"].items():
        checks: dict[str, Any] = {}
        for metric in ("final_window_mse", "test_accuracy"):
            if metric not in dataset_agg["comparisons"]:
                continue
            row = dataset_agg["comparisons"][metric]["mixture_vs_best_mlp"]
            diff = float(row["paired_diff_mean_positive_favors_mixture"])
            checks[metric] = {
                "paired_diff_mean_positive_favors_portfolio": diff,
                "wins_for_portfolio": int(row["wins_for_mixture"]),
                "wins_for_best_mlp": int(row["wins_for_baseline"]),
                "ties": int(row["ties"]),
            }
            all_primary_nonnegative = all_primary_nonnegative and diff >= 0.0
        rows[benchmark] = checks
    datasets = results.get("datasets", {})
    permuted_meta = datasets.get("permuted_mnist_like", {})
    scr_meta = datasets.get("slowly_changing_regression", {})
    true_mnist_source = bool(permuted_meta.get("is_true_mnist", False))
    full_mnist_source = bool(permuted_meta.get("is_full_mnist_split", False))
    full_mnist_blocks = bool(permuted_meta.get("full_mnist_task_blocks", False))
    no_task_id = not bool(permuted_meta.get("task_id_provided_to_learner", True))
    single_pass_task_order = bool(permuted_meta.get("single_pass_examples_within_task", False))
    random_permutations = bool(permuted_meta.get("permutations_are_random_pixel_orders", False))
    opmnist_core_protocol = bool(
        permuted_meta.get("matches_dohare_opmnist_core_protocol", False)
    )
    opmnist_published_task_count = bool(
        permuted_meta.get("matches_dohare_opmnist_published_task_count", False)
    )
    prediction_before_update = bool(
        permuted_meta.get("prediction_before_update_every_step", False)
    )
    all_experts_update = bool(permuted_meta.get("all_experts_update_every_step", False))
    completed_task_blocks = int(permuted_meta.get("completed_full_task_blocks", 0) or 0)
    completed_published_blocks = int(
        permuted_meta.get("opmnist_completed_full_60000_task_blocks", 0) or 0
    )
    dohare_scr_config = bool(scr_meta.get("matches_dohare_public_config", False))
    scr_steps = int(scr_meta.get("steps", 0) or 0)
    scr_no_task_id = not bool(scr_meta.get("task_id_provided_to_learner", True))
    scr_online_stream_only = bool(scr_meta.get("uses_online_stream_only", False))
    scr_fixed_target = bool(scr_meta.get("uses_fixed_target_network", False))
    scr_published_steps = scr_steps >= DOHARE_SCR_MIN_PUBLISHED_STEPS
    scr_public_protocol = bool(
        dohare_scr_config
        and scr_published_steps
        and scr_no_task_id
        and scr_online_stream_only
        and scr_fixed_target
    )
    has_permuted_mnist = "permuted_mnist_like" in datasets
    has_scr = "slowly_changing_regression" in datasets
    published_scale_opmnist = bool(
        has_permuted_mnist
        and true_mnist_source
        and full_mnist_source
        and full_mnist_blocks
        and no_task_id
        and single_pass_task_order
        and random_permutations
        and prediction_before_update
        and all_experts_update
        and opmnist_core_protocol
        and opmnist_published_task_count
    )
    published_scale_scr = bool(
        has_scr and scr_public_protocol and bool(all_primary_nonnegative)
    )
    return {
        "all_primary_nonnegative_vs_best_mlp": bool(all_primary_nonnegative),
        "uses_true_mnist": true_mnist_source,
        "uses_true_openml_mnist": bool(
            permuted_meta.get("source_kind") == "openml_mnist_784"
        ),
        "uses_torchvision_mnist": bool(
            permuted_meta.get("source_kind") == "torchvision_mnist"
        ),
        "uses_full_mnist_split": full_mnist_source,
        "uses_full_openml_mnist_split": bool(
            permuted_meta.get("source_kind") == "openml_mnist_784" and full_mnist_source
        ),
        "uses_full_mnist_task_blocks": full_mnist_blocks,
        "task_id_provided_to_learner": not no_task_id,
        "single_pass_examples_within_task": single_pass_task_order,
        "uses_random_pixel_permutations_for_all_tasks": random_permutations,
        "matches_dohare_opmnist_core_protocol": opmnist_core_protocol,
        "matches_dohare_opmnist_published_task_count": opmnist_published_task_count,
        "opmnist_steps": int(permuted_meta.get("steps", 0) or 0),
        "opmnist_n_permutations": int(permuted_meta.get("n_permutations", 0) or 0),
        "opmnist_completed_full_task_blocks": completed_task_blocks,
        "opmnist_completed_full_60000_task_blocks": completed_published_blocks,
        "opmnist_streaming_runner": bool(permuted_meta.get("streaming_runner", False)),
        "opmnist_resumable_runner": bool(permuted_meta.get("resumable_runner", False)),
        "prediction_before_update_every_step": prediction_before_update,
        "all_experts_update_every_step": all_experts_update,
        "uses_dohare_public_scr_config": dohare_scr_config,
        "scr_steps": scr_steps,
        "scr_min_published_steps": DOHARE_SCR_MIN_PUBLISHED_STEPS,
        "scr_meets_published_step_count": bool(scr_published_steps),
        "scr_task_id_provided_to_learner": not scr_no_task_id,
        "scr_uses_online_stream_only": scr_online_stream_only,
        "scr_uses_fixed_target_network": scr_fixed_target,
        "matches_dohare_public_scr_protocol": scr_public_protocol,
        "published_scale_scr_claim_supported": published_scale_scr,
        "published_scale_external_claim_supported": published_scale_opmnist
        and bool(all_primary_nonnegative),
        "checks": rows,
    }


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary."""
    cfg = results["config"]
    status = results["status"]
    lines = [
        "# Step 2 Published-Style External Stressors",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} online steps, "
            f"final window {cfg['final_window']}, benchmarks="
            f"{', '.join(cfg['benchmarks'])}."
        ),
        "",
        "The learner side is the existing Step 2 strict prediction-space portfolio "
        "over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired "
        "differences favor the portfolio; for MSE the value is best MLP minus "
        "portfolio, and for accuracy it is portfolio minus best MLP.",
        "",
        "Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and "
        "Slowly-Changing Regression. This runner is compact unless real OpenML "
        "MNIST is explicitly enabled.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
    ]

    for dataset, dataset_agg in results["aggregate"].items():
        meta = results["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"Source/protocol: `{meta.get('source_kind', meta.get('protocol'))}`.",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHOD_NAMES:
            row = dataset_agg[method]
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} |"
            )
        lines.append("")
        for metric in ("final_window_mse", "test_accuracy"):
            line = comparison_line(dataset_agg, metric)
            if line is not None:
                lines.append(line)
        lines.extend(["", "Limits:", ""])
        limits = meta.get("limitations") or meta.get("published_protocol_delta")
        if limits:
            lines.append(f"- {limits}")
        if meta.get("published_protocol_delta") and meta.get("limitations"):
            lines.append(f"- {meta['published_protocol_delta']}")
        lines.append("")

    lines.extend(
        [
            "## Assessment",
            "",
            (
                "Primary comparator status: "
                f"`all_primary_nonnegative_vs_best_mlp="
                f"{status['all_primary_nonnegative_vs_best_mlp']}`."
            ),
            (
                "Published-scale status: "
                f"`published_scale_external_claim_supported="
                f"{status['published_scale_external_claim_supported']}` "
                f"(true MNIST={status['uses_true_mnist']}, "
                f"OpenML MNIST={status['uses_true_openml_mnist']}, "
                f"torchvision MNIST={status['uses_torchvision_mnist']}, "
                f"full MNIST split={status['uses_full_mnist_split']}, "
                f"full MNIST task blocks={status['uses_full_mnist_task_blocks']}, "
                f"single-pass task order={status['single_pass_examples_within_task']}, "
                f"random permutations={status['uses_random_pixel_permutations_for_all_tasks']}, "
                f"no task id={not status['task_id_provided_to_learner']}, "
                f"prediction-before-update={status['prediction_before_update_every_step']}, "
                f"all experts update={status['all_experts_update_every_step']}, "
                f"OPMNIST tasks={status['opmnist_n_permutations']}, "
                f"full blocks={status['opmnist_completed_full_task_blocks']}, "
                f"60k blocks={status['opmnist_completed_full_60000_task_blocks']}, "
                f"OPMNIST steps={status['opmnist_steps']}, "
                f"SCR public config={status['uses_dohare_public_scr_config']}, "
                f"SCR steps={status['scr_steps']}/"
                f"{status['scr_min_published_steps']}, "
                f"SCR protocol="
                f"{status['matches_dohare_public_scr_protocol']})."
            ),
            (
                "Published-scale SCR status: "
                f"`published_scale_scr_claim_supported="
                f"{status['published_scale_scr_claim_supported']}`."
            ),
            "",
            "This should be reported as published-scale external evidence only "
            "when `published_scale_external_claim_supported=true`. With the "
            "default sklearn-digits fallback or shorter SCR settings, the result "
            "narrows the gap but is not a full published-scale reproduction.",
            "",
            "## References",
            "",
        ]
    )
    for ref in results["references"]:
        lines.append(f"- [{ref['name']}]({ref['url']}): {ref['relevance']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines))


def config_dict(args: argparse.Namespace, benchmarks: Iterable[str]) -> dict[str, Any]:
    """Return JSON-serializable command config."""
    return {
        "benchmarks": list(benchmarks),
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "final_window": args.final_window,
        "mnist_source": args.mnist_source,
        "allow_openml_download": args.allow_openml_download,
        "allow_torchvision_download": args.allow_torchvision_download,
        "mnist_split": args.mnist_split,
        "openml_data_home": str(args.openml_data_home)
        if args.openml_data_home is not None
        else None,
        "torchvision_data_home": str(args.torchvision_data_home)
        if args.torchvision_data_home is not None
        else None,
        "openml_n_retries": args.openml_n_retries,
        "openml_retry_delay": args.openml_retry_delay,
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
        "mnist_published_scale": args.mnist_published_scale,
        "opmnist_streaming": args.opmnist_streaming,
        "opmnist_chunk_size": args.opmnist_chunk_size,
        "opmnist_resume": args.opmnist_resume,
        "opmnist_resume_path": str(args.opmnist_resume_path)
        if args.opmnist_resume_path is not None
        else None,
        "opmnist_force_restart": args.opmnist_force_restart,
        "opmnist_status_target_steps": args.opmnist_status_target_steps,
        "opmnist_status_output": str(args.opmnist_status_output)
        if args.opmnist_status_output is not None
        else None,
        "scr_preset": args.scr_preset,
        "long_scr": args.long_scr,
        "regression_bits": args.regression_bits,
        "regression_slow_bits": args.regression_slow_bits,
        "regression_flip_interval": args.regression_flip_interval,
        "regression_target_hidden": args.regression_target_hidden,
        "regression_beta": args.regression_beta,
        "regression_noise_std": args.regression_noise_std,
        "expert_names": list(EXPERT_NAMES),
        "mlp_comparator_methods": list(MLP_METHODS),
        "step_size": args.step_size,
        "sparsity": args.sparsity,
        "perturbation_sigma": args.perturbation_sigma,
        "perturbation_warmup_steps": args.perturbation_warmup_steps,
        "perturbation_ramp_steps": args.perturbation_ramp_steps,
        "dynamic_hidden_size": args.dynamic_hidden_size,
        "dynamic_utility_decay": args.dynamic_utility_decay,
        "dynamic_rewire_interval": args.dynamic_rewire_interval,
        "dynamic_unit_replacement_rate": args.dynamic_unit_replacement_rate,
        "hedge_eta": args.hedge_eta,
        "hedge_discount": args.hedge_discount,
        "router_policy": args.router_policy,
        "router_decay": args.router_decay,
        "guard_tolerance": args.guard_tolerance,
        "digits_deployment_objective": args.digits_deployment_objective,
        "online_retention_mse_guard": args.online_retention_mse_guard,
        "online_retention_min_lifetime_class_fraction": (
            args.online_retention_min_lifetime_class_fraction
        ),
        "online_retention_max_recent_class_fraction": (
            args.online_retention_max_recent_class_fraction
        ),
        "retention_router": args.retention_router,
        "retention_upgd_deployment_weight": args.retention_upgd_deployment_weight,
        "retention_min_lifetime_class_fraction": args.retention_min_lifetime_class_fraction,
        "retention_max_recent_class_fraction": args.retention_max_recent_class_fraction,
        "output_dir": str(args.output_dir),
        "result_prefix": args.result_prefix,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument(
        "--mnist-source",
        choices=(
            "auto",
            "openml",
            "torchvision",
            "sklearn_digits_28x28",
            "sklearn_digits_8x8",
        ),
        default="auto",
    )
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument(
        "--allow-torchvision-download",
        action="store_true",
        help=(
            "Allow torchvision.datasets.MNIST to download MNIST when "
            "--mnist-source torchvision or auto is used. torchvision remains "
            "an optional dependency."
        ),
    )
    parser.add_argument(
        "--mnist-split",
        choices=("stratified", "canonical"),
        default="stratified",
        help=(
            "Split for true OpenML MNIST. 'canonical' uses the standard "
            "60,000/10,000 source split; sklearn fallbacks remain stratified."
        ),
    )
    parser.add_argument(
        "--openml-data-home",
        type=Path,
        default=None,
        help="Optional sklearn OpenML cache directory.",
    )
    parser.add_argument(
        "--torchvision-data-home",
        type=Path,
        default=None,
        help="Optional torchvision MNIST cache directory.",
    )
    parser.add_argument("--openml-n-retries", type=int, default=2)
    parser.add_argument("--openml-retry-delay", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=optional_positive_int, default=4000)
    parser.add_argument("--max-test-examples", type=optional_positive_int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--task-block-size", type=int, default=300)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument(
        "--include-identity-permutation",
        action="store_true",
        help=(
            "Use the unpermuted pixel order as task 0. Leave disabled for the "
            "Dohare-style protocol, where every task is a random pixel permutation."
        ),
    )
    parser.add_argument(
        "--task-sampling",
        choices=("random", "sequential_epoch"),
        default="random",
        help=(
            "Within-task example order. sequential_epoch walks a shuffled epoch "
            "per permutation and is the closest local analogue to one-pass "
            "Online Permuted MNIST task blocks."
        ),
    )
    parser.add_argument(
        "--max-test-permutation-views",
        type=optional_positive_int,
        default=None,
        help=(
            "Cap held-out evaluation views to avoid materializing very large "
            "test tensors. Use 'all' for every permutation."
        ),
    )
    parser.add_argument(
        "--evaluate-all-permutation-views",
        action="store_true",
        help=(
            "Evaluate held-out accuracy on every configured permutation, including "
            "tasks not yet observed. By default, held-out metrics cover observed "
            "task views only."
        ),
    )
    parser.add_argument(
        "--mnist-published-scale",
        action="store_true",
        help=(
            "Configure the MNIST side for the full OpenML 60k/10k split and "
            "60k sequential task blocks. Requires --allow-openml-download to "
            "actually use OpenML; otherwise auto mode still falls back locally."
        ),
    )
    parser.add_argument(
        "--opmnist-streaming",
        action="store_true",
        help=(
            "Use the chunked/resumable OPMNIST path instead of materializing the "
            "entire online classification stream."
        ),
    )
    parser.add_argument("--opmnist-chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--opmnist-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load/save per-seed chunked OPMNIST checkpoints.",
    )
    parser.add_argument(
        "--opmnist-resume-path",
        type=Path,
        default=None,
        help="Optional checkpoint path; multi-seed runs append the seed to the stem.",
    )
    parser.add_argument(
        "--opmnist-force-restart",
        action="store_true",
        help="Delete an existing OPMNIST checkpoint before a chunked run.",
    )
    parser.add_argument(
        "--opmnist-status-checkpoint",
        type=Path,
        default=None,
        help=(
            "Print JSON progress/ETA for a chunked OPMNIST checkpoint and exit. "
            "Reads the lightweight sidecar when available."
        ),
    )
    parser.add_argument(
        "--opmnist-status-results",
        type=Path,
        default=None,
        help="Print JSON full-scale ETA from a completed OPMNIST results file and exit.",
    )
    parser.add_argument(
        "--opmnist-status-target-steps",
        type=int,
        default=DOHARE_OPMNIST_TOTAL_STEPS,
        help="Target step count for OPMNIST status/ETA reporting.",
    )
    parser.add_argument(
        "--opmnist-status-output",
        type=Path,
        default=None,
        help="Optional path for writing status-only JSON atomically.",
    )
    parser.add_argument(
        "--scr-preset",
        choices=("compact", "dohare_small", "dohare_paper"),
        default="compact",
        help=(
            "SCR parameter preset. dohare_paper matches the public reproduction "
            "config m=20, f=15, T=10000, target hidden=100; dohare_small keeps "
            "the same bit/target shape but uses T=1000 for shorter local runs."
        ),
    )
    parser.add_argument(
        "--long-scr",
        action="store_true",
        help=(
            "Moderate longer SCR run: 3 seeds, 20000 steps, final window 5000, "
            "Dohare-small SCR parameters."
        ),
    )
    parser.add_argument("--regression-bits", type=int, default=20)
    parser.add_argument("--regression-slow-bits", type=int, default=5)
    parser.add_argument("--regression-flip-interval", type=int, default=50)
    parser.add_argument("--regression-target-hidden", type=int, default=100)
    parser.add_argument("--regression-beta", type=float, default=0.7)
    parser.add_argument("--regression-noise-std", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-4)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamic-hidden-size", type=int, default=64)
    parser.add_argument("--dynamic-utility-decay", type=float, default=0.99)
    parser.add_argument("--dynamic-rewire-interval", type=int, default=240)
    parser.add_argument("--dynamic-unit-replacement-rate", type=float, default=0.05)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument(
        "--router-policy",
        choices=(
            "convex",
            "all_selector",
            "mlp_convex",
            "mlp_selector",
            "guarded_convex",
            "guarded_best_mlp",
            "meta",
        ),
        default="convex",
    )
    parser.add_argument("--router-decay", type=float, default=0.02)
    parser.add_argument("--guard-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--digits-deployment-objective",
        choices=("mse", "accuracy", "mlp_h128", "dynamic_sparse"),
        default="mse",
    )
    parser.add_argument(
        "--online-retention-mse-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--online-retention-min-lifetime-class-fraction", type=float, default=0.7)
    parser.add_argument("--online-retention-max-recent-class-fraction", type=float, default=0.5)
    parser.add_argument(
        "--retention-router",
        choices=("none", "class_imbalance"),
        default="none",
    )
    parser.add_argument("--retention-upgd-deployment-weight", type=float, default=1.0)
    parser.add_argument("--retention-min-lifetime-class-fraction", type=float, default=0.8)
    parser.add_argument("--retention-max-recent-class-fraction", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast 1-seed harness check with tiny streams.",
    )
    parser.add_argument(
        "--canonical-ish",
        action="store_true",
        help="Moderate local run: 5 seeds, 1500 steps, 5 permutations.",
    )
    return parser.parse_args()


def apply_run_preset(args: argparse.Namespace) -> None:
    """Apply smoke or canonical-ish presets in-place."""
    if sum(bool(value) for value in (args.smoke, args.canonical_ish, args.long_scr)) > 1:
        raise ValueError("--smoke, --canonical-ish, and --long-scr are mutually exclusive")

    if args.mnist_published_scale:
        args.mnist_split = "canonical"
        args.max_train_examples = None
        args.max_test_examples = None
        args.task_block_size = 60_000
        args.task_sampling = "sequential_epoch"
        args.sample_with_replacement = False
        args.opmnist_streaming = True

    if args.scr_preset == "dohare_small":
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 1_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7
    elif args.scr_preset == "dohare_paper":
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 10_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7

    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.n_permutations = 2
        args.task_block_size = 40
        args.max_train_examples = 200
        args.max_test_examples = 80
        args.regression_bits = 8
        args.regression_slow_bits = 3
        args.regression_flip_interval = 20
        args.regression_target_hidden = 16
        args.dynamic_rewire_interval = 60
    elif args.canonical_ish:
        args.steps = 1500
        args.n_seeds = 5
        args.final_window = 300
        args.n_permutations = 5
        args.task_block_size = 300
        args.max_train_examples = 4000
        args.max_test_examples = 1000
        args.regression_bits = 20
        args.regression_slow_bits = 5
        args.regression_flip_interval = 50
        args.regression_target_hidden = 100
        args.dynamic_rewire_interval = 240
    elif args.long_scr:
        args.steps = 20_000
        args.n_seeds = 3
        args.final_window = 5_000
        args.scr_preset = "dohare_small"
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 1_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7
        args.dynamic_rewire_interval = 500


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.opmnist_status_target_steps <= 0:
        raise ValueError("--opmnist-status-target-steps must be positive")
    if args.opmnist_status_checkpoint is not None and not args.opmnist_status_checkpoint.exists():
        raise ValueError("--opmnist-status-checkpoint does not exist")
    if args.opmnist_status_results is not None and not args.opmnist_status_results.exists():
        raise ValueError("--opmnist-status-results does not exist")
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
    if args.opmnist_chunk_size <= 0:
        raise ValueError("--opmnist-chunk-size must be positive")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive or 'all'")
    if args.max_test_examples is not None and args.max_test_examples <= 0:
        raise ValueError("--max-test-examples must be positive or 'all'")
    if args.openml_n_retries < 0:
        raise ValueError("--openml-n-retries must be non-negative")
    if args.openml_retry_delay < 0.0:
        raise ValueError("--openml-retry-delay must be non-negative")
    if args.mnist_source not in {"openml", "torchvision"} and args.mnist_split == "canonical":
        print(
            "--mnist-split canonical applies only to true MNIST sources; "
            "local fallbacks use a stratified split."
        )
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")
    if args.regression_bits <= 0:
        raise ValueError("--regression-bits must be positive")
    if not 0 < args.regression_slow_bits <= args.regression_bits:
        raise ValueError("--regression-slow-bits must be in [1, regression_bits]")
    if args.regression_flip_interval <= 0:
        raise ValueError("--regression-flip-interval must be positive")
    if args.regression_target_hidden <= 0:
        raise ValueError("--regression-target-hidden must be positive")
    if args.regression_noise_std < 0.0:
        raise ValueError("--regression-noise-std must be non-negative")
    if args.perturbation_warmup_steps < 0:
        raise ValueError("--perturbation-warmup-steps must be non-negative")
    if args.perturbation_ramp_steps < 0:
        raise ValueError("--perturbation-ramp-steps must be non-negative")
    if args.dynamic_rewire_interval <= 0:
        raise ValueError("--dynamic-rewire-interval must be positive")
    if not 0.0 <= args.dynamic_unit_replacement_rate <= 1.0:
        raise ValueError("--dynamic-unit-replacement-rate must be in [0, 1]")
    if not 0.0 <= args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in [0, 1]")
    if not 0.0 < args.router_decay <= 1.0:
        raise ValueError("--router-decay must be in (0, 1]")


def main() -> None:
    """Run the selected published-style stressors."""
    args = parse_args()
    validate_args(args)
    if args.opmnist_status_checkpoint is not None or args.opmnist_status_results is not None:
        reports: dict[str, Any] = {}
        if args.opmnist_status_checkpoint is not None:
            reports["checkpoint"] = opmnist_status_from_checkpoint(
                args.opmnist_status_checkpoint,
                target_steps=args.opmnist_status_target_steps,
                task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
            )
        if args.opmnist_status_results is not None:
            reports["results"] = opmnist_eta_from_results(
                args.opmnist_status_results,
                target_steps=args.opmnist_status_target_steps,
                task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
            )
        reports["schema"] = "alberta.opmnist.status_report.v1"
        reports["created_at_utc"] = utc_now_iso()
        if args.opmnist_status_output is not None:
            atomic_write_json(args.opmnist_status_output, reports)
        print(json.dumps(reports, indent=2, sort_keys=True))
        if args.opmnist_status_output is not None:
            print(f"wrote {args.opmnist_status_output}")
        return
    apply_run_preset(args)
    validate_args(args)
    benchmarks = expand_benchmarks(args.benchmarks)

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.result_prefix}_manifest.json"
    setattr(args, "run_manifest_path", str(manifest_path))
    atomic_write_json(manifest_path, run_manifest(args, benchmarks))
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}

    for benchmark in benchmarks:
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            print(f"benchmark={benchmark} seed={seed}: running portfolio")
            record, meta = run_one_seed(benchmark, seed, args)
            records.append(record)
            datasets_meta[benchmark] = meta
            methods = record["methods"]
            best_mlp = min(methods[name]["final_window_mse"] for name in MLP_METHODS)
            print(
                f"benchmark={benchmark} seed={seed}: final MSE "
                f"portfolio={methods['mixture']['final_window_mse']:.4f}, "
                f"best_mlp={best_mlp:.4f}"
            )

    results = {
        "config": config_dict(args, benchmarks),
        "datasets": datasets_meta,
        "records": records,
        "aggregate": aggregate_records(records),
        "status": {},
        "references": list(REFERENCES),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "compact_published_style_external_stressors",
        "manifest_path": str(manifest_path),
    }
    results["status"] = benchmark_status(results)

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

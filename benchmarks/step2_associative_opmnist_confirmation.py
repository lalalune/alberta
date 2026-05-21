#!/usr/bin/env python3
"""Guarded OPMNIST-scale confirmation runner for Step 2 associative memory.

The runner is intentionally a protocol surface, not a default 48M-example job.
Smoke mode executes quickly against bundled sklearn digits when available and
falls back to a deterministic synthetic digit stream. Partial and full modes
share the same manifest/status fields so a later large run can be audited
without changing result schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.associative_memory import (  # noqa: E402
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    AssociativeMemoryState,
    run_associative_memory_arrays,
)

ScaleName = Literal["smoke", "partial", "full"]
DatasetSource = Literal["auto", "sklearn_digits", "synthetic", "mnist_npz", "openml"]

N_CLASSES = 10
DOHARE_OPMNIST_TASKS = 800
DOHARE_OPMNIST_TASK_BLOCK_SIZE = 60_000
DOHARE_OPMNIST_TOTAL_STEPS = DOHARE_OPMNIST_TASKS * DOHARE_OPMNIST_TASK_BLOCK_SIZE
DEFAULT_OUTPUT_DIR = Path("outputs/step2_associative_opmnist_confirmation")
METRIC_NAMES = (
    "nll",
    "accuracy",
    "active_features",
    "occupied_features",
    "mean_feature_weight",
    "allocations",
    "replacements",
    "total_weight",
)


@dataclass(frozen=True)
class ScaleDefaults:
    """Default run sizes for one confirmation mode."""

    steps: int
    n_permutations: int
    task_block_size: int
    n_seeds: int
    chunk_size: int
    final_window: int
    max_train_examples: int | None
    max_test_examples: int | None
    max_test_permutation_views: int | None
    max_features: int


@dataclass(frozen=True)
class ClassificationDataset:
    """Arrays and source metadata for a multiclass confirmation stream."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MetricAccumulator:
    """Streaming metric sums plus final-window tail."""

    n_steps: int
    sums: np.ndarray
    tail: np.ndarray


SCALE_DEFAULTS: dict[ScaleName, ScaleDefaults] = {
    "smoke": ScaleDefaults(
        steps=128,
        n_permutations=2,
        task_block_size=64,
        n_seeds=1,
        chunk_size=128,
        final_window=32,
        max_train_examples=512,
        max_test_examples=128,
        max_test_permutation_views=2,
        max_features=512,
    ),
    "partial": ScaleDefaults(
        steps=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        n_permutations=DOHARE_OPMNIST_TASKS,
        task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        n_seeds=3,
        chunk_size=5_000,
        final_window=5_000,
        max_train_examples=None,
        max_test_examples=2_000,
        max_test_permutation_views=5,
        max_features=8_192,
    ),
    "full": ScaleDefaults(
        steps=DOHARE_OPMNIST_TOTAL_STEPS,
        n_permutations=DOHARE_OPMNIST_TASKS,
        task_block_size=DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        n_seeds=5,
        chunk_size=10_000,
        final_window=10_000,
        max_train_examples=None,
        max_test_examples=None,
        max_test_permutation_views=DOHARE_OPMNIST_TASKS,
        max_features=16_384,
    ),
}


def parse_seed_list(raw: str | None, n_seeds: int) -> list[int]:
    """Parse ``--seeds`` or fall back to ``range(n_seeds)``."""
    if raw is None:
        if n_seeds <= 0:
            raise ValueError("--n-seeds must be positive")
        return list(range(n_seeds))
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def scale_default(scale: str) -> ScaleDefaults:
    """Return defaults for a scale name."""
    if scale not in SCALE_DEFAULTS:
        raise ValueError(f"unknown scale: {scale}")
    return SCALE_DEFAULTS[cast(ScaleName, scale)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Guarded Step 2 associative OPMNIST confirmation runner."
    )
    parser.add_argument("--scale", choices=tuple(SCALE_DEFAULTS), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-published-scale",
        action="store_true",
        help="Required to execute --scale full instead of writing only a dry-run plan.",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--final-window", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--task-block-size", type=int, default=None)
    parser.add_argument(
        "--mnist-source",
        choices=("auto", "sklearn_digits", "synthetic", "mnist_npz", "openml"),
        default="auto",
    )
    parser.add_argument("--mnist-npz", type=Path, default=None)
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--suffix-length", type=int, default=6)
    parser.add_argument("--pixel-bins", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--feature-family",
        choices=("position_token", "suffix_pair", "token_suffix_pair"),
        default="token_suffix_pair",
    )
    parser.add_argument("--write-lr", type=float, default=1.0)
    parser.add_argument("--retention", type=float, default=0.8)
    parser.add_argument("--utility-lr", type=float, default=0.1)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--logit-scale", type=float, default=4.0)
    parser.add_argument("--adaptive-feature-family", action="store_true")
    parser.add_argument("--adaptive-window", action="store_true")
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--scope-lr", type=float, default=0.05)
    parser.add_argument("--budget-lr", type=float, default=0.05)
    parser.add_argument("--initial-budget-fraction", type=float, default=0.5)
    parser.add_argument("--min-effective-budget", type=int, default=1)
    parser.add_argument("--scope-logit-clip", type=float, default=8.0)
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--evaluate-all-permutation-views", action="store_true")
    parser.add_argument("--max-test-permutation-views", type=int, default=None)
    parser.add_argument("--include-identity-permutation", action="store_true")
    return parser


def apply_scale_defaults(args: argparse.Namespace) -> None:
    """Fill unset arguments from the selected scale preset."""
    defaults = scale_default(args.scale)
    if args.steps is None:
        args.steps = defaults.steps
    if args.n_permutations is None:
        args.n_permutations = defaults.n_permutations
    if args.task_block_size is None:
        args.task_block_size = defaults.task_block_size
    if args.n_seeds is None:
        args.n_seeds = defaults.n_seeds
    if args.chunk_size is None:
        args.chunk_size = defaults.chunk_size
    if args.final_window is None:
        args.final_window = defaults.final_window
    if args.max_train_examples is None:
        args.max_train_examples = defaults.max_train_examples
    if args.max_test_examples is None:
        args.max_test_examples = defaults.max_test_examples
    if args.max_test_permutation_views is None:
        args.max_test_permutation_views = defaults.max_test_permutation_views
    if args.max_features is None:
        args.max_features = defaults.max_features
    if args.result_prefix is None:
        args.result_prefix = f"associative_opmnist_{args.scale}"
    args.seed_list = parse_seed_list(args.seeds, args.n_seeds)


def validate_args(args: argparse.Namespace) -> None:
    """Validate resolved CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive")
    if args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.suffix_length < 2 or args.suffix_length > args.block_size:
        raise ValueError("--suffix-length must be in [2, --block-size]")
    if args.pixel_bins < 2:
        raise ValueError("--pixel-bins must be at least 2")
    if args.train_fraction <= 0.0 or args.train_fraction >= 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive when set")
    if args.max_test_examples is not None and args.max_test_examples <= 0:
        raise ValueError("--max-test-examples must be positive when set")
    if (
        args.max_test_permutation_views is not None
        and args.max_test_permutation_views <= 0
    ):
        raise ValueError("--max-test-permutation-views must be positive when set")
    if args.scale == "full" and not (args.allow_published_scale or args.dry_run):
        raise ValueError(
            "--scale full would request 48,000,000 examples; pass "
            "--dry-run to write a plan or --allow-published-scale to execute."
        )
    if args.scope_lr < 0.0:
        raise ValueError("--scope-lr must be non-negative")
    if args.budget_lr < 0.0:
        raise ValueError("--budget-lr must be non-negative")
    if not 0.0 < args.initial_budget_fraction <= 1.0:
        raise ValueError("--initial-budget-fraction must be in (0, 1]")
    if args.min_effective_budget < 1:
        raise ValueError("--min-effective-budget must be positive")
    if args.min_effective_budget > args.max_features:
        raise ValueError("--min-effective-budget must be <= --max-features")
    if args.scope_logit_clip <= 0.0:
        raise ValueError("--scope-logit-clip must be positive")
    if args.mnist_source == "mnist_npz" and args.mnist_npz is None:
        raise ValueError("--mnist-source mnist_npz requires --mnist-npz")
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse, default, and validate CLI arguments."""
    args = build_parser().parse_args(argv)
    apply_scale_defaults(args)
    validate_args(args)
    return args


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_metadata() -> dict[str, Any]:
    """Return git commit and dirty status when available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}
    if commit.returncode != 0:
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def environment_metadata() -> dict[str, Any]:
    """Capture lightweight runtime environment metadata."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "scikit_learn": _package_version("scikit-learn"),
        "alberta_framework": _package_version("alberta-framework"),
    }


def _standardize_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale feature arrays into a stable [0, 1] range."""
    train = np.asarray(x_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    max_value = float(np.max(train)) if train.size else 1.0
    min_value = float(np.min(train)) if train.size else 0.0
    if min_value >= 0.0 and max_value > 1.0:
        scale = max_value if max_value > 0.0 else 1.0
        return (train / scale).astype(np.float32), (test / scale).astype(np.float32)
    if min_value < 0.0 or max_value > 1.0:
        span = max(max_value - min_value, 1e-6)
        return (
            ((train - min_value) / span).astype(np.float32),
            ((test - min_value) / span).astype(np.float32),
        )
    return train.astype(np.float32), test.astype(np.float32)


def _limit_split(
    x: np.ndarray,
    y: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or x.shape[0] <= limit:
        return x, y
    return x[:limit], y[:limit]


def _split_indices(
    n_examples: int,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_examples)
    n_train = max(1, min(n_examples - 1, int(round(n_examples * train_fraction))))
    return order[:n_train], order[n_train:]


def make_synthetic_dataset(
    *,
    seed: int,
    train_fraction: float,
    max_train_examples: int | None,
    max_test_examples: int | None,
    feature_dim: int = 64,
    n_total: int = 1_000,
) -> ClassificationDataset:
    """Create a deterministic digit-like fallback dataset."""
    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.0, 0.25, size=(N_CLASSES, feature_dim)).astype(np.float32)
    for label in range(N_CLASSES):
        start = (label * feature_dim) // N_CLASSES
        end = ((label + 1) * feature_dim) // N_CLASSES
        prototypes[label, start:end] += 0.75
    labels: np.ndarray = np.arange(n_total, dtype=np.int32) % N_CLASSES
    labels = labels[rng.permutation(n_total)]
    noise = rng.normal(0.0, 0.08, size=(n_total, feature_dim)).astype(np.float32)
    x = np.clip(prototypes[labels] + noise, 0.0, 1.0).astype(np.float32)
    train_idx, test_idx = _split_indices(n_total, seed + 17, train_fraction)
    x_train, y_train = _limit_split(x[train_idx], labels[train_idx], max_train_examples)
    x_test, y_test = _limit_split(x[test_idx], labels[test_idx], max_test_examples)
    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "source": "synthetic",
            "source_kind": "deterministic_synthetic_digits",
            "description": "Synthetic class-prototype digit fallback; not external evidence.",
            "fallback_used": True,
            "is_true_mnist": False,
            "is_full_mnist_split": False,
            "feature_dim": int(feature_dim),
            "n_classes": N_CLASSES,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "train_fraction": float(train_fraction),
        },
    )


def load_sklearn_digits_dataset(
    *,
    seed: int,
    train_fraction: float,
    max_train_examples: int | None,
    max_test_examples: int | None,
) -> ClassificationDataset:
    """Load bundled sklearn digits without network access."""
    try:
        from sklearn import datasets  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("scikit-learn is not installed") from exc
    raw = datasets.load_digits()
    x = np.asarray(raw.data, dtype=np.float32) / 16.0
    y = np.asarray(raw.target, dtype=np.int32)
    train_idx, test_idx = _split_indices(x.shape[0], seed, train_fraction)
    x_train, y_train = _limit_split(x[train_idx], y[train_idx], max_train_examples)
    x_test, y_test = _limit_split(x[test_idx], y[test_idx], max_test_examples)
    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "source": "sklearn_digits",
            "source_kind": "sklearn_digits_8x8",
            "description": "Bundled sklearn 8x8 handwritten digits.",
            "fallback_used": False,
            "is_true_mnist": False,
            "is_full_mnist_split": False,
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "train_fraction": float(train_fraction),
        },
    )


def load_mnist_npz_dataset(
    path: Path,
    *,
    max_train_examples: int | None,
    max_test_examples: int | None,
) -> ClassificationDataset:
    """Load a local MNIST-style npz file with x/y train/test arrays."""
    with np.load(path) as payload:
        x_train = np.asarray(payload["x_train"], dtype=np.float32).reshape(
            payload["x_train"].shape[0],
            -1,
        )
        y_train = np.asarray(payload["y_train"], dtype=np.int32)
        x_test = np.asarray(payload["x_test"], dtype=np.float32).reshape(
            payload["x_test"].shape[0],
            -1,
        )
        y_test = np.asarray(payload["y_test"], dtype=np.int32)
    x_train, x_test = _standardize_features(x_train, x_test)
    x_train, y_train = _limit_split(x_train, y_train, max_train_examples)
    x_test, y_test = _limit_split(x_test, y_test, max_test_examples)
    is_full = x_train.shape[0] == 60_000 and x_test.shape[0] == 10_000
    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "source": "mnist_npz",
            "source_kind": "local_mnist_npz",
            "path": str(path),
            "fallback_used": False,
            "is_true_mnist": True,
            "is_full_mnist_split": bool(is_full),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "train_fraction": None,
        },
    )


def load_openml_mnist_dataset(
    *,
    allow_openml_download: bool,
    max_train_examples: int | None,
    max_test_examples: int | None,
) -> ClassificationDataset:
    """Load canonical MNIST from OpenML when explicitly allowed."""
    if not allow_openml_download:
        raise RuntimeError("OpenML download requires --allow-openml-download")
    try:
        from sklearn.datasets import fetch_openml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("scikit-learn is required for OpenML MNIST") from exc
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x_all = np.asarray(raw.data, dtype=np.float32)
    y_all = np.asarray(raw.target, dtype=np.int32)
    x_train, x_test = _standardize_features(x_all[:60_000], x_all[60_000:70_000])
    y_train = y_all[:60_000]
    y_test = y_all[60_000:70_000]
    x_train, y_train = _limit_split(x_train, y_train, max_train_examples)
    x_test, y_test = _limit_split(x_test, y_test, max_test_examples)
    is_full = x_train.shape[0] == 60_000 and x_test.shape[0] == 10_000
    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "source": "openml",
            "source_kind": "openml_mnist_784",
            "fallback_used": False,
            "is_true_mnist": True,
            "is_full_mnist_split": bool(is_full),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "train_fraction": "canonical_60000_10000",
        },
    )


def load_confirmation_dataset(args: argparse.Namespace) -> ClassificationDataset:
    """Load the configured dataset, using synthetic fallback only in auto mode."""
    source = cast(DatasetSource, args.mnist_source)
    if source == "auto" and args.scale == "full":
        source = "openml"
    if source == "auto":
        try:
            return load_sklearn_digits_dataset(
                seed=args.dataset_seed,
                train_fraction=args.train_fraction,
                max_train_examples=args.max_train_examples,
                max_test_examples=args.max_test_examples,
            )
        except RuntimeError:
            return make_synthetic_dataset(
                seed=args.dataset_seed,
                train_fraction=args.train_fraction,
                max_train_examples=args.max_train_examples,
                max_test_examples=args.max_test_examples,
            )
    if source == "sklearn_digits":
        return load_sklearn_digits_dataset(
            seed=args.dataset_seed,
            train_fraction=args.train_fraction,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
        )
    if source == "synthetic":
        return make_synthetic_dataset(
            seed=args.dataset_seed,
            train_fraction=args.train_fraction,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
        )
    if source == "mnist_npz":
        return load_mnist_npz_dataset(
            cast(Path, args.mnist_npz),
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
        )
    return load_openml_mnist_dataset(
        allow_openml_download=args.allow_openml_download,
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
    )


def make_feature_orders(
    *,
    n_permutations: int,
    feature_dim: int,
    seed: int,
    include_identity_permutation: bool,
) -> tuple[np.ndarray, ...]:
    """Create deterministic pixel orders for each configured task."""
    rng = np.random.default_rng(seed)
    orders: list[np.ndarray] = []
    for task_id in range(n_permutations):
        if task_id == 0 and include_identity_permutation:
            orders.append(np.arange(feature_dim, dtype=np.int32))
        else:
            orders.append(rng.permutation(feature_dim).astype(np.int32))
    return tuple(orders)


def encode_images_as_contexts(
    images: np.ndarray,
    *,
    block_size: int,
    pixel_bins: int,
) -> np.ndarray:
    """Encode dense image vectors as fixed-length integer token contexts."""
    if images.ndim != 2:
        raise ValueError("images must be a 2D array")
    if block_size > images.shape[1]:
        raise ValueError("--block-size cannot exceed feature dimension")
    top: np.ndarray = np.argsort(-images, axis=1, kind="stable")[:, :block_size]
    values: np.ndarray = np.take_along_axis(images, top, axis=1)
    bins: np.ndarray = np.clip(
        (values * float(pixel_bins - 1)).astype(np.int32),
        0,
        pixel_bins - 1,
    )
    encoded = (top.astype(np.int32) * int(pixel_bins) + bins).astype(np.int32)
    return cast(np.ndarray, encoded)


def make_online_chunk(
    dataset: ClassificationDataset,
    feature_orders: Sequence[np.ndarray],
    *,
    start: int,
    length: int,
    task_block_size: int,
    block_size: int,
    pixel_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic sequential OPMNIST-style training chunk."""
    offsets: np.ndarray = np.arange(start, start + length, dtype=np.int64)
    train_indices = (offsets % dataset.x_train.shape[0]).astype(np.int64)
    task_ids: np.ndarray = ((offsets // task_block_size) % len(feature_orders)).astype(
        np.int64
    )
    contexts: np.ndarray = np.empty((length, block_size), dtype=np.int32)
    for task_id in np.unique(task_ids):
        row_mask = task_ids == task_id
        order = feature_orders[int(task_id)]
        images = dataset.x_train[train_indices[row_mask]][:, order]
        contexts[row_mask] = encode_images_as_contexts(
            images,
            block_size=block_size,
            pixel_bins=pixel_bins,
        )
    labels = dataset.y_train[train_indices].astype(np.int32)
    return contexts, labels, task_ids.astype(np.int32)


def observed_task_ids_for_steps(
    *,
    steps: int,
    task_block_size: int,
    n_permutations: int,
) -> list[int]:
    """Return task ids touched by a sequential task-block stream."""
    n_observed = min(n_permutations, max(1, math.ceil(steps / task_block_size)))
    return list(range(n_observed))


def test_task_ids_for_protocol(
    args: argparse.Namespace,
    observed_task_ids: Sequence[int],
) -> list[int]:
    """Choose held-out permutation views for evaluation."""
    if args.evaluate_all_permutation_views:
        return list(range(args.n_permutations))
    limit = args.max_test_permutation_views
    if limit is None:
        return list(observed_task_ids)
    return list(observed_task_ids[:limit])


def init_accumulator(final_window: int) -> MetricAccumulator:
    """Create an empty metric accumulator."""
    return MetricAccumulator(
        n_steps=0,
        sums=np.zeros((len(METRIC_NAMES),), dtype=np.float64),
        tail=np.zeros((0, len(METRIC_NAMES)), dtype=np.float64),
    )


def update_accumulator(
    accumulator: MetricAccumulator,
    metrics: np.ndarray,
    *,
    final_window: int,
) -> MetricAccumulator:
    """Add chunk metrics to an accumulator."""
    metric_array = np.asarray(metrics, dtype=np.float64)
    tail = np.concatenate([accumulator.tail, metric_array], axis=0)[-final_window:]
    return MetricAccumulator(
        n_steps=int(accumulator.n_steps + metric_array.shape[0]),
        sums=accumulator.sums + np.sum(metric_array, axis=0),
        tail=tail,
    )


def summarize_accumulator(accumulator: MetricAccumulator) -> dict[str, float]:
    """Summarize online and final-window metric means."""
    if accumulator.n_steps <= 0:
        raise RuntimeError("cannot summarize an empty accumulator")
    online = accumulator.sums / float(accumulator.n_steps)
    final = np.mean(accumulator.tail, axis=0)
    summary: dict[str, float] = {}
    for idx, name in enumerate(METRIC_NAMES):
        summary[f"online_mean_{name}"] = float(online[idx])
        summary[f"final_window_{name}"] = float(final[idx])
    return summary


def evaluate_state(
    learner: AssociativeMemoryLearner,
    state: AssociativeMemoryState,
    contexts: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final associative state without writes."""
    contexts_j = jnp.asarray(contexts, dtype=jnp.int32)
    labels_j = jnp.asarray(labels, dtype=jnp.int32)

    def step_fn(carry: None, inputs: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
        context, label = inputs
        prediction = learner.predict(state, context)
        safe_prob = jnp.maximum(prediction.probabilities[label], 1e-12)
        nll = -jnp.log(safe_prob)
        accuracy = (jnp.argmax(prediction.probabilities) == label).astype(jnp.float32)
        return carry, jnp.stack([nll, accuracy])

    _, metrics = jax.lax.scan(step_fn, None, (contexts_j, labels_j))
    metrics_np = np.asarray(metrics, dtype=np.float64)
    return {
        "test_nll": float(np.mean(metrics_np[:, 0])),
        "test_accuracy": float(np.mean(metrics_np[:, 1])),
    }


def evaluate_heldout_views(
    learner: AssociativeMemoryLearner,
    state: AssociativeMemoryState,
    dataset: ClassificationDataset,
    feature_orders: Sequence[np.ndarray],
    test_task_ids: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate final state on selected held-out permutation views."""
    if args.skip_heldout or not test_task_ids:
        return {
            "heldout_evaluated": False,
            "test_task_ids": list(test_task_ids),
            "test_nll": None,
            "test_accuracy": None,
            "per_view": [],
        }
    per_view: list[dict[str, Any]] = []
    for task_id in test_task_ids:
        order = feature_orders[task_id]
        contexts = encode_images_as_contexts(
            dataset.x_test[:, order],
            block_size=args.block_size,
            pixel_bins=args.pixel_bins,
        )
        metrics = evaluate_state(learner, state, contexts, dataset.y_test)
        per_view.append({"task_id": int(task_id), **metrics})
    return {
        "heldout_evaluated": True,
        "test_task_ids": [int(task_id) for task_id in test_task_ids],
        "test_nll": float(np.mean([row["test_nll"] for row in per_view])),
        "test_accuracy": float(np.mean([row["test_accuracy"] for row in per_view])),
        "per_view": per_view,
    }


def make_learner(args: argparse.Namespace, n_classes: int) -> AssociativeMemoryLearner:
    """Create the Step 2 associative core learner under confirmation."""
    config = AssociativeMemoryConfig(
        vocab_size=n_classes,
        block_size=args.block_size,
        suffix_length=args.suffix_length,
        feature_family=args.feature_family,
        max_features=args.max_features,
        write_lr=args.write_lr,
        retention=args.retention,
        utility_lr=args.utility_lr,
        utility_decay=args.utility_decay,
        logit_scale=args.logit_scale,
        adaptive_feature_family=args.adaptive_feature_family,
        adaptive_window=args.adaptive_window,
        adaptive_budget=args.adaptive_budget,
        scope_lr=args.scope_lr,
        budget_lr=args.budget_lr,
        initial_budget_fraction=args.initial_budget_fraction,
        min_effective_budget=args.min_effective_budget,
        scope_logit_clip=args.scope_logit_clip,
    )
    return AssociativeMemoryLearner(config)


def run_seed(
    *,
    seed: int,
    dataset: ClassificationDataset,
    args: argparse.Namespace,
    observed_task_ids: Sequence[int],
    test_task_ids: Sequence[int],
) -> dict[str, Any]:
    """Run one prequential confirmation seed."""
    feature_dim = int(dataset.x_train.shape[1])
    feature_orders = make_feature_orders(
        n_permutations=args.n_permutations,
        feature_dim=feature_dim,
        seed=seed + 50_000,
        include_identity_permutation=args.include_identity_permutation,
    )
    learner = make_learner(args, int(dataset.metadata["n_classes"]))
    state = learner.init()
    accumulator = init_accumulator(args.final_window)
    started = time.perf_counter()
    completed_steps = 0
    while completed_steps < args.steps:
        chunk_steps = min(args.chunk_size, args.steps - completed_steps)
        contexts, labels, _ = make_online_chunk(
            dataset,
            feature_orders,
            start=completed_steps,
            length=chunk_steps,
            task_block_size=args.task_block_size,
            block_size=args.block_size,
            pixel_bins=args.pixel_bins,
        )
        result = run_associative_memory_arrays(
            learner,
            state,
            jnp.asarray(contexts, dtype=jnp.int32),
            jnp.asarray(labels, dtype=jnp.int32),
        )
        state = result.state
        accumulator = update_accumulator(
            accumulator,
            np.asarray(result.metrics),
            final_window=args.final_window,
        )
        completed_steps += chunk_steps
    train_summary = summarize_accumulator(accumulator)
    heldout = evaluate_heldout_views(
        learner,
        state,
        dataset,
        feature_orders,
        test_task_ids,
        args,
    )
    elapsed_s = time.perf_counter() - started
    return {
        "seed": int(seed),
        "steps": int(completed_steps),
        "methods": {
            "associative_core": {
                **train_summary,
                "test_nll": heldout["test_nll"],
                "test_accuracy": heldout["test_accuracy"],
            }
        },
        "heldout": heldout,
        "state": {
            "allocations": int(state.allocations),
            "replacements": int(state.replacements),
            "occupied_features": int(np.sum(np.asarray(state.counts) > 0.0)),
            "max_features": int(args.max_features),
        },
        "elapsed_s": float(elapsed_s),
        "steps_per_second": float(completed_steps / elapsed_s) if elapsed_s > 0.0 else None,
        "observed_task_ids": [int(task_id) for task_id in observed_task_ids],
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed records for the single associative candidate."""
    if not records:
        return {}
    metric_names = [
        "online_mean_nll",
        "online_mean_accuracy",
        "final_window_nll",
        "final_window_accuracy",
        "test_nll",
        "test_accuracy",
    ]
    aggregate: dict[str, Any] = {"associative_core": {}}
    for metric in metric_names:
        values = [
            record["methods"]["associative_core"][metric]
            for record in records
            if record["methods"]["associative_core"][metric] is not None
        ]
        if not values:
            aggregate["associative_core"][metric] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate["associative_core"][metric] = {
            "mean": float(np.mean(arr)),
            "stderr": float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))
            if arr.shape[0] > 1
            else 0.0,
            "per_seed": arr.tolist(),
        }
    aggregate["method_order"] = ["associative_core"]
    return aggregate


def published_scale_guard(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the explicit guard around the 48M-example run."""
    planned_full = (
        args.n_permutations == DOHARE_OPMNIST_TASKS
        and args.task_block_size == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and args.steps == DOHARE_OPMNIST_TOTAL_STEPS
    )
    return {
        "guard_schema": "alberta.step2.associative_opmnist.guard.v1",
        "scale": args.scale,
        "published_reference_tasks": DOHARE_OPMNIST_TASKS,
        "published_reference_task_block_size": DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        "published_reference_total_steps": DOHARE_OPMNIST_TOTAL_STEPS,
        "configured_for_full_published_scale": bool(planned_full),
        "requires_allow_published_scale": args.scale == "full",
        "allow_published_scale": bool(args.allow_published_scale),
        "dry_run": bool(args.dry_run),
        "will_execute_training": bool(
            not args.dry_run
            and (args.scale != "full" or args.allow_published_scale)
        ),
        "smoke_or_partial_never_counts_as_published": args.scale in {"smoke", "partial"},
    }


def protocol_metadata(
    args: argparse.Namespace,
    dataset_meta: dict[str, Any],
    *,
    completed_steps: int,
    observed_task_ids: Sequence[int],
    test_task_ids: Sequence[int],
) -> dict[str, Any]:
    """Build OPMNIST protocol metadata for manifest and status checks."""
    is_true_mnist = bool(dataset_meta.get("is_true_mnist", False))
    is_full_mnist_split = bool(dataset_meta.get("is_full_mnist_split", False))
    random_pixel_orders = not bool(args.include_identity_permutation)
    single_pass_full_blocks = (
        args.task_block_size == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and int(dataset_meta.get("n_train", 0) or 0) == DOHARE_OPMNIST_TASK_BLOCK_SIZE
    )
    core_protocol = bool(
        is_true_mnist
        and is_full_mnist_split
        and args.n_permutations == DOHARE_OPMNIST_TASKS
        and args.task_block_size == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and random_pixel_orders
        and single_pass_full_blocks
    )
    completed_full_blocks = (
        completed_steps // DOHARE_OPMNIST_TASK_BLOCK_SIZE
        if args.task_block_size == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        else 0
    )
    planned_full_blocks = (
        args.steps // DOHARE_OPMNIST_TASK_BLOCK_SIZE
        if args.task_block_size == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        else 0
    )
    test_covers_observed = set(observed_task_ids).issubset(set(test_task_ids))
    test_covers_all = len(test_task_ids) == args.n_permutations
    return {
        "protocol_schema": "alberta.step2.associative_opmnist.protocol.v1",
        "scale": args.scale,
        "planned_steps": int(args.steps),
        "completed_steps": int(completed_steps),
        "n_permutations": int(args.n_permutations),
        "task_block_size": int(args.task_block_size),
        "task_sampling": "sequential_task_blocks",
        "task_id_provided_to_learner": False,
        "prediction_before_update_every_step": True,
        "associative_core_updates_every_step": True,
        "context_encoder": {
            "kind": "top_pixel_token_context",
            "block_size": int(args.block_size),
            "pixel_bins": int(args.pixel_bins),
            "feature_family": args.feature_family,
        },
        "uses_random_pixel_permutations_for_all_tasks": bool(random_pixel_orders),
        "include_identity_permutation": bool(args.include_identity_permutation),
        "full_mnist_task_blocks": bool(single_pass_full_blocks),
        "matches_dohare_opmnist_core_protocol": core_protocol,
        "configured_for_dohare_opmnist_published_task_count": bool(
            args.n_permutations == DOHARE_OPMNIST_TASKS
            and args.steps == DOHARE_OPMNIST_TOTAL_STEPS
        ),
        "matches_dohare_opmnist_published_task_count": bool(
            completed_steps >= DOHARE_OPMNIST_TOTAL_STEPS
            and args.n_permutations == DOHARE_OPMNIST_TASKS
        ),
        "planned_full_60000_task_blocks": int(planned_full_blocks),
        "opmnist_completed_full_60000_task_blocks": int(completed_full_blocks),
        "task_ids_observed": [int(task_id) for task_id in observed_task_ids],
        "test_task_ids_evaluated": [int(task_id) for task_id in test_task_ids],
        "test_views_cover_observed_permutations": bool(test_covers_observed),
        "test_views_cover_all_permutations": bool(test_covers_all),
    }


def benchmark_status(
    *,
    args: argparse.Namespace,
    dataset_meta: dict[str, Any],
    protocol: dict[str, Any],
    completed_steps: int,
) -> dict[str, Any]:
    """Return claim-support flags from dataset/protocol/run metadata."""
    full_execution = bool(
        args.scale == "full"
        and args.allow_published_scale
        and not args.dry_run
        and completed_steps >= DOHARE_OPMNIST_TOTAL_STEPS
    )
    published_supported = bool(
        full_execution
        and dataset_meta.get("is_true_mnist", False)
        and dataset_meta.get("is_full_mnist_split", False)
        and protocol.get("matches_dohare_opmnist_core_protocol", False)
        and protocol.get("matches_dohare_opmnist_published_task_count", False)
        and protocol.get("test_views_cover_all_permutations", False)
    )
    return {
        "status_schema": "alberta.step2.associative_opmnist.status.v1",
        "scale": args.scale,
        "completed_steps": int(completed_steps),
        "full_published_run_executed": full_execution,
        "uses_true_mnist_source": bool(dataset_meta.get("is_true_mnist", False)),
        "uses_full_mnist_split": bool(dataset_meta.get("is_full_mnist_split", False)),
        "matches_dohare_opmnist_core_protocol": bool(
            protocol.get("matches_dohare_opmnist_core_protocol", False)
        ),
        "matches_dohare_opmnist_published_task_count": bool(
            protocol.get("matches_dohare_opmnist_published_task_count", False)
        ),
        "published_scale_external_claim_supported": published_supported,
        "claim_scope": (
            "published_scale_opmnist_confirmation"
            if published_supported
            else f"{args.scale}_protocol_probe_not_published_confirmation"
        ),
    }


def published_scale_completion_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify that a result payload has a complete guarded OPMNIST manifest."""
    def int_at_least(value: object, minimum: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum

    manifest = payload.get("manifest")
    protocol = payload.get("protocol")
    guard = payload.get("published_scale_guard")
    status = payload.get("status")
    datasets = payload.get("datasets")
    records = payload.get("records")

    manifest_dict = manifest if isinstance(manifest, dict) else {}
    protocol_dict = protocol if isinstance(protocol, dict) else {}
    guard_dict = guard if isinstance(guard, dict) else {}
    status_dict = status if isinstance(status, dict) else {}
    datasets_dict = datasets if isinstance(datasets, dict) else {}
    dataset = datasets_dict.get("permuted_mnist_like", {})
    dataset_dict = dataset if isinstance(dataset, dict) else {}
    records_list = records if isinstance(records, list) else []
    manifest_seed_list = manifest_dict.get("seed_list", [])
    seed_list = manifest_seed_list if isinstance(manifest_seed_list, list) else []
    seed_values = {seed for seed in seed_list if isinstance(seed, int)}
    record_seeds = {
        int(record["seed"])
        for record in records_list
        if isinstance(record, dict) and isinstance(record.get("seed"), int)
    }

    schema_ok = payload.get("schema") == "alberta.step2.associative_opmnist.results.v1"
    manifest_schema_ok = (
        manifest_dict.get("schema") == "alberta.step2.associative_opmnist.manifest.v1"
    )
    protocol_schema_ok = (
        protocol_dict.get("protocol_schema")
        == "alberta.step2.associative_opmnist.protocol.v1"
    )
    guard_schema_ok = (
        guard_dict.get("guard_schema") == "alberta.step2.associative_opmnist.guard.v1"
    )
    status_schema_ok = (
        status_dict.get("status_schema") == "alberta.step2.associative_opmnist.status.v1"
    )
    manifest_consistent = bool(
        manifest_dict.get("protocol") == protocol_dict
        and manifest_dict.get("published_scale_guard") == guard_dict
        and manifest_dict.get("dataset") == dataset_dict
        and manifest_dict.get("scale") == "full"
    )
    guard_complete = bool(
        guard_dict.get("scale") == "full"
        and guard_dict.get("configured_for_full_published_scale") is True
        and guard_dict.get("requires_allow_published_scale") is True
        and guard_dict.get("allow_published_scale") is True
        and guard_dict.get("dry_run") is False
        and guard_dict.get("will_execute_training") is True
        and guard_dict.get("smoke_or_partial_never_counts_as_published") is False
    )
    protocol_complete = bool(
        protocol_dict.get("scale") == "full"
        and protocol_dict.get("planned_steps") == DOHARE_OPMNIST_TOTAL_STEPS
        and int_at_least(
            protocol_dict.get("completed_steps"),
            DOHARE_OPMNIST_TOTAL_STEPS,
        )
        and protocol_dict.get("n_permutations") == DOHARE_OPMNIST_TASKS
        and protocol_dict.get("task_block_size") == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and protocol_dict.get("task_id_provided_to_learner") is False
        and protocol_dict.get("prediction_before_update_every_step") is True
        and protocol_dict.get("associative_core_updates_every_step") is True
        and protocol_dict.get("uses_random_pixel_permutations_for_all_tasks") is True
        and protocol_dict.get("include_identity_permutation") is False
        and protocol_dict.get("full_mnist_task_blocks") is True
        and protocol_dict.get("matches_dohare_opmnist_core_protocol") is True
        and protocol_dict.get("configured_for_dohare_opmnist_published_task_count") is True
        and protocol_dict.get("matches_dohare_opmnist_published_task_count") is True
        and int_at_least(
            protocol_dict.get("opmnist_completed_full_60000_task_blocks"),
            DOHARE_OPMNIST_TASKS,
        )
        and protocol_dict.get("test_views_cover_all_permutations") is True
    )
    dataset_complete = bool(
        dataset_dict.get("fallback_used") is False
        and dataset_dict.get("is_true_mnist") is True
        and dataset_dict.get("is_full_mnist_split") is True
        and dataset_dict.get("n_train") == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and dataset_dict.get("n_test") == 10_000
        and dataset_dict.get("n_classes") == N_CLASSES
    )
    records_complete = bool(
        seed_list
        and records_list
        and len(seed_values) == len(seed_list)
        and record_seeds == seed_values
        and all(
            isinstance(record, dict)
            and int_at_least(record.get("steps"), DOHARE_OPMNIST_TOTAL_STEPS)
            for record in records_list
        )
    )
    status_complete = bool(
        status_dict.get("scale") == "full"
        and int_at_least(status_dict.get("completed_steps"), DOHARE_OPMNIST_TOTAL_STEPS)
        and status_dict.get("full_published_run_executed") is True
        and status_dict.get("uses_true_mnist_source") is True
        and status_dict.get("uses_full_mnist_split") is True
        and status_dict.get("matches_dohare_opmnist_core_protocol") is True
        and status_dict.get("matches_dohare_opmnist_published_task_count") is True
        and status_dict.get("published_scale_external_claim_supported") is True
    )
    not_dry_run = payload.get("dry_run") is False
    full_guarded_manifest = bool(
        schema_ok
        and manifest_schema_ok
        and protocol_schema_ok
        and guard_schema_ok
        and status_schema_ok
        and manifest_consistent
        and guard_complete
        and protocol_complete
        and dataset_complete
        and records_complete
        and status_complete
        and not_dry_run
    )
    return {
        "completion_schema": "alberta.step2.associative_opmnist.completion.v1",
        "result_schema_ok": bool(schema_ok),
        "manifest_schema_ok": bool(manifest_schema_ok),
        "protocol_schema_ok": bool(protocol_schema_ok),
        "guard_schema_ok": bool(guard_schema_ok),
        "status_schema_ok": bool(status_schema_ok),
        "manifest_consistent": bool(manifest_consistent),
        "guard_complete": bool(guard_complete),
        "protocol_complete": bool(protocol_complete),
        "dataset_complete": bool(dataset_complete),
        "records_complete": bool(records_complete),
        "status_complete": bool(status_complete),
        "not_dry_run": bool(not_dry_run),
        "full_guarded_manifest": bool(full_guarded_manifest),
        "published_scale_confirmed": bool(full_guarded_manifest),
    }


def canonical_opmnist_artifact_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Audit a canonical OPMNIST artifact without strengthening its claim.

    The packaged Step 2 canonical artifacts predate the guarded manifest schema
    above.  This verifier checks their protocol fields directly and keeps two
    separate questions apart: whether the artifact reached Dohare-style
    published scale, and whether it is strong enough to claim Step 2 is solved
    on OPMNIST.
    """

    def positive_candidate_diff(metric: str, method: str) -> bool:
        comparison = comparisons.get(metric, {}).get("candidate_vs_best_mlp", {})
        method_row = comparison.get(method, {})
        if not isinstance(method_row, dict):
            return False
        value = method_row.get("diff_mean_positive_favors_candidate")
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0

    config = payload.get("config")
    datasets = payload.get("datasets")
    records = payload.get("records")
    aggregate = payload.get("aggregate")
    config_dict = config if isinstance(config, dict) else {}
    datasets_dict = datasets if isinstance(datasets, dict) else {}
    dataset = datasets_dict.get("permuted_mnist_like", {})
    dataset_dict = dataset if isinstance(dataset, dict) else {}
    records_list = records if isinstance(records, list) else []
    aggregate_dict = aggregate if isinstance(aggregate, dict) else {}
    task_aggregate = aggregate_dict.get("permuted_mnist_like", {})
    task_aggregate_dict = task_aggregate if isinstance(task_aggregate, dict) else {}
    comparisons = task_aggregate_dict.get("comparisons", {})
    comparisons = comparisons if isinstance(comparisons, dict) else {}
    primary_method = payload.get("primary_method")
    primary_name = primary_method if isinstance(primary_method, str) else ""

    configured_seed_count = config_dict.get("n_seeds")
    if isinstance(configured_seed_count, int) and not isinstance(configured_seed_count, bool):
        n_configured_seeds = configured_seed_count
    else:
        n_configured_seeds = len(records_list)
    record_seeds = {
        int(record["seed"])
        for record in records_list
        if isinstance(record, dict) and isinstance(record.get("seed"), int)
    }

    def record_completed_steps(record: dict[str, Any]) -> object:
        direct_steps = record.get("steps")
        if direct_steps is not None:
            return direct_steps
        nested_dataset = record.get("dataset")
        if isinstance(nested_dataset, dict):
            return nested_dataset.get("steps")
        return None

    record_steps_complete = bool(
        records_list
        and all(
            isinstance(record, dict)
            and isinstance(record_completed_steps(record), int)
            and not isinstance(record_completed_steps(record), bool)
            and record_completed_steps(record) >= DOHARE_OPMNIST_TOTAL_STEPS
            for record in records_list
        )
    )
    protocol_complete = bool(
        config_dict.get("mnist_published_scale") is True
        and dataset_dict.get("is_true_mnist") is True
        and dataset_dict.get("is_full_mnist_split") is True
        and dataset_dict.get("n_train") == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and dataset_dict.get("n_test") == 10_000
        and dataset_dict.get("n_permutations") == DOHARE_OPMNIST_TASKS
        and dataset_dict.get("task_block_size") == DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and dataset_dict.get("steps") == DOHARE_OPMNIST_TOTAL_STEPS
        and dataset_dict.get("completed_full_task_blocks") == DOHARE_OPMNIST_TASKS
        and dataset_dict.get("opmnist_completed_full_60000_task_blocks")
        == DOHARE_OPMNIST_TASKS
        and dataset_dict.get("matches_dohare_opmnist_core_protocol") is True
        and dataset_dict.get("matches_dohare_opmnist_published_task_count") is True
        and dataset_dict.get("prediction_before_update_every_step") is True
        and dataset_dict.get("task_id_provided_to_learner") is False
        and dataset_dict.get("test_views_cover_all_permutations") is True
        and record_steps_complete
    )
    multi_seed_full_scale = bool(
        protocol_complete
        and n_configured_seeds >= 3
        and len(record_seeds) >= 3
        and len(record_seeds) == len(records_list)
    )
    core_metric_names = (
        "online_mean_mse",
        "online_mean_accuracy",
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
    )
    primary_beats_best_mlp = {
        metric: positive_candidate_diff(metric, primary_name) for metric in core_metric_names
    }
    primary_all_core_metrics_win = bool(
        primary_name and all(primary_beats_best_mlp.values())
    )
    solved_opmnist_step2 = bool(multi_seed_full_scale and primary_all_core_metrics_win)
    return {
        "artifact_schema": "alberta.step2.opmnist_artifact_audit.v1",
        "protocol_complete": bool(protocol_complete),
        "published_scale_single_or_more_seed": bool(protocol_complete),
        "configured_seed_count": int(n_configured_seeds),
        "completed_record_seed_count": int(len(record_seeds)),
        "multi_seed_full_scale": bool(multi_seed_full_scale),
        "primary_method": primary_name,
        "primary_beats_best_mlp_by_metric": primary_beats_best_mlp,
        "primary_all_core_metrics_win": bool(primary_all_core_metrics_win),
        "solved_opmnist_step2": bool(solved_opmnist_step2),
        "claim_scope": (
            "multi_seed_full_scale_opmnist_solution"
            if solved_opmnist_step2
            else "limited_opmnist_evidence_not_step2_solution"
        ),
    }


def args_config(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize relevant args into JSON-safe config."""
    keys = [
        "scale",
        "steps",
        "chunk_size",
        "final_window",
        "n_permutations",
        "task_block_size",
        "mnist_source",
        "mnist_npz",
        "allow_openml_download",
        "allow_published_scale",
        "dry_run",
        "dataset_seed",
        "train_fraction",
        "max_train_examples",
        "max_test_examples",
        "block_size",
        "suffix_length",
        "pixel_bins",
        "max_features",
        "feature_family",
        "write_lr",
        "retention",
        "utility_lr",
        "utility_decay",
        "logit_scale",
        "adaptive_feature_family",
        "adaptive_window",
        "adaptive_budget",
        "scope_lr",
        "budget_lr",
        "initial_budget_fraction",
        "min_effective_budget",
        "scope_logit_clip",
        "skip_heldout",
        "evaluate_all_permutation_views",
        "max_test_permutation_views",
        "include_identity_permutation",
        "result_prefix",
        "output_dir",
    ]
    config: dict[str, Any] = {}
    for key in keys:
        value = getattr(args, key)
        config[key] = str(value) if isinstance(value, Path) else value
    config["seeds"] = list(args.seed_list)
    return config


def build_manifest(
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    dataset_meta: dict[str, Any],
    protocol: dict[str, Any],
    guard: dict[str, Any],
) -> dict[str, Any]:
    """Build the reproducibility manifest."""
    return {
        "schema": "alberta.step2.associative_opmnist.manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runner": "benchmarks/step2_associative_opmnist_confirmation.py",
        "argv": list(argv),
        "git": git_metadata(),
        "environment": environment_metadata(),
        "scale": args.scale,
        "seed_list": list(args.seed_list),
        "config": args_config(args),
        "dataset": dataset_meta,
        "protocol": protocol,
        "published_scale_guard": guard,
    }


def dry_run_dataset_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return dataset metadata when no arrays were loaded."""
    source = "openml" if args.mnist_source == "auto" and args.scale == "full" else args.mnist_source
    return {
        "source": source,
        "source_kind": "not_loaded_dry_run",
        "fallback_used": False,
        "is_true_mnist": source in {"openml", "mnist_npz"},
        "is_full_mnist_split": False,
        "feature_dim": 784 if source in {"openml", "mnist_npz"} else None,
        "n_classes": N_CLASSES,
        "n_train": None,
        "n_test": None,
        "dry_run": True,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact atomically enough for local benchmark use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def run_confirmation(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    """Run or dry-run the confirmation protocol and return the result payload."""
    guard = published_scale_guard(args)
    if args.dry_run:
        dataset_meta = dry_run_dataset_metadata(args)
        observed = observed_task_ids_for_steps(
            steps=args.steps,
            task_block_size=args.task_block_size,
            n_permutations=args.n_permutations,
        )
        test_ids = test_task_ids_for_protocol(args, observed)
        protocol = protocol_metadata(
            args,
            dataset_meta,
            completed_steps=0,
            observed_task_ids=observed,
            test_task_ids=test_ids,
        )
        status = benchmark_status(
            args=args,
            dataset_meta=dataset_meta,
            protocol=protocol,
            completed_steps=0,
        )
        manifest = build_manifest(
            args=args,
            argv=argv,
            dataset_meta=dataset_meta,
            protocol=protocol,
            guard=guard,
        )
        return {
            "schema": "alberta.step2.associative_opmnist.results.v1",
            "dry_run": True,
            "manifest": manifest,
            "datasets": {"permuted_mnist_like": dataset_meta},
            "protocol": protocol,
            "published_scale_guard": guard,
            "records": [],
            "aggregate": {},
            "status": status,
        }

    dataset = load_confirmation_dataset(args)
    observed = observed_task_ids_for_steps(
        steps=args.steps,
        task_block_size=args.task_block_size,
        n_permutations=args.n_permutations,
    )
    test_ids = test_task_ids_for_protocol(args, observed)
    records = [
        run_seed(
            seed=seed,
            dataset=dataset,
            args=args,
            observed_task_ids=observed,
            test_task_ids=test_ids,
        )
        for seed in args.seed_list
    ]
    completed_steps = min(record["steps"] for record in records) if records else 0
    protocol = protocol_metadata(
        args,
        dataset.metadata,
        completed_steps=completed_steps,
        observed_task_ids=observed,
        test_task_ids=test_ids,
    )
    status = benchmark_status(
        args=args,
        dataset_meta=dataset.metadata,
        protocol=protocol,
        completed_steps=completed_steps,
    )
    manifest = build_manifest(
        args=args,
        argv=argv,
        dataset_meta=dataset.metadata,
        protocol=protocol,
        guard=guard,
    )
    return {
        "schema": "alberta.step2.associative_opmnist.results.v1",
        "dry_run": False,
        "manifest": manifest,
        "datasets": {"permuted_mnist_like": dataset.metadata},
        "protocol": protocol,
        "published_scale_guard": guard,
        "records": records,
        "aggregate": {"permuted_mnist_like": aggregate_records(records)},
        "status": status,
    }


def write_artifacts(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write result and standalone manifest JSON artifacts."""
    result_path = args.output_dir / f"{args.result_prefix}_results.json"
    manifest_path = args.output_dir / f"{args.result_prefix}_manifest.json"
    atomic_write_json(result_path, payload)
    atomic_write_json(manifest_path, cast(dict[str, Any], payload["manifest"]))
    return result_path, manifest_path


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint. Returns the payload for tests and notebooks."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_confirmation(args, raw_argv)
    result_path, manifest_path = write_artifacts(args, payload)
    print(f"wrote {result_path}")
    print(f"wrote {manifest_path}")
    return payload


if __name__ == "__main__":
    main()

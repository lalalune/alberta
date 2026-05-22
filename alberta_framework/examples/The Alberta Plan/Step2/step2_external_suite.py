#!/usr/bin/env python3
"""Step 2 external/non-synthetic online benchmark suite.

This suite broadens the first sklearn-digits sanity check into several
locally available, reproducible streams:

* shuffled digits
* class-blocked digits
* permuted digits
* mask/noise digits
* shuffled wine
* shuffled breast-cancer
* shuffled diabetes regression
* dense exact-zero regression
* sparse multilabel classification
* temporal delayed/history regression

The protocol is online and prequential. At each step each learner predicts the
current example, the suite records pre-update loss/accuracy, and only then does
the learner update on the current one-hot target. Held-out test metrics are
computed from the final state and never used for updates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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

from alberta_framework import (  # noqa: E402
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    MultiHeadMLPLearner,
    ObGDBounding,
    UPGDLearner,
)

DEFAULT_BENCHMARKS = (
    "digits_shuffled",
    "digits_class_blocked",
    "digits_permuted",
    "digits_mask_noise",
    "wine_shuffled",
    "breast_cancer_shuffled",
    "diabetes_regression",
    "dense_exact_zero",
    "sparse_multilabel",
    "temporal_delayed_history",
)
DEFAULT_METHODS = (
    "linear",
    "mlp",
    "mlp_deep",
    "upgd",
    "upgd_fast",
    "upgd_mean",
    "upgd_wide",
    "cbp",
)
ALLOWED_METHODS = (
    *DEFAULT_METHODS,
    "mlp_context",
    "upgd_reg_k2",
    "upgd_reg_noln",
    "upgd_reg_deep",
    "upgd_reg_input",
    "upgd_context",
    "upgd_context_input",
    "upgd_reg_passthrough_deep",
    "upgd_temporal_fast_passthrough",
    "upgd_temporal_passthrough_no_mutation",
)
LOSS_METRICS = ("online_mean_mse", "final_window_mse", "test_mse")
ACCURACY_METRICS = (
    "online_mean_accuracy",
    "final_window_accuracy",
    "test_accuracy",
)


@dataclass(frozen=True)
class LoadedDataset:
    """Train/test arrays plus metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkSpec:
    """A concrete online stream benchmark."""

    name: str
    dataset_name: str
    protocol: str
    task_kind: str = "multiclass"
    n_permutations: int = 1


def _import_sklearn_datasets() -> Any:
    try:
        import sklearn.datasets as datasets
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for this suite. Install with "
            "`pip install -e '.[external]'`. The datasets used here are bundled "
            "with scikit-learn and do not require network access at runtime."
        )
        raise RuntimeError(msg) from exc
    return datasets


def benchmark_specs(names: tuple[str, ...], n_permutations: int) -> tuple[BenchmarkSpec, ...]:
    """Map command-line benchmark names to stream specs."""
    registry = {
        "digits_shuffled": BenchmarkSpec(
            name="digits_shuffled",
            dataset_name="digits",
            protocol="shuffled",
        ),
        "digits_class_blocked": BenchmarkSpec(
            name="digits_class_blocked",
            dataset_name="digits",
            protocol="class_blocked",
        ),
        "digits_permuted": BenchmarkSpec(
            name="digits_permuted",
            dataset_name="digits",
            protocol="permuted",
            n_permutations=n_permutations,
        ),
        "digits_mask_noise": BenchmarkSpec(
            name="digits_mask_noise",
            dataset_name="digits",
            protocol="mask_noise",
            n_permutations=n_permutations,
        ),
        "wine_shuffled": BenchmarkSpec(
            name="wine_shuffled",
            dataset_name="wine",
            protocol="shuffled",
        ),
        "breast_cancer_shuffled": BenchmarkSpec(
            name="breast_cancer_shuffled",
            dataset_name="breast_cancer",
            protocol="shuffled",
        ),
        "diabetes_regression": BenchmarkSpec(
            name="diabetes_regression",
            dataset_name="diabetes",
            protocol="shuffled",
            task_kind="regression",
        ),
        "dense_exact_zero": BenchmarkSpec(
            name="dense_exact_zero",
            dataset_name="dense_exact_zero",
            protocol="shuffled",
            task_kind="regression",
        ),
        "sparse_multilabel": BenchmarkSpec(
            name="sparse_multilabel",
            dataset_name="sparse_multilabel",
            protocol="shuffled",
            task_kind="multilabel",
        ),
        "temporal_delayed_history": BenchmarkSpec(
            name="temporal_delayed_history",
            dataset_name="temporal_delayed_history",
            protocol="sequential",
            task_kind="regression",
        ),
    }
    missing = sorted(set(names) - set(registry))
    if missing:
        raise ValueError(f"unknown benchmark(s): {', '.join(missing)}")
    return tuple(registry[name] for name in names)


def load_dataset(dataset_name: str, seed: int, train_fraction: float) -> LoadedDataset:
    """Load a bundled sklearn dataset and create a stratified train/test split."""
    if dataset_name == "dense_exact_zero":
        return make_dense_exact_zero_dataset(seed=seed)
    if dataset_name == "sparse_multilabel":
        return make_sparse_multilabel_dataset(seed=seed)
    if dataset_name == "temporal_delayed_history":
        return make_temporal_delayed_history_dataset(seed=seed)

    datasets = _import_sklearn_datasets()

    if dataset_name == "digits":
        raw = datasets.load_digits()
        x = np.asarray(raw.data, dtype=np.float32) / 16.0
        description = "sklearn load_digits: 8x8 handwritten digit images"
        task_kind = "multiclass"
    elif dataset_name == "wine":
        raw = datasets.load_wine()
        x = np.asarray(raw.data, dtype=np.float32)
        description = "sklearn load_wine: chemical analysis of wines"
        task_kind = "multiclass"
    elif dataset_name == "breast_cancer":
        raw = datasets.load_breast_cancer()
        x = np.asarray(raw.data, dtype=np.float32)
        description = "sklearn load_breast_cancer: diagnostic tabular dataset"
        task_kind = "multiclass"
    elif dataset_name == "diabetes":
        raw = datasets.load_diabetes()
        x = np.asarray(raw.data, dtype=np.float32)
        y_reg = np.asarray(raw.target, dtype=np.float32)
        train_idx, test_idx = regression_split_indices(
            n_examples=x.shape[0],
            seed=seed,
            train_fraction=train_fraction,
        )
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y_reg[train_idx]
        y_test = y_reg[test_idx]
        mean = x_train.mean(axis=0, keepdims=True)
        raw_std = x_train.std(axis=0, keepdims=True)
        std = np.where(raw_std < 1e-6, 1.0, raw_std)
        target_mean = float(y_train.mean())
        target_std = float(y_train.std() if y_train.std() >= 1e-6 else 1.0)
        x_train = ((x_train - mean) / std).astype(np.float32)
        x_test = ((x_test - mean) / std).astype(np.float32)
        y_train = ((y_train - target_mean) / target_std).astype(np.float32)[:, None]
        y_test = ((y_test - target_mean) / target_std).astype(np.float32)[:, None]
        return LoadedDataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata={
                "dataset": "sklearn.datasets.load_diabetes",
                "description": "sklearn load_diabetes: tabular regression dataset",
                "task_kind": "regression",
                "n_total": int(x.shape[0]),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "feature_dim": int(x_train.shape[1]),
                "n_classes": 1,
                "target_dim": 1,
                "train_fraction": float(train_fraction),
                "split_seed": int(seed),
            },
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    y = np.asarray(raw.target, dtype=np.int32)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for cls in classes:
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        n_train = int(round(train_fraction * len(cls_indices)))
        n_train = min(max(n_train, 1), len(cls_indices) - 1)
        train_indices.extend(cls_indices[:n_train].tolist())
        test_indices.extend(cls_indices[n_train:].tolist())

    train_idx = np.asarray(train_indices, dtype=np.int32)
    test_idx = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    metadata = {
        "dataset": f"sklearn.datasets.load_{dataset_name}",
        "description": description,
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": int(classes.shape[0]),
        "target_dim": int(classes.shape[0]),
        "task_kind": task_kind,
        "train_fraction": float(train_fraction),
        "split_seed": int(seed),
    }
    return LoadedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )


def regression_split_indices(
    n_examples: int,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shuffled train/test indices for non-stratified regression data."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_examples).astype(np.int32)
    n_train = int(round(train_fraction * n_examples))
    n_train = min(max(n_train, 1), n_examples - 1)
    return idx[:n_train], idx[n_train:]


def make_dense_exact_zero_dataset(seed: int) -> LoadedDataset:
    """Dense continuous inputs with an exactly zero vector target."""
    rng = np.random.default_rng(seed)
    n_train = 512
    n_test = 256
    feature_dim = 48
    x_train = rng.normal(size=(n_train, feature_dim)).astype(np.float32)
    x_test = rng.normal(size=(n_test, feature_dim)).astype(np.float32)
    y_train = np.zeros((n_train, 1), dtype=np.float32)
    y_test = np.zeros((n_test, 1), dtype=np.float32)
    return LoadedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "dataset": "generated.dense_exact_zero",
            "description": "Dense Gaussian observations with exact-zero regression targets.",
            "task_kind": "regression",
            "n_total": n_train + n_test,
            "n_train": n_train,
            "n_test": n_test,
            "feature_dim": feature_dim,
            "n_classes": 1,
            "target_dim": 1,
            "split_seed": int(seed),
        },
    )


def make_sparse_multilabel_dataset(seed: int) -> LoadedDataset:
    """Sparse binary features with sparse multi-hot targets."""
    rng = np.random.default_rng(seed)
    n_train = 768
    n_test = 256
    feature_dim = 96
    target_dim = 6
    density = 0.06
    weights = rng.normal(size=(feature_dim, target_dim)).astype(np.float32)

    def sample(n_examples: int) -> tuple[np.ndarray, np.ndarray]:
        x = (rng.random((n_examples, feature_dim)) < density).astype(np.float32)
        logits = x @ weights + 0.15 * rng.normal(size=(n_examples, target_dim))
        thresholds = np.quantile(logits, 0.72, axis=1, keepdims=True)
        y = (logits >= thresholds).astype(np.float32)
        empty = np.sum(y, axis=1) == 0.0
        y[empty, np.argmax(logits[empty], axis=1)] = 1.0
        return x, y.astype(np.float32)

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return LoadedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "dataset": "generated.sparse_multilabel",
            "description": "Sparse binary observations with sparse multi-hot targets.",
            "task_kind": "multilabel",
            "n_total": n_train + n_test,
            "n_train": n_train,
            "n_test": n_test,
            "feature_dim": feature_dim,
            "n_classes": target_dim,
            "target_dim": target_dim,
            "feature_density": density,
            "target_density": float(np.mean(y_train)),
            "split_seed": int(seed),
        },
    )


def make_temporal_delayed_history_dataset(seed: int) -> LoadedDataset:
    """Sequential stream whose target depends on delayed observation history."""
    rng = np.random.default_rng(seed)
    n_total = 1400
    n_train = 1000
    feature_dim = 5
    delay = 4
    raw = rng.normal(size=(n_total + delay, feature_dim)).astype(np.float32)
    signal = (
        0.8 * raw[delay:, 0]
        - 0.6 * raw[:-delay, 1]
        + 0.4 * np.sin(raw[:-delay, 2])
    )
    y = (signal / max(float(np.std(signal)), 1e-6)).astype(np.float32)[:, None]
    x = raw[delay:].astype(np.float32)
    return LoadedDataset(
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_test=x[n_train:],
        y_test=y[n_train:],
        metadata={
            "dataset": "generated.temporal_delayed_history",
            "description": "Sequential regression target depends on current and delayed inputs.",
            "task_kind": "regression",
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_total - n_train,
            "feature_dim": feature_dim,
            "n_classes": 1,
            "target_dim": 1,
            "delay": delay,
            "split_seed": int(seed),
        },
    )


def _epoch_indices(
    y_train: np.ndarray,
    rng: np.random.Generator,
    protocol: str,
) -> np.ndarray:
    """Generate one train-set epoch for shuffled or class-blocked protocols."""
    if protocol == "shuffled":
        return rng.permutation(len(y_train)).astype(np.int32)

    if protocol == "sequential":
        return np.arange(len(y_train), dtype=np.int32)

    if protocol == "class_blocked":
        parts: list[np.ndarray] = []
        classes = np.unique(y_train)
        for cls in rng.permutation(classes):
            cls_idx = np.flatnonzero(y_train == cls)
            rng.shuffle(cls_idx)
            parts.append(cls_idx)
        return np.concatenate(parts).astype(np.int32)

    raise ValueError(f"unsupported epoch protocol: {protocol}")


def make_online_sequence(
    dataset: LoadedDataset,
    spec: BenchmarkSpec,
    steps: int,
    seed: int,
    permutation_block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[np.ndarray, ...], dict[str, Any]]:
    """Create a finite online stream from train data."""
    rng = np.random.default_rng(seed)
    n_classes = int(dataset.metadata["n_classes"])
    feature_dim = int(dataset.metadata["feature_dim"])
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    feature_orders: tuple[np.ndarray, ...]

    if spec.protocol in {"shuffled", "class_blocked", "sequential"}:
        total = 0
        while total < steps:
            idx = _epoch_indices(dataset.y_train, rng, spec.protocol)
            observations.append(dataset.x_train[idx])
            labels.append(dataset.y_train[idx])
            total += len(idx)
        feature_orders = (np.arange(feature_dim, dtype=np.int32),)
    elif spec.protocol == "permuted":
        if spec.n_permutations < 2:
            raise ValueError("permuted protocol requires at least two permutations")
        if permutation_block_size <= 0:
            raise ValueError("permutation_block_size must be positive")
        perms = [np.arange(feature_dim, dtype=np.int32)]
        perms.extend(
            rng.permutation(feature_dim).astype(np.int32)
            for _ in range(spec.n_permutations - 1)
        )
        feature_orders = tuple(perms)
        total = 0
        block_idx = 0
        while total < steps:
            block_len = min(permutation_block_size, steps - total)
            replace = block_len > len(dataset.y_train)
            idx = rng.choice(len(dataset.y_train), size=block_len, replace=replace)
            perm = feature_orders[block_idx % len(feature_orders)]
            observations.append(dataset.x_train[idx][:, perm])
            labels.append(dataset.y_train[idx])
            total += block_len
            block_idx += 1
    elif spec.protocol == "mask_noise":
        if spec.n_permutations < 2:
            raise ValueError("mask_noise protocol requires at least two masks")
        if permutation_block_size <= 0:
            raise ValueError("permutation_block_size must be positive")
        keep_fraction = 0.5
        n_keep = max(1, int(round(feature_dim * keep_fraction)))
        masks = []
        for _ in range(spec.n_permutations):
            keep = rng.choice(feature_dim, size=n_keep, replace=False)
            mask = np.zeros(feature_dim, dtype=np.float32)
            mask[keep] = 1.0
            masks.append(mask)
        feature_orders = (np.arange(feature_dim, dtype=np.int32),)
        total = 0
        block_idx = 0
        while total < steps:
            block_len = min(permutation_block_size, steps - total)
            replace = block_len > len(dataset.y_train)
            idx = rng.choice(len(dataset.y_train), size=block_len, replace=replace)
            masked = dataset.x_train[idx] * masks[block_idx % len(masks)]
            masked = masked + rng.normal(0.0, 0.05, size=masked.shape).astype(np.float32)
            observations.append(masked.astype(np.float32))
            labels.append(dataset.y_train[idx])
            total += block_len
            block_idx += 1
    else:
        raise ValueError(f"unknown protocol: {spec.protocol}")

    obs = np.concatenate(observations, axis=0)[:steps].astype(np.float32)
    raw_targets = np.concatenate(labels, axis=0)[:steps]
    if spec.task_kind == "multiclass":
        label_arr = raw_targets.astype(np.int32)
        targets = np.eye(n_classes, dtype=np.float32)[label_arr]
    elif spec.task_kind == "multilabel":
        targets = raw_targets.astype(np.float32)
        label_arr = np.argmax(targets, axis=1).astype(np.int32)
    else:
        targets = raw_targets.astype(np.float32)
        if targets.ndim == 1:
            targets = targets[:, None]
        label_arr = np.zeros(targets.shape[0], dtype=np.int32)
    stream_meta = {
        "protocol": spec.protocol,
        "task_kind": spec.task_kind,
        "steps": int(steps),
        "sequence_seed": int(seed),
        "n_permutations": int(len(feature_orders)),
        "permutation_block_size": (
            int(permutation_block_size)
            if spec.protocol in {"permuted", "mask_noise"}
            else None
        ),
    }
    return (
        jnp.asarray(obs),
        jnp.asarray(targets),
        jnp.asarray(label_arr),
        feature_orders,
        stream_meta,
    )


def make_learner(
    method: str,
    task_kind: str,
    n_classes: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    perturbation_sigma: float,
    utility_decay: float,
    cbp_decay_rate: float,
    cbp_replacement_rate: float,
    cbp_maturity_threshold: int,
) -> Any:
    """Build one learner under matched architecture and update settings."""
    bounder = ObGDBounding(kappa=2.0)
    if method == "linear":
        return MultiHeadMLPLearner(
            n_heads=n_classes,
            hidden_sizes=(),
            step_size=step_size,
            bounder=bounder,
            sparsity=sparsity,
            use_layer_norm=True,
        )
    if method == "mlp":
        return MultiHeadMLPLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=bounder,
            sparsity=sparsity,
            use_layer_norm=True,
        )
    if method == "mlp_deep":
        return MultiHeadMLPLearner(
            n_heads=n_classes,
            hidden_sizes=(hidden_sizes[0], hidden_sizes[0]),
            step_size=step_size,
            bounder=bounder,
            sparsity=sparsity,
            use_layer_norm=True,
        )
    if method == "mlp_context":
        return MultiHeadMLPLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=bounder,
            sparsity=sparsity,
            use_layer_norm=True,
        )
    if method == "upgd":
        readout_mode = "softmax_ce" if task_kind == "multiclass" else "linear_mse"
        return UPGDLearner.step2_default(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            readout_mode=readout_mode,
        )
    if method == "upgd_wide":
        readout_mode = "softmax_ce" if task_kind == "multiclass" else "linear_mse"
        return UPGDLearner.step2_default(
            n_heads=n_classes,
            hidden_sizes=(hidden_sizes[0] * 2,),
            step_size=step_size,
            readout_mode=readout_mode,
        )
    if method == "upgd_fast":
        readout_mode = "softmax_ce" if task_kind == "multiclass" else "linear_mse"
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size * 0.6,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="target_structure",
            readout_mode=readout_mode,
        )
    if method == "upgd_mean":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size * 0.6,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
        )
    if method == "upgd_reg_k2":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
        )
    if method == "upgd_reg_noln":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=False,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
        )
    if method == "upgd_reg_deep":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=(hidden_sizes[0], hidden_sizes[0]),
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
        )
    if method == "upgd_reg_input":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
            readout_input_mode="hidden_plus_input",
        )
    if method == "upgd_context":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
        )
    if method == "upgd_context_input":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=ObGDBounding(kappa=2.0),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            utility_decay=utility_decay,
            perturbation_beta=2.0,
            perturbation_interval=1,
            loss_normalization="mean",
            readout_mode="linear_mse",
            readout_input_mode="hidden_plus_input",
        )
    if method == "upgd_reg_passthrough_deep":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=(hidden_sizes[0], hidden_sizes[0]),
            step_size=step_size * 0.6,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            perturbation_beta=2.0,
            perturbation_interval=16,
            perturbation_warmup_steps=200,
            perturbation_ramp_steps=400,
            utility_decay=utility_decay,
            loss_normalization="mean",
            readout_mode="linear_mse",
            readout_input_mode="hidden_plus_input",
            readout_head_normalization="hidden_norm",
            track_unit_utilities=False,
            track_gradient_history=False,
        )
    if method == "upgd_temporal_fast_passthrough":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size * 0.6,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="normal",
            perturbation_beta=2.0,
            perturbation_interval=1,
            utility_decay=utility_decay,
            loss_normalization="mean",
            head_step_size_multiplier=2.0,
            readout_mode="linear_mse",
            readout_input_mode="hidden_plus_input",
            readout_head_normalization="hidden_norm",
        )
    if method == "upgd_temporal_passthrough_no_mutation":
        return UPGDLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size * 0.6,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=sparsity,
            use_layer_norm=True,
            perturbation_sigma=0.0,
            utility_decay=utility_decay,
            loss_normalization="mean",
            head_step_size_multiplier=2.0,
            readout_mode="linear_mse",
            readout_input_mode="hidden_plus_input",
            readout_head_normalization="hidden_norm",
            track_unit_utilities=False,
            track_gradient_history=False,
        )
    if method == "cbp":
        return CBPMultiHeadMLPLearner(
            n_heads=n_classes,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            bounder=bounder,
            sparsity=sparsity,
            use_layer_norm=True,
            cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
                decay_rate=cbp_decay_rate,
                replacement_rate=cbp_replacement_rate,
                maturity_threshold=cbp_maturity_threshold,
                enabled=True,
            ),
        )
    raise ValueError(f"unknown method: {method}")


def run_online_stream(
    learner: Any,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    task_kind: str,
) -> tuple[Any, np.ndarray]:
    """Run one learner through one prequential online stream."""
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, label = inputs
        predictions = learner.predict(carry, obs)
        mse = jnp.mean((predictions - target) ** 2)
        if task_kind == "multilabel":
            correct = jnp.mean(((predictions > 0.5) == (target > 0.5)).astype(jnp.float32))
        elif task_kind == "regression":
            correct = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            correct = (jnp.argmax(predictions) == label).astype(jnp.float32)
        result = learner.update(carry, obs, target)
        return result.state, jnp.stack([mse, correct])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def method_uses_temporal_context(method: str) -> bool:
    """Whether a method receives causal handoff context features."""
    return method in {"mlp_context", "upgd_context", "upgd_context_input"}


def augment_temporal_context_np(observations: np.ndarray, decay: float = 0.9) -> np.ndarray:
    """Append causal previous-observation, EMA, and delta context features.

    The appended context at time ``t`` uses only observations strictly before
    ``t``. The current observation is included separately as the first block.
    """
    obs = observations.astype(np.float32, copy=False)
    prev = np.zeros_like(obs)
    ema = np.zeros_like(obs)
    running = np.zeros(obs.shape[1], dtype=np.float32)
    last = np.zeros(obs.shape[1], dtype=np.float32)
    for idx, row in enumerate(obs):
        prev[idx] = last
        ema[idx] = running
        running = decay * running + (1.0 - decay) * row
        last = row
    delta = obs - prev
    return np.concatenate([obs, prev, ema, delta], axis=1).astype(np.float32)


def evaluate_classifier(
    learner: Any,
    state: Any,
    dataset: LoadedDataset,
    feature_orders: tuple[np.ndarray, ...],
    method: str = "",
) -> dict[str, float]:
    """Evaluate a final state on held-out data under all stream transforms."""
    n_classes = int(dataset.metadata["n_classes"])
    task_kind = str(dataset.metadata.get("task_kind", "multiclass"))
    if task_kind == "multiclass":
        targets_np = np.eye(n_classes, dtype=np.float32)[dataset.y_test.astype(np.int32)]
        labels_np = dataset.y_test.astype(np.int32)
    else:
        targets_np = dataset.y_test.astype(np.float32)
        if targets_np.shape[1] > 1:
            labels_np = np.argmax(targets_np, axis=1).astype(np.int32)
        else:
            labels_np = np.zeros(targets_np.shape[0], dtype=np.int32)
    targets = jnp.asarray(targets_np)
    labels = jnp.asarray(labels_np)
    mse_values: list[float] = []
    accuracy_values: list[float] = []

    for order in feature_orders:
        observations_np = dataset.x_test[:, order].astype(np.float32)
        if method_uses_temporal_context(method):
            observations_np = augment_temporal_context_np(observations_np)
        observations = jnp.asarray(observations_np)
        preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        mse = jnp.mean((preds - targets) ** 2)
        if task_kind == "multilabel":
            accuracy = jnp.mean(((preds > 0.5) == (targets > 0.5)).astype(jnp.float32))
        elif task_kind == "regression":
            accuracy = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        mse_values.append(float(mse))
        accuracy_values.append(float(accuracy))

    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def summarize_curve(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize prequential per-step [mse, correct] metrics."""
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def stderr(values: np.ndarray) -> float:
    """Standard error of the mean."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def metric_summary(records: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    """Mean/stderr summary for every method and metric."""
    summary: dict[str, Any] = {}
    for method in methods:
        method_summary: dict[str, Any] = {}
        for metric in (*LOSS_METRICS, *ACCURACY_METRICS):
            values = np.asarray(
                [record["methods"][method][metric] for record in records],
                dtype=np.float64,
            )
            method_summary[metric] = {
                "mean": float(np.mean(values)),
                "stderr": stderr(values),
            }
        summary[method] = method_summary
    return summary


def paired_vs_mlp(
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    """Paired method-vs-MLP summaries for all non-MLP methods."""
    output: dict[str, Any] = {}
    if "mlp" not in methods:
        return output

    for method in methods:
        if method == "mlp":
            continue
        metric_rows: dict[str, Any] = {}
        for metric in LOSS_METRICS:
            mlp = np.asarray(
                [record["methods"]["mlp"][metric] for record in records],
                dtype=np.float64,
            )
            candidate = np.asarray(
                [record["methods"][method][metric] for record in records],
                dtype=np.float64,
            )
            diff = mlp - candidate
            sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
            metric_rows[metric] = {
                "metric": metric,
                "direction": "positive means method lower than MLP",
                "mlp_mean": float(np.mean(mlp)),
                "method_mean": float(np.mean(candidate)),
                "mlp_minus_method_mean": float(np.mean(diff)),
                "mlp_minus_method_stderr": stderr(diff),
                "wins_for_method": int(np.sum(diff > 0.0)),
                "wins_for_mlp": int(np.sum(diff < 0.0)),
                "ties": int(np.sum(diff == 0.0)),
                "n_seeds": int(diff.shape[0]),
                "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
            }
        for metric in ACCURACY_METRICS:
            mlp = np.asarray(
                [record["methods"]["mlp"][metric] for record in records],
                dtype=np.float64,
            )
            candidate = np.asarray(
                [record["methods"][method][metric] for record in records],
                dtype=np.float64,
            )
            diff = candidate - mlp
            sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
            metric_rows[metric] = {
                "metric": metric,
                "direction": "positive means method more accurate than MLP",
                "mlp_mean": float(np.mean(mlp)),
                "method_mean": float(np.mean(candidate)),
                "method_minus_mlp_mean": float(np.mean(diff)),
                "method_minus_mlp_stderr": stderr(diff),
                "wins_for_method": int(np.sum(diff > 0.0)),
                "wins_for_mlp": int(np.sum(diff < 0.0)),
                "ties": int(np.sum(diff == 0.0)),
                "n_seeds": int(diff.shape[0]),
                "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
            }
        output[method] = metric_rows
    return output


def best_by_metric(method_stats: dict[str, Any], methods: tuple[str, ...]) -> dict[str, str]:
    """Find the best mean method per metric."""
    best: dict[str, str] = {}
    for metric in LOSS_METRICS:
        best[metric] = min(methods, key=lambda method: method_stats[method][metric]["mean"])
    for metric in ACCURACY_METRICS:
        best[metric] = max(methods, key=lambda method: method_stats[method][metric]["mean"])
    return best


def run_benchmark(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    """Run all methods/seeds for one benchmark spec."""
    hidden_sizes = (args.hidden_size,)
    records: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] | None = None
    stream_meta: dict[str, Any] | None = None

    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        dataset = load_dataset(spec.dataset_name, seed, args.train_fraction)
        dataset_meta = dataset.metadata
        observations, targets, labels, feature_orders, stream_meta = make_online_sequence(
            dataset=dataset,
            spec=spec,
            steps=args.steps,
            seed=seed + 10_000,
            permutation_block_size=args.permutation_block_size,
        )
        n_classes = int(dataset.metadata["n_classes"])
        method_results: dict[str, Any] = {}

        for method_idx, method in enumerate(methods):
            learner = make_learner(
                method=method,
                task_kind=spec.task_kind,
                n_classes=n_classes,
                hidden_sizes=hidden_sizes,
                step_size=args.step_size,
                sparsity=args.sparsity,
                perturbation_sigma=args.perturbation_sigma,
                utility_decay=args.utility_decay,
                cbp_decay_rate=args.cbp_decay_rate,
                cbp_replacement_rate=args.cbp_replacement_rate,
                cbp_maturity_threshold=args.cbp_maturity_threshold,
            )
            key = jr.key(seed * 100 + method_idx)
            method_observations = observations
            if method_uses_temporal_context(method):
                method_observations = jnp.asarray(
                    augment_temporal_context_np(np.asarray(observations))
                )
            print(f"{spec.name} seed={seed}: running {method}")
            final_state, online_metrics = run_online_stream(
                learner=learner,
                key=key,
                observations=method_observations,
                targets=targets,
                labels=labels,
                task_kind=spec.task_kind,
            )
            summary = summarize_curve(online_metrics, args.final_window)
            summary.update(
                evaluate_classifier(
                    learner,
                    final_state,
                    dataset,
                    feature_orders,
                    method=method,
                )
            )
            method_results[method] = summary
            print(
                f"{spec.name} seed={seed} {method}: "
                f"final_mse={summary['final_window_mse']:.4f}, "
                f"final_acc={summary['final_window_accuracy']:.3f}, "
                f"test_acc={summary['test_accuracy']:.3f}"
            )

        records.append({"seed": seed, "methods": method_results})

    if dataset_meta is None or stream_meta is None:
        raise RuntimeError(f"benchmark {spec.name} produced no records")

    method_stats = metric_summary(records, methods)
    return {
        "spec": {
            "name": spec.name,
            "dataset_name": spec.dataset_name,
            "protocol": spec.protocol,
            "task_kind": spec.task_kind,
            "n_permutations": spec.n_permutations,
        },
        "dataset": dataset_meta,
        "stream": stream_meta,
        "records": records,
        "aggregate": {
            "method_stats": method_stats,
            "paired_vs_mlp": paired_vs_mlp(records, methods),
            "best_by_metric": best_by_metric(method_stats, methods),
        },
    }


def outcome_label(row: dict[str, Any]) -> str:
    """Compact paired outcome label."""
    mean = row.get("mlp_minus_method_mean", row.get("method_minus_mlp_mean", 0.0))
    wins = row["wins_for_method"]
    n_seeds = row["n_seeds"]
    if mean > 0.0 and wins > n_seeds / 2:
        return "beats MLP"
    if mean < 0.0 and row["wins_for_mlp"] > n_seeds / 2:
        return "loses to MLP"
    return "mixed/tied"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a markdown summary for quick review."""
    lines = [
        "# Worker S2C Step 2 External Benchmark Suite",
        "",
        "All benchmarks use bundled scikit-learn datasets only. The online "
        "protocol is prequential: predict, score, then update on the current "
        "example. Held-out test splits are never used for updates.",
        "",
        "## Command Configuration",
        "",
        "```json",
        json.dumps(results["config"], indent=2),
        "```",
        "",
        "## Per-Benchmark Aggregate Metrics",
        "",
    ]

    for bench_name, bench in results["benchmarks"].items():
        dataset = bench["dataset"]
        stream = bench["stream"]
        aggregate = bench["aggregate"]
        method_stats = aggregate["method_stats"]
        lines.extend(
            [
                f"### {bench_name}",
                "",
                (
                    f"Dataset: `{dataset['dataset']}`; protocol: "
                    f"`{stream['protocol']}`; train/test: "
                    f"{dataset['n_train']}/{dataset['n_test']}; "
                    f"features/classes: {dataset['feature_dim']}/{dataset['n_classes']}."
                    f" task_kind=`{dataset.get('task_kind', 'multiclass')}`."
                ),
                "",
                "| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, stats in method_stats.items():
            lines.append(
                f"| {method} | "
                f"{stats['final_window_mse']['mean']:.4f} +/- "
                f"{stats['final_window_mse']['stderr']:.4f} | "
                f"{stats['final_window_accuracy']['mean']:.4f} +/- "
                f"{stats['final_window_accuracy']['stderr']:.4f} | "
                f"{stats['test_mse']['mean']:.4f} +/- "
                f"{stats['test_mse']['stderr']:.4f} | "
                f"{stats['test_accuracy']['mean']:.4f} +/- "
                f"{stats['test_accuracy']['stderr']:.4f} |"
            )

        lines.extend(
            [
                "",
                "Paired vs MLP:",
                "",
                "| Method | Final-window MSE | Test accuracy |",
                "|---|---:|---:|",
            ]
        )
        paired = aggregate["paired_vs_mlp"]
        if not paired:
            lines.append("| n/a | n/a | n/a |")
        for method, rows in paired.items():
            final_row = rows["final_window_mse"]
            test_acc_row = rows["test_accuracy"]
            lines.append(
                f"| {method} | {outcome_label(final_row)} "
                f"({final_row['wins_for_method']}/{final_row['n_seeds']} wins, "
                f"diff={final_row['mlp_minus_method_mean']:+.4f}) | "
                f"{outcome_label(test_acc_row)} "
                f"({test_acc_row['wins_for_method']}/{test_acc_row['n_seeds']} wins, "
                f"diff={test_acc_row['method_minus_mlp_mean']:+.4f}) |"
            )
        lines.extend(["", f"Best by metric: `{aggregate['best_by_metric']}`", ""])

    lines.extend(["## Cross-Benchmark Reading", ""])
    for method in ("upgd", "cbp"):
        if method not in results["config"]["methods"]:
            continue
        wins = []
        losses = []
        mixed = []
        for bench_name, bench in results["benchmarks"].items():
            paired = bench["aggregate"]["paired_vs_mlp"].get(method)
            if paired is None:
                continue
            label = outcome_label(paired["final_window_mse"])
            if label == "beats MLP":
                wins.append(bench_name)
            elif label == "loses to MLP":
                losses.append(bench_name)
            else:
                mixed.append(bench_name)
        lines.append(
            f"- `{method}` final-window MSE vs MLP: wins={wins or 'none'}, "
            f"losses={losses or 'none'}, mixed/tied={mixed or 'none'}."
        )

    lines.extend(
        [
            "",
            "This suite is an external-data sanity check, not a replacement for "
            "the synthetic out-of-hypothesis-class Step 2 suite. A positive "
            "UPGD/CBP result here would be useful evidence of broader "
            "robustness; a negative result means the synthetic UPGD win should "
            "not be generalized without more work.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated command-line field."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-3)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--cbp-decay-rate", type=float, default=0.99)
    parser.add_argument("--cbp-replacement-rate", type=float, default=1e-4)
    parser.add_argument("--cbp-maturity-threshold", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--permutation-block-size", type=int, default=500)
    parser.add_argument(
        "--benchmarks",
        type=parse_csv,
        default=DEFAULT_BENCHMARKS,
        help=f"comma-separated subset of {DEFAULT_BENCHMARKS}",
    )
    parser.add_argument(
        "--methods",
        type=parse_csv,
        default=DEFAULT_METHODS,
        help=f"comma-separated subset of {ALLOWED_METHODS}",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/worker_s2c_external"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast 1-seed, 200-step subset for harness checks.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    unknown_methods = sorted(set(args.methods) - set(ALLOWED_METHODS))
    if unknown_methods:
        raise ValueError(f"unknown method(s): {', '.join(unknown_methods)}")
    if "mlp" not in args.methods:
        raise ValueError("--methods must include mlp so paired comparisons are defined")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 200
        args.n_seeds = 1
        args.final_window = 50
        args.benchmarks = ("digits_shuffled", "digits_permuted")
    validate_args(args)

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = tuple(args.methods)
    specs = benchmark_specs(tuple(args.benchmarks), args.n_permutations)
    benchmarks: dict[str, Any] = {}
    for spec in specs:
        benchmarks[spec.name] = run_benchmark(spec, args, methods)

    results = {
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "final_window": args.final_window,
            "hidden_size": args.hidden_size,
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "perturbation_sigma": args.perturbation_sigma,
            "utility_decay": args.utility_decay,
            "cbp_decay_rate": args.cbp_decay_rate,
            "cbp_replacement_rate": args.cbp_replacement_rate,
            "cbp_maturity_threshold": args.cbp_maturity_threshold,
            "n_permutations": args.n_permutations,
            "permutation_block_size": args.permutation_block_size,
            "benchmarks": list(args.benchmarks),
            "methods": list(methods),
            "output_dir": str(output_dir),
            "upgd_factory": "UPGDLearner.step2_default",
            "upgd_loss_normalization": "target_structure",
            "upgd_multiclass_readout_mode": "softmax_ce",
            "upgd_other_readout_mode": "linear_mse",
        },
        "benchmarks": benchmarks,
        "wall_clock_s": time.time() - t0,
        "evidence_level": "external_sklearn_online_prequential_suite",
    }

    json_path = output_dir / "external_suite_results.json"
    summary_path = output_dir / "external_suite_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(summary_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

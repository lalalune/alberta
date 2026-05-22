#!/usr/bin/env python3
"""Step 2 learned resource manager on harder stateful external streams.

This runner targets the remaining Step 2 gap after the strict universal
portfolio result: a learned resource manager and harder external continual
benchmarks.  Its default suite uses scikit-learn's bundled digits dataset, but
turns it into stateful streams that recur through external-data transformations:

* recurring pixel permutations,
* recurring feature masks with noise,
* class-blocked retention.

It also has an opt-in harder image regime:

* delayed contextual permutations over Fashion-MNIST from OpenML, with a
  no-network 28x28 sklearn-digits fallback for local smoke tests.

The learned manager is a contextual Hedge controller over resource policies:

* static fair MLP (no extra plasticity resource),
* low-noise UPGD,
* high-noise UPGD,
* CBP replacement.

At each time step all policies make a pre-update prediction, the manager emits a
causal allocation from its prior state, the mixture is scored, all policies
update, and then the manager updates its allocation from the observed policy
losses.  Held-out evaluation averages across all recurrent external states.
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
    LearnedResourceManager,
    MultiHeadMLPLearner,
    ObGDBounding,
    UPGDLearner,
)

N_CLASSES = 10
POLICY_NAMES = ("mlp_static", "upgd_low", "upgd_high", "cbp_replace")
MANAGER_NAMES = ("resource_manager", "resource_manager_retention")
METHOD_NAMES = (*POLICY_NAMES, *MANAGER_NAMES)
VALID_BENCHMARKS = (
    "digits_recurrent_permutation",
    "digits_recurrent_mask_noise",
    "digits_class_blocked_retention",
    "external_delayed_contextual_permutation",
)
DEFAULT_BENCHMARKS = (
    "digits_recurrent_permutation",
    "digits_recurrent_mask_noise",
    "digits_class_blocked_retention",
)
DEFAULT_OUTPUT_DIR = Path("outputs/step2_resource_manager_stateful_external")

MIX_MSE_COL = 0
MIX_ACC_COL = 1
RETENTION_MIX_MSE_COL = 2
RETENTION_MIX_ACC_COL = 3
POLICY_MSE_START = 4
POLICY_ACC_START = POLICY_MSE_START + len(POLICY_NAMES)
WEIGHT_START = POLICY_ACC_START + len(POLICY_NAMES)
RETENTION_WEIGHT_START = WEIGHT_START + len(POLICY_NAMES)


@dataclass(frozen=True)
class DigitsData:
    """Train/test split of a ten-class image classification dataset."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StatefulDigitsStream:
    """Materialized online stream plus held-out state views."""

    observations: jax.Array
    targets: jax.Array
    labels: jax.Array
    state_ids: jax.Array
    test_views: jax.Array
    test_labels: jax.Array
    metadata: dict[str, Any]

    @property
    def n_contexts(self) -> int:
        return int(self.test_views.shape[0])


def load_digits_data(seed: int, train_fraction: float) -> DigitsData:
    """Load and standardize sklearn digits from train-split statistics only."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required; install with `pip install -e '.[external]'`. "
            "The digits dataset is bundled with scikit-learn and needs no network."
        )
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in range(N_CLASSES):
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
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    return DigitsData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "dataset": "sklearn.datasets.load_digits",
            "n_total": int(x.shape[0]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "train_fraction": float(train_fraction),
            "split_seed": int(seed),
        },
    )


def _standardize_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features from train-split statistics only."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def _stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_fraction: float,
    sample_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a balanced-ish split, optionally capping examples per class."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    per_class_limit = None
    if sample_limit is not None and sample_limit > 0:
        per_class_limit = max(2, int(math.ceil(sample_limit / N_CLASSES)))

    for cls in range(N_CLASSES):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        if per_class_limit is not None:
            cls_idx = cls_idx[:per_class_limit]
        n_train = int(round(train_fraction * len(cls_idx)))
        n_train = min(max(n_train, 1), len(cls_idx) - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    train = np.asarray(train_idx, dtype=np.int32)
    test = np.asarray(test_idx, dtype=np.int32)
    rng.shuffle(train)
    rng.shuffle(test)
    return x[train], y[train], x[test], y[test]


def _expand_digits_to_28x28(x: np.ndarray) -> np.ndarray:
    """Nearest-neighbor expand sklearn 8x8 digits into centered 28x28 images."""
    images = x.reshape((-1, 8, 8))
    expanded = np.kron(images, np.ones((3, 3), dtype=np.float32))
    padded = np.pad(expanded, ((0, 0), (2, 2), (2, 2)), mode="constant")
    return padded.reshape((x.shape[0], 28 * 28)).astype(np.float32)


def _load_expanded_digits_fallback(
    seed: int,
    train_fraction: float,
    sample_limit: int,
    requested_source: str,
    fallback_reason: str,
) -> DigitsData:
    """Load a local 28x28 image fallback for no-network smoke tests."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for the local external fallback; install "
            "with `pip install -e '.[external]'`."
        )
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = _expand_digits_to_28x28(np.asarray(digits.data, dtype=np.float32) / 16.0)
    y = np.asarray(digits.target, dtype=np.int32)
    x_train, y_train, x_test, y_test = _stratified_train_test_split(
        x,
        y,
        seed=seed,
        train_fraction=train_fraction,
        sample_limit=sample_limit,
    )
    x_train, x_test = _standardize_split(x_train, x_test)
    return DigitsData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "dataset": "sklearn.datasets.load_digits_expanded_28x28",
            "requested_external_source": requested_source,
            "used_fallback": True,
            "fallback_reason": fallback_reason,
            "n_total": int(x.shape[0]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "train_fraction": float(train_fraction),
            "split_seed": int(seed),
            "sample_limit": int(sample_limit),
        },
    )


def load_external_image_data(
    seed: int,
    train_fraction: float,
    source: str,
    allow_openml_download: bool,
    sample_limit: int,
) -> DigitsData:
    """Load a harder ten-class image dataset, falling back locally if needed."""
    if source not in {"digits_28x28_fallback", "openml_fashion_mnist"}:
        raise ValueError(f"unknown external image source {source!r}")
    if source == "digits_28x28_fallback":
        return _load_expanded_digits_fallback(
            seed=seed,
            train_fraction=train_fraction,
            sample_limit=sample_limit,
            requested_source=source,
            fallback_reason="explicit fallback source",
        )
    if not allow_openml_download:
        return _load_expanded_digits_fallback(
            seed=seed,
            train_fraction=train_fraction,
            sample_limit=sample_limit,
            requested_source=source,
            fallback_reason="OpenML download disabled",
        )

    try:
        from sklearn.datasets import fetch_openml

        dataset = fetch_openml(
            name="Fashion-MNIST",
            version=1,
            as_frame=False,
        )
        x = np.asarray(dataset.data, dtype=np.float32) / 255.0
        y = np.asarray(dataset.target, dtype=np.int32)
        keep = np.isin(y, np.arange(N_CLASSES))
        x = x[keep]
        y = y[keep]
        x_train, y_train, x_test, y_test = _stratified_train_test_split(
            x,
            y,
            seed=seed,
            train_fraction=train_fraction,
            sample_limit=sample_limit,
        )
        x_train, x_test = _standardize_split(x_train, x_test)
        return DigitsData(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata={
                "dataset": "OpenML Fashion-MNIST",
                "requested_external_source": source,
                "used_fallback": False,
                "openml_name": "Fashion-MNIST",
                "openml_version": 1,
                "n_total": int(x.shape[0]),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "feature_dim": int(x_train.shape[1]),
                "n_classes": N_CLASSES,
                "train_fraction": float(train_fraction),
                "split_seed": int(seed),
                "sample_limit": int(sample_limit),
            },
        )
    except Exception as exc:  # pragma: no cover - network/service dependent
        return _load_expanded_digits_fallback(
            seed=seed,
            train_fraction=train_fraction,
            sample_limit=sample_limit,
            requested_source=source,
            fallback_reason=f"OpenML load failed: {type(exc).__name__}: {exc}",
        )


def _sample_block_indices(
    rng: np.random.Generator,
    n_train: int,
    block_size: int,
) -> np.ndarray:
    replace = block_size > n_train
    return rng.choice(n_train, size=block_size, replace=replace).astype(np.int32)


def make_recurrent_permutation_stream(
    data: DigitsData,
    steps: int,
    seed: int,
    n_states: int,
    block_size: int,
) -> StatefulDigitsStream:
    """Digits stream cycling through recurring pixel permutations."""
    rng = np.random.default_rng(seed)
    feature_dim = data.x_train.shape[1]
    permutations = [np.arange(feature_dim, dtype=np.int32)]
    permutations.extend(rng.permutation(feature_dim).astype(np.int32) for _ in range(n_states - 1))

    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    total = 0
    block_idx = 0
    while total < steps:
        state_id = block_idx % n_states
        length = min(block_size, steps - total)
        idx = _sample_block_indices(rng, data.x_train.shape[0], length)
        obs_parts.append(data.x_train[idx][:, permutations[state_id]])
        label_parts.append(data.y_train[idx])
        state_parts.append(np.full(length, state_id, dtype=np.int32))
        total += length
        block_idx += 1

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    state_ids = np.concatenate(state_parts, axis=0).astype(np.int32)
    test_views = np.stack([data.x_test[:, perm] for perm in permutations], axis=0)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return StatefulDigitsStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        state_ids=jnp.asarray(state_ids),
        test_views=jnp.asarray(test_views.astype(np.float32)),
        test_labels=jnp.asarray(data.y_test.astype(np.int32)),
        metadata={
            "benchmark": "digits_recurrent_permutation",
            "description": (
                "Recurring pixel-permutation states; held-out evaluation averages "
                "over all recurrent permutations."
            ),
            "n_states": int(n_states),
            "block_size": int(block_size),
            "steps": int(steps),
        },
    )


def make_recurrent_mask_noise_stream(
    data: DigitsData,
    steps: int,
    seed: int,
    n_states: int,
    block_size: int,
    keep_fraction: float,
    noise_std: float,
) -> StatefulDigitsStream:
    """Digits stream cycling through recurring feature masks plus noise."""
    rng = np.random.default_rng(seed)
    feature_dim = data.x_train.shape[1]
    n_keep = max(1, min(feature_dim, int(round(feature_dim * keep_fraction))))
    masks = []
    for _ in range(n_states):
        keep = rng.choice(feature_dim, size=n_keep, replace=False)
        mask = np.zeros(feature_dim, dtype=np.float32)
        mask[keep] = 1.0
        masks.append(mask)

    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    total = 0
    block_idx = 0
    while total < steps:
        state_id = block_idx % n_states
        length = min(block_size, steps - total)
        idx = _sample_block_indices(rng, data.x_train.shape[0], length)
        obs = data.x_train[idx] * masks[state_id]
        if noise_std > 0.0:
            obs = obs + rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)
            obs = obs * masks[state_id]
        obs_parts.append(obs)
        label_parts.append(data.y_train[idx])
        state_parts.append(np.full(length, state_id, dtype=np.int32))
        total += length
        block_idx += 1

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    state_ids = np.concatenate(state_parts, axis=0).astype(np.int32)
    test_views = np.stack([data.x_test * mask for mask in masks], axis=0)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return StatefulDigitsStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        state_ids=jnp.asarray(state_ids),
        test_views=jnp.asarray(test_views.astype(np.float32)),
        test_labels=jnp.asarray(data.y_test.astype(np.int32)),
        metadata={
            "benchmark": "digits_recurrent_mask_noise",
            "description": (
                "Recurring feature-mask states with online noise; held-out "
                "evaluation averages over the recurring masks."
            ),
            "n_states": int(n_states),
            "block_size": int(block_size),
            "keep_fraction": float(keep_fraction),
            "noise_std": float(noise_std),
            "steps": int(steps),
        },
    )


def make_class_blocked_retention_stream(
    data: DigitsData,
    steps: int,
    seed: int,
) -> StatefulDigitsStream:
    """Class-blocked stream with balanced held-out retention evaluation."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(data.y_train == cls) for cls in range(N_CLASSES)]
    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    total = 0
    while total < steps:
        for cls in rng.permutation(N_CLASSES):
            cls_idx = class_indices[int(cls)].copy()
            rng.shuffle(cls_idx)
            length = min(len(cls_idx), steps - total)
            obs_parts.append(data.x_train[cls_idx[:length]])
            label_parts.append(data.y_train[cls_idx[:length]])
            total += length
            if total >= steps:
                break

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    state_ids = np.zeros(labels.shape[0], dtype=np.int32)
    return StatefulDigitsStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        state_ids=jnp.asarray(state_ids),
        test_views=jnp.asarray(data.x_test[None, :, :].astype(np.float32)),
        test_labels=jnp.asarray(data.y_test.astype(np.int32)),
        metadata={
            "benchmark": "digits_class_blocked_retention",
            "description": (
                "Digit-class blocks create current-block specialization pressure; "
                "held-out evaluation is balanced over all classes."
            ),
            "n_states": 1,
            "steps": int(steps),
        },
    )


def make_delayed_contextual_permutation_stream(
    data: DigitsData,
    steps: int,
    seed: int,
    n_states: int,
    block_size: int,
    context_delay_blocks: int,
) -> StatefulDigitsStream:
    """Image stream with recurring permutations and delayed manager context.

    The image transformation follows the current hidden block state, but the
    resource manager receives a context id delayed by whole blocks.  This makes
    allocation harder than the ordinary recurrent-permutation digits stream
    while keeping held-out views interpretable by true recurrent state.
    """
    rng = np.random.default_rng(seed)
    feature_dim = data.x_train.shape[1]
    permutations = [np.arange(feature_dim, dtype=np.int32)]
    permutations.extend(rng.permutation(feature_dim).astype(np.int32) for _ in range(n_states - 1))

    obs_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    manager_state_parts: list[np.ndarray] = []
    true_state_parts: list[np.ndarray] = []
    total = 0
    block_idx = 0
    while total < steps:
        true_state_id = block_idx % n_states
        manager_state_id = max(block_idx - context_delay_blocks, 0) % n_states
        length = min(block_size, steps - total)
        idx = _sample_block_indices(rng, data.x_train.shape[0], length)
        obs_parts.append(data.x_train[idx][:, permutations[true_state_id]])
        label_parts.append(data.y_train[idx])
        manager_state_parts.append(np.full(length, manager_state_id, dtype=np.int32))
        true_state_parts.append(np.full(length, true_state_id, dtype=np.int32))
        total += length
        block_idx += 1

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int32)
    manager_state_ids = np.concatenate(manager_state_parts, axis=0).astype(np.int32)
    true_state_ids = np.concatenate(true_state_parts, axis=0).astype(np.int32)
    test_views = np.stack([data.x_test[:, perm] for perm in permutations], axis=0)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return StatefulDigitsStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        state_ids=jnp.asarray(manager_state_ids),
        test_views=jnp.asarray(test_views.astype(np.float32)),
        test_labels=jnp.asarray(data.y_test.astype(np.int32)),
        metadata={
            "benchmark": "external_delayed_contextual_permutation",
            "description": (
                "Fashion-MNIST-style 28x28 image stream when OpenML is enabled, "
                "otherwise expanded sklearn digits fallback; recurring pixel "
                "permutations use true hidden states while manager context ids "
                "are delayed by whole blocks."
            ),
            "n_states": int(n_states),
            "block_size": int(block_size),
            "context_delay_blocks": int(context_delay_blocks),
            "steps": int(steps),
            "state_id_semantics": "manager receives delayed context ids",
            "true_state_ids_head": true_state_ids[: min(20, true_state_ids.shape[0])].tolist(),
            "manager_state_ids_head": manager_state_ids[
                : min(20, manager_state_ids.shape[0])
            ].tolist(),
        },
    )


def make_stream(
    benchmark: str,
    data: DigitsData,
    args: argparse.Namespace,
    seed: int,
) -> StatefulDigitsStream:
    """Create a benchmark stream by name."""
    if benchmark == "digits_recurrent_permutation":
        return make_recurrent_permutation_stream(
            data=data,
            steps=args.steps,
            seed=seed,
            n_states=args.n_states,
            block_size=args.block_size,
        )
    if benchmark == "digits_recurrent_mask_noise":
        return make_recurrent_mask_noise_stream(
            data=data,
            steps=args.steps,
            seed=seed,
            n_states=args.n_states,
            block_size=args.block_size,
            keep_fraction=args.mask_keep_fraction,
            noise_std=args.noise_std,
        )
    if benchmark == "digits_class_blocked_retention":
        return make_class_blocked_retention_stream(data=data, steps=args.steps, seed=seed)
    if benchmark == "external_delayed_contextual_permutation":
        return make_delayed_contextual_permutation_stream(
            data=data,
            steps=args.steps,
            seed=seed,
            n_states=args.n_states,
            block_size=args.block_size,
            context_delay_blocks=args.context_delay_blocks,
        )
    raise ValueError(f"unknown benchmark {benchmark!r}")


def make_policy_learners(
    hidden_size: int,
    step_size: float,
    sparsity: float,
    low_sigma: float,
    high_sigma: float,
    cbp_decay: float,
    cbp_replacement_rate: float,
    cbp_maturity: int,
) -> tuple[Any, Any, Any, Any]:
    """Create the resource-policy learners."""
    bounder = ObGDBounding(kappa=2.0)
    mlp = MultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=bounder,
        sparsity=sparsity,
        use_layer_norm=True,
    )
    upgd_low = UPGDLearner(
        n_heads=N_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=bounder,
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=low_sigma,
    )
    upgd_high = UPGDLearner(
        n_heads=N_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=bounder,
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=high_sigma,
    )
    cbp = CBPMultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=bounder,
        sparsity=sparsity,
        use_layer_norm=True,
        cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
            decay_rate=cbp_decay,
            replacement_rate=cbp_replacement_rate,
            maturity_threshold=cbp_maturity,
            enabled=True,
        ),
    )
    return mlp, upgd_low, upgd_high, cbp


def _pred_metrics(
    predictions: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mse = jnp.mean((predictions - targets[None, :]) ** 2, axis=1)
    accuracy = (jnp.argmax(predictions, axis=1) == labels).astype(jnp.float32)
    return mse, accuracy


def run_resource_manager_stream(
    stream: StatefulDigitsStream,
    args: argparse.Namespace,
    key: jax.Array,
) -> tuple[
    tuple[Any, Any, Any, Any],
    Any,
    LearnedResourceManager,
    Any,
    LearnedResourceManager,
    Any,
    np.ndarray,
]:
    """Run all resource policies plus the learned manager through one stream."""
    learners = make_policy_learners(
        hidden_size=args.hidden_size,
        step_size=args.step_size,
        sparsity=args.sparsity,
        low_sigma=args.low_sigma,
        high_sigma=args.high_sigma,
        cbp_decay=args.cbp_decay,
        cbp_replacement_rate=args.cbp_replacement_rate,
        cbp_maturity=args.cbp_maturity,
    )
    keys = jr.split(key, len(POLICY_NAMES) + 1)
    states = tuple(
        learner.init(stream.observations.shape[1], keys[idx])
        for idx, learner in enumerate(learners)
    )
    tracking_manager = LearnedResourceManager(
        n_actions=len(POLICY_NAMES),
        n_contexts=stream.n_contexts,
        learning_rate=args.manager_learning_rate,
        discount=args.manager_discount,
        exploration=args.manager_exploration,
        loss_decay=args.manager_loss_decay,
        cost_weight=args.resource_cost_weight,
    )
    tracking_manager_state = tracking_manager.init()
    retention_manager = LearnedResourceManager(
        n_actions=len(POLICY_NAMES),
        n_contexts=stream.n_contexts,
        learning_rate=args.retention_manager_learning_rate,
        discount=args.retention_manager_discount,
        exploration=args.manager_exploration,
        loss_decay=args.manager_loss_decay,
        cost_weight=args.resource_cost_weight,
    )
    retention_manager_state = retention_manager.init()
    prototypes = jnp.zeros(
        (stream.n_contexts, N_CLASSES, stream.observations.shape[1]),
        dtype=jnp.float32,
    )
    prototype_counts = jnp.zeros((stream.n_contexts, N_CLASSES), dtype=jnp.float32)
    resource_costs = jnp.asarray(
        [0.0, args.low_sigma_cost, args.high_sigma_cost, args.cbp_cost],
        dtype=jnp.float32,
    )
    mlp, upgd_low, upgd_high, cbp = learners

    def step_fn(
        carry: tuple[Any, Any, Any, Any, Any, Any, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[Any, Any, Any, Any, Any, Any, jax.Array, jax.Array], jax.Array]:
        (
            mlp_s,
            upgd_low_s,
            upgd_high_s,
            cbp_s,
            tracking_manager_s,
            retention_manager_s,
            prototype_s,
            prototype_count_s,
        ) = carry
        obs, target, label, context_id = inputs
        predictions = jnp.stack(
            [
                mlp.predict(mlp_s, obs),
                upgd_low.predict(upgd_low_s, obs),
                upgd_high.predict(upgd_high_s, obs),
                cbp.predict(cbp_s, obs),
            ],
            axis=0,
        )
        weights = tracking_manager.weights(tracking_manager_s, context_id)
        retention_weights = retention_manager.weights(retention_manager_s, context_id)
        mixture_pred = jnp.sum(weights[:, None] * predictions, axis=0)
        retention_mixture_pred = jnp.sum(retention_weights[:, None] * predictions, axis=0)
        mixture_mse = jnp.mean((mixture_pred - target) ** 2)
        mixture_acc = (jnp.argmax(mixture_pred) == label).astype(jnp.float32)
        retention_mixture_mse = jnp.mean((retention_mixture_pred - target) ** 2)
        retention_mixture_acc = (jnp.argmax(retention_mixture_pred) == label).astype(
            jnp.float32
        )
        policy_mse, policy_acc = _pred_metrics(predictions, target, label)

        context_prototypes = prototype_s[context_id]
        context_counts = prototype_count_s[context_id]
        seen = context_counts > 0.0
        seen_count = jnp.maximum(jnp.sum(seen.astype(jnp.float32)), 1.0)
        prototype_targets = jnp.eye(N_CLASSES, dtype=jnp.float32)

        def prototype_loss(learner: Any, learner_state: Any) -> jax.Array:
            proto_preds = jax.vmap(lambda proto: learner.predict(learner_state, proto))(
                context_prototypes
            )
            per_class = jnp.mean((proto_preds - prototype_targets) ** 2, axis=1)
            return jnp.sum(jnp.where(seen, per_class, 0.0)) / seen_count

        prototype_losses = jnp.stack(
            [
                prototype_loss(mlp, mlp_s),
                prototype_loss(upgd_low, upgd_low_s),
                prototype_loss(upgd_high, upgd_high_s),
                prototype_loss(cbp, cbp_s),
            ],
            axis=0,
        )

        mlp_result = mlp.update(mlp_s, obs, target)
        upgd_low_result = upgd_low.update(upgd_low_s, obs, target)
        upgd_high_result = upgd_high.update(upgd_high_s, obs, target)
        cbp_result = cbp.update(cbp_s, obs, target)
        tracking_manager_result = tracking_manager.update(
            tracking_manager_s,
            policy_mse,
            context_id=context_id,
            resource_costs=resource_costs,
        )
        retention_manager_result = retention_manager.update(
            retention_manager_s,
            prototype_losses,
            context_id=context_id,
            resource_costs=resource_costs,
        )

        old_count = prototype_count_s[context_id, label]
        rate = 1.0 / (old_count + 1.0)
        old_proto = prototype_s[context_id, label]
        new_proto = old_proto + rate * (obs - old_proto)
        new_prototypes = prototype_s.at[context_id, label].set(new_proto)
        new_prototype_counts = prototype_count_s.at[context_id, label].add(1.0)

        metrics = jnp.concatenate(
            [
                jnp.asarray(
                    [
                        mixture_mse,
                        mixture_acc,
                        retention_mixture_mse,
                        retention_mixture_acc,
                    ],
                    dtype=jnp.float32,
                ),
                policy_mse,
                policy_acc,
                weights,
                retention_weights,
            ],
            axis=0,
        )
        return (
            mlp_result.state,
            upgd_low_result.state,
            upgd_high_result.state,
            cbp_result.state,
            tracking_manager_result.state,
            retention_manager_result.state,
            new_prototypes,
            new_prototype_counts,
        ), metrics

    final_carry, metrics = jax.lax.scan(
        step_fn,
        (
            *states,
            tracking_manager_state,
            retention_manager_state,
            prototypes,
            prototype_counts,
        ),
        (stream.observations, stream.targets, stream.labels, stream.state_ids),
    )
    metrics.block_until_ready()
    final_states = final_carry[: len(POLICY_NAMES)]
    final_tracking_manager_state = final_carry[len(POLICY_NAMES)]
    final_retention_manager_state = final_carry[len(POLICY_NAMES) + 1]
    return (
        learners,
        final_states,
        tracking_manager,
        final_tracking_manager_state,
        retention_manager,
        final_retention_manager_state,
        np.asarray(metrics),
    )


def evaluate_resource_manager(
    learners: tuple[Any, Any, Any, Any],
    states: tuple[Any, Any, Any, Any],
    tracking_manager: LearnedResourceManager,
    tracking_manager_state: Any,
    retention_manager: LearnedResourceManager,
    retention_manager_state: Any,
    stream: StatefulDigitsStream,
) -> dict[str, dict[str, float]]:
    """Evaluate resource policies and manager on held-out recurrent state views."""
    labels = stream.test_labels
    targets = jnp.eye(N_CLASSES, dtype=jnp.float32)[labels]
    policy_mse_values: dict[str, list[float]] = {name: [] for name in POLICY_NAMES}
    policy_acc_values: dict[str, list[float]] = {name: [] for name in POLICY_NAMES}
    tracking_manager_mse_values: list[float] = []
    tracking_manager_acc_values: list[float] = []
    retention_manager_mse_values: list[float] = []
    retention_manager_acc_values: list[float] = []

    for context_id in range(stream.n_contexts):
        observations = stream.test_views[context_id]
        policy_preds = jnp.stack(
            [
                jax.vmap(lambda obs, learner=learner, state=state: learner.predict(state, obs))(
                    observations
                )
                for learner, state in zip(learners, states, strict=True)
            ],
            axis=0,
        )
        tracking_weights = tracking_manager.weights(tracking_manager_state, context_id)
        retention_weights = retention_manager.weights(retention_manager_state, context_id)
        tracking_manager_preds = jnp.sum(tracking_weights[:, None, None] * policy_preds, axis=0)
        retention_manager_preds = jnp.sum(
            retention_weights[:, None, None] * policy_preds,
            axis=0,
        )
        for policy_idx, name in enumerate(POLICY_NAMES):
            preds = policy_preds[policy_idx]
            mse = jnp.mean((preds - targets) ** 2)
            acc = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
            mse.block_until_ready()
            policy_mse_values[name].append(float(mse))
            policy_acc_values[name].append(float(acc))

        tracking_mse = jnp.mean((tracking_manager_preds - targets) ** 2)
        tracking_acc = jnp.mean(
            (jnp.argmax(tracking_manager_preds, axis=1) == labels).astype(jnp.float32)
        )
        retention_mse = jnp.mean((retention_manager_preds - targets) ** 2)
        retention_acc = jnp.mean(
            (jnp.argmax(retention_manager_preds, axis=1) == labels).astype(jnp.float32)
        )
        tracking_mse.block_until_ready()
        retention_mse.block_until_ready()
        tracking_manager_mse_values.append(float(tracking_mse))
        tracking_manager_acc_values.append(float(tracking_acc))
        retention_manager_mse_values.append(float(retention_mse))
        retention_manager_acc_values.append(float(retention_acc))

    out: dict[str, dict[str, float]] = {}
    for name in POLICY_NAMES:
        out[name] = {
            "test_mse": float(np.mean(policy_mse_values[name])),
            "test_accuracy": float(np.mean(policy_acc_values[name])),
        }
    out["resource_manager"] = {
        "test_mse": float(np.mean(tracking_manager_mse_values)),
        "test_accuracy": float(np.mean(tracking_manager_acc_values)),
    }
    out["resource_manager_retention"] = {
        "test_mse": float(np.mean(retention_manager_mse_values)),
        "test_accuracy": float(np.mean(retention_manager_acc_values)),
    }
    return out


def summarize_metrics(metrics: np.ndarray, final_window: int) -> dict[str, dict[str, float]]:
    """Summarize online curves for policies and manager."""
    window = min(final_window, metrics.shape[0])
    out: dict[str, dict[str, float]] = {
        "resource_manager": {
            "online_mean_mse": float(np.mean(metrics[:, MIX_MSE_COL])),
            "online_mean_accuracy": float(np.mean(metrics[:, MIX_ACC_COL])),
            "final_window_mse": float(np.mean(metrics[-window:, MIX_MSE_COL])),
            "final_window_accuracy": float(np.mean(metrics[-window:, MIX_ACC_COL])),
        },
        "resource_manager_retention": {
            "online_mean_mse": float(np.mean(metrics[:, RETENTION_MIX_MSE_COL])),
            "online_mean_accuracy": float(np.mean(metrics[:, RETENTION_MIX_ACC_COL])),
            "final_window_mse": float(np.mean(metrics[-window:, RETENTION_MIX_MSE_COL])),
            "final_window_accuracy": float(np.mean(metrics[-window:, RETENTION_MIX_ACC_COL])),
        }
    }
    for idx, name in enumerate(POLICY_NAMES):
        mse_col = POLICY_MSE_START + idx
        acc_col = POLICY_ACC_START + idx
        out[name] = {
            "online_mean_mse": float(np.mean(metrics[:, mse_col])),
            "online_mean_accuracy": float(np.mean(metrics[:, acc_col])),
            "final_window_mse": float(np.mean(metrics[-window:, mse_col])),
            "final_window_accuracy": float(np.mean(metrics[-window:, acc_col])),
        }
    return out


def manager_weight_summary(metrics: np.ndarray, final_window: int) -> dict[str, Any]:
    """Summarize learned allocation weights."""
    window = min(final_window, metrics.shape[0])
    weights = metrics[:, WEIGHT_START : WEIGHT_START + len(POLICY_NAMES)]
    retention_weights = metrics[
        :,
        RETENTION_WEIGHT_START : RETENTION_WEIGHT_START + len(POLICY_NAMES),
    ]
    final_weights = weights[-window:]
    final_retention_weights = retention_weights[-window:]
    return {
        "tracking_mean_weights": {
            name: float(np.mean(weights[:, idx]))
            for idx, name in enumerate(POLICY_NAMES)
        },
        "tracking_final_window_mean_weights": {
            name: float(np.mean(final_weights[:, idx]))
            for idx, name in enumerate(POLICY_NAMES)
        },
        "tracking_last_weights": {
            name: float(weights[-1, idx]) for idx, name in enumerate(POLICY_NAMES)
        },
        "retention_mean_weights": {
            name: float(np.mean(retention_weights[:, idx]))
            for idx, name in enumerate(POLICY_NAMES)
        },
        "retention_final_window_mean_weights": {
            name: float(np.mean(final_retention_weights[:, idx]))
            for idx, name in enumerate(POLICY_NAMES)
        },
        "retention_last_weights": {
            name: float(retention_weights[-1, idx])
            for idx, name in enumerate(POLICY_NAMES)
        },
    }


def stderr(values: np.ndarray) -> float:
    """Standard error."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def aggregate_method_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate method metrics over seeds."""
    metrics = (
        "online_mean_mse",
        "final_window_mse",
        "test_mse",
        "online_mean_accuracy",
        "final_window_accuracy",
        "test_accuracy",
    )
    out: dict[str, Any] = {}
    for method in METHOD_NAMES:
        rows: dict[str, Any] = {}
        for metric in metrics:
            values = np.asarray(
                [record["methods"][method][metric] for record in records],
                dtype=np.float64,
            )
            rows[metric] = {"mean": float(np.mean(values)), "stderr": stderr(values)}
        out[method] = rows
    return out


def paired_resource_manager_vs_method(
    records: list[dict[str, Any]],
    manager_method: str,
    baseline: str,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Paired manager-method comparison against one baseline method."""
    manager = np.asarray(
        [record["methods"][manager_method][metric] for record in records],
        dtype=np.float64,
    )
    base = np.asarray(
        [record["methods"][baseline][metric] for record in records],
        dtype=np.float64,
    )
    diff = manager - base if higher_is_better else base - manager
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "baseline": baseline,
        "manager_method": manager_method,
        "metric": metric,
        "direction": f"positive favors {manager_method}",
        "resource_manager_mean": float(np.mean(manager)),
        "baseline_mean": float(np.mean(base)),
        "paired_diff_mean_positive_favors_resource_manager": float(np.mean(diff)),
        "paired_diff_stderr": stderr(diff),
        "wins_for_resource_manager": int(np.sum(diff > 0.0)),
        "wins_for_baseline": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def paired_resource_manager_vs(
    records: list[dict[str, Any]],
    baseline: str,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Paired tracking resource-manager comparison against one baseline."""
    return paired_resource_manager_vs_method(
        records,
        manager_method="resource_manager",
        baseline=baseline,
        metric=metric,
        higher_is_better=higher_is_better,
    )


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one benchmark."""
    comparisons: dict[str, Any] = {}
    for baseline in POLICY_NAMES:
        baseline_rows: dict[str, Any] = {}
        for metric in ("online_mean_mse", "final_window_mse", "test_mse"):
            baseline_rows[metric] = paired_resource_manager_vs(
                records,
                baseline,
                metric,
                higher_is_better=False,
            )
        for metric in ("online_mean_accuracy", "final_window_accuracy", "test_accuracy"):
            baseline_rows[metric] = paired_resource_manager_vs(
                records,
                baseline,
                metric,
                higher_is_better=True,
            )
        comparisons[f"resource_manager_vs_{baseline}"] = baseline_rows
    retention_rows: dict[str, Any] = {}
    for metric in ("online_mean_mse", "final_window_mse", "test_mse"):
        retention_rows[metric] = paired_resource_manager_vs_method(
            records,
            manager_method="resource_manager_retention",
            baseline="mlp_static",
            metric=metric,
            higher_is_better=False,
        )
    for metric in ("online_mean_accuracy", "final_window_accuracy", "test_accuracy"):
        retention_rows[metric] = paired_resource_manager_vs_method(
            records,
            manager_method="resource_manager_retention",
            baseline="mlp_static",
            metric=metric,
            higher_is_better=True,
        )
    comparisons["resource_manager_retention_vs_mlp_static"] = retention_rows
    return {
        "method_stats": aggregate_method_stats(records),
        "comparisons": comparisons,
    }


def run_benchmark(benchmark: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run all seeds for one benchmark."""
    records: list[dict[str, Any]] = []
    stream_meta: dict[str, Any] | None = None
    dataset_meta: dict[str, Any] | None = None
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        if benchmark == "external_delayed_contextual_permutation":
            data = load_external_image_data(
                seed=seed,
                train_fraction=args.train_fraction,
                source=args.external_image_source,
                allow_openml_download=args.allow_openml_download,
                sample_limit=args.external_sample_limit,
            )
        else:
            data = load_digits_data(seed, args.train_fraction)
        dataset_meta = data.metadata
        stream = make_stream(benchmark, data, args, seed + 10_000)
        stream_meta = stream.metadata
        print(f"{benchmark} seed={seed}: running learned resource manager")
        (
            learners,
            states,
            tracking_manager,
            tracking_manager_state,
            retention_manager,
            retention_manager_state,
            metrics,
        ) = run_resource_manager_stream(stream, args, jr.key(seed + 100_000))
        summaries = summarize_metrics(metrics, args.final_window)
        held_out = evaluate_resource_manager(
            learners,
            states,
            tracking_manager,
            tracking_manager_state,
            retention_manager,
            retention_manager_state,
            stream,
        )
        for method, held_out_summary in held_out.items():
            summaries[method].update(held_out_summary)

        records.append(
            {
                "seed": seed,
                "methods": summaries,
                "resource_weights": manager_weight_summary(metrics, args.final_window),
            }
        )
        rm = summaries["resource_manager"]
        mlp = summaries["mlp_static"]
        print(
            f"{benchmark} seed={seed}: rm final_mse={rm['final_window_mse']:.4f}, "
            f"mlp final_mse={mlp['final_window_mse']:.4f}, "
            f"rm test_acc={rm['test_accuracy']:.3f}, mlp test_acc={mlp['test_accuracy']:.3f}"
        )

    if stream_meta is None or dataset_meta is None:
        raise RuntimeError(f"{benchmark} produced no records")

    return {
        "dataset": dataset_meta,
        "stream": stream_meta,
        "records": records,
        "aggregate": aggregate_records(records),
    }


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary."""
    cfg = results["config"]
    lines = [
        "# Step 2 Learned Resource Manager On Stateful External Benchmarks",
        "",
        "Default streams use the bundled sklearn digits dataset.  The optional "
        "external image stream uses OpenML Fashion-MNIST only when explicitly "
        "enabled, otherwise a local 28x28 sklearn-digits fallback.  The manager "
        "is causal: each prediction uses allocation weights learned before "
        "seeing the current label.",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
    ]
    for name, bench in results["benchmarks"].items():
        agg = bench["aggregate"]
        lines.extend(
            [
                f"## {name}",
                "",
                bench["stream"]["description"],
                "",
                "| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHOD_NAMES:
            stats = agg["method_stats"][method]
            lines.append(
                f"| `{method}` | "
                f"{stats['final_window_mse']['mean']:.4f} +/- "
                f"{stats['final_window_mse']['stderr']:.4f} | "
                f"{stats['final_window_accuracy']['mean']:.4f} +/- "
                f"{stats['final_window_accuracy']['stderr']:.4f} | "
                f"{stats['test_mse']['mean']:.4f} +/- "
                f"{stats['test_mse']['stderr']:.4f} | "
                f"{stats['test_accuracy']['mean']:.4f} +/- "
                f"{stats['test_accuracy']['stderr']:.4f} |"
            )
        lines.extend(["", "Paired resource-manager vs `mlp_static`:", ""])
        for metric in ("final_window_mse", "test_accuracy"):
            row = agg["comparisons"]["resource_manager_vs_mlp_static"][metric]
            lines.append(
                f"- `{metric}`: diff "
                f"{row['paired_diff_mean_positive_favors_resource_manager']:+.4f} "
                f"+/- {row['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{row['wins_for_resource_manager']}/"
                f"{row['wins_for_baseline']}/{row['ties']}."
            )
        retention_row = agg["comparisons"]["resource_manager_retention_vs_mlp_static"][
            "test_accuracy"
        ]
        lines.append(
            "- `resource_manager_retention` held-out `test_accuracy`: diff "
            f"{retention_row['paired_diff_mean_positive_favors_resource_manager']:+.4f} "
            f"+/- {retention_row['paired_diff_stderr']:.4f}; wins/losses/ties "
            f"{retention_row['wins_for_resource_manager']}/"
            f"{retention_row['wins_for_baseline']}/{retention_row['ties']}."
        )
        lines.append("")

    lines.extend(
        [
            "## Assessment Rule",
            "",
            "Positive paired differences favor the learned manager.  For MSE, "
            "the difference is baseline minus manager; for accuracy, it is "
            "manager minus baseline.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI values."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--low-sigma", type=float, default=1e-4)
    parser.add_argument("--high-sigma", type=float, default=1e-3)
    parser.add_argument("--cbp-decay", type=float, default=0.99)
    parser.add_argument("--cbp-replacement-rate", type=float, default=5e-4)
    parser.add_argument("--cbp-maturity", type=int, default=100)
    parser.add_argument("--manager-learning-rate", type=float, default=2.0)
    parser.add_argument("--manager-discount", type=float, default=0.995)
    parser.add_argument("--retention-manager-learning-rate", type=float, default=4.0)
    parser.add_argument("--retention-manager-discount", type=float, default=0.999)
    parser.add_argument("--manager-exploration", type=float, default=0.01)
    parser.add_argument("--manager-loss-decay", type=float, default=0.99)
    parser.add_argument("--resource-cost-weight", type=float, default=0.0)
    parser.add_argument("--low-sigma-cost", type=float, default=0.1)
    parser.add_argument("--high-sigma-cost", type=float, default=1.0)
    parser.add_argument("--cbp-cost", type=float, default=0.5)
    parser.add_argument("--n-states", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=240)
    parser.add_argument("--context-delay-blocks", type=int, default=1)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--external-image-source",
        choices=("digits_28x28_fallback", "openml_fashion_mnist"),
        default="digits_28x28_fallback",
        help=(
            "Image source for external_delayed_contextual_permutation. "
            "OpenML is attempted only with --allow-openml-download."
        ),
    )
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--external-sample-limit", type=int, default=3000)
    parser.add_argument(
        "--benchmarks",
        type=parse_csv,
        default=DEFAULT_BENCHMARKS,
        help=f"Comma-separated subset of {VALID_BENCHMARKS}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--note-path",
        type=Path,
        default=Path("docs/research/step2_resource_manager_stateful_external.md"),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0 or args.n_seeds <= 0 or args.final_window <= 0:
        raise ValueError("--steps, --n-seeds, and --final-window must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.n_states <= 0 or args.block_size <= 0:
        raise ValueError("--n-states and --block-size must be positive")
    if args.context_delay_blocks < 0:
        raise ValueError("--context-delay-blocks must be non-negative")
    if args.external_sample_limit <= 1:
        raise ValueError("--external-sample-limit must be greater than 1")
    if not 0.0 < args.mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative")
    unknown = sorted(set(args.benchmarks) - set(VALID_BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")


def main() -> None:
    """Run the benchmark suite."""
    args = parse_args()
    if args.smoke:
        args.steps = 160
        args.n_seeds = 1
        args.final_window = 40
        args.benchmarks = ("digits_recurrent_permutation",)
        args.block_size = 40
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.note_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    benchmarks = {name: run_benchmark(name, args) for name in args.benchmarks}
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
            "low_sigma": args.low_sigma,
            "high_sigma": args.high_sigma,
            "cbp_decay": args.cbp_decay,
            "cbp_replacement_rate": args.cbp_replacement_rate,
            "cbp_maturity": args.cbp_maturity,
            "manager_learning_rate": args.manager_learning_rate,
            "manager_discount": args.manager_discount,
            "retention_manager_learning_rate": args.retention_manager_learning_rate,
            "retention_manager_discount": args.retention_manager_discount,
            "manager_exploration": args.manager_exploration,
            "manager_loss_decay": args.manager_loss_decay,
            "resource_cost_weight": args.resource_cost_weight,
            "resource_policy_names": list(POLICY_NAMES),
            "resource_policy_costs": {
                "mlp_static": 0.0,
                "upgd_low": args.low_sigma_cost,
                "upgd_high": args.high_sigma_cost,
                "cbp_replace": args.cbp_cost,
            },
            "n_states": args.n_states,
            "block_size": args.block_size,
            "context_delay_blocks": args.context_delay_blocks,
            "mask_keep_fraction": args.mask_keep_fraction,
            "noise_std": args.noise_std,
            "external_image_source": args.external_image_source,
            "allow_openml_download": args.allow_openml_download,
            "external_sample_limit": args.external_sample_limit,
            "benchmarks": list(args.benchmarks),
        },
        "benchmarks": benchmarks,
        "wall_clock_s": time.time() - t0,
        "evidence_level": "learned_contextual_resource_manager_stateful_external",
    }

    json_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_summary(summary_path, results)
    write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

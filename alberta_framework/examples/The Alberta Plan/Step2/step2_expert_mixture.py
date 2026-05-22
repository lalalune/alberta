#!/usr/bin/env python3
"""Direction 9: online MLP/UPGD expert mixture across Step 2 regimes.

This script runs a small prequential expert portfolio over two existing Step 2
learners:

* fair MLP: ``MultiHeadMLPLearner``
* UPGD: ``UPGDLearner`` with the same trunk, heads, step size, sparsity,
  layer norm, and ObGD bounding, plus utility-scaled perturbations

The mixture is deliberately simple. At each time step it predicts with both
experts before either expert updates, forms a convex prediction using discounted
Hedge weights, records pre-update loss, then updates both experts on the same
example. This tests whether the portfolio can keep MLP's behavior on external
digits while taking advantage of UPGD when it is better on synthetic streams.

Outputs default to ``output/direction9_expert_mixture_scaled/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    CompositionalStream,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
)

N_DIGIT_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("output/direction9_expert_mixture_scaled")
SYNTHETIC_REGIMES = (
    "synthetic_polynomial",
    "synthetic_frequency",
    "synthetic_compositional",
)
DIGITS_REGIMES = (
    "digits_iid",
    "digits_class_blocked",
    "digits_permuted_pixels",
    "digits_mask_noise",
    "digits_label_drift",
)
VALID_DATASETS = (*SYNTHETIC_REGIMES, *DIGITS_REGIMES)
RETENTION_ROUTERS = ("none", "class_imbalance")


def make_mlp(
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
) -> MultiHeadMLPLearner:
    """Create the non-perturbed MLP baseline."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )


def make_upgd(
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    perturbation_sigma: float,
    perturbation_warmup_steps: int = 0,
    perturbation_ramp_steps: int = 0,
) -> UPGDLearner:
    """Create the architecture-matched UPGD learner."""
    return UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=perturbation_sigma,
        perturbation_warmup_steps=perturbation_warmup_steps,
        perturbation_ramp_steps=perturbation_ramp_steps,
    )


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a scan stream into observation and target arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, stream_state, jnp.arange(num_steps)
    )
    return observations, targets


def make_synthetic_stream(
    steps: int,
    seed: int,
    regime: str = "synthetic_polynomial",
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Create one synthetic out-of-class stream realization."""
    context_length = max(100, steps // 4)
    stream: Any
    meta: dict[str, Any]
    if regime == "synthetic_polynomial":
        stream = OutOfClassPolynomialStream(
            feature_dim=8,
            n_tasks=3,
            n_contexts=4,
            context_length=context_length,
            active_triples_per_context=2,
            noise_std=0.05,
        )
        meta = {
            "dataset": "synthetic_out_of_class_polynomial",
            "feature_dim": 8,
            "n_heads": 3,
            "steps": steps,
            "stream_seed": seed,
            "description": "Triple-product oracle outside the pair-product feature class.",
        }
    elif regime == "synthetic_frequency":
        stream = FrequencyMismatchStream(
            feature_dim=4,
            n_tasks=2,
            n_components_per_task=3,
            n_contexts=4,
            context_length=context_length,
            noise_std=0.05,
        )
        meta = {
            "dataset": "synthetic_frequency_mismatch",
            "feature_dim": 4,
            "n_heads": 2,
            "steps": steps,
            "stream_seed": seed,
            "description": "Trigonometric target features outside the learner feature class.",
        }
    elif regime == "synthetic_compositional":
        stream = CompositionalStream(
            feature_dim=6,
            n_tasks=3,
            inner_hidden=4,
            outer_components=5,
            n_contexts=4,
            context_length=context_length,
            noise_std=0.05,
        )
        meta = {
            "dataset": "synthetic_compositional",
            "feature_dim": 6,
            "n_heads": 3,
            "steps": steps,
            "stream_seed": seed,
            "description": "Two-hidden-layer tanh oracle outside shallow feature classes.",
        }
    else:
        raise ValueError(f"unknown synthetic regime {regime!r}")

    observations, targets = collect_stream_arrays(stream, steps, jr.key(seed))
    meta["regime"] = regime
    meta["context_length"] = context_length
    return observations, targets, meta


def load_digits_arrays(
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and standardize sklearn digits using train-split statistics only."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for --datasets digits/all. "
            "Install the external extra or run with --datasets synthetic."
        )
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for cls in range(N_DIGIT_CLASSES):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    train_indices_arr = np.asarray(train_indices, dtype=np.int32)
    test_indices_arr = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_indices_arr)
    rng.shuffle(test_indices_arr)

    x_train = x[train_indices_arr]
    y_train = y[train_indices_arr]
    x_test = x[test_indices_arr]
    y_test = y[test_indices_arr]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    meta = {
        "dataset": "sklearn.datasets.load_digits",
        "feature_dim": int(x_train.shape[1]),
        "n_heads": N_DIGIT_CLASSES,
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "train_fraction": train_fraction,
        "split_seed": seed,
    }
    return x_train, y_train, x_test, y_test, meta


def make_digits_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    class_blocked: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create a finite online digit stream from shuffled train-set epochs."""
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_DIGIT_CLASSES)]

    while total < steps:
        if class_blocked:
            epoch_parts: list[np.ndarray] = []
            for cls in rng.permutation(N_DIGIT_CLASSES):
                cls_idx = class_indices[int(cls)].copy()
                rng.shuffle(cls_idx)
                epoch_parts.append(cls_idx)
            indices = np.concatenate(epoch_parts)
        else:
            indices = rng.permutation(len(y_train))
        chunks_x.append(x_train[indices])
        chunks_y.append(y_train[indices])
        total += len(indices)

    observations = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float32)[labels]
    return jnp.asarray(observations), jnp.asarray(targets), jnp.asarray(labels)


def phase_count(steps: int, phase_length: int) -> int:
    return int(math.ceil(steps / phase_length))


def label_permutations(n_phases: int, rng: np.random.Generator) -> list[np.ndarray]:
    perms = [np.arange(N_DIGIT_CLASSES, dtype=np.int32)]
    for _ in range(1, n_phases):
        perms.append(rng.permutation(N_DIGIT_CLASSES).astype(np.int32))
    return perms


def pixel_permutations(
    feature_dim: int,
    n_phases: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    perms = [np.arange(feature_dim, dtype=np.int32)]
    for _ in range(1, n_phases):
        perms.append(rng.permutation(feature_dim).astype(np.int32))
    return perms


def feature_masks(
    feature_dim: int,
    n_phases: int,
    keep_fraction: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_keep = max(1, min(feature_dim, int(round(feature_dim * keep_fraction))))
    masks: list[np.ndarray] = []
    for _ in range(n_phases):
        keep = rng.choice(feature_dim, size=n_keep, replace=False)
        mask = np.zeros(feature_dim, dtype=np.float32)
        mask[keep] = 1.0
        masks.append(mask)
    return masks


def make_digits_regime_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    steps: int,
    seed: int,
    regime: str,
    phase_length: int,
    mask_keep_fraction: float,
    mask_noise_std: float,
) -> tuple[jax.Array, jax.Array, jax.Array, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a stationary or nonstationary online digits stream."""
    if regime not in DIGITS_REGIMES:
        raise ValueError(f"unknown digits regime {regime!r}")
    if phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if mask_noise_std < 0.0:
        raise ValueError("--mask-noise-std must be non-negative")

    observations, _, labels_jax = make_digits_sequence(
        x_train=x_train,
        y_train=y_train,
        steps=steps,
        seed=seed,
        class_blocked=regime == "digits_class_blocked",
    )
    obs_np = np.asarray(observations).astype(np.float32)
    labels = np.asarray(labels_jax).astype(np.int32)
    test_x = x_test.astype(np.float32).copy()
    test_y = y_test.astype(np.int32).copy()

    phase_ids = np.arange(steps, dtype=np.int32) // phase_length
    n_phases = phase_count(steps, phase_length)
    final_phase = int(phase_ids[-1])
    rng = np.random.default_rng(seed + 50_000)
    meta: dict[str, Any] = {
        "regime": regime,
        "phase_length": phase_length,
        "n_phases": n_phases,
        "final_phase": final_phase,
        "class_blocked": regime == "digits_class_blocked",
    }

    if regime == "digits_permuted_pixels":
        permutations = pixel_permutations(obs_np.shape[1], n_phases, rng)
        for phase, perm in enumerate(permutations):
            mask = phase_ids == phase
            obs_np[mask] = obs_np[mask][:, perm]
        test_x = test_x[:, permutations[final_phase]]
        meta["description"] = "Pixel order changes by phase; held-out uses final phase."
    elif regime == "digits_mask_noise":
        masks = feature_masks(obs_np.shape[1], n_phases, mask_keep_fraction, rng)
        if mask_noise_std > 0.0:
            obs_np = obs_np + rng.normal(0.0, mask_noise_std, size=obs_np.shape).astype(
                np.float32
            )
        for phase, mask in enumerate(masks):
            phase_mask = phase_ids == phase
            obs_np[phase_mask] = obs_np[phase_mask] * mask
        test_rng = np.random.default_rng(seed + 60_000 + final_phase)
        if mask_noise_std > 0.0:
            test_x = test_x + test_rng.normal(
                0.0, mask_noise_std, size=test_x.shape
            ).astype(np.float32)
        test_x = test_x * masks[final_phase]
        meta.update(
            {
                "description": "Visible feature mask rotates by phase with feature noise.",
                "mask_keep_fraction": mask_keep_fraction,
                "mask_noise_std": mask_noise_std,
            }
        )
    elif regime == "digits_label_drift":
        perms = label_permutations(n_phases, rng)
        for phase, perm in enumerate(perms):
            phase_mask = phase_ids == phase
            labels[phase_mask] = perm[labels[phase_mask]]
        test_y = perms[final_phase][test_y]
        meta["description"] = "Class-head meanings are permuted by phase."
    elif regime == "digits_class_blocked":
        meta["description"] = "Training stream is grouped into digit-class blocks."
    else:
        meta["description"] = "Stationary IID shuffled-epoch digits control."

    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float32)[labels]
    return (
        jnp.asarray(obs_np.astype(np.float32)),
        jnp.asarray(targets),
        jnp.asarray(labels),
        test_x.astype(np.float32),
        test_y.astype(np.int32),
        meta,
    )


def run_expert_mixture_stream(
    mlp: MultiHeadMLPLearner,
    upgd: UPGDLearner,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
    hedge_eta: float,
    hedge_discount: float,
) -> tuple[Any, Any, np.ndarray]:
    """Run discounted Hedge over MLP and UPGD on a prequential stream.

    Metric columns:
        0 mixture MSE, 1 MLP MSE, 2 UPGD MSE, 3 MLP weight, 4 UPGD weight,
        5 mixture predicted class, 6 MLP predicted class, 7 UPGD predicted class.
    """
    mlp_key, upgd_key = jr.split(key)
    feature_dim = int(observations.shape[1])
    mlp_state = mlp.init(feature_dim, mlp_key)
    upgd_state = upgd.init(feature_dim, upgd_key)
    log_weights = jnp.zeros(2, dtype=jnp.float32)

    eta = jnp.array(hedge_eta, dtype=jnp.float32)
    discount = jnp.array(hedge_discount, dtype=jnp.float32)

    def step_fn(
        carry: tuple[Any, Any, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[Any, Any, jax.Array], jax.Array]:
        mlp_s, upgd_s, log_w = carry
        obs, tgt = inputs

        weights = jax.nn.softmax(log_w)
        mlp_pred = mlp.predict(mlp_s, obs)
        upgd_pred = upgd.predict(upgd_s, obs)
        mixture_pred = weights[0] * mlp_pred + weights[1] * upgd_pred

        mlp_loss = jnp.mean((mlp_pred - tgt) ** 2)
        upgd_loss = jnp.mean((upgd_pred - tgt) ** 2)
        mixture_loss = jnp.mean((mixture_pred - tgt) ** 2)

        expert_losses = jnp.stack([mlp_loss, upgd_loss])
        new_log_w = discount * log_w - eta * expert_losses
        new_log_w = new_log_w - jnp.max(new_log_w)

        mlp_result = mlp.update(mlp_s, obs, tgt)
        upgd_result = upgd.update(upgd_s, obs, tgt)

        metric = jnp.stack(
            [
                mixture_loss,
                mlp_loss,
                upgd_loss,
                weights[0],
                weights[1],
                jnp.argmax(mixture_pred).astype(jnp.float32),
                jnp.argmax(mlp_pred).astype(jnp.float32),
                jnp.argmax(upgd_pred).astype(jnp.float32),
            ]
        )
        return (mlp_result.state, upgd_result.state, new_log_w), metric

    (final_mlp, final_upgd, _), metrics = jax.lax.scan(
        step_fn, (mlp_state, upgd_state, log_weights), (observations, targets)
    )
    metrics.block_until_ready()
    return final_mlp, final_upgd, np.asarray(metrics)


def summarize_prequential(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Summarize mixture and expert curves."""
    window = min(final_window, metrics.shape[0])
    names = ["mixture", "mlp", "upgd"]
    summary: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        entry = {
            "online_mean_mse": float(np.mean(metrics[:, idx])),
            "final_window_mse": float(np.mean(metrics[-window:, idx])),
        }
        if labels is not None:
            pred_col = 5 + idx
            correct = metrics[:, pred_col].astype(np.int32) == labels
            entry["online_mean_accuracy"] = float(np.mean(correct))
            entry["final_window_accuracy"] = float(np.mean(correct[-window:]))
        summary[name] = entry
    summary["mixture"]["mean_mlp_weight"] = float(np.mean(metrics[:, 3]))
    summary["mixture"]["mean_upgd_weight"] = float(np.mean(metrics[:, 4]))
    summary["mixture"]["final_mlp_weight"] = float(metrics[-1, 3])
    summary["mixture"]["final_upgd_weight"] = float(metrics[-1, 4])
    return summary


def evaluate_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate an individual expert on the held-out digits split."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def evaluate_mixture_classifier(
    mlp: MultiHeadMLPLearner,
    mlp_state: Any,
    upgd: UPGDLearner,
    upgd_state: Any,
    final_weights: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final convex expert mixture on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    weights = jnp.asarray(final_weights.astype(np.float32))

    def predict(obs: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            weights[0] * mlp.predict(mlp_state, obs)
            + weights[1] * upgd.predict(upgd_state, obs),
        )

    preds = jax.vmap(predict)(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def class_imbalance_retention_signal(
    labels: np.ndarray | None,
    n_heads: int,
    final_window: int,
    min_lifetime_class_fraction: float,
    max_recent_class_fraction: float,
) -> dict[str, float | int | bool]:
    """Detect when recent one-step loss is a poor retained-class proxy.

    Class-blocked streams can make the tracking router overfit to the final
    block: recent loss is excellent, but retained held-out predictions over all
    classes are poor.  This signal uses only labels observed in the online
    stream.  It triggers when the lifetime stream covered most classes, but the
    recent window covers only a small fraction of those observed classes.
    """
    if labels is None or labels.size == 0 or n_heads <= 0:
        return {
            "retention_hazard": False,
            "lifetime_class_count": 0,
            "recent_class_count": 0,
            "lifetime_class_fraction": 0.0,
            "recent_class_fraction": 0.0,
            "recent_fraction_of_lifetime": 0.0,
        }

    window = min(max(int(final_window), 1), int(labels.shape[0]))
    lifetime_classes = np.unique(labels)
    recent_classes = np.unique(labels[-window:])
    lifetime_count = int(lifetime_classes.shape[0])
    recent_count = int(recent_classes.shape[0])
    lifetime_fraction = float(lifetime_count / n_heads)
    recent_fraction = float(recent_count / n_heads)
    recent_fraction_of_lifetime = float(recent_count / max(lifetime_count, 1))
    hazard = (
        lifetime_fraction >= min_lifetime_class_fraction
        and recent_fraction_of_lifetime <= max_recent_class_fraction
    )
    return {
        "retention_hazard": bool(hazard),
        "lifetime_class_count": lifetime_count,
        "recent_class_count": recent_count,
        "lifetime_class_fraction": lifetime_fraction,
        "recent_class_fraction": recent_fraction,
        "recent_fraction_of_lifetime": recent_fraction_of_lifetime,
    }


def retention_deployment_weights(
    tracking_weights: np.ndarray,
    labels: np.ndarray | None,
    n_heads: int,
    final_window: int,
    retention_router: str,
    retention_upgd_deployment_weight: float,
    min_lifetime_class_fraction: float,
    max_recent_class_fraction: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Return final held-out deployment weights for the expert mixture."""
    weights = np.asarray(tracking_weights, dtype=np.float32).copy()
    weights = np.clip(weights, 0.0, 1.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.asarray([0.5, 0.5], dtype=np.float32)
    else:
        weights = weights / total

    signal: dict[str, float | int | bool | str] = dict(
        class_imbalance_retention_signal(
            labels=labels,
            n_heads=n_heads,
            final_window=final_window,
            min_lifetime_class_fraction=min_lifetime_class_fraction,
            max_recent_class_fraction=max_recent_class_fraction,
        )
    )
    signal["router"] = retention_router
    signal["tracking_mlp_weight"] = float(weights[0])
    signal["tracking_upgd_weight"] = float(weights[1])
    signal["deployment_source"] = "tracking"

    if retention_router == "class_imbalance" and signal["retention_hazard"]:
        upgd_weight = max(float(weights[1]), retention_upgd_deployment_weight)
        upgd_weight = min(max(upgd_weight, 0.0), 1.0)
        weights = np.asarray([1.0 - upgd_weight, upgd_weight], dtype=np.float32)
        signal["deployment_source"] = "class_imbalance_retention"

    signal["deployment_mlp_weight"] = float(weights[0])
    signal["deployment_upgd_weight"] = float(weights[1])
    return weights, signal


def stderr(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed records by dataset, method, and metric."""
    aggregate: dict[str, Any] = {}
    datasets = sorted({record["dataset_name"] for record in records})
    methods = ["mixture", "mlp", "upgd"]
    for dataset in datasets:
        dataset_records = [r for r in records if r["dataset_name"] == dataset]
        aggregate[dataset] = {}
        for method in methods:
            method_metrics: dict[str, Any] = {}
            keys = sorted(dataset_records[0]["methods"][method].keys())
            for key in keys:
                values = np.asarray(
                    [r["methods"][method][key] for r in dataset_records],
                    dtype=np.float64,
                )
                method_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "stderr": stderr(values),
                    "values": values.tolist(),
                }
            aggregate[dataset][method] = method_metrics
        aggregate[dataset]["comparisons"] = paired_comparisons(dataset_records)
        aggregate[dataset]["best_expert_regret"] = aggregate_best_expert_regret(
            dataset_records
        )
    return aggregate


def best_expert_comparison(methods: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Compute mixture regret or shortfall against the better expert."""
    comparison: dict[str, Any] = {}
    candidate_metrics = [
        "final_window_mse",
        "online_mean_mse",
        "test_mse",
        "final_window_accuracy",
        "online_mean_accuracy",
        "test_accuracy",
    ]
    for metric in candidate_metrics:
        if metric not in methods["mixture"]:
            continue
        expert_values = {
            "mlp": float(methods["mlp"][metric]),
            "upgd": float(methods["upgd"][metric]),
        }
        mixture_value = float(methods["mixture"][metric])
        if metric.endswith("accuracy"):
            best_expert = max(expert_values, key=expert_values.__getitem__)
            best_value = expert_values[best_expert]
            regret = best_value - mixture_value
        else:
            best_expert = min(expert_values, key=expert_values.__getitem__)
            best_value = expert_values[best_expert]
            regret = mixture_value - best_value
        comparison[metric] = {
            "best_expert": best_expert,
            "best_expert_value": best_value,
            "mixture_value": mixture_value,
            "regret_positive_favors_best_expert": float(regret),
            "failure": bool(regret > 0.0),
        }
    return comparison


def aggregate_best_expert_regret(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mixture regret against the better expert by metric."""
    metrics = sorted(records[0]["best_expert_comparison"].keys())
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        rows = [r["best_expert_comparison"][metric] for r in records]
        regrets = np.asarray(
            [row["regret_positive_favors_best_expert"] for row in rows],
            dtype=np.float64,
        )
        best_experts = [row["best_expert"] for row in rows]
        aggregate[metric] = {
            "mean_regret_positive_favors_best_expert": float(np.mean(regrets)),
            "stderr": stderr(regrets),
            "values": regrets.tolist(),
            "failures": int(np.sum(regrets > 0.0)),
            "ties_or_beats_best": int(np.sum(regrets <= 0.0)),
            "best_expert_counts": {
                "mlp": int(best_experts.count("mlp")),
                "upgd": int(best_experts.count("upgd")),
            },
        }
    return aggregate


def paired_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired differences of mixture against each expert for main metrics."""
    comparisons: dict[str, Any] = {}
    candidate_metrics = [
        "final_window_mse",
        "online_mean_mse",
        "test_mse",
        "final_window_accuracy",
        "online_mean_accuracy",
        "test_accuracy",
    ]
    for metric in candidate_metrics:
        if metric not in records[0]["methods"]["mixture"]:
            continue
        mix = np.asarray(
            [r["methods"]["mixture"][metric] for r in records], dtype=np.float64
        )
        comparisons[metric] = {}
        for expert in ["mlp", "upgd"]:
            exp = np.asarray(
                [r["methods"][expert][metric] for r in records], dtype=np.float64
            )
            if metric.endswith("accuracy"):
                diff = mix - exp
                wins = np.sum(diff > 0.0)
                losses = np.sum(diff < 0.0)
            else:
                diff = exp - mix
                wins = np.sum(diff > 0.0)
                losses = np.sum(diff < 0.0)
            sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
            comparisons[metric][f"mixture_vs_{expert}"] = {
                "paired_diff_mean_positive_favors_mixture": float(np.mean(diff)),
                "paired_diff_stderr": stderr(diff),
                "wins_for_mixture": int(wins),
                "wins_for_expert": int(losses),
                "ties": int(np.sum(diff == 0.0)),
                "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
            }
    return comparisons


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    cfg = results["config"]
    lines = [
        "# Direction 9 Expert Mixture Scaled",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} steps, "
            f"final window {cfg['final_window']}, Hedge eta={cfg['hedge_eta']}, "
            f"discount={cfg['hedge_discount']}, "
            f"retention router={cfg.get('retention_router', 'none')}."
        ),
        "",
        "Methods: architecture-matched MLP and UPGD experts plus a discounted "
        "Hedge convex mixture. Expert metrics are the same experts used inside "
        "the mixture, updated on the same stream.",
        "",
    ]

    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend([f"## {dataset}", ""])
        metric = (
            "final_window_accuracy"
            if "final_window_accuracy" in dataset_agg["mixture"]
            else "final_window_mse"
        )
        lines.extend(
            [
                f"Primary metric: `{metric}`.",
                "",
                "| Method | final_window_mse | online_mean_mse | "
                "final_window_accuracy | test_accuracy | weights |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in ["mixture", "mlp", "upgd"]:
            row = dataset_agg[method]

            def fmt(key: str) -> str:
                if key not in row:
                    return ""
                return f"{row[key]['mean']:.4f} +/- {row[key]['stderr']:.4f}"

            if method == "mixture":
                weights = (
                    f"track MLP {row['final_mlp_weight']['mean']:.3f}, "
                    f"UPGD {row['final_upgd_weight']['mean']:.3f}"
                )
                if "deployment_upgd_weight" in row:
                    weights = (
                        f"{weights}; deploy MLP "
                        f"{row['deployment_mlp_weight']['mean']:.3f}, "
                        f"UPGD {row['deployment_upgd_weight']['mean']:.3f}"
                    )
            else:
                weights = ""
            lines.append(
                f"| {method} | {fmt('final_window_mse')} | "
                f"{fmt('online_mean_mse')} | {fmt('final_window_accuracy')} | "
                f"{fmt('test_accuracy')} | {weights} |"
            )
        lines.append("")

        regret = dataset_agg["best_expert_regret"].get(metric)
        if regret is not None:
            lines.extend(
                [
                    (
                        "Best-expert regret: "
                        f"{regret['mean_regret_positive_favors_best_expert']:.4f} "
                        f"+/- {regret['stderr']:.4f}; "
                        f"failures {regret['failures']}/{cfg['n_seeds']}."
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Critical Interpretation",
            "",
            "This is a small universality probe, not a broad universality result. "
            "A useful outcome is a mixture that is close to the stronger expert on "
            "each stream without knowing the stream identity. Any gain here should "
            "be treated as routing/adaptation evidence until it survives more seeds, "
            "larger streams, and additional out-of-class generators.",
            "",
            "Failure cases are regimes where the mixture's primary metric trails "
            "the better of MLP and UPGD. Positive best-expert regret favors the "
            "better expert; zero or negative means the mixture tied or beat both "
            "experts on that paired seed.",
            "",
        ]
    )
    if cfg.get("retention_router", "none") != "none":
        lines.extend(
            [
                "The retention router affects held-out deployment weights only. "
                "Prequential final-window metrics still use the ordinary tracking "
                "Hedge predictor. The class-imbalance trigger uses observed stream "
                "labels to detect when lifetime class coverage is broad but the "
                "recent window is class-narrow, which is exactly the failure mode "
                "where one-step loss can erase retained-class performance.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def expand_dataset_names(dataset_arg: str) -> list[str]:
    """Expand CLI dataset aliases into concrete regimes."""
    aliases = {
        "all": list(VALID_DATASETS),
        "synthetic": list(SYNTHETIC_REGIMES),
        "digits": list(DIGITS_REGIMES),
        "synthetic_all": list(SYNTHETIC_REGIMES),
        "digits_all": list(DIGITS_REGIMES),
    }
    names: list[str] = []
    for raw_name in dataset_arg.split(","):
        name = raw_name.strip()
        if not name:
            continue
        names.extend(aliases.get(name, [name]))
    unknown = sorted({name for name in names if name not in VALID_DATASETS})
    if unknown:
        valid = ", ".join(("all", "synthetic", "digits", *VALID_DATASETS))
        raise ValueError(f"unknown --datasets entries {unknown}; valid entries: {valid}")
    return list(dict.fromkeys(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated regimes or aliases: all, synthetic, digits, "
            f"{', '.join(VALID_DATASETS)}."
        ),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-3)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument("--hedge-eta", type=float, default=8.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument(
        "--retention-router",
        choices=RETENTION_ROUTERS,
        default="none",
        help=(
            "Held-out deployment router. 'class_imbalance' preserves the "
            "tracking Hedge predictor online, but switches deployment weights "
            "toward UPGD when the lifetime stream covered most classes and the "
            "recent window covers only a small fraction of them."
        ),
    )
    parser.add_argument(
        "--retention-upgd-deployment-weight",
        type=float,
        default=1.0,
        help="Minimum UPGD deployment weight when the retention router triggers.",
    )
    parser.add_argument(
        "--retention-min-lifetime-class-fraction",
        type=float,
        default=0.8,
        help="Minimum lifetime class coverage required to trigger retention mode.",
    )
    parser.add_argument(
        "--retention-max-recent-class-fraction",
        type=float,
        default=0.4,
        help=(
            "Maximum recent/lifetime class coverage fraction allowed before "
            "retention mode triggers."
        ),
    )
    parser.add_argument("--class-blocked", action="store_true")
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Run one dataset/seed combination."""
    hidden_sizes = (args.hidden_size,)
    labels_np: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    if dataset_name in SYNTHETIC_REGIMES:
        observations, targets, dataset_meta = make_synthetic_stream(
            steps=args.steps,
            seed=seed + 20_000,
            regime=dataset_name,
        )
        n_heads = int(targets.shape[1])
    elif dataset_name in DIGITS_REGIMES:
        x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
            seed=seed,
            train_fraction=args.train_fraction,
        )
        digits_regime = (
            "digits_class_blocked"
            if args.class_blocked and dataset_name == "digits_iid"
            else dataset_name
        )
        (
            observations,
            targets,
            labels,
            x_test,
            y_test,
            stream_meta,
        ) = make_digits_regime_sequence(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            steps=args.steps,
            seed=seed + 10_000,
            regime=digits_regime,
            phase_length=args.phase_length,
            mask_keep_fraction=args.mask_keep_fraction,
            mask_noise_std=args.mask_noise_std,
        )
        dataset_meta.update(stream_meta)
        labels_np = np.asarray(labels)
        n_heads = N_DIGIT_CLASSES
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    mlp = make_mlp(n_heads, hidden_sizes, args.step_size, args.sparsity)
    upgd = make_upgd(
        n_heads,
        hidden_sizes,
        args.step_size,
        args.sparsity,
        args.perturbation_sigma,
        args.perturbation_warmup_steps,
        args.perturbation_ramp_steps,
    )
    mlp_state, upgd_state, metrics = run_expert_mixture_stream(
        mlp=mlp,
        upgd=upgd,
        key=jr.key(seed),
        observations=observations,
        targets=targets,
        hedge_eta=args.hedge_eta,
        hedge_discount=args.hedge_discount,
    )
    methods = summarize_prequential(metrics, args.final_window, labels_np)
    retention_signal: dict[str, float | int | bool | str] | None = None

    if dataset_name in DIGITS_REGIMES:
        assert x_test is not None and y_test is not None
        tracking_weights = metrics[-1, 3:5]
        final_weights, retention_signal = retention_deployment_weights(
            tracking_weights=tracking_weights,
            labels=labels_np,
            n_heads=n_heads,
            final_window=args.final_window,
            retention_router=args.retention_router,
            retention_upgd_deployment_weight=args.retention_upgd_deployment_weight,
            min_lifetime_class_fraction=args.retention_min_lifetime_class_fraction,
            max_recent_class_fraction=args.retention_max_recent_class_fraction,
        )
        methods["mlp"].update(evaluate_classifier(mlp, mlp_state, x_test, y_test))
        methods["upgd"].update(evaluate_classifier(upgd, upgd_state, x_test, y_test))
        methods["mixture"].update(
            evaluate_mixture_classifier(
                mlp, mlp_state, upgd, upgd_state, final_weights, x_test, y_test
            )
        )
        methods["mixture"].update(
            {
                "deployment_mlp_weight": float(final_weights[0]),
                "deployment_upgd_weight": float(final_weights[1]),
                "retention_hazard": float(
                    bool(retention_signal["retention_hazard"])
                ),
                "retention_recent_class_fraction": float(
                    retention_signal["recent_fraction_of_lifetime"]
                ),
                "retention_lifetime_class_fraction": float(
                    retention_signal["lifetime_class_fraction"]
                ),
            }
        )

    record = {
        "dataset_name": dataset_name,
        "seed": seed,
        "dataset": dataset_meta,
        "methods": methods,
        "best_expert_comparison": best_expert_comparison(methods),
    }
    if retention_signal is not None:
        record["retention_router"] = retention_signal
    return record, dataset_meta, metrics


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if not 0.0 <= args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in [0, 1]")
    if args.phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < args.mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if args.mask_noise_std < 0.0:
        raise ValueError("--mask-noise-std must be non-negative")
    if args.perturbation_warmup_steps < 0:
        raise ValueError("--perturbation-warmup-steps must be non-negative")
    if args.perturbation_ramp_steps < 0:
        raise ValueError("--perturbation-ramp-steps must be non-negative")
    if not 0.0 <= args.retention_upgd_deployment_weight <= 1.0:
        raise ValueError("--retention-upgd-deployment-weight must be in [0, 1]")
    if not 0.0 <= args.retention_min_lifetime_class_fraction <= 1.0:
        raise ValueError("--retention-min-lifetime-class-fraction must be in [0, 1]")
    if not 0.0 <= args.retention_max_recent_class_fraction <= 1.0:
        raise ValueError("--retention-max-recent-class-fraction must be in [0, 1]")

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = expand_dataset_names(args.datasets)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}

    for dataset_name in dataset_names:
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            print(f"dataset={dataset_name} seed={seed}: running expert mixture")
            record, meta, metrics = run_one_dataset_seed(dataset_name, seed, args)
            records.append(record)
            datasets_meta[dataset_name] = meta
            npz_path = output_dir / f"{dataset_name}_seed{seed}_curves.npz"
            np.savez_compressed(npz_path, metrics=metrics)
            m = record["methods"]
            print(
                f"dataset={dataset_name} seed={seed}: final MSE "
                f"mix={m['mixture']['final_window_mse']:.4f}, "
                f"mlp={m['mlp']['final_window_mse']:.4f}, "
                f"upgd={m['upgd']['final_window_mse']:.4f}; "
                f"final weights mlp={m['mixture']['final_mlp_weight']:.3f}, "
                f"upgd={m['mixture']['final_upgd_weight']:.3f}"
            )

    results = {
        "config": {
            "datasets": dataset_names,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "final_window": args.final_window,
            "hidden_sizes": [args.hidden_size],
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "perturbation_sigma": args.perturbation_sigma,
            "perturbation_warmup_steps": args.perturbation_warmup_steps,
            "perturbation_ramp_steps": args.perturbation_ramp_steps,
            "hedge_eta": args.hedge_eta,
            "hedge_discount": args.hedge_discount,
            "retention_router": args.retention_router,
            "retention_upgd_deployment_weight": (
                args.retention_upgd_deployment_weight
            ),
            "retention_min_lifetime_class_fraction": (
                args.retention_min_lifetime_class_fraction
            ),
            "retention_max_recent_class_fraction": (
                args.retention_max_recent_class_fraction
            ),
            "class_blocked": args.class_blocked,
            "phase_length": args.phase_length,
            "mask_keep_fraction": args.mask_keep_fraction,
            "mask_noise_std": args.mask_noise_std,
        },
        "datasets": datasets_meta,
        "records": records,
        "aggregate": aggregate_records(records),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "small_prequential_online_expert_mixture_probe",
    }

    json_path = output_dir / "results.json"
    md_path = output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

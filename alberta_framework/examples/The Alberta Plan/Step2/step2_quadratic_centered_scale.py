#!/usr/bin/env python3
"""Scale the positive Step 2 moonshots: centered targets and quadratic lifts.

This script leans into the two moonshot ideas that showed signal:

* M01: centered classification targets.
* M04: explicit hashed quadratic interaction features.

The standard is intentionally stricter than the smoke tests.  A candidate must
beat the best fair MLP comparator on the same stream.  The suite includes an
interaction stream where quadratic features should help, a nonlinear tanh
control where they need not help, and external digits regimes where target
encoding and interaction features are tested against one-hot MLP baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    InteractionFeatureDiscoveryStream,
    MultiHeadMLPLearner,
    NonlinearFeatureDiscoveryStream,
    ObGDBounding,
    run_multi_head_learning_loop,
)

N_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("outputs/step2_quadratic_centered_scale")
DEFAULT_NOTE_PATH = Path("docs/research/step2_quadratic_centered_scale.md")


@dataclass(frozen=True)
class ScaleConfig:
    """Configuration for the scaling experiment."""

    num_steps: int = 900
    num_seeds: int = 3
    final_window: int = 180
    feature_dim: int = 10
    n_tasks: int = 2
    n_contexts: int = 4
    context_length: int = 150
    interaction_active_pairs: int = 1
    nonlinear_latents: int = 32
    nonlinear_active_latents: int = 4
    noise_std: float = 0.01
    linear_scale: float = 0.01
    learner_step_size: float = 0.03
    obgd_kappa: float = 2.0
    mlp_sparsity: float = 0.5
    train_fraction: float = 0.7
    phase_length: int = 300
    hash_dims: tuple[int, ...] = (128, 512)
    regression_suites: tuple[str, ...] = ("interaction", "nonlinear")
    digits_regimes: tuple[str, ...] = ("iid", "label_drift", "permuted_pixels")


@dataclass(frozen=True)
class MethodSpec:
    """One learner configuration."""

    hidden_sizes: tuple[int, ...]
    hash_dim: int | None
    target_code: str | None = None


@dataclass(frozen=True)
class DigitsSplit:
    """Standardized sklearn digits arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class DigitStream:
    """Materialized online digits stream and final-phase held-out view."""

    observations: jax.Array
    labels: jax.Array
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def one_hot_targets(labels: np.ndarray) -> np.ndarray:
    """Return one-hot vectors."""
    return np.eye(N_CLASSES, dtype=np.float32)[labels]


def centered_targets(labels: np.ndarray) -> np.ndarray:
    """Return zero-sum centered class codes."""
    targets = np.full((labels.shape[0], N_CLASSES), -1.0 / (N_CLASSES - 1), dtype=np.float32)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return targets


def targets_for(labels: np.ndarray, target_code: str) -> np.ndarray:
    """Return class targets for a named code."""
    if target_code == "one_hot":
        return one_hot_targets(labels)
    if target_code == "centered":
        return centered_targets(labels)
    raise ValueError(f"unknown target code {target_code!r}")


def pair_hash_arrays(
    feature_dim: int,
    hash_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return deterministic pair indices, buckets, and signs for a hash sketch."""
    left: list[int] = []
    right: list[int] = []
    buckets: list[int] = []
    signs: list[float] = []
    mask64 = (1 << 64) - 1

    for i in range(feature_dim):
        for j in range(i + 1, feature_dim):
            h = (
                (i + 1) * 0x9E3779B185EBCA87
                ^ (j + 1) * 0xC2B2AE3D27D4EB4F
                ^ hash_dim * 0x165667B19E3779F9
            ) & mask64
            left.append(i)
            right.append(j)
            buckets.append(h % hash_dim)
            signs.append(1.0 if ((h >> 63) & 1) == 0 else -1.0)

    return (
        jnp.asarray(left, dtype=jnp.int32),
        jnp.asarray(right, dtype=jnp.int32),
        jnp.asarray(buckets, dtype=jnp.int32),
        jnp.asarray(signs, dtype=jnp.float32),
    )


def hashed_quadratic_lift(observations: jax.Array, hash_dim: int) -> jax.Array:
    """Append signed hashed pair products to observations."""
    left, right, buckets, signs = pair_hash_arrays(int(observations.shape[1]), hash_dim)

    def lift_one(obs: jax.Array) -> jax.Array:
        products = obs[left] * obs[right] * signs
        hashed = jnp.zeros(hash_dim, dtype=obs.dtype).at[buckets].add(products)
        return jnp.concatenate([obs, hashed], axis=0)

    return jax.vmap(lift_one)(observations)


def maybe_lift(observations: jax.Array, hash_dim: int | None) -> jax.Array:
    """Apply the quadratic lift when requested."""
    if hash_dim is None:
        return observations
    return hashed_quadratic_lift(observations, hash_dim)


def make_learner(n_heads: int, spec: MethodSpec, config: ScaleConfig) -> MultiHeadMLPLearner:
    """Construct a MultiHeadMLPLearner for one method."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=spec.hidden_sizes,
        step_size=config.learner_step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=0.0 if spec.hidden_sizes == () else config.mlp_sparsity,
        use_layer_norm=spec.hidden_sizes != (),
    )


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a ScanStream-like source."""
    state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(step_fn, state, jnp.arange(num_steps))
    return observations, targets


def regression_loss_curve(
    learner: MultiHeadMLPLearner,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> np.ndarray:
    """Run a vector regression learner and return per-step mean head loss."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    per_head = np.asarray(result.per_head_metrics)
    return np.nanmean(per_head[:, :, 0], axis=1)


def run_classifier_stream(
    learner: MultiHeadMLPLearner,
    key: jax.Array,
    observations: jax.Array,
    labels: jax.Array,
    target_code: str,
) -> tuple[Any, np.ndarray]:
    """Run online classification and return final state plus [mse, correct]."""
    targets = jnp.asarray(targets_for(np.asarray(labels), target_code))
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        mse = jnp.mean((result.predictions - target) ** 2)
        correct = jnp.argmax(result.predictions) == label
        return result.state, jnp.stack([mse, correct.astype(jnp.float32)])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def evaluate_classifier(
    learner: MultiHeadMLPLearner,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_code: str,
    hash_dim: int | None,
) -> dict[str, float]:
    """Evaluate a classifier on held-out final-phase data."""
    observations = maybe_lift(jnp.asarray(x_test.astype(np.float32)), hash_dim)
    labels = jnp.asarray(y_test.astype(np.int32))
    targets = jnp.asarray(targets_for(y_test, target_code))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def final_window(values: np.ndarray, window: int) -> float:
    """Return final-window mean."""
    return float(np.mean(values[-min(window, values.shape[0]) :]))


def stderr(values: Sequence[float]) -> float:
    """Return standard error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def mean_row(values: Sequence[float]) -> dict[str, Any]:
    """Summarize one metric over seeds."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "stderr": stderr(arr), "per_seed": arr.tolist()}


def paired_lower(method: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    """Summarize lower-is-better paired differences; positive favors method."""
    diff = np.asarray(baseline, dtype=np.float64) - np.asarray(method, dtype=np.float64)
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_stderr": stderr(diff),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n": int(diff.shape[0]),
        "diffs": diff.tolist(),
    }


def paired_higher(method: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    """Summarize higher-is-better paired differences; positive favors method."""
    diff = np.asarray(method, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_stderr": stderr(diff),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n": int(diff.shape[0]),
        "diffs": diff.tolist(),
    }


def load_digits_split(seed: int, train_fraction: float) -> DigitsSplit:
    """Load and standardize sklearn digits using train-split statistics."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required; install `.[external]`.") from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for cls in range(N_CLASSES):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        n_train = max(1, min(len(cls_idx) - 1, n_train))
        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    train_idx = np.asarray(train_indices, dtype=np.int32)
    test_idx = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    mean = x_train.mean(axis=0, keepdims=True)
    train_std = x_train.std(axis=0, keepdims=True)
    std = np.where(train_std < 1e-6, 1.0, train_std)
    return DigitsSplit(
        x_train=((x_train - mean) / std).astype(np.float32),
        y_train=y_train,
        x_test=((x_test - mean) / std).astype(np.float32),
        y_test=y_test,
        meta={
            "dataset": "sklearn.datasets.load_digits",
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
        },
    )


def make_digit_base_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    class_blocked: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an online sequence from repeated shuffled train-set epochs."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_CLASSES)]
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    while total < steps:
        if class_blocked:
            parts: list[np.ndarray] = []
            for cls in rng.permutation(N_CLASSES):
                idx = class_indices[int(cls)].copy()
                rng.shuffle(idx)
                parts.append(idx)
            indices = np.concatenate(parts)
        else:
            indices = rng.permutation(len(y_train))
        chunks_x.append(x_train[indices])
        chunks_y.append(y_train[indices])
        total += len(indices)
    return (
        np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32),
        np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32),
    )


def make_digit_stream(
    regime: str,
    split: DigitsSplit,
    config: ScaleConfig,
    seed: int,
) -> DigitStream:
    """Create one digits online stream."""
    observations, labels = make_digit_base_sequence(
        split.x_train,
        split.y_train,
        config.num_steps,
        seed,
        class_blocked=regime == "class_blocked",
    )
    x_test = split.x_test.copy()
    y_test = split.y_test.copy()
    phase_ids = np.arange(config.num_steps, dtype=np.int32) // config.phase_length
    n_phases = int(math.ceil(config.num_steps / config.phase_length))
    final_phase = int(phase_ids[-1])
    rng = np.random.default_rng(seed + 50_000)
    meta = {"regime": regime, "n_phases": n_phases, "final_phase": final_phase}

    if regime == "label_drift":
        perms = [np.arange(N_CLASSES, dtype=np.int32)]
        perms.extend(rng.permutation(N_CLASSES).astype(np.int32) for _ in range(1, n_phases))
        for phase, perm in enumerate(perms):
            labels[phase_ids == phase] = perm[labels[phase_ids == phase]]
        y_test = perms[final_phase][y_test]
        meta["description"] = "class-head labels are permuted by phase"
    elif regime == "permuted_pixels":
        perms = [np.arange(observations.shape[1], dtype=np.int32)]
        perms.extend(
            rng.permutation(observations.shape[1]).astype(np.int32)
            for _ in range(1, n_phases)
        )
        for phase, perm in enumerate(perms):
            observations[phase_ids == phase] = observations[phase_ids == phase][:, perm]
        x_test = x_test[:, perms[final_phase]]
        meta["description"] = "pixel order is permuted by phase"
    elif regime == "class_blocked":
        meta["description"] = "samples are grouped into class blocks"
    else:
        meta["description"] = "stationary shuffled iid control"

    return DigitStream(
        observations=jnp.asarray(observations),
        labels=jnp.asarray(labels),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
        meta=meta,
    )


def regression_stream_factory(name: str, config: ScaleConfig) -> tuple[Any, str]:
    """Create one regression benchmark stream."""
    if name == "interaction":
        return (
            InteractionFeatureDiscoveryStream(
                feature_dim=config.feature_dim,
                n_tasks=config.n_tasks,
                n_contexts=config.n_contexts,
                context_length=config.context_length,
                active_pairs_per_context=config.interaction_active_pairs,
                noise_std=config.noise_std,
                linear_scale=config.linear_scale,
            ),
            "pair-product oracle stream where quadratic lifts should help",
        )
    if name == "nonlinear":
        return (
            NonlinearFeatureDiscoveryStream(
                feature_dim=config.feature_dim,
                n_tasks=config.n_tasks,
                n_latents=config.nonlinear_latents,
                n_contexts=config.n_contexts,
                context_length=config.context_length,
                active_latents_per_context=config.nonlinear_active_latents,
                noise_std=config.noise_std,
                linear_scale=config.linear_scale,
            ),
            "tanh-latent negative control not generated by quadratic features",
        )
    raise ValueError(f"unknown regression suite {name!r}")


def regression_methods(config: ScaleConfig) -> dict[str, MethodSpec]:
    """Return regression method grid."""
    methods = {
        "mlp_h64": MethodSpec(hidden_sizes=(64,), hash_dim=None),
        "mlp_h128": MethodSpec(hidden_sizes=(128,), hash_dim=None),
        "mlp_h64_64": MethodSpec(hidden_sizes=(64, 64), hash_dim=None),
        "quad_mlp_h8": MethodSpec(hidden_sizes=(8,), hash_dim=max(config.hash_dims)),
    }
    for dim in config.hash_dims:
        methods[f"quad_linear_h{dim}"] = MethodSpec(hidden_sizes=(), hash_dim=dim)
    return methods


def digit_methods(config: ScaleConfig) -> dict[str, MethodSpec]:
    """Return digits method grid."""
    hash_dim = max(config.hash_dims)
    return {
        "mlp_one_hot_h64": MethodSpec(hidden_sizes=(64,), hash_dim=None, target_code="one_hot"),
        "mlp_one_hot_h128": MethodSpec(hidden_sizes=(128,), hash_dim=None, target_code="one_hot"),
        "mlp_centered_h64": MethodSpec(hidden_sizes=(64,), hash_dim=None, target_code="centered"),
        "quad_linear_centered": MethodSpec(
            hidden_sizes=(),
            hash_dim=hash_dim,
            target_code="centered",
        ),
        "quad_mlp_centered_h8": MethodSpec(
            hidden_sizes=(8,),
            hash_dim=hash_dim,
            target_code="centered",
        ),
    }


def run_regression_suite(name: str, config: ScaleConfig) -> dict[str, Any]:
    """Run one vector-regression suite."""
    stream, description = regression_stream_factory(name, config)
    methods = regression_methods(config)
    finals: dict[str, list[float]] = {method: [] for method in methods}
    means: dict[str, list[float]] = {method: [] for method in methods}

    for seed in range(config.num_seeds):
        root_key = jr.key(seed + 10_000 * (1 + list(config.regression_suites).index(name)))
        keys = jr.split(root_key, len(methods) + 1)
        observations, targets = collect_stream_arrays(stream, config.num_steps, keys[0])
        for idx, (method_name, spec) in enumerate(methods.items(), start=1):
            method_obs = maybe_lift(observations, spec.hash_dim)
            learner = make_learner(config.n_tasks, spec, config)
            curve = regression_loss_curve(learner, method_obs, targets, keys[idx])
            finals[method_name].append(final_window(curve, config.final_window))
            means[method_name].append(float(np.mean(curve)))
        print(f"regression={name} seed={seed} complete")

    per_method = {
        method: {
            "final_window_loss": mean_row(finals[method]),
            "mean_loss": mean_row(means[method]),
        }
        for method in methods
    }
    mlp_methods = [m for m in methods if m.startswith("mlp_")]
    best_mlp = min(mlp_methods, key=lambda m: per_method[m]["final_window_loss"]["mean"])
    paired_vs_best_mlp = {
        method: paired_lower(finals[method], finals[best_mlp])
        for method in methods
        if method != best_mlp
    }
    best_candidate = min(
        [m for m in methods if not m.startswith("mlp_")],
        key=lambda m: per_method[m]["final_window_loss"]["mean"],
    )
    candidate_pair = paired_vs_best_mlp[best_candidate]
    return {
        "description": description,
        "per_method": per_method,
        "best_mlp": best_mlp,
        "best_candidate": best_candidate,
        "paired_vs_best_mlp": paired_vs_best_mlp,
        "candidate_beats_best_mlp": bool(
            candidate_pair["diff_mean"] > 0.0 and candidate_pair["wins"] > candidate_pair["losses"]
        ),
    }


def summarize_classifier_metrics(metrics: np.ndarray, final_window_size: int) -> dict[str, float]:
    """Summarize [mse, correct] metrics."""
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "final_window_mse": final_window(metrics[:, 0], final_window_size),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_accuracy": final_window(metrics[:, 1], final_window_size),
    }


def run_digits_regime(regime: str, config: ScaleConfig) -> dict[str, Any]:
    """Run one digits regime."""
    methods = digit_methods(config)
    rows: list[dict[str, Any]] = []
    metrics_by_method: dict[str, dict[str, list[float]]] = {
        method: {
            "final_window_accuracy": [],
            "test_accuracy": [],
            "final_window_mse": [],
            "test_mse": [],
        }
        for method in methods
    }

    for seed in range(config.num_seeds):
        split = load_digits_split(seed, config.train_fraction)
        stream = make_digit_stream(regime, split, config, seed + 20_000)
        root_key = jr.key(seed + 30_000)
        keys = jr.split(root_key, len(methods))
        for idx, (method_name, spec) in enumerate(methods.items()):
            obs = maybe_lift(stream.observations, spec.hash_dim)
            learner = make_learner(N_CLASSES, spec, config)
            assert spec.target_code is not None
            state, curve_metrics = run_classifier_stream(
                learner,
                keys[idx],
                obs,
                stream.labels,
                spec.target_code,
            )
            summary = summarize_classifier_metrics(curve_metrics, config.final_window)
            summary.update(
                evaluate_classifier(
                    learner,
                    state,
                    stream.x_test,
                    stream.y_test,
                    spec.target_code,
                    spec.hash_dim,
                )
            )
            rows.append({"seed": seed, "method": method_name, **summary})
            for metric in metrics_by_method[method_name]:
                metrics_by_method[method_name][metric].append(summary[metric])
        print(f"digits={regime} seed={seed} complete")

    per_method = {
        method: {metric: mean_row(values) for metric, values in method_metrics.items()}
        for method, method_metrics in metrics_by_method.items()
    }
    mlp_methods = [m for m in methods if m.startswith("mlp_one_hot")]
    best_mlp = max(mlp_methods, key=lambda m: per_method[m]["final_window_accuracy"]["mean"])
    candidate_methods = [m for m in methods if not m.startswith("mlp_one_hot")]
    best_candidate = max(
        candidate_methods,
        key=lambda m: per_method[m]["final_window_accuracy"]["mean"],
    )
    paired_vs_best_mlp = {
        method: {
            "final_window_accuracy": paired_higher(
                metrics_by_method[method]["final_window_accuracy"],
                metrics_by_method[best_mlp]["final_window_accuracy"],
            ),
            "test_accuracy": paired_higher(
                metrics_by_method[method]["test_accuracy"],
                metrics_by_method[best_mlp]["test_accuracy"],
            ),
        }
        for method in methods
        if method != best_mlp
    }
    candidate_pair = paired_vs_best_mlp[best_candidate]["final_window_accuracy"]
    return {
        "description": make_digit_stream(
            regime,
            load_digits_split(0, config.train_fraction),
            config,
            99,
        ).meta["description"],
        "rows": rows,
        "per_method": per_method,
        "best_mlp": best_mlp,
        "best_candidate": best_candidate,
        "paired_vs_best_mlp": paired_vs_best_mlp,
        "candidate_beats_best_mlp": bool(
            candidate_pair["diff_mean"] > 0.0 and candidate_pair["wins"] > candidate_pair["losses"]
        ),
    }


def run(config: ScaleConfig) -> dict[str, Any]:
    """Run all configured suites."""
    t0 = time.time()
    regression = {
        name: run_regression_suite(name, config)
        for name in config.regression_suites
    }
    digits = {regime: run_digits_regime(regime, config) for regime in config.digits_regimes}
    all_flags = [suite["candidate_beats_best_mlp"] for suite in regression.values()]
    all_flags.extend(regime["candidate_beats_best_mlp"] for regime in digits.values())
    return {
        "experiment": "step2_quadratic_centered_scale",
        "config": asdict(config),
        "elapsed_s": float(time.time() - t0),
        "regression": regression,
        "digits": digits,
        "universal_candidate_win": bool(all(all_flags)),
        "suite_win_count": int(sum(all_flags)),
        "suite_count": int(len(all_flags)),
    }


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format a mean +/- stderr cell."""
    data = row[metric]
    return f"{data['mean']:.4f} +/- {data['stderr']:.4f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    lines = [
        "# Step 2 Quadratic + Centered Scaling",
        "",
        "Goal: scale the strongest moonshot signal without weakening the comparator. "
        "Candidates are compared against the best fair MLP in each suite.",
        "",
        f"Universal candidate win: `{payload['universal_candidate_win']}` "
        f"({payload['suite_win_count']}/{payload['suite_count']} suites).",
        "",
        "## Regression Suites",
        "",
    ]
    for name, suite in payload["regression"].items():
        lines.extend([
            f"### {name}",
            "",
            suite["description"],
            "",
            f"Best MLP: `{suite['best_mlp']}`. Best candidate: `{suite['best_candidate']}`. "
            f"Candidate beats best MLP: `{suite['candidate_beats_best_mlp']}`.",
            "",
            "| Method | Final-window loss | Mean loss |",
            "|---|---:|---:|",
        ])
        for method, data in sorted(
            suite["per_method"].items(),
            key=lambda item: item[1]["final_window_loss"]["mean"],
        ):
            lines.append(
                f"| `{method}` | {metric_cell(data, 'final_window_loss')} | "
                f"{metric_cell(data, 'mean_loss')} |"
            )
        lines.append("")

    lines.extend(["## Digits Regimes", ""])
    for regime, suite in payload["digits"].items():
        lines.extend([
            f"### {regime}",
            "",
            suite["description"],
            "",
            f"Best MLP: `{suite['best_mlp']}`. Best candidate: `{suite['best_candidate']}`. "
            f"Candidate beats best MLP: `{suite['candidate_beats_best_mlp']}`.",
            "",
            "| Method | Final-window acc | Test acc | Final-window MSE |",
            "|---|---:|---:|---:|",
        ])
        for method, data in sorted(
            suite["per_method"].items(),
            key=lambda item: item[1]["final_window_accuracy"]["mean"],
            reverse=True,
        ):
            lines.append(
                f"| `{method}` | {metric_cell(data, 'final_window_accuracy')} | "
                f"{metric_cell(data, 'test_accuracy')} | "
                f"{metric_cell(data, 'final_window_mse')} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "A universal claim requires candidate wins in every configured suite. "
        "A partial win identifies the stream family where the mechanism is real "
        "and the counterexamples where the next iteration must improve.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any], output_dir: Path, note_path: Path) -> None:
    """Write JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_summary(output_dir / "SUMMARY.md", payload)
    write_summary(note_path, payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--num-steps", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--final-window", type=int, default=None)
    parser.add_argument("--hash-dims", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--regression-suites", nargs="+", default=["interaction", "nonlinear"])
    parser.add_argument(
        "--digits-regimes",
        nargs="+",
        default=["iid", "label_drift", "permuted_pixels"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the scaling experiment."""
    args = parse_args()
    final_window_size = args.final_window or max(1, args.num_steps // 5)
    config = ScaleConfig(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        final_window=final_window_size,
        hash_dims=tuple(args.hash_dims),
        regression_suites=tuple(args.regression_suites),
        digits_regimes=tuple(args.digits_regimes),
    )
    payload = run(config)
    write_outputs(payload, args.output_dir, args.note_path)
    print(
        f"universal_candidate_win={payload['universal_candidate_win']} "
        f"suite_wins={payload['suite_win_count']}/{payload['suite_count']}"
    )
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

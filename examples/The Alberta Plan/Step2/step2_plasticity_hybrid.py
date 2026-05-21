#!/usr/bin/env python3
"""Worker S2E Step 2 plasticity and hybrid benchmark.

This script tests plasticity mechanisms against a fair MLP baseline on:

* synthetic out-of-hypothesis-class streams from Step 2; and
* externally grounded sklearn digits online streams.

Methods share the same hidden trunk, base step size, sparsity, layer norm, and
ObGD bounding. Plasticity variants add only the advertised mechanism:

* ``mlp``: ordinary ``MultiHeadMLPLearner``.
* ``upgd``: UPGD perturbations.
* ``upgd_low_noise`` / scheduled UPGD variants: lower or delayed
  perturbation intended to preserve current-task tracking while retaining
  UPGD plasticity.
* ``cbp``: Continual Backprop low-utility unit replacement.
* ``upgd_reset_hybrid``: UPGD plus a local CBP-style reset of mature,
  low-utility hidden units, implemented in this experiment script without core
  library edits.

The hybrid uses UPGD's per-weight utilities, aggregates them to per-unit
utility, and periodically reinitializes the lowest-utility mature unit in each
hidden layer while zeroing that unit's outgoing weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    CBPMultiHeadMLPLearner,
    CompositionalStream,
    ContinualBackpropConfig,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
    run_cbp_learning_loop,
    run_multi_head_learning_loop,
    run_upgd_arrays,
)
from alberta_framework.core.initializers import sparse_init  # noqa: E402
from alberta_framework.core.types import MLPParams  # noqa: E402
from alberta_framework.core.upgd import UPGDState  # noqa: E402

N_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("output/worker_s2e_plasticity")
MethodName = Literal[
    "mlp",
    "upgd",
    "upgd_low_noise",
    "upgd_tiny_noise",
    "upgd_delayed_low_noise",
    "upgd_ramped_low_noise",
    "cbp",
    "cbp_fast",
    "upgd_reset_hybrid",
    "upgd_reset_hybrid_fast",
]


@chex.dataclass(frozen=True)
class HybridState:
    """UPGD state plus local CBP-style reset tracker."""

    upgd_state: UPGDState
    ages: tuple[Array, ...]
    replacement_accumulators: Array
    reset_key: Array


@chex.dataclass(frozen=True)
class HybridLearningResult:
    """Result from the script-local UPGD plus reset hybrid."""

    state: HybridState
    metrics: Array
    replacements: Array


SYNTHETIC_STREAM_SPECS: list[dict[str, Any]] = [
    {
        "name": "out_of_class_polynomial",
        "feature_dim": 8,
        "n_tasks": 3,
        "factory": lambda: OutOfClassPolynomialStream(
            feature_dim=8,
            n_tasks=3,
            n_contexts=4,
            context_length=400,
            active_triples_per_context=2,
            noise_std=0.05,
        ),
    },
    {
        "name": "frequency_mismatch",
        "feature_dim": 4,
        "n_tasks": 2,
        "factory": lambda: FrequencyMismatchStream(
            feature_dim=4,
            n_tasks=2,
            n_components_per_task=3,
            n_contexts=4,
            context_length=400,
            noise_std=0.05,
        ),
    },
    {
        "name": "compositional",
        "feature_dim": 6,
        "n_tasks": 3,
        "factory": lambda: CompositionalStream(
            feature_dim=6,
            n_tasks=3,
            inner_hidden=4,
            outer_components=5,
            n_contexts=4,
            context_length=400,
            noise_std=0.05,
        ),
    },
]


DIGITS_VARIANTS = ("iid", "class_blocked", "permuted_blocks")


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: Array,
) -> tuple[Array, Array]:
    """Materialize a scan-compatible stream into observation and target arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: Array) -> tuple[Any, tuple[Array, Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    return observations, targets


def make_mlp(
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
) -> MultiHeadMLPLearner:
    """Create the fair MLP baseline used by all comparisons."""
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
    """Create an architecture-matched UPGD learner."""
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


def make_cbp(
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    replacement_rate: float,
    maturity_threshold: int,
) -> CBPMultiHeadMLPLearner:
    """Create an architecture-matched CBP learner."""
    return CBPMultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        cbp_config=ContinualBackpropConfig(
            decay_rate=0.99,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            enabled=True,
        ),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )


def _mlp_loss_curve(per_head_metrics: Array) -> Array:
    """Average active-head squared error into one per-step curve."""
    return jnp.nanmean(per_head_metrics[..., 0], axis=-1)


def run_mlp_loss_curve(
    learner: MultiHeadMLPLearner,
    key: Array,
    observations: Array,
    targets: Array,
) -> Array:
    """Run MLP on array data and return a per-step MSE curve."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    return _mlp_loss_curve(result.per_head_metrics)


def run_upgd_loss_curve(
    learner: UPGDLearner,
    key: Array,
    observations: Array,
    targets: Array,
) -> Array:
    """Run UPGD on array data and return a per-step MSE curve."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_upgd_arrays(learner, state, observations, targets)
    return result.metrics[:, 0]


def run_cbp_loss_curve(
    learner: CBPMultiHeadMLPLearner,
    key: Array,
    observations: Array,
    targets: Array,
) -> tuple[Array, Array]:
    """Run CBP on array data and return per-step MSE and replacement counts."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_cbp_learning_loop(learner, state, observations, targets)
    return _mlp_loss_curve(result.per_head_metrics), jnp.sum(
        result.replacements_made.astype(jnp.int32),
        axis=-1,
    )


def _unit_utilities(weight_utilities: tuple[Array, ...]) -> tuple[Array, ...]:
    """Aggregate UPGD per-weight utilities to per-hidden-unit utilities."""
    return tuple(jnp.mean(u, axis=1) for u in weight_utilities)


def _replace_one_upgd_unit(
    state: UPGDState,
    layer_idx: int,
    unit_idx: Array,
    do_replace: Array,
    reset_key: Array,
    sparsity: float,
) -> UPGDState:
    """Reinitialize one hidden unit in an UPGD trunk, guarded by ``do_replace``."""
    n_layers = len(state.trunk_params.weights)
    weight_layer = state.trunk_params.weights[layer_idx]
    fan_in = weight_layer.shape[1]
    sampled = sparse_init(reset_key, (1, fan_in), sparsity=sparsity)
    new_row = jnp.where(do_replace, sampled[0], weight_layer[unit_idx])
    new_weight_layer = weight_layer.at[unit_idx].set(new_row)

    bias_layer = state.trunk_params.biases[layer_idx]
    new_bias = jnp.where(do_replace, jnp.float32(0.0), bias_layer[unit_idx])
    new_bias_layer = bias_layer.at[unit_idx].set(new_bias)

    new_trunk_weights = list(state.trunk_params.weights)
    new_trunk_biases = list(state.trunk_params.biases)
    new_trunk_weights[layer_idx] = new_weight_layer
    new_trunk_biases[layer_idx] = new_bias_layer

    new_head_weights = list(state.head_params.weights)
    if layer_idx < n_layers - 1:
        next_weight_layer = new_trunk_weights[layer_idx + 1]
        zero_col = jnp.zeros(next_weight_layer.shape[0], dtype=next_weight_layer.dtype)
        new_col = jnp.where(do_replace, zero_col, next_weight_layer[:, unit_idx])
        new_trunk_weights[layer_idx + 1] = next_weight_layer.at[:, unit_idx].set(new_col)
    else:
        for head_idx in range(len(new_head_weights)):
            head_weight = new_head_weights[head_idx]
            zero_col = jnp.zeros(head_weight.shape[0], dtype=head_weight.dtype)
            new_col = jnp.where(do_replace, zero_col, head_weight[:, unit_idx])
            new_head_weights[head_idx] = head_weight.at[:, unit_idx].set(new_col)

    new_utilities = list(state.utilities)
    utility_layer = new_utilities[layer_idx]
    zero_row = jnp.zeros(utility_layer.shape[1], dtype=utility_layer.dtype)
    new_utility_row = jnp.where(do_replace, zero_row, utility_layer[unit_idx])
    new_utilities[layer_idx] = utility_layer.at[unit_idx].set(new_utility_row)

    return state.replace(  # type: ignore[attr-defined]
        trunk_params=MLPParams(  # type: ignore[call-arg]
            weights=tuple(new_trunk_weights),
            biases=tuple(new_trunk_biases),
        ),
        head_params=MLPParams(  # type: ignore[call-arg]
            weights=tuple(new_head_weights),
            biases=state.head_params.biases,
        ),
        utilities=tuple(new_utilities),
    )


def _maybe_reset_upgd_units(
    hybrid_state: HybridState,
    replacement_rate: float,
    maturity_threshold: int,
    sparsity: float,
) -> tuple[HybridState, Array]:
    """Apply one CBP-style reset pass to an UPGD state."""
    n_layers = len(hybrid_state.upgd_state.trunk_params.weights)
    if n_layers == 0:
        return hybrid_state, jnp.zeros((1,), dtype=jnp.bool_)

    unit_utilities = _unit_utilities(hybrid_state.upgd_state.utilities)
    new_state = hybrid_state.upgd_state
    new_ages = [age + 1 for age in hybrid_state.ages]
    accumulators = hybrid_state.replacement_accumulators
    new_accumulators: list[Array] = []
    reset_flags: list[Array] = []
    reset_key = hybrid_state.reset_key

    for layer_idx in range(n_layers):
        layer_size = unit_utilities[layer_idx].shape[0]
        accum = accumulators[layer_idx] + jnp.float32(replacement_rate * layer_size)
        budgeted = accum >= 1.0
        mature = new_ages[layer_idx] >= maturity_threshold
        masked_utility = jnp.where(mature, unit_utilities[layer_idx], jnp.inf)
        has_candidate = jnp.any(mature)
        unit_idx = jnp.argmin(masked_utility).astype(jnp.int32)
        do_replace = jnp.logical_and(budgeted, has_candidate)
        reset_key, subkey = jr.split(reset_key)
        new_state = _replace_one_upgd_unit(
            new_state,
            layer_idx,
            unit_idx,
            do_replace,
            subkey,
            sparsity,
        )
        age_layer = new_ages[layer_idx]
        replacement_age = jnp.where(do_replace, jnp.int32(0), age_layer[unit_idx])
        new_ages[layer_idx] = age_layer.at[unit_idx].set(replacement_age)
        new_accumulators.append(jnp.where(do_replace, accum - 1.0, accum))
        reset_flags.append(do_replace)

    return (
        HybridState(  # type: ignore[call-arg]
            upgd_state=new_state,
            ages=tuple(new_ages),
            replacement_accumulators=jnp.stack(new_accumulators),
            reset_key=reset_key,
        ),
        jnp.stack(reset_flags),
    )


def run_upgd_reset_hybrid_arrays(
    learner: UPGDLearner,
    key: Array,
    observations: Array,
    targets: Array,
    replacement_rate: float,
    maturity_threshold: int,
    sparsity: float,
) -> HybridLearningResult:
    """Run UPGD plus script-local CBP-style unit resets over arrays."""
    init_key, reset_key = jr.split(key)
    upgd_state = learner.init(int(observations.shape[1]), init_key)
    ages = tuple(
        jnp.zeros(w.shape[0], dtype=jnp.int32)
        for w in upgd_state.trunk_params.weights
    )
    hybrid_state = HybridState(  # type: ignore[call-arg]
        upgd_state=upgd_state,
        ages=ages,
        replacement_accumulators=jnp.zeros(len(ages), dtype=jnp.float32),
        reset_key=reset_key,
    )

    def step_fn(
        carry: HybridState,
        inputs: tuple[Array, Array],
    ) -> tuple[HybridState, tuple[Array, Array]]:
        obs, target = inputs
        result = learner.update(carry.upgd_state, obs, target)
        updated = carry.replace(upgd_state=result.state)  # type: ignore[attr-defined]
        updated, reset_flags = _maybe_reset_upgd_units(
            updated,
            replacement_rate,
            maturity_threshold,
            sparsity,
        )
        return updated, (result.metrics, reset_flags)

    final_state, (metrics, replacements) = jax.lax.scan(
        step_fn,
        hybrid_state,
        (observations, targets),
    )
    return HybridLearningResult(  # type: ignore[call-arg]
        state=final_state,
        metrics=metrics,
        replacements=replacements,
    )


def summarize_loss_curve(curve: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize a one-dimensional loss curve."""
    window = min(final_window, curve.shape[0])
    return {
        "mean_mse": float(np.mean(curve)),
        "final_window_mse": float(np.mean(curve[-window:])),
        "last_mse": float(curve[-1]),
    }


def stderr(values: np.ndarray) -> float:
    """Standard error of a vector."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def cohens_d(diffs: np.ndarray) -> float:
    """Paired Cohen's d."""
    if diffs.shape[0] <= 1:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(diffs) / sd)


def method_names(include_tuned: bool) -> list[MethodName]:
    """Return method list for the requested grid size."""
    names: list[MethodName] = ["mlp", "upgd", "cbp", "upgd_reset_hybrid"]
    if include_tuned:
        names.extend([
            "upgd_low_noise",
            "upgd_tiny_noise",
            "upgd_delayed_low_noise",
            "upgd_ramped_low_noise",
            "cbp_fast",
            "upgd_reset_hybrid_fast",
        ])
    return names


def upgd_hyperparams_for_method(
    method: MethodName,
    args: argparse.Namespace,
) -> tuple[float, int, int]:
    """Return ``(sigma, warmup_steps, ramp_steps)`` for a UPGD method."""
    if method == "upgd_low_noise":
        return args.upgd_low_noise_sigma, 0, 0
    if method == "upgd_tiny_noise":
        return args.upgd_tiny_noise_sigma, 0, 0
    if method == "upgd_delayed_low_noise":
        return args.upgd_low_noise_sigma, args.upgd_warmup_steps, 0
    if method == "upgd_ramped_low_noise":
        return (
            args.upgd_low_noise_sigma,
            args.upgd_warmup_steps,
            args.upgd_ramp_steps,
        )
    return args.upgd_sigma, 0, 0


def run_synthetic_method(
    method: MethodName,
    n_tasks: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    key: Array,
    observations: Array,
    targets: Array,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    """Run one method on one synthetic stream realization."""
    if method == "mlp":
        learner = make_mlp(n_tasks, hidden_sizes, step_size, sparsity)
        curve = run_mlp_loss_curve(learner, key, observations, targets)
        replacements = jnp.array(0)
    elif method in {
        "upgd",
        "upgd_low_noise",
        "upgd_tiny_noise",
        "upgd_delayed_low_noise",
        "upgd_ramped_low_noise",
    }:
        sigma, warmup_steps, ramp_steps = upgd_hyperparams_for_method(method, args)
        learner = make_upgd(
            n_tasks,
            hidden_sizes,
            step_size,
            sparsity,
            sigma,
            warmup_steps,
            ramp_steps,
        )
        curve = run_upgd_loss_curve(learner, key, observations, targets)
        replacements = jnp.array(0)
    elif method in {"cbp", "cbp_fast"}:
        rate = args.cbp_fast_replacement_rate if method == "cbp_fast" else args.cbp_replacement_rate
        learner = make_cbp(
            n_tasks,
            hidden_sizes,
            step_size,
            sparsity,
            rate,
            args.maturity_threshold,
        )
        curve, replacement_curve = run_cbp_loss_curve(learner, key, observations, targets)
        replacements = jnp.sum(replacement_curve)
    elif method in {"upgd_reset_hybrid", "upgd_reset_hybrid_fast"}:
        rate = (
            args.hybrid_fast_replacement_rate
            if method == "upgd_reset_hybrid_fast"
            else args.hybrid_replacement_rate
        )
        learner = make_upgd(n_tasks, hidden_sizes, step_size, sparsity, args.upgd_sigma)
        result = run_upgd_reset_hybrid_arrays(
            learner,
            key,
            observations,
            targets,
            rate,
            args.maturity_threshold,
            sparsity,
        )
        curve = result.metrics[:, 0]
        replacements = jnp.sum(result.replacements.astype(jnp.int32))
    else:
        raise ValueError(f"unknown method {method}")

    curve.block_until_ready()
    replacements.block_until_ready()
    return np.asarray(curve), int(replacements)


def run_synthetic_suite(
    args: argparse.Namespace,
    methods: list[MethodName],
) -> list[dict[str, Any]]:
    """Run all synthetic out-of-class scenarios."""
    records: list[dict[str, Any]] = []
    hidden_sizes = (args.hidden_size,)

    for spec in SYNTHETIC_STREAM_SPECS:
        if args.synthetic_stream != "all" and args.synthetic_stream != spec["name"]:
            continue
        print(f"\n[SYNTHETIC] {spec['name']}")
        stream = spec["factory"]()
        for seed_idx in range(args.n_seeds):
            seed = args.seed + seed_idx
            root = jr.key(seed)
            split_keys = jr.split(root, len(methods) + 1)
            observations, targets = collect_stream_arrays(
                stream,
                args.synthetic_steps,
                split_keys[0],
            )
            observations.block_until_ready()
            targets.block_until_ready()
            for method_idx, method in enumerate(methods):
                t0 = time.time()
                curve, replacements = run_synthetic_method(
                    method,
                    int(spec["n_tasks"]),
                    hidden_sizes,
                    args.step_size,
                    args.sparsity,
                    split_keys[method_idx + 1],
                    observations,
                    targets,
                    args,
                )
                summary = summarize_loss_curve(curve, args.final_window)
                records.append(
                    {
                        "family": "synthetic",
                        "scenario": spec["name"],
                        "seed": seed,
                        "method": method,
                        "metrics": summary,
                        "replacements": replacements,
                        "wall_clock_s": time.time() - t0,
                    }
                )
                print(
                    f"  seed={seed} {method}: "
                    f"final_mse={summary['final_window_mse']:.5f} "
                    f"mean_mse={summary['mean_mse']:.5f} resets={replacements}"
                )
    return records


def load_digits_arrays(
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load sklearn digits and standardize from train-set statistics."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        msg = "scikit-learn is required; install the project's external extra."
        raise RuntimeError(msg) from exc

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
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)
    meta = {
        "dataset": "sklearn.datasets.load_digits",
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_CLASSES,
        "train_fraction": train_fraction,
    }
    return x_train, y_train, x_test, y_test, meta


def make_digits_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    variant: str,
    block_size: int,
) -> tuple[Array, Array, Array, list[np.ndarray]]:
    """Create a digits online sequence with optional distribution shifts."""
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_CLASSES)]

    permutations = [
        np.arange(x_train.shape[1], dtype=np.int32),
        rng.permutation(x_train.shape[1]).astype(np.int32),
        rng.permutation(x_train.shape[1]).astype(np.int32),
        rng.permutation(x_train.shape[1]).astype(np.int32),
    ]
    block_id = 0
    while total < steps:
        if variant == "class_blocked":
            epoch_parts: list[np.ndarray] = []
            for cls in rng.permutation(N_CLASSES):
                cls_idx = class_indices[int(cls)].copy()
                rng.shuffle(cls_idx)
                epoch_parts.append(cls_idx)
            indices = np.concatenate(epoch_parts)
        else:
            indices = rng.permutation(len(y_train))

        x_epoch = x_train[indices].copy()
        if variant == "permuted_blocks":
            pieces: list[np.ndarray] = []
            for start in range(0, x_epoch.shape[0], block_size):
                perm = permutations[block_id % len(permutations)]
                pieces.append(x_epoch[start : start + block_size, perm])
                block_id += 1
            x_epoch = np.concatenate(pieces, axis=0)

        chunks_x.append(x_epoch)
        chunks_y.append(y_train[indices])
        total += len(indices)

    observations = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return (
        jnp.asarray(observations),
        jnp.asarray(targets),
        jnp.asarray(labels),
        permutations if variant == "permuted_blocks" else [permutations[0]],
    )


def _digit_metrics(predictions: Array, target: Array, label: Array) -> Array:
    """Return [mse, correct] for a classifier step."""
    mse = jnp.mean((predictions - target) ** 2)
    correct = (jnp.argmax(predictions) == label).astype(jnp.float32)
    return jnp.stack([mse, correct])


def run_mlp_digits(
    learner: MultiHeadMLPLearner,
    key: Array,
    observations: Array,
    targets: Array,
    labels: Array,
) -> tuple[Any, np.ndarray]:
    """Run MLP on a digits online sequence."""
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(carry: Any, inputs: tuple[Array, Array, Array]) -> tuple[Any, Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        return result.state, _digit_metrics(result.predictions, target, label)

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def run_upgd_digits(
    learner: UPGDLearner,
    key: Array,
    observations: Array,
    targets: Array,
    labels: Array,
) -> tuple[Any, np.ndarray]:
    """Run UPGD on a digits online sequence."""
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(carry: UPGDState, inputs: tuple[Array, Array, Array]) -> tuple[UPGDState, Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        return result.state, _digit_metrics(result.predictions, target, label)

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def run_cbp_digits(
    learner: CBPMultiHeadMLPLearner,
    key: Array,
    observations: Array,
    targets: Array,
    labels: Array,
) -> tuple[Any, np.ndarray, int]:
    """Run CBP on a digits online sequence."""
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(carry: Any, inputs: tuple[Array, Array, Array]) -> tuple[Any, tuple[Array, Array]]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        return (
            result.state,
            (
                _digit_metrics(result.predictions, target, label),
                result.replacements_made,
            ),
        )

    final_state, (metrics, replacements) = jax.lax.scan(
        step_fn,
        state,
        (observations, targets, labels),
    )
    metrics.block_until_ready()
    replacements.block_until_ready()
    return final_state, np.asarray(metrics), int(jnp.sum(replacements.astype(jnp.int32)))


def run_hybrid_digits(
    learner: UPGDLearner,
    key: Array,
    observations: Array,
    targets: Array,
    labels: Array,
    replacement_rate: float,
    maturity_threshold: int,
    sparsity: float,
) -> tuple[HybridState, np.ndarray, int]:
    """Run the script-local UPGD plus reset hybrid on digits."""
    init_key, reset_key = jr.split(key)
    upgd_state = learner.init(int(observations.shape[1]), init_key)
    hybrid_state = HybridState(  # type: ignore[call-arg]
        upgd_state=upgd_state,
        ages=tuple(
            jnp.zeros(w.shape[0], dtype=jnp.int32)
            for w in upgd_state.trunk_params.weights
        ),
        replacement_accumulators=jnp.zeros(
            len(upgd_state.trunk_params.weights),
            dtype=jnp.float32,
        ),
        reset_key=reset_key,
    )

    def step_fn(
        carry: HybridState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[HybridState, tuple[Array, Array]]:
        obs, target, label = inputs
        result = learner.update(carry.upgd_state, obs, target)
        updated = carry.replace(upgd_state=result.state)  # type: ignore[attr-defined]
        updated, reset_flags = _maybe_reset_upgd_units(
            updated,
            replacement_rate,
            maturity_threshold,
            sparsity,
        )
        return updated, (_digit_metrics(result.predictions, target, label), reset_flags)

    final_state, (metrics, replacements) = jax.lax.scan(
        step_fn,
        hybrid_state,
        (observations, targets, labels),
    )
    metrics.block_until_ready()
    replacements.block_until_ready()
    return final_state, np.asarray(metrics), int(jnp.sum(replacements.astype(jnp.int32)))


def summarize_digits_metrics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize online classification metrics."""
    window = min(final_window, metrics.shape[0])
    return {
        "mean_mse": float(np.mean(metrics[:, 0])),
        "mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def evaluate_digits(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    permutations: list[np.ndarray],
    is_hybrid: bool = False,
) -> dict[str, float]:
    """Evaluate final classifier averaged over one or more pixel contexts."""
    labels = jnp.asarray(y_test.astype(np.int32))
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[y_test])
    mses: list[float] = []
    accuracies: list[float] = []
    model_state = state.upgd_state if is_hybrid else state
    for perm in permutations:
        observations = jnp.asarray(x_test[:, perm].astype(np.float32))
        preds = jax.vmap(lambda obs: learner.predict(model_state, obs))(observations)
        mse = jnp.mean((preds - targets) ** 2)
        acc = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
        mse.block_until_ready()
        acc.block_until_ready()
        mses.append(float(mse))
        accuracies.append(float(acc))
    return {
        "test_mse": float(np.mean(mses)),
        "test_accuracy": float(np.mean(accuracies)),
    }


def run_digits_method(
    method: MethodName,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    key: Array,
    observations: Array,
    targets: Array,
    labels: Array,
    x_test: np.ndarray,
    y_test: np.ndarray,
    permutations: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, float], int]:
    """Run one method on one digits online sequence."""
    if method == "mlp":
        learner = make_mlp(N_CLASSES, hidden_sizes, step_size, sparsity)
        state, metrics = run_mlp_digits(learner, key, observations, targets, labels)
        summary = summarize_digits_metrics(metrics, args.final_window)
        summary.update(evaluate_digits(learner, state, x_test, y_test, permutations))
        return summary, 0
    if method in {
        "upgd",
        "upgd_low_noise",
        "upgd_tiny_noise",
        "upgd_delayed_low_noise",
        "upgd_ramped_low_noise",
    }:
        sigma, warmup_steps, ramp_steps = upgd_hyperparams_for_method(method, args)
        learner = make_upgd(
            N_CLASSES,
            hidden_sizes,
            step_size,
            sparsity,
            sigma,
            warmup_steps,
            ramp_steps,
        )
        state, metrics = run_upgd_digits(learner, key, observations, targets, labels)
        summary = summarize_digits_metrics(metrics, args.final_window)
        summary.update(evaluate_digits(learner, state, x_test, y_test, permutations))
        return summary, 0
    if method in {"cbp", "cbp_fast"}:
        rate = args.cbp_fast_replacement_rate if method == "cbp_fast" else args.cbp_replacement_rate
        learner = make_cbp(
            N_CLASSES,
            hidden_sizes,
            step_size,
            sparsity,
            rate,
            args.maturity_threshold,
        )
        state, metrics, replacements = run_cbp_digits(
            learner,
            key,
            observations,
            targets,
            labels,
        )
        summary = summarize_digits_metrics(metrics, args.final_window)
        summary.update(evaluate_digits(learner, state, x_test, y_test, permutations))
        return summary, replacements
    if method in {"upgd_reset_hybrid", "upgd_reset_hybrid_fast"}:
        rate = (
            args.hybrid_fast_replacement_rate
            if method == "upgd_reset_hybrid_fast"
            else args.hybrid_replacement_rate
        )
        learner = make_upgd(N_CLASSES, hidden_sizes, step_size, sparsity, args.upgd_sigma)
        state, metrics, replacements = run_hybrid_digits(
            learner,
            key,
            observations,
            targets,
            labels,
            rate,
            args.maturity_threshold,
            sparsity,
        )
        summary = summarize_digits_metrics(metrics, args.final_window)
        summary.update(
            evaluate_digits(
                learner,
                state,
                x_test,
                y_test,
                permutations,
                is_hybrid=True,
            )
        )
        return summary, replacements
    raise ValueError(f"unknown method {method}")


def run_digits_suite(
    args: argparse.Namespace,
    methods: list[MethodName],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all selected digits variants."""
    records: list[dict[str, Any]] = []
    hidden_sizes = (args.hidden_size,)
    dataset_meta: dict[str, Any] = {}
    variants = DIGITS_VARIANTS if args.digits_variant == "all" else (args.digits_variant,)

    for variant in variants:
        print(f"\n[DIGITS] {variant}")
        for seed_idx in range(args.n_seeds):
            seed = args.seed + seed_idx
            x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
                seed,
                args.train_fraction,
            )
            observations, targets, labels, permutations = make_digits_sequence(
                x_train,
                y_train,
                args.digits_steps,
                seed + 10_000,
                variant,
                args.digits_block_size,
            )
            split_keys = jr.split(jr.key(seed + 50_000), len(methods))
            for method_idx, method in enumerate(methods):
                t0 = time.time()
                summary, replacements = run_digits_method(
                    method,
                    hidden_sizes,
                    args.step_size,
                    args.sparsity,
                    split_keys[method_idx],
                    observations,
                    targets,
                    labels,
                    x_test,
                    y_test,
                    permutations,
                    args,
                )
                records.append(
                    {
                        "family": "digits",
                        "scenario": f"digits_{variant}",
                        "seed": seed,
                        "method": method,
                        "metrics": summary,
                        "replacements": replacements,
                        "wall_clock_s": time.time() - t0,
                    }
                )
                print(
                    f"  seed={seed} {method}: "
                    f"final_acc={summary['final_window_accuracy']:.3f} "
                    f"test_acc={summary['test_accuracy']:.3f} "
                    f"final_mse={summary['final_window_mse']:.5f} resets={replacements}"
                )
    return records, dataset_meta


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed records and pair methods against MLP."""
    aggregate: dict[str, Any] = {}
    scenarios = sorted({str(r["scenario"]) for r in records})
    methods = sorted({str(r["method"]) for r in records})
    for scenario in scenarios:
        scenario_records = [r for r in records if r["scenario"] == scenario]
        aggregate[scenario] = {"methods": {}, "paired_vs_mlp": {}}
        for method in methods:
            method_records = [r for r in scenario_records if r["method"] == method]
            if not method_records:
                continue
            metric_keys = sorted(method_records[0]["metrics"].keys())
            aggregate[scenario]["methods"][method] = {}
            for metric_key in metric_keys:
                values = np.asarray(
                    [r["metrics"][metric_key] for r in method_records],
                    dtype=np.float64,
                )
                aggregate[scenario]["methods"][method][metric_key] = {
                    "mean": float(np.mean(values)),
                    "stderr": stderr(values),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            aggregate[scenario]["methods"][method]["replacements"] = {
                "mean": float(np.mean([r["replacements"] for r in method_records])),
                "sum": int(np.sum([r["replacements"] for r in method_records])),
            }

        mlp_records = sorted(
            [r for r in scenario_records if r["method"] == "mlp"],
            key=lambda r: int(r["seed"]),
        )
        if not mlp_records:
            continue
        for method in methods:
            if method == "mlp":
                continue
            method_records = sorted(
                [r for r in scenario_records if r["method"] == method],
                key=lambda r: int(r["seed"]),
            )
            if len(method_records) != len(mlp_records):
                continue
            comparisons: dict[str, Any] = {}
            for metric_key in mlp_records[0]["metrics"]:
                mlp_values = np.asarray(
                    [r["metrics"][metric_key] for r in mlp_records],
                    dtype=np.float64,
                )
                method_values = np.asarray(
                    [r["metrics"][metric_key] for r in method_records],
                    dtype=np.float64,
                )
                higher_is_better = metric_key.endswith("accuracy")
                diff = (
                    method_values - mlp_values
                    if higher_is_better
                    else mlp_values - method_values
                )
                comparisons[metric_key] = {
                    "positive_means_method_better": float(np.mean(diff)),
                    "stderr": stderr(diff),
                    "wins_for_method": int(np.sum(diff > 0.0)),
                    "wins_for_mlp": int(np.sum(diff < 0.0)),
                    "ties": int(np.sum(diff == 0.0)),
                    "cohens_d": cohens_d(diff),
                }
            aggregate[scenario]["paired_vs_mlp"][method] = comparisons
    return aggregate


def choose_best_candidate(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Pick the plasticity method with the best broad final-window evidence."""
    scores: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    for scenario_summary in aggregate.values():
        paired = scenario_summary["paired_vs_mlp"]
        for method, metrics in paired.items():
            if "final_window_mse" not in metrics:
                continue
            scores.setdefault(method, []).append(
                metrics["final_window_mse"]["positive_means_method_better"]
            )
            wins[method] = wins.get(method, 0) + int(
                metrics["final_window_mse"]["positive_means_method_better"] > 0.0
            )
    if not scores:
        return {
            "method": None,
            "reason": "No paired plasticity-vs-MLP records were produced.",
        }

    ranked = sorted(
        scores,
        key=lambda method: (
            wins.get(method, 0),
            float(np.mean(scores[method])),
            -float(np.std(scores[method])),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "method": best,
        "scenario_wins_on_final_window_mse": wins.get(best, 0),
        "n_scenarios": len(scores[best]),
        "mean_mlp_minus_method_final_window_mse": float(np.mean(scores[best])),
        "all_method_scores": {
            method: {
                "scenario_wins": wins.get(method, 0),
                "mean_mlp_minus_method_final_window_mse": float(np.mean(vals)),
                "values": vals,
            }
            for method, vals in sorted(scores.items())
        },
    }


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary from JSON-ready results."""
    aggregate = results["aggregate"]
    best = results["best_candidate"]
    lines = [
        "# Worker S2E Step 2 Plasticity Hybrid Results",
        "",
        "This worker compared fair MLP, UPGD, CBP, and a script-local UPGD plus "
        "low-utility unit reset hybrid. All methods used the same hidden width, "
        "step size, sparsity, layer norm, and ObGD bounding.",
        "",
        f"Best broad candidate by paired final-window MSE: `{best.get('method')}`.",
        "",
        "Positive paired differences mean the plasticity method beat MLP.",
        "",
    ]

    for scenario, scenario_summary in aggregate.items():
        lines.extend([f"## {scenario}", ""])
        method_summary = scenario_summary["methods"]
        metric = "final_window_mse"
        lines.extend([
            "| Method | Final-window MSE | Mean MSE | Final-window accuracy | "
            "Test accuracy | Replacements |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method, values in sorted(method_summary.items()):
            final_mse = values.get(metric, {})
            mean_mse = values.get("mean_mse", {})
            final_acc = values.get("final_window_accuracy", {})
            test_acc = values.get("test_accuracy", {})
            repl = values.get("replacements", {})
            lines.append(
                f"| {method} | "
                f"{final_mse.get('mean', float('nan')):.5f} +/- "
                f"{final_mse.get('stderr', 0.0):.5f} | "
                f"{mean_mse.get('mean', float('nan')):.5f} +/- "
                f"{mean_mse.get('stderr', 0.0):.5f} | "
                f"{final_acc.get('mean', float('nan')):.3f} +/- "
                f"{final_acc.get('stderr', 0.0):.3f} | "
                f"{test_acc.get('mean', float('nan')):.3f} +/- "
                f"{test_acc.get('stderr', 0.0):.3f} | "
                f"{repl.get('mean', 0.0):.1f} |"
            )
        lines.extend(["", "### Paired vs MLP", ""])
        lines.extend([
            "| Method | MLP minus method final MSE | Wins | MLP wins | Cohen d |",
            "|---|---:|---:|---:|---:|",
        ])
        for method, comparisons in sorted(scenario_summary["paired_vs_mlp"].items()):
            row = comparisons.get("final_window_mse", {})
            lines.append(
                f"| {method} | "
                f"{row.get('positive_means_method_better', float('nan')):+.5f} +/- "
                f"{row.get('stderr', 0.0):.5f} | "
                f"{row.get('wins_for_method', 0)} | "
                f"{row.get('wins_for_mlp', 0)} | "
                f"{row.get('cohens_d', 0.0):+.2f} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "The script reports evidence, not a canonical conclusion. A method should be "
        "considered robust only if it improves final-window MSE across most "
        "scenarios and does not degrade external test accuracy.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "digits", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--include-tuned", action="store_true")
    parser.add_argument("--synthetic-steps", type=int, default=3000)
    parser.add_argument(
        "--synthetic-stream",
        choices=("all", "out_of_class_polynomial", "frequency_mismatch", "compositional"),
        default="all",
    )
    parser.add_argument("--digits-steps", type=int, default=2500)
    parser.add_argument("--digits-variant", choices=("all", *DIGITS_VARIANTS), default="all")
    parser.add_argument("--digits-block-size", type=int, default=250)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--upgd-sigma", type=float, default=1e-3)
    parser.add_argument("--upgd-low-noise-sigma", type=float, default=1e-4)
    parser.add_argument("--upgd-tiny-noise-sigma", type=float, default=3e-5)
    parser.add_argument("--upgd-warmup-steps", type=int, default=200)
    parser.add_argument("--upgd-ramp-steps", type=int, default=400)
    parser.add_argument("--maturity-threshold", type=int, default=100)
    parser.add_argument("--cbp-replacement-rate", type=float, default=1e-4)
    parser.add_argument("--cbp-fast-replacement-rate", type=float, default=5e-4)
    parser.add_argument("--hybrid-replacement-rate", type=float, default=1e-4)
    parser.add_argument("--hybrid-fast-replacement-rate", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    """Run the selected S2E benchmark suite."""
    args = parse_args()
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.synthetic_steps <= 0 or args.digits_steps <= 0:
        raise ValueError("step counts must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.upgd_warmup_steps < 0 or args.upgd_ramp_steps < 0:
        raise ValueError("UPGD warmup/ramp steps must be non-negative")

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = method_names(args.include_tuned)
    records: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}

    if args.suite in {"synthetic", "both"}:
        records.extend(run_synthetic_suite(args, methods))
    if args.suite in {"digits", "both"}:
        digits_records, dataset_meta = run_digits_suite(args, methods)
        records.extend(digits_records)

    aggregate = aggregate_records(records)
    results = {
        "config": {
            "suite": args.suite,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "final_window": args.final_window,
            "include_tuned": args.include_tuned,
            "synthetic_steps": args.synthetic_steps,
            "digits_steps": args.digits_steps,
            "upgd_sigma": args.upgd_sigma,
            "upgd_low_noise_sigma": args.upgd_low_noise_sigma,
            "upgd_tiny_noise_sigma": args.upgd_tiny_noise_sigma,
            "upgd_warmup_steps": args.upgd_warmup_steps,
            "upgd_ramp_steps": args.upgd_ramp_steps,
            "maturity_threshold": args.maturity_threshold,
            "cbp_replacement_rate": args.cbp_replacement_rate,
            "cbp_fast_replacement_rate": args.cbp_fast_replacement_rate,
            "hybrid_replacement_rate": args.hybrid_replacement_rate,
            "hybrid_fast_replacement_rate": args.hybrid_fast_replacement_rate,
        },
        "methods": methods,
        "dataset": dataset_meta,
        "records": records,
        "aggregate": aggregate,
        "best_candidate": choose_best_candidate(aggregate),
        "wall_clock_s": time.time() - t0,
    }
    results_path = args.output_dir / "step2_plasticity_hybrid_results.json"
    summary_path = args.output_dir / "step2_plasticity_hybrid_SUMMARY.md"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(summary_path, results)
    print(f"\nwrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

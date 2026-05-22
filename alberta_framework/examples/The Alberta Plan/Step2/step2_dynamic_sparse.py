#!/usr/bin/env python3
"""Direction 7: dynamic sparse / rewiring Step 2 baseline.

This script prototypes a deliberately simple dynamic sparse MLP baseline:

* one sparse hidden layer with a fixed per-unit input fan-in budget,
* dense linear heads,
* online SGD with the same ObGD bounder used by the fair MLP and UPGD baselines,
* an EMA utility estimate ``|w * grad|`` for active hidden input weights,
* periodic low-utility hidden-unit reset under a fixed hidden-input budget.

The comparison is intentionally narrow:

1. One synthetic out-of-hypothesis-class stream:
   ``OutOfClassPolynomialStream``.
2. One externally grounded smoke:
   ``sklearn.datasets.load_digits`` in an online class-blocked stream.

Outputs:
    output/direction7_dynamic_sparse/dynamic_sparse_results.json
    output/direction7_dynamic_sparse/dynamic_sparse_SUMMARY.md
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

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
    run_multi_head_learning_loop,
    run_upgd_arrays,
)

N_DIGIT_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("output/direction7_dynamic_sparse")


@chex.dataclass(frozen=True)
class DynamicSparseState:
    """State for the local dynamic sparse learner."""

    input_weights: jax.Array
    hidden_bias: jax.Array
    head_weights: jax.Array
    head_bias: jax.Array
    mask: jax.Array
    utilities: jax.Array
    key: jax.Array
    step_count: jax.Array


@chex.dataclass(frozen=True)
class DynamicSparseUpdateResult:
    """One online update result."""

    state: DynamicSparseState
    predictions: jax.Array
    metrics: jax.Array


class DynamicSparseMLP:
    """One-hidden-layer MLP with fixed-budget hidden-input rewiring.

    The fixed budget is enforced on the first layer: every hidden unit keeps
    exactly ``fan_in_budget`` live input weights. Rewiring resets the lowest
    utility hidden units to fresh sparse input masks and zero outgoing head
    weights, preserving the total number of live hidden input connections.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int = 64,
        step_size: float = 0.03,
        sparsity: float = 0.5,
        utility_decay: float = 0.99,
        rewire_interval: int = 100,
        unit_replacement_rate: float = 0.05,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        obgd_kappa: float = 2.0,
    ) -> None:
        if n_heads < 1:
            raise ValueError("n_heads must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        if rewire_interval < 1:
            raise ValueError("rewire_interval must be positive")
        if not 0.0 <= unit_replacement_rate <= 1.0:
            raise ValueError("unit_replacement_rate must be in [0, 1]")

        self._n_heads = int(n_heads)
        self._hidden_size = int(hidden_size)
        self._step_size = float(step_size)
        self._sparsity = float(sparsity)
        self._utility_decay = float(utility_decay)
        self._rewire_interval = int(rewire_interval)
        self._unit_replacement_rate = float(unit_replacement_rate)
        self._n_replace = (
            0
            if unit_replacement_rate == 0.0
            else max(1, int(round(hidden_size * unit_replacement_rate)))
        )
        self._leaky_relu_slope = float(leaky_relu_slope)
        self._use_layer_norm = bool(use_layer_norm)
        self._bounder = ObGDBounding(kappa=obgd_kappa)

    def _fan_in_budget(self, feature_dim: int) -> int:
        return max(1, min(feature_dim, int(round((1.0 - self._sparsity) * feature_dim))))

    def init(self, feature_dim: int, key: jax.Array) -> DynamicSparseState:
        """Initialize hidden sparse masks and dense linear heads."""
        key, k_w, k_mask, k_head = jr.split(key, 4)
        fan_in_budget = self._fan_in_budget(feature_dim)

        score = jr.uniform(k_mask, (self._hidden_size, feature_dim), dtype=jnp.float32)
        order = jnp.argsort(score, axis=1)
        mask = jnp.zeros((self._hidden_size, feature_dim), dtype=jnp.float32)
        rows = jnp.arange(self._hidden_size)[:, None]
        mask = mask.at[rows, order[:, :fan_in_budget]].set(1.0)

        in_scale = 1.0 / math.sqrt(feature_dim)
        input_weights = jr.uniform(
            k_w,
            (self._hidden_size, feature_dim),
            dtype=jnp.float32,
            minval=-in_scale,
            maxval=in_scale,
        )
        input_weights = input_weights * mask

        head_scale = 1.0 / math.sqrt(self._hidden_size)
        head_weights = jr.uniform(
            k_head,
            (self._n_heads, self._hidden_size),
            dtype=jnp.float32,
            minval=-head_scale,
            maxval=head_scale,
        )

        return DynamicSparseState(  # type: ignore[call-arg]
            input_weights=input_weights,
            hidden_bias=jnp.zeros(self._hidden_size, dtype=jnp.float32),
            head_weights=head_weights,
            head_bias=jnp.zeros(self._n_heads, dtype=jnp.float32),
            mask=mask,
            utilities=jnp.zeros_like(input_weights),
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    def _hidden_forward(
        input_weights: jax.Array,
        hidden_bias: jax.Array,
        mask: jax.Array,
        observation: jax.Array,
        leaky_relu_slope: float,
        use_layer_norm: bool,
    ) -> jax.Array:
        hidden = (input_weights * mask) @ observation + hidden_bias
        if use_layer_norm:
            hidden = (hidden - jnp.mean(hidden)) / jnp.sqrt(jnp.var(hidden) + 1e-5)
        return jnp.where(hidden >= 0.0, hidden, leaky_relu_slope * hidden)

    @staticmethod
    def _forward(
        input_weights: jax.Array,
        hidden_bias: jax.Array,
        head_weights: jax.Array,
        head_bias: jax.Array,
        mask: jax.Array,
        observation: jax.Array,
        leaky_relu_slope: float,
        use_layer_norm: bool,
    ) -> jax.Array:
        hidden = DynamicSparseMLP._hidden_forward(
            input_weights,
            hidden_bias,
            mask,
            observation,
            leaky_relu_slope,
            use_layer_norm,
        )
        return head_weights @ hidden + head_bias

    @jax.jit(static_argnums=(0,))
    def predict(self, state: DynamicSparseState, observation: jax.Array) -> jax.Array:
        return self._forward(
            state.input_weights,
            state.hidden_bias,
            state.head_weights,
            state.head_bias,
            state.mask,
            observation,
            self._leaky_relu_slope,
            self._use_layer_norm,
        )

    @jax.jit(static_argnums=(0,))
    def update(
        self,
        state: DynamicSparseState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> DynamicSparseUpdateResult:
        """Run one prequential prediction, SGD update, and optional rewire."""
        step_size = jnp.array(self._step_size, dtype=jnp.float32)
        decay = jnp.array(self._utility_decay, dtype=jnp.float32)
        active_mask = ~jnp.isnan(targets)
        safe_targets = jnp.where(active_mask, targets, 0.0)

        def loss_fn(
            input_weights: jax.Array,
            hidden_bias: jax.Array,
            head_weights: jax.Array,
            head_bias: jax.Array,
        ) -> jax.Array:
            preds = self._forward(
                input_weights,
                hidden_bias,
                head_weights,
                head_bias,
                state.mask,
                observation,
                self._leaky_relu_slope,
                self._use_layer_norm,
            )
            sq = jnp.where(active_mask, (preds - safe_targets) ** 2, 0.0)
            n_active = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32)), 1.0)
            return 0.5 * jnp.sum(sq) / n_active

        predictions = self.predict(state, observation)
        loss_value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
            state.input_weights,
            state.hidden_bias,
            state.head_weights,
            state.head_bias,
        )
        grad_w, grad_b, grad_head_w, grad_head_b = grads

        steps = (
            -step_size * grad_w,
            -step_size * grad_b,
            -step_size * grad_head_w,
            -step_size * grad_head_b,
        )
        errors_for_bound = jnp.where(active_mask, predictions - safe_targets, 0.0)
        mean_abs_error = jnp.sum(jnp.abs(errors_for_bound)) / jnp.maximum(
            jnp.sum(active_mask.astype(jnp.float32)), 1.0
        )
        bounded_steps, bound_scale = self._bounder.bound(
            steps,
            mean_abs_error,
            (
                state.input_weights,
                state.hidden_bias,
                state.head_weights,
                state.head_bias,
            ),
        )
        step_w, step_b, step_head_w, step_head_b = bounded_steps

        input_weights = (state.input_weights + step_w) * state.mask
        hidden_bias = state.hidden_bias + step_b
        head_weights = state.head_weights + step_head_w
        head_bias = state.head_bias + step_head_b

        utilities = (
            decay * state.utilities
            + (1.0 - decay) * jnp.abs((state.input_weights * state.mask) * grad_w)
        ) * state.mask

        input_weights, hidden_bias, head_weights, mask, utilities, key, rewired = (
            self._maybe_rewire(
                state=state,
                input_weights=input_weights,
                hidden_bias=hidden_bias,
                head_weights=head_weights,
                mask=state.mask,
                utilities=utilities,
            )
        )

        new_state = DynamicSparseState(  # type: ignore[call-arg]
            input_weights=input_weights,
            hidden_bias=hidden_bias,
            head_weights=head_weights,
            head_bias=head_bias,
            mask=mask,
            utilities=utilities,
            key=key,
            step_count=state.step_count + 1,
        )
        mean_utility = jnp.mean(jnp.sum(utilities, axis=1))
        metrics = jnp.stack(
            [
                loss_value,
                mean_utility,
                rewired.astype(jnp.float32),
                bound_scale,
            ]
        )
        return DynamicSparseUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            predictions=predictions,
            metrics=metrics,
        )

    def _maybe_rewire(
        self,
        state: DynamicSparseState,
        input_weights: jax.Array,
        hidden_bias: jax.Array,
        head_weights: jax.Array,
        mask: jax.Array,
        utilities: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Replace low-utility hidden units while preserving hidden-input budget."""
        if self._n_replace == 0:
            return (
                input_weights,
                hidden_bias,
                head_weights,
                mask,
                utilities,
                state.key,
                jnp.array(0, dtype=jnp.int32),
            )

        feature_dim = input_weights.shape[1]
        fan_in_budget = self._fan_in_budget(feature_dim)
        do_rewire = jnp.logical_and(
            state.step_count > 0,
            (state.step_count % self._rewire_interval) == 0,
        )

        key, k_mask, k_w = jr.split(state.key, 3)
        input_score = jnp.sum(utilities, axis=1) / float(fan_in_budget)
        head_score = 0.01 * jnp.mean(jnp.abs(head_weights), axis=0)
        unit_score = input_score + head_score
        replace_idx = jnp.argsort(unit_score)[: self._n_replace]

        score = jr.uniform(k_mask, (self._n_replace, feature_dim), dtype=jnp.float32)
        order = jnp.argsort(score, axis=1)
        new_mask_rows = jnp.zeros((self._n_replace, feature_dim), dtype=jnp.float32)
        rows = jnp.arange(self._n_replace)[:, None]
        new_mask_rows = new_mask_rows.at[rows, order[:, :fan_in_budget]].set(1.0)

        scale = 1.0 / math.sqrt(feature_dim)
        new_w_rows = jr.uniform(
            k_w,
            (self._n_replace, feature_dim),
            dtype=jnp.float32,
            minval=-scale,
            maxval=scale,
        )
        new_w_rows = new_w_rows * new_mask_rows

        rewired_input_weights = input_weights.at[replace_idx, :].set(new_w_rows)
        rewired_hidden_bias = hidden_bias.at[replace_idx].set(0.0)
        rewired_head_weights = head_weights.at[:, replace_idx].set(0.0)
        rewired_mask = mask.at[replace_idx, :].set(new_mask_rows)
        rewired_utilities = utilities.at[replace_idx, :].set(0.0)

        return (
            jnp.where(do_rewire, rewired_input_weights, input_weights),
            jnp.where(do_rewire, rewired_hidden_bias, hidden_bias),
            jnp.where(do_rewire, rewired_head_weights, head_weights),
            jnp.where(do_rewire, rewired_mask, mask),
            jnp.where(do_rewire, rewired_utilities, utilities),
            key,
            jnp.where(do_rewire, self._n_replace, 0),
        )


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize one supervised stream realization."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, stream_state, jnp.arange(num_steps)
    )
    return observations, targets


def mlp_loss_curve(metrics: jax.Array) -> jax.Array:
    """Return per-step mean MSE from MultiHeadMLPLearner metrics."""
    return jnp.nanmean(metrics[..., 0], axis=-1)


def run_mlp_arrays(
    n_heads: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
) -> tuple[Any, np.ndarray]:
    learner = MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )
    state = learner.init(feature_dim, key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    curve = mlp_loss_curve(result.per_head_metrics)
    curve.block_until_ready()
    return result.state, np.asarray(curve)


def run_upgd_loss_arrays(
    n_heads: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    perturbation_sigma: float,
) -> tuple[Any, np.ndarray]:
    learner = UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=perturbation_sigma,
    )
    state = learner.init(feature_dim, key)
    result = run_upgd_arrays(learner, state, observations, targets)
    curve = result.metrics[:, 0]
    curve.block_until_ready()
    return result.state, np.asarray(curve)


def run_dynamic_sparse_arrays(
    n_heads: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    utility_decay: float,
    rewire_interval: int,
    unit_replacement_rate: float,
) -> tuple[DynamicSparseMLP, DynamicSparseState, np.ndarray]:
    learner = DynamicSparseMLP(
        n_heads=n_heads,
        hidden_size=hidden_size,
        step_size=step_size,
        sparsity=sparsity,
        utility_decay=utility_decay,
        rewire_interval=rewire_interval,
        unit_replacement_rate=unit_replacement_rate,
    )
    state = learner.init(feature_dim, key)

    def step_fn(
        carry: DynamicSparseState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[DynamicSparseState, jax.Array]:
        obs, tgt = inputs
        result = learner.update(carry, obs, tgt)
        return result.state, result.metrics

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    return learner, final_state, np.asarray(metrics)


def stderr(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def summarize_losses(curves: dict[str, list[np.ndarray]], final_window: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method, method_curves in curves.items():
        arr = np.stack(method_curves, axis=0)
        window = min(final_window, arr.shape[1])
        final = np.mean(arr[:, -window:], axis=1)
        total = np.mean(arr, axis=1)
        out[method] = {
            "mean_final_window_mse": float(np.mean(final)),
            "stderr_final_window_mse": stderr(final),
            "mean_total_mse": float(np.mean(total)),
            "stderr_total_mse": stderr(total),
            "per_seed_final_window_mse": final.astype(float).tolist(),
        }
    return out


def paired_loss_summary(
    aggregate_source: dict[str, list[np.ndarray]],
    baseline: str,
    method: str,
    final_window: int,
) -> dict[str, Any]:
    base = np.stack(aggregate_source[baseline], axis=0)
    other = np.stack(aggregate_source[method], axis=0)
    window = min(final_window, base.shape[1])
    base_final = np.mean(base[:, -window:], axis=1)
    other_final = np.mean(other[:, -window:], axis=1)
    diff = base_final - other_final
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "baseline": baseline,
        "method": method,
        "metric": "final_window_mse",
        "baseline_minus_method_mean": float(np.mean(diff)),
        "baseline_minus_method_stderr": stderr(diff),
        "wins_for_method": int(np.sum(diff > 0.0)),
        "wins_for_baseline": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n_seeds": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    """Run one synthetic out-of-class polynomial stream."""
    stream = OutOfClassPolynomialStream(
        feature_dim=8,
        n_tasks=3,
        n_contexts=4,
        context_length=max(100, args.synthetic_steps // 4),
        active_triples_per_context=2,
        noise_std=0.05,
    )
    curves: dict[str, list[np.ndarray]] = {"mlp": [], "upgd": [], "dynamic_sparse": []}
    rewires: list[int] = []

    for seed_idx in range(args.n_seeds):
        seed = args.seed + seed_idx
        stream_key, mlp_key, upgd_key, dyn_key = jr.split(jr.key(seed), 4)
        observations, targets = collect_stream_arrays(stream, args.synthetic_steps, stream_key)
        observations.block_until_ready()
        targets.block_until_ready()

        print(f"synthetic seed={seed}: running MLP")
        _, mlp_curve = run_mlp_arrays(
            n_heads=stream.target_dim,
            feature_dim=stream.feature_dim,
            observations=observations,
            targets=targets,
            key=mlp_key,
            hidden_size=args.hidden_size,
            step_size=args.step_size,
            sparsity=args.sparsity,
        )
        print(f"synthetic seed={seed}: running UPGD")
        _, upgd_curve = run_upgd_loss_arrays(
            n_heads=stream.target_dim,
            feature_dim=stream.feature_dim,
            observations=observations,
            targets=targets,
            key=upgd_key,
            hidden_size=args.hidden_size,
            step_size=args.step_size,
            sparsity=args.sparsity,
            perturbation_sigma=args.perturbation_sigma,
        )
        print(f"synthetic seed={seed}: running dynamic_sparse")
        _, _, dyn_metrics = run_dynamic_sparse_arrays(
            n_heads=stream.target_dim,
            feature_dim=stream.feature_dim,
            observations=observations,
            targets=targets,
            key=dyn_key,
            hidden_size=args.hidden_size,
            step_size=args.step_size,
            sparsity=args.sparsity,
            utility_decay=args.utility_decay,
            rewire_interval=args.rewire_interval,
            unit_replacement_rate=args.unit_replacement_rate,
        )

        curves["mlp"].append(mlp_curve)
        curves["upgd"].append(upgd_curve)
        curves["dynamic_sparse"].append(dyn_metrics[:, 0])
        rewires.append(int(np.sum(dyn_metrics[:, 2])))

    aggregate = summarize_losses(curves, args.final_window)
    paired = {
        "dynamic_sparse_vs_mlp": paired_loss_summary(
            curves, "mlp", "dynamic_sparse", args.final_window
        ),
        "dynamic_sparse_vs_upgd": paired_loss_summary(
            curves, "upgd", "dynamic_sparse", args.final_window
        ),
        "upgd_vs_mlp": paired_loss_summary(curves, "mlp", "upgd", args.final_window),
    }
    return {
        "stream": "out_of_class_polynomial",
        "config": {
            "feature_dim": stream.feature_dim,
            "n_tasks": stream.target_dim,
            "steps": args.synthetic_steps,
            "n_seeds": args.n_seeds,
        },
        "aggregate": aggregate,
        "paired": paired,
        "dynamic_sparse_rewired_units_per_seed": rewires,
    }


@dataclass(frozen=True)
class DigitsStream:
    observations: jax.Array
    targets: jax.Array
    labels: jax.Array
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def load_digits_split(
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:
        msg = "scikit-learn is required for the digits smoke benchmark."
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
        "n_classes": N_DIGIT_CLASSES,
        "train_fraction": train_fraction,
    }
    return x_train, y_train, x_test, y_test, meta


def make_class_blocked_digits_stream(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    steps: int,
    seed: int,
    meta: dict[str, Any],
) -> DigitsStream:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_DIGIT_CLASSES)]
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    while total < steps:
        for cls in rng.permutation(N_DIGIT_CLASSES):
            cls_idx = class_indices[int(cls)].copy()
            rng.shuffle(cls_idx)
            chunks_x.append(x_train[cls_idx])
            chunks_y.append(y_train[cls_idx])
            total += len(cls_idx)
            if total >= steps:
                break

    observations = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float32)[labels]
    stream_meta = {
        **meta,
        "stream": "class_blocked_digits",
        "description": "Online sklearn digits stream grouped by digit-class blocks.",
        "steps": steps,
    }
    return DigitsStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        x_test=x_test,
        y_test=y_test,
        meta=stream_meta,
    )


def classifier_curve_from_predictions(
    predictions: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    mse = jnp.mean((predictions - targets) ** 2, axis=1)
    correct = (jnp.argmax(predictions, axis=1) == labels).astype(jnp.float32)
    return jnp.stack([mse, correct], axis=1)


def run_mlp_digits(
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
) -> tuple[MultiHeadMLPLearner, Any, np.ndarray]:
    learner = MultiHeadMLPLearner(
        n_heads=N_DIGIT_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        metric = classifier_curve_from_predictions(
            result.predictions[None, :], tgt[None, :], label[None]
        )[0]
        return result.state, metric

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return learner, final_state, np.asarray(metrics)


def run_upgd_digits(
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    perturbation_sigma: float,
) -> tuple[UPGDLearner, Any, np.ndarray]:
    learner = UPGDLearner(
        n_heads=N_DIGIT_CLASSES,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=perturbation_sigma,
    )
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        metric = classifier_curve_from_predictions(
            result.predictions[None, :], tgt[None, :], label[None]
        )[0]
        return result.state, metric

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return learner, final_state, np.asarray(metrics)


def run_dynamic_sparse_digits(
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    utility_decay: float,
    rewire_interval: int,
    unit_replacement_rate: float,
) -> tuple[DynamicSparseMLP, DynamicSparseState, np.ndarray]:
    learner = DynamicSparseMLP(
        n_heads=N_DIGIT_CLASSES,
        hidden_size=hidden_size,
        step_size=step_size,
        sparsity=sparsity,
        utility_decay=utility_decay,
        rewire_interval=rewire_interval,
        unit_replacement_rate=unit_replacement_rate,
    )
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: DynamicSparseState,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[DynamicSparseState, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        metric = classifier_curve_from_predictions(
            result.predictions[None, :], tgt[None, :], label[None]
        )[0]
        return result.state, metric

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return learner, final_state, np.asarray(metrics)


def evaluate_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def summarize_digit_metrics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def aggregate_digit_records(
    records: list[dict[str, Any]],
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    methods = ["mlp", "upgd", "dynamic_sparse"]
    values = {
        method: np.asarray(
            [r["methods"][method][metric] for r in records], dtype=np.float64
        )
        for method in methods
    }
    paired: dict[str, Any] = {}
    for baseline in ("mlp", "upgd"):
        diff = values["dynamic_sparse"] - values[baseline]
        if not higher_is_better:
            diff = values[baseline] - values["dynamic_sparse"]
        paired[f"dynamic_sparse_vs_{baseline}"] = {
            "diff_mean_positive_favors_dynamic_sparse": float(np.mean(diff)),
            "diff_stderr": stderr(diff),
            "wins_for_dynamic_sparse": int(np.sum(diff > 0.0)),
            "wins_for_baseline": int(np.sum(diff < 0.0)),
            "ties": int(np.sum(diff == 0.0)),
            "n_seeds": int(diff.shape[0]),
        }
    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "means": {method: float(np.mean(vals)) for method, vals in values.items()},
        "stderrs": {method: stderr(vals) for method, vals in values.items()},
        "paired": paired,
    }


def run_digits(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] | None = None

    for seed_idx in range(args.n_seeds):
        seed = args.seed + 10_000 + seed_idx
        x_train, y_train, x_test, y_test, dataset_meta = load_digits_split(
            seed=seed,
            train_fraction=args.train_fraction,
        )
        stream = make_class_blocked_digits_stream(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            steps=args.digits_steps,
            seed=seed + 123,
            meta=dataset_meta,
        )
        mlp_key, upgd_key, dyn_key = jr.split(jr.key(seed), 3)

        print(f"digits seed={seed}: running MLP")
        mlp, mlp_state, mlp_metrics = run_mlp_digits(
            stream.observations,
            stream.targets,
            stream.labels,
            mlp_key,
            args.hidden_size,
            args.step_size,
            args.sparsity,
        )
        print(f"digits seed={seed}: running UPGD")
        upgd, upgd_state, upgd_metrics = run_upgd_digits(
            stream.observations,
            stream.targets,
            stream.labels,
            upgd_key,
            args.hidden_size,
            args.step_size,
            args.sparsity,
            args.perturbation_sigma,
        )
        print(f"digits seed={seed}: running dynamic_sparse")
        dyn, dyn_state, dyn_metrics = run_dynamic_sparse_digits(
            stream.observations,
            stream.targets,
            stream.labels,
            dyn_key,
            args.hidden_size,
            args.step_size,
            args.sparsity,
            args.utility_decay,
            args.rewire_interval,
            args.unit_replacement_rate,
        )

        mlp_summary = summarize_digit_metrics(mlp_metrics, args.final_window)
        mlp_summary.update(evaluate_classifier(mlp, mlp_state, stream.x_test, stream.y_test))
        upgd_summary = summarize_digit_metrics(upgd_metrics, args.final_window)
        upgd_summary.update(evaluate_classifier(upgd, upgd_state, stream.x_test, stream.y_test))
        dyn_summary = summarize_digit_metrics(dyn_metrics, args.final_window)
        dyn_summary.update(evaluate_classifier(dyn, dyn_state, stream.x_test, stream.y_test))

        records.append(
            {
                "seed": seed,
                "methods": {
                    "mlp": mlp_summary,
                    "upgd": upgd_summary,
                    "dynamic_sparse": dyn_summary,
                },
            }
        )
        print(
            f"digits seed={seed}: final acc "
            f"mlp={mlp_summary['final_window_accuracy']:.3f}, "
            f"upgd={upgd_summary['final_window_accuracy']:.3f}, "
            f"dyn={dyn_summary['final_window_accuracy']:.3f}"
        )

    aggregate = {
        "final_window_mse": aggregate_digit_records(
            records, "final_window_mse", higher_is_better=False
        ),
        "test_mse": aggregate_digit_records(records, "test_mse", higher_is_better=False),
        "final_window_accuracy": aggregate_digit_records(
            records, "final_window_accuracy", higher_is_better=True
        ),
        "test_accuracy": aggregate_digit_records(
            records, "test_accuracy", higher_is_better=True
        ),
    }
    return {
        "stream": "class_blocked_digits",
        "dataset": dataset_meta,
        "records": records,
        "aggregate": aggregate,
    }


def _fmt(mean: float, se: float) -> str:
    return f"{mean:.4f} +/- {se:.4f}"


def _loss_method_order(aggregate: dict[str, Any]) -> list[str]:
    return sorted(aggregate, key=lambda m: aggregate[m]["mean_final_window_mse"])


def write_summary(path: Path, results: dict[str, Any]) -> None:
    cfg = results["config"]
    synthetic = results["synthetic"]
    digits = results.get("digits")
    lines: list[str] = [
        "# Direction 7 Dynamic Sparse Rewiring Baseline",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, hidden_size={cfg['hidden_size']}, "
            f"step_size={cfg['step_size']}, sparsity={cfg['sparsity']}, "
            f"final_window={cfg['final_window']}."
        ),
        (
            "Dynamic sparse keeps a fixed first-layer fan-in budget and resets "
            f"the lowest-utility hidden units every {cfg['rewire_interval']} steps."
        ),
        "",
        "## Synthetic Out-of-Class Polynomial",
        "",
        "| Method | Final-window MSE | Total MSE |",
        "|---|---:|---:|",
    ]

    for method in _loss_method_order(synthetic["aggregate"]):
        row = synthetic["aggregate"][method]
        lines.append(
            f"| `{method}` | "
            f"{_fmt(row['mean_final_window_mse'], row['stderr_final_window_mse'])} | "
            f"{_fmt(row['mean_total_mse'], row['stderr_total_mse'])} |"
        )
    lines.extend(
        [
            "",
            "Paired final-window MSE differences. Positive favors the method after `vs`.",
            "",
        ]
    )
    lines.append("| Comparison | Diff | Wins |")
    lines.append("|---|---:|---:|")
    for name, row in synthetic["paired"].items():
        lines.append(
            f"| `{name}` | "
            f"{row['baseline_minus_method_mean']:+.4f} +/- "
            f"{row['baseline_minus_method_stderr']:.4f} | "
            f"{row['wins_for_method']}/{row['n_seeds']} |"
        )

    if digits is None:
        lines.extend(["", "## Digits Smoke", "", "Skipped: scikit-learn is unavailable."])
    else:
        lines.extend(
            [
                "",
                "## Digits Smoke",
                "",
                "Stream: sklearn digits, class-blocked online train stream, held-out split.",
                "",
                "| Metric | MLP | UPGD | Dynamic Sparse | Dyn vs MLP | Dyn vs UPGD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric_name, row in digits["aggregate"].items():
            means = row["means"]
            stderrs = row["stderrs"]
            dyn_mlp = row["paired"]["dynamic_sparse_vs_mlp"]
            dyn_upgd = row["paired"]["dynamic_sparse_vs_upgd"]
            lines.append(
                f"| `{metric_name}` | "
                f"{_fmt(means['mlp'], stderrs['mlp'])} | "
                f"{_fmt(means['upgd'], stderrs['upgd'])} | "
                f"{_fmt(means['dynamic_sparse'], stderrs['dynamic_sparse'])} | "
                f"{dyn_mlp['diff_mean_positive_favors_dynamic_sparse']:+.4f} "
                f"({dyn_mlp['wins_for_dynamic_sparse']}/{dyn_mlp['n_seeds']}) | "
                f"{dyn_upgd['diff_mean_positive_favors_dynamic_sparse']:+.4f} "
                f"({dyn_upgd['wins_for_dynamic_sparse']}/{dyn_upgd['n_seeds']}) |"
            )

    verdict = results["verdict"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_verdict(results: dict[str, Any]) -> str:
    synthetic = results["synthetic"]
    dyn_vs_mlp = synthetic["paired"]["dynamic_sparse_vs_mlp"]
    dyn_vs_upgd = synthetic["paired"]["dynamic_sparse_vs_upgd"]
    synthetic_good = (
        dyn_vs_mlp["baseline_minus_method_mean"] > 0.0
        or dyn_vs_upgd["baseline_minus_method_mean"] > 0.0
    )

    digits = results.get("digits")
    digits_good = False
    if digits is not None:
        acc = digits["aggregate"]["final_window_accuracy"]
        digits_good = (
            acc["paired"]["dynamic_sparse_vs_mlp"][
                "diff_mean_positive_favors_dynamic_sparse"
            ]
            > 0.0
            or acc["paired"]["dynamic_sparse_vs_upgd"][
                "diff_mean_positive_favors_dynamic_sparse"
            ]
            > 0.0
        )

    if synthetic_good and digits_good:
        return (
            "Dynamic sparse rewiring produced at least one positive comparison "
            "on both the synthetic and digits smoke settings. It is worth a "
            "small follow-up sweep, especially over replacement interval/rate."
        )
    if synthetic_good or digits_good:
        return (
            "Dynamic sparse rewiring showed a conditional win, but not a broad "
            "improvement over the fair MLP/UPGD baselines. It is only worth "
            "further work if the next step targets the specific regime where it won."
        )
    return (
        "Dynamic sparse rewiring did not improve over the fair MLP or UPGD in "
        "these smoke comparisons. This version is not worth expanding before "
        "changing the utility signal or rewiring rule."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-steps", type=int, default=1200)
    parser.add_argument("--digits-steps", type=int, default=900)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-3)
    parser.add_argument("--utility-decay", type=float, default=0.99)
    parser.add_argument("--rewire-interval", type=int, default=100)
    parser.add_argument("--unit-replacement-rate", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-digits",
        action="store_true",
        help="Skip sklearn digits smoke.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for harness checks: 1 seed, 200 synthetic steps, 200 digits steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_seeds = 1
        args.synthetic_steps = 200
        args.digits_steps = 200
        args.final_window = 50
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.synthetic_steps <= 0 or args.digits_steps <= 0:
        raise ValueError("step counts must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    synthetic = run_synthetic(args)
    digits: dict[str, Any] | None = None
    digits_error: str | None = None
    if not args.skip_digits:
        try:
            digits = run_digits(args)
        except RuntimeError as exc:
            digits_error = str(exc)

    results: dict[str, Any] = {
        "config": {
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "synthetic_steps": args.synthetic_steps,
            "digits_steps": args.digits_steps,
            "final_window": args.final_window,
            "hidden_size": args.hidden_size,
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "perturbation_sigma": args.perturbation_sigma,
            "utility_decay": args.utility_decay,
            "rewire_interval": args.rewire_interval,
            "unit_replacement_rate": args.unit_replacement_rate,
            "train_fraction": args.train_fraction,
        },
        "synthetic": synthetic,
        "digits": digits,
        "digits_error": digits_error,
        "wall_clock_s": time.time() - t0,
    }
    results["verdict"] = build_verdict(results)

    json_path = args.output_dir / "dynamic_sparse_results.json"
    md_path = args.output_dir / "dynamic_sparse_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(results["verdict"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Conclusive Step 2 learner candidate.

This runner tests a single causal prediction-space learner against the fair MLP
grid across the controlled recursive suite and the existing strict universal
Step 2 regimes.  The learner is a guarded selector over a fixed set of experts:

* robust recursive compositional features,
* fair MLP controls,
* UPGD low-noise plasticity, and
* dynamic sparse rewiring.

Every expert predicts before any update at every time step, every expert then
updates on the same example, and the conclusive learner's route is chosen only
from loss EMAs available before the current target is used for updates.  The
baseline comparison is the best MLP expert from the same run and seed, avoiding
separate-initialization artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

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

from step2_dynamic_sparse import DynamicSparseMLP  # noqa: E402
from step2_expert_mixture import (  # noqa: E402
    DIGITS_REGIMES,
    N_DIGIT_CLASSES,
    SYNTHETIC_REGIMES,
    expand_dataset_names,
    load_digits_arrays,
    make_digits_regime_sequence,
    make_mlp,
    make_synthetic_stream,
    make_upgd,
)
from step2_recursive_feature_utility_probe import (  # noqa: E402
    SUITE_TASKS,
)
from step2_recursive_feature_utility_probe import (  # noqa: E402
    make_data as make_recursive_suite_data,
)

from alberta_framework.core.compositional_features import (  # noqa: E402
    CompositionalFeatureLearner,
)
from alberta_framework.core.future_utility import trace_decay_from_half_life  # noqa: E402
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner  # noqa: E402
from alberta_framework.core.optimizers import ObGDBounding  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/step2_conclusive_learner")
DEFAULT_NOTE_PATH = Path("docs/research/step2_conclusive_learner.md")

EXPERT_NAMES = (
    "recursive_features",
    "polynomial_features",
    "fourier_features",
    "tanh_random_features",
    "mlp_32x32_s01_no_ln",
    "mlp_64x64_s01_no_ln",
    "mlp_32x32",
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_64",
    "upgd_low_noise",
    "dynamic_sparse",
)
MLP_METHODS = (
    "mlp_32x32_s01_no_ln",
    "mlp_64x64_s01_no_ln",
    "mlp_32x32",
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_64",
)
METHOD_NAMES = ("conclusive", *EXPERT_NAMES)
MLP_ROUTE_NAMES = tuple(f"expert_{name}" for name in MLP_METHODS)
SAFE_SOURCE_NAMES = (
    "recursive_features",
    "polynomial_features",
    "fourier_features",
    "tanh_random_features",
)


def safe_route_name(source: str, anchor: str) -> str:
    """Return a stable route name for an MLP-safe specialist interpolation."""
    label = source.removesuffix("_features")
    return f"safe_{label}_{anchor}"


SAFE_ROUTE_SPECS = tuple(
    (source, anchor) for source in SAFE_SOURCE_NAMES for anchor in MLP_METHODS
)
SAFE_ROUTE_NAMES = tuple(
    safe_route_name(source, anchor) for source, anchor in SAFE_ROUTE_SPECS
)
SAFE_ROUTE_BY_NAME = dict(zip(SAFE_ROUTE_NAMES, SAFE_ROUTE_SPECS, strict=True))
ROUTE_NAMES = (
    "all_convex",
    "all_selector",
    "stacked_predictions",
    *SAFE_ROUTE_NAMES,
    "mlp_convex",
    "mlp_selector",
    *MLP_ROUTE_NAMES,
)
SAFE_ROUTE_START = ROUTE_NAMES.index(SAFE_ROUTE_NAMES[0])
MLP_ROUTE_START = ROUTE_NAMES.index("mlp_convex")
CONTROLLED_DATASETS = tuple(f"controlled_{task}" for task in SUITE_TASKS)
LOSS_START = 1
ALL_SELECTOR_COL = LOSS_START + len(EXPERT_NAMES)
MLP_SELECTOR_COL = ALL_SELECTOR_COL + 1
META_ROUTE_COL = MLP_SELECTOR_COL + 1
SAFE_GATE_START = META_ROUTE_COL + 1
ALL_WEIGHT_START = SAFE_GATE_START + len(SAFE_ROUTE_NAMES)
MLP_WEIGHT_START = ALL_WEIGHT_START + len(EXPERT_NAMES)
PRED_START = MLP_WEIGHT_START + len(EXPERT_NAMES)
ROUTE_DISABLE_GROUPS = {
    "safe_recursive": tuple(
        name
        for name, (source, _) in SAFE_ROUTE_BY_NAME.items()
        if source == "recursive_features"
    ),
    "safe": SAFE_ROUTE_NAMES,
    "safe_polynomial": tuple(
        name
        for name, (source, _) in SAFE_ROUTE_BY_NAME.items()
        if source == "polynomial_features"
    ),
    "safe_fourier": tuple(
        name
        for name, (source, _) in SAFE_ROUTE_BY_NAME.items()
        if source == "fourier_features"
    ),
    "safe_tanh_random": tuple(
        name
        for name, (source, _) in SAFE_ROUTE_BY_NAME.items()
        if source == "tanh_random_features"
    ),
    "mlp_experts": MLP_ROUTE_NAMES,
    "fixed_mlp_experts": MLP_ROUTE_NAMES,
    "all_non_mlp": ROUTE_NAMES[:MLP_ROUTE_START],
    "all_routes": ROUTE_NAMES,
}


def stderr(values: np.ndarray) -> float:
    """Return standard error."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def make_recursive_expert(
    n_heads: int,
    feature_dim: int,
    extra_features: int,
) -> CompositionalFeatureLearner:
    """Construct the robust recursive feature expert for one stream shape."""
    n_features = max(36, feature_dim + extra_features)
    return CompositionalFeatureLearner(
        n_features=n_features,
        n_tasks=n_heads,
        candidate_count=n_features,
        step_size_output=0.05,
        step_size_theta=0.005,
        utility_decay=0.99,
        replacement_interval=15,
        min_feature_age=30,
        candidate_min_age=12,
        promotion_margin=1.0,
        promotion_blend=0.6,
        max_depth=3,
        use_obgd=True,
        obgd_kappa=2.0,
        generation_strategy="robust_recursive",
        promotion_output_mode="blend",
        parent_temperature=0.75,
        parent_novelty_weight=0.01,
        parent_depth_prior=0.05,
        retention_depth_bonus=0.02,
        residual_guidance=0.75,
        candidate_imprint_scale=0.2,
        future_utility_mix=0.65,
        future_utility_trace_decay=float(trace_decay_from_half_life(16.0)),
        future_utility_trace_mode="contribution",
        future_utility_rare_task_power=0.25,
    )


def make_dynamic_sparse(
    n_heads: int,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    utility_decay: float,
    rewire_interval: int,
    unit_replacement_rate: float,
) -> DynamicSparseMLP:
    """Construct the dynamic sparse expert."""
    return DynamicSparseMLP(
        n_heads=n_heads,
        hidden_size=hidden_size,
        step_size=step_size,
        sparsity=sparsity,
        utility_decay=utility_decay,
        rewire_interval=rewire_interval,
        unit_replacement_rate=unit_replacement_rate,
        use_layer_norm=True,
        obgd_kappa=2.0,
    )


class PolynomialState(NamedTuple):
    """State for normalized LMS over bounded polynomial features."""

    weights: jax.Array


class PolynomialUpdateResult(NamedTuple):
    """Result for one polynomial feature update."""

    state: PolynomialState


class PolynomialFeatureLearner:
    """Online linear learner over degree-1/2/3 recursive monomial features."""

    def __init__(
        self,
        n_heads: int,
        max_input_dim: int,
        step_size: float,
        feature_clip: float,
    ) -> None:
        self._n_heads = int(n_heads)
        self._max_input_dim = int(max_input_dim)
        self._step_size = float(step_size)
        self._feature_clip = float(feature_clip)

    def _feature_dim(self, input_dim: int) -> int:
        d = min(input_dim, self._max_input_dim)
        return 1 + d + (d * (d + 1)) // 2 + (d * (d + 1) * (d + 2)) // 6

    def _features(self, observation: jax.Array) -> jax.Array:
        x = jnp.clip(
            observation[: self._max_input_dim],
            -self._feature_clip,
            self._feature_clip,
        )
        terms = [jnp.asarray(1.0, dtype=jnp.float32)]
        terms.extend([x[i] for i in range(x.shape[0])])
        terms.extend([x[i] * x[j] for i in range(x.shape[0]) for j in range(i, x.shape[0])])
        terms.extend(
            [
                x[i] * x[j] * x[k]
                for i in range(x.shape[0])
                for j in range(i, x.shape[0])
                for k in range(j, x.shape[0])
            ]
        )
        return jnp.stack(terms).astype(jnp.float32)

    def init(self, feature_dim: int, key: jax.Array) -> PolynomialState:
        """Initialize weights with zeros so this expert starts conservative."""
        del key
        return PolynomialState(
            weights=jnp.zeros((self._n_heads, self._feature_dim(feature_dim)), dtype=jnp.float32)
        )

    def predict(self, state: PolynomialState, observation: jax.Array) -> jax.Array:
        """Predict all heads."""
        return state.weights @ self._features(observation)

    def update(
        self,
        state: PolynomialState,
        observation: jax.Array,
        target: jax.Array,
    ) -> PolynomialUpdateResult:
        """Update active heads with normalized LMS."""
        features = self._features(observation)
        prediction = state.weights @ features
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        errors = jnp.where(active, safe_target - prediction, 0.0)
        normalizer = 1.0 + jnp.sum(features * features)
        delta = (
            jnp.asarray(self._step_size, dtype=jnp.float32)
            * errors[:, None]
            * features[None, :]
            / normalizer
        )
        return PolynomialUpdateResult(state=PolynomialState(weights=state.weights + delta))


class FourierState(NamedTuple):
    """State for normalized LMS over Fourier features."""

    weights: jax.Array


class FourierUpdateResult(NamedTuple):
    """Result for one Fourier feature update."""

    state: FourierState


class FourierFeatureLearner:
    """Online linear learner over sin/cos features for frequency streams."""

    def __init__(
        self,
        n_heads: int,
        max_input_dim: int,
        step_size: float,
        frequencies: tuple[float, ...],
    ) -> None:
        self._n_heads = int(n_heads)
        self._max_input_dim = int(max_input_dim)
        self._step_size = float(step_size)
        self._frequencies = tuple(float(freq) for freq in frequencies)

    def _feature_dim(self, input_dim: int) -> int:
        d = min(input_dim, self._max_input_dim)
        return 1 + d + 2 * d * len(self._frequencies)

    def _features(self, observation: jax.Array) -> jax.Array:
        x = observation[: self._max_input_dim]
        terms = [jnp.asarray(1.0, dtype=jnp.float32)]
        terms.extend([x[i] for i in range(x.shape[0])])
        for i in range(x.shape[0]):
            for frequency in self._frequencies:
                z = jnp.asarray(frequency, dtype=jnp.float32) * x[i]
                terms.append(jnp.sin(z))
                terms.append(jnp.cos(z))
        return jnp.stack(terms).astype(jnp.float32)

    def init(self, feature_dim: int, key: jax.Array) -> FourierState:
        """Initialize output weights."""
        del key
        return FourierState(
            weights=jnp.zeros((self._n_heads, self._feature_dim(feature_dim)), dtype=jnp.float32)
        )

    def predict(self, state: FourierState, observation: jax.Array) -> jax.Array:
        """Predict all heads."""
        return state.weights @ self._features(observation)

    def update(
        self,
        state: FourierState,
        observation: jax.Array,
        target: jax.Array,
    ) -> FourierUpdateResult:
        """Update active heads with normalized LMS."""
        features = self._features(observation)
        prediction = state.weights @ features
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        errors = jnp.where(active, safe_target - prediction, 0.0)
        normalizer = 1.0 + jnp.sum(features * features)
        delta = (
            jnp.asarray(self._step_size, dtype=jnp.float32)
            * errors[:, None]
            * features[None, :]
            / normalizer
        )
        return FourierUpdateResult(state=FourierState(weights=state.weights + delta))


class RandomTanhState(NamedTuple):
    """State for normalized LMS over fixed random tanh features."""

    output_weights: jax.Array
    feature_weights: jax.Array
    feature_biases: jax.Array


class RandomTanhUpdateResult(NamedTuple):
    """Result for one random tanh feature update."""

    state: RandomTanhState


class RandomTanhFeatureLearner:
    """Online linear learner over fixed random tanh features."""

    def __init__(
        self,
        n_heads: int,
        width: int,
        step_size: float,
        weight_scale: float,
    ) -> None:
        self._n_heads = int(n_heads)
        self._width = int(width)
        self._step_size = float(step_size)
        self._weight_scale = float(weight_scale)

    def _features(self, state: RandomTanhState, observation: jax.Array) -> jax.Array:
        hidden = jnp.tanh(state.feature_weights @ observation + state.feature_biases)
        return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), hidden])

    def init(self, feature_dim: int, key: jax.Array) -> RandomTanhState:
        """Initialize fixed random features and output weights."""
        key_w, key_b = jr.split(key)
        scale = self._weight_scale / math.sqrt(max(feature_dim, 1))
        feature_weights = scale * jr.normal(
            key_w, (self._width, feature_dim), dtype=jnp.float32
        )
        feature_biases = jr.uniform(
            key_b,
            (self._width,),
            dtype=jnp.float32,
            minval=-self._weight_scale,
            maxval=self._weight_scale,
        )
        output_weights = jnp.zeros((self._n_heads, self._width + 1), dtype=jnp.float32)
        return RandomTanhState(
            output_weights=output_weights,
            feature_weights=feature_weights,
            feature_biases=feature_biases,
        )

    def predict(self, state: RandomTanhState, observation: jax.Array) -> jax.Array:
        """Predict all heads."""
        return state.output_weights @ self._features(state, observation)

    def update(
        self,
        state: RandomTanhState,
        observation: jax.Array,
        target: jax.Array,
    ) -> RandomTanhUpdateResult:
        """Update active output heads with normalized LMS."""
        features = self._features(state, observation)
        prediction = state.output_weights @ features
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        errors = jnp.where(active, safe_target - prediction, 0.0)
        normalizer = 1.0 + jnp.sum(features * features)
        delta = (
            jnp.asarray(self._step_size, dtype=jnp.float32)
            * errors[:, None]
            * features[None, :]
            / normalizer
        )
        return RandomTanhUpdateResult(
            state=RandomTanhState(
                output_weights=state.output_weights + delta,
                feature_weights=state.feature_weights,
                feature_biases=state.feature_biases,
            )
        )


def masked_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error over active target heads."""
    active = ~jnp.isnan(target)
    safe_target = jnp.where(active, target, 0.0)
    active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(active, (prediction - safe_target) ** 2, 0.0)) / active_count


def route_scoring_mse(
    prediction: jax.Array,
    target: jax.Array,
    rare_active_step_weight: float,
) -> jax.Array:
    """Route-selection loss with optional emphasis on multi-head rare steps.

    Some controlled streams expose a rare target head only on a small fraction of
    time steps. A plain route window can hide rare-step spikes behind frequent
    head wins. This scorer leaves the reported MSE unchanged, but can upweight
    route-selection state updates when more than one target head is active.
    """
    loss = masked_mse(prediction, target)
    if rare_active_step_weight == 0.0:
        return loss
    active_count = jnp.sum((~jnp.isnan(target)).astype(jnp.float32))
    multiplier = jnp.where(
        active_count > 1.0,
        1.0 + jnp.asarray(rare_active_step_weight, dtype=jnp.float32),
        1.0,
    )
    return loss * multiplier


def split_name_spec(spec: Any) -> frozenset[str]:
    """Parse a comma-separated name list."""
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        return frozenset(part.strip() for part in spec.split(",") if part.strip())
    return frozenset(str(part).strip() for part in spec if str(part).strip())


def disabled_expert_names(args: argparse.Namespace) -> frozenset[str]:
    """Return disabled conclusive-routing expert names from args."""
    return split_name_spec(getattr(args, "disable_experts", ""))


def enabled_safe_route_sources(args: argparse.Namespace) -> frozenset[str]:
    """Return specialist sources allowed to form MLP-safe routes."""
    return split_name_spec(getattr(args, "safe_route_sources", "recursive_features"))


def disabled_route_names(args: argparse.Namespace) -> frozenset[str]:
    """Return disabled conclusive route names from args, expanding groups."""
    disabled: set[str] = set()
    for name in split_name_spec(getattr(args, "disable_routes", "")):
        if name in ROUTE_DISABLE_GROUPS:
            disabled.update(ROUTE_DISABLE_GROUPS[name])
        else:
            disabled.add(name)
    expert_disabled = disabled_expert_names(args)
    safe_sources = enabled_safe_route_sources(args)
    for route_name, (source, anchor) in SAFE_ROUTE_BY_NAME.items():
        if (
            source not in safe_sources
            or source in expert_disabled
            or anchor in expert_disabled
        ):
            disabled.add(route_name)
    for expert in expert_disabled:
        fixed_route = f"expert_{expert}"
        if fixed_route in ROUTE_NAMES:
            disabled.add(fixed_route)
    return frozenset(disabled)


def expert_enabled_mask(args: argparse.Namespace) -> np.ndarray:
    """Return a boolean mask for experts available to the conclusive router."""
    disabled = disabled_expert_names(args)
    return np.asarray([name not in disabled for name in EXPERT_NAMES], dtype=np.bool_)


def route_enabled_mask(args: argparse.Namespace) -> np.ndarray:
    """Return a boolean mask for routes available to the conclusive router."""
    disabled = disabled_route_names(args)
    return np.asarray([name not in disabled for name in ROUTE_NAMES], dtype=np.bool_)


def expert_softmax_weights(
    loss_ema: jax.Array,
    eta: float,
    enabled_mask: jax.Array | None = None,
) -> jax.Array:
    """Return loss-weighted convex expert weights."""
    logits = -jnp.asarray(eta, dtype=jnp.float32) * loss_ema
    if enabled_mask is not None:
        logits = jnp.where(enabled_mask, logits, -jnp.inf)
    return jax.nn.softmax(logits)


def mlp_softmax_weights(
    loss_ema: jax.Array,
    eta: float,
    enabled_mask: jax.Array | None = None,
) -> jax.Array:
    """Return full-length convex weights over the MLP subset."""
    mlp_indices = jnp.asarray([EXPERT_NAMES.index(name) for name in MLP_METHODS])
    mlp_logits = -jnp.asarray(eta, dtype=jnp.float32) * loss_ema[mlp_indices]
    if enabled_mask is not None:
        mlp_logits = jnp.where(enabled_mask[mlp_indices], mlp_logits, -jnp.inf)
    local_weights = jax.nn.softmax(mlp_logits)
    return jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32).at[mlp_indices].set(
        local_weights
    )


def masked_log_weight_softmax(
    log_weights: jax.Array,
    enabled_mask: jax.Array,
) -> jax.Array:
    """Return softmax weights while excluding disabled entries."""
    masked = jnp.where(enabled_mask, log_weights, -jnp.inf)
    return jax.nn.softmax(masked)


def route_softmax_weights_from_scores(
    route_scores: jax.Array,
    route_enabled: jax.Array,
    eta: float,
) -> jax.Array:
    """Return route weights that favor low causal route scores."""
    logits = -jnp.asarray(eta, dtype=jnp.float32) * route_scores
    enabled = route_enabled & jnp.isfinite(route_scores)
    return masked_log_weight_softmax(logits, enabled)


def blend_predictions(
    base_prediction: jax.Array,
    floor_prediction: jax.Array,
    floor_weight: float,
) -> jax.Array:
    """Blend a route prediction toward a protective floor prediction."""
    weight = jnp.asarray(floor_weight, dtype=jnp.float32)
    return (1.0 - weight) * base_prediction + weight * floor_prediction


def rare_active_step_mask(target: jax.Array, target_missing_seen: jax.Array) -> jax.Array:
    """Return true for rare-active multi-head steps after sparse-head evidence."""
    active = ~jnp.isnan(target)
    active_count = jnp.sum(active.astype(jnp.float32))
    return jnp.logical_and(active_count > 1.0, target_missing_seen)


def effective_mlp_floor_weight(
    target: jax.Array,
    target_missing_seen: jax.Array,
    base_weight: float,
    rare_active_extra_weight: float,
) -> jax.Array:
    """Return floor weight, optionally stronger on multi-head rare-active steps."""
    extra = jnp.where(
        rare_active_step_mask(target, target_missing_seen),
        jnp.asarray(rare_active_extra_weight, dtype=jnp.float32),
        0.0,
    )
    return jnp.asarray(base_weight, dtype=jnp.float32) + extra


def full_mlp_weights_from_local(local_weights: jax.Array) -> jax.Array:
    """Expand local MLP weights to the full expert axis."""
    mlp_indices = jnp.asarray([EXPERT_NAMES.index(name) for name in MLP_METHODS])
    return jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32).at[mlp_indices].set(
        local_weights
    )


def stacker_features(preds: jax.Array) -> jax.Array:
    """Return per-head prediction-stacker features with a bias term."""
    bias = jnp.ones((1, preds.shape[1]), dtype=jnp.float32)
    return jnp.concatenate([bias, preds.astype(jnp.float32)], axis=0).T


def stacker_predict(weights: jax.Array, preds: jax.Array) -> jax.Array:
    """Predict one target vector from expert predictions."""
    features = stacker_features(preds)
    return jnp.sum(weights * features, axis=1)


def stacker_update(
    weights: jax.Array,
    preds: jax.Array,
    target: jax.Array,
    step_size: float,
) -> jax.Array:
    """Update per-head stacker weights with normalized LMS."""
    features = stacker_features(preds)
    prediction = jnp.sum(weights * features, axis=1)
    active = ~jnp.isnan(target)
    safe_target = jnp.where(active, target, 0.0)
    errors = jnp.where(active, safe_target - prediction, 0.0)
    normalizer = 1.0 + jnp.sum(features * features, axis=1)
    delta = (
        jnp.asarray(step_size, dtype=jnp.float32)
        * errors[:, None]
        * features
        / normalizer[:, None]
    )
    return weights + delta


def safe_gate_gradients(
    gates: jax.Array,
    source_preds: jax.Array,
    anchor_preds: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Return gradients for MLP-anchored safe specialist gates."""
    active = ~jnp.isnan(target)
    safe_target = jnp.where(active, target, 0.0)
    active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    residual = source_preds - anchor_preds
    safe_preds = anchor_preds + gates[:, None] * residual
    errors = safe_preds - safe_target[None, :]
    masked_grad = jnp.where(active[None, :], errors * residual, 0.0)
    return 2.0 * jnp.sum(masked_grad, axis=1) / active_count


def route_loss_window_scores(
    route_window_sums: jax.Array,
    route_window_square_sums: jax.Array,
    route_buffer_count: jax.Array,
    fallback_scores: jax.Array,
    stderr_penalty: float,
) -> jax.Array:
    """Return causal route scores from a recent loss window.

    Scores are `mean + stderr_penalty * stderr`, so positive penalties prefer
    routes with both low recent MSE and low recent uncertainty.
    """
    count = route_buffer_count.astype(jnp.float32)
    safe_count = jnp.maximum(count, 1.0)
    mean = route_window_sums / safe_count
    centered_sum_squares = jnp.maximum(
        route_window_square_sums - (route_window_sums * route_window_sums) / safe_count,
        0.0,
    )
    sample_variance = jnp.where(
        count > 1.0,
        centered_sum_squares / jnp.maximum(count - 1.0, 1.0),
        0.0,
    )
    standard_error = jnp.sqrt(sample_variance / safe_count)
    window_scores = mean + jnp.asarray(stderr_penalty, dtype=jnp.float32) * standard_error
    return jnp.where(route_buffer_count > 0, window_scores, fallback_scores)


def route_loss_history_window_scores(
    route_loss_buffer: jax.Array,
    route_buffer_idx: jax.Array,
    route_buffer_count: jax.Array,
    window_sizes: jax.Array,
    fallback_scores: jax.Array,
    stderr_penalty: float,
) -> jax.Array:
    """Return causal route scores for multiple recent-loss windows."""
    max_window = route_loss_buffer.shape[0]
    offsets = jnp.arange(max_window, dtype=jnp.int32)
    history_indices = (route_buffer_idx - 1 - offsets) % max_window
    history = route_loss_buffer[history_indices]
    history_squares = history * history

    def score_one(window_size: jax.Array) -> jax.Array:
        count = jnp.minimum(route_buffer_count, window_size)
        mask = offsets < count
        weighted_history = jnp.where(mask[:, None], history, 0.0)
        weighted_squares = jnp.where(mask[:, None], history_squares, 0.0)
        sums = jnp.sum(weighted_history, axis=0)
        square_sums = jnp.sum(weighted_squares, axis=0)
        return route_loss_window_scores(
            sums,
            square_sums,
            count,
            fallback_scores,
            stderr_penalty,
        )

    return jax.vmap(score_one)(window_sizes)


def parse_route_selector_windows(args: argparse.Namespace) -> tuple[int, ...]:
    """Return positive route selector windows from CLI args."""
    if getattr(args, "route_policy_mode", "score") == "telemetry_worker_b":
        return (100, 150)
    base_window = int(args.selector_window or args.final_window)
    spec = str(getattr(args, "route_selector_windows", "")).strip()
    if not spec:
        return (base_window,)
    windows = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not windows:
        return (base_window,)
    if any(window <= 0 for window in windows):
        raise ValueError("--route-selector-windows must contain positive integers")
    return tuple(dict.fromkeys(windows))


def telemetry_worker_b_gate(
    route_id_buffer: jax.Array,
    selector_id_buffer: jax.Array,
    telemetry_count: jax.Array,
) -> jax.Array:
    """Return a causal gate for Worker-B-style routing from recent telemetry."""
    count = telemetry_count.astype(jnp.float32)
    denom = jnp.maximum(count, 1.0)
    valid = jnp.arange(route_id_buffer.shape[0], dtype=jnp.int32) < telemetry_count
    valid_f = valid.astype(jnp.float32)

    def route_fraction(route_name: str) -> jax.Array:
        route_id = jnp.asarray(ROUTE_NAMES.index(route_name), dtype=jnp.int32)
        return jnp.sum(((route_id_buffer == route_id) & valid).astype(jnp.float32)) / denom

    def route_group_fraction(prefix: str) -> jax.Array:
        mask = jnp.asarray(
            [name.startswith(prefix) for name in ROUTE_NAMES],
            dtype=bool,
        )
        return jnp.sum(mask[route_id_buffer] * valid_f) / denom

    def selector_fraction(expert_name: str) -> jax.Array:
        expert_id = jnp.asarray(EXPERT_NAMES.index(expert_name), dtype=jnp.int32)
        return (
            jnp.sum(((selector_id_buffer == expert_id) & valid).astype(jnp.float32))
            / denom
        )

    all_selector_tanh = selector_fraction("tanh_random_features")
    all_selector_poly = selector_fraction("polynomial_features")
    all_selector_fourier = selector_fraction("fourier_features")
    all_selector_recursive = selector_fraction("recursive_features")
    route_mlp_convex = route_fraction("mlp_convex")
    route_safe_poly = route_group_fraction("safe_polynomial")
    route_safe_rec = route_group_fraction("safe_recursive")
    route_expert_mlp = route_group_fraction("expert_mlp")

    rule_tanh = all_selector_tanh > 0.90
    rule_poly_mlp = jnp.logical_and(all_selector_poly > 0.95, route_mlp_convex > 0.05)
    rule_safe_churn = jnp.logical_and(
        route_safe_poly + route_safe_rec > 0.75,
        jnp.logical_and(
            route_expert_mlp < 0.10,
            jnp.logical_and(all_selector_fourier < 0.50, all_selector_recursive < 0.10),
        ),
    )
    return jnp.logical_and(
        telemetry_count > 0,
        jnp.logical_or(rule_tanh, jnp.logical_or(rule_poly_mlp, rule_safe_churn)),
    )


def contextual_route_enabled_mask(
    route_enabled: jax.Array,
    all_selector_idx: jax.Array,
    mode: str,
) -> jax.Array:
    """Return dynamic route mask using only pre-current-step selector context."""
    if mode == "off":
        return route_enabled
    safe_route_sources = jnp.asarray(
        [EXPERT_NAMES.index(source) for source, _ in SAFE_ROUTE_SPECS],
        dtype=jnp.int32,
    )
    safe_context_enabled = safe_route_sources == all_selector_idx
    return route_enabled.at[SAFE_ROUTE_START:MLP_ROUTE_START].set(
        route_enabled[SAFE_ROUTE_START:MLP_ROUTE_START] & safe_context_enabled
    )


def route_selector(
    route_ema: jax.Array,
    step_count: jax.Array,
    warmup_steps: int,
    guard_margin: float,
    current_route: jax.Array,
    switch_margin: float,
) -> jax.Array:
    """Choose a route using only pre-current-step route EMAs."""
    all_route_losses = route_ema[:MLP_ROUTE_START]
    mlp_route_losses = route_ema[MLP_ROUTE_START:]
    best_all_route = jnp.argmin(all_route_losses).astype(jnp.int32)
    best_mlp_route = (
        MLP_ROUTE_START + jnp.argmin(mlp_route_losses)
    ).astype(jnp.int32)
    best_all_loss = route_ema[best_all_route]
    best_mlp_loss = route_ema[best_mlp_route]
    routed = jnp.where(
        best_all_loss <= best_mlp_loss - jnp.asarray(guard_margin, dtype=jnp.float32),
        best_all_route,
        best_mlp_route,
    )
    warmup_route = jnp.asarray(
        ROUTE_NAMES.index("expert_mlp_32x32"),
        dtype=jnp.int32,
    )
    warmup_route = jnp.where(
        jnp.isfinite(route_ema[warmup_route]),
        warmup_route,
        best_mlp_route,
    )
    current_loss = route_ema[current_route]
    should_switch = route_ema[routed] <= current_loss - jnp.asarray(
        switch_margin, dtype=jnp.float32
    )
    stable_route = jnp.where(should_switch, routed, current_route)
    return jnp.where(
        step_count < jnp.asarray(warmup_steps, dtype=jnp.int32),
        warmup_route,
        stable_route.astype(jnp.int32),
    )


def run_conclusive_stream(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[Any, Any]], np.ndarray]:
    """Run the conclusive learner and all experts on one materialized stream."""
    n_heads = int(targets.shape[1])
    feature_dim = int(observations.shape[1])
    keys = jr.split(key, len(EXPERT_NAMES))
    selector_windows = parse_route_selector_windows(args)
    selector_window = max(selector_windows)
    selector_window_array = jnp.asarray(selector_windows, dtype=jnp.int32)
    telemetry_window = int(args.route_telemetry_window or args.final_window)
    expert_enabled = jnp.asarray(expert_enabled_mask(args))
    route_enabled = jnp.asarray(route_enabled_mask(args))
    recursive = make_recursive_expert(n_heads, feature_dim, args.recursive_extra_features)
    polynomial = PolynomialFeatureLearner(
        n_heads=n_heads,
        max_input_dim=args.polynomial_max_input_dim,
        step_size=args.polynomial_step_size,
        feature_clip=args.polynomial_feature_clip,
    )
    fourier = FourierFeatureLearner(
        n_heads=n_heads,
        max_input_dim=args.fourier_max_input_dim,
        step_size=args.fourier_step_size,
        frequencies=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    )
    tanh_random = RandomTanhFeatureLearner(
        n_heads=n_heads,
        width=args.tanh_random_width,
        step_size=args.tanh_random_step_size,
        weight_scale=args.tanh_random_weight_scale,
    )
    mlp32_s01 = MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(32, 32),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
    mlp64_s01 = MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(64, 64),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
    mlp32 = MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(32, 32),
        step_size=args.step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
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
    learners = (
        recursive,
        polynomial,
        fourier,
        tanh_random,
        mlp32_s01,
        mlp64_s01,
        mlp32,
        mlp64,
        mlp128,
        mlp6464,
        upgd,
        dynamic,
    )
    disabled_experts = disabled_expert_names(args)
    stacker_anchor = "mlp_32x32"
    if stacker_anchor in disabled_experts:
        stacker_anchor = next(name for name in MLP_METHODS if name not in disabled_experts)
    stacker_anchor_idx = EXPERT_NAMES.index(stacker_anchor)
    stacker_initial_weights = jnp.zeros(
        (n_heads, len(EXPERT_NAMES) + 1),
        dtype=jnp.float32,
    ).at[:, 1 + stacker_anchor_idx].set(1.0)
    initial_states = (
        recursive.init(feature_dim, keys[0]),
        polynomial.init(feature_dim, keys[1]),
        fourier.init(feature_dim, keys[2]),
        tanh_random.init(feature_dim, keys[3]),
        mlp32_s01.init(feature_dim, keys[4]),
        mlp64_s01.init(feature_dim, keys[5]),
        mlp32.init(feature_dim, keys[6]),
        mlp64.init(feature_dim, keys[7]),
        mlp128.init(feature_dim, keys[8]),
        mlp6464.init(feature_dim, keys[9]),
        upgd.init(feature_dim, keys[10]),
        dynamic.init(feature_dim, keys[11]),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(MLP_METHODS), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(ROUTE_NAMES), dtype=jnp.float32),
        jnp.zeros(len(SAFE_ROUTE_NAMES), dtype=jnp.float32),
        stacker_initial_weights,
        jnp.zeros((selector_window, len(EXPERT_NAMES)), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros((selector_window, len(ROUTE_NAMES)), dtype=jnp.float32),
        jnp.zeros(len(ROUTE_NAMES), dtype=jnp.float32),
        jnp.zeros(len(ROUTE_NAMES), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.full((selector_window,), -1, dtype=jnp.int32),
        jnp.zeros(n_heads, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.full((telemetry_window,), -1, dtype=jnp.int32),
        jnp.full((telemetry_window,), -1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.full(
            (len(selector_windows),),
            ROUTE_NAMES.index("expert_mlp_32x32"),
            dtype=jnp.int32,
        ),
        jnp.asarray(False),
        jnp.array(0, dtype=jnp.int32),
    )
    route_decay = jnp.asarray(args.route_loss_decay, dtype=jnp.float32)
    hedge_discount = jnp.asarray(args.hedge_discount, dtype=jnp.float32)
    hedge_eta = jnp.asarray(args.hedge_eta, dtype=jnp.float32)
    use_discounted_hedge = args.weighting_scheme == "discounted_hedge"
    use_window_expert_selectors = args.expert_selector_score_source == "window"
    classification_guard_enabled = n_heads == N_DIGIT_CLASSES
    safe_source_indices = jnp.asarray(
        [EXPERT_NAMES.index(source) for source, _ in SAFE_ROUTE_SPECS],
        dtype=jnp.int32,
    )
    safe_anchor_indices = jnp.asarray(
        [EXPERT_NAMES.index(anchor) for _, anchor in SAFE_ROUTE_SPECS],
        dtype=jnp.int32,
    )

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        (
            recursive_s,
            polynomial_s,
            fourier_s,
            tanh_random_s,
            mlp32_s01_s,
            mlp64_s01_s,
            mlp32_s,
            mlp64_s,
            mlp128_s,
            mlp6464_s,
            upgd_s,
            dynamic_s,
            log_weights,
            mlp_log_weights,
            loss_ema,
            route_ema,
            safe_gates,
            stacker_weights,
            expert_loss_buffer,
            expert_window_sums,
            route_loss_buffer,
            route_window_sums,
            route_window_square_sums,
            route_buffer_idx,
            route_buffer_count,
            label_buffer,
            label_counts,
            label_buffer_idx,
            label_buffer_count,
            route_id_buffer,
            selector_id_buffer,
            telemetry_buffer_idx,
            telemetry_buffer_count,
            current_routes,
            target_missing_seen,
            step_count,
        ) = carry
        obs, tgt = sample
        preds = jnp.stack(
            [
                recursive.predict(recursive_s, obs),
                polynomial.predict(polynomial_s, obs),
                fourier.predict(fourier_s, obs),
                tanh_random.predict(tanh_random_s, obs),
                mlp32_s01.predict(mlp32_s01_s, obs),
                mlp64_s01.predict(mlp64_s01_s, obs),
                mlp32.predict(mlp32_s, obs),
                mlp64.predict(mlp64_s, obs),
                mlp128.predict(mlp128_s, obs),
                mlp6464.predict(mlp6464_s, obs),
                upgd.predict(upgd_s, obs),
                dynamic.predict(dynamic_s, obs),
            ],
            axis=0,
        )
        losses = jax.vmap(lambda pred: masked_mse(pred, tgt))(preds)
        mlp_indices = jnp.asarray([EXPERT_NAMES.index(name) for name in MLP_METHODS])
        mlp_enabled = expert_enabled[mlp_indices]
        if use_discounted_hedge:
            all_weights = masked_log_weight_softmax(log_weights, expert_enabled)
            local_mlp_weights = masked_log_weight_softmax(mlp_log_weights, mlp_enabled)
            mlp_weights = full_mlp_weights_from_local(local_mlp_weights)
        else:
            all_weights = expert_softmax_weights(
                loss_ema,
                args.hedge_eta,
                enabled_mask=expert_enabled,
            )
            mlp_weights = mlp_softmax_weights(
                loss_ema,
                args.hedge_eta,
                enabled_mask=expert_enabled,
            )
        expert_scores = jnp.where(
            route_buffer_count > 0,
            expert_window_sums
            / jnp.maximum(route_buffer_count.astype(jnp.float32), 1.0),
            loss_ema,
        )
        selector_scores = expert_scores if use_window_expert_selectors else loss_ema
        masked_selector_scores = jnp.where(expert_enabled, selector_scores, jnp.inf)
        all_selector_idx = jnp.argmin(masked_selector_scores).astype(jnp.int32)
        mlp_selector_idx = mlp_indices[
            jnp.argmin(jnp.where(mlp_enabled, selector_scores[mlp_indices], jnp.inf))
        ].astype(jnp.int32)
        safe_route_preds = preds[safe_anchor_indices] + safe_gates[:, None] * (
            preds[safe_source_indices] - preds[safe_anchor_indices]
        )
        stacked_pred = stacker_predict(stacker_weights, preds)
        route_preds = jnp.concatenate(
            [
                jnp.stack(
                    [
                        jnp.sum(all_weights[:, None] * preds, axis=0),
                        preds[all_selector_idx],
                        stacked_pred,
                    ],
                    axis=0,
                ),
                safe_route_preds,
                jnp.stack(
                    [
                        jnp.sum(mlp_weights[:, None] * preds, axis=0),
                        preds[mlp_selector_idx],
                    ],
                    axis=0,
                ),
                preds[mlp_indices],
            ],
            axis=0,
        )
        route_eval_losses = jax.vmap(lambda pred: masked_mse(pred, tgt))(route_preds)
        route_score_losses = jax.vmap(
            lambda pred: route_scoring_mse(
                pred,
                tgt,
                args.route_rare_active_step_weight,
            )
        )(route_preds)
        contextual_route_enabled = contextual_route_enabled_mask(
            route_enabled,
            all_selector_idx,
            args.safe_route_context_gate,
        )
        if len(selector_windows) == 1:
            route_score_matrix = route_loss_window_scores(
                route_window_sums,
                route_window_square_sums,
                route_buffer_count,
                route_ema,
                args.route_variance_penalty,
            )[None, :]
        else:
            route_score_matrix = route_loss_history_window_scores(
                route_loss_buffer,
                route_buffer_idx,
                route_buffer_count,
                selector_window_array,
                route_ema,
                args.route_variance_penalty,
            )
        masked_route_score_matrix = jnp.where(
            contextual_route_enabled[None, :],
            route_score_matrix,
            jnp.inf,
        )
        if args.route_policy_mode == "telemetry_worker_b":
            candidate_route_indices = jnp.stack(
                [
                    route_selector(
                        masked_route_score_matrix[0],
                        step_count,
                        args.warmup_steps,
                        args.guard_margin,
                        current_routes[0],
                        0.0,
                    ),
                    route_selector(
                        masked_route_score_matrix[1],
                        step_count,
                        args.warmup_steps,
                        args.guard_margin,
                        current_routes[1],
                        args.worker_b_switch_margin,
                    ),
                ]
            )
            use_worker_b = telemetry_worker_b_gate(
                route_id_buffer,
                selector_id_buffer,
                telemetry_buffer_count,
            )
            selected_window_idx = jnp.where(use_worker_b, 1, 0).astype(jnp.int32)
        else:
            candidate_route_indices = jax.vmap(
                lambda scores, current_route: route_selector(
                    scores,
                    step_count,
                    args.warmup_steps,
                    args.guard_margin,
                    current_route,
                    args.route_switch_margin,
                )
            )(masked_route_score_matrix, current_routes)
            candidate_window_scores = masked_route_score_matrix[
                jnp.arange(len(selector_windows)),
                candidate_route_indices,
            ]
            selected_window_idx = jnp.argmin(candidate_window_scores).astype(jnp.int32)
        meta_route_idx = candidate_route_indices[selected_window_idx]
        masked_route_scores = masked_route_score_matrix[selected_window_idx]
        best_mlp_guard_route = jnp.asarray(
            ROUTE_NAMES.index("expert_mlp_h64_64"),
            dtype=jnp.int32,
        )
        recent_class_count = jnp.sum((label_counts > 0).astype(jnp.int32))
        class_guard_active = jnp.logical_and(
            jnp.asarray(classification_guard_enabled),
            jnp.logical_and(
                label_buffer_count
                >= jnp.asarray(args.classification_guard_min_window, dtype=jnp.int32),
                recent_class_count
                <= jnp.asarray(args.classification_guard_max_recent_classes, dtype=jnp.int32),
            ),
        )
        class_guard_active = jnp.logical_and(
            class_guard_active,
            route_enabled[best_mlp_guard_route],
        )
        meta_route_idx = jnp.where(class_guard_active, best_mlp_guard_route, meta_route_idx)
        conclusive_pred = route_preds[meta_route_idx]
        conclusive_loss = route_eval_losses[meta_route_idx]
        if args.route_deployment_mode in ("softmax", "softmax_rare_active"):
            route_weights = route_softmax_weights_from_scores(
                masked_route_scores,
                contextual_route_enabled,
                args.route_softmax_eta,
            )
            softmax_pred = jnp.sum(route_weights[:, None] * route_preds, axis=0)
            softmax_loss = masked_mse(softmax_pred, tgt)
            rare_active_step = rare_active_step_mask(tgt, target_missing_seen)
            softmax_mode_enabled = jnp.logical_or(
                args.route_deployment_mode == "softmax",
                rare_active_step,
            )
            softmax_enabled = jnp.logical_and(
                step_count >= jnp.asarray(args.warmup_steps, dtype=jnp.int32),
                jnp.logical_and(~class_guard_active, softmax_mode_enabled),
            )
            conclusive_pred = jnp.where(softmax_enabled, softmax_pred, conclusive_pred)
            conclusive_loss = jnp.where(softmax_enabled, softmax_loss, conclusive_loss)
        if (
            args.mlp_floor_blend_weight
            + args.mlp_floor_rare_active_extra_weight
            > 0.0
        ):
            if args.mlp_floor_source == "convex":
                mlp_floor_pred = jnp.sum(mlp_weights[:, None] * preds, axis=0)
            elif args.mlp_floor_source == "selector":
                mlp_floor_pred = preds[mlp_selector_idx]
            else:
                mlp_floor_pred = preds[EXPERT_NAMES.index(args.mlp_floor_source)]
            floor_weight = effective_mlp_floor_weight(
                tgt,
                target_missing_seen,
                args.mlp_floor_blend_weight,
                args.mlp_floor_rare_active_extra_weight,
            )
            floor_pred = blend_predictions(
                conclusive_pred,
                mlp_floor_pred,
                floor_weight,
            )
            floor_loss = masked_mse(floor_pred, tgt)
            floor_enabled = jnp.logical_and(
                step_count >= jnp.asarray(args.warmup_steps, dtype=jnp.int32),
                ~class_guard_active,
            )
            conclusive_pred = jnp.where(floor_enabled, floor_pred, conclusive_pred)
            conclusive_loss = jnp.where(floor_enabled, floor_loss, conclusive_loss)
        new_log_weights = hedge_discount * log_weights - hedge_eta * losses
        new_log_weights = new_log_weights - jnp.max(
            jnp.where(expert_enabled, new_log_weights, -jnp.inf)
        )
        new_mlp_log_weights = (
            hedge_discount * mlp_log_weights - hedge_eta * losses[mlp_indices]
        )
        new_mlp_log_weights = new_mlp_log_weights - jnp.max(
            jnp.where(mlp_enabled, new_mlp_log_weights, -jnp.inf)
        )
        new_loss_ema = route_decay * loss_ema + (1.0 - route_decay) * losses
        new_route_ema = route_decay * route_ema + (1.0 - route_decay) * route_score_losses
        gate_grads = safe_gate_gradients(
            safe_gates,
            preds[safe_source_indices],
            preds[safe_anchor_indices],
            tgt,
        )
        new_safe_gates = jnp.clip(
            safe_gates - jnp.asarray(args.safe_gate_step_size, dtype=jnp.float32) * gate_grads,
            0.0,
            1.0,
        )
        new_stacker_weights = stacker_update(
            stacker_weights,
            preds,
            tgt,
            args.stacker_step_size,
        )
        old_expert_losses = expert_loss_buffer[route_buffer_idx]
        old_route_losses = route_loss_buffer[route_buffer_idx]
        buffer_full = route_buffer_count >= selector_window
        new_expert_window_sums = jnp.where(
            buffer_full,
            expert_window_sums - old_expert_losses + losses,
            expert_window_sums + losses,
        )
        new_expert_loss_buffer = expert_loss_buffer.at[route_buffer_idx].set(losses)
        new_route_window_sums = jnp.where(
            buffer_full,
            route_window_sums - old_route_losses + route_score_losses,
            route_window_sums + route_score_losses,
        )
        route_loss_squares = route_score_losses * route_score_losses
        old_route_loss_squares = old_route_losses * old_route_losses
        new_route_window_square_sums = jnp.where(
            buffer_full,
            route_window_square_sums - old_route_loss_squares + route_loss_squares,
            route_window_square_sums + route_loss_squares,
        )
        new_route_loss_buffer = route_loss_buffer.at[route_buffer_idx].set(route_score_losses)
        new_route_buffer_idx = (route_buffer_idx + 1) % selector_window
        new_route_buffer_count = jnp.minimum(
            route_buffer_count + 1,
            jnp.asarray(selector_window, dtype=jnp.int32),
        )
        label = jnp.argmax(tgt).astype(jnp.int32)
        old_label = label_buffer[label_buffer_idx]
        label_buffer_full = label_buffer_count >= selector_window
        decremented_label_counts = jnp.where(
            label_buffer_full,
            label_counts.at[old_label].add(-1),
            label_counts,
        )
        new_label_counts = decremented_label_counts.at[label].add(1)
        new_label_buffer = label_buffer.at[label_buffer_idx].set(label)
        new_label_buffer_idx = (label_buffer_idx + 1) % selector_window
        new_label_buffer_count = jnp.minimum(
            label_buffer_count + 1,
            jnp.asarray(selector_window, dtype=jnp.int32),
        )
        new_route_id_buffer = route_id_buffer.at[telemetry_buffer_idx].set(
            candidate_route_indices[0]
        )
        new_selector_id_buffer = selector_id_buffer.at[telemetry_buffer_idx].set(
            all_selector_idx
        )
        new_telemetry_buffer_idx = (telemetry_buffer_idx + 1) % telemetry_window
        new_telemetry_buffer_count = jnp.minimum(
            telemetry_buffer_count + 1,
            jnp.asarray(telemetry_window, dtype=jnp.int32),
        )

        recursive_result = recursive.update(recursive_s, obs, tgt)
        polynomial_result = polynomial.update(polynomial_s, obs, tgt)
        fourier_result = fourier.update(fourier_s, obs, tgt)
        tanh_random_result = tanh_random.update(tanh_random_s, obs, tgt)
        mlp32_s01_result = mlp32_s01.update(mlp32_s01_s, obs, tgt)
        mlp64_s01_result = mlp64_s01.update(mlp64_s01_s, obs, tgt)
        mlp32_result = mlp32.update(mlp32_s, obs, tgt)
        mlp64_result = mlp64.update(mlp64_s, obs, tgt)
        mlp128_result = mlp128.update(mlp128_s, obs, tgt)
        mlp6464_result = mlp6464.update(mlp6464_s, obs, tgt)
        upgd_result = upgd.update(upgd_s, obs, tgt)
        dynamic_result = dynamic.update(dynamic_s, obs, tgt)

        pred_classes = jnp.concatenate(
            [
                jnp.asarray([jnp.argmax(conclusive_pred)], dtype=jnp.float32),
                jnp.argmax(preds, axis=1).astype(jnp.float32),
            ]
        )
        metric = jnp.concatenate(
            [
                jnp.asarray([conclusive_loss], dtype=jnp.float32),
                losses.astype(jnp.float32),
                jnp.asarray([all_selector_idx.astype(jnp.float32)], dtype=jnp.float32),
                jnp.asarray([mlp_selector_idx.astype(jnp.float32)], dtype=jnp.float32),
                jnp.asarray([meta_route_idx.astype(jnp.float32)], dtype=jnp.float32),
                new_safe_gates.astype(jnp.float32),
                all_weights.astype(jnp.float32),
                mlp_weights.astype(jnp.float32),
                pred_classes,
            ]
        )
        new_target_missing_seen = jnp.logical_or(
            target_missing_seen,
            jnp.any(jnp.isnan(tgt)),
        )
        return (
            recursive_result.state,
            polynomial_result.state,
            fourier_result.state,
            tanh_random_result.state,
            mlp32_s01_result.state,
            mlp64_s01_result.state,
            mlp32_result.state,
            mlp64_result.state,
            mlp128_result.state,
            mlp6464_result.state,
            upgd_result.state,
            dynamic_result.state,
            new_log_weights,
            new_mlp_log_weights,
            new_loss_ema,
            new_route_ema,
            new_safe_gates,
            new_stacker_weights,
            new_expert_loss_buffer,
            new_expert_window_sums,
            new_route_loss_buffer,
            new_route_window_sums,
            new_route_window_square_sums,
            new_route_buffer_idx,
            new_route_buffer_count,
            new_label_buffer,
            new_label_counts,
            new_label_buffer_idx,
            new_label_buffer_count,
            new_route_id_buffer,
            new_selector_id_buffer,
            new_telemetry_buffer_idx,
            new_telemetry_buffer_count,
            candidate_route_indices,
            new_target_missing_seen,
            step_count + 1,
        ), metric

    final_tuple, metrics = jax.lax.scan(step_fn, initial_states, (observations, targets))
    metrics.block_until_ready()
    final_states = {
        name: (learner, final_tuple[idx])
        for idx, (name, learner) in enumerate(zip(EXPERT_NAMES, learners, strict=True))
    }
    final_states["_stacker_weights"] = (None, final_tuple[len(EXPERT_NAMES) + 5])
    return final_states, np.asarray(metrics)


def summarize_prequential(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Summarize online losses and optional accuracies for every method."""
    window = min(final_window, metrics.shape[0])
    summary: dict[str, dict[str, float]] = {}
    for idx, method in enumerate(METHOD_NAMES):
        loss_col = 0 if method == "conclusive" else LOSS_START + idx - 1
        entry = {
            "online_mean_mse": float(np.mean(metrics[:, loss_col])),
            "final_window_mse": float(np.mean(metrics[-window:, loss_col])),
        }
        if labels is not None:
            pred_col = PRED_START + idx
            correct = metrics[:, pred_col].astype(np.int32) == labels
            entry["online_mean_accuracy"] = float(np.mean(correct))
            entry["final_window_accuracy"] = float(np.mean(correct[-window:]))
        if method == "conclusive":
            routes = metrics[:, META_ROUTE_COL].astype(np.int32)
            counts = Counter(ROUTE_NAMES[int(route)] for route in routes)
            for route in ROUTE_NAMES:
                entry[f"meta_route_fraction_{route}"] = counts.get(route, 0) / len(routes)
            all_selectors = metrics[:, ALL_SELECTOR_COL].astype(np.int32)
            all_counts = Counter(EXPERT_NAMES[int(route)] for route in all_selectors)
            mlp_selectors = metrics[:, MLP_SELECTOR_COL].astype(np.int32)
            mlp_counts = Counter(EXPERT_NAMES[int(route)] for route in mlp_selectors)
            for expert in EXPERT_NAMES:
                entry[f"all_selector_fraction_{expert}"] = all_counts.get(
                    expert, 0
                ) / len(all_selectors)
                entry[f"mlp_selector_fraction_{expert}"] = mlp_counts.get(
                    expert, 0
                ) / len(mlp_selectors)
        summary[method] = entry
    return summary


def evaluate_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate one classifier expert on the held-out digits split."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def evaluate_conclusive_classifier(
    final_states: dict[str, tuple[Any, Any]],
    metrics: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    online_labels: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluate the final conclusive route on held-out classification."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    expert_enabled = expert_enabled_mask(args)
    expert_preds = []
    for expert in EXPERT_NAMES:
        learner, state = final_states[expert]
        expert_preds.append(jax.vmap(lambda obs: learner.predict(state, obs))(observations))
    preds = jnp.stack(expert_preds, axis=0)

    deployment_source = "online_mse_route"
    meta_route = int(round(float(metrics[-1, META_ROUTE_COL])))
    all_selector = int(round(float(metrics[-1, ALL_SELECTOR_COL])))
    mlp_selector = int(round(float(metrics[-1, MLP_SELECTOR_COL])))
    safe_gates = jnp.asarray(
        metrics[-1, SAFE_GATE_START : SAFE_GATE_START + len(SAFE_ROUTE_NAMES)].astype(
            np.float32
        )
    )
    all_weights = jnp.asarray(
        metrics[-1, ALL_WEIGHT_START : ALL_WEIGHT_START + len(EXPERT_NAMES)].astype(
            np.float32
        )
    )
    mlp_weights = jnp.asarray(
        metrics[-1, MLP_WEIGHT_START : MLP_WEIGHT_START + len(EXPERT_NAMES)].astype(
            np.float32
        )
    )
    if ROUTE_NAMES[meta_route] == "all_convex":
        conclusive_preds = jnp.sum(all_weights[:, None, None] * preds, axis=0)
    elif ROUTE_NAMES[meta_route] == "all_selector":
        conclusive_preds = preds[all_selector]
    elif ROUTE_NAMES[meta_route] == "stacked_predictions":
        stacker_weights = final_states["_stacker_weights"][1]
        conclusive_preds = jax.vmap(
            lambda sample_preds: stacker_predict(stacker_weights, sample_preds)
        )(jnp.swapaxes(preds, 0, 1))
    elif ROUTE_NAMES[meta_route] in SAFE_ROUTE_BY_NAME:
        source, anchor = SAFE_ROUTE_BY_NAME[ROUTE_NAMES[meta_route]]
        source_idx = EXPERT_NAMES.index(source)
        anchor_idx = EXPERT_NAMES.index(anchor)
        gate_idx = SAFE_ROUTE_NAMES.index(ROUTE_NAMES[meta_route])
        conclusive_preds = preds[anchor_idx] + safe_gates[gate_idx] * (
            preds[source_idx] - preds[anchor_idx]
        )
    elif ROUTE_NAMES[meta_route] == "mlp_convex":
        conclusive_preds = jnp.sum(mlp_weights[:, None, None] * preds, axis=0)
    elif ROUTE_NAMES[meta_route] == "mlp_selector":
        conclusive_preds = preds[mlp_selector]
    else:
        expert = ROUTE_NAMES[meta_route].removeprefix("expert_")
        conclusive_preds = preds[EXPERT_NAMES.index(expert)]

    if args.digits_deployment_objective == "all_h128_blend":
        blend_weight = jnp.asarray(args.h128_blend_weight, dtype=jnp.float32)
        conclusive_preds = (1.0 - blend_weight) * jnp.sum(
            all_weights[:, None, None] * preds,
            axis=0,
        ) + blend_weight * preds[EXPERT_NAMES.index("mlp_h128")]
        deployment_source = "all_h128_blend"

    if (
        online_labels is not None
        and online_labels.size > 0
        and args.digits_deployment_objective == "accuracy"
    ):
        window = min(args.final_window, int(online_labels.shape[0]))
        expert_accuracies = []
        for idx, enabled in enumerate(expert_enabled):
            if not enabled:
                expert_accuracies.append(-math.inf)
                continue
            pred_col = PRED_START + 1 + idx
            correct = metrics[-window:, pred_col].astype(np.int32) == online_labels[-window:]
            expert_accuracies.append(float(np.mean(correct)))
        best_expert_idx = int(np.argmax(np.asarray(expert_accuracies)))
        conclusive_preds = preds[best_expert_idx]
        deployment_source = "online_accuracy_expert"

    if online_labels is not None and online_labels.size > 0:
        window = min(args.final_window, int(online_labels.shape[0]))
        lifetime_classes = np.unique(online_labels)
        recent_classes = np.unique(online_labels[-window:])
        lifetime_fraction = lifetime_classes.shape[0] / N_DIGIT_CLASSES
        recent_fraction = recent_classes.shape[0] / max(lifetime_classes.shape[0], 1)
        if (
            expert_enabled[EXPERT_NAMES.index("recursive_features")]
            and
            lifetime_fraction >= args.retention_min_lifetime_class_fraction
            and recent_fraction <= args.retention_max_recent_class_fraction
        ):
            conclusive_preds = preds[EXPERT_NAMES.index("recursive_features")]
            deployment_source = "class_imbalance_recursive_retention"

    mse = jnp.mean((conclusive_preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(conclusive_preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    result = {"test_mse": float(mse), "test_accuracy": float(accuracy)}
    result["deployment_route_id"] = float(meta_route)
    result["deployment_all_selector_id"] = float(all_selector)
    result["deployment_mlp_selector_id"] = float(mlp_selector)
    result["deployment_source_id"] = float(
        {
            "online_mse_route": 0,
            "online_accuracy_expert": 1,
            "class_imbalance_recursive_retention": 2,
            "all_h128_blend": 3,
        }[deployment_source]
    )
    return result


def is_higher_better(metric: str) -> bool:
    """Return whether a metric is higher-is-better."""
    return metric.endswith("accuracy")


def better_value(values: dict[str, float], metric: str) -> tuple[str, float]:
    """Return the best method/value for one metric."""
    method = max(values, key=values.__getitem__) if is_higher_better(metric) else min(
        values,
        key=values.__getitem__,
    )
    return method, values[method]


def paired_diff(method_value: float, baseline_value: float, metric: str) -> float:
    """Positive paired difference favors the method."""
    if is_higher_better(metric):
        return method_value - baseline_value
    return baseline_value - method_value


def paired_conclusive_vs_group(
    records: list[dict[str, Any]],
    metric: str,
    group: tuple[str, ...],
    group_name: str,
) -> dict[str, Any]:
    """Compare conclusive learner to the per-seed best baseline group member."""
    diffs: list[float] = []
    best_methods: list[str] = []
    for record in records:
        methods = record["methods"]
        method_value = float(methods["conclusive"][metric])
        group_values = {method: float(methods[method][metric]) for method in group}
        best_method, best_value = better_value(group_values, metric)
        best_methods.append(best_method)
        diffs.append(paired_diff(method_value, best_value, metric))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "metric": metric,
        "baseline_group": group_name,
        "paired_diff_mean_positive_favors_conclusive": float(np.mean(diff_arr)),
        "paired_diff_stderr": stderr(diff_arr),
        "wins_for_conclusive": int(np.sum(diff_arr > 0.0)),
        "wins_for_baseline": int(np.sum(diff_arr < 0.0)),
        "ties": int(np.sum(diff_arr == 0.0)),
        "n": int(diff_arr.shape[0]),
        "diffs": diff_arr.tolist(),
        "best_baseline_counts": dict(Counter(best_methods)),
    }


def summary_row(values: np.ndarray) -> dict[str, Any]:
    """Return mean/stderr/values for one aggregate row."""
    return {
        "mean": float(np.mean(values)),
        "stderr": stderr(values),
        "values": values.tolist(),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate records by dataset and method."""
    aggregate: dict[str, Any] = {}
    datasets = sorted({record["dataset_name"] for record in records})
    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset_name"] == dataset]
        metrics = sorted(dataset_records[0]["methods"]["conclusive"])
        aggregate[dataset] = {}
        for method in METHOD_NAMES:
            aggregate[dataset][method] = {}
            for metric in metrics:
                if metric not in dataset_records[0]["methods"][method]:
                    continue
                values = np.asarray(
                    [record["methods"][method][metric] for record in dataset_records],
                    dtype=np.float64,
                )
                aggregate[dataset][method][metric] = summary_row(values)
        main_metrics = [
            metric
            for metric in (
                "final_window_mse",
                "online_mean_mse",
                "test_mse",
                "final_window_accuracy",
                "online_mean_accuracy",
                "test_accuracy",
            )
            if metric in dataset_records[0]["methods"]["conclusive"]
        ]
        aggregate[dataset]["comparisons"] = {
            metric: {
                "conclusive_vs_best_mlp": paired_conclusive_vs_group(
                    dataset_records,
                    metric,
                    MLP_METHODS,
                    "best_mlp",
                ),
                "conclusive_vs_best_expert": paired_conclusive_vs_group(
                    dataset_records,
                    metric,
                    EXPERT_NAMES,
                    "best_expert",
                ),
            }
            for metric in main_metrics
        }
    return aggregate


def controlled_dataset_name(task_mode: str) -> str:
    """Return canonical name for one controlled recursive task."""
    return f"controlled_{task_mode}"


def expand_benchmark_names(spec: str) -> list[str]:
    """Expand benchmark aliases to concrete dataset names."""
    if spec == "controlled":
        return list(CONTROLLED_DATASETS)
    if spec == "universal":
        return expand_dataset_names("all")
    if spec == "all":
        return list(CONTROLLED_DATASETS) + expand_dataset_names("all")
    names: list[str] = []
    for item in [part.strip() for part in spec.split(",") if part.strip()]:
        if item in {"controlled", "universal", "all"}:
            names.extend(expand_benchmark_names(item))
        elif item in CONTROLLED_DATASETS:
            names.append(item)
        elif item in SUITE_TASKS:
            names.append(controlled_dataset_name(item))
        else:
            names.extend(expand_dataset_names(item))
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Run one dataset/seed combination."""
    labels_np: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    if dataset_name.startswith("controlled_"):
        task_mode = dataset_name.removeprefix("controlled_")
        observations, targets = make_recursive_suite_data(
            seed=seed,
            num_steps=args.steps,
            feature_dim=args.feature_dim,
            noise_std=args.noise_std,
            task_mode=task_mode,
            rare_period=args.rare_period,
        )
        dataset_meta: dict[str, Any] = {
            "benchmark_family": "controlled_recursive_suite",
            "task_mode": task_mode,
            "feature_dim": args.feature_dim,
        }
    elif dataset_name in SYNTHETIC_REGIMES:
        observations, targets, dataset_meta = make_synthetic_stream(
            steps=args.steps,
            seed=seed + 20_000,
            regime=dataset_name,
        )
    elif dataset_name in DIGITS_REGIMES:
        x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
            seed=seed,
            train_fraction=args.train_fraction,
        )
        observations, targets, labels, x_test, y_test, stream_meta = (
            make_digits_regime_sequence(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                steps=args.steps,
                seed=seed + 10_000,
                regime=dataset_name,
                phase_length=args.phase_length,
                mask_keep_fraction=args.mask_keep_fraction,
                mask_noise_std=args.mask_noise_std,
            )
        )
        dataset_meta.update(stream_meta)
        labels_np = np.asarray(labels)
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    final_states, metrics = run_conclusive_stream(
        observations=observations,
        targets=targets,
        key=jr.key(seed),
        args=args,
    )
    methods = summarize_prequential(metrics, args.final_window, labels_np)
    if dataset_name in DIGITS_REGIMES:
        assert x_test is not None and y_test is not None
        for expert in EXPERT_NAMES:
            learner, state = final_states[expert]
            methods[expert].update(evaluate_classifier(learner, state, x_test, y_test))
        methods["conclusive"].update(
            evaluate_conclusive_classifier(
                final_states,
                metrics,
                x_test,
                y_test,
                labels_np,
                args,
            )
        )

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "dataset": dataset_meta,
            "methods": methods,
        },
        dataset_meta,
        metrics,
    )


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    lines = [
        "# Step 2 Conclusive Learner Candidate",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} steps, final "
            f"window {cfg['final_window']}; route loss decay="
            f"{cfg['route_loss_decay']}, warmup={cfg['warmup_steps']}, "
            f"guard margin={cfg['guard_margin']}."
        ),
        "",
        "Positive conclusive-vs-best-MLP differences favor the conclusive learner. "
        "For MSE this is best MLP minus conclusive; for accuracy this is "
        "conclusive minus best MLP.",
        "",
    ]
    positive_tasks = 0
    total_tasks = 0
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
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
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            total_tasks += 1
            cmp_row = comparisons["final_window_mse"]["conclusive_vs_best_mlp"]
            diff = cmp_row["paired_diff_mean_positive_favors_conclusive"]
            if diff > 0.0:
                positive_tasks += 1
            lines.append(
                f"`final_window_mse` conclusive-vs-best-MLP diff: {diff:+.4f} "
                f"+/- {cmp_row['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{cmp_row['wins_for_conclusive']}/{cmp_row['wins_for_baseline']}/"
                f"{cmp_row['ties']}."
            )
        if "test_accuracy" in comparisons:
            cmp_row = comparisons["test_accuracy"]["conclusive_vs_best_mlp"]
            lines.append(
                f"`test_accuracy` conclusive-vs-best-MLP diff: "
                f"{cmp_row['paired_diff_mean_positive_favors_conclusive']:+.4f} "
                f"+/- {cmp_row['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{cmp_row['wins_for_conclusive']}/{cmp_row['wins_for_baseline']}/"
                f"{cmp_row['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Suite Summary",
            "",
            (
                f"Conclusive learner has positive mean final-window MSE delta "
                f"against best fair MLP on {positive_tasks}/{total_tasks} "
                "configured datasets."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default="controlled",
        help=(
            "Comma-separated benchmark names or aliases: controlled, universal, "
            "all, controlled task names, or universal portfolio dataset names."
        ),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-4)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamic-hidden-size", type=int, default=64)
    parser.add_argument("--dynamic-utility-decay", type=float, default=0.99)
    parser.add_argument("--dynamic-rewire-interval", type=int, default=180)
    parser.add_argument("--dynamic-unit-replacement-rate", type=float, default=0.05)
    parser.add_argument("--recursive-extra-features", type=int, default=32)
    parser.add_argument("--polynomial-max-input-dim", type=int, default=8)
    parser.add_argument("--polynomial-step-size", type=float, default=0.5)
    parser.add_argument("--polynomial-feature-clip", type=float, default=3.0)
    parser.add_argument("--fourier-max-input-dim", type=int, default=8)
    parser.add_argument("--fourier-step-size", type=float, default=0.3)
    parser.add_argument("--tanh-random-width", type=int, default=256)
    parser.add_argument("--tanh-random-step-size", type=float, default=0.4)
    parser.add_argument("--tanh-random-weight-scale", type=float, default=1.0)
    parser.add_argument("--route-loss-decay", type=float, default=0.99)
    parser.add_argument(
        "--selector-window",
        type=int,
        default=0,
        help="Causal route-loss window; 0 uses --final-window.",
    )
    parser.add_argument(
        "--route-selector-windows",
        default="",
        help=(
            "Optional comma-separated causal route-loss windows. When multiple "
            "windows are supplied, each proposes a route and the route with the "
            "lowest causal window score is deployed."
        ),
    )
    parser.add_argument(
        "--expert-selector-score-source",
        choices=("ema", "window"),
        default="ema",
        help=(
            "Loss score used by hard expert selectors. 'ema' preserves the "
            "original long-memory selectors; 'window' uses the same causal "
            "selector window as route selection."
        ),
    )
    parser.add_argument("--hedge-eta", type=float, default=8.0)
    parser.add_argument(
        "--weighting-scheme",
        choices=("ema_softmax", "discounted_hedge"),
        default="ema_softmax",
        help=(
            "Convex route weighting. 'ema_softmax' matches the original "
            "conclusive runner; 'discounted_hedge' uses cumulative discounted "
            "expert losses, matching the stronger universal portfolio router."
        ),
    )
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument(
        "--route-deployment-mode",
        choices=("hard", "softmax", "softmax_rare_active"),
        default="hard",
        help=(
            "Top-level online route deployment. 'hard' chooses one route; "
            "'softmax' causally averages route predictions using prior route "
            "scores; 'softmax_rare_active' does this only on multi-head "
            "rare-active target steps."
        ),
    )
    parser.add_argument("--route-softmax-eta", type=float, default=8.0)
    parser.add_argument(
        "--route-policy-mode",
        choices=("score", "telemetry_worker_b"),
        default="score",
        help=(
            "How to choose among route-memory policies. 'score' uses the lowest "
            "causal route score. 'telemetry_worker_b' runs current and "
            "Worker-B-style route policies and chooses Worker B under a causal "
            "route/selector telemetry gate."
        ),
    )
    parser.add_argument("--worker-b-switch-margin", type=float, default=0.010)
    parser.add_argument(
        "--route-telemetry-window",
        type=int,
        default=0,
        help=(
            "Recent telemetry window for --route-policy-mode telemetry_worker_b; "
            "0 uses final-window."
        ),
    )
    parser.add_argument("--mlp-floor-blend-weight", type=float, default=0.0)
    parser.add_argument("--mlp-floor-rare-active-extra-weight", type=float, default=0.0)
    parser.add_argument(
        "--mlp-floor-source",
        choices=("selector", "convex", *MLP_METHODS),
        default="selector",
        help=(
            "MLP prediction used by --mlp-floor-blend-weight. 'selector' uses "
            "the current hard MLP selector; 'convex' uses the MLP Hedge mixture; "
            "MLP method names use that fixed expert."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--guard-margin", type=float, default=0.0)
    parser.add_argument("--route-switch-margin", type=float, default=0.0)
    parser.add_argument(
        "--route-variance-penalty",
        type=float,
        default=0.0,
        help=(
            "Nonnegative multiplier on each route's causal recent-window "
            "standard error. The default 0.0 preserves mean-loss routing."
        ),
    )
    parser.add_argument(
        "--route-rare-active-step-weight",
        type=float,
        default=0.0,
        help=(
            "Nonnegative route-selection multiplier for time steps with multiple "
            "active target heads. The default 0.0 preserves raw-MSE routing."
        ),
    )
    parser.add_argument("--safe-gate-step-size", type=float, default=0.05)
    parser.add_argument("--stacker-step-size", type=float, default=0.006)
    parser.add_argument(
        "--safe-route-sources",
        default="recursive_features",
        help=(
            "Comma-separated non-MLP experts allowed to form MLP-safe "
            "interpolation routes. Valid sources: "
            + ", ".join(SAFE_SOURCE_NAMES)
            + "."
        ),
    )
    parser.add_argument(
        "--safe-route-context-gate",
        choices=("off", "source_selector"),
        default="off",
        help=(
            "Dynamic causal safe-route mask. 'source_selector' allows an MLP-safe "
            "specialist route only when its specialist source is the current hard "
            "all-selector expert."
        ),
    )
    parser.add_argument(
        "--disable-experts",
        default="",
        help=(
            "Comma-separated experts to remove from conclusive routing. Experts "
            "still run for paired diagnostics, but their predictions cannot be "
            "selected, mixed, or used by deployment guards."
        ),
    )
    parser.add_argument(
        "--disable-routes",
        default="",
        help=(
            "Comma-separated routes or route groups to remove from conclusive "
            "routing. Groups: safe_recursive, safe, mlp_experts, "
            "fixed_mlp_experts, all_non_mlp, all_routes."
        ),
    )
    parser.add_argument("--classification-guard-min-window", type=int, default=1)
    parser.add_argument("--classification-guard-max-recent-classes", type=int, default=6)
    parser.add_argument("--retention-min-lifetime-class-fraction", type=float, default=0.8)
    parser.add_argument("--retention-max-recent-class-fraction", type=float, default=0.6)
    parser.add_argument(
        "--digits-deployment-objective",
        choices=("mse", "accuracy", "all_h128_blend"),
        default="mse",
    )
    parser.add_argument("--h128-blend-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.feature_dim < 4:
        raise ValueError("--feature-dim must be at least 4")
    if not 0.0 < args.route_loss_decay < 1.0:
        raise ValueError("--route-loss-decay must be in (0, 1)")
    if args.selector_window < 0:
        raise ValueError("--selector-window must be non-negative")
    _ = parse_route_selector_windows(args)
    if args.hedge_eta <= 0.0:
        raise ValueError("--hedge-eta must be positive")
    if not 0.0 < args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in (0, 1]")
    if args.route_softmax_eta <= 0.0:
        raise ValueError("--route-softmax-eta must be positive")
    if not 0.0 <= args.mlp_floor_blend_weight <= 1.0:
        raise ValueError("--mlp-floor-blend-weight must be in [0, 1]")
    if args.mlp_floor_rare_active_extra_weight < 0.0:
        raise ValueError("--mlp-floor-rare-active-extra-weight must be non-negative")
    if args.mlp_floor_blend_weight + args.mlp_floor_rare_active_extra_weight > 1.0:
        raise ValueError(
            "--mlp-floor-blend-weight + --mlp-floor-rare-active-extra-weight "
            "must be at most 1"
        )
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.route_switch_margin < 0.0:
        raise ValueError("--route-switch-margin must be non-negative")
    if args.worker_b_switch_margin < 0.0:
        raise ValueError("--worker-b-switch-margin must be non-negative")
    if args.route_telemetry_window < 0:
        raise ValueError("--route-telemetry-window must be non-negative")
    if args.route_variance_penalty < 0.0:
        raise ValueError("--route-variance-penalty must be non-negative")
    if args.route_rare_active_step_weight < 0.0:
        raise ValueError("--route-rare-active-step-weight must be non-negative")
    if args.safe_gate_step_size < 0.0:
        raise ValueError("--safe-gate-step-size must be non-negative")
    if args.stacker_step_size < 0.0:
        raise ValueError("--stacker-step-size must be non-negative")
    unknown_safe_sources = enabled_safe_route_sources(args).difference(SAFE_SOURCE_NAMES)
    if unknown_safe_sources:
        raise ValueError(
            "--safe-route-sources contains unknown experts: "
            + ", ".join(sorted(unknown_safe_sources))
        )
    unknown_experts = disabled_expert_names(args).difference(EXPERT_NAMES)
    if unknown_experts:
        raise ValueError(
            "--disable-experts contains unknown experts: "
            + ", ".join(sorted(unknown_experts))
        )
    raw_disabled_routes = split_name_spec(getattr(args, "disable_routes", ""))
    unknown_routes = {
        name
        for name in raw_disabled_routes
        if name not in ROUTE_NAMES and name not in ROUTE_DISABLE_GROUPS
    }
    if unknown_routes:
        raise ValueError(
            "--disable-routes contains unknown routes or groups: "
            + ", ".join(sorted(unknown_routes))
        )
    enabled_experts = expert_enabled_mask(args)
    if not np.any(enabled_experts):
        raise ValueError("--disable-experts cannot disable every expert")
    if not np.any(enabled_experts[[EXPERT_NAMES.index(name) for name in MLP_METHODS]]):
        raise ValueError("--disable-experts must leave at least one fair MLP route enabled")
    enabled_routes = route_enabled_mask(args)
    if not np.any(enabled_routes):
        raise ValueError("--disable-routes cannot disable every route")
    if not np.any(enabled_routes[MLP_ROUTE_START:]):
        raise ValueError("--disable-routes must leave at least one MLP-protected route enabled")
    if args.classification_guard_min_window < 0:
        raise ValueError("--classification-guard-min-window must be non-negative")
    if args.classification_guard_max_recent_classes <= 0:
        raise ValueError("--classification-guard-max-recent-classes must be positive")
    if not 0.0 <= args.retention_min_lifetime_class_fraction <= 1.0:
        raise ValueError("--retention-min-lifetime-class-fraction must be in [0, 1]")
    if not 0.0 <= args.retention_max_recent_class_fraction <= 1.0:
        raise ValueError("--retention-max-recent-class-fraction must be in [0, 1]")
    if not 0.0 <= args.h128_blend_weight <= 1.0:
        raise ValueError("--h128-blend-weight must be in [0, 1]")
    if (
        args.digits_deployment_objective == "all_h128_blend"
        and "mlp_h128" in disabled_expert_names(args)
    ):
        raise ValueError("--digits-deployment-objective all_h128_blend requires mlp_h128")
    if args.dynamic_rewire_interval <= 0:
        raise ValueError("--dynamic-rewire-interval must be positive")
    if not 0.0 <= args.dynamic_unit_replacement_rate <= 1.0:
        raise ValueError("--dynamic-unit-replacement-rate must be in [0, 1]")
    if args.polynomial_max_input_dim <= 0:
        raise ValueError("--polynomial-max-input-dim must be positive")
    if args.polynomial_step_size < 0.0:
        raise ValueError("--polynomial-step-size must be non-negative")
    if args.polynomial_feature_clip <= 0.0:
        raise ValueError("--polynomial-feature-clip must be positive")
    if args.fourier_max_input_dim <= 0:
        raise ValueError("--fourier-max-input-dim must be positive")
    if args.fourier_step_size < 0.0:
        raise ValueError("--fourier-step-size must be non-negative")
    if args.tanh_random_width <= 0:
        raise ValueError("--tanh-random-width must be positive")
    if args.tanh_random_step_size < 0.0:
        raise ValueError("--tanh-random-step-size must be non-negative")
    if args.tanh_random_weight_scale <= 0.0:
        raise ValueError("--tanh-random-weight-scale must be positive")


def main() -> None:
    """Run the configured benchmark matrix."""
    args = parse_args()
    validate_args(args)
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = expand_benchmark_names(args.benchmarks)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in dataset_names:
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            print(f"dataset={dataset_name} seed={seed}: running conclusive learner")
            record, meta, metrics = run_one_dataset_seed(dataset_name, seed, args)
            records.append(record)
            datasets_meta[dataset_name] = meta
            np.savez_compressed(
                args.output_dir / f"{dataset_name}_seed{seed}_curves.npz",
                metrics=metrics,
            )
            methods = record["methods"]
            best_mlp = min(methods[name]["final_window_mse"] for name in MLP_METHODS)
            print(
                f"dataset={dataset_name} seed={seed}: final MSE "
                f"conclusive={methods['conclusive']['final_window_mse']:.4f}, "
                f"best_mlp={best_mlp:.4f}, route="
                f"{ROUTE_NAMES[int(round(float(metrics[-1, META_ROUTE_COL])))]}"
            )

    results = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "note_path": str(args.note_path),
            "benchmarks": dataset_names,
            "expert_names": list(EXPERT_NAMES),
            "mlp_comparator_methods": list(MLP_METHODS),
            "route_names": list(ROUTE_NAMES),
        },
        "datasets": datasets_meta,
        "records": records,
        "aggregate": aggregate_records(records),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "conclusive_step2_guarded_recursive_portfolio_candidate",
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

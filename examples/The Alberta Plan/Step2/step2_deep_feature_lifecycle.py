#!/usr/bin/env python3
"""Step 2 native deep feature lifecycle comparison.

This runner evaluates ``DeepFeatureGeneratingMultiHeadMLPLearner`` against
compact Step 2 comparators on nonlinear, interaction, compositional, and
digits-like streams:

* fair MLP baselines: ``mlp_64`` and ``mlp_64_64``
* ``UPGDLearner``
* native deep feature lifecycle variants

The native variants are small switches around the same learner: static shadow
candidates, residual-gradient imprinting, orthogonalized candidate
initialization, age/utility protection, layer-wise replacement budgeting, a
perturbation/plasticity hybrid, and opt-in Net2Net-style function-preserving
promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CBPMultiHeadMLPLearner,
    CompositionalFeatureLearner,
    CompositionalStream,
    ContinualBackpropConfig,
    DeepFeatureGeneratingMultiHeadMLPLearner,
    DeepFeatureLifecycleConfig,
    FrequencyMismatchStream,
    InteractionFeatureDiscoveryStream,
    MultiHeadMLPLearner,
    NonlinearFeatureDiscoveryStream,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
    run_cbp_learning_loop,
    run_compositional_arrays,
    run_deep_feature_lifecycle_arrays,
    run_multi_head_learning_loop,
    run_upgd_arrays,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_deep_net2net_attempt")
DEFAULT_NOTE = Path("docs/research/step2_deep_net2net_attempt.md")
OBGD_KAPPA = 2.0
STEP_SIZE = 0.03
SPARSITY = 0.5
FINAL_WINDOW_DEFAULT = 500

BASE_METHOD_NAMES = [
    "mlp_64",
    "mlp_64_64",
    "upgd",
]

DEEP_METHOD_NAMES = [
    "deep_feature_lifecycle",
    "deep_lr_low",
    "deep_lr_high",
    "deep_imprint",
    "deep_imprint_orthogonal",
    "deep_protected",
    "deep_first",
    "deep_final",
    "deep_utility_norm",
    "deep_no_layernorm",
    "deep_warmup",
    "deep_bank8",
    "deep_fast_cadence",
    "deep_upgd_hybrid",
    "deep_preserve_outgoing",
    "deep_preserve_no_layernorm",
    "deep_active_perturb",
    "deep_active_perturb_low",
    "deep_active_perturb_preserve",
    "deep_soft_gate_final",
    "deep_soft_gate_fast",
    "deep_soft_gate_l1",
    "deep_soft_gate_bank8",
    "deep_net2net",
    "deep_net2net_guarded",
    "deep_net2net_final",
    "deep_net2net_fast",
    "deep_shallow",
    "deep_shallow_final",
    "deep_shallow_nlms",
    "deep_shallow_soft_gate",
    "deep_shallow_net2net",
]

METHOD_NAMES = [
    *BASE_METHOD_NAMES,
    *DEEP_METHOD_NAMES,
]

STREAM_SPECS: list[dict[str, Any]] = [
    {
        "name": "nonlinear",
        "factory": lambda: NonlinearFeatureDiscoveryStream(
            feature_dim=8,
            n_tasks=3,
            n_latents=24,
            n_contexts=4,
            context_length=500,
            active_latents_per_context=4,
            noise_std=0.05,
        ),
        "feature_dim": 8,
        "n_tasks": 3,
    },
    {
        "name": "interaction",
        "factory": lambda: InteractionFeatureDiscoveryStream(
            feature_dim=8,
            n_tasks=3,
            n_contexts=4,
            context_length=500,
            active_pairs_per_context=3,
            noise_std=0.05,
        ),
        "feature_dim": 8,
        "n_tasks": 3,
    },
    {
        "name": "out_of_class_polynomial",
        "factory": lambda: OutOfClassPolynomialStream(
            feature_dim=8,
            n_tasks=3,
            n_contexts=4,
            context_length=500,
            active_triples_per_context=2,
            noise_std=0.05,
        ),
        "feature_dim": 8,
        "n_tasks": 3,
    },
    {
        "name": "frequency_mismatch",
        "factory": lambda: FrequencyMismatchStream(
            feature_dim=4,
            n_tasks=2,
            n_components_per_task=3,
            n_contexts=4,
            context_length=500,
            noise_std=0.05,
        ),
        "feature_dim": 4,
        "n_tasks": 2,
    },
    {
        "name": "compositional",
        "factory": lambda: CompositionalStream(
            feature_dim=6,
            n_tasks=3,
            inner_hidden=4,
            outer_components=5,
            n_contexts=4,
            context_length=500,
            noise_std=0.05,
        ),
        "feature_dim": 6,
        "n_tasks": 3,
    },
]


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a stream into observation and target arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    return observations, targets


def collect_digits_arrays(
    seed: int,
    num_steps: int,
    train_fraction: float = 0.7,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a small sklearn-digits prequential one-hot stream."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        msg = "scikit-learn is required for the optional digits stream."
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    for cls in range(10):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        train_indices.extend(cls_idx[:n_train].tolist())
    train_indices_arr = np.asarray(train_indices, dtype=np.int32)
    rng.shuffle(train_indices_arr)

    x_train = x[train_indices_arr]
    y_train = y[train_indices_arr]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)

    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    while total < num_steps:
        order = rng.permutation(len(x_train))
        chunks_x.append(x_train[order])
        chunks_y.append(y_train[order])
        total += len(order)
    observations = np.concatenate(chunks_x, axis=0)[:num_steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:num_steps].astype(np.int32)
    targets = np.eye(10, dtype=np.float32)[labels]
    return jnp.asarray(observations), jnp.asarray(targets)


def mse_curve_from_mlp_metrics(metrics: jax.Array) -> jax.Array:
    """Convert per-head MLP metrics to per-step mean MSE."""
    return jnp.nanmean(metrics[..., 0], axis=-1)


def make_mlp(n_heads: int, hidden_sizes: tuple[int, ...]) -> MultiHeadMLPLearner:
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        step_size=STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=SPARSITY,
        use_layer_norm=True,
    )


def make_upgd(n_heads: int) -> UPGDLearner:
    return UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=(64,),
        step_size=STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        perturbation_sigma=1e-3,
        perturbation_warmup_steps=100,
        perturbation_ramp_steps=400,
        sparsity=SPARSITY,
        use_layer_norm=True,
    )


def make_cbp(n_heads: int) -> CBPMultiHeadMLPLearner:
    return CBPMultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(64,),
        cbp_config=ContinualBackpropConfig(  # type: ignore[call-arg]
            decay_rate=0.99,
            replacement_rate=1e-4,
            maturity_threshold=100,
            enabled=True,
        ),
        step_size=STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=SPARSITY,
        use_layer_norm=True,
    )


def make_compositional(n_heads: int) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner(
        n_features=16,
        n_tasks=n_heads,
        candidate_count=16,
        max_depth=4,
        use_obgd=True,
        obgd_kappa=OBGD_KAPPA,
    )


def make_deep_lifecycle(
    n_heads: int,
    variant: str = "deep_feature_lifecycle",
) -> DeepFeatureGeneratingMultiHeadMLPLearner:
    lifecycle_kwargs: dict[str, Any] = {
        "candidate_count": 4,
        "candidate_step_size": 0.03,
        "candidate_utility_decay": 0.99,
        "active_utility_decay": 0.99,
        "promotion_interval": 100,
        "min_unit_age": 100,
        "candidate_min_age": 50,
        "promotion_ratio": 1.05,
        "refresh_on_failed_promotion": True,
    }
    use_layer_norm = True
    hidden_sizes: tuple[int, ...] = (64, 64)
    if variant == "deep_lr_low":
        lifecycle_kwargs.update(candidate_step_size=0.01)
    elif variant == "deep_lr_high":
        lifecycle_kwargs.update(candidate_step_size=0.10)
    elif variant == "deep_imprint":
        lifecycle_kwargs.update(candidate_weight_step_size=0.003)
    elif variant == "deep_imprint_orthogonal":
        lifecycle_kwargs.update(
            candidate_weight_step_size=0.003,
            candidate_init="orthogonalized",
        )
    elif variant == "deep_protected":
        lifecycle_kwargs.update(replacement_utility_quantile=0.35, min_unit_age=150)
    elif variant == "deep_first":
        lifecycle_kwargs.update(promotion_layer_mode="first", promotion_ratio=1.0)
    elif variant == "deep_final":
        lifecycle_kwargs.update(promotion_layer_mode="final", promotion_ratio=1.0)
    elif variant == "deep_utility_norm":
        lifecycle_kwargs.update(
            promotion_utility_mode="mean_normalized",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
        )
    elif variant == "deep_no_layernorm":
        use_layer_norm = False
    elif variant == "deep_warmup":
        lifecycle_kwargs.update(
            replacement_warmup_steps=250,
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
        )
    elif variant == "deep_bank8":
        lifecycle_kwargs.update(candidate_count=8, promotion_ratio=1.0)
    elif variant == "deep_fast_cadence":
        lifecycle_kwargs.update(promotion_interval=50, promotion_ratio=1.0)
    elif variant == "deep_upgd_hybrid":
        lifecycle_kwargs.update(
            candidate_weight_step_size=0.001,
            candidate_perturbation_std=1e-3,
            candidate_perturbation_utility_scaled=True,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
            promotion_ratio=1.0,
        )
    elif variant == "deep_preserve_outgoing":
        lifecycle_kwargs.update(
            early_promotion_outgoing_mode="preserve",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
        )
    elif variant == "deep_preserve_no_layernorm":
        lifecycle_kwargs.update(
            early_promotion_outgoing_mode="preserve",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
        )
        use_layer_norm = False
    elif variant == "deep_active_perturb":
        lifecycle_kwargs.update(
            active_perturbation_std=1e-3,
            active_perturbation_warmup_steps=100,
            active_perturbation_ramp_steps=400,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
            promotion_ratio=1.0,
        )
    elif variant == "deep_active_perturb_low":
        lifecycle_kwargs.update(
            active_perturbation_std=1e-4,
            active_perturbation_warmup_steps=100,
            active_perturbation_ramp_steps=400,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
            promotion_ratio=1.0,
        )
    elif variant == "deep_active_perturb_preserve":
        lifecycle_kwargs.update(
            early_promotion_outgoing_mode="preserve",
            active_perturbation_std=1e-3,
            active_perturbation_warmup_steps=100,
            active_perturbation_ramp_steps=400,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
            promotion_ratio=1.0,
        )
    elif variant == "deep_soft_gate_final":
        lifecycle_kwargs.update(
            soft_gated_candidates=True,
            candidate_step_size=0.0,
            candidate_gate_init=0.0,
            candidate_gate_step_size=0.03,
            candidate_gate_max_abs=0.20,
            candidate_weight_step_size=0.003,
            promotion_layer_mode="final",
            soft_gate_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_soft_gate_fast":
        lifecycle_kwargs.update(
            soft_gated_candidates=True,
            candidate_step_size=0.0,
            candidate_gate_init=0.0,
            candidate_gate_step_size=0.05,
            candidate_gate_max_abs=0.25,
            candidate_weight_step_size=0.003,
            promotion_interval=50,
            candidate_min_age=25,
            promotion_layer_mode="final",
            soft_gate_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_soft_gate_l1":
        lifecycle_kwargs.update(
            soft_gated_candidates=True,
            candidate_step_size=0.0,
            candidate_gate_init=0.0,
            candidate_gate_step_size=0.05,
            candidate_gate_l1=1e-4,
            candidate_gate_max_abs=0.20,
            candidate_weight_step_size=0.003,
            promotion_layer_mode="final",
            soft_gate_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_soft_gate_bank8":
        lifecycle_kwargs.update(
            soft_gated_candidates=True,
            candidate_count=8,
            candidate_step_size=0.0,
            candidate_gate_init=0.0,
            candidate_gate_step_size=0.03,
            candidate_gate_max_abs=0.20,
            candidate_weight_step_size=0.003,
            promotion_layer_mode="final",
            soft_gate_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_net2net":
        lifecycle_kwargs.update(
            candidate_init="active_perturbation",
            active_candidate_perturbation_std=0.01,
            candidate_weight_step_size=0.003,
            function_preserving_promotion=True,
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_net2net_guarded":
        lifecycle_kwargs.update(
            candidate_init="active_perturbation",
            active_candidate_perturbation_std=0.01,
            candidate_weight_step_size=0.003,
            function_preserving_promotion=True,
            promotion_output_change_threshold=0.05,
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_net2net_final":
        lifecycle_kwargs.update(
            candidate_init="active_perturbation",
            active_candidate_perturbation_std=0.01,
            candidate_weight_step_size=0.003,
            function_preserving_promotion=True,
            promotion_output_change_threshold=0.05,
            promotion_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_net2net_fast":
        lifecycle_kwargs.update(
            candidate_init="active_perturbation",
            active_candidate_perturbation_std=0.01,
            candidate_weight_step_size=0.003,
            function_preserving_promotion=True,
            promotion_output_change_threshold=0.10,
            promotion_interval=50,
            candidate_min_age=25,
            min_unit_age=50,
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
            layer_promotion_budget=1,
        )
    elif variant == "deep_shallow":
        hidden_sizes = (64,)
        lifecycle_kwargs.update(
            promotion_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.5,
        )
    elif variant == "deep_shallow_final":
        hidden_sizes = (64,)
        lifecycle_kwargs.update(
            candidate_step_size=0.01,
            promotion_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.35,
            layer_promotion_budget=1,
        )
    elif variant == "deep_shallow_nlms":
        hidden_sizes = (64,)
        lifecycle_kwargs.update(
            candidate_step_size=0.03,
            candidate_weight_step_size=0.003,
            candidate_normalized_updates=True,
            promotion_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.35,
            layer_promotion_budget=1,
        )
    elif variant == "deep_shallow_soft_gate":
        hidden_sizes = (64,)
        lifecycle_kwargs.update(
            soft_gated_candidates=True,
            candidate_count=8,
            candidate_step_size=0.0,
            candidate_gate_init=0.0,
            candidate_gate_step_size=0.03,
            candidate_gate_l1=1e-4,
            candidate_gate_max_abs=0.20,
            candidate_weight_step_size=0.003,
            candidate_normalized_updates=True,
            promotion_layer_mode="final",
            soft_gate_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.35,
            layer_promotion_budget=1,
        )
    elif variant == "deep_shallow_net2net":
        hidden_sizes = (64,)
        lifecycle_kwargs.update(
            candidate_init="active_perturbation",
            active_candidate_perturbation_std=0.01,
            candidate_weight_step_size=0.003,
            candidate_normalized_updates=True,
            function_preserving_promotion=True,
            promotion_output_change_threshold=0.05,
            promotion_layer_mode="final",
            promotion_ratio=1.0,
            replacement_utility_quantile=0.35,
            layer_promotion_budget=1,
        )
    elif variant != "deep_feature_lifecycle":
        raise ValueError(f"unknown deep lifecycle variant: {variant}")

    return DeepFeatureGeneratingMultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        lifecycle_config=DeepFeatureLifecycleConfig(**lifecycle_kwargs),
        step_size=STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=SPARSITY,
        use_layer_norm=use_layer_norm,
    )


def run_single_method(
    method: str,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array | None]:
    """Run one method on one materialized stream.

    Returns the loss curve and, for deep lifecycle methods, the number of
    promotions made by the run.
    """
    feature_dim = int(observations.shape[1])
    n_heads = int(targets.shape[1])
    if method == "mlp_64":
        mlp = make_mlp(n_heads, (64,))
        mlp_state = mlp.init(feature_dim, key)
        return (
            mse_curve_from_mlp_metrics(
                run_multi_head_learning_loop(
                    mlp, mlp_state, observations, targets
                ).per_head_metrics
            ),
            None,
        )
    if method == "mlp_64_64":
        mlp = make_mlp(n_heads, (64, 64))
        mlp_state = mlp.init(feature_dim, key)
        return (
            mse_curve_from_mlp_metrics(
                run_multi_head_learning_loop(
                    mlp, mlp_state, observations, targets
                ).per_head_metrics
            ),
            None,
        )
    if method == "upgd":
        upgd = make_upgd(n_heads)
        upgd_state = upgd.init(feature_dim, key)
        return (
            run_upgd_arrays(upgd, upgd_state, observations, targets).metrics[:, 0],
            None,
        )
    if method == "cbp":
        cbp = make_cbp(n_heads)
        cbp_state = cbp.init(feature_dim, key)
        return (
            mse_curve_from_mlp_metrics(
                run_cbp_learning_loop(cbp, cbp_state, observations, targets).per_head_metrics
            ),
            None,
        )
    if method == "compositional":
        compositional = make_compositional(n_heads)
        compositional_state = compositional.init(feature_dim, key)
        return (
            run_compositional_arrays(
                compositional, compositional_state, observations, targets
            ).metrics[:, 0],
            None,
        )
    if method in DEEP_METHOD_NAMES:
        deep = make_deep_lifecycle(n_heads, method)
        deep_state = deep.init(feature_dim, key)
        result = run_deep_feature_lifecycle_arrays(
            deep,
            deep_state,
            observations,
            targets,
        )
        return (
            mse_curve_from_mlp_metrics(result.per_head_metrics),
            jnp.sum(result.lifecycle_metrics[:, 0]),
        )
    raise ValueError(f"unknown method: {method}")


def count_array_params(tree: Any, include_int: bool = False) -> int:
    """Count scalar array entries in a learner state tree."""
    total = 0
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "shape") and (
            jnp.issubdtype(leaf.dtype, jnp.floating)
            or (include_int and jnp.issubdtype(leaf.dtype, jnp.integer))
        ):
            total += int(np.prod(leaf.shape))
    return total


def estimate_method_params(method: str, feature_dim: int, n_heads: int) -> dict[str, int]:
    """Estimate active and candidate parameter counts for one probe shape."""
    key = jr.key(999)
    if method == "mlp_64":
        mlp_state = make_mlp(n_heads, (64,)).init(feature_dim, key)
        active = count_array_params(mlp_state.trunk_params) + count_array_params(
            mlp_state.head_params
        )
        return {"active": active, "candidate": 0}
    if method == "mlp_64_64":
        mlp_state = make_mlp(n_heads, (64, 64)).init(feature_dim, key)
        active = count_array_params(mlp_state.trunk_params) + count_array_params(
            mlp_state.head_params
        )
        return {"active": active, "candidate": 0}
    if method == "upgd":
        upgd_state = make_upgd(n_heads).init(feature_dim, key)
        active = count_array_params(upgd_state.trunk_params) + count_array_params(
            upgd_state.head_params
        )
        return {"active": active, "candidate": 0}
    if method in DEEP_METHOD_NAMES:
        deep_state = make_deep_lifecycle(n_heads, method).init(feature_dim, key)
        candidate = (
            count_array_params(deep_state.candidate_weights)
            + count_array_params(deep_state.candidate_biases)
            + count_array_params(deep_state.candidate_output_weights)
            + count_array_params(deep_state.candidate_gates)
        )
        return {
            "active": count_array_params(deep_state.mlp_state.trunk_params)
            + count_array_params(deep_state.mlp_state.head_params),
            "candidate": candidate,
        }
    return {"active": 0, "candidate": 0}


def run_portfolio(
    observations: jax.Array,
    targets: jax.Array,
    keys: tuple[jax.Array, ...],
    hedge_eta: float,
    hedge_discount: float,
) -> jax.Array:
    """Run a causal Hedge portfolio over all non-portfolio experts."""
    feature_dim = int(observations.shape[1])
    n_heads = int(targets.shape[1])
    mlp64 = make_mlp(n_heads, (64,))
    mlp6464 = make_mlp(n_heads, (64, 64))
    upgd = make_upgd(n_heads)
    cbp = make_cbp(n_heads)
    comp = make_compositional(n_heads)
    deep = make_deep_lifecycle(n_heads)

    init_carry = (
        mlp64.init(feature_dim, keys[0]),
        mlp6464.init(feature_dim, keys[1]),
        upgd.init(feature_dim, keys[2]),
        cbp.init(feature_dim, keys[3]),
        comp.init(feature_dim, keys[4]),
        deep.init(feature_dim, keys[5]),
        jnp.zeros(6, dtype=jnp.float32),
    )
    eta = jnp.asarray(hedge_eta, dtype=jnp.float32)
    discount = jnp.asarray(hedge_discount, dtype=jnp.float32)

    def step_fn(
        carry: tuple[Any, ...], inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[Any, ...], jax.Array]:
        mlp64_s, mlp6464_s, upgd_s, cbp_s, comp_s, deep_s, log_w = carry
        obs, tgt = inputs
        preds = jnp.stack(
            [
                mlp64.predict(mlp64_s, obs),
                mlp6464.predict(mlp6464_s, obs),
                upgd.predict(upgd_s, obs),
                cbp.predict(cbp_s, obs),
                comp.predict(comp_s, obs),
                deep.predict(deep_s, obs),
            ],
            axis=0,
        )
        weights = jax.nn.softmax(log_w)
        mixture_pred = jnp.sum(weights[:, None] * preds, axis=0)
        active = ~jnp.isnan(tgt)
        safe_tgt = jnp.where(active, tgt, 0.0)
        active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        expert_losses = jnp.sum(
            jnp.where(active[None, :], (preds - safe_tgt[None, :]) ** 2, 0.0),
            axis=1,
        ) / active_count
        mixture_loss = (
            jnp.sum(jnp.where(active, (mixture_pred - safe_tgt) ** 2, 0.0))
            / active_count
        )
        new_log_w = discount * log_w - eta * expert_losses

        mlp64_r = mlp64.update(mlp64_s, obs, tgt)
        mlp6464_r = mlp6464.update(mlp6464_s, obs, tgt)
        upgd_r = upgd.update(upgd_s, obs, tgt)
        cbp_r = cbp.update(cbp_s, obs, tgt)
        comp_r = comp.update(comp_s, obs, tgt)
        deep_r = deep.update(deep_s, obs, tgt)
        return (
            mlp64_r.state,
            mlp6464_r.state,
            upgd_r.state,
            cbp_r.state,
            comp_r.state,
            deep_r.state,
            new_log_w,
        ), mixture_loss

    _, curve = jax.lax.scan(step_fn, init_carry, (observations, targets))
    return jnp.asarray(curve)


def stderr(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def run_stream(
    spec: dict[str, Any],
    seeds: int,
    num_steps: int,
    final_window: int,
    methods: list[str],
    hedge_eta: float,
    hedge_discount: float,
) -> dict[str, Any]:
    """Run all methods for one stream spec."""
    stream = spec["factory"]() if "factory" in spec else None
    curves: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    deep_methods = [m for m in methods if m in DEEP_METHOD_NAMES]
    promotion_counts: dict[str, list[float]] = {m: [] for m in deep_methods}
    print(f"\n=== {spec['name']} seeds={seeds} steps={num_steps} ===")
    parameter_counts = {
        method: estimate_method_params(method, spec["feature_dim"], spec["n_tasks"])
        for method in methods
    }
    for seed in range(seeds):
        root = jr.key(seed)
        split = jr.split(root, len(methods) + 2)
        if "array_factory" in spec:
            observations, targets = spec["array_factory"](seed, num_steps)
        else:
            observations, targets = collect_stream_arrays(stream, num_steps, split[0])
        observations.block_until_ready()
        targets.block_until_ready()
        for i, method in enumerate(methods):
            curve, promotions = run_single_method(
                method, observations, targets, split[i + 1]
            )
            curve.block_until_ready()
            curves[method].append(np.asarray(curve))
            if promotions is not None:
                promotions.block_until_ready()
                promotion_counts[method].append(float(promotions))

        msg = ", ".join(
            f"{m}={np.mean(curves[m][-1][-final_window:]):.4f}" for m in methods
        )
        print(f"  seed {seed}: {msg}")

    stacked = {m: np.stack(v) for m, v in curves.items()}
    aggregate: dict[str, Any] = {}
    for method, arr in stacked.items():
        final = np.mean(arr[:, -final_window:], axis=1)
        aggregate[method] = {
            "mean_final": float(np.mean(final)),
            "stderr_final": stderr(final),
            "mean_total": float(np.mean(arr)),
            "stderr_total": stderr(np.mean(arr, axis=1)),
        }

    best_mlp = min(["mlp_64", "mlp_64_64"], key=lambda m: aggregate[m]["mean_final"])
    best_mlp_final = np.mean(stacked[best_mlp][:, -final_window:], axis=1)
    paired: dict[str, Any] = {"_best_mlp": best_mlp}
    for method, arr in stacked.items():
        if method == best_mlp:
            continue
        method_final = np.mean(arr[:, -final_window:], axis=1)
        diff = best_mlp_final - method_final
        paired[method] = {
            "best_mlp_minus_method": float(np.mean(diff)),
            "stderr": stderr(diff),
            "wins": int(np.sum(diff > 0.0)),
            "n_seeds": int(seeds),
        }

    return {
        "stream": spec["name"],
        "aggregate": aggregate,
        "paired_vs_best_mlp": paired,
        "mean_deep_promotions": {
            method: float(np.mean(counts)) if counts else 0.0
            for method, counts in promotion_counts.items()
        },
        "parameter_counts": parameter_counts,
    }


def write_note(path: Path, results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Write a compact Markdown research note."""
    lines = [
        "# Step 2 Deep Feature Lifecycle",
        "",
        f"Seeds: {config['seeds']}. Steps: {config['num_steps']}. "
        f"Final window: {config['final_window']}.",
        "",
        "Positive paired differences mean the method beat the best fair MLP.",
        "",
    ]
    for result in results:
        lines.append(f"## `{result['stream']}`")
        lines.append("")
        lines.append("| Method | Final-window MSE | Total MSE |")
        lines.append("|---|---:|---:|")
        aggregate = result["aggregate"]
        for method in sorted(aggregate, key=lambda m: aggregate[m]["mean_final"]):
            stats = aggregate[method]
            lines.append(
                f"| `{method}` | {stats['mean_final']:.4f} +/- "
                f"{stats['stderr_final']:.4f} | {stats['mean_total']:.4f} +/- "
                f"{stats['stderr_total']:.4f} |"
            )
        lines.append("")
        paired = result["paired_vs_best_mlp"]
        lines.append(f"Best fair MLP: `{paired['_best_mlp']}`.")
        lines.append("")
        lines.append("| Method | best_mlp - method | Wins |")
        lines.append("|---|---:|---:|")
        for method, stats in paired.items():
            if method == "_best_mlp":
                continue
            lines.append(
                f"| `{method}` | {stats['best_mlp_minus_method']:+.4f} +/- "
                f"{stats['stderr']:.4f} | {stats['wins']}/{stats['n_seeds']} |"
            )
        lines.append("")
        promotion_summary = ", ".join(
            f"`{method}`={count:.2f}"
            for method, count in result["mean_deep_promotions"].items()
        )
        lines.append(f"Mean deep-feature promotions per run: {promotion_summary}.")
        lines.append("")
        lines.append("| Method | Active params | Candidate params | Temporal uniformity |")
        lines.append("|---|---:|---:|---|")
        for method in sorted(aggregate, key=lambda m: aggregate[m]["mean_final"]):
            counts = result["parameter_counts"].get(
                method, {"active": 0, "candidate": 0}
            )
            temporal = (
                "active + candidates update every step"
                if method in DEEP_METHOD_NAMES
                else "active learner updates every step"
            )
            lines.append(
                f"| `{method}` | {counts['active']} | {counts['candidate']} | "
                f"{temporal} |"
            )
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    deep_methods = [
        method
        for method in DEEP_METHOD_NAMES
        if all(method in result["paired_vs_best_mlp"] for result in results)
    ]
    deep_wins_by_method = {
        method: sum(
            1
            for result in results
            if result["paired_vs_best_mlp"][method]["best_mlp_minus_method"] > 0.0
        )
        for method in deep_methods
    }
    if deep_wins_by_method:
        best_deep_method = max(
            deep_wins_by_method,
            key=lambda method: deep_wins_by_method[method],
        )
        lines.append(
            f"The best native deep feature lifecycle variant was `{best_deep_method}`, "
            f"which beat the best fair MLP on "
            f"{deep_wins_by_method[best_deep_method]}/{len(results)} streams. "
            "A single general deep feature-construction algorithm should be "
            "treated as a partial or negative Step 2 result unless a native "
            "variant wins robustly across the full matrix."
        )
    else:
        lines.append("No native deep feature lifecycle variants were included.")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--final-window", type=int, default=FINAL_WINDOW_DEFAULT)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--hedge-discount", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_NAMES,
        help="Subset of methods to run. Must include mlp_64 and mlp_64_64.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=None,
        help="Optional subset of stream names to run.",
    )
    parser.add_argument("--skip-digits", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.num_steps = 300
        args.final_window = 100
    unknown_methods = sorted(set(args.methods) - set(METHOD_NAMES))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    if "mlp_64" not in args.methods or "mlp_64_64" not in args.methods:
        raise ValueError("--methods must include mlp_64 and mlp_64_64")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.note_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    specs = list(STREAM_SPECS)
    if not args.skip_digits:
        specs.append(
            {
                "name": "digits_iid",
                "array_factory": collect_digits_arrays,
                "feature_dim": 64,
                "n_tasks": 10,
            }
        )
    if args.streams is not None:
        wanted_streams = set(args.streams)
        known_streams = {spec["name"] for spec in specs}
        unknown_streams = sorted(wanted_streams - known_streams)
        if unknown_streams:
            raise ValueError(f"unknown streams: {unknown_streams}")
        specs = [spec for spec in specs if spec["name"] in wanted_streams]

    results = [
        run_stream(
            spec,
            seeds=args.seeds,
            num_steps=args.num_steps,
            final_window=args.final_window,
            methods=list(args.methods),
            hedge_eta=args.hedge_eta,
            hedge_discount=args.hedge_discount,
        )
        for spec in specs
    ]
    config = {
        "seeds": args.seeds,
        "num_steps": args.num_steps,
        "final_window": args.final_window,
        "hedge_eta": args.hedge_eta,
        "hedge_discount": args.hedge_discount,
        "skip_digits": args.skip_digits,
        "wall_clock_s": time.time() - start,
        "methods": list(args.methods),
    }
    output_path = args.output_dir / "deep_feature_lifecycle_results.json"
    output_path.write_text(json.dumps({"config": config, "results": results}, indent=2))
    write_note(args.note_path, results, config)
    print(f"\nwrote {output_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

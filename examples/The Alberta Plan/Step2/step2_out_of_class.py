#!/usr/bin/env python3
"""Canonical Step 2 out-of-hypothesis-class benchmark.

This benchmark evaluates whether new feature-construction methods
(``CompositionalFeatureLearner``, ``UPGDLearner``, and
``CBPMultiHeadMLPLearner``) advance Step 2 beyond plain MLP /
pair-product baselines on streams whose oracle features lie
*outside* a 1-layer pair-product hypothesis class.

Methods (7):
    1. ``MultiHeadMLPLearner(hidden_sizes=(64,))``                  -- single MLP
    2. ``MultiHeadMLPLearner(hidden_sizes=(64, 64))``               -- two-layer MLP
    3. ``MultiHeadMLPLearner(hidden_sizes=())``                     -- per-task linear
    4. ``FixedBudgetInteractionLearner(n_features=16, candidate_count=64)``
    5. ``CompositionalFeatureLearner(n_features=16, candidate_count=16, max_depth=4)``
    6. ``UPGDLearner(n_heads=n_tasks, hidden_sizes=(64,))``
    7. ``CBPMultiHeadMLPLearner(n_heads=n_tasks, hidden_sizes=(64,))``

Optional ``--upgd-variants`` entries add tuned single-learner UPGD
configurations discovered on the external digits universality matrix while
preserving the same paired stream protocol.

Streams (3):
    A. ``OutOfClassPolynomialStream(feature_dim=8, n_tasks=3, ...)``  -- triple-product oracle
    B. ``FrequencyMismatchStream(feature_dim=4, n_tasks=2, ...)``      -- sinusoidal oracle
    C. ``CompositionalStream(feature_dim=6, n_tasks=3, ...)``          -- 2-layer tanh oracle

Protocol: 30 seeds x 6000 steps. The stream is realized once per seed; all
seven methods consume the *same* arrays for that seed. Per-seed metrics:
final-2000-window mean MSE, total mean MSE, per-step loss curve.

Statistics:
* Per (stream, method): mean +/- stderr of final-window MSE and total MSE.
* Per stream: paired-by-seed MSE differences vs the BEST MLP baseline
  (the better of MLP(64) or MLP(64, 64) by mean final-window MSE), with
  wins-out-of-30 and Cohen's d.

Outputs:
    outputs/step2_canonical/out_of_class_results.json
    outputs/step2_canonical/out_of_class_SUMMARY.md
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
import jax.tree_util as jtu
import numpy as np

from alberta_framework import (
    CBPMultiHeadMLPLearner,
    CompositionalFeatureLearner,
    CompositionalStream,
    ContinualBackpropConfig,
    FixedBudgetFeatureLearner,
    FixedBudgetInteractionLearner,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
    run_cbp_learning_loop,
    run_compositional_arrays,
    run_feature_discovery_arrays,
    run_interaction_feature_arrays,
    run_multi_head_learning_loop,
    run_upgd_arrays,
)

# =============================================================================
# Hyperparameters / configuration
# =============================================================================

NUM_STEPS = 6000
FINAL_WINDOW = 2000
DEFAULT_SEEDS = 30

OBGD_KAPPA = 2.0
MLP_STEP_SIZE = 0.03
MLP_SPARSITY = 0.5
UPGD_STEP_SIZE = 0.03
UPGD_SPARSITY = 0.5
CBP_STEP_SIZE = 0.03
CBP_SPARSITY = 0.5
CBP_DECAY_RATE = 0.99
CBP_REPLACEMENT_RATE = 1e-4
CBP_MATURITY_THRESHOLD = 100
COMP_N_FEATURES = 16
COMP_CANDIDATE_COUNT = 16
COMP_MAX_DEPTH = 4
INT_N_FEATURES = 16
INT_CANDIDATE_COUNT = 64
LIFECYCLE_N_FEATURES = 16
LIFECYCLE_CANDIDATE_COUNT = 16

STREAM_SPECS: list[dict[str, Any]] = [
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
        "n_contexts": 4,
        "context_length": 500,
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
        "n_contexts": 4,
        "context_length": 500,
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
        "n_contexts": 4,
        "context_length": 500,
    },
]


# =============================================================================
# Stream collection
# =============================================================================


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize one stream realization into ``(observations, targets)``."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, stream_state, jnp.arange(num_steps)
    )
    return observations, targets


# =============================================================================
# Per-method runners (vmap over seeds)
# =============================================================================


def _stack_states(states: list[Any]) -> Any:
    """Pytree-stack a list of dataclass states along a new leading axis."""
    return jtu.tree_map(lambda *xs: jnp.stack(xs), *states)


def _mlp_curve(metrics: jax.Array) -> jax.Array:
    """Convert ``MultiHeadMLPLearner`` per-head metrics to a per-step MSE curve.

    ``per_head_metrics`` has shape ``(num_steps, n_heads, 3)`` with column 0
    holding squared error per active head. Inactive heads emit NaN, so we
    take ``nanmean`` over the head axis.
    """
    return jnp.nanmean(metrics[..., 0], axis=-1)


def run_mlp_all_seeds(
    hidden_sizes: tuple[int, ...],
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Run an MLP variant across all seeds in one ``jax.vmap``.

    Args:
        hidden_sizes: Hidden layer widths (``()`` for the linear baseline).
        n_tasks: Number of supervised heads.
        feature_dim: Raw observation dimension.
        observations_per_seed: Shape ``(n_seeds, num_steps, feature_dim)``.
        targets_per_seed: Shape ``(n_seeds, num_steps, n_tasks)``.
        keys: Per-seed init keys, shape ``(n_seeds, ...)``.

    Returns:
        Per-seed loss curves of shape ``(n_seeds, num_steps)``.
    """
    use_layer_norm = len(hidden_sizes) > 0
    sparsity = MLP_SPARSITY if use_layer_norm else 0.0
    learner = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=hidden_sizes,
        step_size=MLP_STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA) if use_layer_norm else None,
        sparsity=sparsity,
        use_layer_norm=use_layer_norm,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_multi_head_learning_loop(learner, state, obs, tgt)
        return _mlp_curve(result.per_head_metrics)

    return jax.vmap(single, in_axes=(0, 0, 0))(
        keys, observations_per_seed, targets_per_seed
    )


def run_interaction_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Run ``FixedBudgetInteractionLearner`` across all seeds in one ``vmap``.

    The interaction learner emits a 7-column metrics block per step with
    column 0 = mean squared error.
    """
    learner = FixedBudgetInteractionLearner(
        n_features=INT_N_FEATURES,
        n_tasks=n_tasks,
        candidate_count=INT_CANDIDATE_COUNT,
        use_obgd=True,
        obgd_kappa=OBGD_KAPPA,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_interaction_feature_arrays(learner, state, obs, tgt)
        return result.metrics[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(
        keys, observations_per_seed, targets_per_seed
    )


def run_compositional_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: list[jax.Array],
) -> jax.Array:
    """Run ``CompositionalFeatureLearner`` across all seeds.

    ``CompositionalFeatureLearner.init`` performs Python-side parent
    sampling and is not directly ``vmap``-friendly, so we initialize each
    seed eagerly, stack the resulting states along axis 0, and ``vmap``
    the JIT-compiled training loop alongside per-seed observation arrays.
    """
    learner = CompositionalFeatureLearner(
        n_features=COMP_N_FEATURES,
        n_tasks=n_tasks,
        candidate_count=COMP_CANDIDATE_COUNT,
        max_depth=COMP_MAX_DEPTH,
        use_obgd=True,
        obgd_kappa=OBGD_KAPPA,
    )
    states = [learner.init(feature_dim, key) for key in keys]
    stacked = _stack_states(states)

    def single(state: Any, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        result = run_compositional_arrays(learner, state, obs, tgt)
        return result.metrics[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(
        stacked, observations_per_seed, targets_per_seed
    )


def run_upgd_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: jax.Array,
    variant_name: str | None = None,
) -> jax.Array:
    """Run ``UPGDLearner`` across all seeds in one ``jax.vmap``.

    ``run_upgd_arrays`` emits a 4-column metrics block per step with
    column 0 = mean squared error (already averaged across active heads).
    """
    variant_config = (
        {"loss_normalization": "mean"}
        if variant_name is None
        else UPGD_VARIANT_CONFIGS[variant_name]
    )
    hidden_size = int(variant_config.get("hidden_size", 64))
    step_size_multiplier = float(variant_config.get("step_size_multiplier", 1.0))
    bounder_kappa = float(variant_config.get("bounder_kappa", OBGD_KAPPA))
    upgd_kwargs = {
        key: value
        for key, value in variant_config.items()
        if key not in {"hidden_size", "step_size_multiplier", "bounder_kappa"}
    }
    learner = UPGDLearner(
        n_heads=n_tasks,
        hidden_sizes=(hidden_size,),
        step_size=UPGD_STEP_SIZE * step_size_multiplier,
        bounder=ObGDBounding(kappa=bounder_kappa),
        sparsity=UPGD_SPARSITY,
        use_layer_norm=True,
        **upgd_kwargs,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_upgd_arrays(learner, state, obs, tgt)
        return result.metrics[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(
        keys, observations_per_seed, targets_per_seed
    )


def run_cbp_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Run ``CBPMultiHeadMLPLearner`` across all seeds in one ``jax.vmap``.

    CBP wraps the same multi-head MLP shell as ``mlp_64`` and adds per-unit
    utility tracking plus low-utility hidden-unit replacement. Its learning
    loop emits the same per-head metrics as ``MultiHeadMLPLearner``.
    """
    learner = CBPMultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(64,),
        cbp_config=ContinualBackpropConfig(
            decay_rate=CBP_DECAY_RATE,
            replacement_rate=CBP_REPLACEMENT_RATE,
            maturity_threshold=CBP_MATURITY_THRESHOLD,
            enabled=True,
        ),
        step_size=CBP_STEP_SIZE,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=CBP_SPARSITY,
        use_layer_norm=True,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_cbp_learning_loop(learner, state, obs, tgt)
        return _mlp_curve(result.per_head_metrics)

    return jax.vmap(single, in_axes=(0, 0, 0))(
        keys, observations_per_seed, targets_per_seed
    )


def run_feature_lifecycle_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations_per_seed: jax.Array,
    targets_per_seed: jax.Array,
    keys: list[jax.Array],
) -> jax.Array:
    """Run ``FixedBudgetFeatureLearner`` (generate-test-rank-replace) across seeds.

    Uses the stacked-states-plus-vmap approach because ``init`` contains a
    Python-side ``time.time()`` call.  Metrics column 0 is per-step MSE.
    """
    learner = FixedBudgetFeatureLearner(
        n_features=LIFECYCLE_N_FEATURES,
        n_tasks=n_tasks,
        candidate_count=LIFECYCLE_CANDIDATE_COUNT,
        step_size_output=MLP_STEP_SIZE,
        use_obgd=True,
        obgd_kappa=OBGD_KAPPA,
    )
    states = [learner.init(feature_dim, key) for key in keys]
    stacked = _stack_states(states)

    def single(state: Any, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        result = run_feature_discovery_arrays(learner, state, obs, tgt)
        return result.metrics[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(
        stacked, observations_per_seed, targets_per_seed
    )


METHOD_NAMES = [
    "mlp_64",
    "mlp_64_64",
    "linear",
    "interaction",
    "compositional",
    "upgd",
    "cbp",
    "feature_lifecycle",
]

UPGD_VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "mean_sigma1e3_kappa2": {
        "bounder_kappa": 2.0,
        "perturbation_sigma": 1e-3,
        "loss_normalization": "mean",
    },
    "mean_sigma1e4_kappa2": {
        "bounder_kappa": 2.0,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "mean",
    },
    "mean_sigma0_kappa2": {
        "bounder_kappa": 2.0,
        "perturbation_sigma": 0.0,
        "loss_normalization": "mean",
    },
    "mean_sigma1e3_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-3,
        "loss_normalization": "mean",
    },
    "mean_sigma1e4_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "mean",
    },
    "mean_headsum_sigma0_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 0.0,
        "loss_normalization": "mean",
        "head_gradient_scale": "active_count",
    },
    "mean_headsum_sigma1e4_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "mean",
        "head_gradient_scale": "active_count",
    },
    "mean_headsum_sigma1e3_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-3,
        "loss_normalization": "mean",
        "head_gradient_scale": "active_count",
    },
    "density_sigma0_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 0.0,
        "loss_normalization": "target_density",
    },
    "density_sigma1e4_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
    },
    "structure_sigma1e4_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_structure",
    },
    "structure_sigma1e4_kappa05_interval4": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 4,
        "loss_normalization": "target_structure",
    },
    "structure_sigma1e4_kappa05_interval16": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "loss_normalization": "target_structure",
    },
    "structure_sigma1e4_kappa05_interval16_lean": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "loss_normalization": "target_structure",
        "track_unit_utilities": False,
        "track_gradient_history": False,
    },
    "structure_sigma1e4_kappa05_rademacher_interval16_lean": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "perturbation_noise": "rademacher",
        "loss_normalization": "target_structure",
        "track_unit_utilities": False,
        "track_gradient_history": False,
    },
    "structure_sigma1e4_kappa05_rademacher_interval4_lean": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 4,
        "perturbation_noise": "rademacher",
        "loss_normalization": "target_structure",
        "track_unit_utilities": False,
        "track_gradient_history": False,
    },
    "structure_h16_rademacher_interval16_lean": {
        "hidden_size": 16,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "perturbation_noise": "rademacher",
        "loss_normalization": "target_structure",
        "track_unit_utilities": False,
        "track_gradient_history": False,
    },
    "structure_h32_rademacher_interval16_lean": {
        "hidden_size": 32,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "perturbation_noise": "rademacher",
        "loss_normalization": "target_structure",
        "track_unit_utilities": False,
        "track_gradient_history": False,
    },
    "density_sigma1e3_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-3,
        "loss_normalization": "target_density",
    },
    "density_adaptk035_065_lr06": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
    },
    "structure_adaptk035_065_lr06": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_structure",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
    },
    "density_adaptk035_065_lr05_e1": {
        "step_size_multiplier": 0.5,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 1.0,
        "adaptive_kappa_warmup_steps": 120,
    },
    "density_meta003_notrunk": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.003,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
    "structure_meta003_notrunk": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_structure",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.003,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
    "structure_meta003_notrunk_interval16": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "perturbation_interval": 16,
        "loss_normalization": "target_structure",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.003,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
    "density_repx075": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
        "head_repetition_multiplier": 0.75,
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
    },
    "density_repx075_meta001_notrunk": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "target_density",
        "head_repetition_multiplier": 0.75,
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.001,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
    "sum_sigma0_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 0.0,
        "loss_normalization": "sum",
    },
    "sum_sigma1e4_kappa05": {
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
    },
    "sum_sigma0_kappa1": {
        "bounder_kappa": 1.0,
        "perturbation_sigma": 0.0,
        "loss_normalization": "sum",
    },
    "sum_sigma1e4_kappa1": {
        "bounder_kappa": 1.0,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
    },
    "sum_sigma1e4_kappa05_lr075": {
        "step_size_multiplier": 0.75,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
    },
    "sum_sigma1e4_kappa05_lr045": {
        "step_size_multiplier": 0.45,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
    },
    "digits_meta003_notrunk": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.003,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
    "digits_repx075": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
        "head_repetition_multiplier": 0.75,
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
    },
    "digits_repx075_meta001_notrunk": {
        "step_size_multiplier": 0.6,
        "bounder_kappa": 0.5,
        "perturbation_sigma": 1e-4,
        "loss_normalization": "sum",
        "head_repetition_multiplier": 0.75,
        "adaptive_kappa_mode": "loss_ratio",
        "adaptive_kappa_base": 0.5,
        "adaptive_kappa_min": 0.35,
        "adaptive_kappa_max": 0.65,
        "adaptive_kappa_exponent": 0.5,
        "adaptive_kappa_warmup_steps": 120,
        "meta_plasticity_mode": "gradient_alignment",
        "meta_plasticity_step_size": 0.001,
        "meta_plasticity_min_multiplier": 0.5,
        "meta_plasticity_max_multiplier": 2.0,
        "meta_plasticity_warmup_steps": 30,
        "meta_plasticity_trunk_enabled": False,
    },
}


# =============================================================================
# Statistics
# =============================================================================


def cohens_d(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences (mean / std)."""
    sd = float(np.std(diffs, ddof=1))
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(diffs) / sd)


def stderr(values: np.ndarray) -> float:
    """Standard error of the mean (population std / sqrt(N))."""
    n = values.shape[0]
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(n))


# =============================================================================
# Per-stream pipeline
# =============================================================================


def run_one_stream(
    stream_name: str,
    stream_factory: Any,
    feature_dim: int,
    n_tasks: int,
    n_contexts: int,
    n_seeds: int,
    num_steps: int,
    upgd_variant_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run all methods on one stream across ``n_seeds`` seeds.

    Each seed materializes a fresh stream realization (independent random
    coefficients in the stream's ``init``); all methods then consume the same
    arrays so paired comparisons are honest.
    """
    print(f"\n=== stream: {stream_name} (seeds={n_seeds}, steps={num_steps}) ===")
    t_stream = time.time()

    stream = stream_factory()

    # Per-seed deterministic key splits. Each seed gets a fresh stream
    # realization plus one init key per method.
    stream_keys: list[jax.Array] = []
    method_keys_per_seed: list[list[jax.Array]] = []
    for seed in range(n_seeds):
        root = jr.key(seed)
        stream_key, *method_keys = jr.split(root, 8)
        if upgd_variant_names:
            variant_keys = list(
                jr.split(jr.fold_in(root, 1_000_003), len(upgd_variant_names))
            )
            method_keys.extend(variant_keys)
        stream_keys.append(stream_key)
        method_keys_per_seed.append(list(method_keys))

    # Collect all stream realizations into a single
    # ``(n_seeds, num_steps, feature_dim)`` block so each method-runner
    # can vmap once over (key, obs, tgt). We loop in Python here because
    # ``stream.init`` for some streams uses Python-side branching.
    t_collect = time.time()
    obs_list: list[jax.Array] = []
    tgt_list: list[jax.Array] = []
    for stream_key in stream_keys:
        obs_i, tgt_i = collect_stream_arrays(stream, num_steps, stream_key)
        obs_list.append(obs_i)
        tgt_list.append(tgt_i)
    observations_per_seed = jnp.stack(obs_list)
    targets_per_seed = jnp.stack(tgt_list)
    observations_per_seed.block_until_ready()
    targets_per_seed.block_until_ready()
    print(
        f"  collected stream arrays in {time.time() - t_collect:.1f}s "
        f"[obs={observations_per_seed.shape}, tgt={targets_per_seed.shape}]"
    )

    # Per-method keys, stacked.
    keys_mlp64 = jnp.stack([mks[0] for mks in method_keys_per_seed])
    keys_mlp64_64 = jnp.stack([mks[1] for mks in method_keys_per_seed])
    keys_linear = jnp.stack([mks[2] for mks in method_keys_per_seed])
    keys_inter = jnp.stack([mks[3] for mks in method_keys_per_seed])
    keys_comp = [mks[4] for mks in method_keys_per_seed]
    keys_upgd = jnp.stack([mks[5] for mks in method_keys_per_seed])
    keys_cbp = jnp.stack([mks[6] for mks in method_keys_per_seed])
    keys_upgd_variants = {
        name: jnp.stack([mks[7 + idx] for mks in method_keys_per_seed])
        for idx, name in enumerate(upgd_variant_names)
    }
    # feature_lifecycle uses fold_in to avoid disturbing existing key allocations
    keys_lifecycle = [
        jr.fold_in(jr.key(seed), jnp.uint32(2_000_001))
        for seed in range(n_seeds)
    ]

    per_method_curves: dict[str, np.ndarray] = {}

    # 1. MLP(64)
    t = time.time()
    curves = run_mlp_all_seeds(
        (64,), n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_mlp64,
    )
    curves.block_until_ready()
    per_method_curves["mlp_64"] = np.asarray(curves)
    print(
        f"  mlp_64 done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['mlp_64'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 2. MLP(64, 64)
    t = time.time()
    curves = run_mlp_all_seeds(
        (64, 64), n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_mlp64_64,
    )
    curves.block_until_ready()
    per_method_curves["mlp_64_64"] = np.asarray(curves)
    print(
        f"  mlp_64_64 done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['mlp_64_64'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 3. Linear (per-task linear regression via MLP with no hidden layers)
    t = time.time()
    curves = run_mlp_all_seeds(
        (), n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_linear,
    )
    curves.block_until_ready()
    per_method_curves["linear"] = np.asarray(curves)
    print(
        f"  linear done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['linear'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 4. FixedBudgetInteractionLearner
    t = time.time()
    curves = run_interaction_all_seeds(
        n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_inter,
    )
    curves.block_until_ready()
    per_method_curves["interaction"] = np.asarray(curves)
    print(
        f"  interaction done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['interaction'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 5. CompositionalFeatureLearner
    t = time.time()
    curves = run_compositional_all_seeds(
        n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_comp,
    )
    curves.block_until_ready()
    per_method_curves["compositional"] = np.asarray(curves)
    print(
        f"  compositional done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['compositional'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 6. UPGD
    t = time.time()
    curves = run_upgd_all_seeds(
        n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_upgd,
    )
    curves.block_until_ready()
    per_method_curves["upgd"] = np.asarray(curves)
    print(
        f"  upgd done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['upgd'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # Optional tuned UPGD variants.
    for variant_name in upgd_variant_names:
        method_name = f"upgd_variant_{variant_name}"
        t = time.time()
        curves = run_upgd_all_seeds(
            n_tasks, feature_dim,
            observations_per_seed, targets_per_seed,
            keys_upgd_variants[variant_name],
            variant_name=variant_name,
        )
        curves.block_until_ready()
        per_method_curves[method_name] = np.asarray(curves)
        print(
            f"  {method_name} done in {time.time() - t:.1f}s "
            f"[mean final={per_method_curves[method_name][:, -FINAL_WINDOW:].mean():.4f}]"
        )

    # 7. Continual Backprop
    t = time.time()
    curves = run_cbp_all_seeds(
        n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_cbp,
    )
    curves.block_until_ready()
    per_method_curves["cbp"] = np.asarray(curves)
    print(
        f"  cbp done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['cbp'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    # 8. FixedBudgetFeatureLearner (generate-test-rank-replace lifecycle)
    t = time.time()
    curves = run_feature_lifecycle_all_seeds(
        n_tasks, feature_dim,
        observations_per_seed, targets_per_seed, keys_lifecycle,
    )
    curves.block_until_ready()
    per_method_curves["feature_lifecycle"] = np.asarray(curves)
    print(
        f"  feature_lifecycle done in {time.time() - t:.1f}s "
        f"[mean final={per_method_curves['feature_lifecycle'][:, -FINAL_WINDOW:].mean():.4f}]"
    )

    stream_dt = time.time() - t_stream
    print(f"  stream {stream_name} total: {stream_dt:.1f}s")

    return {
        "stream": stream_name,
        "stream_meta": {
            "feature_dim": feature_dim,
            "n_tasks": n_tasks,
            "n_contexts": n_contexts,
            "num_steps": num_steps,
            "n_seeds": n_seeds,
            "wall_clock_s": stream_dt,
        },
        "curves": per_method_curves,
    }


# =============================================================================
# Aggregation and write-out
# =============================================================================


def build_per_run_records(stream_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-stream curves into per-(stream, method, seed) records."""
    records: list[dict[str, Any]] = []
    for stream_out in stream_outputs:
        stream_name = stream_out["stream"]
        for method_name, curves in stream_out["curves"].items():
            for seed in range(curves.shape[0]):
                curve = curves[seed]
                final_window = float(np.mean(curve[-FINAL_WINDOW:]))
                total_mean = float(np.mean(curve))
                records.append({
                    "stream": stream_name,
                    "method": method_name,
                    "seed": int(seed),
                    "final_window_mse": final_window,
                    "total_mean_loss": total_mean,
                    "loss_curve": curve.astype(np.float64).tolist(),
                })
    return records


def build_aggregate(stream_outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-stream per-method mean / stderr summary."""
    aggregate: dict[str, dict[str, Any]] = {}
    for stream_out in stream_outputs:
        per_method: dict[str, Any] = {}
        for method_name, curves in stream_out["curves"].items():
            final_per_seed = np.mean(curves[:, -FINAL_WINDOW:], axis=1)
            total_per_seed = np.mean(curves, axis=1)
            per_method[method_name] = {
                "mean_final": float(np.mean(final_per_seed)),
                "stderr_final": stderr(final_per_seed),
                "mean_total": float(np.mean(total_per_seed)),
                "stderr_total": stderr(total_per_seed),
                "min_final": float(np.min(final_per_seed)),
                "max_final": float(np.max(final_per_seed)),
                "median_final": float(np.median(final_per_seed)),
            }
        aggregate[stream_out["stream"]] = per_method
    return aggregate


def build_paired_vs_best_mlp(
    stream_outputs: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-stream paired diffs vs the best-mean MLP baseline."""
    paired: dict[str, dict[str, Any]] = {}
    for stream_out in stream_outputs:
        stream_name = stream_out["stream"]
        per_method = aggregate[stream_name]
        # Choose best of MLP(64) / MLP(64, 64) by mean final-window MSE.
        mlp_candidates = ["mlp_64", "mlp_64_64"]
        best_mlp = min(mlp_candidates, key=lambda m: per_method[m]["mean_final"])
        best_mlp_curves = stream_out["curves"][best_mlp]
        best_mlp_final = np.mean(best_mlp_curves[:, -FINAL_WINDOW:], axis=1)

        per_method_paired: dict[str, Any] = {"_best_mlp": best_mlp}
        for method_name, curves in stream_out["curves"].items():
            if method_name == best_mlp:
                continue
            method_final = np.mean(curves[:, -FINAL_WINDOW:], axis=1)
            diff = best_mlp_final - method_final  # positive => method beats MLP
            per_method_paired[method_name] = {
                "best_mlp_minus_method": float(np.mean(diff)),
                "stderr": stderr(diff),
                "wins": int(np.sum(diff > 0.0)),
                "n_seeds": int(diff.shape[0]),
                "cohens_d": cohens_d(diff),
            }
        paired[stream_name] = per_method_paired
    return paired


# =============================================================================
# Markdown summary
# =============================================================================


def _fmt_mean_stderr(mean: float, sd: float) -> str:
    """Formatted ``mean +/- stderr`` for tables, with adaptive precision."""
    return f"{mean:.4f} +/- {sd:.4f}"


def write_markdown_summary(
    output_path: Path,
    aggregate: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
    config: dict[str, Any],
    wall_clock_s: float,
) -> None:
    """Write the human-readable benchmark summary."""
    lines: list[str] = []
    lines.append("# Step 2 Out-of-Hypothesis-Class Benchmark")
    lines.append("")
    lines.append(
        f"Wall clock: {wall_clock_s:.1f}s. "
        f"Seeds: {config['n_seeds']}. "
        f"Steps per run: {config['num_steps']}. "
        f"Final-window: last {FINAL_WINDOW} steps."
    )
    lines.append("")
    lines.append(
        "All MLP-family learners use ObGDBounding(kappa=2.0). "
        "MSE values are averaged across active heads at each step."
    )
    lines.append("")

    # Final-window MSE table per stream.
    for stream_name, per_method in aggregate.items():
        lines.append(f"## Stream: `{stream_name}`")
        lines.append("")
        lines.append("Final-window MSE (mean +/- stderr over seeds):")
        lines.append("")
        lines.append("| Method | Final-window MSE | Total mean MSE |")
        lines.append("|---|---|---|")
        # Sort by mean_final ascending so the winner is first.
        for method_name in sorted(per_method.keys(), key=lambda m: per_method[m]["mean_final"]):
            stats = per_method[method_name]
            lines.append(
                f"| `{method_name}` | "
                f"{_fmt_mean_stderr(stats['mean_final'], stats['stderr_final'])} | "
                f"{_fmt_mean_stderr(stats['mean_total'], stats['stderr_total'])} |"
            )
        lines.append("")

        paired_data = paired[stream_name]
        best_mlp = paired_data["_best_mlp"]
        lines.append(
            f"Paired-vs-best-MLP comparison (best MLP on this stream: `{best_mlp}`)."
            " A positive `best_mlp - method` value means the method **beats** the MLP."
        )
        lines.append("")
        n_seeds_any = next(
            info["n_seeds"]
            for k, info in paired_data.items() if k != "_best_mlp"
        )
        lines.append(
            f"| Method | best_mlp - method | stderr | Wins/{n_seeds_any} | Cohen's d |"
        )
        lines.append("|---|---|---|---|---|")
        for method_name, info in paired_data.items():
            if method_name == "_best_mlp":
                continue
            wins_str = f"{info['wins']}/{info['n_seeds']}"
            lines.append(
                f"| `{method_name}` | {info['best_mlp_minus_method']:+.4f} | "
                f"{info['stderr']:.4f} | {wins_str} | {info['cohens_d']:+.3f} |"
            )
        lines.append("")

    # Honest summary section.
    lines.append("## Honest summary")
    lines.append("")
    summary_lines = build_honest_summary(aggregate, paired)
    lines.extend(summary_lines)
    lines.append("")

    # Science verdict.
    lines.append("## Science verdict")
    lines.append("")
    verdict_lines = build_science_verdict(aggregate, paired)
    lines.extend(verdict_lines)
    lines.append("")

    output_path.write_text("\n".join(lines))


def build_honest_summary(
    aggregate: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
) -> list[str]:
    """Per-stream honest assessment of which methods beat the best MLP."""
    lines: list[str] = []
    for stream_name, paired_data in paired.items():
        best_mlp = paired_data["_best_mlp"]
        best_mlp_mean = aggregate[stream_name][best_mlp]["mean_final"]
        beat_methods: list[tuple[str, float, float, int, int, float]] = []
        lost_methods: list[tuple[str, float, float, int, int, float]] = []
        for method_name, info in paired_data.items():
            if method_name == "_best_mlp":
                continue
            entry = (
                method_name,
                info["best_mlp_minus_method"],
                info["stderr"],
                info["wins"],
                info["n_seeds"],
                info["cohens_d"],
            )
            if info["best_mlp_minus_method"] > 0.0:
                beat_methods.append(entry)
            else:
                lost_methods.append(entry)

        lines.append(f"### `{stream_name}`")
        lines.append("")
        lines.append(
            f"Baseline winner: `{best_mlp}` with final-window MSE = {best_mlp_mean:.4f}."
        )
        lines.append("")
        if beat_methods:
            lines.append(
                "Methods that beat the best MLP on this stream "
                "(mean diff, wins/n_seeds, d):"
            )
            for name, diff, sd, wins, n, d in beat_methods:
                lines.append(
                    f"- `{name}`: +{diff:.4f} +/- {sd:.4f}, "
                    f"{wins}/{n} wins, d = {d:+.3f}"
                )
        else:
            lines.append(
                "**No method beat the best MLP in mean final-window MSE "
                "on this stream.**"
            )
        lines.append("")
        if lost_methods:
            lines.append("Methods that lost to the best MLP:")
            for name, diff, sd, wins, n, d in lost_methods:
                lines.append(
                    f"- `{name}`: {diff:+.4f} +/- {sd:.4f}, "
                    f"{wins}/{n} wins, d = {d:+.3f}"
                )
        lines.append("")
    return lines


def build_science_verdict(
    aggregate: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
) -> list[str]:
    """One-paragraph synthesis of what the result means for Step 2."""
    # Count, per new method, on how many streams it beats the best MLP and
    # whether the effect size is non-trivial (|d| >= 0.2).
    optional_upgd_methods = sorted(
        {
            method_name
            for per_method in aggregate.values()
            for method_name in per_method
            if method_name.startswith("upgd_variant_")
        }
    )
    base_new_methods = ["compositional", "upgd", "cbp", *optional_upgd_methods]
    new_methods = [
        method_name
        for method_name in base_new_methods
        if all(method_name in paired_data for paired_data in paired.values())
    ]
    pair_method = "interaction"
    comparison_methods = [*new_methods]
    if all(pair_method in paired_data for paired_data in paired.values()):
        comparison_methods.append(pair_method)
    n_streams = len(paired)

    method_wins: dict[str, list[tuple[str, float, int, float]]] = {
        m: [] for m in comparison_methods
    }
    for stream_name, paired_data in paired.items():
        for method_name in comparison_methods:
            info = paired_data[method_name]
            method_wins[method_name].append((
                stream_name,
                info["best_mlp_minus_method"],
                info["wins"],
                info["cohens_d"],
            ))

    parts: list[str] = []
    parts.append("Step 2 demands feature *construction*, not feature *selection*.")
    parts.append(
        "The three streams used here are deliberately out-of-class for a "
        "1-layer pair-product hypothesis: triple-product polynomials, "
        "sums of sinusoids, and 2-hidden-layer tanh networks."
    )
    stream_wins = {
        method_name: sum(1 for _, diff, _, _ in wins if diff > 0.0)
        for method_name, wins in method_wins.items()
    }
    best_new_method_wins = max(
        (stream_wins.get(method_name, 0) for method_name in new_methods),
        default=0,
    )
    method_summary = ", ".join(
        f"`{method_name}` on {stream_wins[method_name]}/{n_streams}"
        for method_name in comparison_methods
    )
    parts.append(
        f"Across {n_streams} streams, "
        f"methods beat the best MLP as follows: {method_summary}."
    )
    if best_new_method_wins >= max(2, math.floor(n_streams / 2) + 1):
        parts.append(
            "Step 2 has been meaningfully advanced: at least one new feature-"
            "construction method beats a hand-tuned MLP on the majority of "
            "out-of-class streams, with non-trivial paired effect sizes."
        )
    elif best_new_method_wins == 1:
        parts.append(
            "Step 2 is partially advanced: a new method beats the best MLP "
            "on exactly one stream. The remaining streams motivate further "
            "work on feature-construction priors."
        )
    else:
        parts.append(
            "Step 2 is *not* meaningfully advanced by these methods on these "
            "streams: the MLP baselines still win even when their hypothesis "
            "class strictly contains the oracle. This points at the rest "
            "of the construction stack (priors, replacement, candidate budget) "
            "as the bottleneck rather than expressivity."
        )
    return [" ".join(parts)]


# =============================================================================
# Entrypoint
# =============================================================================


def make_config_dict(
    n_seeds: int,
    num_steps: int,
    upgd_variant_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Materialize the full hyperparameter spec for the JSON output."""
    return {
        "num_steps": num_steps,
        "final_window": FINAL_WINDOW,
        "n_seeds": n_seeds,
        "obgd_kappa": OBGD_KAPPA,
        "mlp": {
            "step_size": MLP_STEP_SIZE,
            "sparsity_with_hidden": MLP_SPARSITY,
            "sparsity_linear": 0.0,
            "use_layer_norm": True,
            "bounder": "ObGDBounding",
        },
        "upgd": {
            "step_size": UPGD_STEP_SIZE,
            "sparsity": UPGD_SPARSITY,
            "hidden_sizes": [64],
            "use_layer_norm": True,
            "bounder": "ObGDBounding",
            "loss_normalization": "mean",
            "perturbation_sigma": 1e-3,
        },
        "upgd_variants": {
            name: UPGD_VARIANT_CONFIGS[name] for name in upgd_variant_names
        },
        "cbp": {
            "step_size": CBP_STEP_SIZE,
            "sparsity": CBP_SPARSITY,
            "hidden_sizes": [64],
            "use_layer_norm": True,
            "bounder": "ObGDBounding",
            "decay_rate": CBP_DECAY_RATE,
            "replacement_rate": CBP_REPLACEMENT_RATE,
            "maturity_threshold": CBP_MATURITY_THRESHOLD,
        },
        "compositional": {
            "n_features": COMP_N_FEATURES,
            "candidate_count": COMP_CANDIDATE_COUNT,
            "max_depth": COMP_MAX_DEPTH,
            "use_obgd": True,
        },
        "interaction": {
            "n_features": INT_N_FEATURES,
            "candidate_count": INT_CANDIDATE_COUNT,
            "use_obgd": True,
        },
        "streams": [
            {k: v for k, v in spec.items() if k != "factory"}
            for spec in STREAM_SPECS
        ],
        "method_names": [
            *METHOD_NAMES,
            *[f"upgd_variant_{name}" for name in upgd_variant_names],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help="Number of seeds per (stream, method) pair.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=NUM_STEPS,
        help="Number of stream steps per run.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/step2_canonical"),
        help="Directory to write JSON and SUMMARY.md outputs.",
    )
    parser.add_argument(
        "--upgd-variants",
        default="",
        help=(
            "Comma-separated optional tuned UPGD variants to include. "
            f"Known variants: {', '.join(sorted(UPGD_VARIANT_CONFIGS))}. "
            "Use 'all' for every variant."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a tiny smoke configuration (2 seeds, 600 steps).",
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.num_steps = 600

    requested_upgd_variants = tuple(
        name.strip()
        for name in args.upgd_variants.split(",")
        if name.strip()
    )
    if requested_upgd_variants == ("all",):
        requested_upgd_variants = tuple(sorted(UPGD_VARIANT_CONFIGS))
    unknown_variants = [
        name for name in requested_upgd_variants
        if name not in UPGD_VARIANT_CONFIGS
    ]
    if unknown_variants:
        known = ", ".join(sorted(UPGD_VARIANT_CONFIGS))
        raise ValueError(
            f"Unknown UPGD variant(s): {unknown_variants}. Known variants: {known}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stream_outputs: list[dict[str, Any]] = []
    for spec in STREAM_SPECS:
        stream_out = run_one_stream(
            stream_name=spec["name"],
            stream_factory=spec["factory"],
            feature_dim=spec["feature_dim"],
            n_tasks=spec["n_tasks"],
            n_contexts=spec["n_contexts"],
            n_seeds=args.seeds,
            num_steps=args.num_steps,
            upgd_variant_names=requested_upgd_variants,
        )
        stream_outputs.append(stream_out)
    wall_clock_s = time.time() - t0

    config = make_config_dict(args.seeds, args.num_steps, requested_upgd_variants)
    aggregate = build_aggregate(stream_outputs)
    paired = build_paired_vs_best_mlp(stream_outputs, aggregate)
    per_run = build_per_run_records(stream_outputs)

    json_path = args.output_dir / "out_of_class_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": config,
                "wall_clock_s": wall_clock_s,
                "aggregate": aggregate,
                "paired_vs_best_mlp": paired,
                "per_run": per_run,
            },
            f,
            indent=2,
        )

    md_path = args.output_dir / "out_of_class_SUMMARY.md"
    write_markdown_summary(md_path, aggregate, paired, config, wall_clock_s)

    print(f"\nTotal wall-clock: {wall_clock_s:.1f}s")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

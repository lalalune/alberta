#!/usr/bin/env python3
"""Focused recursive feature-utility probe for Step 2.

The target is a triple product ``x0 * x1 * x2``. A raw-only or pair-product
feature bank cannot represent it exactly; a compositional DAG can only do so by
constructing a feature of an existing feature, e.g. ``(x0 * x1) * x2``.

This script compares the historical one-step future-utility signal against the
new causal trace signal. It is deliberately a probe, not a canonical closure
claim: the output reports final-window MSE and how often the final feature bank
contains active depth-2+ structure.
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

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    run_compositional_arrays,
)
from alberta_framework.core.future_utility import trace_decay_from_half_life
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner
from alberta_framework.core.optimizers import ObGDBounding

DEFAULT_METHODS = (
    "single_mechanism",
    "single_mechanism_signed_tanh",
    "single_mechanism_retention",
    "single_mechanism_energy_novelty",
    "mlp_32x32_no_ln",
    "mlp_64x64_no_ln",
)

RETENTION_HEDGE_METHODS = (
    "recursive_retention_hedge",
    "recursive_retention_hedge_recent",
    "recursive_retention_hedge_sharp",
    "recursive_retention_hedge_sharp_recent",
    "recursive_retention_switch_recent",
)
ROUTER_METHODS = ("recursive_mlp_router",) + RETENTION_HEDGE_METHODS
SINGLE_MECHANISM_METHODS = (
    "single_mechanism",
    "single_mechanism_signed_tanh",
    "single_mechanism_retention",
    "single_mechanism_retention_fast",
    "single_mechanism_retention_tanh24",
    "single_mechanism_retention_tanh24_tanh_heavy_conservative",
    "single_mechanism_retention_theta",
    "single_mechanism_tanh_shadow",
    "single_mechanism_energy_novelty",
)

SUITE_TASKS = (
    "nonlinear",
    "interaction",
    "triple",
    "rare",
    "polynomial",
    "frequency",
)


def make_data(
    seed: int,
    num_steps: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> tuple[jax.Array, jax.Array]:
    """Return observations and masked Step 2 targets."""
    rng = np.random.default_rng(seed)
    if task_mode == "frequency":
        phase = rng.uniform(-np.pi, np.pi, size=num_steps).astype(np.float32)
        observations = rng.standard_normal((num_steps, feature_dim)).astype(np.float32)
        observations[:, 0] = np.sin(phase)
        observations[:, 1] = np.cos(phase)
    else:
        observations = rng.standard_normal((num_steps, feature_dim)).astype(np.float32)

    triple = observations[:, 0] * observations[:, 1] * observations[:, 2]
    if task_mode == "rare":
        frequent = 0.5 * observations[:, 0] - 0.25 * observations[:, 1]
        targets = np.stack([frequent, triple], axis=1)
        targets += noise_std * rng.standard_normal(targets.shape).astype(np.float32)
        rare_mask = np.arange(num_steps) % rare_period == 0
        targets[~rare_mask, 1] = np.nan
        return jnp.asarray(observations), jnp.asarray(targets.astype(np.float32))
    if task_mode == "nonlinear":
        target_signal = (
            np.tanh(1.2 * observations[:, 0] - 0.7 * observations[:, 1])
            + 0.25 * observations[:, 2] ** 2
        )
    elif task_mode == "interaction":
        target_signal = (
            observations[:, 0] * observations[:, 1]
            + 0.5 * observations[:, 2] * observations[:, 3]
        )
    elif task_mode == "triple":
        target_signal = triple
    elif task_mode == "polynomial":
        target_signal = (
            0.4 * observations[:, 0] ** 2
            - 0.25 * observations[:, 1] ** 3
            + 0.6 * triple
        )
    elif task_mode == "frequency":
        # Given sin(theta), cos(theta), this is sin(3 theta) = 3s - 4s^3.
        target_signal = 3.0 * observations[:, 0] - 4.0 * observations[:, 0] ** 3
    else:
        raise ValueError(f"unknown task_mode: {task_mode}")
    noise = noise_std * rng.standard_normal(num_steps).astype(np.float32)
    return jnp.asarray(observations), jnp.asarray((target_signal + noise)[:, None])


def variant_config(method: str) -> dict[str, Any]:
    """Return future-utility knobs for one named estimator variant."""
    if method in SINGLE_MECHANISM_METHODS:
        knobs = {
            "future_utility_mix": 0.65,
            "future_utility_trace_decay": float(trace_decay_from_half_life(16.0)),
            "future_utility_trace_mode": "contribution",
            "future_utility_rare_task_power": 0.25,
        }
        if method == "single_mechanism_energy_novelty":
            knobs.update(
                {
                    "candidate_scoring_mode": "energy_novelty",
                    "candidate_score_trace_decay": float(
                        trace_decay_from_half_life(16.0)
                    ),
                    "candidate_novelty_weight": 0.75,
                    "candidate_novelty_power": 1.0,
                    "candidate_novelty_floor": 0.05,
                }
            )
        if method == "single_mechanism_retention":
            knobs.update(
                {
                    "retention_slow_utility_decay": float(
                        trace_decay_from_half_life(32.0)
                    ),
                    "retention_tanh_min_count": 12,
                    "retention_product_min_count": 0,
                }
            )
        if method in {
            "single_mechanism_retention_fast",
            "single_mechanism_retention_tanh24",
            "single_mechanism_retention_tanh24_tanh_heavy_conservative",
            "single_mechanism_retention_theta",
        }:
            knobs.update(
                {
                    "retention_slow_utility_decay": float(
                        trace_decay_from_half_life(24.0)
                    ),
                    "retention_tanh_min_count": 24
                    if method
                    in {
                        "single_mechanism_retention_tanh24",
                        "single_mechanism_retention_tanh24_tanh_heavy_conservative",
                    }
                    else 12,
                    "retention_product_min_count": 0,
                }
            )
        return knobs
    base_method = method.removeprefix("recursive_")
    if base_method == "current":
        return {"future_utility_mix": 0.0}
    if base_method == "one_step":
        return {"future_utility_mix": 1.0, "future_utility_trace_decay": 0.0}
    if base_method.startswith("contrib_h"):
        half_life = float(base_method.removeprefix("contrib_h"))
        return {
            "future_utility_mix": 1.0,
            "future_utility_trace_decay": float(trace_decay_from_half_life(half_life)),
            "future_utility_trace_mode": "contribution",
        }
    if base_method.startswith("marginal_h"):
        half_life = float(base_method.removeprefix("marginal_h"))
        return {
            "future_utility_mix": 1.0,
            "future_utility_trace_decay": float(trace_decay_from_half_life(half_life)),
            "future_utility_trace_mode": "marginal",
        }
    if base_method == "two_timescale_h16":
        return {
            "future_utility_mix": 0.5,
            "future_utility_trace_decay": float(trace_decay_from_half_life(16.0)),
            "future_utility_trace_mode": "contribution",
        }
    if base_method == "uncertainty_age_h16":
        return {
            "future_utility_mix": 1.0,
            "future_utility_trace_decay": float(trace_decay_from_half_life(16.0)),
            "future_utility_trace_mode": "contribution",
            "future_utility_normalization": "uncertainty_age",
        }
    if base_method == "rare_credit_h16":
        return {
            "future_utility_mix": 1.0,
            "future_utility_trace_decay": float(trace_decay_from_half_life(16.0)),
            "future_utility_trace_mode": "contribution",
            "future_utility_rare_task_power": 0.5,
        }
    raise ValueError(f"unknown method: {method}")


def make_learner(method: str, n_tasks: int) -> CompositionalFeatureLearner:
    """Create one compositional learner variant."""
    knobs = variant_config(method)
    generation_strategy = "residual_imprint"
    parent_novelty_weight = 0.0
    parent_depth_prior = 0.0
    retention_depth_bonus = 0.0
    operation_prior = None
    if method.startswith("recursive_"):
        generation_strategy = "recursive_product"
    if method in SINGLE_MECHANISM_METHODS:
        generation_strategy = "robust_recursive"
        parent_novelty_weight = 0.01
        parent_depth_prior = 0.05
        retention_depth_bonus = 0.02
    if method == "single_mechanism_retention_tanh24_tanh_heavy_conservative":
        operation_prior = (0.0, 0.35, 0.0, 0.65, 0.0)
    return CompositionalFeatureLearner(
        n_features=36 if method in SINGLE_MECHANISM_METHODS else 20,
        n_tasks=n_tasks,
        candidate_count=36 if method in SINGLE_MECHANISM_METHODS else 20,
        step_size_output=(
            0.08 if method == "single_mechanism_retention_fast" else 0.05
        ),
        step_size_theta=(
            0.012 if method == "single_mechanism_retention_fast" else 0.005
        ),
        utility_decay=0.99,
        replacement_interval=15 if method in SINGLE_MECHANISM_METHODS else 20,
        min_feature_age=30 if method in SINGLE_MECHANISM_METHODS else 40,
        candidate_min_age=24
        if method == "single_mechanism_retention"
        else 12
        if method in SINGLE_MECHANISM_METHODS
        else 20,
        promotion_margin=1.0 if method in SINGLE_MECHANISM_METHODS else 1.05,
        promotion_blend=(
            0.35
            if method == "single_mechanism_retention_tanh24_tanh_heavy_conservative"
            else 0.6
            if method in SINGLE_MECHANISM_METHODS
            else 0.5
        ),
        max_depth=3,
        use_obgd=True,
        obgd_kappa=2.0,
        generation_strategy=generation_strategy,
        promotion_output_mode="blend",
        parent_temperature=0.75,
        parent_novelty_weight=parent_novelty_weight,
        parent_depth_prior=parent_depth_prior,
        retention_depth_bonus=retention_depth_bonus,
        residual_guidance=0.75 if method in SINGLE_MECHANISM_METHODS else 0.5,
        candidate_imprint_scale=(
            0.1
            if method == "single_mechanism_retention_tanh24_tanh_heavy_conservative"
            else 0.2
        ),
        train_candidate_theta=method
        in {"single_mechanism_tanh_shadow", "single_mechanism_retention_theta"},
        signed_tanh_scaffold_count=(
            24
            if method == "single_mechanism_retention_tanh24"
            or method == "single_mechanism_retention_tanh24_tanh_heavy_conservative"
            else 12
            if method
            in {
                "single_mechanism_retention",
                "single_mechanism_retention_fast",
                "single_mechanism_retention_theta",
            }
            else 6
            if method
            in {
                "single_mechanism_signed_tanh",
                "single_mechanism_tanh_shadow",
                "single_mechanism_energy_novelty",
            }
            else 0
        ),
        operation_prior=operation_prior,
        **knobs,
    )


def run_mlp_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run one fair MLP control."""
    observations, targets = make_data(
        seed,
        num_steps,
        feature_dim,
        noise_std,
        task_mode,
        rare_period,
    )
    use_layer_norm = method.endswith("_ln")
    width = 64 if "64x64" in method else 32
    learner = MultiHeadMLPLearner(
        n_heads=int(targets.shape[1]),
        hidden_sizes=(width, width),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=use_layer_norm,
    )
    state = learner.init(feature_dim=feature_dim, key=jr.key(seed + 20_000))

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        observation, target = sample
        result = learner.update(carry, observation, target)
        return result.state, result.per_head_metrics

    t0 = time.time()
    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    elapsed = time.time() - t0
    mse = np.nanmean(np.asarray(metrics[:, :, 0], dtype=np.float64), axis=1)
    held_obs, held_targets = make_data(
        seed + 100_000,
        max(final_window, 200),
        feature_dim,
        0.0,
        task_mode,
        rare_period,
    )
    held_preds = jax.vmap(lambda obs: learner.predict(final_state, obs))(held_obs)
    held_errors = np.asarray(held_preds - held_targets)
    held_mask = ~np.isnan(np.asarray(held_targets))
    held_mse_by_task = []
    for task in range(targets.shape[1]):
        task_mask = held_mask[:, task]
        held_mse_by_task.append(float(np.mean(held_errors[task_mask, task] ** 2)))
    return {
        "method": method,
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(mse[-final_window:])),
        "initial_window_mse": float(np.mean(mse[:final_window])),
        "mean_mse": float(np.mean(mse)),
        "heldout_mse_by_task": held_mse_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_mse_by_task))),
        "active_depth2_count": 0,
        "max_depth2_utility": 0.0,
        "wall_clock_s": elapsed,
    }


def run_recursive_mlp_router_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run a causal resource router over recursive features and fair MLPs."""
    del method
    observations, targets = make_data(
        seed,
        num_steps,
        feature_dim,
        noise_std,
        task_mode,
        rare_period,
    )
    n_tasks = int(targets.shape[1])
    recursive = make_learner("single_mechanism", n_tasks)
    mlp32 = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(32, 32),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
    mlp64 = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=(64, 64),
        step_size=0.1,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
    recursive_state = recursive.init(feature_dim=feature_dim, key=jr.key(seed + 30_000))
    mlp32_state = mlp32.init(feature_dim=feature_dim, key=jr.key(seed + 40_000))
    mlp64_state = mlp64.init(feature_dim=feature_dim, key=jr.key(seed + 50_000))
    router_decay = jnp.array(0.99, dtype=jnp.float32)
    warmup_steps = jnp.array(min(500, max(50, num_steps // 10)), dtype=jnp.int32)

    def masked_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        sq_error = jnp.where(active, (prediction - safe_target) ** 2, 0.0)
        return jnp.sum(sq_error) / active_count

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        recursive_s, mlp32_s, mlp64_s, loss_ema, step_count = carry
        observation, target = sample
        recursive_pred = recursive.predict(recursive_s, observation)
        mlp32_pred = mlp32.predict(mlp32_s, observation)
        mlp64_pred = mlp64.predict(mlp64_s, observation)
        expert_preds = jnp.stack([recursive_pred, mlp32_pred, mlp64_pred])
        expert_losses = jnp.stack(
            [
                masked_loss(recursive_pred, target),
                masked_loss(mlp32_pred, target),
                masked_loss(mlp64_pred, target),
            ]
        )
        selected_idx = jnp.where(
            step_count < warmup_steps,
            jnp.array(1, dtype=jnp.int32),
            jnp.argmin(loss_ema).astype(jnp.int32),
        )
        router_loss = masked_loss(expert_preds[selected_idx], target)
        new_loss_ema = router_decay * loss_ema + (1.0 - router_decay) * expert_losses

        recursive_result = recursive.update(recursive_s, observation, target)
        mlp32_result = mlp32.update(mlp32_s, observation, target)
        mlp64_result = mlp64.update(mlp64_s, observation, target)
        new_carry = (
            recursive_result.state,
            mlp32_result.state,
            mlp64_result.state,
            new_loss_ema,
            step_count + 1,
        )
        return new_carry, router_loss

    t0 = time.time()
    initial_carry = (
        recursive_state,
        mlp32_state,
        mlp64_state,
        jnp.zeros(3, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    final_carry, losses = jax.lax.scan(step_fn, initial_carry, (observations, targets))
    losses.block_until_ready()
    elapsed = time.time() - t0
    final_recursive_state, final_mlp32_state, final_mlp64_state, final_loss_ema, _ = (
        final_carry
    )

    held_obs, held_targets = make_data(
        seed + 100_000,
        max(final_window, 200),
        feature_dim,
        0.0,
        task_mode,
        rare_period,
    )
    recursive_held = jax.vmap(lambda obs: recursive.predict(final_recursive_state, obs))(
        held_obs
    )
    mlp32_held = jax.vmap(lambda obs: mlp32.predict(final_mlp32_state, obs))(held_obs)
    mlp64_held = jax.vmap(lambda obs: mlp64.predict(final_mlp64_state, obs))(held_obs)
    held_stack = jnp.stack([recursive_held, mlp32_held, mlp64_held])
    held_selected_idx = int(jnp.argmin(final_loss_ema))
    held_preds = held_stack[held_selected_idx]
    held_errors = np.asarray(held_preds - held_targets)
    held_mask = ~np.isnan(np.asarray(held_targets))
    held_mse_by_task = []
    for task in range(n_tasks):
        task_mask = held_mask[:, task]
        held_mse_by_task.append(float(np.mean(held_errors[task_mask, task] ** 2)))

    depths = np.asarray(final_recursive_state.depth)
    utilities = np.asarray(final_recursive_state.utilities)
    depth2_mask = depths >= 2
    loss_np = np.asarray(losses, dtype=np.float64)
    return {
        "method": "recursive_mlp_router",
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(loss_np[-final_window:])),
        "initial_window_mse": float(np.mean(loss_np[:final_window])),
        "mean_mse": float(np.mean(loss_np)),
        "heldout_mse_by_task": held_mse_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_mse_by_task))),
        "active_depth2_count": int(np.sum(depth2_mask)),
        "max_depth2_utility": (
            float(np.max(utilities[depth2_mask])) if np.any(depth2_mask) else 0.0
        ),
        "wall_clock_s": elapsed,
    }


def run_recursive_retention_hedge_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run a causal Hedge selector over the two strongest recursive variants."""
    observations, targets = make_data(
        seed,
        num_steps,
        feature_dim,
        noise_std,
        task_mode,
        rare_period,
    )
    n_tasks = int(targets.shape[1])
    robust = make_learner("single_mechanism_retention", n_tasks)
    nonlinear = make_learner("single_mechanism_retention_tanh24", n_tasks)
    robust_state = robust.init(feature_dim=feature_dim, key=jr.key(seed + 10_000))
    nonlinear_state = nonlinear.init(feature_dim=feature_dim, key=jr.key(seed + 10_000))
    if method == "recursive_retention_hedge_recent":
        hedge_eta = jnp.array(4.0, dtype=jnp.float32)
        hedge_discount = jnp.array(0.98, dtype=jnp.float32)
        hard_switch = False
    elif method == "recursive_retention_hedge_sharp":
        hedge_eta = jnp.array(12.0, dtype=jnp.float32)
        hedge_discount = jnp.array(0.995, dtype=jnp.float32)
        hard_switch = False
    elif method == "recursive_retention_hedge_sharp_recent":
        hedge_eta = jnp.array(12.0, dtype=jnp.float32)
        hedge_discount = jnp.array(0.98, dtype=jnp.float32)
        hard_switch = False
    elif method == "recursive_retention_switch_recent":
        hedge_eta = jnp.array(8.0, dtype=jnp.float32)
        hedge_discount = jnp.array(0.98, dtype=jnp.float32)
        hard_switch = True
    else:
        hedge_eta = jnp.array(4.0, dtype=jnp.float32)
        hedge_discount = jnp.array(0.995, dtype=jnp.float32)
        hard_switch = False

    def masked_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        sq_error = jnp.where(active, (prediction - safe_target) ** 2, 0.0)
        return jnp.sum(sq_error) / active_count

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        robust_s, nonlinear_s, log_weights = carry
        observation, target = sample
        robust_pred = robust.predict(robust_s, observation)
        nonlinear_pred = nonlinear.predict(nonlinear_s, observation)
        expert_preds = jnp.stack([robust_pred, nonlinear_pred])
        weights = jax.nn.softmax(log_weights)
        selected_weights = jnp.where(
            hard_switch,
            jax.nn.one_hot(jnp.argmax(weights), 2, dtype=jnp.float32),
            weights,
        )
        hedge_pred = jnp.sum(selected_weights[:, None] * expert_preds, axis=0)
        hedge_loss = masked_loss(hedge_pred, target)
        expert_losses = jnp.stack(
            [
                masked_loss(robust_pred, target),
                masked_loss(nonlinear_pred, target),
            ]
        )
        centered_losses = expert_losses - jnp.mean(expert_losses)
        new_log_weights = hedge_discount * log_weights - hedge_eta * centered_losses
        new_log_weights = jnp.clip(new_log_weights, -20.0, 20.0)

        robust_result = robust.update(robust_s, observation, target)
        nonlinear_result = nonlinear.update(nonlinear_s, observation, target)
        return (
            robust_result.state,
            nonlinear_result.state,
            new_log_weights,
        ), hedge_loss

    t0 = time.time()
    initial_carry = (
        robust_state,
        nonlinear_state,
        jnp.zeros(2, dtype=jnp.float32),
    )
    final_carry, losses = jax.lax.scan(step_fn, initial_carry, (observations, targets))
    losses.block_until_ready()
    elapsed = time.time() - t0
    final_robust_state, final_nonlinear_state, final_log_weights = final_carry

    held_obs, held_targets = make_data(
        seed + 100_000,
        max(final_window, 200),
        feature_dim,
        0.0,
        task_mode,
        rare_period,
    )
    robust_held = jax.vmap(lambda obs: robust.predict(final_robust_state, obs))(held_obs)
    nonlinear_held = jax.vmap(lambda obs: nonlinear.predict(final_nonlinear_state, obs))(
        held_obs
    )
    final_weights = jax.nn.softmax(final_log_weights)
    selected_final_weights = jnp.where(
        hard_switch,
        jax.nn.one_hot(jnp.argmax(final_weights), 2, dtype=jnp.float32),
        final_weights,
    )
    held_preds = jnp.sum(
        selected_final_weights[:, None, None] * jnp.stack([robust_held, nonlinear_held]),
        axis=0,
    )
    held_errors = np.asarray(held_preds - held_targets)
    held_mask = ~np.isnan(np.asarray(held_targets))
    held_mse_by_task = []
    for task in range(n_tasks):
        task_mask = held_mask[:, task]
        held_mse_by_task.append(float(np.mean(held_errors[task_mask, task] ** 2)))

    robust_depths = np.asarray(final_robust_state.depth)
    nonlinear_depths = np.asarray(final_nonlinear_state.depth)
    robust_utilities = np.asarray(final_robust_state.utilities)
    nonlinear_utilities = np.asarray(final_nonlinear_state.utilities)
    depth2_utilities = np.concatenate(
        [
            robust_utilities[robust_depths >= 2],
            nonlinear_utilities[nonlinear_depths >= 2],
        ]
    )
    loss_np = np.asarray(losses, dtype=np.float64)
    return {
        "method": method,
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(loss_np[-final_window:])),
        "initial_window_mse": float(np.mean(loss_np[:final_window])),
        "mean_mse": float(np.mean(loss_np)),
        "heldout_mse_by_task": held_mse_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_mse_by_task))),
        "active_depth2_count": int(
            np.sum(robust_depths >= 2) + np.sum(nonlinear_depths >= 2)
        ),
        "max_depth2_utility": (
            float(np.max(depth2_utilities)) if depth2_utilities.size else 0.0
        ),
        "wall_clock_s": elapsed,
        "final_hedge_weights": [float(value) for value in np.asarray(final_weights)],
        "final_selected_weights": [
            float(value) for value in np.asarray(selected_final_weights)
        ],
    }


def run_variant(
    method: str,
    seed: int,
    num_steps: int,
    final_window: int,
    feature_dim: int,
    noise_std: float,
    task_mode: str,
    rare_period: int,
) -> dict[str, Any]:
    """Run one method/seed pair and return scalar diagnostics."""
    if method in RETENTION_HEDGE_METHODS:
        return run_recursive_retention_hedge_variant(
            method,
            seed,
            num_steps,
            final_window,
            feature_dim,
            noise_std,
            task_mode,
            rare_period,
        )
    if method in ROUTER_METHODS:
        return run_recursive_mlp_router_variant(
            method,
            seed,
            num_steps,
            final_window,
            feature_dim,
            noise_std,
            task_mode,
            rare_period,
        )
    if method.startswith("mlp_"):
        return run_mlp_variant(
            method,
            seed,
            num_steps,
            final_window,
            feature_dim,
            noise_std,
            task_mode,
            rare_period,
        )
    observations, targets = make_data(
        seed,
        num_steps,
        feature_dim,
        noise_std,
        task_mode,
        rare_period,
    )
    n_tasks = int(targets.shape[1])
    learner = make_learner(method, n_tasks)
    state = learner.init(feature_dim=feature_dim, key=jr.key(seed + 10_000))
    t0 = time.time()
    result = run_compositional_arrays(learner, state, observations, targets)
    result.metrics.block_until_ready()
    elapsed = time.time() - t0
    mse = np.asarray(result.metrics[:, 0], dtype=np.float64)
    depths = np.asarray(result.state.depth)
    utilities = np.asarray(result.state.utilities)
    depth2_mask = depths >= 2
    held_obs, held_targets = make_data(
        seed + 100_000,
        max(final_window, 200),
        feature_dim,
        0.0,
        task_mode,
        rare_period,
    )
    held_preds = jax.vmap(lambda obs: learner.predict(result.state, obs))(held_obs)
    held_errors = np.asarray(held_preds - held_targets)
    held_mask = ~np.isnan(np.asarray(held_targets))
    held_mse_by_task = []
    for task in range(n_tasks):
        task_mask = held_mask[:, task]
        if np.any(task_mask):
            held_mse_by_task.append(float(np.mean(held_errors[task_mask, task] ** 2)))
        else:
            held_mse_by_task.append(float("nan"))
    return {
        "method": method,
        "seed": seed,
        "task_mode": task_mode,
        "final_window_mse": float(np.mean(mse[-final_window:])),
        "initial_window_mse": float(np.mean(mse[:final_window])),
        "mean_mse": float(np.mean(mse)),
        "heldout_mse_by_task": held_mse_by_task,
        "heldout_mean_mse": float(np.nanmean(np.asarray(held_mse_by_task))),
        "active_depth2_count": int(np.sum(depth2_mask)),
        "max_depth2_utility": (
            float(np.max(utilities[depth2_mask])) if np.any(depth2_mask) else 0.0
        ),
        "wall_clock_s": elapsed,
    }


def stderr(values: list[float]) -> float:
    """Return standard error for a list of scalars."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def paired_delta(
    records: list[dict[str, Any]],
    left_method: str,
    right_method: str,
) -> dict[str, Any]:
    """Return paired final-window MSE deltas; positive favors right_method."""
    paired = []
    for seed in sorted({int(record["seed"]) for record in records}):
        left = next(
            row
            for row in records
            if row["method"] == left_method and row["seed"] == seed
        )
        right = next(
            row
            for row in records
            if row["method"] == right_method and row["seed"] == seed
        )
        paired.append(float(left["final_window_mse"]) - float(right["final_window_mse"]))
    return {
        "left_method": left_method,
        "right_method": right_method,
        "mean_final_window_mse_delta": float(np.mean(paired)),
        "right_wins": int(sum(delta > 0.0 for delta in paired)),
        "n": len(paired),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed records by method."""
    task_modes = sorted({record["task_mode"] for record in records})
    out: dict[str, Any] = {}
    for task_mode in task_modes:
        task_rows = [record for record in records if record["task_mode"] == task_mode]
        methods = sorted({record["method"] for record in task_rows})
        out[task_mode] = {}
        for method in methods:
            rows = [record for record in task_rows if record["method"] == method]
            finals = [float(row["final_window_mse"]) for row in rows]
            held = [float(row["heldout_mean_mse"]) for row in rows]
            rare = [
                float(row["heldout_mse_by_task"][-1])
                for row in rows
                if len(row["heldout_mse_by_task"]) > 1
            ]
            depth_counts = [float(row["active_depth2_count"]) for row in rows]
            out[task_mode][method] = {
                "mean_final_window_mse": float(np.mean(finals)),
                "stderr_final_window_mse": stderr(finals),
                "mean_heldout_mse": float(np.mean(held)),
                "stderr_heldout_mse": stderr(held),
                "mean_rare_head_heldout_mse": (
                    float(np.mean(rare)) if rare else float("nan")
                ),
                "mean_active_depth2_count": float(np.mean(depth_counts)),
                "depth2_present_seeds": int(sum(count > 0 for count in depth_counts)),
            }
        mlp_methods = [method for method in methods if method.startswith("mlp_")]
        candidate_methods = [method for method in methods if not method.startswith("mlp_")]
        if candidate_methods and mlp_methods:
            best_mlp = min(
                mlp_methods,
                key=lambda method: out[task_mode][method]["mean_final_window_mse"],
            )
            out[task_mode]["best_mlp_method"] = best_mlp
            for candidate in candidate_methods:
                out[task_mode][f"paired_best_mlp_minus_{candidate}"] = paired_delta(
                    task_rows,
                    best_mlp,
                    candidate,
                )
    suite_candidates = sorted(
        {
            method
            for task in task_modes
            for method in out[task]
            if not method.startswith("mlp_")
            and not method.startswith("paired_")
            and method != "best_mlp_method"
        }
    )
    if suite_candidates:
        suite_summary: dict[str, Any] = {"tasks": len(task_modes)}
        for candidate in suite_candidates:
            closes = 0
            ties = 0
            for task in task_modes:
                comparison = out[task].get(f"paired_best_mlp_minus_{candidate}")
                if not comparison:
                    continue
                delta = comparison["mean_final_window_mse_delta"]
                if delta > 0.0:
                    closes += 1
                elif abs(delta) <= 0.02:
                    ties += 1
            suite_summary[f"{candidate}_beats_best_mlp_tasks"] = closes
            suite_summary[f"{candidate}_ties_best_mlp_tasks"] = ties
        out["suite_summary"] = suite_summary
    return out


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    lines = [
        "# Step 2 Recursive Feature Utility Probe",
        "",
        (
            f"Seeds: {results['config']['seeds']}; steps: "
            f"{results['config']['num_steps']}; final-window: "
            f"{results['config']['final_window']}."
        ),
        "",
    ]
    for task_mode, task_stats in results["aggregate"].items():
        if task_mode == "suite_summary":
            continue
        lines.extend(
            [
                "",
                f"## {task_mode}",
                "",
                "| Method | Final MSE | Heldout MSE | Active depth>=2 | Seeds depth>=2 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, stats in task_stats.items():
            if method.startswith("paired_") or method == "best_mlp_method":
                continue
            lines.append(
                f"| `{method}` | "
                f"{stats['mean_final_window_mse']:.4f} +/- "
                f"{stats['stderr_final_window_mse']:.4f} | "
                f"{stats['mean_heldout_mse']:.4f} +/- "
                f"{stats['stderr_heldout_mse']:.4f} | "
                f"{stats['mean_active_depth2_count']:.2f} | "
                f"{stats['depth2_present_seeds']}/{results['config']['seeds']} |"
            )
        comparisons = [
            (label, stats)
            for label, stats in task_stats.items()
            if label.startswith("paired_best_mlp_minus_")
        ]
        if comparisons:
            lines.append("")
        for label, comparison in comparisons:
            candidate = label.removeprefix("paired_best_mlp_minus_")
            lines.extend(
                [
                    (
                        f"Best fair MLP: `{task_stats['best_mlp_method']}`. "
                        f"Paired delta best MLP minus `{candidate}`: "
                        f"{comparison['mean_final_window_mse_delta']:.4f}; "
                        f"`{candidate}` wins {comparison['right_wins']}/"
                        f"{comparison['n']} seeds."
                    ),
                ]
            )
    suite = results["aggregate"].get("suite_summary")
    if suite:
        lines.extend(
            [
                "",
                "## Suite summary",
                "",
            ]
        )
        for key, value in sorted(suite.items()):
            if not key.endswith("_beats_best_mlp_tasks"):
                continue
            candidate = key.removesuffix("_beats_best_mlp_tasks")
            ties = suite.get(f"{candidate}_ties_best_mlp_tasks", 0)
            lines.append(
                f"`{candidate}` beats best fair MLP on {value}/{suite['tasks']} "
                f"tasks and ties within 0.02 MSE on {ties}/{suite['tasks']} tasks."
            )
    lines.extend(
        [
            "",
            "Interpretation: `single_mechanism` is the robust recursive "
            "configuration: contribution-trace utility, residual imprint, "
            "product-biased operation priors, utility/novelty parent choice, "
            "and depth retention. `single_mechanism_energy_novelty` keeps the "
            "same signed-tanh scaffold budget as `single_mechanism_signed_tanh` "
            "but changes candidate credit to energy-normalized residual "
            "alignment gated by active-feature correlation novelty. "
            "`single_mechanism_retention` keeps the signed-tanh scaffold "
            "budget, uses energy-normalized residual alignment with no "
            "novelty penalty, and adds opt-in slow utility/family quota "
            "retention. Each "
            "non-MLP candidate is judged against the best fair MLP run for "
            "each task.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=2500)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated estimator variants to run.",
    )
    parser.add_argument(
        "--task-mode",
        choices=SUITE_TASKS,
        default="triple",
        help="Task to run when --suite is not set.",
    )
    parser.add_argument("--suite", action="store_true", help="Run all suite tasks.")
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_future_utility_probe"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 2 seeds, 800 steps, and a 200-step final window.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 2
        args.num_steps = 800
        args.final_window = 200
        args.methods = "single_mechanism,mlp_32x32_no_ln"
    args.final_window = min(args.final_window, args.num_steps)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    task_modes = list(SUITE_TASKS) if args.suite else [args.task_mode]
    for task_mode in task_modes:
        for seed in range(args.seeds):
            for method in methods:
                record = run_variant(
                    method,
                    seed,
                    args.num_steps,
                    args.final_window,
                    args.feature_dim,
                    args.noise_std,
                    task_mode,
                    args.rare_period,
                )
                records.append(record)
                print(
                    f"task={task_mode} seed={seed:02d} {method}: "
                    f"final={record['final_window_mse']:.4f}, "
                    f"held={record['heldout_mean_mse']:.4f}, "
                    f"depth2={record['active_depth2_count']}"
                )

    results = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "methods": methods,
            "task_modes": task_modes,
        },
        "aggregate": aggregate(records),
        "per_run": records,
    }
    json_path = args.output_dir / "recursive_feature_utility_results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_summary(summary_path, results)
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

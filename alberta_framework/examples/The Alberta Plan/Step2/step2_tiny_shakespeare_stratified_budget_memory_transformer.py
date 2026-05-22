#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare stratified-budget memory transformer.

This runner attacks several static knobs in the replay-capped advantage-memory
candidate. It keeps the prior `gate_max=0.15` post-FFN replay-gated memory as a
paired comparator, then replaces the hard cap and FIFO replay objective with a
stratified replay controller.

The learned budget is a scalar cap state. Replay advantage supplies utility,
hard-negative and positive strata make the utility estimate less recency-bound,
uncertainty samples probe stale decisions, and online controllers adapt gate L2
and the slow-memory learning-rate multiplier.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_tiny_shakespeare_advantage_memory_transformer import (
    AdvantageMemoryState,
    eval_advantage_fast_only,
    eval_advantage_memory_transformer,
    init_advantage_params_state,
    model_parts,
    prediction_advantage,
    reset_proto_row,
    run_advantage_memory_transformer,
    sgd_step_decoupled,
)
from step2_tiny_shakespeare_proto_basis_transformer import (
    make_proto_block,
    select_center_slot,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    clip_grads,
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    encode_text,
    ensure_tiny_shakespeare,
    eval_transformer,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    stderr,
    summarize_online,
)

from alberta_framework.core.prototype_basis import (
    PrototypeBasisBlock,
    PrototypeBasisState,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured in result artifacts."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    proto_count: int
    eval_steps: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    fast_lr: float
    slow_lr: float
    grad_clip: float
    proto_update_rate: float
    proto_novelty_threshold: float
    proto_bandwidth: float
    proto_adaptive_bandwidth: bool
    proto_bandwidth_update_rate: float
    gate_init_logit: float
    gate_lr: float
    gate_decay: float
    advantage_margin: float
    gate_l2: float
    static_gate_max: float
    replay_size: int
    train_loss_mode: str
    memory_loss_weight: float
    reset_mode: str
    placement: str
    budget_init: float
    budget_min: float
    budget_max: float
    budget_lr: float
    budget_ema_decay: float
    budget_target_utilization: float
    budget_cost: float
    budget_pressure_floor: float
    budget_advantage_floor: float
    gate_l2_min: float
    gate_l2_max: float
    gate_l2_lr: float
    gate_l2_pressure_weight: float
    slow_lr_multiplier_min: float
    slow_lr_multiplier_max: float
    slow_lr_control_lr: float
    slow_lr_control_cost: float
    replay_strata: str
    data_path: str
    output_dir: str
    seed: int


@chex.dataclass(frozen=True)
class LearnedBudgetMemoryState:
    """Slow memory state plus learned resource-controller scalars."""

    proto_state: PrototypeBasisState
    gate_logit: jax.Array
    advantage_ema: jax.Array
    replay_advantage_ema: jax.Array
    replay_advantage_sq_ema: jax.Array
    budget_logit: jax.Array
    gate_l2_logit: jax.Array
    slow_lr_multiplier_logit: jax.Array
    init_value: jax.Array
    step_count: jax.Array


def _safe_logit(probability: jax.Array) -> jax.Array:
    """Return a numerically safe logit."""
    clipped = jnp.clip(probability, 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped / (1.0 - clipped))


def init_budget_logit(
    *,
    budget_init: float,
    budget_min: float,
    budget_max: float,
) -> jax.Array:
    """Map an initial budget in `[budget_min, budget_max]` to logit space."""
    span = max(budget_max - budget_min, 1e-6)
    fraction = (budget_init - budget_min) / span
    return _safe_logit(jnp.asarray(fraction, dtype=jnp.float32))


def budget_from_logit(
    budget_logit: jax.Array,
    *,
    budget_min: float,
    budget_max: float,
) -> jax.Array:
    """Map budget logit state to the active scalar gate cap."""
    min_value = jnp.asarray(budget_min, dtype=jnp.float32)
    span = jnp.asarray(budget_max - budget_min, dtype=jnp.float32)
    return min_value + span * jax.nn.sigmoid(budget_logit)


def bounded_logit(
    *,
    value: float,
    lower: float,
    upper: float,
) -> jax.Array:
    """Map a bounded scalar to logit space."""
    span = max(upper - lower, 1e-6)
    fraction = (value - lower) / span
    return _safe_logit(jnp.asarray(fraction, dtype=jnp.float32))


def bounded_from_logit(
    logit: jax.Array,
    *,
    lower: float,
    upper: float,
) -> jax.Array:
    """Map a logit to a bounded scalar interval."""
    min_value = jnp.asarray(lower, dtype=jnp.float32)
    span = jnp.asarray(upper - lower, dtype=jnp.float32)
    return min_value + span * jax.nn.sigmoid(logit)


def init_learned_budget_params_state(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    gate_init_logit: float,
    budget_init: float,
    budget_min: float,
    budget_max: float,
    gate_l2_init: float,
    gate_l2_min: float,
    gate_l2_max: float,
    slow_lr_multiplier_init: float,
    slow_lr_multiplier_min: float,
    slow_lr_multiplier_max: float,
) -> tuple[dict[str, Any], LearnedBudgetMemoryState]:
    """Initialize transformer params and stratified-budget memory state."""
    params, base_state = init_advantage_params_state(
        key,
        block=block,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        gate_init_logit=gate_init_logit,
        gate_mode="scalar",
    )
    state = LearnedBudgetMemoryState(
        proto_state=base_state.proto_state,
        gate_logit=base_state.gate_logit,
        advantage_ema=base_state.advantage_ema,
        replay_advantage_ema=jnp.asarray(0.0, dtype=jnp.float32),
        replay_advantage_sq_ema=jnp.asarray(0.0, dtype=jnp.float32),
        budget_logit=init_budget_logit(
            budget_init=budget_init,
            budget_min=budget_min,
            budget_max=budget_max,
        ),
        gate_l2_logit=bounded_logit(
            value=gate_l2_init,
            lower=gate_l2_min,
            upper=gate_l2_max,
        ),
        slow_lr_multiplier_logit=bounded_logit(
            value=slow_lr_multiplier_init,
            lower=slow_lr_multiplier_min,
            upper=slow_lr_multiplier_max,
        ),
        init_value=base_state.init_value,
        step_count=base_state.step_count,
    )
    return params, state


def run_learned_budget_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: LearnedBudgetMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    advantage_margin: float,
    gate_l2: float,
    gate_init_logit: float,
    replay_size: int,
    train_loss_mode: str,
    memory_loss_weight: float,
    reset_mode: str,
    budget_min: float,
    budget_max: float,
    budget_lr: float,
    budget_ema_decay: float,
    budget_target_utilization: float,
    budget_cost: float,
    budget_pressure_floor: float,
    budget_advantage_floor: float,
    gate_l2_min: float,
    gate_l2_max: float,
    gate_l2_lr: float,
    gate_l2_pressure_weight: float,
    slow_lr_multiplier_min: float,
    slow_lr_multiplier_max: float,
    slow_lr_control_lr: float,
    slow_lr_control_cost: float,
) -> tuple[dict[str, Any], LearnedBudgetMemoryState, np.ndarray]:
    """Train a stratified replay memory transformer with learned controllers."""
    replay_capacity = max(1, replay_size)

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: LearnedBudgetMemoryState,
    ) -> tuple[tuple[Any, ...], jax.Array]:
        def step(
            carry: tuple[Any, ...],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[Any, ...], jax.Array]:
            (
                params,
                state,
                replay_contexts,
                replay_labels,
                replay_advantages,
                replay_count,
                replay_index,
            ) = carry
            context, label = inputs

            budget = budget_from_logit(
                state.budget_logit,
                budget_min=budget_min,
                budget_max=budget_max,
            )
            learned_gate_l2 = bounded_from_logit(
                state.gate_l2_logit,
                lower=gate_l2_min,
                upper=gate_l2_max,
            )
            slow_lr_multiplier = bounded_from_logit(
                state.slow_lr_multiplier_logit,
                lower=slow_lr_multiplier_min,
                upper=slow_lr_multiplier_max,
            )
            active_slow_lr = (
                jnp.asarray(slow_lr, dtype=jnp.float32) * slow_lr_multiplier
            )

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[
                jax.Array,
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                base_logits, memory_logits, basis_input, activations, gate = model_parts(
                    block,
                    candidate,
                    state,
                    context,
                    placement=placement,
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                memory_loss = cross_entropy_from_logits(memory_logits, label)
                if train_loss_mode == "memory":
                    train_loss = memory_loss
                else:
                    weight = jnp.asarray(memory_loss_weight, dtype=jnp.float32)
                    train_loss = (1.0 - weight) * base_loss + weight * memory_loss
                return train_loss, (base_loss, memory_logits, basis_input, activations, gate)

            (_train_loss, (base_loss, logits, basis_input, activations, gate)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(params)
            )
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled(
                params,
                grads,
                fast_lr=fast_lr,
                slow_lr=cast(float, active_slow_lr),
            )
            memory_loss = cross_entropy_from_logits(logits, label)
            advantage = base_loss - memory_loss
            slot, novel = select_center_slot(block, state.proto_state, basis_input)
            new_proto_state, center_metrics = block.update_centers(
                state.proto_state,
                basis_input,
            )
            new_params = reset_proto_row(
                new_params,
                state,
                slot,
                novel,
                reset_mode=reset_mode,
            )

            slots = jnp.arange(replay_capacity, dtype=jnp.int32)
            valid = slots < replay_count
            invalid_score = jnp.asarray(-1e9, dtype=jnp.float32)
            positive_slot = jnp.argmax(
                jnp.where(valid, replay_advantages, invalid_score)
            ).astype(jnp.int32)
            hard_negative_slot = jnp.argmax(
                jnp.where(valid, -replay_advantages, invalid_score)
            ).astype(jnp.int32)
            uncertain_slot = jnp.argmax(
                jnp.where(
                    valid,
                    jnp.abs(replay_advantages - state.replay_advantage_ema),
                    invalid_score,
                )
            ).astype(jnp.int32)
            recent_slot = jnp.mod(replay_index - jnp.array(1, dtype=jnp.int32), replay_capacity)
            phase = jnp.mod(state.step_count, jnp.array(4, dtype=jnp.int32))
            replay_slot = jnp.where(
                phase == 0,
                hard_negative_slot,
                jnp.where(
                    phase == 1,
                    positive_slot,
                    jnp.where(phase == 2, uncertain_slot, recent_slot),
                ),
            )
            replay_slot = jnp.where(
                replay_count > 0,
                replay_slot,
                jnp.mod(replay_index, replay_capacity),
            )
            replay_state = LearnedBudgetMemoryState(
                proto_state=new_proto_state,
                gate_logit=state.gate_logit,
                advantage_ema=state.advantage_ema,
                replay_advantage_ema=state.replay_advantage_ema,
                replay_advantage_sq_ema=state.replay_advantage_sq_ema,
                budget_logit=state.budget_logit,
                gate_l2_logit=state.gate_l2_logit,
                slow_lr_multiplier_logit=state.slow_lr_multiplier_logit,
                init_value=state.init_value,
                step_count=state.step_count,
            )
            replay_advantage = prediction_advantage(
                block,
                new_params,
                cast(AdvantageMemoryState, replay_state),
                replay_contexts[replay_slot],
                replay_labels[replay_slot],
                placement=placement,
            )
            gate_update_advantage = jnp.where(
                replay_count > 0,
                replay_advantage,
                advantage,
            )

            gate_signal = gate_update_advantage - jnp.asarray(
                advantage_margin,
                dtype=jnp.float32,
            )
            gate_signal = gate_signal - learned_gate_l2 * gate
            new_gate_logit = (
                jnp.asarray(gate_decay, dtype=jnp.float32) * state.gate_logit
                + jnp.asarray(gate_lr, dtype=jnp.float32)
                * jnp.clip(gate_signal, -1.0, 1.0)
            )
            max_logit = _safe_logit(budget)
            new_gate_logit = jnp.clip(new_gate_logit, -8.0, jnp.minimum(max_logit, 8.0))
            new_gate_logit = jnp.where(
                novel,
                jnp.maximum(new_gate_logit, jnp.asarray(gate_init_logit, dtype=jnp.float32)),
                new_gate_logit,
            )

            decay = jnp.asarray(budget_ema_decay, dtype=jnp.float32)
            new_replay_advantage_ema = (
                decay * state.replay_advantage_ema
                + (1.0 - decay) * gate_update_advantage
            )
            new_replay_advantage_sq_ema = (
                decay * state.replay_advantage_sq_ema
                + (1.0 - decay) * gate_update_advantage**2
            )
            replay_variance = jnp.maximum(
                new_replay_advantage_sq_ema - new_replay_advantage_ema**2,
                0.0,
            )
            replay_std = jnp.sqrt(
                replay_variance
                + jnp.asarray(budget_advantage_floor, dtype=jnp.float32) ** 2
            )
            normalized_utility = gate_signal / replay_std
            utility_signal = jnp.tanh(normalized_utility)
            budget_pressure = gate / jnp.maximum(budget, 1e-6)
            pressure_gate = jax.nn.sigmoid(
                8.0
                * (
                    budget_pressure
                    - jnp.asarray(budget_target_utilization, dtype=jnp.float32)
                )
            )
            pressure_gate = jnp.maximum(
                pressure_gate,
                jnp.asarray(budget_pressure_floor, dtype=jnp.float32),
            )
            normalized_budget = (budget - jnp.asarray(budget_min, dtype=jnp.float32)) / (
                jnp.asarray(budget_max - budget_min, dtype=jnp.float32)
            )
            budget_signal = (
                utility_signal * pressure_gate
                - jnp.asarray(budget_cost, dtype=jnp.float32) * normalized_budget
            )
            new_budget_logit = jnp.clip(
                state.budget_logit
                + jnp.asarray(budget_lr, dtype=jnp.float32)
                * jnp.clip(budget_signal, -1.0, 1.0),
                -8.0,
                8.0,
            )
            new_budget = budget_from_logit(
                new_budget_logit,
                budget_min=budget_min,
                budget_max=budget_max,
            )
            new_max_logit = _safe_logit(new_budget)
            new_gate_logit = jnp.clip(
                new_gate_logit,
                -8.0,
                jnp.minimum(new_max_logit, 8.0),
            )
            l2_pressure = budget_pressure - jnp.asarray(
                budget_target_utilization,
                dtype=jnp.float32,
            )
            gate_l2_signal = (
                -utility_signal
                + jnp.asarray(gate_l2_pressure_weight, dtype=jnp.float32) * l2_pressure
            )
            new_gate_l2_logit = jnp.clip(
                state.gate_l2_logit
                + jnp.asarray(gate_l2_lr, dtype=jnp.float32)
                * jnp.clip(gate_l2_signal, -1.0, 1.0),
                -8.0,
                8.0,
            )
            slow_lr_signal = (
                utility_signal
                - jnp.asarray(slow_lr_control_cost, dtype=jnp.float32)
                * (slow_lr_multiplier - 1.0)
            )
            new_slow_lr_multiplier_logit = jnp.clip(
                state.slow_lr_multiplier_logit
                + jnp.asarray(slow_lr_control_lr, dtype=jnp.float32)
                * jnp.clip(slow_lr_signal, -1.0, 1.0),
                -8.0,
                8.0,
            )

            residual = block.transform(new_params["proto"], activations)
            improved = advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * residual,
                state.init_value,
            )
            new_state = LearnedBudgetMemoryState(
                proto_state=new_proto_state,
                gate_logit=new_gate_logit,
                advantage_ema=0.99 * state.advantage_ema + 0.01 * advantage,
                replay_advantage_ema=new_replay_advantage_ema,
                replay_advantage_sq_ema=new_replay_advantage_sq_ema,
                budget_logit=new_budget_logit,
                gate_l2_logit=new_gate_l2_logit,
                slow_lr_multiplier_logit=new_slow_lr_multiplier_logit,
                init_value=init_value,
                step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
            )

            updated_replay_advantages = replay_advantages.at[replay_slot].set(
                jnp.where(replay_count > 0, gate_update_advantage, replay_advantages[replay_slot])
            )
            write_slot = jnp.mod(replay_index, replay_capacity)
            new_replay_contexts = replay_contexts.at[write_slot].set(context)
            new_replay_labels = replay_labels.at[write_slot].set(label)
            new_replay_advantages = updated_replay_advantages.at[write_slot].set(
                advantage,
            )
            new_replay_count = jnp.minimum(
                replay_count + jnp.array(1, dtype=jnp.int32),
                jnp.array(replay_capacity, dtype=jnp.int32),
            )
            new_replay_index = jnp.mod(
                replay_index + jnp.array(1, dtype=jnp.int32),
                jnp.array(replay_capacity, dtype=jnp.int32),
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    memory_loss,
                    acc,
                    base_loss,
                    advantage,
                    jnp.mean(gate),
                    jnp.mean(new_gate_logit),
                    center_metrics[0],
                    center_metrics[1],
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    gate_update_advantage,
                    budget,
                    new_budget,
                    budget_signal,
                    budget_pressure,
                    new_replay_advantage_ema,
                    replay_std,
                    learned_gate_l2,
                    slow_lr_multiplier,
                    active_slow_lr,
                    replay_advantages[replay_slot],
                    replay_slot.astype(jnp.float32),
                    phase.astype(jnp.float32),
                ]
            )
            return (
                new_params,
                new_state,
                new_replay_contexts,
                new_replay_labels,
                new_replay_advantages,
                new_replay_count,
                new_replay_index,
            ), metrics

        replay_contexts = jnp.zeros(
            (replay_capacity, contexts.shape[1]),
            dtype=contexts.dtype,
        )
        replay_labels = jnp.zeros((replay_capacity,), dtype=labels.dtype)
        replay_advantages = jnp.zeros((replay_capacity,), dtype=jnp.float32)
        initial = (
            params,
            state,
            replay_contexts,
            replay_labels,
            replay_advantages,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(step, initial, (contexts, labels))

    (final_params, final_state, *_), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def summarize_learned_budget(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize online, gate, replay, and controller diagnostics."""
    online = summarize_online(metrics[:, :2], final_window)
    window = metrics[-min(final_window, metrics.shape[0]) :]
    online.update(
        {
            "final_window_base_nll": float(np.mean(window[:, 2])),
            "final_window_advantage": float(np.mean(window[:, 3])),
            "final_window_gate": float(np.mean(window[:, 4])),
            "final_window_gate_logit": float(np.mean(window[:, 5])),
            "final_window_active_prototypes": float(np.mean(window[:, 6])),
            "final_window_allocation_rate": float(np.mean(window[:, 7])),
            "final_window_active_features": float(np.mean(window[:, 8])),
            "final_window_gate_update_advantage": float(np.mean(window[:, 9])),
            "final_window_budget": float(np.mean(window[:, 10])),
            "final_window_new_budget": float(np.mean(window[:, 11])),
            "final_window_budget_signal": float(np.mean(window[:, 12])),
            "final_window_budget_pressure": float(np.mean(window[:, 13])),
            "final_window_replay_advantage_ema": float(np.mean(window[:, 14])),
            "final_window_replay_advantage_std": float(np.mean(window[:, 15])),
            "final_window_learned_gate_l2": float(np.mean(window[:, 16])),
            "final_window_slow_lr_multiplier": float(np.mean(window[:, 17])),
            "final_window_active_slow_lr": float(np.mean(window[:, 18])),
            "final_window_selected_replay_prior": float(np.mean(window[:, 19])),
            "final_window_replay_slot": float(np.mean(window[:, 20])),
            "final_window_replay_phase": float(np.mean(window[:, 21])),
            "final_budget": float(metrics[-1, 11]),
            "final_learned_gate_l2": float(metrics[-1, 16]),
            "final_slow_lr_multiplier": float(metrics[-1, 17]),
        }
    )
    return online


def aggregate_metric(records: list[dict[str, Any]], method: str, metric: str) -> np.ndarray:
    """Collect metric values across seeds."""
    fallback = {
        "final_window_base_nll": "final_window_nll",
        "eval_fast_nll": "eval_nll",
        "eval_fast_perplexity": "eval_perplexity",
    }
    values = []
    for row in records:
        if row["method"] != method:
            continue
        summary = row["summary"]
        values.append(summary[metric] if metric in summary else summary[fallback[metric]])
    return np.asarray(values, dtype=np.float64)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "static_post_ffn_memory",
        "stratified_budget_post_ffn_memory",
    ]
    labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "static_post_ffn_memory": "Static `gate_max=0.15`",
        "stratified_budget_post_ffn_memory": "Stratified Budget",
    }
    metrics = [
        "final_window_nll",
        "final_window_base_nll",
        "final_window_accuracy",
        "eval_nll",
        "eval_fast_nll",
        "eval_accuracy",
        "eval_perplexity",
        "eval_fast_perplexity",
        "train_s",
        "train_steps_per_s",
    ]
    lower = {
        "final_window_nll",
        "final_window_base_nll",
        "eval_nll",
        "eval_fast_nll",
        "eval_perplexity",
        "eval_fast_perplexity",
        "train_s",
    }
    lines = [
        "# Tiny Shakespeare Stratified-Budget Memory Transformer",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        "",
        "## Mechanism",
        "",
        "The learned path replaces the fixed memory cap with a scalar budget state,",
        "samples replay from hard-negative/positive/uncertainty/recency strata,",
        "and adapts gate L2 plus the slow-memory LR multiplier online.",
        "",
        "## Architecture and State",
        "",
        "| Method | Trainable params | Trainable bytes | "
        "Extra state elements | Extra state bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        profile = payload["profiles"][method]
        lines.append(
            f"| {labels[method]} | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    lines.extend(["", "## Metrics", ""])
    lines.append("| Metric | " + " | ".join(labels[m] for m in methods) + " |")
    lines.append("|---|" + "---:|" * len(methods))
    for metric in metrics:
        cells = []
        for method in methods:
            values = aggregate_metric(records, method, metric)
            cells.append(f"{np.mean(values):.4f} +/- {stderr(values):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(["", "## Diffs vs Static Cap", ""])
    lines.append("| Metric | Stratified budget minus static, positive favors stratified |")
    lines.append("|---|---:|")
    for metric in metrics:
        static = aggregate_metric(records, "static_post_ffn_memory", metric)
        learned = aggregate_metric(records, "stratified_budget_post_ffn_memory", metric)
        diff = static - learned if metric in lower else learned - static
        lines.append(f"| `{metric}` | {np.mean(diff):+.4f} +/- {stderr(diff):.4f} |")
    lines.extend(["", "## Stratified-Budget Diagnostics", ""])
    lines.append("| Metric | Mean +/- stderr |")
    lines.append("|---|---:|")
    rows = [row for row in records if row["method"] == "stratified_budget_post_ffn_memory"]
    for metric in [
        "final_window_advantage",
        "final_window_gate_update_advantage",
        "final_window_gate",
        "final_window_budget",
        "final_window_new_budget",
        "final_budget",
        "final_window_budget_signal",
        "final_window_budget_pressure",
        "final_window_replay_advantage_ema",
        "final_window_replay_advantage_std",
        "final_window_learned_gate_l2",
        "final_window_slow_lr_multiplier",
        "final_window_active_slow_lr",
        "final_window_selected_replay_prior",
        "final_window_active_features",
    ]:
        values = np.asarray([row["summary"][metric] for row in rows], dtype=np.float64)
        lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--fast-lr", type=float, default=0.15)
    parser.add_argument("--slow-lr", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.0002)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--proto-adaptive-bandwidth", action="store_true")
    parser.add_argument("--proto-bandwidth-update-rate", type=float, default=0.1)
    parser.add_argument("--gate-init-logit", type=float, default=-3.0)
    parser.add_argument("--gate-lr", type=float, default=0.5)
    parser.add_argument("--gate-decay", type=float, default=0.995)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--gate-l2", type=float, default=0.1)
    parser.add_argument(
        "--static-gate-max",
        type=float,
        default=0.15,
        help="Comparator cap for the prior replay-gated candidate.",
    )
    parser.add_argument("--replay-size", type=int, default=128)
    parser.add_argument(
        "--train-loss-mode",
        choices=("memory", "blend"),
        default="memory",
    )
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-mode", choices=("none", "zero", "meta_ema"), default="meta_ema")
    parser.add_argument("--placement", choices=("post_ffn", "pre_ffn_kv"), default="post_ffn")
    parser.add_argument("--budget-init", type=float, default=0.15)
    parser.add_argument("--budget-min", type=float, default=0.02)
    parser.add_argument("--budget-max", type=float, default=0.35)
    parser.add_argument("--budget-lr", type=float, default=0.0005)
    parser.add_argument("--budget-ema-decay", type=float, default=0.995)
    parser.add_argument("--budget-target-utilization", type=float, default=0.85)
    parser.add_argument("--budget-cost", type=float, default=0.005)
    parser.add_argument(
        "--budget-pressure-floor",
        type=float,
        default=0.25,
        help="Minimum utility weight for budget opening even below target utilization.",
    )
    parser.add_argument("--budget-advantage-floor", type=float, default=0.01)
    parser.add_argument("--gate-l2-min", type=float, default=0.0)
    parser.add_argument("--gate-l2-max", type=float, default=0.2)
    parser.add_argument("--gate-l2-lr", type=float, default=0.001)
    parser.add_argument("--gate-l2-pressure-weight", type=float, default=0.25)
    parser.add_argument("--slow-lr-multiplier-min", type=float, default=0.5)
    parser.add_argument("--slow-lr-multiplier-max", type=float, default=1.5)
    parser.add_argument("--slow-lr-control-lr", type=float, default=0.0005)
    parser.add_argument("--slow-lr-control-cost", type=float, default=0.1)
    parser.add_argument(
        "--replay-strata",
        choices=("hard_positive_uncertain_recent",),
        default="hard_positive_uncertain_recent",
        help="Replay sampling schedule used by the stratified controller.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/stratified_budget_memory_transformer"),
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments."""
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.d_model < 1 or args.mlp_hidden < 1 or args.proto_count < 1:
        raise ValueError("--d-model, --mlp-hidden, and --proto-count must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if min(args.baseline_lr, args.fast_lr, args.slow_lr, args.gate_lr) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 < args.proto_update_rate <= 1.0:
        raise ValueError("--proto-update-rate must be in (0, 1]")
    if args.proto_novelty_threshold < 0.0:
        raise ValueError("--proto-novelty-threshold must be non-negative")
    if args.proto_bandwidth <= 0.0:
        raise ValueError("--proto-bandwidth must be positive")
    if not 0.0 <= args.proto_bandwidth_update_rate <= 1.0:
        raise ValueError("--proto-bandwidth-update-rate must be in [0, 1]")
    if not 0.0 <= args.gate_decay <= 1.0:
        raise ValueError("--gate-decay must be in [0, 1]")
    if not 0.0 < args.static_gate_max < 1.0:
        raise ValueError("--static-gate-max must be in (0, 1)")
    if not 0.0 <= args.memory_loss_weight <= 1.0:
        raise ValueError("--memory-loss-weight must be in [0, 1]")
    if args.replay_size < 1:
        raise ValueError("--replay-size must be positive")
    if not 0.0 < args.budget_min < args.budget_max < 1.0:
        raise ValueError("--budget-min < --budget-max must hold inside (0, 1)")
    if not args.budget_min <= args.budget_init <= args.budget_max:
        raise ValueError("--budget-init must be within [budget-min, budget-max]")
    if args.budget_lr < 0.0:
        raise ValueError("--budget-lr must be non-negative")
    if not 0.0 <= args.budget_ema_decay < 1.0:
        raise ValueError("--budget-ema-decay must be in [0, 1)")
    if not 0.0 < args.budget_target_utilization <= 1.0:
        raise ValueError("--budget-target-utilization must be in (0, 1]")
    if args.budget_cost < 0.0:
        raise ValueError("--budget-cost must be non-negative")
    if not 0.0 <= args.budget_pressure_floor <= 1.0:
        raise ValueError("--budget-pressure-floor must be in [0, 1]")
    if args.budget_advantage_floor <= 0.0:
        raise ValueError("--budget-advantage-floor must be positive")
    if not 0.0 <= args.gate_l2_min <= args.gate_l2 <= args.gate_l2_max:
        raise ValueError("--gate-l2 must be within [gate-l2-min, gate-l2-max]")
    if args.gate_l2_max <= args.gate_l2_min:
        raise ValueError("--gate-l2-max must exceed --gate-l2-min")
    if args.gate_l2_lr < 0.0 or args.gate_l2_pressure_weight < 0.0:
        raise ValueError("--gate-l2-lr and --gate-l2-pressure-weight must be non-negative")
    if not 0.0 < args.slow_lr_multiplier_min <= 1.0 <= args.slow_lr_multiplier_max:
        raise ValueError(
            "--slow-lr-multiplier-min <= 1 <= --slow-lr-multiplier-max must hold"
        )
    if args.slow_lr_multiplier_max <= args.slow_lr_multiplier_min:
        raise ValueError("--slow-lr-multiplier-max must exceed --slow-lr-multiplier-min")
    if args.slow_lr_control_lr < 0.0 or args.slow_lr_control_cost < 0.0:
        raise ValueError("--slow-lr-control-lr and --slow-lr-control-cost must be non-negative")


def static_summary(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdvantageMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    eval_contexts: jax.Array,
    eval_labels: jax.Array,
    args: argparse.Namespace,
    final_window: int,
) -> tuple[dict[str, Any], AdvantageMemoryState, np.ndarray, dict[str, float]]:
    """Run the static-cap comparator and summarize it."""
    final_params, final_state, metrics = run_advantage_memory_transformer(
        block,
        params,
        state,
        contexts,
        labels,
        placement=args.placement,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        grad_clip=args.grad_clip,
        gate_lr=args.gate_lr,
        gate_decay=args.gate_decay,
        gate_max=args.static_gate_max,
        advantage_margin=args.advantage_margin,
        gate_l2=args.gate_l2,
        gate_init_logit=args.gate_init_logit,
        gate_objective="replay",
        replay_size=args.replay_size,
        train_loss_mode=args.train_loss_mode,
        memory_loss_weight=args.memory_loss_weight,
        reset_mode=args.reset_mode,
    )
    from step2_tiny_shakespeare_advantage_memory_transformer import summarize_advantage

    summary = {
        **summarize_advantage(metrics, final_window),
        **eval_advantage_memory_transformer(
            block,
            final_params,
            final_state,
            eval_contexts,
            eval_labels,
            placement=args.placement,
        ),
        **eval_advantage_fast_only(
            block,
            final_params,
            final_state,
            eval_contexts,
            eval_labels,
            placement=args.placement,
        ),
    }
    return final_params, final_state, metrics, summary


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = metadata["vocab_size"]
    final_window = args.final_window if args.final_window > 0 else args.eval_steps
    block = make_proto_block(args)

    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        proto_count=args.proto_count,
        eval_steps=args.eval_steps,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=args.proto_adaptive_bandwidth,
        proto_bandwidth_update_rate=args.proto_bandwidth_update_rate,
        gate_init_logit=args.gate_init_logit,
        gate_lr=args.gate_lr,
        gate_decay=args.gate_decay,
        advantage_margin=args.advantage_margin,
        gate_l2=args.gate_l2,
        static_gate_max=args.static_gate_max,
        replay_size=args.replay_size,
        train_loss_mode=args.train_loss_mode,
        memory_loss_weight=args.memory_loss_weight,
        reset_mode=args.reset_mode,
        placement=args.placement,
        budget_init=args.budget_init,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        budget_lr=args.budget_lr,
        budget_ema_decay=args.budget_ema_decay,
        budget_target_utilization=args.budget_target_utilization,
        budget_cost=args.budget_cost,
        budget_pressure_floor=args.budget_pressure_floor,
        budget_advantage_floor=args.budget_advantage_floor,
        gate_l2_min=args.gate_l2_min,
        gate_l2_max=args.gate_l2_max,
        gate_l2_lr=args.gate_l2_lr,
        gate_l2_pressure_weight=args.gate_l2_pressure_weight,
        slow_lr_multiplier_min=args.slow_lr_multiplier_min,
        slow_lr_multiplier_max=args.slow_lr_multiplier_max,
        slow_lr_control_lr=args.slow_lr_control_lr,
        slow_lr_control_cost=args.slow_lr_control_cost,
        replay_strata=args.replay_strata,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    profile_key = jr.fold_in(root, 999)
    baseline_profile = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    static_profile, static_state = init_advantage_params_state(
        profile_key,
        block=block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_logit=args.gate_init_logit,
        gate_mode="scalar",
    )
    learned_profile, learned_state = init_learned_budget_params_state(
        profile_key,
        block=block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_logit=args.gate_init_logit,
        budget_init=args.budget_init,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        gate_l2_init=args.gate_l2,
        gate_l2_min=args.gate_l2_min,
        gate_l2_max=args.gate_l2_max,
        slow_lr_multiplier_init=1.0,
        slow_lr_multiplier_min=args.slow_lr_multiplier_min,
        slow_lr_multiplier_max=args.slow_lr_multiplier_max,
    )
    replay_state_elements = args.replay_size * args.block_size + 2 * args.replay_size + 2
    replay_state_bytes = 4 * replay_state_elements
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile),
            "trainable_bytes": count_array_bytes(baseline_profile),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "static_post_ffn_memory": {
            "trainable_params": count_array_elements(static_profile),
            "trainable_bytes": count_array_bytes(static_profile),
            "state_elements": count_array_elements(static_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(static_state, include_int=True)
            + replay_state_bytes,
        },
        "stratified_budget_post_ffn_memory": {
            "trainable_params": count_array_elements(learned_profile),
            "trainable_bytes": count_array_bytes(learned_profile),
            "state_elements": count_array_elements(learned_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(learned_state, include_int=True)
            + replay_state_bytes,
        },
    }

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        run_key = jr.fold_in(root, seed_idx)
        param_key, offset_key = jr.split(run_key, 2)
        max_offset = max(1, int(train_tokens.shape[0]) - args.block_size - args.steps - 1)
        train_offset = int(jr.randint(offset_key, (), 0, max_offset))
        contexts, labels = make_examples(
            train_tokens,
            steps=args.steps,
            block_size=args.block_size,
            offset=train_offset,
        )
        eval_contexts, eval_labels = make_examples(
            eval_tokens,
            steps=args.eval_steps,
            block_size=args.block_size,
            offset=seed_idx * args.eval_steps,
        )

        baseline_params = init_transformer_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
        )
        method_start = time.perf_counter()
        final_baseline, baseline_metrics = run_baseline_transformer(
            baseline_params,
            contexts,
            labels,
            step_size=args.baseline_lr,
            grad_clip=args.grad_clip,
        )
        train_s = time.perf_counter() - method_start
        summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_transformer(final_baseline, eval_contexts, eval_labels),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {"seed": seed_idx, "method": "baseline_ffn_transformer", "summary": summary}
        )
        print(
            f"seed={seed_idx} baseline: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, train_s={train_s:.2f}"
        )

        static_params, static_state = init_advantage_params_state(
            param_key,
            block=block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            gate_init_logit=args.gate_init_logit,
            gate_mode="scalar",
        )
        method_start = time.perf_counter()
        _final_params, static_final_state, _metrics, summary = static_summary(
            block,
            static_params,
            static_state,
            contexts,
            labels,
            eval_contexts,
            eval_labels,
            args,
            final_window,
        )
        static_final_state.step_count.block_until_ready()
        train_s = time.perf_counter() - method_start
        summary.update({"train_s": train_s, "train_steps_per_s": args.steps / train_s})
        records.append({"seed": seed_idx, "method": "static_post_ffn_memory", "summary": summary})
        print(
            f"seed={seed_idx} static: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, "
            f"gate={summary['final_window_gate']:.3f}, train_s={train_s:.2f}"
        )

        learned_params, learned_state = init_learned_budget_params_state(
            param_key,
            block=block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            gate_init_logit=args.gate_init_logit,
            budget_init=args.budget_init,
            budget_min=args.budget_min,
            budget_max=args.budget_max,
            gate_l2_init=args.gate_l2,
            gate_l2_min=args.gate_l2_min,
            gate_l2_max=args.gate_l2_max,
            slow_lr_multiplier_init=1.0,
            slow_lr_multiplier_min=args.slow_lr_multiplier_min,
            slow_lr_multiplier_max=args.slow_lr_multiplier_max,
        )
        method_start = time.perf_counter()
        learned_final_params, learned_final_state, metrics = (
            run_learned_budget_memory_transformer(
                block,
                learned_params,
                learned_state,
                contexts,
                labels,
                placement=args.placement,
                fast_lr=args.fast_lr,
                slow_lr=args.slow_lr,
                grad_clip=args.grad_clip,
                gate_lr=args.gate_lr,
                gate_decay=args.gate_decay,
                advantage_margin=args.advantage_margin,
                gate_l2=args.gate_l2,
                gate_init_logit=args.gate_init_logit,
                replay_size=args.replay_size,
                train_loss_mode=args.train_loss_mode,
                memory_loss_weight=args.memory_loss_weight,
                reset_mode=args.reset_mode,
                budget_min=args.budget_min,
                budget_max=args.budget_max,
                budget_lr=args.budget_lr,
                budget_ema_decay=args.budget_ema_decay,
                budget_target_utilization=args.budget_target_utilization,
                budget_cost=args.budget_cost,
                budget_pressure_floor=args.budget_pressure_floor,
                budget_advantage_floor=args.budget_advantage_floor,
                gate_l2_min=args.gate_l2_min,
                gate_l2_max=args.gate_l2_max,
                gate_l2_lr=args.gate_l2_lr,
                gate_l2_pressure_weight=args.gate_l2_pressure_weight,
                slow_lr_multiplier_min=args.slow_lr_multiplier_min,
                slow_lr_multiplier_max=args.slow_lr_multiplier_max,
                slow_lr_control_lr=args.slow_lr_control_lr,
                slow_lr_control_cost=args.slow_lr_control_cost,
            )
        )
        learned_final_state.step_count.block_until_ready()
        train_s = time.perf_counter() - method_start
        summary = {
            **summarize_learned_budget(metrics, final_window),
            **eval_advantage_memory_transformer(
                block,
                learned_final_params,
                cast(AdvantageMemoryState, learned_final_state),
                eval_contexts,
                eval_labels,
                placement=args.placement,
            ),
            **eval_advantage_fast_only(
                block,
                learned_final_params,
                cast(AdvantageMemoryState, learned_final_state),
                eval_contexts,
                eval_labels,
                placement=args.placement,
            ),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "stratified_budget_post_ffn_memory",
                "summary": summary,
            }
        )
        print(
            f"seed={seed_idx} stratified: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, "
            f"gate={summary['final_window_gate']:.3f}, "
            f"budget={summary['final_budget']:.3f}, "
            f"gate_l2={summary['final_learned_gate_l2']:.3f}, "
            f"slow_x={summary['final_slow_lr_multiplier']:.3f}, train_s={train_s:.2f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "profiles": profiles,
        "prototype_block": block.to_config(),
        "elapsed_s": time.perf_counter() - start,
        "records": records,
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

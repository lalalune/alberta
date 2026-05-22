#!/usr/bin/env python3
"""Throughput probe for the Step 2 replay-capped transformer memory runner.

The research runner defines JIT scans inside per-method Python functions.  This
benchmark keeps equivalent scan bodies but hoists each JIT once, so timings
separate compile+first-run cost from steady-state production-style throughput.
It also includes an exact fused center-slot path and a cached-replay ablation.
Fusing center-slot selection is required to match the current runner output;
cached replay intentionally changes the science contract by using stale replay
features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

STEP2_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "The Alberta Plan" / "Step2"
)
sys.path.insert(0, str(STEP2_EXAMPLE_DIR))

from step2_tiny_shakespeare_advantage_memory_transformer import (  # type: ignore[import-not-found] # noqa: E402
    AdvantageMemoryState,
    eval_advantage_fast_only,
    eval_advantage_memory_transformer,
    init_advantage_params_state,
    model_parts,
    reset_proto_row,
    sgd_step_decoupled,
    summarize_advantage,
)
from step2_tiny_shakespeare_proto_basis_transformer import (  # type: ignore[import-not-found] # noqa: E402
    make_proto_block,
    select_center_slot,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (  # type: ignore[import-not-found] # noqa: E402
    clip_grads,
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    encode_text,
    ensure_tiny_shakespeare,
    eval_transformer,
    ffn_transform,
    init_transformer_params,
    make_examples,
    sgd_step,
    summarize_online,
    transformer_logits,
)

from alberta_framework.core.prototype_basis import (  # noqa: E402
    PrototypeBasisBlock,
)


@dataclass(frozen=True)
class BenchConfig:
    """Serializable benchmark configuration."""

    steps: int
    repeats: int
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
    gate_init_logit: float
    gate_lr: float
    gate_decay: float
    gate_max: float
    advantage_margin: float
    gate_l2: float
    gate_mode: str
    train_loss_mode: str
    memory_loss_weight: float
    reset_mode: str
    replay_size: int
    seed: int
    data_path: str
    output_dir: str


@dataclass(frozen=True)
class Variant:
    """One throughput variant."""

    name: str
    kind: str
    placement: str = "post_ffn"
    gate_objective: str = "current"
    fuse_center_slot: bool = False
    cache_replay_features: bool = False
    behavior_contract: str = "exact"
    center_update_path: str = "separate_slot_then_update"
    replay_feature_source: str = "none"
    manifest_note: str = ""


def _block_until_ready(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        leaves[0].block_until_ready()


def _tree_exact_match_summary(left: Any, right: Any) -> dict[str, Any]:
    left_leaves, left_treedef = jax.tree_util.tree_flatten(left)
    right_leaves, right_treedef = jax.tree_util.tree_flatten(right)
    if left_treedef != right_treedef:
        raise AssertionError("tree structures differ")
    max_abs_diff = 0.0
    mismatched_leaves: list[int] = []
    for idx, (left_leaf, right_leaf) in enumerate(
        zip(left_leaves, right_leaves, strict=True)
    ):
        left_arr = np.asarray(left_leaf)
        right_arr = np.asarray(right_leaf)
        if left_arr.shape != right_arr.shape:
            raise AssertionError(
                f"leaf {idx} shape differs: {left_arr.shape} != {right_arr.shape}"
            )
        if left_arr.dtype != right_arr.dtype:
            raise AssertionError(
                f"leaf {idx} dtype differs: {left_arr.dtype} != {right_arr.dtype}"
            )
        if left_arr.size > 0 and np.issubdtype(left_arr.dtype, np.number):
            diff = np.max(
                np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))
            )
            max_abs_diff = max(max_abs_diff, float(diff))
        if not np.array_equal(left_arr, right_arr):
            mismatched_leaves.append(idx)
    return {
        "leaves_checked": len(left_leaves),
        "mismatched_leaves": mismatched_leaves,
        "max_abs_diff": max_abs_diff,
        "exact_equal": len(mismatched_leaves) == 0,
    }


def _mean_stderr(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    err = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, err


def _attention_forward_macs(block_size: int, d_model: int) -> int:
    return int(4 * block_size * d_model * d_model + 2 * block_size * block_size * d_model)


def _ffn_forward_macs(d_model: int, hidden: int) -> int:
    return int(2 * d_model * hidden)


def _readout_forward_macs(d_model: int, vocab_size: int) -> int:
    return int(d_model * vocab_size)


def _prototype_activation_macs(d_model: int, proto_count: int) -> int:
    return int(d_model * proto_count)


def _prototype_value_macs(d_model: int, proto_count: int) -> int:
    return int(d_model * proto_count)


def _forward_macs(
    variant: Variant,
    *,
    block_size: int,
    d_model: int,
    hidden: int,
    proto_count: int,
    vocab_size: int,
) -> int:
    attention = _attention_forward_macs(block_size, d_model)
    ffn = _ffn_forward_macs(d_model, hidden)
    readout = _readout_forward_macs(d_model, vocab_size)
    proto = _prototype_activation_macs(d_model, proto_count) + _prototype_value_macs(
        d_model,
        proto_count,
    )
    if variant.kind == "baseline":
        return attention + ffn + readout
    if variant.placement == "pre_ffn_kv":
        current = attention + 2 * ffn + proto + 2 * readout
    else:
        current = attention + ffn + proto + 2 * readout
    if variant.cache_replay_features:
        replay = proto + readout
        if variant.placement == "pre_ffn_kv":
            replay += ffn
        return current + replay
    if variant.gate_objective == "replay":
        return current * 2
    return current


def _prototype_distance_passes(variant: Variant) -> int:
    if variant.kind == "baseline":
        return 0
    current = 2 if variant.fuse_center_slot else 3
    replay = 1 if variant.gate_objective == "replay" else 0
    return current + replay


def _ffn_forward_paths(variant: Variant) -> int:
    if variant.kind == "baseline":
        return 1
    current = 2 if variant.placement == "pre_ffn_kv" else 1
    if variant.cache_replay_features:
        replay = 1 if variant.placement == "pre_ffn_kv" else 0
        return current + replay
    replay = current if variant.gate_objective == "replay" else 0
    return current + replay


def _readout_paths(variant: Variant) -> int:
    if variant.kind == "baseline":
        return 1
    current = 2
    if variant.cache_replay_features:
        return current + 1
    replay = 2 if variant.gate_objective == "replay" else 0
    return current + replay


def make_baseline_runner(step_size: float, grad_clip: float) -> Any:
    """Build a reusable JIT runner equivalent to the FFN baseline."""

    @jax.jit
    def run(params: dict[str, Any], contexts: jax.Array, labels: jax.Array) -> Any:
        def step(carry: dict[str, Any], inputs: tuple[jax.Array, jax.Array]) -> Any:
            context, label = inputs

            def loss_fn(candidate: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
                logits = transformer_logits(candidate, context)
                return cross_entropy_from_logits(logits, label), logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(carry)
            clipped = clip_grads(grads, grad_clip)
            new_params = sgd_step(carry, clipped, step_size)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            return new_params, jnp.stack([loss, acc])

        return jax.lax.scan(step, params, (contexts, labels))

    return run


def make_advantage_runner(
    block: PrototypeBasisBlock,
    variant: Variant,
    *,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    gate_max: float,
    advantage_margin: float,
    gate_l2: float,
    gate_init_logit: float,
    replay_size: int,
    train_loss_mode: str,
    memory_loss_weight: float,
    reset_mode: str,
) -> Any:
    """Build a reusable JIT runner equivalent to the advantage-memory scan."""
    if variant.cache_replay_features:
        return make_advantage_cached_replay_runner(
            block,
            variant,
            fast_lr=fast_lr,
            slow_lr=slow_lr,
            grad_clip=grad_clip,
            gate_lr=gate_lr,
            gate_decay=gate_decay,
            gate_max=gate_max,
            advantage_margin=advantage_margin,
            gate_l2=gate_l2,
            gate_init_logit=gate_init_logit,
            replay_size=replay_size,
            train_loss_mode=train_loss_mode,
            memory_loss_weight=memory_loss_weight,
            reset_mode=reset_mode,
        )
    replay_capacity = max(1, replay_size)

    @jax.jit
    def run(
        params: dict[str, Any],
        state: AdvantageMemoryState,
        contexts: jax.Array,
        labels: jax.Array,
    ) -> Any:
        def step(
            carry: tuple[Any, ...],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[Any, ...], jax.Array]:
            params, state, replay_contexts, replay_labels, replay_count, replay_index = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[
                jax.Array,
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                base_logits, memory_logits, basis_input, activations, gate = model_parts(
                    block,
                    candidate,
                    state,
                    context,
                    placement=variant.placement,
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                memory_loss = cross_entropy_from_logits(memory_logits, label)
                if train_loss_mode == "memory":
                    train_loss = memory_loss
                else:
                    weight = jnp.asarray(memory_loss_weight, dtype=jnp.float32)
                    train_loss = (1.0 - weight) * base_loss + weight * memory_loss
                return train_loss, (
                    base_loss,
                    memory_loss,
                    memory_logits,
                    basis_input,
                    activations,
                    gate,
                )

            (
                _train_loss,
                (base_loss, memory_loss, logits, basis_input, activations, gate),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            clipped = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled(
                params,
                clipped,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )
            advantage = base_loss - memory_loss
            if variant.fuse_center_slot:
                new_proto_state, center_metrics, slot, novel = block.update_centers_with_slot(
                    state.proto_state,
                    basis_input,
                )
            else:
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
            if variant.gate_objective == "replay":
                replay_slot = jnp.mod(replay_index, replay_capacity)
                replay_state = AdvantageMemoryState(
                    proto_state=new_proto_state,
                    gate_logit=state.gate_logit,
                    advantage_ema=state.advantage_ema,
                    init_value=state.init_value,
                    step_count=state.step_count,
                )
                replay_base, replay_memory, _, _, _ = model_parts(
                    block,
                    new_params,
                    replay_state,
                    replay_contexts[replay_slot],
                    placement=variant.placement,
                )
                replay_advantage = cross_entropy_from_logits(
                    replay_base,
                    replay_labels[replay_slot],
                ) - cross_entropy_from_logits(replay_memory, replay_labels[replay_slot])
                gate_update_advantage = jnp.where(
                    replay_count > 0,
                    replay_advantage,
                    advantage,
                )
            else:
                gate_update_advantage = advantage
            gate_signal = gate_update_advantage - jnp.asarray(
                advantage_margin,
                dtype=jnp.float32,
            )
            if state.gate_logit.ndim == 0:
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                new_gate_logit = (
                    jnp.asarray(gate_decay, dtype=jnp.float32) * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
            else:
                credit = activations / jnp.maximum(jnp.max(activations), 1e-6)
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                active_decay = 1.0 - credit * (
                    1.0 - jnp.asarray(gate_decay, dtype=jnp.float32)
                )
                new_gate_logit = (
                    active_decay * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * credit
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
                slots = jnp.arange(state.gate_logit.shape[0], dtype=slot.dtype)
                new_gate_logit = jnp.where(
                    (slots == slot) & novel,
                    jnp.asarray(gate_init_logit, dtype=jnp.float32),
                    new_gate_logit,
                )
            max_logit = jnp.log(
                jnp.asarray(gate_max, dtype=jnp.float32)
                / jnp.maximum(1.0 - jnp.asarray(gate_max, dtype=jnp.float32), 1e-6)
            )
            new_gate_logit = jnp.clip(new_gate_logit, -8.0, jnp.minimum(max_logit, 8.0))
            residual = block.transform(new_params["proto"], activations)
            improved = advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * residual,
                state.init_value,
            )
            new_state = AdvantageMemoryState(
                proto_state=new_proto_state,
                gate_logit=new_gate_logit,
                advantage_ema=0.99 * state.advantage_ema + 0.01 * advantage,
                init_value=init_value,
                step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
            )
            write_slot = jnp.mod(replay_index, replay_capacity)
            new_replay_contexts = replay_contexts.at[write_slot].set(context)
            new_replay_labels = replay_labels.at[write_slot].set(label)
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
                ]
            )
            return (
                new_params,
                new_state,
                new_replay_contexts,
                new_replay_labels,
                new_replay_count,
                new_replay_index,
            ), metrics

        replay_contexts = jnp.zeros(
            (replay_capacity, contexts.shape[1]),
            dtype=contexts.dtype,
        )
        replay_labels = jnp.zeros((replay_capacity,), dtype=labels.dtype)
        initial = (
            params,
            state,
            replay_contexts,
            replay_labels,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(step, initial, (contexts, labels))

    return run


def make_advantage_cached_replay_runner(
    block: PrototypeBasisBlock,
    variant: Variant,
    *,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    gate_max: float,
    advantage_margin: float,
    gate_l2: float,
    gate_init_logit: float,
    replay_size: int,
    train_loss_mode: str,
    memory_loss_weight: float,
    reset_mode: str,
) -> Any:
    """Build an experimental replay runner with stale cached replay features."""
    if variant.gate_objective != "replay":
        raise ValueError("cached replay requires gate_objective='replay'")
    replay_capacity = max(1, replay_size)

    @jax.jit
    def run(
        params: dict[str, Any],
        state: AdvantageMemoryState,
        contexts: jax.Array,
        labels: jax.Array,
    ) -> Any:
        def memory_logits_from_basis(
            candidate: dict[str, Any],
            memory_state: AdvantageMemoryState,
            basis_input: jax.Array,
        ) -> jax.Array:
            replay_activations = block.activations(memory_state.proto_state, basis_input)
            replay_gate = jax.nn.sigmoid(memory_state.gate_logit)
            if memory_state.gate_logit.ndim == 0:
                residual = replay_gate * block.transform(candidate["proto"], replay_activations)
            else:
                proto_params = candidate["proto"]
                gated_activations = replay_activations * replay_gate
                bias_scale = jnp.sum(gated_activations)
                residual = (
                    gated_activations @ proto_params.values
                    + bias_scale * proto_params.bias
                )
            if variant.placement == "pre_ffn_kv":
                memory_hidden = ffn_transform(candidate["ffn"], basis_input + residual)
            else:
                memory_hidden = basis_input + residual
            readout = candidate["readout"]
            logits = memory_hidden @ readout["w"] + readout["b"]
            return cast(jax.Array, logits)

        def step(
            carry: tuple[Any, ...],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[Any, ...], jax.Array]:
            (
                params,
                state,
                replay_basis_inputs,
                replay_base_losses,
                replay_labels,
                replay_count,
                replay_index,
            ) = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[
                jax.Array,
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                base_logits, memory_logits, basis_input, activations, gate = model_parts(
                    block,
                    candidate,
                    state,
                    context,
                    placement=variant.placement,
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                memory_loss = cross_entropy_from_logits(memory_logits, label)
                if train_loss_mode == "memory":
                    train_loss = memory_loss
                else:
                    weight = jnp.asarray(memory_loss_weight, dtype=jnp.float32)
                    train_loss = (1.0 - weight) * base_loss + weight * memory_loss
                return train_loss, (
                    base_loss,
                    memory_loss,
                    memory_logits,
                    basis_input,
                    activations,
                    gate,
                )

            (
                _train_loss,
                (base_loss, memory_loss, logits, basis_input, activations, gate),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            clipped = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled(
                params,
                clipped,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )
            advantage = base_loss - memory_loss
            if variant.fuse_center_slot:
                new_proto_state, center_metrics, slot, novel = block.update_centers_with_slot(
                    state.proto_state,
                    basis_input,
                )
            else:
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
            replay_slot = jnp.mod(replay_index, replay_capacity)
            replay_state = AdvantageMemoryState(
                proto_state=new_proto_state,
                gate_logit=state.gate_logit,
                advantage_ema=state.advantage_ema,
                init_value=state.init_value,
                step_count=state.step_count,
            )
            replay_memory_logits = memory_logits_from_basis(
                new_params,
                replay_state,
                replay_basis_inputs[replay_slot],
            )
            replay_advantage = replay_base_losses[replay_slot] - cross_entropy_from_logits(
                replay_memory_logits,
                replay_labels[replay_slot],
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
            if state.gate_logit.ndim == 0:
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                new_gate_logit = (
                    jnp.asarray(gate_decay, dtype=jnp.float32) * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
            else:
                credit = activations / jnp.maximum(jnp.max(activations), 1e-6)
                gate_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * gate
                active_decay = 1.0 - credit * (
                    1.0 - jnp.asarray(gate_decay, dtype=jnp.float32)
                )
                new_gate_logit = (
                    active_decay * state.gate_logit
                    + jnp.asarray(gate_lr, dtype=jnp.float32)
                    * credit
                    * jnp.clip(gate_signal, -1.0, 1.0)
                )
                slots = jnp.arange(state.gate_logit.shape[0], dtype=slot.dtype)
                new_gate_logit = jnp.where(
                    (slots == slot) & novel,
                    jnp.asarray(gate_init_logit, dtype=jnp.float32),
                    new_gate_logit,
                )
            max_logit = jnp.log(
                jnp.asarray(gate_max, dtype=jnp.float32)
                / jnp.maximum(1.0 - jnp.asarray(gate_max, dtype=jnp.float32), 1e-6)
            )
            new_gate_logit = jnp.clip(new_gate_logit, -8.0, jnp.minimum(max_logit, 8.0))
            residual = block.transform(new_params["proto"], activations)
            improved = advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * residual,
                state.init_value,
            )
            new_state = AdvantageMemoryState(
                proto_state=new_proto_state,
                gate_logit=new_gate_logit,
                advantage_ema=0.99 * state.advantage_ema + 0.01 * advantage,
                init_value=init_value,
                step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
            )
            write_slot = jnp.mod(replay_index, replay_capacity)
            new_replay_basis_inputs = replay_basis_inputs.at[write_slot].set(basis_input)
            new_replay_base_losses = replay_base_losses.at[write_slot].set(base_loss)
            new_replay_labels = replay_labels.at[write_slot].set(label)
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
                ]
            )
            return (
                new_params,
                new_state,
                new_replay_basis_inputs,
                new_replay_base_losses,
                new_replay_labels,
                new_replay_count,
                new_replay_index,
            ), metrics

        replay_basis_inputs = jnp.zeros(
            (replay_capacity, block.config.input_dim),
            dtype=jnp.float32,
        )
        replay_base_losses = jnp.zeros((replay_capacity,), dtype=jnp.float32)
        replay_labels = jnp.zeros((replay_capacity,), dtype=labels.dtype)
        initial = (
            params,
            state,
            replay_basis_inputs,
            replay_base_losses,
            replay_labels,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(step, initial, (contexts, labels))

    return run


def _replay_state_profile(
    replay_size: int,
    block_size: int,
    d_model: int,
    *,
    use_replay: bool,
    cache_replay_features: bool,
) -> tuple[int, int]:
    if not use_replay:
        return 0, 0
    if cache_replay_features:
        elements = replay_size * d_model + replay_size + replay_size + 2
    else:
        elements = replay_size * block_size + replay_size + 2
    return elements, elements * 4


def _state_profile(
    state: AdvantageMemoryState,
    *,
    replay_size: int,
    block_size: int,
    d_model: int,
    use_replay: bool,
    cache_replay_features: bool,
) -> tuple[int, int]:
    replay_elements, replay_bytes = _replay_state_profile(
        replay_size,
        block_size,
        d_model,
        use_replay=use_replay,
        cache_replay_features=cache_replay_features,
    )
    return (
        count_array_elements(state, include_int=True) + replay_elements,
        count_array_bytes(state, include_int=True) + replay_bytes,
    )


def bench_variant(
    variant: Variant,
    *,
    block: PrototypeBasisBlock,
    contexts: jax.Array,
    labels: jax.Array,
    eval_contexts: jax.Array,
    eval_labels: jax.Array,
    param_key: jax.Array,
    vocab_size: int,
    config: BenchConfig,
) -> dict[str, Any]:
    """Benchmark one variant."""
    if variant.kind == "baseline":
        params = init_transformer_params(
            param_key,
            vocab_size=vocab_size,
            block_size=config.block_size,
            d_model=config.d_model,
            ffn_hidden=config.mlp_hidden,
        )
        runner = make_baseline_runner(config.baseline_lr, config.grad_clip)
        first_start = time.perf_counter()
        final_params, metrics = runner(params, contexts, labels)
        _block_until_ready((final_params, metrics))
        compile_plus_first_s = time.perf_counter() - first_start
        times: list[float] = []
        final_params = params
        metrics = jnp.zeros((config.steps, 2), dtype=jnp.float32)
        for _ in range(config.repeats):
            start = time.perf_counter()
            final_params, metrics = runner(params, contexts, labels)
            _block_until_ready((final_params, metrics))
            times.append(time.perf_counter() - start)
        summary = {
            **summarize_online(np.asarray(metrics), config.final_window),
            **eval_transformer(final_params, eval_contexts, eval_labels),
        }
        state_elements = 0
        state_bytes = 0
        trainable_params = count_array_elements(params)
        trainable_bytes = count_array_bytes(params)
    else:
        params, state = init_advantage_params_state(
            param_key,
            block=block,
            vocab_size=vocab_size,
            block_size=config.block_size,
            d_model=config.d_model,
            ffn_hidden=config.mlp_hidden,
            gate_init_logit=config.gate_init_logit,
            gate_mode=config.gate_mode,
        )
        runner = make_advantage_runner(
            block,
            variant,
            fast_lr=config.fast_lr,
            slow_lr=config.slow_lr,
            grad_clip=config.grad_clip,
            gate_lr=config.gate_lr,
            gate_decay=config.gate_decay,
            gate_max=config.gate_max,
            advantage_margin=config.advantage_margin,
            gate_l2=config.gate_l2,
            gate_init_logit=config.gate_init_logit,
            replay_size=config.replay_size,
            train_loss_mode=config.train_loss_mode,
            memory_loss_weight=config.memory_loss_weight,
            reset_mode=config.reset_mode,
        )
        first_start = time.perf_counter()
        first_final, metrics = runner(params, state, contexts, labels)
        _block_until_ready((first_final, metrics))
        compile_plus_first_s = time.perf_counter() - first_start
        times = []
        final_pack = first_final
        metrics = jnp.zeros((config.steps, 10), dtype=jnp.float32)
        for _ in range(config.repeats):
            start = time.perf_counter()
            final_pack, metrics = runner(params, state, contexts, labels)
            _block_until_ready((final_pack, metrics))
            times.append(time.perf_counter() - start)
        final_params = final_pack[0]
        final_state = final_pack[1]
        summary = {
            **summarize_advantage(np.asarray(metrics), config.final_window),
            **eval_advantage_memory_transformer(
                block,
                final_params,
                final_state,
                eval_contexts,
                eval_labels,
                placement=variant.placement,
            ),
            **eval_advantage_fast_only(
                block,
                final_params,
                final_state,
                eval_contexts,
                eval_labels,
                placement=variant.placement,
            ),
        }
        state_elements, state_bytes = _state_profile(
            state,
            replay_size=config.replay_size,
            block_size=config.block_size,
            d_model=config.d_model,
            use_replay=variant.gate_objective == "replay",
            cache_replay_features=variant.cache_replay_features,
        )
        trainable_params = count_array_elements(params)
        trainable_bytes = count_array_bytes(params)

    mean_s, stderr_s = _mean_stderr(times)
    return {
        **asdict(variant),
        "steps": config.steps,
        "compile_plus_first_s": compile_plus_first_s,
        "steady_mean_s": mean_s,
        "steady_stderr_s": stderr_s,
        "steady_steps_per_s": config.steps / mean_s,
        "trainable_params": trainable_params,
        "trainable_bytes": trainable_bytes,
        "state_elements": state_elements,
        "state_bytes": state_bytes,
        "prototype_distance_passes_per_step": _prototype_distance_passes(variant),
        "ffn_forward_paths_per_step": _ffn_forward_paths(variant),
        "readout_paths_per_step": _readout_paths(variant),
        "forward_macs_per_step": _forward_macs(
            variant,
            block_size=config.block_size,
            d_model=config.d_model,
            hidden=config.mlp_hidden,
            proto_count=config.proto_count,
            vocab_size=vocab_size,
        ),
        "summary": summary,
    }


def run_exact_fused_center_check(
    *,
    block: PrototypeBasisBlock,
    contexts: jax.Array,
    labels: jax.Array,
    param_key: jax.Array,
    vocab_size: int,
    config: BenchConfig,
) -> dict[str, Any]:
    """Assert that fused center-slot update preserves current runner output."""
    params, state = init_advantage_params_state(
        param_key,
        block=block,
        vocab_size=vocab_size,
        block_size=config.block_size,
        d_model=config.d_model,
        ffn_hidden=config.mlp_hidden,
        gate_init_logit=config.gate_init_logit,
        gate_mode=config.gate_mode,
    )
    reference = Variant(
        "exactness_reference_unfused_current_centers",
        "advantage",
        "post_ffn",
        "replay",
        False,
        False,
        "exact_current_replay",
        "separate_slot_then_update",
        "raw_context_latest_params",
        "Reference path matching the current research runner.",
    )
    candidate = Variant(
        "exactness_candidate_fused_center",
        "advantage",
        "post_ffn",
        "replay",
        True,
        False,
        "exact_current_replay",
        "fused_slot_and_update",
        "raw_context_latest_params",
        "Production path that must match the reference output exactly.",
    )
    runner_kwargs = {
        "fast_lr": config.fast_lr,
        "slow_lr": config.slow_lr,
        "grad_clip": config.grad_clip,
        "gate_lr": config.gate_lr,
        "gate_decay": config.gate_decay,
        "gate_max": config.gate_max,
        "advantage_margin": config.advantage_margin,
        "gate_l2": config.gate_l2,
        "gate_init_logit": config.gate_init_logit,
        "replay_size": config.replay_size,
        "train_loss_mode": config.train_loss_mode,
        "memory_loss_weight": config.memory_loss_weight,
        "reset_mode": config.reset_mode,
    }
    reference_runner = make_advantage_runner(block, reference, **runner_kwargs)
    candidate_runner = make_advantage_runner(block, candidate, **runner_kwargs)
    reference_pack, reference_metrics = reference_runner(params, state, contexts, labels)
    candidate_pack, candidate_metrics = candidate_runner(params, state, contexts, labels)
    _block_until_ready((reference_pack, reference_metrics, candidate_pack, candidate_metrics))
    pack_summary = _tree_exact_match_summary(reference_pack, candidate_pack)
    metrics_summary = _tree_exact_match_summary(reference_metrics, candidate_metrics)
    passed = bool(pack_summary["exact_equal"] and metrics_summary["exact_equal"])
    if not passed:
        raise AssertionError(
            "exact fused center update changed runner output: "
            f"pack={pack_summary}, metrics={metrics_summary}"
        )
    return {
        "name": "exact_fused_center_matches_current_output",
        "passed": passed,
        "reference_variant": asdict(reference),
        "candidate_variant": asdict(candidate),
        "steps_checked": int(contexts.shape[0]),
        "pack": pack_summary,
        "metrics": metrics_summary,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    rows = payload["results"]
    baseline = next(row for row in rows if row["name"] == "baseline_ffn")
    lines = [
        "# Step 2 Transformer Memory Throughput",
        "",
        "This benchmark profiles the replay-capped advantage-memory transformer "
        "candidate with reusable JIT scan functions. Steady timings exclude "
        "the compile+first run.",
        "",
        f"Device: `{payload['device']}`.",
        f"Steps: `{payload['config']['steps']}`. Repeats: `{payload['config']['repeats']}`.",
        f"Shape: block `{payload['config']['block_size']}`, d_model "
        f"`{payload['config']['d_model']}`, hidden `{payload['config']['mlp_hidden']}`, "
        f"prototypes `{payload['config']['proto_count']}`, vocab `{payload['vocab_size']}`.",
        "",
        "## Variant Manifest",
        "",
        "| Variant | Contract | Center update | Replay feature source | Note |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['behavior_contract']}` | "
            f"`{row['center_update_path']}` | `{row['replay_feature_source']}` | "
            f"{row['manifest_note']} |"
        )
    lines.extend(
        [
            "",
            "## Exactness Checks",
            "",
        ]
    )
    for check in payload.get("exactness_checks", []):
        lines.append(
            f"- `{check['name']}`: passed={check['passed']}, "
            f"steps={check['steps_checked']}, "
            f"pack max abs diff={check['pack']['max_abs_diff']:.1e}, "
            f"metrics max abs diff={check['metrics']['max_abs_diff']:.1e}."
        )
    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Variant | Compile+first s | Steady s | Steps/s | vs FFN | Trainable | State bytes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ratio = row["steady_steps_per_s"] / baseline["steady_steps_per_s"]
        lines.append(
            f"| `{row['name']}` | {row['compile_plus_first_s']:.4f} | "
            f"{row['steady_mean_s']:.4f} +/- {row['steady_stderr_s']:.4f} | "
            f"{row['steady_steps_per_s']:.1f} | {ratio:.3f}x | "
            f"{row['trainable_params']} | {row['state_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Step Work Model",
            "",
            "| Variant | Forward MACs | Prototype distance passes | FFN paths | Readout paths |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['forward_macs_per_step']} | "
            f"{row['prototype_distance_passes_per_step']} | "
            f"{row['ffn_forward_paths_per_step']} | {row['readout_paths_per_step']} |"
        )
    lines.extend(
        [
            "",
            "## Online Metrics",
            "",
            "| Variant | Final NLL | Eval NLL | Eval PPL | Gate | Advantage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        summary = row["summary"]
        gate = summary.get("final_window_gate", float("nan"))
        advantage = summary.get("final_window_advantage", float("nan"))
        lines.append(
            f"| `{row['name']}` | {summary['final_window_nll']:.4f} | "
            f"{summary['eval_nll']:.4f} | {summary['eval_perplexity']:.2f} | "
            f"{gate:.4f} | {advantage:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Bottleneck Notes",
            "",
            "- The unfused memory variants traverse prototype centers three times on the "
            "current token: activation, reset-slot selection, then center update.",
            "- Replay-gated variants add another full `model_parts` call on the ring-buffer "
            "sample, adding one prototype lookup plus another attention/FFN/readout "
            "forward outside the gradient path.",
            "- The exact fused-center variant is the production behavior path: it matches "
            "the current separate slot/update output but removes the standalone "
            "slot-selection center pass.",
            "- The cached-replay variant is a science-changing ablation: it stores replay "
            "`basis_input` and base loss, so it removes the second attention/FFN pass but "
            "uses stale fast features instead of exact latest-parameter replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--final-window", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--fast-lr", type=float, default=0.15)
    parser.add_argument("--slow-lr", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.0002)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--gate-init-logit", type=float, default=-2.0)
    parser.add_argument("--gate-lr", type=float, default=1.0)
    parser.add_argument("--gate-decay", type=float, default=0.995)
    parser.add_argument("--gate-max", type=float, default=0.9997)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--gate-l2", type=float, default=0.0)
    parser.add_argument("--gate-mode", choices=("scalar", "prototype"), default="scalar")
    parser.add_argument("--train-loss-mode", choices=("memory", "blend"), default="memory")
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-mode", choices=("none", "zero", "meta_ema"), default="meta_ema")
    parser.add_argument("--replay-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/step2_transformer_memory_throughput"),
    )
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.eval_steps) <= 0:
        raise ValueError("--steps, --repeats, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if min(args.d_model, args.mlp_hidden, args.proto_count) <= 0:
        raise ValueError("--d-model, --mlp-hidden, and --proto-count must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if min(args.baseline_lr, args.fast_lr, args.slow_lr, args.gate_lr) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 < args.proto_update_rate <= 1.0:
        raise ValueError("--proto-update-rate must be in (0, 1]")
    if args.proto_novelty_threshold < 0.0 or args.proto_bandwidth <= 0.0:
        raise ValueError("prototype threshold must be non-negative and bandwidth positive")
    if not 0.0 <= args.gate_decay <= 1.0:
        raise ValueError("--gate-decay must be in [0, 1]")
    if not 0.0 < args.gate_max < 1.0:
        raise ValueError("--gate-max must be in (0, 1)")
    if not 0.0 <= args.memory_loss_weight <= 1.0:
        raise ValueError("--memory-loss-weight must be in [0, 1]")
    if args.replay_size < 1:
        raise ValueError("--replay-size must be positive")
    return args


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    vocab_size = int(metadata["vocab_size"])
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    final_window = min(args.final_window, args.steps)
    config = BenchConfig(
        steps=args.steps,
        repeats=args.repeats,
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
        gate_init_logit=args.gate_init_logit,
        gate_lr=args.gate_lr,
        gate_decay=args.gate_decay,
        gate_max=args.gate_max,
        advantage_margin=args.advantage_margin,
        gate_l2=args.gate_l2,
        gate_mode=args.gate_mode,
        train_loss_mode=args.train_loss_mode,
        memory_loss_weight=args.memory_loss_weight,
        reset_mode=args.reset_mode,
        replay_size=args.replay_size,
        seed=args.seed,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
    )

    block_args = argparse.Namespace(
        d_model=args.d_model,
        proto_count=args.proto_count,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=False,
        proto_bandwidth_update_rate=0.1,
    )
    block = make_proto_block(block_args)
    root = jr.key(args.seed)
    param_key, offset_key = jr.split(root, 2)
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
        offset=args.eval_steps,
    )
    variants = [
        Variant(
            name="baseline_ffn",
            kind="baseline",
            behavior_contract="baseline",
            center_update_path="none",
            replay_feature_source="none",
            manifest_note="FFN-only baseline.",
        ),
        Variant(
            name="advantage_post_current_exact_reference",
            kind="advantage",
            placement="post_ffn",
            gate_objective="current",
            behavior_contract="exact_current_token",
            center_update_path="separate_slot_then_update",
            replay_feature_source="none",
            manifest_note="Current-token gate objective; no replay buffer.",
        ),
        Variant(
            name="advantage_post_replay_exact_reference",
            kind="advantage",
            placement="post_ffn",
            gate_objective="replay",
            behavior_contract="exact_current_replay",
            center_update_path="separate_slot_then_update",
            replay_feature_source="raw_context_latest_params",
            manifest_note="Reference behavior matching the current research runner.",
        ),
        Variant(
            name="advantage_post_replay_exact_fused_center",
            kind="advantage",
            placement="post_ffn",
            gate_objective="replay",
            fuse_center_slot=True,
            behavior_contract="exact_current_replay",
            center_update_path="fused_slot_and_update",
            replay_feature_source="raw_context_latest_params",
            manifest_note="Exact behavior path; checked against the reference output.",
        ),
        Variant(
            name="advantage_post_replay_cached_basis_ablation",
            kind="advantage",
            placement="post_ffn",
            gate_objective="replay",
            fuse_center_slot=True,
            cache_replay_features=True,
            behavior_contract="cached_replay_ablation_changes_behavior",
            center_update_path="fused_slot_and_update",
            replay_feature_source="stale_basis_input_and_base_loss",
            manifest_note="Science-changing ablation; not the exact production path.",
        ),
        Variant(
            name="advantage_pre_kv_replay_exact_reference",
            kind="advantage",
            placement="pre_ffn_kv",
            gate_objective="replay",
            behavior_contract="exact_current_replay",
            center_update_path="separate_slot_then_update",
            replay_feature_source="raw_context_latest_params",
            manifest_note="Pre-FFN KV placement reference with exact replay.",
        ),
    ]
    print("checking exact fused-center output match ...", flush=True)
    exactness_checks = [
        run_exact_fused_center_check(
            block=block,
            contexts=contexts,
            labels=labels,
            param_key=param_key,
            vocab_size=vocab_size,
            config=config,
        )
    ]
    print("  exact fused-center output match passed", flush=True)
    results = []
    for variant in variants:
        print(f"benchmarking {variant.name} ...", flush=True)
        row = bench_variant(
            variant,
            block=block,
            contexts=contexts,
            labels=labels,
            eval_contexts=eval_contexts,
            eval_labels=eval_labels,
            param_key=param_key,
            vocab_size=vocab_size,
            config=config,
        )
        print(
            f"  {row['steady_steps_per_s']:.1f} steps/s steady "
            f"(compile+first {row['compile_plus_first_s']:.3f}s)",
            flush=True,
        )
        results.append(row)

    payload = {
        "config": asdict(config),
        "device": str(jax.devices()[0]),
        "vocab_size": vocab_size,
        "prototype_block": block.to_config(),
        "exactness_checks": exactness_checks,
        "results": results,
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

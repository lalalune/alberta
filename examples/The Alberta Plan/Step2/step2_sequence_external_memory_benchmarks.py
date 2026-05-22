#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""External sequence probes for the replay-capped Step 2 memory transformer.

This runner deliberately avoids another Tiny Shakespeare-only validation pass.
It compares the existing FFN transformer baseline with the current replay-capped
post-FFN advantage memory candidate and local gate/replacement rescue variants
on small locally generated continual sequence probes:

* ``block_shift_markov``: a contiguous next-token stream whose transition
  grammar changes in repeating blocks.
* ``delayed_copy``: an algorithmic copy task whose queried lag changes in
  blocks while the context contains distractor payload tokens.
* ``sparse_kv_recall``: one-shot random key/value bindings with distractor
  interference and held-out queries against the final binding table.
* ``local_text_motif``: a deterministic non-Shakespeare text-like stream used
  when no cached public corpus is present.

The goal is bounded evidence, not benchmark saturation.  Defaults run three
small seeds and finish quickly on CPU after JAX compilation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from step2_tiny_shakespeare_advantage_memory_transformer import (
    eval_advantage_fast_only,
    eval_advantage_memory_transformer,
    init_advantage_params_state,
    run_advantage_memory_transformer,
    summarize_advantage,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    causal_attention_sequence,
    clip_grads,
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    eval_transformer,
    ffn_transform,
    init_transformer_params,
    run_baseline_transformer,
    stderr,
    summarize_online,
)

from alberta_framework.core.prototype_basis import (
    PrototypeBasisBlock,
    PrototypeBasisConfig,
    PrototypeBasisParams,
    PrototypeBasisState,
)

BenchmarkName = Literal[
    "block_shift_markov",
    "delayed_copy",
    "sparse_kv_recall",
    "local_text_motif",
]
MethodName = Literal["baseline_ffn_transformer", "replay_capped_post_ffn_memory"]
MemoryVariantName = Literal[
    "static_replay",
    "rescue_close_gate",
    "rescue_proto_utility",
    "rescue_split_update_utility",
]

ALL_MEMORY_VARIANTS: tuple[MemoryVariantName, ...] = (
    "static_replay",
    "rescue_close_gate",
    "rescue_proto_utility",
    "rescue_split_update_utility",
)

BASELINE_METHOD = "baseline_ffn_transformer"
STATIC_REPLAY_METHOD = "replay_capped_post_ffn_memory"

VARIANT_METHODS: dict[MemoryVariantName, str] = {
    "static_replay": STATIC_REPLAY_METHOD,
    "rescue_close_gate": "rescue_close_gate_memory",
    "rescue_proto_utility": "rescue_proto_utility_memory",
    "rescue_split_update_utility": "rescue_split_update_utility_memory",
}

METHOD_LABELS: dict[str, str] = {
    BASELINE_METHOD: "FFN",
    STATIC_REPLAY_METHOD: "Static replay",
    "rescue_close_gate_memory": "Close-gate rescue",
    "rescue_proto_utility_memory": "Prototype utility rescue",
    "rescue_split_update_utility_memory": "Split-gate utility rescue",
}

ALL_BENCHMARKS: tuple[BenchmarkName, ...] = (
    "block_shift_markov",
    "delayed_copy",
    "sparse_kv_recall",
    "local_text_motif",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured in result artifacts."""

    benchmarks: list[str]
    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    proto_count: int
    eval_steps: int
    final_window: int
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
    gate_max: float
    gate_l2: float
    replay_size: int
    memory_variants: list[str]
    negative_gate_multiplier: float
    utility_decay: float
    utility_replace_threshold: float
    utility_novelty_bonus: float
    min_update_gate: float
    seed: int
    output_dir: str


@dataclass(frozen=True)
class SequenceBenchmark:
    """Concrete train/eval tensors for one sequence probe."""

    name: BenchmarkName
    description: str
    train_contexts: jax.Array
    train_labels: jax.Array
    eval_contexts: jax.Array
    eval_labels: jax.Array
    vocab_size: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MechanismSpec:
    """Local external-memory rescue mechanism settings."""

    variant: MemoryVariantName
    gate_mode: Literal["scalar", "prototype"]
    close_negative: bool
    use_utility_replacement: bool
    split_update_gate: bool
    reset_mode: Literal["zero", "meta_ema", "none"]


@chex.dataclass(frozen=True)
class RescueMemoryState:
    """Advantage memory state with per-prototype utility for replacement."""

    proto_state: PrototypeBasisState
    gate_logit: jax.Array
    advantage_ema: jax.Array
    utility_ema: jax.Array
    init_value: jax.Array
    step_count: jax.Array


def make_memory_block(args: argparse.Namespace) -> PrototypeBasisBlock:
    """Create the replay-memory residual block used by the candidate."""
    return PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=args.d_model,
            output_dim=args.d_model,
            n_prototypes=args.proto_count,
            step_size=0.0,
            update_rate=args.proto_update_rate,
            novelty_threshold=args.proto_novelty_threshold,
            bandwidth=args.proto_bandwidth,
            adaptive_bandwidth=args.proto_adaptive_bandwidth,
            bandwidth_update_rate=args.proto_bandwidth_update_rate,
            min_bandwidth=1e-4,
            max_bandwidth=10.0,
            normalize_activations=True,
        )
    )


def mechanism_spec(variant: MemoryVariantName) -> MechanismSpec:
    """Return the concrete rescue mechanism for a variant name."""
    if variant == "rescue_close_gate":
        return MechanismSpec(
            variant=variant,
            gate_mode="scalar",
            close_negative=True,
            use_utility_replacement=False,
            split_update_gate=False,
            reset_mode="zero",
        )
    if variant == "rescue_proto_utility":
        return MechanismSpec(
            variant=variant,
            gate_mode="prototype",
            close_negative=True,
            use_utility_replacement=True,
            split_update_gate=False,
            reset_mode="zero",
        )
    if variant == "rescue_split_update_utility":
        return MechanismSpec(
            variant=variant,
            gate_mode="prototype",
            close_negative=True,
            use_utility_replacement=True,
            split_update_gate=True,
            reset_mode="zero",
        )
    msg = f"static replay has no local rescue mechanism: {variant}"
    raise ValueError(msg)


def init_rescue_params_state(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    gate_init_logit: float,
    gate_mode: str,
) -> tuple[dict[str, Any], RescueMemoryState]:
    """Initialize params and utility-augmented rescue memory state."""
    params, base_state = init_advantage_params_state(
        key,
        block=block,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        gate_init_logit=gate_init_logit,
        gate_mode=gate_mode,
    )
    state = RescueMemoryState(
        proto_state=base_state.proto_state,
        gate_logit=base_state.gate_logit,
        advantage_ema=base_state.advantage_ema,
        utility_ema=jnp.zeros((block.config.n_prototypes,), dtype=jnp.float32),
        init_value=base_state.init_value,
        step_count=base_state.step_count,
    )
    return params, state


def gate_logit_from_gate(gate: jax.Array) -> jax.Array:
    """Map a gate probability to a numerically safe logit."""
    clipped = jnp.clip(gate, 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped / (1.0 - clipped))


def rescue_model_parts(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: RescueMemoryState,
    context: jax.Array,
    *,
    placement: str,
    gate_override: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return model parts with an optional gate override for update-only use."""
    attn_hidden = causal_attention_sequence(params["attn"], context)[-1]
    if placement == "pre_ffn_kv":
        basis_input = attn_hidden
        base_hidden = ffn_transform(params["ffn"], basis_input)
    else:
        basis_input = ffn_transform(params["ffn"], attn_hidden)
        base_hidden = basis_input
    activations = block.activations(state.proto_state, basis_input)
    gate = jax.nn.sigmoid(state.gate_logit) if gate_override is None else gate_override
    if gate.ndim == 0:
        residual = gate * block.transform(params["proto"], activations)
    else:
        proto_params = params["proto"]
        gated_activations = activations * gate
        bias_scale = jnp.sum(gated_activations)
        residual = gated_activations @ proto_params.values + bias_scale * proto_params.bias
    if placement == "pre_ffn_kv":
        memory_hidden = ffn_transform(params["ffn"], basis_input + residual)
    else:
        memory_hidden = basis_input + residual
    readout = params["readout"]
    base_logits = base_hidden @ readout["w"] + readout["b"]
    memory_logits = memory_hidden @ readout["w"] + readout["b"]
    return base_logits, memory_logits, basis_input, activations, gate


def rescue_prediction_advantage(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: RescueMemoryState,
    context: jax.Array,
    label: jax.Array,
    *,
    placement: str,
    gate_override: jax.Array | None = None,
) -> jax.Array:
    """Return fast-minus-memory loss under a chosen prediction gate."""
    base_logits, memory_logits, _, _, _ = rescue_model_parts(
        block,
        params,
        state,
        context,
        placement=placement,
        gate_override=gate_override,
    )
    return cross_entropy_from_logits(base_logits, label) - cross_entropy_from_logits(
        memory_logits,
        label,
    )


def sgd_step_decoupled_local(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    fast_lr: float,
    slow_lr: float,
) -> dict[str, Any]:
    """Apply separate rates to fast transformer and slow memory values."""

    def step_tree(tree: Any, grad_tree: Any, lr: float) -> Any:
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, tree, grad_tree)

    return {
        "attn": step_tree(params["attn"], grads["attn"], fast_lr),
        "ffn": step_tree(params["ffn"], grads["ffn"], fast_lr),
        "readout": step_tree(params["readout"], grads["readout"], fast_lr),
        "proto": step_tree(params["proto"], grads["proto"], slow_lr),
    }


def reset_rescue_proto_row(
    params: dict[str, Any],
    state: RescueMemoryState,
    slot: jax.Array,
    novel: jax.Array,
    *,
    reset_mode: str,
) -> dict[str, Any]:
    """Reset replaced prototype value rows for the local rescue mechanisms."""
    if reset_mode == "none":
        return params
    proto_params = params["proto"]
    row = proto_params.values[slot]
    target = jnp.zeros_like(row) if reset_mode == "zero" else state.init_value
    new_row = jnp.where(novel, target, row)
    new_proto = PrototypeBasisParams(
        values=proto_params.values.at[slot].set(new_row),
        bias=proto_params.bias,
    )
    return {**params, "proto": new_proto}


def update_centers_with_utility(
    block: PrototypeBasisBlock,
    state: PrototypeBasisState,
    observation: jax.Array,
    utility: jax.Array,
    *,
    use_utility_replacement: bool,
    utility_replace_threshold: float,
    utility_novelty_bonus: float,
) -> tuple[PrototypeBasisState, jax.Array, jax.Array, jax.Array]:
    """Update prototype centers with optional utility-governed replacement."""
    used = state.counts > 0.0
    has_used = jnp.any(used)
    has_empty = jnp.any(~used)
    distances = jnp.mean((state.centers - observation[None, :]) ** 2, axis=1)
    used_distances = jnp.where(used, distances, jnp.inf)
    nearest_slot = jnp.argmin(used_distances)
    nearest_distance = used_distances[nearest_slot]
    empty_slot = jnp.argmax((~used).astype(jnp.int32))
    count_replacement_slot = block._replacement_slot(state)
    utility_replacement_slot = jnp.argmin(jnp.where(used, utility, jnp.inf))
    use_utility = jnp.asarray(use_utility_replacement)
    replacement_slot = jnp.where(
        use_utility,
        utility_replacement_slot,
        count_replacement_slot,
    )
    nearest_utility = jnp.where(has_used, utility[nearest_slot], 0.0)
    effective_threshold = jnp.asarray(
        block.config.novelty_threshold,
        dtype=jnp.float32,
    ) + jnp.asarray(utility_novelty_bonus, dtype=jnp.float32) * jnp.maximum(
        nearest_utility,
        0.0,
    )
    novel_request = (~has_used) | (nearest_distance > effective_threshold)
    replacement_allowed = (~use_utility) | (
        utility[utility_replacement_slot]
        <= jnp.asarray(utility_replace_threshold, dtype=jnp.float32)
    )
    can_allocate = (~has_used) | has_empty | replacement_allowed
    novel = novel_request & can_allocate
    slot = jnp.where(
        ~has_used,
        jnp.array(0, dtype=nearest_slot.dtype),
        jnp.where(
            novel & has_empty,
            empty_slot,
            jnp.where(novel, replacement_slot, nearest_slot),
        ),
    )
    eta = jnp.asarray(block.config.update_rate, dtype=jnp.float32)
    old_center = state.centers[slot]
    new_center = jnp.where(
        novel,
        observation,
        old_center + eta * (observation - old_center),
    )
    old_bandwidth = state.bandwidths[slot]
    distance_for_bandwidth = jnp.maximum(nearest_distance, block.config.min_bandwidth)
    bandwidth_eta = jnp.asarray(block.config.bandwidth_update_rate, dtype=jnp.float32)
    adapted_bandwidth = old_bandwidth + bandwidth_eta * (
        distance_for_bandwidth - old_bandwidth
    )
    new_bandwidth = jnp.where(
        block.config.adaptive_bandwidth & (~novel),
        jnp.clip(
            adapted_bandwidth,
            block.config.min_bandwidth,
            block.config.max_bandwidth,
        ),
        jnp.asarray(block.config.bandwidth, dtype=jnp.float32),
    )
    new_count = jnp.where(novel, 1.0, state.counts[slot] + 1.0)
    new_state = PrototypeBasisState(
        centers=state.centers.at[slot].set(new_center),
        bandwidths=state.bandwidths.at[slot].set(new_bandwidth),
        counts=state.counts.at[slot].set(new_count),
        last_update=state.last_update.at[slot].set(state.step_count + 1),
        step_count=state.step_count + 1,
    )
    active_count = jnp.sum(new_state.counts > 0.0).astype(jnp.float32)
    replacement_blocked = novel_request & (~can_allocate)
    metrics = jnp.asarray(
        [
            active_count,
            novel.astype(jnp.float32),
            jnp.where(has_used, nearest_distance, jnp.array(0.0, dtype=jnp.float32)),
            replacement_blocked.astype(jnp.float32),
        ],
        dtype=jnp.float32,
    )
    return new_state, metrics, slot, novel


def run_rescue_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: RescueMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    mechanism: MechanismSpec,
    placement: str,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    gate_max: float,
    advantage_margin: float,
    gate_l2: float,
    gate_init_logit: float,
    gate_objective: str,
    replay_size: int,
    train_loss_mode: str,
    memory_loss_weight: float,
    negative_gate_multiplier: float,
    utility_decay: float,
    utility_replace_threshold: float,
    utility_novelty_bonus: float,
    min_update_gate: float,
) -> tuple[dict[str, Any], RescueMemoryState, np.ndarray]:
    """Train one local external-memory rescue variant."""
    replay_capacity = max(1, replay_size)

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: RescueMemoryState,
    ) -> tuple[tuple[Any, ...], jax.Array]:
        def step(
            carry: tuple[Any, ...],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[Any, ...], jax.Array]:
            params, state, replay_contexts, replay_labels, replay_count, replay_index = carry
            context, label = inputs
            pred_gate = jax.nn.sigmoid(state.gate_logit)
            update_gate = pred_gate
            if mechanism.split_update_gate:
                update_gate = jnp.maximum(
                    pred_gate,
                    jnp.asarray(min_update_gate, dtype=jnp.float32),
                )

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[
                jax.Array,
                tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                base_logits, memory_logits, basis_input, activations, gate = (
                    rescue_model_parts(
                        block,
                        candidate,
                        state,
                        context,
                        placement=placement,
                        gate_override=update_gate,
                    )
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                memory_loss = cross_entropy_from_logits(memory_logits, label)
                if train_loss_mode == "memory":
                    train_loss = memory_loss
                else:
                    weight = jnp.asarray(memory_loss_weight, dtype=jnp.float32)
                    train_loss = (1.0 - weight) * base_loss + weight * memory_loss
                return train_loss, (base_loss, memory_logits, basis_input, activations, gate)

            (_train_loss, (_, _, basis_input, activations, _)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(params)
            )
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled_local(
                params,
                grads,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )

            base_logits, pred_logits, _, _, pred_gate = rescue_model_parts(
                block,
                params,
                state,
                context,
                placement=placement,
            )
            base_loss = cross_entropy_from_logits(base_logits, label)
            memory_loss = cross_entropy_from_logits(pred_logits, label)
            advantage = base_loss - memory_loss

            new_proto_state, center_metrics, slot, novel = update_centers_with_utility(
                block,
                state.proto_state,
                basis_input,
                state.utility_ema,
                use_utility_replacement=mechanism.use_utility_replacement,
                utility_replace_threshold=utility_replace_threshold,
                utility_novelty_bonus=utility_novelty_bonus,
            )
            new_params = reset_rescue_proto_row(
                new_params,
                state,
                slot,
                novel,
                reset_mode=mechanism.reset_mode,
            )

            replay_slot = jnp.mod(replay_index, replay_capacity)
            gate_probe = pred_gate
            if mechanism.split_update_gate:
                gate_probe = update_gate
            probe_state = RescueMemoryState(
                proto_state=new_proto_state,
                gate_logit=gate_logit_from_gate(gate_probe),
                advantage_ema=state.advantage_ema,
                utility_ema=state.utility_ema,
                init_value=state.init_value,
                step_count=state.step_count,
            )
            if gate_objective == "replay":
                replay_advantage = rescue_prediction_advantage(
                    block,
                    new_params,
                    probe_state,
                    replay_contexts[replay_slot],
                    replay_labels[replay_slot],
                    placement=placement,
                )
                current_probe_advantage = rescue_prediction_advantage(
                    block,
                    new_params,
                    probe_state,
                    context,
                    label,
                    placement=placement,
                )
                gate_update_advantage = jnp.where(
                    replay_count > 0,
                    replay_advantage,
                    current_probe_advantage,
                )
            else:
                gate_update_advantage = rescue_prediction_advantage(
                    block,
                    new_params,
                    probe_state,
                    context,
                    label,
                    placement=placement,
                )

            gate_signal = gate_update_advantage - jnp.asarray(
                advantage_margin,
                dtype=jnp.float32,
            )
            if mechanism.close_negative:
                gate_signal = jnp.where(
                    gate_signal < 0.0,
                    gate_signal * jnp.asarray(negative_gate_multiplier, dtype=jnp.float32),
                    gate_signal,
                )
            gate_signal = jnp.clip(gate_signal, -1.0, 1.0)
            gate_decay_value = jnp.asarray(gate_decay, dtype=jnp.float32)
            gate_lr_value = jnp.asarray(gate_lr, dtype=jnp.float32)
            gate_init = jnp.asarray(gate_init_logit, dtype=jnp.float32)
            if state.gate_logit.ndim == 0:
                scalar_signal = gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * pred_gate
                decayed_logit = gate_init + gate_decay_value * (state.gate_logit - gate_init)
                new_gate_logit = decayed_logit + gate_lr_value * scalar_signal
            else:
                credit = activations / jnp.maximum(jnp.max(activations), 1e-6)
                per_proto_signal = credit * (
                    gate_signal - jnp.asarray(gate_l2, dtype=jnp.float32) * pred_gate
                )
                decayed_logit = gate_init + gate_decay_value * (state.gate_logit - gate_init)
                new_gate_logit = decayed_logit + gate_lr_value * jnp.clip(
                    per_proto_signal,
                    -1.0,
                    1.0,
                )
                slots = jnp.arange(state.gate_logit.shape[0], dtype=slot.dtype)
                new_gate_logit = jnp.where((slots == slot) & novel, gate_init, new_gate_logit)
            max_logit = jnp.log(
                jnp.asarray(gate_max, dtype=jnp.float32)
                / jnp.maximum(1.0 - jnp.asarray(gate_max, dtype=jnp.float32), 1e-6)
            )
            new_gate_logit = jnp.clip(new_gate_logit, -8.0, jnp.minimum(max_logit, 8.0))

            utility_credit = activations / jnp.maximum(jnp.sum(activations), 1e-6)
            new_utility = (
                jnp.asarray(utility_decay, dtype=jnp.float32) * state.utility_ema
                + utility_credit * gate_update_advantage
            )
            slots = jnp.arange(state.utility_ema.shape[0], dtype=slot.dtype)
            new_utility = jnp.where((slots == slot) & novel, 0.0, new_utility)

            residual = block.transform(new_params["proto"], activations)
            improved = advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * residual,
                state.init_value,
            )
            new_state = RescueMemoryState(
                proto_state=new_proto_state,
                gate_logit=new_gate_logit,
                advantage_ema=0.99 * state.advantage_ema + 0.01 * advantage,
                utility_ema=new_utility,
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
            acc = (jnp.argmax(pred_logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    memory_loss,
                    acc,
                    base_loss,
                    advantage,
                    jnp.mean(pred_gate),
                    jnp.mean(new_gate_logit),
                    center_metrics[0],
                    center_metrics[1],
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    gate_update_advantage,
                    jnp.mean(update_gate),
                    center_metrics[2],
                    jnp.mean(new_utility),
                    jnp.min(new_utility),
                    center_metrics[3],
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

    (final_params, final_state, *_), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def summarize_rescue(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize online metrics and rescue-specific diagnostics."""
    summary = summarize_advantage(metrics, final_window)
    window = metrics[-min(final_window, metrics.shape[0]) :]
    summary.update(
        {
            "final_window_update_gate": float(np.mean(window[:, 10])),
            "final_window_nearest_distance": float(np.mean(window[:, 11])),
            "final_window_utility_mean": float(np.mean(window[:, 12])),
            "final_window_utility_min": float(np.mean(window[:, 13])),
            "final_window_replacement_blocked_rate": float(np.mean(window[:, 14])),
        }
    )
    return summary


def _contexts_from_tokens(
    tokens: npt.NDArray[np.int32],
    steps: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    starts: npt.NDArray[np.int32] = np.arange(steps, dtype=np.int32)
    contexts: npt.NDArray[np.int32] = np.stack(
        [tokens[start : start + block_size] for start in starts],
        axis=0,
    )
    labels: npt.NDArray[np.int32] = tokens[starts + block_size]
    return jnp.asarray(contexts, dtype=jnp.int32), jnp.asarray(labels, dtype=jnp.int32)


def make_block_shift_markov(
    *,
    seed: int,
    steps: int,
    eval_steps: int,
    block_size: int,
    vocab_size: int = 32,
    shift_period: int = 96,
    regime_count: int = 3,
    noise_prob: float = 0.03,
) -> SequenceBenchmark:
    """Generate a contiguous next-token stream with recurring grammar shifts."""
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(vocab_size) for _ in range(regime_count)])
    offsets = rng.integers(1, vocab_size, size=(regime_count,), endpoint=False)
    total_tokens = steps + eval_steps + 2 * block_size + 1
    tokens: npt.NDArray[np.int32] = np.empty(total_tokens, dtype=np.int32)
    tokens[0] = int(rng.integers(0, vocab_size))
    tokens[1] = int(rng.integers(0, vocab_size))
    for index in range(2, total_tokens):
        regime = (index // shift_period) % regime_count
        previous = int(tokens[index - 1])
        older = int(tokens[index - 2])
        deterministic = (
            int(permutations[regime, previous]) + older + int(offsets[regime])
        ) % vocab_size
        if rng.random() < noise_prob:
            tokens[index] = int(rng.integers(0, vocab_size))
        else:
            tokens[index] = deterministic

    train_contexts, train_labels = _contexts_from_tokens(tokens, steps, block_size)
    eval_tokens = tokens[steps:]
    eval_contexts, eval_labels = _contexts_from_tokens(eval_tokens, eval_steps, block_size)
    return SequenceBenchmark(
        name="block_shift_markov",
        description=(
            "Contiguous character-like stream with three recurring second-order "
            "Markov grammars and abrupt block shifts."
        ),
        train_contexts=train_contexts,
        train_labels=train_labels,
        eval_contexts=eval_contexts,
        eval_labels=eval_labels,
        vocab_size=vocab_size,
        metadata={
            "shift_period": shift_period,
            "regime_count": regime_count,
            "noise_prob": noise_prob,
        },
    )


def make_delayed_copy(
    *,
    seed: int,
    steps: int,
    eval_steps: int,
    block_size: int,
    payload_vocab: int = 24,
    filler_vocab: int = 8,
    segment_length: int = 96,
    delays: tuple[int, ...] = (8, 16, 24),
) -> SequenceBenchmark:
    """Generate lag-copy examples with blockwise lag changes and distractors."""
    if max(delays) >= block_size:
        msg = "all delayed-copy lags must be smaller than --block-size"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    total = steps + eval_steps
    vocab_size = payload_vocab + filler_vocab + 1
    query_token = vocab_size - 1
    contexts: npt.NDArray[np.int32] = np.empty((total, block_size), dtype=np.int32)
    labels: npt.NDArray[np.int32] = np.empty((total,), dtype=np.int32)
    for index in range(total):
        delay = delays[(index // segment_length) % len(delays)]
        payload = rng.integers(0, payload_vocab, size=(block_size,), dtype=np.int32)
        filler_mask = rng.random(block_size) < 0.45
        filler = rng.integers(
            payload_vocab,
            payload_vocab + filler_vocab,
            size=(block_size,),
            dtype=np.int32,
        )
        context = np.where(filler_mask, filler, payload).astype(np.int32)
        context[-1] = query_token
        label_pos = block_size - delay - 1
        labels[index] = int(context[label_pos] % payload_vocab)
        contexts[index] = context

    return SequenceBenchmark(
        name="delayed_copy",
        description=(
            "Algorithmic next-token probe where the target is copied from a "
            "block-dependent lag in a context with payload-token distractors."
        ),
        train_contexts=jnp.asarray(contexts[:steps], dtype=jnp.int32),
        train_labels=jnp.asarray(labels[:steps], dtype=jnp.int32),
        eval_contexts=jnp.asarray(contexts[steps:], dtype=jnp.int32),
        eval_labels=jnp.asarray(labels[steps:], dtype=jnp.int32),
        vocab_size=vocab_size,
        metadata={
            "payload_vocab": payload_vocab,
            "filler_vocab": filler_vocab,
            "segment_length": segment_length,
            "delays": list(delays),
        },
    )


def _kv_context(
    rng: np.random.Generator,
    *,
    block_size: int,
    key_a: int,
    key_b: int,
    filler_start: int,
    filler_vocab: int,
    marker_base: int,
) -> npt.NDArray[np.int32]:
    """Build a repeated-key query context with random distractor tokens."""
    context = rng.integers(
        filler_start,
        filler_start + filler_vocab,
        size=(block_size,),
        dtype=np.int32,
    )
    context[-8:] = np.asarray(
        [
            marker_base,
            key_a,
            marker_base + 1,
            key_b,
            key_a,
            marker_base + 2,
            key_b,
            marker_base + 3,
        ],
        dtype=np.int32,
    )
    return context


def _distractor_context(
    rng: np.random.Generator,
    *,
    block_size: int,
    key_symbol_count: int,
    filler_start: int,
    filler_vocab: int,
    marker_base: int,
) -> tuple[npt.NDArray[np.int32], int]:
    """Build an interference example that does not answer a stored binding."""
    context = rng.integers(
        filler_start,
        filler_start + filler_vocab,
        size=(block_size,),
        dtype=np.int32,
    )
    if block_size >= 8:
        context[-4] = marker_base + 4
        context[-3] = int(rng.integers(0, key_symbol_count))
        context[-2] = int(rng.integers(0, key_symbol_count))
        context[-1] = marker_base + 5
    label = int(rng.integers(filler_start, filler_start + filler_vocab))
    return context, label


def make_sparse_kv_recall(
    *,
    seed: int,
    steps: int,
    eval_steps: int,
    block_size: int,
    key_symbol_count: int = 16,
    value_vocab: int = 16,
    filler_vocab: int = 16,
    max_active_keys: int = 64,
    distractor_span: int = 3,
) -> SequenceBenchmark:
    """Generate one-shot random key/value bindings with sparse delayed queries."""
    if block_size < 8:
        msg = "sparse key/value recall requires --block-size at least 8"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    total_key_count = key_symbol_count * key_symbol_count
    active_key_count = min(max_active_keys, total_key_count, max(4, steps // 2))
    value_start = key_symbol_count
    filler_start = value_start + value_vocab
    marker_base = filler_start + filler_vocab
    marker_vocab = 6
    vocab_size = marker_base + marker_vocab
    active_keys = rng.choice(total_key_count, size=active_key_count, replace=False)
    contexts: list[npt.NDArray[np.int32]] = []
    labels: list[int] = []

    def append_binding(mapping: npt.NDArray[np.int32], key_id: int) -> None:
        key_a = key_id // key_symbol_count
        key_b = key_id % key_symbol_count
        contexts.append(
            _kv_context(
                rng,
                block_size=block_size,
                key_a=int(key_a),
                key_b=int(key_b),
                filler_start=filler_start,
                filler_vocab=filler_vocab,
                marker_base=marker_base,
            )
        )
        labels.append(value_start + int(mapping[key_id]))

    def append_distractor() -> None:
        context, label = _distractor_context(
            rng,
            block_size=block_size,
            key_symbol_count=key_symbol_count,
            filler_start=filler_start,
            filler_vocab=filler_vocab,
            marker_base=marker_base,
        )
        contexts.append(context)
        labels.append(label)

    prefix_length = max(0, steps - active_key_count)
    current_mapping = rng.integers(0, value_vocab, size=(total_key_count,), dtype=np.int32)
    while len(contexts) < prefix_length:
        current_mapping = rng.integers(0, value_vocab, size=(total_key_count,), dtype=np.int32)
        for key_id in rng.permutation(active_keys):
            if len(contexts) >= prefix_length:
                break
            append_binding(current_mapping, int(key_id))
        while len(contexts) < prefix_length:
            for _ in range(distractor_span):
                if len(contexts) >= prefix_length:
                    break
                append_distractor()
            if len(contexts) >= prefix_length:
                break
            append_binding(current_mapping, int(rng.choice(active_keys)))
            if rng.random() < 0.35:
                break

    final_mapping = rng.integers(0, value_vocab, size=(total_key_count,), dtype=np.int32)
    for key_id in rng.permutation(active_keys):
        if len(contexts) >= steps:
            break
        append_binding(final_mapping, int(key_id))
    while len(contexts) < steps:
        append_binding(final_mapping, int(rng.choice(active_keys)))

    eval_contexts: list[npt.NDArray[np.int32]] = []
    eval_labels: list[int] = []
    for _ in range(eval_steps):
        key_id = int(rng.choice(active_keys))
        key_a = key_id // key_symbol_count
        key_b = key_id % key_symbol_count
        eval_contexts.append(
            _kv_context(
                rng,
                block_size=block_size,
                key_a=int(key_a),
                key_b=int(key_b),
                filler_start=filler_start,
                filler_vocab=filler_vocab,
                marker_base=marker_base,
            )
        )
        eval_labels.append(value_start + int(final_mapping[key_id]))

    return SequenceBenchmark(
        name="sparse_kv_recall",
        description=(
            "Sparse random key/value recall: key pairs are rebound across "
            "episodes, distractors intervene, and held-out queries target the "
            "final one-shot binding table."
        ),
        train_contexts=jnp.asarray(np.stack(contexts), dtype=jnp.int32),
        train_labels=jnp.asarray(np.asarray(labels, dtype=np.int32), dtype=jnp.int32),
        eval_contexts=jnp.asarray(np.stack(eval_contexts), dtype=jnp.int32),
        eval_labels=jnp.asarray(np.asarray(eval_labels, dtype=np.int32), dtype=jnp.int32),
        vocab_size=vocab_size,
        metadata={
            "key_symbol_count": key_symbol_count,
            "active_key_count": active_key_count,
            "value_vocab": value_vocab,
            "filler_vocab": filler_vocab,
            "distractor_span": distractor_span,
            "query_type": "held-out queries against final binding table",
        },
    )


def _local_text_like_corpus(seed: int, min_chars: int) -> str:
    """Create a deterministic non-Shakespeare prose-like corpus."""
    rng = np.random.default_rng(seed)
    subjects = [
        "ada",
        "bert",
        "cyra",
        "dion",
        "elin",
        "faro",
        "gita",
        "hollis",
    ]
    places = [
        "harbor",
        "library",
        "garden",
        "workshop",
        "station",
        "archive",
    ]
    objects = [
        "lantern",
        "ledger",
        "map",
        "signal",
        "engine",
        "notebook",
        "compass",
    ]
    verbs = [
        "marks",
        "carries",
        "checks",
        "returns",
        "copies",
        "sorts",
        "tests",
    ]
    clauses = [
        "before the bell rings",
        "while rain taps the glass",
        "after the quiet report",
        "when the north door opens",
        "as the small clock turns",
    ]
    lines: list[str] = []
    index = 0
    while len("\n".join(lines)) < min_chars:
        subject = subjects[(index + seed) % len(subjects)]
        partner = subjects[(index * 3 + 1 + seed) % len(subjects)]
        place = places[int(rng.integers(0, len(places)))]
        obj = objects[(index * 5 + seed) % len(objects)]
        verb = verbs[int(rng.integers(0, len(verbs)))]
        clause = clauses[(index + int(rng.integers(0, len(clauses)))) % len(clauses)]
        lines.append(
            f"{subject} {verb} the {obj} in the {place}, {clause}. "
            f"{partner} asks for the {obj}; {subject} answers with the {place} note."
        )
        index += 1
    return "\n".join(lines).lower()


def make_local_text_motif(
    *,
    seed: int,
    steps: int,
    eval_steps: int,
    block_size: int,
) -> SequenceBenchmark:
    """Generate a non-Shakespeare character stream without network dependency."""
    total_tokens = steps + eval_steps + 2 * block_size + 1
    text = _local_text_like_corpus(seed, total_tokens + 512)
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    tokens = np.asarray([stoi[char] for char in text], dtype=np.int32)
    train_contexts, train_labels = _contexts_from_tokens(tokens, steps, block_size)
    eval_tokens = tokens[steps:]
    eval_contexts, eval_labels = _contexts_from_tokens(eval_tokens, eval_steps, block_size)
    return SequenceBenchmark(
        name="local_text_motif",
        description=(
            "Deterministic non-Shakespeare prose-like character stream. It is "
            "used because no cached non-Shakespeare public text corpus is present."
        ),
        train_contexts=train_contexts,
        train_labels=train_labels,
        eval_contexts=eval_contexts,
        eval_labels=eval_labels,
        vocab_size=len(chars),
        metadata={
            "source": "deterministic_local_generator",
            "cached_public_corpus_found": False,
            "char_count": len(text),
            "vocab": chars,
        },
    )


def make_benchmark(
    name: BenchmarkName,
    *,
    seed: int,
    steps: int,
    eval_steps: int,
    block_size: int,
) -> SequenceBenchmark:
    """Dispatch benchmark generation."""
    if name == "block_shift_markov":
        return make_block_shift_markov(
            seed=seed,
            steps=steps,
            eval_steps=eval_steps,
            block_size=block_size,
        )
    if name == "delayed_copy":
        return make_delayed_copy(
            seed=seed,
            steps=steps,
            eval_steps=eval_steps,
            block_size=block_size,
        )
    if name == "sparse_kv_recall":
        return make_sparse_kv_recall(
            seed=seed,
            steps=steps,
            eval_steps=eval_steps,
            block_size=block_size,
        )
    if name == "local_text_motif":
        return make_local_text_motif(
            seed=seed,
            steps=steps,
            eval_steps=eval_steps,
            block_size=block_size,
        )
    raise ValueError(f"unknown benchmark: {name}")


def aggregate_metric(
    records: list[dict[str, Any]],
    benchmark: str,
    method: str,
    metric: str,
) -> np.ndarray:
    """Collect one metric across seeds."""
    fallback = {
        "eval_fast_nll": "eval_nll",
        "eval_fast_accuracy": "eval_accuracy",
        "eval_fast_perplexity": "eval_perplexity",
        "final_window_base_nll": "final_window_nll",
    }
    values = []
    for row in records:
        if row["benchmark"] != benchmark or row["method"] != method:
            continue
        summary = row["summary"]
        if metric in summary:
            values.append(summary[metric])
        else:
            values.append(summary[fallback[metric]])
    return np.asarray(values, dtype=np.float64)


def profile_models(
    args: argparse.Namespace,
    block: PrototypeBasisBlock,
    vocab_size: int,
) -> dict[str, dict[str, int]]:
    """Return trainable and state footprint for the compared methods."""
    profile_key = jr.key(args.seed + 10_007)
    baseline_params = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    static_params, static_state = init_advantage_params_state(
        profile_key,
        block=block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_logit=args.gate_init_logit,
        gate_mode="scalar",
    )
    replay_state_elements = args.replay_size * args.block_size + args.replay_size + 2
    replay_state_bytes = 4 * replay_state_elements
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_params),
            "trainable_bytes": count_array_bytes(baseline_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "replay_capped_post_ffn_memory": {
            "trainable_params": count_array_elements(static_params),
            "trainable_bytes": count_array_bytes(static_params),
            "state_elements": count_array_elements(static_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(static_state, include_int=True)
            + replay_state_bytes,
        },
    }
    for variant in args.memory_variants:
        if variant == "static_replay":
            continue
        spec = mechanism_spec(variant)
        method = VARIANT_METHODS[variant]
        rescue_params, rescue_state = init_rescue_params_state(
            profile_key,
            block=block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            gate_init_logit=args.gate_init_logit,
            gate_mode=spec.gate_mode,
        )
        profiles[method] = {
            "trainable_params": count_array_elements(rescue_params),
            "trainable_bytes": count_array_bytes(rescue_params),
            "state_elements": count_array_elements(rescue_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(rescue_state, include_int=True)
            + replay_state_bytes,
        }
    return profiles


def run_one_seed(
    args: argparse.Namespace,
    block: PrototypeBasisBlock,
    benchmark: SequenceBenchmark,
    seed_idx: int,
) -> list[dict[str, Any]]:
    """Run FFN and replay-capped memory for one benchmark seed."""
    benchmark_seed = ALL_BENCHMARKS.index(benchmark.name)
    run_key = jr.fold_in(jr.key(args.seed), 10_000 * (seed_idx + 1) + benchmark_seed)
    param_key = jr.fold_in(run_key, 17)
    final_window = args.final_window if args.final_window > 0 else args.eval_steps
    records: list[dict[str, Any]] = []

    baseline_params = init_transformer_params(
        param_key,
        vocab_size=benchmark.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    method_start = time.perf_counter()
    final_baseline, baseline_metrics = run_baseline_transformer(
        baseline_params,
        benchmark.train_contexts,
        benchmark.train_labels,
        step_size=args.baseline_lr,
        grad_clip=args.grad_clip,
    )
    train_s = time.perf_counter() - method_start
    baseline_summary = {
        **summarize_online(baseline_metrics, final_window),
        **eval_transformer(final_baseline, benchmark.eval_contexts, benchmark.eval_labels),
        "train_s": train_s,
        "train_steps_per_s": args.steps / train_s,
    }
    records.append(
        {
            "benchmark": benchmark.name,
            "seed": seed_idx,
            "method": BASELINE_METHOD,
            "summary": baseline_summary,
        }
    )
    print(
        f"{benchmark.name} seed={seed_idx} baseline: "
        f"fw_nll={baseline_summary['final_window_nll']:.3f}, "
        f"eval_nll={baseline_summary['eval_nll']:.3f}, "
        f"acc={baseline_summary['eval_accuracy']:.3f}, train_s={train_s:.2f}"
    )

    for variant in args.memory_variants:
        method = VARIANT_METHODS[variant]
        if variant == "static_replay":
            memory_params, memory_state = init_advantage_params_state(
                param_key,
                block=block,
                vocab_size=benchmark.vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                ffn_hidden=args.mlp_hidden,
                gate_init_logit=args.gate_init_logit,
                gate_mode="scalar",
            )
            method_start = time.perf_counter()
            final_memory_params, final_memory_state, memory_metrics = (
                run_advantage_memory_transformer(
                    block,
                    memory_params,
                    memory_state,
                    benchmark.train_contexts,
                    benchmark.train_labels,
                    placement="post_ffn",
                    fast_lr=args.fast_lr,
                    slow_lr=args.slow_lr,
                    grad_clip=args.grad_clip,
                    gate_lr=args.gate_lr,
                    gate_decay=args.gate_decay,
                    gate_max=args.gate_max,
                    advantage_margin=0.0,
                    gate_l2=args.gate_l2,
                    gate_init_logit=args.gate_init_logit,
                    gate_objective="replay",
                    replay_size=args.replay_size,
                    train_loss_mode="memory",
                    memory_loss_weight=1.0,
                    reset_mode="meta_ema",
                )
            )
            summary_fn = summarize_advantage
        else:
            spec = mechanism_spec(variant)
            memory_params, memory_state = init_rescue_params_state(
                param_key,
                block=block,
                vocab_size=benchmark.vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                ffn_hidden=args.mlp_hidden,
                gate_init_logit=args.gate_init_logit,
                gate_mode=spec.gate_mode,
            )
            method_start = time.perf_counter()
            final_memory_params, final_memory_state, memory_metrics = (
                run_rescue_memory_transformer(
                    block,
                    memory_params,
                    memory_state,
                    benchmark.train_contexts,
                    benchmark.train_labels,
                    mechanism=spec,
                    placement="post_ffn",
                    fast_lr=args.fast_lr,
                    slow_lr=args.slow_lr,
                    grad_clip=args.grad_clip,
                    gate_lr=args.gate_lr,
                    gate_decay=args.gate_decay,
                    gate_max=args.gate_max,
                    advantage_margin=0.0,
                    gate_l2=args.gate_l2,
                    gate_init_logit=args.gate_init_logit,
                    gate_objective="replay",
                    replay_size=args.replay_size,
                    train_loss_mode="memory",
                    memory_loss_weight=1.0,
                    negative_gate_multiplier=args.negative_gate_multiplier,
                    utility_decay=args.utility_decay,
                    utility_replace_threshold=args.utility_replace_threshold,
                    utility_novelty_bonus=args.utility_novelty_bonus,
                    min_update_gate=args.min_update_gate,
                )
            )
            summary_fn = summarize_rescue
        final_memory_state.step_count.block_until_ready()
        train_s = time.perf_counter() - method_start
        memory_summary = {
            **summary_fn(memory_metrics, final_window),
            **eval_advantage_memory_transformer(
                block,
                final_memory_params,
                final_memory_state,
                benchmark.eval_contexts,
                benchmark.eval_labels,
                placement="post_ffn",
            ),
            **eval_advantage_fast_only(
                block,
                final_memory_params,
                final_memory_state,
                benchmark.eval_contexts,
                benchmark.eval_labels,
                placement="post_ffn",
            ),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {
                "benchmark": benchmark.name,
                "seed": seed_idx,
                "method": method,
                "variant": variant,
                "summary": memory_summary,
            }
        )
        print(
            f"{benchmark.name} seed={seed_idx} {method}: "
            f"fw_nll={memory_summary['final_window_nll']:.3f}, "
            f"eval_nll={memory_summary['eval_nll']:.3f}, "
            f"fast_eval_nll={memory_summary['eval_fast_nll']:.3f}, "
            f"gate={memory_summary['final_window_gate']:.3f}, train_s={train_s:.2f}"
        )
    return records


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary for the run."""
    records = payload["records"]
    benchmark_names = [item["name"] for item in payload["benchmarks"]]
    methods = [
        BASELINE_METHOD,
        *[VARIANT_METHODS[variant] for variant in payload["config"]["memory_variants"]],
    ]
    labels = METHOD_LABELS
    metrics = [
        "final_window_nll",
        "final_window_base_nll",
        "final_window_accuracy",
        "eval_nll",
        "eval_fast_nll",
        "eval_accuracy",
        "eval_perplexity",
        "train_s",
        "train_steps_per_s",
    ]
    lower = {
        "final_window_nll",
        "final_window_base_nll",
        "eval_nll",
        "eval_fast_nll",
        "eval_perplexity",
        "train_s",
    }
    lines = [
        "# Step 2 External Sequence Memory Benchmarks",
        "",
        "Replay-capped post-FFN memory is compared with the FFN transformer on "
        "generated continual sequence probes outside Tiny Shakespeare.",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        f"Replay cap: `{payload['config']['replay_size']}` examples.",
        "Memory variants: "
        + ", ".join(f"`{variant}`" for variant in payload["config"]["memory_variants"])
        + ".",
        "",
        "## Benchmarks",
        "",
    ]
    for item in payload["benchmarks"]:
        lines.append(f"- `{item['name']}`: {item['description']}")
    lines.extend(["", "## Architecture and State", ""])
    for benchmark_name in benchmark_names:
        lines.append(f"### `{benchmark_name}`")
        lines.append("")
        lines.append(
            "| Method | Trainable params | Trainable bytes | State elements | State bytes |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        profiles = payload["profiles"][benchmark_name]
        for method in methods:
            profile = profiles[method]
            lines.append(
                f"| {labels[method]} | {profile['trainable_params']} | "
                f"{profile['trainable_bytes']} | {profile['state_elements']} | "
                f"{profile['state_bytes']} |"
            )
        lines.append("")
    lines.extend(["## Metrics", ""])
    for benchmark_name in benchmark_names:
        lines.append(f"### `{benchmark_name}`")
        lines.append("")
        lines.append("| Metric | Method | Value | Diff vs FFN | Diff vs static replay |")
        lines.append("|---|---|---:|---:|---:|")
        for metric in metrics:
            baseline = aggregate_metric(records, benchmark_name, BASELINE_METHOD, metric)
            static = aggregate_metric(records, benchmark_name, STATIC_REPLAY_METHOD, metric)
            for method in methods:
                values = aggregate_metric(records, benchmark_name, method, metric)
                diff_ffn = baseline - values if metric in lower else values - baseline
                diff_static = static - values if metric in lower else values - static
                static_text = (
                    ""
                    if method == STATIC_REPLAY_METHOD
                    else f"{np.mean(diff_static):+.4f} +/- {stderr(diff_static):.4f}"
                )
                lines.append(
                    f"| `{metric}` | {labels[method]} | "
                    f"{np.mean(values):.4f} +/- {stderr(values):.4f} | "
                    f"{np.mean(diff_ffn):+.4f} +/- {stderr(diff_ffn):.4f} | "
                    f"{static_text} |"
                )
        lines.append("")
    lines.extend(["## Replay-Memory Diagnostics", ""])
    diagnostic_metrics = [
        "final_window_advantage",
        "final_window_gate_update_advantage",
        "final_window_gate",
        "final_window_gate_logit",
        "final_window_active_prototypes",
        "final_window_allocation_rate",
        "final_window_active_features",
        "final_window_update_gate",
        "final_window_nearest_distance",
        "final_window_utility_mean",
        "final_window_utility_min",
        "final_window_replacement_blocked_rate",
    ]
    for benchmark_name in benchmark_names:
        lines.append(f"### `{benchmark_name}`")
        lines.append("")
        lines.append("| Method | Metric | Mean +/- stderr |")
        lines.append("|---|---|---:|")
        for method in methods[1:]:
            rows = [
                row
                for row in records
                if row["benchmark"] == benchmark_name and row["method"] == method
            ]
            for metric in diagnostic_metrics:
                diagnostic_values = [
                    row["summary"][metric]
                    for row in rows
                    if metric in row["summary"]
                ]
                if not diagnostic_values:
                    continue
                array = np.asarray(diagnostic_values, dtype=np.float64)
                lines.append(
                    f"| {labels[method]} | `{metric}` | "
                    f"{np.mean(array):.6f} +/- {stderr(array):.6f} |"
                )
        lines.append("")
    lines.append(
        "Positive diffs favor the row method. Wall-clock includes JAX compilation."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_benchmarks(raw: list[str]) -> list[BenchmarkName]:
    """Parse benchmark CLI names."""
    if "all" in raw:
        return list(ALL_BENCHMARKS)
    parsed: list[BenchmarkName] = []
    for name in raw:
        if name not in ALL_BENCHMARKS:
            msg = f"unknown benchmark {name!r}; choose from {ALL_BENCHMARKS} or 'all'"
            raise ValueError(msg)
        parsed.append(name)
    return parsed


def parse_memory_variants(raw: list[str]) -> list[MemoryVariantName]:
    """Parse memory-variant CLI names and keep static replay as comparator."""
    if "all" in raw:
        parsed = list(ALL_MEMORY_VARIANTS)
    else:
        parsed = []
        for name in raw:
            if name not in ALL_MEMORY_VARIANTS:
                msg = (
                    f"unknown memory variant {name!r}; choose from "
                    f"{ALL_MEMORY_VARIANTS} or 'all'"
                )
                raise ValueError(msg)
            parsed.append(name)
    if "static_replay" not in parsed:
        parsed.insert(0, "static_replay")
    deduped: list[MemoryVariantName] = []
    for name in parsed:
        if name not in deduped:
            deduped.append(name)
    return deduped


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument(
        "--memory-variants",
        nargs="+",
        default=["static_replay", "rescue_split_update_utility"],
        help=(
            "Memory mechanisms to compare. Use 'all' for static replay plus "
            "all local rescue variants."
        ),
    )
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
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
    parser.add_argument("--gate-max", type=float, default=0.15)
    parser.add_argument("--gate-l2", type=float, default=0.1)
    parser.add_argument("--replay-size", type=int, default=128)
    parser.add_argument("--negative-gate-multiplier", type=float, default=8.0)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--utility-replace-threshold", type=float, default=-0.002)
    parser.add_argument("--utility-novelty-bonus", type=float, default=0.05)
    parser.add_argument("--min-update-gate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/external_sequence_memory"),
    )
    args = parser.parse_args()
    validate_args(args)
    args.benchmarks = parse_benchmarks(args.benchmarks)
    args.memory_variants = parse_memory_variants(args.memory_variants)
    args.final_window = args.final_window if args.final_window > 0 else args.eval_steps
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 25:
        raise ValueError("--block-size must be at least 25 for the delayed-copy probe")
    if min(args.d_model, args.mlp_hidden, args.proto_count, args.replay_size) < 1:
        raise ValueError("model sizes and replay size must be positive")
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
    if not 0.0 < args.gate_max < 1.0:
        raise ValueError("--gate-max must be in (0, 1)")
    if args.negative_gate_multiplier < 1.0:
        raise ValueError("--negative-gate-multiplier must be >= 1")
    if not 0.0 <= args.utility_decay <= 1.0:
        raise ValueError("--utility-decay must be in [0, 1]")
    if args.utility_novelty_bonus < 0.0:
        raise ValueError("--utility-novelty-bonus must be non-negative")
    if not 0.0 <= args.min_update_gate < 1.0:
        raise ValueError("--min-update-gate must be in [0, 1)")


def main() -> None:
    """Run the benchmark suite."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    block = make_memory_block(args)
    records: list[dict[str, Any]] = []
    benchmark_payloads: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, dict[str, int]]] = {}
    start = time.perf_counter()

    for benchmark_name in args.benchmarks:
        first_benchmark = make_benchmark(
            benchmark_name,
            seed=args.seed,
            steps=args.steps,
            eval_steps=args.eval_steps,
            block_size=args.block_size,
        )
        benchmark_payloads.append(
            {
                "name": first_benchmark.name,
                "description": first_benchmark.description,
                "vocab_size": first_benchmark.vocab_size,
                "metadata": first_benchmark.metadata,
            }
        )
        profiles[first_benchmark.name] = profile_models(args, block, first_benchmark.vocab_size)
        for seed_idx in range(args.seeds):
            benchmark = make_benchmark(
                benchmark_name,
                seed=args.seed + 1009 * seed_idx,
                steps=args.steps,
                eval_steps=args.eval_steps,
                block_size=args.block_size,
            )
            records.extend(run_one_seed(args, block, benchmark, seed_idx))

    config = ExperimentConfig(
        benchmarks=list(args.benchmarks),
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        proto_count=args.proto_count,
        eval_steps=args.eval_steps,
        final_window=args.final_window,
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
        gate_max=args.gate_max,
        gate_l2=args.gate_l2,
        replay_size=args.replay_size,
        memory_variants=list(args.memory_variants),
        negative_gate_multiplier=args.negative_gate_multiplier,
        utility_decay=args.utility_decay,
        utility_replace_threshold=args.utility_replace_threshold,
        utility_novelty_bonus=args.utility_novelty_bonus,
        min_update_gate=args.min_update_gate,
        seed=args.seed,
        output_dir=str(args.output_dir),
    )
    payload = {
        "config": asdict(config),
        "benchmarks": benchmark_payloads,
        "profiles": profiles,
        "records": records,
        "elapsed_s": time.perf_counter() - start,
        "candidate_note": (
            "Static replay is the original post-FFN scalar advantage gate from "
            "the Tiny Shakespeare follow-up. Rescue variants change only "
            "gate/replacement/reset/update-vs-prediction dynamics inside this "
            "external runner."
        ),
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

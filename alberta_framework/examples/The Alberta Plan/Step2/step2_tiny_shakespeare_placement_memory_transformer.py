#!/usr/bin/env python3
# ruff: noqa: E402
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare replay-advantage placement memory transformer.

This runner attacks the post-FFN versus pre-FFN-KV placement issue without
hardcoding a horizon-specific oracle. It keeps the replay-capped memory gate
from the current advantage-memory candidate and adds a learned placement
selector. The selector is updated from delayed replay loss:

``placement signal = replay_post_loss - replay_pre_loss``.

Positive signal opens the pre-KV branch; negative signal closes it toward the
stable post-FFN branch. A small optional placement cost can bias against using
pre-KV unless replay shows measured advantage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _install_core_import_shim() -> None:
    """Import core modules without executing package-level research exports."""
    root = Path(__file__).resolve().parents[3]
    package_path = root / "src" / "alberta_framework"
    if not package_path.exists() or "alberta_framework" in sys.modules:
        return
    package = types.ModuleType("alberta_framework")
    setattr(package, "__path__", [str(package_path)])
    sys.modules["alberta_framework"] = package
    core_package = types.ModuleType("alberta_framework.core")
    setattr(core_package, "__path__", [str(package_path / "core")])
    sys.modules["alberta_framework.core"] = core_package


_install_core_import_shim()

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_tiny_shakespeare_advantage_memory_transformer import (
    eval_advantage_fast_only,
    eval_advantage_memory_transformer,
    init_advantage_params_state,
    run_advantage_memory_transformer,
    summarize_advantage,
)
from step2_tiny_shakespeare_proto_basis_transformer import (
    make_proto_block,
    select_center_slot,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    causal_attention_sequence,
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


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured in result artifacts."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    proto_count: int
    adaptive_post_proto_count: int
    adaptive_pre_proto_count: int
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
    replay_size: int
    train_loss_mode: str
    memory_loss_weight: float
    reset_mode: str
    placement_init_logit: float
    placement_lr: float
    placement_decay: float
    placement_margin: float
    placement_l2: float
    selector_loss_mode: str
    proto_count_is_per_placement: bool
    data_path: str
    output_dir: str
    seed: int


@chex.dataclass(frozen=True)
class AdaptivePlacementMemoryState:
    """Two placement memories plus scalar resource-manager traces."""

    post_proto_state: PrototypeBasisState
    pre_proto_state: PrototypeBasisState
    memory_gate_logit: jax.Array
    placement_logit: jax.Array
    memory_advantage_ema: jax.Array
    placement_advantage_ema: jax.Array
    init_value: jax.Array
    step_count: jax.Array


def make_sized_proto_block(args: argparse.Namespace, n_prototypes: int) -> PrototypeBasisBlock:
    """Create a prototype block with a caller-selected budget."""
    return PrototypeBasisBlock(
        PrototypeBasisConfig(
            input_dim=args.d_model,
            output_dim=args.d_model,
            n_prototypes=n_prototypes,
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


def adaptive_proto_counts(args: argparse.Namespace) -> tuple[int, int]:
    """Return post/pre prototype counts for the adaptive manager."""
    if args.proto_count_is_per_placement:
        return args.proto_count, args.proto_count
    post_count = max(1, (args.proto_count + 1) // 2)
    pre_count = max(1, args.proto_count // 2)
    return post_count, pre_count


def init_adaptive_params_state(
    key: jax.Array,
    *,
    post_block: PrototypeBasisBlock,
    pre_block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    memory_gate_init_logit: float,
    placement_init_logit: float,
) -> tuple[dict[str, Any], AdaptivePlacementMemoryState]:
    """Initialize the transformer and the adaptive placement memory manager."""
    post_key = jr.fold_in(key, 17)
    pre_key = jr.fold_in(key, 23)
    post_proto_params, post_proto_state = post_block.init(post_key)
    pre_proto_params, pre_proto_state = pre_block.init(pre_key)
    params = init_transformer_params(
        key,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
    )
    state = AdaptivePlacementMemoryState(
        post_proto_state=post_proto_state,
        pre_proto_state=pre_proto_state,
        memory_gate_logit=jnp.asarray(memory_gate_init_logit, dtype=jnp.float32),
        placement_logit=jnp.asarray(placement_init_logit, dtype=jnp.float32),
        memory_advantage_ema=jnp.asarray(0.0, dtype=jnp.float32),
        placement_advantage_ema=jnp.asarray(0.0, dtype=jnp.float32),
        init_value=jnp.zeros((d_model,), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    return {
        **params,
        "post_proto": post_proto_params,
        "pre_proto": pre_proto_params,
    }, state


def sgd_step_adaptive(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    fast_lr: float,
    slow_lr: float,
) -> dict[str, Any]:
    """Apply fast rates to the transformer and slow rates to placement memories."""

    def step_tree(tree: Any, grad_tree: Any, lr: float) -> Any:
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, tree, grad_tree)

    return {
        "attn": step_tree(params["attn"], grads["attn"], fast_lr),
        "ffn": step_tree(params["ffn"], grads["ffn"], fast_lr),
        "readout": step_tree(params["readout"], grads["readout"], fast_lr),
        "post_proto": step_tree(params["post_proto"], grads["post_proto"], slow_lr),
        "pre_proto": step_tree(params["pre_proto"], grads["pre_proto"], slow_lr),
    }


def reset_adaptive_proto_row(
    params: dict[str, Any],
    state: AdaptivePlacementMemoryState,
    branch: str,
    slot: jax.Array,
    novel: jax.Array,
    *,
    reset_mode: str,
) -> dict[str, Any]:
    """Reset a newly allocated placement-memory value row."""
    if reset_mode == "none":
        return params
    key = f"{branch}_proto"
    proto_params = params[key]
    old_row = proto_params.values[slot]
    if reset_mode == "zero":
        new_row = jnp.where(novel, jnp.zeros_like(old_row), old_row)
    else:
        new_row = jnp.where(novel, state.init_value, old_row)
    new_proto = PrototypeBasisParams(
        values=proto_params.values.at[slot].set(new_row),
        bias=proto_params.bias,
    )
    return {**params, key: new_proto}


def adaptive_model_parts(
    post_block: PrototypeBasisBlock,
    pre_block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdaptivePlacementMemoryState,
    context: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Return base, branch, and mixed logits plus placement diagnostics."""
    attn_hidden = causal_attention_sequence(params["attn"], context)[-1]
    base_hidden = ffn_transform(params["ffn"], attn_hidden)
    readout = params["readout"]
    base_logits = base_hidden @ readout["w"] + readout["b"]

    memory_gate = jax.nn.sigmoid(state.memory_gate_logit)
    pre_weight = jax.nn.sigmoid(state.placement_logit)

    post_activations = post_block.activations(state.post_proto_state, base_hidden)
    post_residual = memory_gate * post_block.transform(
        params["post_proto"],
        post_activations,
    )
    post_hidden = base_hidden + post_residual
    post_logits = post_hidden @ readout["w"] + readout["b"]

    pre_activations = pre_block.activations(state.pre_proto_state, attn_hidden)
    pre_residual = memory_gate * pre_block.transform(
        params["pre_proto"],
        pre_activations,
    )
    pre_hidden = ffn_transform(params["ffn"], attn_hidden + pre_residual)
    pre_logits = pre_hidden @ readout["w"] + readout["b"]

    mixed_logits = (1.0 - pre_weight) * post_logits + pre_weight * pre_logits
    return (
        base_logits,
        post_logits,
        pre_logits,
        mixed_logits,
        base_hidden,
        attn_hidden,
        post_activations,
        pre_activations,
        memory_gate,
        pre_weight,
        post_residual,
        pre_residual,
    )


def placement_losses(
    post_block: PrototypeBasisBlock,
    pre_block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdaptivePlacementMemoryState,
    context: jax.Array,
    label: jax.Array,
    *,
    selector_loss_mode: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return base, post, pre, selected memory, and pre-weight losses."""
    (
        base_logits,
        post_logits,
        pre_logits,
        mixed_logits,
        _post_basis,
        _pre_basis,
        _post_activations,
        _pre_activations,
        _memory_gate,
        pre_weight,
        _post_residual,
        _pre_residual,
    ) = adaptive_model_parts(post_block, pre_block, params, state, context)
    base_loss = cross_entropy_from_logits(base_logits, label)
    post_loss = cross_entropy_from_logits(post_logits, label)
    pre_loss = cross_entropy_from_logits(pre_logits, label)
    if selector_loss_mode == "weighted_losses":
        selected_loss = (1.0 - pre_weight) * post_loss + pre_weight * pre_loss
    else:
        selected_loss = cross_entropy_from_logits(mixed_logits, label)
    return base_loss, post_loss, pre_loss, selected_loss, pre_weight


def run_adaptive_placement_memory_transformer(
    post_block: PrototypeBasisBlock,
    pre_block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdaptivePlacementMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    gate_lr: float,
    gate_decay: float,
    gate_max: float,
    advantage_margin: float,
    gate_l2: float,
    replay_size: int,
    reset_mode: str,
    placement_init_logit: float,
    placement_lr: float,
    placement_decay: float,
    placement_margin: float,
    placement_l2: float,
    selector_loss_mode: str,
) -> tuple[dict[str, Any], AdaptivePlacementMemoryState, np.ndarray]:
    """Train one adaptive replay-placement memory transformer."""
    replay_capacity = max(1, replay_size)

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: AdaptivePlacementMemoryState,
    ) -> tuple[tuple[Any, ...], jax.Array]:
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
                tuple[
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
            ]:
                (
                    base_logits,
                    post_logits,
                    pre_logits,
                    mixed_logits,
                    post_basis,
                    pre_basis,
                    post_activations,
                    pre_activations,
                    memory_gate,
                    pre_weight,
                    post_residual,
                    pre_residual,
                ) = adaptive_model_parts(
                    post_block,
                    pre_block,
                    candidate,
                    state,
                    context,
                )
                base_loss = cross_entropy_from_logits(base_logits, label)
                post_loss = cross_entropy_from_logits(post_logits, label)
                pre_loss = cross_entropy_from_logits(pre_logits, label)
                if selector_loss_mode == "weighted_losses":
                    selected_loss = (1.0 - pre_weight) * post_loss + pre_weight * pre_loss
                else:
                    selected_loss = cross_entropy_from_logits(mixed_logits, label)
                return selected_loss, (
                    base_loss,
                    post_loss,
                    pre_loss,
                    mixed_logits,
                    post_basis,
                    pre_basis,
                    post_activations,
                    pre_activations,
                    memory_gate,
                    pre_weight,
                    (1.0 - pre_weight) * post_residual + pre_weight * pre_residual,
                )

            (
                selected_loss,
                (
                    base_loss,
                    post_loss,
                    pre_loss,
                    mixed_logits,
                    post_basis,
                    pre_basis,
                    post_activations,
                    pre_activations,
                    memory_gate,
                    pre_weight,
                    mixed_residual,
                ),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step_adaptive(
                params,
                grads,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )

            post_slot, post_novel = select_center_slot(
                post_block,
                state.post_proto_state,
                post_basis,
            )
            pre_slot, pre_novel = select_center_slot(
                pre_block,
                state.pre_proto_state,
                pre_basis,
            )
            new_post_state, post_center_metrics = post_block.update_centers(
                state.post_proto_state,
                post_basis,
            )
            new_pre_state, pre_center_metrics = pre_block.update_centers(
                state.pre_proto_state,
                pre_basis,
            )
            new_params = reset_adaptive_proto_row(
                new_params,
                state,
                "post",
                post_slot,
                post_novel,
                reset_mode=reset_mode,
            )
            new_params = reset_adaptive_proto_row(
                new_params,
                state,
                "pre",
                pre_slot,
                pre_novel,
                reset_mode=reset_mode,
            )

            replay_state = AdaptivePlacementMemoryState(
                post_proto_state=new_post_state,
                pre_proto_state=new_pre_state,
                memory_gate_logit=state.memory_gate_logit,
                placement_logit=state.placement_logit,
                memory_advantage_ema=state.memory_advantage_ema,
                placement_advantage_ema=state.placement_advantage_ema,
                init_value=state.init_value,
                step_count=state.step_count,
            )
            replay_slot = jnp.mod(replay_index, replay_capacity)
            replay_base_loss, replay_post_loss, replay_pre_loss, replay_selected_loss, _ = (
                placement_losses(
                    post_block,
                    pre_block,
                    new_params,
                    replay_state,
                    replay_contexts[replay_slot],
                    replay_labels[replay_slot],
                    selector_loss_mode=selector_loss_mode,
                )
            )
            current_memory_advantage = base_loss - selected_loss
            current_placement_advantage = post_loss - pre_loss
            replay_memory_advantage = replay_base_loss - replay_selected_loss
            replay_placement_advantage = replay_post_loss - replay_pre_loss
            memory_update_advantage = jnp.where(
                replay_count > 0,
                replay_memory_advantage,
                current_memory_advantage,
            )
            placement_update_advantage = jnp.where(
                replay_count > 0,
                replay_placement_advantage,
                current_placement_advantage,
            )

            memory_signal = (
                memory_update_advantage
                - jnp.asarray(advantage_margin, dtype=jnp.float32)
                - jnp.asarray(gate_l2, dtype=jnp.float32) * memory_gate
            )
            new_memory_gate_logit = (
                jnp.asarray(gate_decay, dtype=jnp.float32) * state.memory_gate_logit
                + jnp.asarray(gate_lr, dtype=jnp.float32)
                * jnp.clip(memory_signal, -1.0, 1.0)
            )
            max_gate_logit = jnp.log(
                jnp.asarray(gate_max, dtype=jnp.float32)
                / jnp.maximum(1.0 - jnp.asarray(gate_max, dtype=jnp.float32), 1e-6)
            )
            new_memory_gate_logit = jnp.clip(
                new_memory_gate_logit,
                -8.0,
                jnp.minimum(max_gate_logit, 8.0),
            )

            placement_signal = (
                placement_update_advantage
                - jnp.asarray(placement_margin, dtype=jnp.float32)
                - jnp.asarray(placement_l2, dtype=jnp.float32) * pre_weight
            )
            placement_prior = jnp.asarray(placement_init_logit, dtype=jnp.float32)
            new_placement_logit = (
                placement_prior
                + jnp.asarray(placement_decay, dtype=jnp.float32)
                * (state.placement_logit - placement_prior)
                + jnp.asarray(placement_lr, dtype=jnp.float32)
                * jnp.clip(placement_signal, -1.0, 1.0)
            )
            new_placement_logit = jnp.clip(new_placement_logit, -8.0, 8.0)

            improved = current_memory_advantage > 0.0
            init_value = jnp.where(
                improved,
                0.99 * state.init_value + 0.01 * mixed_residual,
                state.init_value,
            )
            new_state = AdaptivePlacementMemoryState(
                post_proto_state=new_post_state,
                pre_proto_state=new_pre_state,
                memory_gate_logit=new_memory_gate_logit,
                placement_logit=new_placement_logit,
                memory_advantage_ema=0.99 * state.memory_advantage_ema
                + 0.01 * current_memory_advantage,
                placement_advantage_ema=0.99 * state.placement_advantage_ema
                + 0.01 * current_placement_advantage,
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

            acc = (jnp.argmax(mixed_logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    selected_loss,
                    acc,
                    base_loss,
                    post_loss,
                    pre_loss,
                    base_loss - post_loss,
                    base_loss - pre_loss,
                    current_memory_advantage,
                    current_placement_advantage,
                    memory_gate,
                    pre_weight,
                    new_memory_gate_logit,
                    new_placement_logit,
                    memory_update_advantage,
                    placement_update_advantage,
                    post_center_metrics[0],
                    pre_center_metrics[0],
                    post_center_metrics[1],
                    pre_center_metrics[1],
                    jnp.sum(post_activations > 1e-6).astype(jnp.float32),
                    jnp.sum(pre_activations > 1e-6).astype(jnp.float32),
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


def eval_adaptive_placement_memory_transformer(
    post_block: PrototypeBasisBlock,
    pre_block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: AdaptivePlacementMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    selector_loss_mode: str,
) -> dict[str, float]:
    """Evaluate adaptive placement memory and its component branches."""

    @jax.jit
    def run() -> tuple[jax.Array, ...]:
        def one(context: jax.Array, label: jax.Array) -> jax.Array:
            (
                base_logits,
                post_logits,
                pre_logits,
                mixed_logits,
                _post_basis,
                _pre_basis,
                _post_activations,
                _pre_activations,
                _memory_gate,
                pre_weight,
                _post_residual,
                _pre_residual,
            ) = adaptive_model_parts(post_block, pre_block, params, state, context)
            base_loss = cross_entropy_from_logits(base_logits, label)
            post_loss = cross_entropy_from_logits(post_logits, label)
            pre_loss = cross_entropy_from_logits(pre_logits, label)
            if selector_loss_mode == "weighted_losses":
                selected_loss = (1.0 - pre_weight) * post_loss + pre_weight * pre_loss
            else:
                selected_loss = cross_entropy_from_logits(mixed_logits, label)
            acc = (jnp.argmax(mixed_logits) == label).astype(jnp.float32)
            post_acc = (jnp.argmax(post_logits) == label).astype(jnp.float32)
            pre_acc = (jnp.argmax(pre_logits) == label).astype(jnp.float32)
            base_acc = (jnp.argmax(base_logits) == label).astype(jnp.float32)
            return jnp.asarray(
                [
                    selected_loss,
                    acc,
                    base_loss,
                    base_acc,
                    post_loss,
                    post_acc,
                    pre_loss,
                    pre_acc,
                ],
                dtype=jnp.float32,
            )

        rows = jax.vmap(one)(contexts, labels)
        return tuple(rows[:, idx] for idx in range(rows.shape[1]))

    (
        selected_losses,
        acc,
        base_losses,
        base_acc,
        post_losses,
        post_acc,
        pre_losses,
        pre_acc,
    ) = run()
    selected_losses.block_until_ready()
    mean_loss = float(jnp.mean(selected_losses))
    base_nll = float(jnp.mean(base_losses))
    post_nll = float(jnp.mean(post_losses))
    pre_nll = float(jnp.mean(pre_losses))
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": float(jnp.mean(acc)),
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
        "eval_fast_nll": base_nll,
        "eval_fast_accuracy": float(jnp.mean(base_acc)),
        "eval_fast_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(base_nll), 20.0))),
        "eval_post_nll": post_nll,
        "eval_post_accuracy": float(jnp.mean(post_acc)),
        "eval_post_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(post_nll), 20.0))),
        "eval_pre_nll": pre_nll,
        "eval_pre_accuracy": float(jnp.mean(pre_acc)),
        "eval_pre_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(pre_nll), 20.0))),
        "eval_pre_weight": float(jax.nn.sigmoid(state.placement_logit)),
    }


def summarize_adaptive(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize online and adaptive placement diagnostics."""
    online = summarize_online(metrics[:, :2], final_window)
    window = metrics[-min(final_window, metrics.shape[0]) :]
    online.update(
        {
            "final_window_base_nll": float(np.mean(window[:, 2])),
            "final_window_post_nll": float(np.mean(window[:, 3])),
            "final_window_pre_nll": float(np.mean(window[:, 4])),
            "final_window_post_advantage": float(np.mean(window[:, 5])),
            "final_window_pre_advantage": float(np.mean(window[:, 6])),
            "final_window_advantage": float(np.mean(window[:, 7])),
            "final_window_memory_advantage": float(np.mean(window[:, 7])),
            "final_window_placement_advantage": float(np.mean(window[:, 8])),
            "final_window_gate": float(np.mean(window[:, 9])),
            "final_window_memory_gate": float(np.mean(window[:, 9])),
            "final_window_pre_weight": float(np.mean(window[:, 10])),
            "final_window_gate_logit": float(np.mean(window[:, 11])),
            "final_window_placement_logit": float(np.mean(window[:, 12])),
            "final_window_gate_update_advantage": float(np.mean(window[:, 13])),
            "final_window_replay_memory_advantage": float(np.mean(window[:, 13])),
            "final_window_replay_placement_advantage": float(np.mean(window[:, 14])),
            "final_window_post_active_prototypes": float(np.mean(window[:, 15])),
            "final_window_pre_active_prototypes": float(np.mean(window[:, 16])),
            "final_window_post_allocation_rate": float(np.mean(window[:, 17])),
            "final_window_pre_allocation_rate": float(np.mean(window[:, 18])),
            "final_window_post_active_features": float(np.mean(window[:, 19])),
            "final_window_pre_active_features": float(np.mean(window[:, 20])),
        }
    )
    return online


def aggregate_metric(records: list[dict[str, Any]], method: str, metric: str) -> np.ndarray:
    """Collect one metric across seeds."""
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
        "static_pre_ffn_kv_memory",
        "adaptive_placement_memory",
    ]
    labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "static_post_ffn_memory": "Static Post-FFN",
        "static_pre_ffn_kv_memory": "Static Pre-KV",
        "adaptive_placement_memory": "Adaptive Placement",
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
        "# Tiny Shakespeare Placement Memory Transformer",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        f"Replay size: `{payload['config']['replay_size']}`.",
        "",
        "## Architecture and State",
        "",
        "| Method | Trainable params | Trainable bytes | Extra state elements | "
        "Extra state bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        profile = payload["profiles"][method]
        lines.append(
            f"| `{labels[method]}` | {profile['trainable_params']} | "
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

    lines.extend(["", "## Diffs vs Static Post-FFN", ""])
    lines.append("| Metric | Static Pre-KV | Adaptive Placement |")
    lines.append("|---|---:|---:|")
    for metric in metrics:
        post = aggregate_metric(records, "static_post_ffn_memory", metric)
        cells = []
        for method in methods[2:]:
            values = aggregate_metric(records, method, metric)
            diff = post - values if metric in lower else values - post
            cells.append(f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")

    lines.extend(["", "## Adaptive Diagnostics", ""])
    lines.append("| Metric | Mean +/- stderr |")
    lines.append("|---|---:|")
    adaptive_rows = [row for row in records if row["method"] == "adaptive_placement_memory"]
    for metric in [
        "final_window_memory_advantage",
        "final_window_replay_memory_advantage",
        "final_window_placement_advantage",
        "final_window_replay_placement_advantage",
        "final_window_memory_gate",
        "final_window_pre_weight",
        "final_window_gate_logit",
        "final_window_placement_logit",
        "final_window_post_active_prototypes",
        "final_window_pre_active_prototypes",
        "final_window_post_active_features",
        "final_window_pre_active_features",
        "eval_post_perplexity",
        "eval_pre_perplexity",
        "eval_pre_weight",
    ]:
        values = np.asarray([row["summary"][metric] for row in adaptive_rows], dtype=np.float64)
        lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")

    lines.append("")
    lines.append("Positive diffs favor the comparison method over static post-FFN.")
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
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--final-window", type=int, default=512)
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
    parser.add_argument("--gate-max", type=float, default=0.15)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--gate-l2", type=float, default=0.1)
    parser.add_argument(
        "--replay-size",
        type=int,
        default=128,
        help="Ring-buffer size for delayed memory and placement signals.",
    )
    parser.add_argument(
        "--train-loss-mode",
        choices=("memory", "blend"),
        default="memory",
        help="Static-memory train loss mode for the baseline static placements.",
    )
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-mode", choices=("none", "zero", "meta_ema"), default="meta_ema")
    parser.add_argument("--placement-init-logit", type=float, default=-2.0)
    parser.add_argument("--placement-lr", type=float, default=0.5)
    parser.add_argument("--placement-decay", type=float, default=0.995)
    parser.add_argument("--placement-margin", type=float, default=0.0)
    parser.add_argument("--placement-l2", type=float, default=0.01)
    parser.add_argument(
        "--selector-loss-mode",
        choices=("weighted_losses", "mixed_logits"),
        default="weighted_losses",
    )
    parser.add_argument(
        "--proto-count-is-per-placement",
        action="store_true",
        help="Give the adaptive manager this many prototypes to each placement.",
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
        default=Path("outputs/step2_new_directions/placement_memory_transformer"),
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
    if args.d_model < 1 or args.mlp_hidden < 1 or args.proto_count < 2:
        raise ValueError("--d-model and --mlp-hidden must be positive; proto-count >= 2")
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
    if not 0.0 < args.gate_max < 1.0:
        raise ValueError("--gate-max must be in (0, 1)")
    if not 0.0 <= args.memory_loss_weight <= 1.0:
        raise ValueError("--memory-loss-weight must be in [0, 1]")
    if args.replay_size < 1:
        raise ValueError("--replay-size must be positive")
    if not 0.0 <= args.placement_decay <= 1.0:
        raise ValueError("--placement-decay must be in [0, 1]")
    if args.placement_lr < 0.0 or args.placement_l2 < 0.0:
        raise ValueError("placement learning rate and l2 must be non-negative")


def main() -> None:
    """Run the placement comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = metadata["vocab_size"]
    final_window = args.final_window if args.final_window > 0 else args.eval_steps

    static_block = make_proto_block(args)
    post_count, pre_count = adaptive_proto_counts(args)
    post_block = make_sized_proto_block(args, post_count)
    pre_block = make_sized_proto_block(args, pre_count)
    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        proto_count=args.proto_count,
        adaptive_post_proto_count=post_count,
        adaptive_pre_proto_count=pre_count,
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
        replay_size=args.replay_size,
        train_loss_mode=args.train_loss_mode,
        memory_loss_weight=args.memory_loss_weight,
        reset_mode=args.reset_mode,
        placement_init_logit=args.placement_init_logit,
        placement_lr=args.placement_lr,
        placement_decay=args.placement_decay,
        placement_margin=args.placement_margin,
        placement_l2=args.placement_l2,
        selector_loss_mode=args.selector_loss_mode,
        proto_count_is_per_placement=args.proto_count_is_per_placement,
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
        block=static_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_logit=args.gate_init_logit,
        gate_mode="scalar",
    )
    adaptive_profile, adaptive_state = init_adaptive_params_state(
        profile_key,
        post_block=post_block,
        pre_block=pre_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        memory_gate_init_logit=args.gate_init_logit,
        placement_init_logit=args.placement_init_logit,
    )
    replay_state_elements = args.replay_size * args.block_size + args.replay_size + 2
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
        "static_pre_ffn_kv_memory": {
            "trainable_params": count_array_elements(static_profile),
            "trainable_bytes": count_array_bytes(static_profile),
            "state_elements": count_array_elements(static_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(static_state, include_int=True)
            + replay_state_bytes,
        },
        "adaptive_placement_memory": {
            "trainable_params": count_array_elements(adaptive_profile),
            "trainable_bytes": count_array_bytes(adaptive_profile),
            "state_elements": count_array_elements(adaptive_state, include_int=True)
            + replay_state_elements,
            "state_bytes": count_array_bytes(adaptive_state, include_int=True)
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

        for method, placement in (
            ("static_post_ffn_memory", "post_ffn"),
            ("static_pre_ffn_kv_memory", "pre_ffn_kv"),
        ):
            params, state = init_advantage_params_state(
                param_key,
                block=static_block,
                vocab_size=vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                ffn_hidden=args.mlp_hidden,
                gate_init_logit=args.gate_init_logit,
                gate_mode="scalar",
            )
            method_start = time.perf_counter()
            final_params, final_state, metrics = run_advantage_memory_transformer(
                static_block,
                params,
                state,
                contexts,
                labels,
                placement=placement,
                fast_lr=args.fast_lr,
                slow_lr=args.slow_lr,
                grad_clip=args.grad_clip,
                gate_lr=args.gate_lr,
                gate_decay=args.gate_decay,
                gate_max=args.gate_max,
                advantage_margin=args.advantage_margin,
                gate_l2=args.gate_l2,
                gate_init_logit=args.gate_init_logit,
                gate_objective="replay",
                replay_size=args.replay_size,
                train_loss_mode=args.train_loss_mode,
                memory_loss_weight=args.memory_loss_weight,
                reset_mode=args.reset_mode,
            )
            final_state.step_count.block_until_ready()
            train_s = time.perf_counter() - method_start
            summary = {
                **summarize_advantage(metrics, final_window),
                **eval_advantage_memory_transformer(
                    static_block,
                    final_params,
                    final_state,
                    eval_contexts,
                    eval_labels,
                    placement=placement,
                ),
                **eval_advantage_fast_only(
                    static_block,
                    final_params,
                    final_state,
                    eval_contexts,
                    eval_labels,
                    placement=placement,
                ),
                "train_s": train_s,
                "train_steps_per_s": args.steps / train_s,
            }
            records.append({"seed": seed_idx, "method": method, "summary": summary})
            print(
                f"seed={seed_idx} {method}: fw_nll={summary['final_window_nll']:.3f}, "
                f"eval_ppl={summary['eval_perplexity']:.2f}, "
                f"gate={summary['final_window_gate']:.3f}, train_s={train_s:.2f}"
            )

        adaptive_params, adaptive_initial_state = init_adaptive_params_state(
            param_key,
            post_block=post_block,
            pre_block=pre_block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            memory_gate_init_logit=args.gate_init_logit,
            placement_init_logit=args.placement_init_logit,
        )
        method_start = time.perf_counter()
        final_adaptive, final_adaptive_state, adaptive_metrics = (
            run_adaptive_placement_memory_transformer(
                post_block,
                pre_block,
                adaptive_params,
                adaptive_initial_state,
                contexts,
                labels,
                fast_lr=args.fast_lr,
                slow_lr=args.slow_lr,
                grad_clip=args.grad_clip,
                gate_lr=args.gate_lr,
                gate_decay=args.gate_decay,
                gate_max=args.gate_max,
                advantage_margin=args.advantage_margin,
                gate_l2=args.gate_l2,
                replay_size=args.replay_size,
                reset_mode=args.reset_mode,
                placement_init_logit=args.placement_init_logit,
                placement_lr=args.placement_lr,
                placement_decay=args.placement_decay,
                placement_margin=args.placement_margin,
                placement_l2=args.placement_l2,
                selector_loss_mode=args.selector_loss_mode,
            )
        )
        final_adaptive_state.step_count.block_until_ready()
        train_s = time.perf_counter() - method_start
        summary = {
            **summarize_adaptive(adaptive_metrics, final_window),
            **eval_adaptive_placement_memory_transformer(
                post_block,
                pre_block,
                final_adaptive,
                final_adaptive_state,
                eval_contexts,
                eval_labels,
                selector_loss_mode=args.selector_loss_mode,
            ),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {"seed": seed_idx, "method": "adaptive_placement_memory", "summary": summary}
        )
        print(
            f"seed={seed_idx} adaptive: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, "
            f"pre_w={summary['final_window_pre_weight']:.3f}, "
            f"gate={summary['final_window_memory_gate']:.3f}, train_s={train_s:.2f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "static_prototype_block": static_block.to_config(),
        "adaptive_post_prototype_block": post_block.to_config(),
        "adaptive_pre_prototype_block": pre_block.to_config(),
        "profiles": profiles,
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

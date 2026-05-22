#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare learned-gate prototype transformer comparison.

This experiment is a local Step 2 probe derived from
``step2_tiny_shakespeare_proto_basis_transformer.py``. It keeps the same tiny
character-level transformer scaffold and compares:

1. ``baseline_ffn_transformer``: causal attention, GELU FFN residual, readout.
2. ``ungated_hybrid_proto_transformer``: FFN plus an always-on prototype
   residual.
3. ``gated_hybrid_proto_transformer``: FFN plus a learned gate on the prototype
   residual.

The learned gate sees online signals available before the token update:
prototype novelty, recent token loss EMA, and uncertainty summaries from the
current base logits. Learning rates are intentionally decoupled: attention,
FFN, and readout use ``--model-lr``; prototype values use
``--proto-value-lr``; gate parameters use ``--gate-lr``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_tiny_shakespeare_proto_basis_transformer import (
    make_proto_block,
    reset_value_row_if_novel,
    select_center_slot,
    summarize_proto_diagnostics,
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
    PrototypeBasisParams,
    PrototypeBasisState,
)

GATE_FEATURE_NAMES = (
    "bias",
    "novelty_binary",
    "novelty_distance_ratio",
    "recent_loss_log1p",
    "entropy_norm",
    "one_minus_max_prob",
    "margin",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured into the result artifact."""

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
    model_lr: float
    proto_value_lr: float
    gate_lr: float
    grad_clip: float
    proto_update_rate: float
    proto_novelty_threshold: float
    proto_bandwidth: float
    proto_adaptive_bandwidth: bool
    proto_bandwidth_update_rate: float
    reset_new_prototype_values: bool
    gate_mode: str
    gate_groups: int
    gate_bias_init: float
    gate_loss_decay: float
    gate_initial_loss: float
    data_path: str
    output_dir: str
    seed: int


def init_hybrid_transformer(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
) -> tuple[dict[str, Any], PrototypeBasisState]:
    """Initialize a standard FFN transformer plus slow prototype residual."""
    proto_key = jr.fold_in(key, 17)
    proto_params, proto_state = block.init(proto_key)
    params = init_transformer_params(
        key,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
    )
    return {**params, "proto": proto_params}, proto_state


def init_gate_params(
    key: jax.Array,
    *,
    gate_mode: str,
    gate_input_dim: int,
    d_model: int,
    gate_groups: int,
    gate_bias_init: float,
) -> dict[str, jax.Array]:
    """Initialize residual-gate weights for scalar, channel, or group gates."""
    if gate_mode == "scalar":
        out_dim = 1
    elif gate_mode == "channel":
        out_dim = d_model
    elif gate_mode == "group":
        out_dim = gate_groups
    else:
        msg = f"unknown gate mode: {gate_mode}"
        raise ValueError(msg)
    return {
        "w": 0.01 * jr.normal(key, (gate_input_dim, out_dim), dtype=jnp.float32),
        "b": jnp.full((out_dim,), gate_bias_init, dtype=jnp.float32),
    }


def init_gate_state(initial_loss: float) -> dict[str, jax.Array]:
    """Initialize online gate context state."""
    return {
        "loss_ema": jnp.asarray(initial_loss, dtype=jnp.float32),
        "step_count": jnp.array(0, dtype=jnp.int32),
    }


def init_gated_hybrid_transformer(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    gate_mode: str,
    gate_groups: int,
    gate_bias_init: float,
) -> tuple[dict[str, Any], PrototypeBasisState, dict[str, jax.Array]]:
    """Initialize the gated hybrid transformer."""
    hybrid_key, gate_key = jr.split(key, 2)
    params, proto_state = init_hybrid_transformer(
        hybrid_key,
        block=block,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
    )
    params = {
        **params,
        "gate": init_gate_params(
            gate_key,
            gate_mode=gate_mode,
            gate_input_dim=len(GATE_FEATURE_NAMES),
            d_model=d_model,
            gate_groups=gate_groups,
            gate_bias_init=gate_bias_init,
        ),
    }
    return params, proto_state, init_gate_state(math.log(vocab_size))


def partitioned_sgd_step(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    model_step_size: float,
    proto_value_step_size: float,
    gate_step_size: float,
) -> dict[str, Any]:
    """Apply separate step sizes to model, prototype value, and gate leaves."""
    new_params: dict[str, Any] = {}
    for name, value in params.items():
        if name == "proto":
            step_size = proto_value_step_size
        elif name == "gate":
            step_size = gate_step_size
        else:
            step_size = model_step_size
        new_params[name] = jax.tree_util.tree_map(
            lambda p, g: p - step_size * g,
            value,
            grads[name],
        )
    return new_params


def hybrid_transformer_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return logits for FFN plus ungated prototype residual transformer."""
    hidden = causal_attention_sequence(params["attn"], context)[-1]
    basis_input = ffn_transform(params["ffn"], hidden)
    activations = block.activations(state, basis_input)
    hidden = basis_input + block.transform(params["proto"], activations)
    readout = params["readout"]
    logits = hidden @ readout["w"] + readout["b"]
    return logits, basis_input, activations


def _uncertainty_from_logits(logits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    probs = jax.nn.softmax(logits)
    vocab = jnp.asarray(logits.shape[0], dtype=jnp.float32)
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-12)))
    entropy_norm = entropy / jnp.maximum(jnp.log(vocab), 1e-12)
    sorted_probs = jnp.sort(probs)
    max_prob = sorted_probs[-1]
    second_prob = sorted_probs[-2]
    margin = max_prob - second_prob
    return entropy_norm, max_prob, margin


def gate_inputs(
    block: PrototypeBasisBlock,
    state: PrototypeBasisState,
    hidden: jax.Array,
    base_logits: jax.Array,
    loss_ema: jax.Array,
) -> jax.Array:
    """Build gate inputs from novelty, recent loss, and logit uncertainty."""
    _, novel = select_center_slot(block, state, hidden)
    used = state.counts > 0.0
    distances = jnp.mean((state.centers - hidden[None, :]) ** 2, axis=1)
    used_distances = jnp.where(used, distances, jnp.inf)
    nearest_distance = jnp.min(used_distances)
    has_used = jnp.any(used)
    nearest_distance = jnp.where(has_used, nearest_distance, jnp.array(0.0, dtype=jnp.float32))
    threshold = jnp.maximum(
        jnp.asarray(block.config.novelty_threshold, dtype=jnp.float32),
        jnp.array(1e-6, dtype=jnp.float32),
    )
    novelty_ratio = jnp.clip(nearest_distance / threshold, 0.0, 10.0)
    entropy_norm, max_prob, margin = _uncertainty_from_logits(base_logits)
    return jnp.stack(
        [
            jnp.array(1.0, dtype=jnp.float32),
            novel.astype(jnp.float32),
            jnp.log1p(novelty_ratio),
            jnp.log1p(jnp.maximum(loss_ema, 0.0)),
            entropy_norm,
            1.0 - max_prob,
            margin,
        ]
    )


def _group_ids(proto_count: int, gate_groups: int) -> jax.Array:
    groups = max(1, min(proto_count, gate_groups))
    return (jnp.arange(proto_count, dtype=jnp.int32) * groups) // proto_count


def gated_proto_residual(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    activations: jax.Array,
    features: jax.Array,
    *,
    gate_mode: str,
    gate_groups: int,
) -> tuple[jax.Array, jax.Array]:
    """Apply a learned gate to the prototype residual."""
    gate = jax.nn.sigmoid(features @ params["gate"]["w"] + params["gate"]["b"])
    proto_params: PrototypeBasisParams = params["proto"]
    if gate_mode == "group":
        group_ids = _group_ids(activations.shape[0], gate_groups)
        proto_gates = gate[group_ids]
        residual = (activations * proto_gates) @ proto_params.values
        residual = residual + jnp.mean(gate) * proto_params.bias
        return residual, gate
    residual = block.transform(proto_params, activations)
    if gate_mode == "scalar":
        return residual * gate[0], gate
    return residual * gate, gate


def gated_hybrid_transformer_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    gate_state: dict[str, jax.Array],
    context: jax.Array,
    *,
    gate_mode: str,
    gate_groups: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return logits and diagnostics for the gated hybrid transformer."""
    hidden = causal_attention_sequence(params["attn"], context)[-1]
    basis_input = ffn_transform(params["ffn"], hidden)
    readout = params["readout"]
    base_logits = basis_input @ readout["w"] + readout["b"]
    features = gate_inputs(block, state, basis_input, base_logits, gate_state["loss_ema"])
    activations = block.activations(state, basis_input)
    residual, gate = gated_proto_residual(
        block,
        params,
        activations,
        features,
        gate_mode=gate_mode,
        gate_groups=gate_groups,
    )
    logits = (basis_input + residual) @ readout["w"] + readout["b"]
    return logits, basis_input, activations, gate, features


def update_gate_state(
    gate_state: dict[str, jax.Array],
    loss: jax.Array,
    *,
    gate_loss_decay: float,
) -> dict[str, jax.Array]:
    """Update recent token-loss EMA used as a gate input."""
    decay = jnp.asarray(gate_loss_decay, dtype=jnp.float32)
    loss_ema = decay * gate_state["loss_ema"] + (1.0 - decay) * loss
    return {
        "loss_ema": loss_ema,
        "step_count": gate_state["step_count"] + jnp.array(1, dtype=jnp.int32),
    }


def run_ungated_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    model_step_size: float,
    proto_value_step_size: float,
    grad_clip: float,
    reset_new_prototype_values: bool,
) -> tuple[dict[str, Any], PrototypeBasisState, np.ndarray]:
    """Train the ungated FFN plus prototype residual transformer."""

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: PrototypeBasisState,
    ) -> tuple[tuple[dict[str, Any], PrototypeBasisState], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], PrototypeBasisState],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], PrototypeBasisState], jax.Array]:
            params, state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
                logits, hidden, activations = hybrid_transformer_logits(
                    block,
                    candidate,
                    state,
                    context,
                )
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    hidden,
                    activations,
                )

            (loss, (logits, hidden, activations)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grads = clip_grads(grads, grad_clip)
            new_params = partitioned_sgd_step(
                params,
                grads,
                model_step_size=model_step_size,
                proto_value_step_size=proto_value_step_size,
                gate_step_size=0.0,
            )
            slot, novel = select_center_slot(block, state, hidden)
            new_state, center_metrics = block.update_centers(state, hidden)
            if reset_new_prototype_values:
                new_params = reset_value_row_if_novel(new_params, slot, novel)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def run_gated_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    gate_state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    model_step_size: float,
    proto_value_step_size: float,
    gate_step_size: float,
    grad_clip: float,
    reset_new_prototype_values: bool,
    gate_mode: str,
    gate_groups: int,
    gate_loss_decay: float,
) -> tuple[dict[str, Any], PrototypeBasisState, dict[str, jax.Array], np.ndarray]:
    """Train the gated FFN plus prototype residual transformer."""

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: PrototypeBasisState,
        gate_state: dict[str, jax.Array],
    ) -> tuple[tuple[dict[str, Any], PrototypeBasisState, dict[str, jax.Array]], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], PrototypeBasisState, dict[str, jax.Array]],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], PrototypeBasisState, dict[str, jax.Array]], jax.Array]:
            params, state, gate_state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
                logits, hidden, activations, gate, features = gated_hybrid_transformer_logits(
                    block,
                    candidate,
                    state,
                    gate_state,
                    context,
                    gate_mode=gate_mode,
                    gate_groups=gate_groups,
                )
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    hidden,
                    activations,
                    gate,
                    features,
                )

            (loss, (logits, hidden, activations, gate, features)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(params)
            )
            grads = clip_grads(grads, grad_clip)
            new_params = partitioned_sgd_step(
                params,
                grads,
                model_step_size=model_step_size,
                proto_value_step_size=proto_value_step_size,
                gate_step_size=gate_step_size,
            )
            slot, novel = select_center_slot(block, state, hidden)
            new_state, center_metrics = block.update_centers(state, hidden)
            if reset_new_prototype_values:
                new_params = reset_value_row_if_novel(new_params, slot, novel)
            new_gate_state = update_gate_state(
                gate_state,
                loss,
                gate_loss_decay=gate_loss_decay,
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                    jnp.mean(gate),
                    jnp.min(gate),
                    jnp.max(gate),
                    new_gate_state["loss_ema"],
                    features[1],
                    features[4],
                    1.0 - features[5],
                    features[6],
                ]
            )
            return (new_params, new_state, new_gate_state), metrics

        return jax.lax.scan(step, (params, state, gate_state), (contexts, labels))

    (final_params, final_state, final_gate_state), metrics = scan(params, state, gate_state)
    metrics.block_until_ready()
    return final_params, final_state, final_gate_state, np.asarray(metrics)


def eval_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    """Evaluate the ungated hybrid transformer on held-out examples."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(lambda ctx: hybrid_transformer_logits(block, params, state, ctx)[0])(
            contexts
        )
        losses = jax.vmap(cross_entropy_from_logits)(logits, labels)
        acc = jnp.argmax(logits, axis=1) == labels
        return losses, acc.astype(jnp.float32)

    losses, acc = run()
    losses.block_until_ready()
    mean_loss = float(jnp.mean(losses))
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": float(jnp.mean(acc)),
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def eval_gated_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    gate_state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    gate_mode: str,
    gate_groups: int,
) -> dict[str, float]:
    """Evaluate the gated hybrid transformer on held-out examples."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(
            lambda ctx: gated_hybrid_transformer_logits(
                block,
                params,
                state,
                gate_state,
                ctx,
                gate_mode=gate_mode,
                gate_groups=gate_groups,
            )[0]
        )(contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, labels)
        acc = jnp.argmax(logits, axis=1) == labels
        return losses, acc.astype(jnp.float32)

    losses, acc = run()
    losses.block_until_ready()
    mean_loss = float(jnp.mean(losses))
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": float(jnp.mean(acc)),
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def summarize_gate_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize learned gate diagnostics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_gate_mean": float(np.mean(window[:, 6])),
        "final_window_gate_min": float(np.mean(window[:, 7])),
        "final_window_gate_max": float(np.mean(window[:, 8])),
        "final_loss_ema": float(metrics[-1, 9]),
        "novelty_rate": float(np.mean(metrics[:, 10])),
        "final_window_entropy_norm": float(np.mean(window[:, 11])),
        "final_window_max_prob": float(np.mean(window[:, 12])),
        "final_window_margin": float(np.mean(window[:, 13])),
    }


def aggregate_metric(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
) -> np.ndarray:
    """Collect one metric across seeds."""
    return np.asarray(
        [row["summary"][metric] for row in records if row["method"] == method],
        dtype=np.float64,
    )


def _format_metric(records: list[dict[str, Any]], method: str, metric: str) -> str:
    values = aggregate_metric(records, method, metric)
    return f"{np.mean(values):.4f} +/- {stderr(values):.4f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown result summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "ungated_hybrid_proto_transformer",
        "gated_hybrid_proto_transformer",
    ]
    method_labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "ungated_hybrid_proto_transformer": "Ungated Hybrid",
        "gated_hybrid_proto_transformer": "Gated Hybrid",
    }
    metrics = [
        "final_window_nll",
        "final_window_accuracy",
        "final_window_perplexity",
        "eval_nll",
        "eval_accuracy",
        "eval_perplexity",
        "train_s",
        "train_steps_per_s",
    ]
    lower_is_better = {
        "final_window_nll",
        "final_window_perplexity",
        "eval_nll",
        "eval_perplexity",
        "train_s",
    }
    lines = [
        "# Tiny Shakespeare Gated Prototype Transformer",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "The gate is a learned logistic residual gate over the prototype block.",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        f"Gate mode: `{payload['config']['gate_mode']}`.",
        f"Block size: `{payload['config']['block_size']}`. Vocab: `{payload['vocab_size']}`.",
        "",
        "## Architecture and state",
        "",
        "| Method | Trainable params | Trainable bytes | "
        "Extra state elements | Extra state bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        profile = payload["profiles"][method]
        lines.append(
            f"| `{method_labels[method]}` | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    metric_header = "| Metric | " + " | ".join(method_labels[m] for m in methods) + " |"
    metric_separator = "|---|" + "---:|" * len(methods)
    lines.extend(["", "## Metrics", "", metric_header, metric_separator])
    for metric in metrics:
        cells = [_format_metric(records, method, metric) for method in methods]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Diffs vs Baseline",
            "",
            "| Metric | Ungated Hybrid | Gated Hybrid |",
            "|---|---:|---:|",
        ]
    )
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        diff_cells = []
        for method in ("ungated_hybrid_proto_transformer", "gated_hybrid_proto_transformer"):
            values = aggregate_metric(records, method, metric)
            diff = baseline - values if metric in lower_is_better else values - baseline
            diff_cells.append(f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(diff_cells) + " |")
    for method in ("ungated_hybrid_proto_transformer", "gated_hybrid_proto_transformer"):
        method_records = [row for row in records if row["method"] == method]
        lines.extend(
            [
                "",
                f"## {method_labels[method]} Prototype Diagnostics",
                "",
                "| Metric | Mean +/- stderr |",
                "|---|---:|",
            ]
        )
        for metric in [
            "final_window_active_features",
            "final_window_active_prototypes",
            "allocation_rate",
            "final_window_allocation_rate",
            "final_window_nearest_distance",
        ]:
            values = np.asarray(
                [row["summary"][metric] for row in method_records],
                dtype=np.float64,
            )
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    gated_records = [row for row in records if row["method"] == "gated_hybrid_proto_transformer"]
    lines.extend(["", "## Gate Diagnostics", "", "| Metric | Mean +/- stderr |", "|---|---:|"])
    for metric in [
        "final_window_gate_mean",
        "final_window_gate_min",
        "final_window_gate_max",
        "final_loss_ema",
        "novelty_rate",
        "final_window_entropy_norm",
        "final_window_max_prob",
        "final_window_margin",
    ]:
        values = np.asarray([row["summary"][metric] for row in gated_records], dtype=np.float64)
        lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    lines.extend(
        [
            "",
            "Positive diffs favor the hybrid method over the FFN baseline. "
            "Wall-clock includes JAX compilation on the first seed for each method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
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
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--model-lr", type=float, default=0.03)
    parser.add_argument("--proto-value-lr", type=float, default=0.03)
    parser.add_argument("--gate-lr", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.08)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--proto-adaptive-bandwidth", action="store_true")
    parser.add_argument("--proto-bandwidth-update-rate", type=float, default=0.1)
    parser.add_argument(
        "--no-reset-new-prototype-values",
        action="store_false",
        dest="reset_new_prototype_values",
    )
    parser.add_argument("--gate-mode", choices=("scalar", "channel", "group"), default="channel")
    parser.add_argument("--gate-groups", type=int, default=8)
    parser.add_argument("--gate-bias-init", type=float, default=-0.5)
    parser.add_argument("--gate-loss-decay", type=float, default=0.98)
    parser.add_argument("--gate-initial-loss", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/gated_proto_transformer_worker"),
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.d_model < 1 or args.mlp_hidden < 1 or args.proto_count < 1:
        raise ValueError("--d-model, --mlp-hidden, and --proto-count must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    for name in ("baseline_lr", "model_lr", "proto_value_lr", "gate_lr"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
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
    if args.gate_groups < 1:
        raise ValueError("--gate-groups must be positive")
    if not 0.0 <= args.gate_loss_decay < 1.0:
        raise ValueError("--gate-loss-decay must be in [0, 1)")
    if args.gate_initial_loss < 0.0:
        raise ValueError("--gate-initial-loss must be non-negative")


def _record(
    records: list[dict[str, Any]],
    *,
    seed_idx: int,
    method: str,
    summary: dict[str, float],
) -> None:
    records.append({"seed": seed_idx, "method": method, "summary": summary})


def main() -> None:
    """Run the transformer block comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = metadata["vocab_size"]
    final_window = args.final_window if args.final_window > 0 else args.eval_steps
    proto_block = make_proto_block(args)
    gate_initial_loss = args.gate_initial_loss or math.log(vocab_size)

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
        model_lr=args.model_lr,
        proto_value_lr=args.proto_value_lr,
        gate_lr=args.gate_lr,
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=args.proto_adaptive_bandwidth,
        proto_bandwidth_update_rate=args.proto_bandwidth_update_rate,
        reset_new_prototype_values=args.reset_new_prototype_values,
        gate_mode=args.gate_mode,
        gate_groups=args.gate_groups,
        gate_bias_init=args.gate_bias_init,
        gate_loss_decay=args.gate_loss_decay,
        gate_initial_loss=gate_initial_loss,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    profile_key = jr.fold_in(root, 999)
    baseline_profile_params = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    ungated_profile_params, ungated_profile_state = init_hybrid_transformer(
        profile_key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    gated_profile_params, gated_profile_state, gated_profile_gate_state = (
        init_gated_hybrid_transformer(
            profile_key,
            block=proto_block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            gate_mode=args.gate_mode,
            gate_groups=args.gate_groups,
            gate_bias_init=args.gate_bias_init,
        )
    )
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "ungated_hybrid_proto_transformer": {
            "trainable_params": count_array_elements(ungated_profile_params),
            "trainable_bytes": count_array_bytes(ungated_profile_params),
            "state_elements": count_array_elements(ungated_profile_state, include_int=True),
            "state_bytes": count_array_bytes(ungated_profile_state, include_int=True),
        },
        "gated_hybrid_proto_transformer": {
            "trainable_params": count_array_elements(gated_profile_params),
            "trainable_bytes": count_array_bytes(gated_profile_params),
            "state_elements": count_array_elements(gated_profile_state, include_int=True)
            + count_array_elements(gated_profile_gate_state, include_int=True),
            "state_bytes": count_array_bytes(gated_profile_state, include_int=True)
            + count_array_bytes(gated_profile_gate_state, include_int=True),
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
        ungated_params, ungated_state = init_hybrid_transformer(
            param_key,
            block=proto_block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
        )
        gated_params, gated_state, gate_state = init_gated_hybrid_transformer(
            param_key,
            block=proto_block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            gate_mode=args.gate_mode,
            gate_groups=args.gate_groups,
            gate_bias_init=args.gate_bias_init,
        )
        gate_state = init_gate_state(gate_initial_loss)

        method_start = time.perf_counter()
        final_baseline, baseline_metrics = run_baseline_transformer(
            baseline_params,
            contexts,
            labels,
            step_size=args.baseline_lr,
            grad_clip=args.grad_clip,
        )
        baseline_train_s = time.perf_counter() - method_start
        baseline_summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_transformer(final_baseline, eval_contexts, eval_labels),
            "train_s": baseline_train_s,
            "train_steps_per_s": args.steps / baseline_train_s,
        }
        _record(
            records,
            seed_idx=seed_idx,
            method="baseline_ffn_transformer",
            summary=baseline_summary,
        )
        print(
            f"seed={seed_idx} baseline_ffn_transformer: "
            f"fw_nll={baseline_summary['final_window_nll']:.3f}, "
            f"eval_ppl={baseline_summary['eval_perplexity']:.2f}, "
            f"train_s={baseline_train_s:.2f}"
        )

        method_start = time.perf_counter()
        final_ungated, final_ungated_state, ungated_metrics = (
            run_ungated_hybrid_proto_transformer(
                proto_block,
                ungated_params,
                ungated_state,
                contexts,
                labels,
                model_step_size=args.model_lr,
                proto_value_step_size=args.proto_value_lr,
                grad_clip=args.grad_clip,
                reset_new_prototype_values=args.reset_new_prototype_values,
            )
        )
        final_ungated_state.step_count.block_until_ready()
        ungated_train_s = time.perf_counter() - method_start
        ungated_summary = {
            **summarize_online(ungated_metrics, final_window),
            **summarize_proto_diagnostics(ungated_metrics, final_window),
            **eval_hybrid_proto_transformer(
                proto_block,
                final_ungated,
                final_ungated_state,
                eval_contexts,
                eval_labels,
            ),
            "train_s": ungated_train_s,
            "train_steps_per_s": args.steps / ungated_train_s,
        }
        _record(
            records,
            seed_idx=seed_idx,
            method="ungated_hybrid_proto_transformer",
            summary=ungated_summary,
        )
        print(
            f"seed={seed_idx} ungated_hybrid_proto_transformer: "
            f"fw_nll={ungated_summary['final_window_nll']:.3f}, "
            f"eval_ppl={ungated_summary['eval_perplexity']:.2f}, "
            f"train_s={ungated_train_s:.2f}, "
            f"active={ungated_summary['final_window_active_prototypes']:.1f}"
        )

        method_start = time.perf_counter()
        final_gated, final_gated_state, final_gate_state, gated_metrics = (
            run_gated_hybrid_proto_transformer(
                proto_block,
                gated_params,
                gated_state,
                gate_state,
                contexts,
                labels,
                model_step_size=args.model_lr,
                proto_value_step_size=args.proto_value_lr,
                gate_step_size=args.gate_lr,
                grad_clip=args.grad_clip,
                reset_new_prototype_values=args.reset_new_prototype_values,
                gate_mode=args.gate_mode,
                gate_groups=args.gate_groups,
                gate_loss_decay=args.gate_loss_decay,
            )
        )
        final_gated_state.step_count.block_until_ready()
        gated_train_s = time.perf_counter() - method_start
        gated_summary = {
            **summarize_online(gated_metrics, final_window),
            **summarize_proto_diagnostics(gated_metrics, final_window),
            **summarize_gate_diagnostics(gated_metrics, final_window),
            **eval_gated_hybrid_proto_transformer(
                proto_block,
                final_gated,
                final_gated_state,
                final_gate_state,
                eval_contexts,
                eval_labels,
                gate_mode=args.gate_mode,
                gate_groups=args.gate_groups,
            ),
            "train_s": gated_train_s,
            "train_steps_per_s": args.steps / gated_train_s,
        }
        _record(
            records,
            seed_idx=seed_idx,
            method="gated_hybrid_proto_transformer",
            summary=gated_summary,
        )
        print(
            f"seed={seed_idx} gated_hybrid_proto_transformer: "
            f"fw_nll={gated_summary['final_window_nll']:.3f}, "
            f"eval_ppl={gated_summary['eval_perplexity']:.2f}, "
            f"train_s={gated_train_s:.2f}, "
            f"gate={gated_summary['final_window_gate_mean']:.3f}, "
            f"active={gated_summary['final_window_active_prototypes']:.1f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "prototype_block": proto_block.to_config(),
        "profiles": profiles,
        "elapsed_s": time.perf_counter() - start,
        "records": records,
        "note": (
            "The gated method learns a logistic residual gate from prototype "
            "novelty, recent token-loss EMA, and uncertainty from the current "
            "base logits. Model, prototype-value, and gate learning rates are "
            "separate CLI parameters."
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

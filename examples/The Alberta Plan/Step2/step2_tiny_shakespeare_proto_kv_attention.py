#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare prototype-KV attention memory experiment.

This is a Step 2 transformer-memory prototype. It keeps the Tiny Shakespeare
online next-token setup from the nearby FFN transformer scripts, but inserts a
small prototype key/value memory into the attention path:

``causal attention -> retrieve prototype memory value by final query -> readout``

and the hybrid:

``causal attention -> retrieve prototype memory value by final query -> FFN -> readout``.

Memory value rows and a scalar retrieval gate are trained by next-token
cross-entropy gradients. Memory centers are non-gradient prototype keys updated
online every token from the final attention query.
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
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    clip_grads,
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    encode_text,
    ensure_tiny_shakespeare,
    eval_transformer,
    ffn_transform,
    init_attention_params,
    init_readout_params,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    sgd_step,
    stderr,
    summarize_online,
)

ArrayTree = dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured into the result artifact."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    memory_slots: int
    eval_steps: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    adapter_lr: float
    grad_clip: float
    memory_update_rate: float
    memory_novelty_threshold: float
    memory_temperature: float
    memory_value_init_scale: float
    gate_init: float
    reset_new_memory_values: bool
    data_path: str
    output_dir: str
    seed: int


def causal_attention_sequence_with_query(
    attn: dict[str, jax.Array],
    context: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return causal attention hidden states and raw query vectors."""
    x = attn["token_embed"][context] + attn["pos_embed"]
    q = x @ attn["wq"]
    k = x @ attn["wk"]
    v = x @ attn["wv"]
    scores = (q @ k.T) / jnp.sqrt(jnp.asarray(x.shape[-1], dtype=jnp.float32))
    mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), dtype=bool))
    scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = (weights @ v) @ attn["wo"]
    return x + attended, q


def _logit_from_probability(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def init_memory_params(
    key: jax.Array,
    *,
    memory_slots: int,
    d_model: int,
    value_init_scale: float,
    gate_init: float,
) -> dict[str, jax.Array]:
    """Initialize trainable memory values and scalar retrieval gate."""
    return {
        "values": value_init_scale * jr.normal(key, (memory_slots, d_model)),
        "gate_logit": jnp.asarray(_logit_from_probability(gate_init), dtype=jnp.float32),
    }


def init_memory_state(*, memory_slots: int, d_model: int) -> dict[str, jax.Array]:
    """Initialize empty non-gradient prototype key state."""
    return {
        "centers": jnp.zeros((memory_slots, d_model), dtype=jnp.float32),
        "counts": jnp.zeros((memory_slots,), dtype=jnp.float32),
        "last_update": jnp.zeros((memory_slots,), dtype=jnp.int32),
        "step_count": jnp.array(0, dtype=jnp.int32),
    }


def init_proto_kv_attention_params(
    key: jax.Array,
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    memory_slots: int,
    memory_value_init_scale: float,
    gate_init: float,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Initialize attention, trainable prototype values, readout, and state."""
    attn_key, memory_key, readout_key = jr.split(key, 3)
    params = {
        "attn": init_attention_params(attn_key, vocab_size, block_size, d_model),
        "memory": init_memory_params(
            memory_key,
            memory_slots=memory_slots,
            d_model=d_model,
            value_init_scale=memory_value_init_scale,
            gate_init=gate_init,
        ),
        "readout": init_readout_params(readout_key, d_model, vocab_size),
    }
    return params, init_memory_state(memory_slots=memory_slots, d_model=d_model)


def init_hybrid_proto_kv_params(
    key: jax.Array,
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    memory_slots: int,
    memory_value_init_scale: float,
    gate_init: float,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Initialize FFN transformer plus trainable prototype-KV memory."""
    attn_key, ffn_key, memory_key, readout_key = jr.split(key, 4)
    params = {
        "attn": init_attention_params(attn_key, vocab_size, block_size, d_model),
        "ffn": init_transformer_params(
            ffn_key,
            vocab_size=vocab_size,
            block_size=block_size,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
        )["ffn"],
        "memory": init_memory_params(
            memory_key,
            memory_slots=memory_slots,
            d_model=d_model,
            value_init_scale=memory_value_init_scale,
            gate_init=gate_init,
        ),
        "readout": init_readout_params(readout_key, d_model, vocab_size),
    }
    return params, init_memory_state(memory_slots=memory_slots, d_model=d_model)


def memory_activations(
    state: dict[str, jax.Array],
    query: jax.Array,
    *,
    temperature: float,
) -> tuple[jax.Array, jax.Array]:
    """Return similarity weights and nearest used-center distance."""
    used = state["counts"] > 0.0
    has_used = jnp.any(used)
    distances = jnp.mean((state["centers"] - query[None, :]) ** 2, axis=1)
    scores = -distances / jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), 1e-8)
    scores = jnp.where(used, scores, -1e9)
    weights = jnp.where(
        has_used,
        jax.nn.softmax(scores, axis=0),
        jnp.zeros_like(scores),
    )
    nearest_distance = jnp.min(jnp.where(used, distances, jnp.inf))
    nearest_distance = jnp.where(has_used, nearest_distance, jnp.array(0.0, dtype=jnp.float32))
    return weights, nearest_distance


def retrieve_memory(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    query: jax.Array,
    *,
    temperature: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Retrieve a gated memory value from prototype keys."""
    weights, nearest_distance = memory_activations(state, query, temperature=temperature)
    raw_value = weights @ params["memory"]["values"]
    gate = jax.nn.sigmoid(params["memory"]["gate_logit"])
    return gate * raw_value, weights, gate, nearest_distance


def proto_kv_attention_logits(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    context: jax.Array,
    *,
    temperature: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return logits and prototype-KV diagnostics for the adapter model."""
    hidden_sequence, query_sequence = causal_attention_sequence_with_query(
        params["attn"],
        context,
    )
    query = query_sequence[-1]
    memory_value, weights, gate, nearest_distance = retrieve_memory(
        params,
        state,
        query,
        temperature=temperature,
    )
    hidden = hidden_sequence[-1] + memory_value
    readout = params["readout"]
    logits = hidden @ readout["w"] + readout["b"]
    return logits, query, weights, gate, nearest_distance


def hybrid_proto_kv_logits(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    context: jax.Array,
    *,
    temperature: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return logits and prototype-KV diagnostics for the hybrid model."""
    hidden_sequence, query_sequence = causal_attention_sequence_with_query(
        params["attn"],
        context,
    )
    query = query_sequence[-1]
    memory_value, weights, gate, nearest_distance = retrieve_memory(
        params,
        state,
        query,
        temperature=temperature,
    )
    hidden = ffn_transform(params["ffn"], hidden_sequence[-1] + memory_value)
    readout = params["readout"]
    logits = hidden @ readout["w"] + readout["b"]
    return logits, query, weights, gate, nearest_distance


def _replacement_slot(state: dict[str, jax.Array]) -> jax.Array:
    """Choose a least-used and then oldest prototype slot."""
    min_count = jnp.min(state["counts"])
    tied = state["counts"] <= (min_count + 1e-6)
    oldest = jnp.where(
        tied,
        state["last_update"],
        jnp.array(2_147_483_647, dtype=state["last_update"].dtype),
    )
    return jnp.argmin(oldest)


def select_memory_slot(
    state: dict[str, jax.Array],
    query: jax.Array,
    *,
    novelty_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Choose the center slot that the online query update will touch."""
    used = state["counts"] > 0.0
    has_used = jnp.any(used)
    has_empty = jnp.any(~used)
    distances = jnp.mean((state["centers"] - query[None, :]) ** 2, axis=1)
    used_distances = jnp.where(used, distances, jnp.inf)
    nearest_slot = jnp.argmin(used_distances)
    nearest_distance = used_distances[nearest_slot]
    empty_slot = jnp.argmax((~used).astype(jnp.int32))
    novel = (~has_used) | (
        nearest_distance > jnp.asarray(novelty_threshold, dtype=jnp.float32)
    )
    slot = jnp.where(
        ~has_used,
        jnp.array(0, dtype=nearest_slot.dtype),
        jnp.where(
            novel & has_empty,
            empty_slot,
            jnp.where(novel, _replacement_slot(state), nearest_slot),
        ),
    )
    nearest_metric = jnp.where(
        has_used,
        nearest_distance,
        jnp.array(0.0, dtype=jnp.float32),
    )
    return slot, novel, nearest_metric


def update_memory_centers(
    state: dict[str, jax.Array],
    query: jax.Array,
    *,
    update_rate: float,
    novelty_threshold: float,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Update one memory key center from the current token query."""
    slot, novel, nearest_distance = select_memory_slot(
        state,
        query,
        novelty_threshold=novelty_threshold,
    )
    eta = jnp.asarray(update_rate, dtype=jnp.float32)
    old_center = state["centers"][slot]
    new_center = jnp.where(novel, query, old_center + eta * (query - old_center))
    new_count = jnp.where(novel, 1.0, state["counts"][slot] + 1.0)
    new_state = {
        "centers": state["centers"].at[slot].set(new_center),
        "counts": state["counts"].at[slot].set(new_count),
        "last_update": state["last_update"].at[slot].set(state["step_count"] + 1),
        "step_count": state["step_count"] + jnp.array(1, dtype=jnp.int32),
    }
    metrics = jnp.asarray(
        [
            jnp.sum(new_state["counts"] > 0.0).astype(jnp.float32),
            novel.astype(jnp.float32),
            nearest_distance,
        ],
        dtype=jnp.float32,
    )
    return new_state, metrics


def reset_memory_value_row_if_novel(
    params: dict[str, Any],
    slot: jax.Array,
    novel: jax.Array,
) -> dict[str, Any]:
    """Avoid recycled memory keys inheriting stale value rows."""
    memory = params["memory"]
    row = memory["values"][slot]
    new_row = jnp.where(novel, jnp.zeros_like(row), row)
    new_memory = {**memory, "values": memory["values"].at[slot].set(new_row)}
    return {**params, "memory": new_memory}


def run_proto_kv_attention(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
    memory_update_rate: float,
    memory_novelty_threshold: float,
    memory_temperature: float,
    reset_new_memory_values: bool,
) -> tuple[dict[str, Any], dict[str, jax.Array], np.ndarray]:
    """Train the attention-only prototype-KV adapter with one scan."""

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: dict[str, jax.Array],
    ) -> tuple[tuple[dict[str, Any], dict[str, jax.Array]], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], dict[str, jax.Array]],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], dict[str, jax.Array]], jax.Array]:
            params, state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
                logits, query, weights, gate, nearest_distance = proto_kv_attention_logits(
                    candidate,
                    state,
                    context,
                    temperature=memory_temperature,
                )
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    query,
                    weights,
                    jnp.asarray([gate, nearest_distance], dtype=jnp.float32),
                )

            (loss, (logits, query, weights, gate_distance)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step(params, grads, step_size)
            slot, novel, _ = select_memory_slot(
                state,
                query,
                novelty_threshold=memory_novelty_threshold,
            )
            new_state, center_metrics = update_memory_centers(
                state,
                query,
                update_rate=memory_update_rate,
                novelty_threshold=memory_novelty_threshold,
            )
            if reset_new_memory_values:
                new_params = reset_memory_value_row_if_novel(new_params, slot, novel)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.sum(weights > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                    gate_distance[0],
                    gate_distance[1],
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def run_hybrid_proto_kv_attention(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
    memory_update_rate: float,
    memory_novelty_threshold: float,
    memory_temperature: float,
    reset_new_memory_values: bool,
) -> tuple[dict[str, Any], dict[str, jax.Array], np.ndarray]:
    """Train the FFN plus prototype-KV adapter with one scan."""

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: dict[str, jax.Array],
    ) -> tuple[tuple[dict[str, Any], dict[str, jax.Array]], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], dict[str, jax.Array]],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], dict[str, jax.Array]], jax.Array]:
            params, state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
                logits, query, weights, gate, nearest_distance = hybrid_proto_kv_logits(
                    candidate,
                    state,
                    context,
                    temperature=memory_temperature,
                )
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    query,
                    weights,
                    jnp.asarray([gate, nearest_distance], dtype=jnp.float32),
                )

            (loss, (logits, query, weights, gate_distance)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step(params, grads, step_size)
            slot, novel, _ = select_memory_slot(
                state,
                query,
                novelty_threshold=memory_novelty_threshold,
            )
            new_state, center_metrics = update_memory_centers(
                state,
                query,
                update_rate=memory_update_rate,
                novelty_threshold=memory_novelty_threshold,
            )
            if reset_new_memory_values:
                new_params = reset_memory_value_row_if_novel(new_params, slot, novel)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.sum(weights > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                    gate_distance[0],
                    gate_distance[1],
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_proto_kv_attention(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    memory_temperature: float,
    hybrid: bool,
) -> dict[str, float]:
    """Evaluate a prototype-KV model on held-out examples."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        if hybrid:
            logits = jax.vmap(
                lambda ctx: hybrid_proto_kv_logits(
                    params,
                    state,
                    ctx,
                    temperature=memory_temperature,
                )[0]
            )(contexts)
        else:
            logits = jax.vmap(
                lambda ctx: proto_kv_attention_logits(
                    params,
                    state,
                    ctx,
                    temperature=memory_temperature,
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


def summarize_memory_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize prototype-KV diagnostics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_active_memory_weights": float(np.mean(window[:, 2])),
        "final_window_active_memory_slots": float(np.mean(window[:, 3])),
        "allocation_rate": float(np.mean(metrics[:, 4])),
        "final_window_allocation_rate": float(np.mean(window[:, 4])),
        "final_window_center_nearest_distance": float(np.mean(window[:, 5])),
        "final_window_retrieval_nearest_distance": float(np.mean(window[:, 7])),
        "final_window_gate": float(np.mean(window[:, 6])),
        "final_gate": float(metrics[-1, 6]),
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


def _format_mean_stderr(values: np.ndarray) -> str:
    return f"{np.mean(values):.4f} +/- {stderr(values):.4f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown result summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "proto_kv_attention",
        "hybrid_ffn_proto_kv",
    ]
    labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "proto_kv_attention": "Proto-KV",
        "hybrid_ffn_proto_kv": "FFN + Proto-KV",
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
        "# Tiny Shakespeare Prototype-KV Attention Memory",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "The prototype memory is inserted into the attention path. Prototype key "
        "centers update online from the final attention query; memory values and "
        "the retrieval gate train by cross-entropy gradients.",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
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
            f"| `{labels[method]}` | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Baseline FFN | Proto-KV | FFN + Proto-KV |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in metrics:
        cells = [
            _format_mean_stderr(aggregate_metric(records, method, metric))
            for method in methods
        ]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Diffs vs Baseline",
            "",
            "| Metric | Proto-KV | FFN + Proto-KV |",
            "|---|---:|---:|",
        ]
    )
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        diff_cells = []
        for method in ("proto_kv_attention", "hybrid_ffn_proto_kv"):
            values = aggregate_metric(records, method, metric)
            diff = baseline - values if metric in lower_is_better else values - baseline
            diff_cells.append(_format_mean_stderr(diff))
        lines.append(f"| `{metric}` | " + " | ".join(diff_cells) + " |")
    for method in ("proto_kv_attention", "hybrid_ffn_proto_kv"):
        memory_records = [row for row in records if row["method"] == method]
        if not memory_records:
            continue
        lines.extend(
            [
                "",
                f"## {labels[method]} Diagnostics",
                "",
                "| Metric | Mean +/- stderr |",
                "|---|---:|",
            ]
        )
        for metric in [
            "final_window_active_memory_weights",
            "final_window_active_memory_slots",
            "allocation_rate",
            "final_window_allocation_rate",
            "final_window_center_nearest_distance",
            "final_window_retrieval_nearest_distance",
            "final_window_gate",
            "final_gate",
        ]:
            values = np.asarray(
                [row["summary"][metric] for row in memory_records],
                dtype=np.float64,
            )
            lines.append(f"| `{metric}` | {_format_mean_stderr(values)} |")
    lines.extend(
        [
            "",
            "Positive diffs favor the prototype-KV method over the FFN baseline. "
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
    parser.add_argument("--memory-slots", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--adapter-lr", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-update-rate", type=float, default=0.3)
    parser.add_argument("--memory-novelty-threshold", type=float, default=0.08)
    parser.add_argument("--memory-temperature", type=float, default=0.05)
    parser.add_argument("--memory-value-init-scale", type=float, default=0.0)
    parser.add_argument("--gate-init", type=float, default=0.2)
    parser.add_argument(
        "--no-reset-new-memory-values",
        action="store_false",
        dest="reset_new_memory_values",
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
        default=Path("outputs/step2_new_directions/proto_kv_attention_worker"),
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
    if args.d_model < 1 or args.mlp_hidden < 1 or args.memory_slots < 1:
        raise ValueError("--d-model, --mlp-hidden, and --memory-slots must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.baseline_lr < 0.0 or args.adapter_lr < 0.0:
        raise ValueError("learning rates must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 < args.memory_update_rate <= 1.0:
        raise ValueError("--memory-update-rate must be in (0, 1]")
    if args.memory_novelty_threshold < 0.0:
        raise ValueError("--memory-novelty-threshold must be non-negative")
    if args.memory_temperature <= 0.0:
        raise ValueError("--memory-temperature must be positive")
    if args.memory_value_init_scale < 0.0:
        raise ValueError("--memory-value-init-scale must be non-negative")
    if not 0.0 < args.gate_init < 1.0:
        raise ValueError("--gate-init must be in (0, 1)")


def _record_summary(
    *,
    records: list[dict[str, Any]],
    seed_idx: int,
    method: str,
    metrics: np.ndarray,
    final_window: int,
    eval_metrics: dict[str, float],
    train_s: float,
    steps: int,
    include_memory: bool,
) -> dict[str, float]:
    summary = {
        **summarize_online(metrics, final_window),
        **eval_metrics,
        "train_s": train_s,
        "train_steps_per_s": steps / train_s,
    }
    if include_memory:
        summary = {**summary, **summarize_memory_diagnostics(metrics, final_window)}
    records.append({"seed": seed_idx, "method": method, "summary": summary})
    return summary


def main() -> None:
    """Run the prototype-KV attention comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = metadata["vocab_size"]
    final_window = args.final_window if args.final_window > 0 else args.eval_steps

    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        memory_slots=args.memory_slots,
        eval_steps=args.eval_steps,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        adapter_lr=args.adapter_lr,
        grad_clip=args.grad_clip,
        memory_update_rate=args.memory_update_rate,
        memory_novelty_threshold=args.memory_novelty_threshold,
        memory_temperature=args.memory_temperature,
        memory_value_init_scale=args.memory_value_init_scale,
        gate_init=args.gate_init,
        reset_new_memory_values=args.reset_new_memory_values,
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
    proto_profile_params, proto_profile_state = init_proto_kv_attention_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        memory_slots=args.memory_slots,
        memory_value_init_scale=args.memory_value_init_scale,
        gate_init=args.gate_init,
    )
    hybrid_profile_params, hybrid_profile_state = init_hybrid_proto_kv_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        memory_slots=args.memory_slots,
        memory_value_init_scale=args.memory_value_init_scale,
        gate_init=args.gate_init,
    )
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "proto_kv_attention": {
            "trainable_params": count_array_elements(proto_profile_params),
            "trainable_bytes": count_array_bytes(proto_profile_params),
            "state_elements": count_array_elements(proto_profile_state, include_int=True),
            "state_bytes": count_array_bytes(proto_profile_state, include_int=True),
        },
        "hybrid_ffn_proto_kv": {
            "trainable_params": count_array_elements(hybrid_profile_params),
            "trainable_bytes": count_array_bytes(hybrid_profile_params),
            "state_elements": count_array_elements(hybrid_profile_state, include_int=True),
            "state_bytes": count_array_bytes(hybrid_profile_state, include_int=True),
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
        proto_params, proto_state = init_proto_kv_attention_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            memory_slots=args.memory_slots,
            memory_value_init_scale=args.memory_value_init_scale,
            gate_init=args.gate_init,
        )
        hybrid_params, hybrid_state = init_hybrid_proto_kv_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
            memory_slots=args.memory_slots,
            memory_value_init_scale=args.memory_value_init_scale,
            gate_init=args.gate_init,
        )

        method_start = time.perf_counter()
        final_baseline, baseline_metrics = run_baseline_transformer(
            baseline_params,
            contexts,
            labels,
            step_size=args.baseline_lr,
            grad_clip=args.grad_clip,
        )
        baseline_train_s = time.perf_counter() - method_start
        baseline_summary = _record_summary(
            records=records,
            seed_idx=seed_idx,
            method="baseline_ffn_transformer",
            metrics=baseline_metrics,
            final_window=final_window,
            eval_metrics=eval_transformer(final_baseline, eval_contexts, eval_labels),
            train_s=baseline_train_s,
            steps=args.steps,
            include_memory=False,
        )
        print(
            f"seed={seed_idx} baseline_ffn_transformer: "
            f"fw_nll={baseline_summary['final_window_nll']:.3f}, "
            f"eval_ppl={baseline_summary['eval_perplexity']:.2f}, "
            f"train_s={baseline_train_s:.2f}"
        )

        method_start = time.perf_counter()
        final_proto, final_proto_state, proto_metrics = run_proto_kv_attention(
            proto_params,
            proto_state,
            contexts,
            labels,
            step_size=args.adapter_lr,
            grad_clip=args.grad_clip,
            memory_update_rate=args.memory_update_rate,
            memory_novelty_threshold=args.memory_novelty_threshold,
            memory_temperature=args.memory_temperature,
            reset_new_memory_values=args.reset_new_memory_values,
        )
        final_proto_state["step_count"].block_until_ready()
        proto_train_s = time.perf_counter() - method_start
        proto_summary = _record_summary(
            records=records,
            seed_idx=seed_idx,
            method="proto_kv_attention",
            metrics=proto_metrics,
            final_window=final_window,
            eval_metrics=eval_proto_kv_attention(
                final_proto,
                final_proto_state,
                eval_contexts,
                eval_labels,
                memory_temperature=args.memory_temperature,
                hybrid=False,
            ),
            train_s=proto_train_s,
            steps=args.steps,
            include_memory=True,
        )
        print(
            f"seed={seed_idx} proto_kv_attention: "
            f"fw_nll={proto_summary['final_window_nll']:.3f}, "
            f"eval_ppl={proto_summary['eval_perplexity']:.2f}, "
            f"gate={proto_summary['final_window_gate']:.3f}, "
            f"active={proto_summary['final_window_active_memory_slots']:.1f}, "
            f"train_s={proto_train_s:.2f}"
        )

        method_start = time.perf_counter()
        final_hybrid, final_hybrid_state, hybrid_metrics = run_hybrid_proto_kv_attention(
            hybrid_params,
            hybrid_state,
            contexts,
            labels,
            step_size=args.adapter_lr,
            grad_clip=args.grad_clip,
            memory_update_rate=args.memory_update_rate,
            memory_novelty_threshold=args.memory_novelty_threshold,
            memory_temperature=args.memory_temperature,
            reset_new_memory_values=args.reset_new_memory_values,
        )
        final_hybrid_state["step_count"].block_until_ready()
        hybrid_train_s = time.perf_counter() - method_start
        hybrid_summary = _record_summary(
            records=records,
            seed_idx=seed_idx,
            method="hybrid_ffn_proto_kv",
            metrics=hybrid_metrics,
            final_window=final_window,
            eval_metrics=eval_proto_kv_attention(
                final_hybrid,
                final_hybrid_state,
                eval_contexts,
                eval_labels,
                memory_temperature=args.memory_temperature,
                hybrid=True,
            ),
            train_s=hybrid_train_s,
            steps=args.steps,
            include_memory=True,
        )
        print(
            f"seed={seed_idx} hybrid_ffn_proto_kv: "
            f"fw_nll={hybrid_summary['final_window_nll']:.3f}, "
            f"eval_ppl={hybrid_summary['eval_perplexity']:.2f}, "
            f"gate={hybrid_summary['final_window_gate']:.3f}, "
            f"active={hybrid_summary['final_window_active_memory_slots']:.1f}, "
            f"train_s={hybrid_train_s:.2f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "profiles": profiles,
        "elapsed_s": time.perf_counter() - start,
        "records": records,
        "note": (
            "Prototype-KV memory is implemented locally in this experiment. "
            "The trainable value rows/gate receive cross-entropy gradients; "
            "the prototype key centers update online from every final attention query."
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

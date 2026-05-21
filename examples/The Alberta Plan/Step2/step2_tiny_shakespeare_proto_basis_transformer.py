#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare prototype-basis transformer block comparison.

This script turns the D24 prototype-basis candidate into an actual transformer
residual sublayer:

``causal attention -> prototype basis residual block -> linear token readout``.

The baseline keeps the same attention stem and readout shape, but uses a
standard GELU FFN residual block. The prototype block has trainable value rows
and slow novelty-allocated centers. Cross-entropy gradients train the value
rows, attention, and readout; centers update online at every token.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
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
    init_attention_params,
    init_readout_params,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    sgd_step,
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
    proto_lr: float
    grad_clip: float
    proto_update_rate: float
    proto_novelty_threshold: float
    proto_bandwidth: float
    proto_adaptive_bandwidth: bool
    proto_bandwidth_update_rate: float
    reset_new_prototype_values: bool
    data_path: str
    output_dir: str
    seed: int


def make_proto_block(args: argparse.Namespace) -> PrototypeBasisBlock:
    """Create the transformer residual prototype block."""
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


def init_proto_transformer(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
) -> tuple[dict[str, Any], PrototypeBasisState]:
    """Initialize trainable transformer params and slow prototype state."""
    attn_key, proto_key, readout_key = jr.split(key, 3)
    proto_params, proto_state = block.init(proto_key)
    params = {
        "attn": init_attention_params(attn_key, vocab_size, block_size, d_model),
        "proto": proto_params,
        "readout": init_readout_params(readout_key, d_model, vocab_size),
    }
    return params, proto_state


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


def proto_transformer_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return logits, pre-block hidden state, and prototype activations."""
    basis_input = causal_attention_sequence(params["attn"], context)[-1]
    activations = block.activations(state, basis_input)
    residual = block.transform(params["proto"], activations)
    hidden = basis_input + residual
    readout = params["readout"]
    logits = hidden @ readout["w"] + readout["b"]
    return logits, basis_input, activations


def hybrid_transformer_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return logits for FFN plus prototype residual transformer."""
    hidden = causal_attention_sequence(params["attn"], context)[-1]
    basis_input = ffn_transform(params["ffn"], hidden)
    activations = block.activations(state, basis_input)
    hidden = basis_input + block.transform(params["proto"], activations)
    readout = params["readout"]
    logits = hidden @ readout["w"] + readout["b"]
    return logits, basis_input, activations


def select_center_slot(
    block: PrototypeBasisBlock,
    state: PrototypeBasisState,
    observation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mirror ``update_centers`` slot choice so new rows can be reset."""
    used = state.counts > 0.0
    has_used = jnp.any(used)
    has_empty = jnp.any(~used)
    distances = jnp.mean((state.centers - observation[None, :]) ** 2, axis=1)
    used_distances = jnp.where(used, distances, jnp.inf)
    nearest_slot = jnp.argmin(used_distances)
    nearest_distance = used_distances[nearest_slot]
    empty_slot = jnp.argmax((~used).astype(jnp.int32))
    replacement_slot = block._replacement_slot(state)
    novel = (~has_used) | (
        nearest_distance
        > jnp.asarray(block.config.novelty_threshold, dtype=jnp.float32)
    )
    slot = jnp.where(
        ~has_used,
        jnp.array(0, dtype=nearest_slot.dtype),
        jnp.where(
            novel & has_empty,
            empty_slot,
            jnp.where(novel, replacement_slot, nearest_slot),
        ),
    )
    return slot, novel


def reset_value_row_if_novel(
    params: dict[str, Any],
    slot: jax.Array,
    novel: jax.Array,
) -> dict[str, Any]:
    """Prevent recycled centers from inheriting stale residual values."""
    proto_params = params["proto"]
    old_row = proto_params.values[slot]
    new_row = jnp.where(novel, jnp.zeros_like(old_row), old_row)
    new_proto = PrototypeBasisParams(
        values=proto_params.values.at[slot].set(new_row),
        bias=proto_params.bias,
    )
    return {**params, "proto": new_proto}


def run_proto_basis_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
    reset_new_prototype_values: bool,
) -> tuple[dict[str, Any], PrototypeBasisState, np.ndarray]:
    """Train the prototype-basis transformer block with one JIT scan."""

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
                logits, hidden, activations = proto_transformer_logits(
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
            new_params = sgd_step(params, grads, step_size)
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


def run_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
    reset_new_prototype_values: bool,
) -> tuple[dict[str, Any], PrototypeBasisState, np.ndarray]:
    """Train a fast FFN plus slow prototype residual transformer block."""

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
            new_params = sgd_step(params, grads, step_size)
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


def eval_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    """Evaluate the prototype-basis transformer on held-out examples."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(lambda ctx: proto_transformer_logits(block, params, state, ctx)[0])(
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


def eval_hybrid_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    """Evaluate the hybrid FFN plus prototype residual transformer."""

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


def summarize_proto_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize prototype-block diagnostics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_active_features": float(np.mean(window[:, 2])),
        "final_window_active_prototypes": float(np.mean(window[:, 3])),
        "allocation_rate": float(np.mean(metrics[:, 4])),
        "final_window_allocation_rate": float(np.mean(window[:, 4])),
        "final_window_nearest_distance": float(np.mean(window[:, 5])),
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


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown result summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "proto_basis_transformer",
        "hybrid_proto_transformer",
    ]
    method_labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "proto_basis_transformer": "Prototype Basis",
        "hybrid_proto_transformer": "FFN + Prototype",
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
        "# Tiny Shakespeare Prototype-Basis Transformer Block",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "The prototype methods are actual transformer sublayers: pure "
        "`attention -> prototype basis -> readout` and hybrid "
        "`attention -> FFN -> prototype basis -> readout`.",
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
            f"| `{method_labels[method]}` | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    metric_header = "| Metric | " + " | ".join(method_labels[m] for m in methods) + " |"
    metric_separator = "|---|" + "---:|" * len(methods)
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            metric_header,
            metric_separator,
        ]
    )
    for metric in metrics:
        cells = []
        for method in methods:
            values = aggregate_metric(records, method, metric)
            cells.append(f"{np.mean(values):.4f} +/- {stderr(values):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Diffs vs Baseline",
            "",
            "| Metric | Prototype Basis | FFN + Prototype |",
            "|---|---:|---:|",
        ]
    )
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        diff_cells = []
        for method in ("proto_basis_transformer", "hybrid_proto_transformer"):
            values = aggregate_metric(records, method, metric)
            diff = baseline - values if metric in lower_is_better else values - baseline
            diff_cells.append(f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(diff_cells) + " |")
    for method in ("proto_basis_transformer", "hybrid_proto_transformer"):
        proto_records = [row for row in records if row["method"] == method]
        if not proto_records:
            continue
        lines.extend(
            [
                "",
                f"## {method_labels[method]} Diagnostics",
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
                [row["summary"][metric] for row in proto_records],
                dtype=np.float64,
            )
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    lines.extend(
        [
            "",
            "Positive diffs favor the prototype method over the FFN baseline. "
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
    parser.add_argument("--proto-lr", type=float, default=0.03)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/proto_basis_transformer"),
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
    if args.baseline_lr < 0.0 or args.proto_lr < 0.0:
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
        proto_lr=args.proto_lr,
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=args.proto_adaptive_bandwidth,
        proto_bandwidth_update_rate=args.proto_bandwidth_update_rate,
        reset_new_prototype_values=args.reset_new_prototype_values,
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
    proto_profile_params, proto_profile_state = init_proto_transformer(
        profile_key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
    )
    hybrid_profile_params, hybrid_profile_state = init_hybrid_transformer(
        profile_key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "proto_basis_transformer": {
            "trainable_params": count_array_elements(proto_profile_params),
            "trainable_bytes": count_array_bytes(proto_profile_params),
            "state_elements": count_array_elements(proto_profile_state, include_int=True),
            "state_bytes": count_array_bytes(proto_profile_state, include_int=True),
        },
        "hybrid_proto_transformer": {
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
        proto_params, proto_state = init_proto_transformer(
            param_key,
            block=proto_block,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
        )
        hybrid_params, hybrid_state = init_hybrid_transformer(
            param_key,
            block=proto_block,
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
        baseline_train_s = time.perf_counter() - method_start
        baseline_summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_transformer(final_baseline, eval_contexts, eval_labels),
            "train_s": baseline_train_s,
            "train_steps_per_s": args.steps / baseline_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "baseline_ffn_transformer",
                "summary": baseline_summary,
            }
        )
        print(
            f"seed={seed_idx} baseline_ffn_transformer: "
            f"fw_nll={baseline_summary['final_window_nll']:.3f}, "
            f"eval_ppl={baseline_summary['eval_perplexity']:.2f}, "
            f"train_s={baseline_train_s:.2f}"
        )

        method_start = time.perf_counter()
        final_proto, final_proto_state, proto_metrics = run_proto_basis_transformer(
            proto_block,
            proto_params,
            proto_state,
            contexts,
            labels,
            step_size=args.proto_lr,
            grad_clip=args.grad_clip,
            reset_new_prototype_values=args.reset_new_prototype_values,
        )
        final_proto_state.step_count.block_until_ready()
        proto_train_s = time.perf_counter() - method_start
        proto_summary = {
            **summarize_online(proto_metrics, final_window),
            **summarize_proto_diagnostics(proto_metrics, final_window),
            **eval_proto_transformer(
                proto_block,
                final_proto,
                final_proto_state,
                eval_contexts,
                eval_labels,
            ),
            "train_s": proto_train_s,
            "train_steps_per_s": args.steps / proto_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "proto_basis_transformer",
                "summary": proto_summary,
            }
        )
        print(
            f"seed={seed_idx} proto_basis_transformer: "
            f"fw_nll={proto_summary['final_window_nll']:.3f}, "
            f"eval_ppl={proto_summary['eval_perplexity']:.2f}, "
            f"train_s={proto_train_s:.2f}, "
            f"active={proto_summary['final_window_active_prototypes']:.1f}"
        )

        method_start = time.perf_counter()
        final_hybrid, final_hybrid_state, hybrid_metrics = run_hybrid_proto_transformer(
            proto_block,
            hybrid_params,
            hybrid_state,
            contexts,
            labels,
            step_size=args.proto_lr,
            grad_clip=args.grad_clip,
            reset_new_prototype_values=args.reset_new_prototype_values,
        )
        final_hybrid_state.step_count.block_until_ready()
        hybrid_train_s = time.perf_counter() - method_start
        hybrid_summary = {
            **summarize_online(hybrid_metrics, final_window),
            **summarize_proto_diagnostics(hybrid_metrics, final_window),
            **eval_hybrid_proto_transformer(
                proto_block,
                final_hybrid,
                final_hybrid_state,
                eval_contexts,
                eval_labels,
            ),
            "train_s": hybrid_train_s,
            "train_steps_per_s": args.steps / hybrid_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "hybrid_proto_transformer",
                "summary": hybrid_summary,
            }
        )
        print(
            f"seed={seed_idx} hybrid_proto_transformer: "
            f"fw_nll={hybrid_summary['final_window_nll']:.3f}, "
            f"eval_ppl={hybrid_summary['eval_perplexity']:.2f}, "
            f"train_s={hybrid_train_s:.2f}, "
            f"active={hybrid_summary['final_window_active_prototypes']:.1f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "prototype_block": proto_block.to_config(),
        "profiles": profiles,
        "elapsed_s": time.perf_counter() - start,
        "records": records,
        "note": (
            "The prototype basis is trained as a residual transformer block, not "
            "as an external nearest-prototype classifier. The hybrid method keeps "
            "the fast FFN and adds slow prototype residual state."
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

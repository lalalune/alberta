#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare gated fast/slow memory transformer.

This runner tests the next Step 2 transformer hypothesis:

* keep a fast differentiable FFN path;
* add a slow novelty-allocated prototype memory path;
* learn a residual gate from hidden state plus uncertainty, novelty, and
  recent loss;
* decouple fast-path learning rate from memory/gate learning rate; and
* optionally place memory as a KV-like adapter before the FFN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_tiny_shakespeare_proto_basis_transformer import (
    eval_hybrid_proto_transformer,
    init_hybrid_transformer,
    make_proto_block,
    reset_value_row_if_novel,
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
    PrototypeBasisParams,
    PrototypeBasisState,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured in results."""

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
    loss_decay: float
    meta_init_decay: float
    gate_init_bias: float
    reset_mode: str
    data_path: str
    output_dir: str
    seed: int


@chex.dataclass(frozen=True)
class GatedMemoryState:
    """Slow memory state plus online diagnostic traces."""

    proto_state: PrototypeBasisState
    loss_ema: jax.Array
    init_value: jax.Array
    step_count: jax.Array


def init_gate_params(
    key: jax.Array,
    d_model: int,
    gate_init_bias: float,
) -> dict[str, jax.Array]:
    """Initialize per-channel residual gate parameters."""
    return {
        "w": 0.01 * jr.normal(key, (d_model + 4, d_model), dtype=jnp.float32),
        "b": jnp.full((d_model,), gate_init_bias, dtype=jnp.float32),
    }


def init_gated_params_state(
    key: jax.Array,
    *,
    block: PrototypeBasisBlock,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
    gate_init_bias: float,
) -> tuple[dict[str, Any], GatedMemoryState]:
    """Initialize fast transformer params, slow memory params, and gate."""
    base_key, gate_key = jr.split(key)
    params, proto_state = init_hybrid_transformer(
        base_key,
        block=block,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=ffn_hidden,
    )
    params = {**params, "gate": init_gate_params(gate_key, d_model, gate_init_bias)}
    state = GatedMemoryState(
        proto_state=proto_state,
        loss_ema=jnp.asarray(math.log(vocab_size), dtype=jnp.float32),
        init_value=jnp.zeros((d_model,), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    return params, state


def categorical_diagnostics(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return normalized entropy and inverse margin uncertainty."""
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-12)))
    entropy = entropy / jnp.log(jnp.asarray(logits.shape[0], dtype=jnp.float32))
    top2 = jax.lax.top_k(probs, 2)[0]
    margin = top2[0] - top2[1]
    return entropy, 1.0 - margin


def novelty_score(
    block: PrototypeBasisBlock,
    state: PrototypeBasisState,
    observation: jax.Array,
) -> jax.Array:
    """Return threshold-normalized distance to nearest used prototype."""
    used = state.counts > 0.0
    distances = jnp.mean((state.centers - observation[None, :]) ** 2, axis=1)
    nearest = jnp.min(jnp.where(used, distances, jnp.inf))
    nearest = jnp.where(jnp.any(used), nearest, jnp.asarray(block.config.novelty_threshold))
    threshold = jnp.maximum(jnp.asarray(block.config.novelty_threshold), 1e-8)
    return jnp.clip(nearest / threshold, 0.0, 10.0)


def gate_values(
    params: dict[str, Any],
    hidden: jax.Array,
    *,
    entropy: jax.Array,
    uncertainty: jax.Array,
    novelty: jax.Array,
    loss_ema: jax.Array,
) -> jax.Array:
    """Compute a per-channel slow residual gate."""
    diagnostics = jnp.asarray([entropy, uncertainty, novelty, loss_ema], dtype=jnp.float32)
    gate_input = jnp.concatenate([hidden, diagnostics])
    return jax.nn.sigmoid(gate_input @ params["gate"]["w"] + params["gate"]["b"])


def gated_memory_logits(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: GatedMemoryState,
    context: jax.Array,
    *,
    placement: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return logits and memory diagnostics for one context."""
    attn_hidden = causal_attention_sequence(params["attn"], context)[-1]
    if placement == "pre_ffn_kv":
        basis_input = attn_hidden
        base_logits = basis_input @ params["readout"]["w"] + params["readout"]["b"]
    else:
        basis_input = ffn_transform(params["ffn"], attn_hidden)
        base_logits = basis_input @ params["readout"]["w"] + params["readout"]["b"]
    entropy, uncertainty = categorical_diagnostics(base_logits)
    novelty = novelty_score(block, state.proto_state, basis_input)
    activations = block.activations(state.proto_state, basis_input)
    residual = block.transform(params["proto"], activations)
    gate = gate_values(
        params,
        basis_input,
        entropy=entropy,
        uncertainty=uncertainty,
        novelty=novelty,
        loss_ema=state.loss_ema,
    )
    if placement == "pre_ffn_kv":
        hidden = ffn_transform(params["ffn"], basis_input + gate * residual)
    else:
        hidden = basis_input + gate * residual
    logits = hidden @ params["readout"]["w"] + params["readout"]["b"]
    return logits, basis_input, activations, gate, entropy, novelty


def sgd_step_decoupled(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    fast_lr: float,
    slow_lr: float,
) -> dict[str, Any]:
    """Apply separate rates to fast transformer and slow memory/gate params."""

    def step_tree(tree: Any, grad_tree: Any, lr: float) -> Any:
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, tree, grad_tree)

    return {
        "attn": step_tree(params["attn"], grads["attn"], fast_lr),
        "ffn": step_tree(params["ffn"], grads["ffn"], fast_lr),
        "readout": step_tree(params["readout"], grads["readout"], fast_lr),
        "proto": step_tree(params["proto"], grads["proto"], slow_lr),
        "gate": step_tree(params["gate"], grads["gate"], slow_lr),
    }


def reset_proto_row(
    params: dict[str, Any],
    state: GatedMemoryState,
    slot: jax.Array,
    novel: jax.Array,
    *,
    reset_mode: str,
) -> dict[str, Any]:
    """Reset a replaced prototype value row according to the selected policy."""
    if reset_mode == "none":
        return params
    if reset_mode == "zero":
        return reset_value_row_if_novel(params, slot, novel)
    proto_params = params["proto"]
    old_row = proto_params.values[slot]
    new_row = jnp.where(novel, state.init_value, old_row)
    new_proto = PrototypeBasisParams(
        values=proto_params.values.at[slot].set(new_row),
        bias=proto_params.bias,
    )
    return {**params, "proto": new_proto}


def update_meta_init(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: GatedMemoryState,
    activations: jax.Array,
    loss: jax.Array,
    *,
    meta_init_decay: float,
) -> jax.Array:
    """Meta-update the replacement initializer from successful active values."""
    residual = block.transform(params["proto"], activations)
    improved = loss <= state.loss_ema
    decay = jnp.asarray(meta_init_decay, dtype=jnp.float32)
    candidate = decay * state.init_value + (1.0 - decay) * residual
    return jnp.where(improved, candidate, state.init_value)


def run_gated_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: GatedMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    loss_decay: float,
    meta_init_decay: float,
    reset_mode: str,
) -> tuple[dict[str, Any], GatedMemoryState, np.ndarray]:
    """Train one gated memory transformer variant."""

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: GatedMemoryState,
    ) -> tuple[tuple[dict[str, Any], GatedMemoryState], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], GatedMemoryState],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], GatedMemoryState], jax.Array]:
            params, state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
                logits, basis_input, activations, gate, entropy, novelty = gated_memory_logits(
                    block,
                    candidate,
                    state,
                    context,
                    placement=placement,
                )
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    basis_input,
                    activations,
                    gate,
                    jnp.asarray([entropy, novelty], dtype=jnp.float32),
                )

            (loss, (logits, basis_input, activations, gate, diag)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(params)
            )
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step_decoupled(
                params,
                grads,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )
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
            init_value = update_meta_init(
                block,
                new_params,
                state,
                activations,
                loss,
                meta_init_decay=meta_init_decay,
            )
            decay = jnp.asarray(loss_decay, dtype=jnp.float32)
            loss_ema = decay * state.loss_ema + (1.0 - decay) * loss
            new_state = GatedMemoryState(
                proto_state=new_proto_state,
                loss_ema=loss_ema,
                init_value=init_value,
                step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.mean(gate),
                    jnp.max(gate),
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                    diag[0],
                    diag[1],
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_gated_memory_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: GatedMemoryState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    placement: str,
) -> dict[str, float]:
    """Evaluate one gated memory transformer variant."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(
            lambda ctx: gated_memory_logits(block, params, state, ctx, placement=placement)[0]
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


def summarize_gated(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize gate and slow-memory diagnostics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_mean_gate": float(np.mean(window[:, 2])),
        "final_window_max_gate": float(np.mean(window[:, 3])),
        "final_window_active_features": float(np.mean(window[:, 4])),
        "final_window_active_prototypes": float(np.mean(window[:, 5])),
        "allocation_rate": float(np.mean(metrics[:, 6])),
        "final_window_allocation_rate": float(np.mean(window[:, 6])),
        "final_window_nearest_distance": float(np.mean(window[:, 7])),
        "final_window_entropy": float(np.mean(window[:, 8])),
        "final_window_novelty": float(np.mean(window[:, 9])),
    }


def aggregate_metric(records: list[dict[str, Any]], method: str, metric: str) -> np.ndarray:
    """Collect one metric across seeds."""
    return np.asarray(
        [row["summary"][metric] for row in records if row["method"] == method],
        dtype=np.float64,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    records = payload["records"]
    methods = [
        "baseline_ffn_transformer",
        "ungated_hybrid_transformer",
        "gated_post_ffn_memory",
        "gated_pre_ffn_kv_memory",
    ]
    labels = {
        "baseline_ffn_transformer": "Baseline FFN",
        "ungated_hybrid_transformer": "Ungated Hybrid",
        "gated_post_ffn_memory": "Gated Post-FFN",
        "gated_pre_ffn_kv_memory": "Gated Pre-FFN KV",
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
        "# Tiny Shakespeare Gated Memory Transformer",
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
    lines.extend(["", "## Metrics", ""])
    lines.append("| Metric | " + " | ".join(labels[m] for m in methods) + " |")
    lines.append("|---|" + "---:|" * len(methods))
    for metric in metrics:
        cells = []
        for method in methods:
            values = aggregate_metric(records, method, metric)
            cells.append(f"{np.mean(values):.4f} +/- {stderr(values):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.extend(["", "## Diffs vs Baseline", ""])
    lines.append("| Metric | Ungated Hybrid | Gated Post-FFN | Gated Pre-FFN KV |")
    lines.append("|---|---:|---:|---:|")
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        cells = []
        for method in methods[1:]:
            values = aggregate_metric(records, method, metric)
            diff = baseline - values if metric in lower_is_better else values - baseline
            cells.append(f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    for method in methods[2:]:
        lines.extend(["", f"## {labels[method]} Diagnostics", ""])
        lines.append("| Metric | Mean +/- stderr |")
        lines.append("|---|---:|")
        rows = [row for row in records if row["method"] == method]
        for metric in [
            "final_window_mean_gate",
            "final_window_max_gate",
            "final_window_active_features",
            "final_window_active_prototypes",
            "allocation_rate",
            "final_window_allocation_rate",
            "final_window_entropy",
            "final_window_novelty",
        ]:
            values = np.asarray([row["summary"][metric] for row in rows], dtype=np.float64)
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    lines.append("")
    lines.append("Positive diffs favor the memory method over the tuned FFN baseline.")
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
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--fast-lr", type=float, default=0.15)
    parser.add_argument("--slow-lr", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.0002)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--proto-adaptive-bandwidth", action="store_true")
    parser.add_argument("--proto-bandwidth-update-rate", type=float, default=0.1)
    parser.add_argument("--loss-decay", type=float, default=0.98)
    parser.add_argument("--meta-init-decay", type=float, default=0.99)
    parser.add_argument("--gate-init-bias", type=float, default=-1.0)
    parser.add_argument("--reset-mode", choices=("none", "zero", "meta_ema"), default="meta_ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/gated_memory_transformer"),
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
    if min(args.baseline_lr, args.fast_lr, args.slow_lr) < 0.0:
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
    if not 0.0 <= args.loss_decay < 1.0:
        raise ValueError("--loss-decay must be in [0, 1)")
    if not 0.0 <= args.meta_init_decay < 1.0:
        raise ValueError("--meta-init-decay must be in [0, 1)")


def main() -> None:
    """Run the gated memory comparison."""
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
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        loss_decay=args.loss_decay,
        meta_init_decay=args.meta_init_decay,
        gate_init_bias=args.gate_init_bias,
        reset_mode=args.reset_mode,
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
    gated_profile_params, gated_profile_state = init_gated_params_state(
        profile_key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
        gate_init_bias=args.gate_init_bias,
    )
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "ungated_hybrid_transformer": {
            "trainable_params": count_array_elements(ungated_profile_params),
            "trainable_bytes": count_array_bytes(ungated_profile_params),
            "state_elements": count_array_elements(ungated_profile_state, include_int=True),
            "state_bytes": count_array_bytes(ungated_profile_state, include_int=True),
        },
        "gated_post_ffn_memory": {
            "trainable_params": count_array_elements(gated_profile_params),
            "trainable_bytes": count_array_bytes(gated_profile_params),
            "state_elements": count_array_elements(gated_profile_state, include_int=True),
            "state_bytes": count_array_bytes(gated_profile_state, include_int=True),
        },
        "gated_pre_ffn_kv_memory": {
            "trainable_params": count_array_elements(gated_profile_params),
            "trainable_bytes": count_array_bytes(gated_profile_params),
            "state_elements": count_array_elements(gated_profile_state, include_int=True),
            "state_bytes": count_array_bytes(gated_profile_state, include_int=True),
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

        method_start = time.perf_counter()
        final_ungated, final_ungated_state, ungated_metrics = run_hybrid_proto_baseline(
            proto_block,
            ungated_params,
            ungated_state,
            contexts,
            labels,
            fast_lr=args.fast_lr,
            slow_lr=args.slow_lr,
            grad_clip=args.grad_clip,
            reset_mode=args.reset_mode,
        )
        final_ungated_state.step_count.block_until_ready()
        train_s = time.perf_counter() - method_start
        summary = {
            **summarize_online(ungated_metrics, final_window),
            **summarize_ungated(ungated_metrics, final_window),
            **eval_hybrid_proto_transformer(
                proto_block,
                final_ungated,
                final_ungated_state,
                eval_contexts,
                eval_labels,
            ),
            "train_s": train_s,
            "train_steps_per_s": args.steps / train_s,
        }
        records.append(
            {"seed": seed_idx, "method": "ungated_hybrid_transformer", "summary": summary}
        )
        print(
            f"seed={seed_idx} ungated: fw_nll={summary['final_window_nll']:.3f}, "
            f"eval_ppl={summary['eval_perplexity']:.2f}, train_s={train_s:.2f}"
        )

        for method, placement in (
            ("gated_post_ffn_memory", "post_ffn"),
            ("gated_pre_ffn_kv_memory", "pre_ffn_kv"),
        ):
            gated_params, gated_state = init_gated_params_state(
                param_key,
                block=proto_block,
                vocab_size=vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                ffn_hidden=args.mlp_hidden,
                gate_init_bias=args.gate_init_bias,
            )
            method_start = time.perf_counter()
            final_gated, final_gated_state, gated_metrics = run_gated_memory_transformer(
                proto_block,
                gated_params,
                gated_state,
                contexts,
                labels,
                placement=placement,
                fast_lr=args.fast_lr,
                slow_lr=args.slow_lr,
                grad_clip=args.grad_clip,
                loss_decay=args.loss_decay,
                meta_init_decay=args.meta_init_decay,
                reset_mode=args.reset_mode,
            )
            final_gated_state.step_count.block_until_ready()
            train_s = time.perf_counter() - method_start
            summary = {
                **summarize_online(gated_metrics, final_window),
                **summarize_gated(gated_metrics, final_window),
                **eval_gated_memory_transformer(
                    proto_block,
                    final_gated,
                    final_gated_state,
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
                f"gate={summary['final_window_mean_gate']:.3f}, train_s={train_s:.2f}"
            )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "prototype_block": proto_block.to_config(),
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


def run_hybrid_proto_baseline(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    fast_lr: float,
    slow_lr: float,
    grad_clip: float,
    reset_mode: str,
) -> tuple[dict[str, Any], PrototypeBasisState, np.ndarray]:
    """Ungated hybrid with decoupled fast/slow learning rates."""

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
                hidden = causal_attention_sequence(candidate["attn"], context)[-1]
                basis_input = ffn_transform(candidate["ffn"], hidden)
                activations = block.activations(state, basis_input)
                output = basis_input + block.transform(candidate["proto"], activations)
                logits = output @ candidate["readout"]["w"] + candidate["readout"]["b"]
                return cross_entropy_from_logits(logits, label), (
                    logits,
                    basis_input,
                    activations,
                )

            (loss, (logits, basis_input, activations)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grads = clip_grads(grads, grad_clip)
            dummy_gate_grads = {"w": jnp.zeros((1, 1)), "b": jnp.zeros((1,))}
            dummy_gate_params = {"w": jnp.zeros((1, 1)), "b": jnp.zeros((1,))}
            stepped = sgd_step_decoupled(
                {**params, "gate": dummy_gate_params},
                {**grads, "gate": dummy_gate_grads},
                fast_lr=fast_lr,
                slow_lr=slow_lr,
            )
            new_params = {k: stepped[k] for k in ("attn", "ffn", "readout", "proto")}
            slot, novel = select_center_slot(block, state, basis_input)
            new_state, center_metrics = block.update_centers(state, basis_input)
            wrapped_state = GatedMemoryState(
                proto_state=state,
                loss_ema=loss,
                init_value=jnp.zeros_like(params["proto"].bias),
                step_count=jnp.array(0, dtype=jnp.int32),
            )
            new_params = reset_proto_row(
                new_params,
                wrapped_state,
                slot,
                novel,
                reset_mode=reset_mode,
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
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def summarize_ungated(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize ungated hybrid diagnostics with gate-compatible keys."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_active_features": float(np.mean(window[:, 2])),
        "final_window_active_prototypes": float(np.mean(window[:, 3])),
        "allocation_rate": float(np.mean(metrics[:, 4])),
        "final_window_allocation_rate": float(np.mean(window[:, 4])),
        "final_window_nearest_distance": float(np.mean(window[:, 5])),
    }


if __name__ == "__main__":
    main()

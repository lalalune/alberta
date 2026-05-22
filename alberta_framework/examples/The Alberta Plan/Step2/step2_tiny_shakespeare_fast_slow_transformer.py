#!/usr/bin/env python3
# mypy: disable-error-code="arg-type,call-arg,index,no-any-return,operator,untyped-decorator"
"""Tiny Shakespeare fast/slow transformer memory experiment.

This is a small Step 2 memory experiment, not a language-modeling benchmark.
It compares:

1. ``baseline_ffn_transformer``: the existing tuned one-block causal attention
   transformer with a residual GELU FFN.
2. ``fast_slow_transformer``: the same attention stem and token readout, but
   the residual block is a learned gate between a fast FFN residual and a slow
   online prototype residual.

The gate sees the current hidden state plus two causal diagnostics: previous
loss EMA and current prototype novelty.  The fast FFN/readout path and slow
prototype-value path have separate learning rates.
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
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    stderr,
    summarize_online,
    transformer_logits,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured into the result artifact."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    fast_hidden: int
    proto_count: int
    eval_steps: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    fast_lr: float
    slow_lr: float
    gate_lr: float
    center_update_rate: float
    loss_decay: float
    novelty_decay: float
    proto_bandwidth: float
    novelty_threshold: float
    grad_clip: float
    data_path: str
    output_dir: str
    seed: int


def init_slow_params(
    key: jax.Array,
    *,
    proto_count: int,
    d_model: int,
) -> dict[str, jax.Array]:
    """Initialize slow prototype residual values."""
    del key
    return {
        "values": jnp.zeros((proto_count, d_model), dtype=jnp.float32),
        "bias": jnp.zeros((d_model,), dtype=jnp.float32),
    }


def init_gate_params(
    key: jax.Array,
    *,
    d_model: int,
    diag_dim: int = 2,
) -> dict[str, jax.Array]:
    """Initialize a learned gate over fast and slow residuals."""
    hidden_key, diag_key = jr.split(key, 2)
    return {
        "hidden": 0.01 * jr.normal(hidden_key, (d_model, d_model)),
        "diag": 0.01 * jr.normal(diag_key, (diag_dim, d_model)),
        "bias": jnp.full((d_model,), -1.0, dtype=jnp.float32),
    }


def init_fast_slow_params(
    key: jax.Array,
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    fast_hidden: int,
    proto_count: int,
) -> dict[str, Any]:
    """Initialize trainable fast/slow transformer parameters."""
    base = init_transformer_params(
        key,
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        ffn_hidden=fast_hidden,
    )
    slow_key, gate_key = jr.split(jr.fold_in(key, 101))
    return {
        "attn": base["attn"],
        "ffn": base["ffn"],
        "slow": init_slow_params(slow_key, proto_count=proto_count, d_model=d_model),
        "gate": init_gate_params(gate_key, d_model=d_model),
        "readout": base["readout"],
    }


def init_fast_slow_state(proto_count: int, d_model: int) -> dict[str, jax.Array]:
    """Initialize online prototype memory and causal diagnostics."""
    return {
        "centers": jnp.zeros((proto_count, d_model), dtype=jnp.float32),
        "counts": jnp.zeros((proto_count,), dtype=jnp.float32),
        "next_slot": jnp.array(0, dtype=jnp.int32),
        "loss_ema": jnp.array(0.0, dtype=jnp.float32),
        "novelty_ema": jnp.array(0.0, dtype=jnp.float32),
        "step_count": jnp.array(0, dtype=jnp.int32),
    }


def fast_residual(ffn: dict[str, jax.Array], hidden: jax.Array) -> jax.Array:
    """Fast GELU FFN residual without adding the input."""
    features = jax.nn.gelu(hidden @ ffn["w1"] + ffn["b1"])
    return features @ ffn["w2"] + ffn["b2"]


def prototype_activations(
    state: dict[str, jax.Array],
    hidden: jax.Array,
    *,
    bandwidth: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return slow-path activations, nearest distance, and novelty."""
    centers = state["centers"]
    used = state["counts"] > 0.0
    distances = jnp.mean(jnp.square(centers - hidden[None, :]), axis=1)
    masked_distances = jnp.where(used, distances, jnp.inf)
    nearest_distance = jnp.min(masked_distances)
    has_used = jnp.any(used)
    nearest_distance = jnp.where(has_used, nearest_distance, jnp.array(0.0, jnp.float32))
    logits = -distances / jnp.maximum(jnp.asarray(bandwidth, dtype=jnp.float32), 1e-6)
    logits = jnp.where(used, logits, -1e9)
    activations = jax.nn.softmax(logits)
    activations = jnp.where(has_used, activations, jnp.zeros_like(activations))
    novelty = jnp.sqrt(jnp.maximum(nearest_distance, 0.0))
    return activations, nearest_distance, novelty


def select_center_slot(
    state: dict[str, jax.Array],
    hidden: jax.Array,
    nearest_distance: jax.Array,
    *,
    novelty_threshold: float,
) -> tuple[jax.Array, jax.Array]:
    """Select the online prototype slot for the current hidden state."""
    used = state["counts"] > 0.0
    has_used = jnp.any(used)
    has_empty = jnp.any(~used)
    distances = jnp.mean(jnp.square(state["centers"] - hidden[None, :]), axis=1)
    nearest_slot = jnp.argmin(jnp.where(used, distances, jnp.inf))
    empty_slot = jnp.argmax((~used).astype(jnp.int32))
    novel = (~has_used) | (
        nearest_distance > jnp.asarray(novelty_threshold, dtype=jnp.float32)
    )
    slot = jnp.where(
        ~has_used,
        jnp.array(0, dtype=jnp.int32),
        jnp.where(
            novel & has_empty,
            empty_slot,
            jnp.where(novel, state["next_slot"], nearest_slot),
        ),
    )
    return slot, novel


def update_fast_slow_state(
    state: dict[str, jax.Array],
    hidden: jax.Array,
    loss: jax.Array,
    nearest_distance: jax.Array,
    novelty: jax.Array,
    *,
    center_update_rate: float,
    loss_decay: float,
    novelty_decay: float,
    novelty_threshold: float,
) -> dict[str, jax.Array]:
    """Update online slow memory and causal diagnostics."""
    slot, novel = select_center_slot(
        state,
        hidden,
        nearest_distance,
        novelty_threshold=novelty_threshold,
    )
    old_count = state["counts"][slot]
    rate = jnp.asarray(center_update_rate, dtype=jnp.float32)
    old_center = state["centers"][slot]
    updated_center = jnp.where(
        novel | (old_count <= 0.0),
        hidden,
        (1.0 - rate) * old_center + rate * hidden,
    )
    counts = state["counts"].at[slot].set(jnp.where(novel, 1.0, old_count + 1.0))
    next_slot = jnp.where(
        novel,
        (state["next_slot"] + 1) % state["counts"].shape[0],
        state["next_slot"],
    )
    return {
        "centers": state["centers"].at[slot].set(updated_center),
        "counts": counts,
        "next_slot": next_slot.astype(jnp.int32),
        "loss_ema": loss_decay * state["loss_ema"] + (1.0 - loss_decay) * loss,
        "novelty_ema": novelty_decay * state["novelty_ema"]
        + (1.0 - novelty_decay) * novelty,
        "step_count": state["step_count"] + jnp.array(1, dtype=jnp.int32),
    }


def fast_slow_transformer_logits(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    context: jax.Array,
    *,
    proto_bandwidth: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return logits plus diagnostics for one context."""
    hidden = causal_attention_sequence(params["attn"], context)[-1]
    activations, nearest_distance, novelty = prototype_activations(
        state,
        hidden,
        bandwidth=proto_bandwidth,
    )
    fast = fast_residual(params["ffn"], hidden)
    slow = activations @ params["slow"]["values"] + params["slow"]["bias"]
    diag = jnp.asarray(
        [
            jnp.log1p(jnp.maximum(state["loss_ema"], 0.0)),
            jnp.log1p(jnp.maximum(novelty, 0.0)),
        ],
        dtype=jnp.float32,
    )
    gate_logits = hidden @ params["gate"]["hidden"] + diag @ params["gate"]["diag"]
    gate = jax.nn.sigmoid(gate_logits + params["gate"]["bias"])
    mixed = hidden + (1.0 - gate) * fast + gate * slow
    logits = mixed @ params["readout"]["w"] + params["readout"]["b"]
    aux = {
        "hidden": hidden,
        "gate": gate,
        "novelty": novelty,
        "nearest_distance": nearest_distance,
        "max_activation": jnp.max(activations),
    }
    return logits, aux


def tree_global_norm(tree: Any) -> jax.Array:
    """Return global L2 norm of a PyTree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves) + 1e-12)


def fast_slow_sgd_step(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    fast_lr: float,
    slow_lr: float,
    gate_lr: float,
) -> dict[str, Any]:
    """Apply decoupled step-sizes to fast, slow, and gate subtrees."""

    def step_tree(subtree: Any, grad_tree: Any, lr: float) -> Any:
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, subtree, grad_tree)

    return {
        "attn": step_tree(params["attn"], grads["attn"], fast_lr),
        "ffn": step_tree(params["ffn"], grads["ffn"], fast_lr),
        "slow": step_tree(params["slow"], grads["slow"], slow_lr),
        "gate": step_tree(params["gate"], grads["gate"], gate_lr),
        "readout": step_tree(params["readout"], grads["readout"], fast_lr),
    }


def run_fast_slow_transformer(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    fast_lr: float,
    slow_lr: float,
    gate_lr: float,
    center_update_rate: float,
    loss_decay: float,
    novelty_decay: float,
    proto_bandwidth: float,
    novelty_threshold: float,
    grad_clip: float,
) -> tuple[dict[str, Any], dict[str, jax.Array], np.ndarray]:
    """Train the fast/slow transformer with one JIT scan."""

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
            ) -> tuple[jax.Array, tuple[jax.Array, dict[str, jax.Array]]]:
                logits, aux = fast_slow_transformer_logits(
                    candidate,
                    state,
                    context,
                    proto_bandwidth=proto_bandwidth,
                )
                return cross_entropy_from_logits(logits, label), (logits, aux)

            (loss, (logits, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            clipped = clip_grads(grads, grad_clip)
            new_params = fast_slow_sgd_step(
                params,
                clipped,
                fast_lr=fast_lr,
                slow_lr=slow_lr,
                gate_lr=gate_lr,
            )
            new_state = update_fast_slow_state(
                state,
                aux["hidden"],
                loss,
                aux["nearest_distance"],
                aux["novelty"],
                center_update_rate=center_update_rate,
                loss_decay=loss_decay,
                novelty_decay=novelty_decay,
                novelty_threshold=novelty_threshold,
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.mean(aux["gate"]),
                    aux["novelty"],
                    state["loss_ema"],
                    state["novelty_ema"],
                    aux["max_activation"],
                    tree_global_norm(clipped["slow"]),
                    tree_global_norm(clipped["ffn"]),
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_fast_slow_transformer(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    proto_bandwidth: float,
) -> dict[str, float]:
    """Evaluate the fast/slow transformer without updating slow memory."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array, jax.Array]:
        def predict(context: jax.Array) -> tuple[jax.Array, jax.Array]:
            logits, aux = fast_slow_transformer_logits(
                params,
                state,
                context,
                proto_bandwidth=proto_bandwidth,
            )
            return logits, jnp.mean(aux["gate"])

        logits, gates = jax.vmap(predict)(contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, labels)
        acc = jnp.argmax(logits, axis=1) == labels
        return losses, acc.astype(jnp.float32), gates

    losses, acc, gates = run()
    losses.block_until_ready()
    mean_loss = float(jnp.mean(losses))
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": float(jnp.mean(acc)),
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
        "eval_mean_gate": float(jnp.mean(gates)),
    }


def eval_baseline_transformer(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    """Evaluate baseline FFN transformer."""

    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(lambda ctx: transformer_logits(params, ctx))(contexts)
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


def summarize_fast_slow_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize fast/slow-specific metrics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_mean_gate": float(np.mean(window[:, 2])),
        "final_window_novelty": float(np.mean(window[:, 3])),
        "final_window_loss_ema": float(np.mean(window[:, 4])),
        "final_window_novelty_ema": float(np.mean(window[:, 5])),
        "final_window_max_activation": float(np.mean(window[:, 6])),
        "final_window_slow_grad_norm": float(np.mean(window[:, 7])),
        "final_window_fast_grad_norm": float(np.mean(window[:, 8])),
    }


def aggregate_metric(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
) -> np.ndarray:
    """Return a metric array for one method."""
    return np.asarray(
        [row["summary"][metric] for row in records if row["method"] == method],
        dtype=np.float64,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    records = payload["records"]
    methods = ["baseline_ffn_transformer", "fast_slow_transformer"]
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
        "# Tiny Shakespeare Fast/Slow Transformer",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "Both methods use the same causal attention stem and linear token readout.",
        "The candidate gates a fast FFN residual against a slow online prototype residual.",
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
            f"| `{method}` | {profile['trainable_params']} | "
            f"{profile['trainable_bytes']} | {profile['state_elements']} | "
            f"{profile['state_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Baseline FFN | Fast/slow | Diff favoring fast/slow |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        candidate = aggregate_metric(records, "fast_slow_transformer", metric)
        diff = baseline - candidate if metric in lower_is_better else candidate - baseline
        lines.append(
            f"| `{metric}` | {np.mean(baseline):.4f} +/- {stderr(baseline):.4f} | "
            f"{np.mean(candidate):.4f} +/- {stderr(candidate):.4f} | "
            f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f} |"
        )
    candidate_records = [row for row in records if row["method"] == "fast_slow_transformer"]
    if candidate_records:
        lines.extend(
            [
                "",
                "## Fast/slow diagnostics",
                "",
                "| Metric | Mean +/- stderr |",
                "|---|---:|",
            ]
        )
        diagnostic_metrics = [
            "final_window_mean_gate",
            "eval_mean_gate",
            "final_window_novelty",
            "final_window_novelty_ema",
            "final_window_max_activation",
            "final_window_slow_grad_norm",
            "final_window_fast_grad_norm",
        ]
        for metric in diagnostic_metrics:
            values = np.asarray(
                [row["summary"][metric] for row in candidate_records],
                dtype=np.float64,
            )
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    lines.extend(
        [
            "",
            "Positive diffs favor the fast/slow transformer. Wall-clock includes "
            "JAX compilation on first use.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--fast-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--fast-lr", type=float, default=0.03)
    parser.add_argument("--slow-lr", type=float, default=0.3)
    parser.add_argument("--gate-lr", type=float, default=0.1)
    parser.add_argument("--center-update-rate", type=float, default=0.05)
    parser.add_argument("--loss-decay", type=float, default=0.98)
    parser.add_argument("--novelty-decay", type=float, default=0.98)
    parser.add_argument("--proto-bandwidth", type=float, default=5.0)
    parser.add_argument("--novelty-threshold", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/fast_slow_transformer_worker/default"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if min(args.d_model, args.mlp_hidden, args.fast_hidden, args.proto_count) < 1:
        raise ValueError("model dimensions must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if min(args.baseline_lr, args.fast_lr, args.slow_lr, args.gate_lr) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if not 0.0 <= args.center_update_rate <= 1.0:
        raise ValueError("--center-update-rate must be in [0, 1]")
    if not 0.0 <= args.loss_decay < 1.0:
        raise ValueError("--loss-decay must be in [0, 1)")
    if not 0.0 <= args.novelty_decay < 1.0:
        raise ValueError("--novelty-decay must be in [0, 1)")
    if args.proto_bandwidth <= 0.0:
        raise ValueError("--proto-bandwidth must be positive")
    if args.novelty_threshold < 0.0:
        raise ValueError("--novelty-threshold must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    return args


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

    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        fast_hidden=args.fast_hidden,
        proto_count=args.proto_count,
        eval_steps=args.eval_steps,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        gate_lr=args.gate_lr,
        center_update_rate=args.center_update_rate,
        loss_decay=args.loss_decay,
        novelty_decay=args.novelty_decay,
        proto_bandwidth=args.proto_bandwidth,
        novelty_threshold=args.novelty_threshold,
        grad_clip=args.grad_clip,
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
    candidate_profile_params = init_fast_slow_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        fast_hidden=args.fast_hidden,
        proto_count=args.proto_count,
    )
    candidate_profile_state = init_fast_slow_state(args.proto_count, args.d_model)
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "fast_slow_transformer": {
            "trainable_params": count_array_elements(candidate_profile_params),
            "trainable_bytes": count_array_bytes(candidate_profile_params),
            "state_elements": count_array_elements(candidate_profile_state, include_int=True),
            "state_bytes": count_array_bytes(candidate_profile_state, include_int=True),
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
        baseline_train_s = time.perf_counter() - method_start
        baseline_summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_baseline_transformer(final_baseline, eval_contexts, eval_labels),
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
            f"fw_nll={baseline_summary['final_window_nll']:.4f}, "
            f"eval_ppl={baseline_summary['eval_perplexity']:.2f}, "
            f"train_s={baseline_train_s:.2f}"
        )

        candidate_params = init_fast_slow_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            fast_hidden=args.fast_hidden,
            proto_count=args.proto_count,
        )
        candidate_state = init_fast_slow_state(args.proto_count, args.d_model)
        method_start = time.perf_counter()
        final_candidate, final_candidate_state, candidate_metrics = run_fast_slow_transformer(
            candidate_params,
            candidate_state,
            contexts,
            labels,
            fast_lr=args.fast_lr,
            slow_lr=args.slow_lr,
            gate_lr=args.gate_lr,
            center_update_rate=args.center_update_rate,
            loss_decay=args.loss_decay,
            novelty_decay=args.novelty_decay,
            proto_bandwidth=args.proto_bandwidth,
            novelty_threshold=args.novelty_threshold,
            grad_clip=args.grad_clip,
        )
        final_candidate_state["centers"].block_until_ready()
        candidate_train_s = time.perf_counter() - method_start
        candidate_summary = {
            **summarize_online(candidate_metrics, final_window),
            **summarize_fast_slow_diagnostics(candidate_metrics, final_window),
            **eval_fast_slow_transformer(
                final_candidate,
                final_candidate_state,
                eval_contexts,
                eval_labels,
                proto_bandwidth=args.proto_bandwidth,
            ),
            "train_s": candidate_train_s,
            "train_steps_per_s": args.steps / candidate_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "fast_slow_transformer",
                "summary": candidate_summary,
            }
        )
        print(
            f"seed={seed_idx} fast_slow_transformer: "
            f"fw_nll={candidate_summary['final_window_nll']:.4f}, "
            f"eval_ppl={candidate_summary['eval_perplexity']:.2f}, "
            f"gate={candidate_summary['final_window_mean_gate']:.3f}, "
            f"train_s={candidate_train_s:.2f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
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

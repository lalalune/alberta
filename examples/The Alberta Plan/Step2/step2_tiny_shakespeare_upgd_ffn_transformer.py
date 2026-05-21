#!/usr/bin/env python3
"""Tiny Shakespeare FFN-only UPGD transformer comparison.

This is a clean isolation experiment for the transformer FFN sublayer. Both
models use the same character stream, one-head causal attention stem, residual
block shape, and linear softmax token readout. The only intended difference is
the FFN module:

1. ``baseline_ffn_transformer``: ordinary residual GELU FFN trained by clipped
   online SGD through next-token cross-entropy.
2. ``upgd_ffn_transformer``: the same residual GELU FFN forward path, augmented
   with UPGD-style feature utility tracking and low-utility perturbations inside
   the FFN only. The token readout remains the same linear softmax head.

The public ``UPGDLearner`` is deliberately not used here because its update owns
the prediction heads and supervised target loss. Dropping it into this script
would again replace the learner/readout rather than only the hidden FFN
transform. This script implements the closest honest FFN-local mechanism: a
fixed-size nonlinear feature bank with explicit utility EMA and utility-scaled
perturbations on FFN feature slots.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    upgd_hidden: int
    eval_steps: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    upgd_lr: float
    grad_clip: float
    utility_decay: float
    perturbation_sigma: float
    perturbation_beta: float
    perturbation_interval: int
    perturbation_warmup_steps: int
    perturbation_ramp_steps: int
    perturbation_noise: str
    perturb_output_weights: bool
    data_path: str
    output_dir: str
    seed: int


def ensure_tiny_shakespeare(path: Path) -> str:
    """Return Tiny Shakespeare text, downloading it into ``path`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def encode_text(text: str) -> tuple[jax.Array, dict[str, Any]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    tokens = jnp.asarray([stoi[ch] for ch in text], dtype=jnp.int32)
    return tokens, {"vocab_size": len(chars), "itos": chars}


def make_examples(
    tokens: jax.Array,
    *,
    steps: int,
    block_size: int,
    offset: int,
) -> tuple[jax.Array, jax.Array]:
    max_start = int(tokens.shape[0]) - block_size - 1
    if max_start <= 0:
        msg = "token stream is too short for the requested block size"
        raise ValueError(msg)
    starts = (jnp.arange(steps, dtype=jnp.int32) + jnp.asarray(offset)) % max_start

    def slice_context(start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(tokens, (start,), (block_size,))

    contexts = jax.vmap(slice_context)(starts)
    labels = tokens[starts + block_size]
    return contexts, labels


def _sinusoidal_positions(block_size: int, d_model: int) -> jax.Array:
    positions = jnp.arange(block_size, dtype=jnp.float32)[:, None]
    dims = jnp.arange(d_model, dtype=jnp.float32)[None, :]
    rates = 1.0 / (10000.0 ** (2.0 * jnp.floor(dims / 2.0) / d_model))
    angles = positions * rates
    return jnp.where((dims.astype(jnp.int32) % 2) == 0, jnp.sin(angles), jnp.cos(angles))


def init_attention_params(
    key: jax.Array,
    vocab_size: int,
    block_size: int,
    d_model: int,
) -> dict[str, jax.Array]:
    emb_key, q_key, k_key, v_key, o_key = jr.split(key, 5)
    scale = 1.0 / math.sqrt(d_model)
    return {
        "token_embed": 0.02 * jr.normal(emb_key, (vocab_size, d_model)),
        "pos_embed": 0.02 * _sinusoidal_positions(block_size, d_model),
        "wq": scale * jr.normal(q_key, (d_model, d_model)),
        "wk": scale * jr.normal(k_key, (d_model, d_model)),
        "wv": scale * jr.normal(v_key, (d_model, d_model)),
        "wo": scale * jr.normal(o_key, (d_model, d_model)),
    }


def init_ffn_params(
    key: jax.Array,
    d_model: int,
    hidden_size: int,
) -> dict[str, jax.Array]:
    w1_key, w2_key = jr.split(key, 2)
    return {
        "w1": jr.normal(w1_key, (d_model, hidden_size)) / math.sqrt(d_model),
        "b1": jnp.zeros((hidden_size,), dtype=jnp.float32),
        "w2": jr.normal(w2_key, (hidden_size, d_model)) / math.sqrt(hidden_size),
        "b2": jnp.zeros((d_model,), dtype=jnp.float32),
    }


def init_readout_params(
    key: jax.Array,
    d_model: int,
    vocab_size: int,
) -> dict[str, jax.Array]:
    return {
        "w": jr.normal(key, (d_model, vocab_size)) / math.sqrt(d_model),
        "b": jnp.zeros((vocab_size,), dtype=jnp.float32),
    }


def init_transformer_params(
    key: jax.Array,
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    ffn_hidden: int,
) -> dict[str, Any]:
    attn_key, ffn_key, readout_key = jr.split(key, 3)
    return {
        "attn": init_attention_params(attn_key, vocab_size, block_size, d_model),
        "ffn": init_ffn_params(ffn_key, d_model, ffn_hidden),
        "readout": init_readout_params(readout_key, d_model, vocab_size),
    }


def init_upgd_ffn_state(key: jax.Array, hidden_size: int) -> dict[str, jax.Array]:
    return {
        "feature_utility": jnp.zeros((hidden_size,), dtype=jnp.float32),
        "key": key,
        "step_count": jnp.array(0, dtype=jnp.int32),
    }


def causal_attention_sequence(
    attn: dict[str, jax.Array],
    context: jax.Array,
) -> jax.Array:
    x = attn["token_embed"][context] + attn["pos_embed"]
    q = x @ attn["wq"]
    k = x @ attn["wk"]
    v = x @ attn["wv"]
    scores = (q @ k.T) / jnp.sqrt(jnp.asarray(x.shape[-1], dtype=jnp.float32))
    mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), dtype=bool))
    scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = (weights @ v) @ attn["wo"]
    return x + attended


def ffn_transform(ffn: dict[str, jax.Array], hidden: jax.Array) -> jax.Array:
    features = jax.nn.gelu(hidden @ ffn["w1"] + ffn["b1"])
    return hidden + features @ ffn["w2"] + ffn["b2"]


def transformer_logits(params: dict[str, Any], context: jax.Array) -> jax.Array:
    hidden = causal_attention_sequence(params["attn"], context)[-1]
    hidden = ffn_transform(params["ffn"], hidden)
    readout = params["readout"]
    return hidden @ readout["w"] + readout["b"]


def cross_entropy_from_logits(logits: jax.Array, label: jax.Array) -> jax.Array:
    return jax.nn.logsumexp(logits) - logits[label]


def clip_grads(grads: Any, max_norm: float) -> Any:
    leaves = jax.tree_util.tree_leaves(grads)
    sq_norm = sum(jnp.sum(jnp.square(g)) for g in leaves)
    norm = jnp.sqrt(sq_norm + 1e-12)
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=jnp.float32) / norm)
    return jax.tree_util.tree_map(lambda g: scale * g, grads)


def sgd_step(params: Any, grads: Any, step_size: float) -> Any:
    return jax.tree_util.tree_map(lambda p, g: p - step_size * g, params, grads)


def _sample_noise(
    key: jax.Array,
    shape: tuple[int, ...],
    perturbation_noise: str,
) -> jax.Array:
    if perturbation_noise == "rademacher":
        return jr.rademacher(key, shape, dtype=jnp.float32)
    return jr.normal(key, shape, dtype=jnp.float32)


def update_upgd_ffn_state_and_params(
    pre_params: dict[str, Any],
    post_sgd_params: dict[str, Any],
    grads: dict[str, Any],
    state: dict[str, jax.Array],
    *,
    utility_decay: float,
    perturbation_sigma: float,
    perturbation_beta: float,
    perturbation_interval: int,
    perturbation_warmup_steps: int,
    perturbation_ramp_steps: int,
    perturbation_noise: str,
    perturb_output_weights: bool,
) -> tuple[dict[str, Any], dict[str, jax.Array], jax.Array]:
    ffn = pre_params["ffn"]
    ffn_grads = grads["ffn"]
    incoming_utility = jnp.mean(jnp.abs(ffn["w1"] * ffn_grads["w1"]), axis=0)
    outgoing_utility = jnp.mean(jnp.abs(ffn["w2"] * ffn_grads["w2"]), axis=1)
    instantaneous = 0.5 * (incoming_utility + outgoing_utility)
    decay = jnp.asarray(utility_decay, dtype=jnp.float32)
    new_utility = decay * state["feature_utility"] + (1.0 - decay) * instantaneous

    step_count = state["step_count"]
    interval = jnp.asarray(perturbation_interval, dtype=jnp.int32)
    after_first_step = step_count > 0
    on_interval = (step_count % interval) == 0
    after_warmup = step_count >= jnp.asarray(perturbation_warmup_steps, dtype=jnp.int32)
    do_perturb = after_first_step & on_interval & after_warmup
    do_perturb = do_perturb & (jnp.asarray(perturbation_sigma, dtype=jnp.float32) > 0.0)

    ramp_steps = jnp.asarray(perturbation_ramp_steps, dtype=jnp.float32)
    warmup_steps = jnp.asarray(perturbation_warmup_steps, dtype=jnp.float32)
    ramp_progress = jnp.where(
        ramp_steps > 0.0,
        (step_count.astype(jnp.float32) - warmup_steps + 1.0)
        / jnp.maximum(ramp_steps, 1.0),
        1.0,
    )
    schedule_scale = jnp.where(after_warmup, jnp.clip(ramp_progress, 0.0, 1.0), 0.0)
    sigma = jnp.asarray(perturbation_sigma, dtype=jnp.float32) * schedule_scale
    beta = jnp.asarray(perturbation_beta, dtype=jnp.float32)
    utility_norm = new_utility / (jnp.max(new_utility) + 1e-12)
    feature_scale = sigma * jnp.power(jnp.maximum(1.0 - utility_norm, 0.0), beta)

    def perturb_branch(key: jax.Array) -> tuple[jax.Array, dict[str, Any], jax.Array]:
        next_key, w1_key, w2_key = jr.split(key, 3)
        w1_noise = _sample_noise(w1_key, post_sgd_params["ffn"]["w1"].shape, perturbation_noise)
        w1_delta = w1_noise * feature_scale[None, :]
        w2_noise = _sample_noise(w2_key, post_sgd_params["ffn"]["w2"].shape, perturbation_noise)
        w2_delta = jnp.where(
            perturb_output_weights,
            w2_noise * feature_scale[:, None],
            jnp.zeros_like(post_sgd_params["ffn"]["w2"]),
        )
        new_ffn = {
            **post_sgd_params["ffn"],
            "w1": post_sgd_params["ffn"]["w1"] + w1_delta,
            "w2": post_sgd_params["ffn"]["w2"] + w2_delta,
        }
        new_params = {**post_sgd_params, "ffn": new_ffn}
        max_perturb = jnp.maximum(jnp.max(jnp.abs(w1_delta)), jnp.max(jnp.abs(w2_delta)))
        return next_key, new_params, max_perturb

    def skip_branch(key: jax.Array) -> tuple[jax.Array, dict[str, Any], jax.Array]:
        return key, post_sgd_params, jnp.array(0.0, dtype=jnp.float32)

    new_key, new_params, max_perturbation = jax.lax.cond(
        do_perturb,
        perturb_branch,
        skip_branch,
        state["key"],
    )
    new_state = {
        "feature_utility": new_utility,
        "key": new_key,
        "step_count": step_count + jnp.array(1, dtype=jnp.int32),
    }
    return new_params, new_state, max_perturbation


def run_baseline_transformer(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
) -> tuple[dict[str, Any], np.ndarray]:
    @jax.jit
    def scan(params: dict[str, Any]) -> tuple[dict[str, Any], jax.Array]:
        def step(carry: dict[str, Any], inputs: tuple[jax.Array, jax.Array]):
            context, label = inputs

            def loss_fn(candidate: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
                logits = transformer_logits(candidate, context)
                return cross_entropy_from_logits(logits, label), logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(carry)
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step(carry, grads, step_size)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            return new_params, jnp.stack([loss, acc])

        return jax.lax.scan(step, params, (contexts, labels))

    final_params, metrics = scan(params)
    metrics.block_until_ready()
    return final_params, np.asarray(metrics)


def run_upgd_ffn_transformer(
    params: dict[str, Any],
    state: dict[str, jax.Array],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    step_size: float,
    grad_clip: float,
    utility_decay: float,
    perturbation_sigma: float,
    perturbation_beta: float,
    perturbation_interval: int,
    perturbation_warmup_steps: int,
    perturbation_ramp_steps: int,
    perturbation_noise: str,
    perturb_output_weights: bool,
) -> tuple[dict[str, Any], dict[str, jax.Array], np.ndarray]:
    @jax.jit
    def scan(
        params: dict[str, Any],
        state: dict[str, jax.Array],
    ) -> tuple[tuple[dict[str, Any], dict[str, jax.Array]], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], dict[str, jax.Array]],
            inputs: tuple[jax.Array, jax.Array],
        ):
            params, state = carry
            context, label = inputs

            def loss_fn(candidate: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
                logits = transformer_logits(candidate, context)
                return cross_entropy_from_logits(logits, label), logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grads = clip_grads(grads, grad_clip)
            post_sgd_params = sgd_step(params, grads, step_size)
            new_params, new_state, max_perturbation = update_upgd_ffn_state_and_params(
                params,
                post_sgd_params,
                grads,
                state,
                utility_decay=utility_decay,
                perturbation_sigma=perturbation_sigma,
                perturbation_beta=perturbation_beta,
                perturbation_interval=perturbation_interval,
                perturbation_warmup_steps=perturbation_warmup_steps,
                perturbation_ramp_steps=perturbation_ramp_steps,
                perturbation_noise=perturbation_noise,
                perturb_output_weights=perturb_output_weights,
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            utility = new_state["feature_utility"]
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.mean(utility),
                    jnp.min(utility),
                    max_perturbation,
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_transformer(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
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


def summarize_online(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    window = metrics[-min(final_window, metrics.shape[0]) :]
    final_nll = float(np.mean(window[:, 0]))
    online_nll = float(np.mean(metrics[:, 0]))
    return {
        "online_nll": online_nll,
        "online_accuracy": float(np.mean(metrics[:, 1])),
        "online_perplexity": float(np.exp(min(online_nll, 20.0))),
        "final_window_nll": final_nll,
        "final_window_accuracy": float(np.mean(window[:, 1])),
        "final_window_perplexity": float(np.exp(min(final_nll, 20.0))),
    }


def summarize_upgd_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "final_window_mean_utility": float(np.mean(window[:, 2])),
        "final_window_min_utility": float(np.mean(window[:, 3])),
        "max_perturbation": float(np.max(metrics[:, 4])),
        "mean_perturbation": float(np.mean(metrics[:, 4])),
    }


def _array_leaf_kind(leaf: Any) -> str | None:
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        return None
    try:
        dtype = np.dtype(leaf.dtype)
    except TypeError:
        return "key" if str(leaf.dtype).startswith("key") else None
    if np.issubdtype(dtype, np.floating):
        return "float"
    if np.issubdtype(dtype, np.integer):
        return "int"
    return None


def count_array_elements(tree: Any, *, include_int: bool = False) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        kind = _array_leaf_kind(leaf)
        if kind == "float" or (include_int and kind in {"int", "key"}):
            total += int(np.prod(leaf.shape, dtype=np.int64))
    return total


def count_array_bytes(tree: Any, *, include_int: bool = False) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        kind = _array_leaf_kind(leaf)
        if kind == "float" or (include_int and kind in {"int", "key"}):
            if kind == "key":
                total += int(np.prod(leaf.shape, dtype=np.int64)) * 8
            else:
                total += int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
    return total


def stderr(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def aggregate_metric(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [row["summary"][metric] for row in records if row["method"] == method],
        dtype=np.float64,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    methods = ["baseline_ffn_transformer", "upgd_ffn_transformer"]
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
        "# Tiny Shakespeare FFN-only UPGD Transformer",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "Both methods use the same causal attention stem and linear softmax readout.",
        "Only the residual FFN sublayer differs.",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        f"Block size: `{payload['config']['block_size']}`. Vocab: `{payload['vocab_size']}`.",
        "",
        "## Architecture and state",
        "",
        "| Method | Trainable params | Extra state elements | Extra state bytes |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        profile = payload["profiles"][method]
        lines.append(
            f"| `{method}` | {profile['trainable_params']} | "
            f"{profile['state_elements']} | {profile['state_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Baseline FFN | UPGD FFN | Diff favoring UPGD |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in metrics:
        baseline = aggregate_metric(records, "baseline_ffn_transformer", metric)
        upgd = aggregate_metric(records, "upgd_ffn_transformer", metric)
        diff = baseline - upgd if metric in lower_is_better else upgd - baseline
        lines.append(
            f"| `{metric}` | {np.mean(baseline):.4f} +/- {stderr(baseline):.4f} | "
            f"{np.mean(upgd):.4f} +/- {stderr(upgd):.4f} | "
            f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f} |"
        )
    upgd_records = [row for row in records if row["method"] == "upgd_ffn_transformer"]
    if upgd_records:
        lines.extend(
            [
                "",
                "## UPGD FFN diagnostics",
                "",
                "| Metric | Mean +/- stderr |",
                "|---|---:|",
            ]
        )
        for metric in [
            "final_window_mean_utility",
            "final_window_min_utility",
            "max_perturbation",
            "mean_perturbation",
        ]:
            values = np.asarray(
                [row["summary"][metric] for row in upgd_records],
                dtype=np.float64,
            )
            lines.append(f"| `{metric}` | {np.mean(values):.6f} +/- {stderr(values):.6f} |")
    lines.extend(
        [
            "",
            "Positive diffs favor the UPGD FFN. Wall-clock includes JAX compilation "
            "on the first seed for each method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--upgd-hidden", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--upgd-lr", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-4)
    parser.add_argument("--perturbation-beta", type=float, default=2.0)
    parser.add_argument("--perturbation-interval", type=int, default=16)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument(
        "--perturbation-noise",
        choices=("normal", "rademacher"),
        default="rademacher",
    )
    parser.add_argument(
        "--no-perturb-output-weights",
        action="store_false",
        dest="perturb_output_weights",
        help="Only perturb incoming FFN feature weights, not feature output rows.",
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
        default=Path("output/subagents/transformer_ffn/upgd_ffn_transformer"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        msg = "--steps, --seeds, and --eval-steps must be positive"
        raise ValueError(msg)
    if args.block_size < 2:
        msg = "--block-size must be at least 2"
        raise ValueError(msg)
    if args.d_model < 1 or args.mlp_hidden < 1 or args.upgd_hidden < 1:
        msg = "--d-model, --mlp-hidden, and --upgd-hidden must be positive"
        raise ValueError(msg)
    if not 0.0 < args.train_fraction < 1.0:
        msg = "--train-fraction must be in (0, 1)"
        raise ValueError(msg)
    if args.baseline_lr < 0.0 or args.upgd_lr < 0.0:
        msg = "learning rates must be non-negative"
        raise ValueError(msg)
    if args.grad_clip <= 0.0:
        msg = "--grad-clip must be positive"
        raise ValueError(msg)
    if not 0.0 <= args.utility_decay < 1.0:
        msg = "--utility-decay must be in [0, 1)"
        raise ValueError(msg)
    if args.perturbation_sigma < 0.0 or args.perturbation_beta < 0.0:
        msg = "perturbation sigma and beta must be non-negative"
        raise ValueError(msg)
    if args.perturbation_interval < 1:
        msg = "--perturbation-interval must be at least 1"
        raise ValueError(msg)
    if args.perturbation_warmup_steps < 0 or args.perturbation_ramp_steps < 0:
        msg = "perturbation warmup and ramp steps must be non-negative"
        raise ValueError(msg)
    return args


def main() -> None:
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
        upgd_hidden=args.upgd_hidden,
        eval_steps=args.eval_steps,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        upgd_lr=args.upgd_lr,
        grad_clip=args.grad_clip,
        utility_decay=args.utility_decay,
        perturbation_sigma=args.perturbation_sigma,
        perturbation_beta=args.perturbation_beta,
        perturbation_interval=args.perturbation_interval,
        perturbation_warmup_steps=args.perturbation_warmup_steps,
        perturbation_ramp_steps=args.perturbation_ramp_steps,
        perturbation_noise=args.perturbation_noise,
        perturb_output_weights=args.perturb_output_weights,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    profile_key, state_profile_key = jr.split(root, 2)
    baseline_profile_params = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    upgd_profile_params = init_transformer_params(
        profile_key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.upgd_hidden,
    )
    upgd_profile_state = init_upgd_ffn_state(state_profile_key, args.upgd_hidden)
    profiles = {
        "baseline_ffn_transformer": {
            "trainable_params": count_array_elements(baseline_profile_params),
            "trainable_bytes": count_array_bytes(baseline_profile_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "upgd_ffn_transformer": {
            "trainable_params": count_array_elements(upgd_profile_params),
            "trainable_bytes": count_array_bytes(upgd_profile_params),
            "state_elements": count_array_elements(upgd_profile_state, include_int=True),
            "state_bytes": count_array_bytes(upgd_profile_state, include_int=True),
        },
    }

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        run_key = jr.fold_in(root, seed_idx)
        param_key, state_key, offset_key = jr.split(run_key, 3)
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
        upgd_params = init_transformer_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.upgd_hidden,
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

        upgd_state = init_upgd_ffn_state(state_key, args.upgd_hidden)
        method_start = time.perf_counter()
        final_upgd, final_upgd_state, upgd_metrics = run_upgd_ffn_transformer(
            upgd_params,
            upgd_state,
            contexts,
            labels,
            step_size=args.upgd_lr,
            grad_clip=args.grad_clip,
            utility_decay=args.utility_decay,
            perturbation_sigma=args.perturbation_sigma,
            perturbation_beta=args.perturbation_beta,
            perturbation_interval=args.perturbation_interval,
            perturbation_warmup_steps=args.perturbation_warmup_steps,
            perturbation_ramp_steps=args.perturbation_ramp_steps,
            perturbation_noise=args.perturbation_noise,
            perturb_output_weights=args.perturb_output_weights,
        )
        final_upgd_state["feature_utility"].block_until_ready()
        upgd_train_s = time.perf_counter() - method_start
        upgd_summary = {
            **summarize_online(upgd_metrics, final_window),
            **summarize_upgd_diagnostics(upgd_metrics, final_window),
            **eval_transformer(final_upgd, eval_contexts, eval_labels),
            "train_s": upgd_train_s,
            "train_steps_per_s": args.steps / upgd_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "upgd_ffn_transformer",
                "summary": upgd_summary,
            }
        )
        print(
            f"seed={seed_idx} upgd_ffn_transformer: "
            f"fw_nll={upgd_summary['final_window_nll']:.3f}, "
            f"eval_ppl={upgd_summary['eval_perplexity']:.2f}, "
            f"train_s={upgd_train_s:.2f}, "
            f"max_perturb={upgd_summary['max_perturbation']:.6f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "profiles": profiles,
        "elapsed_s": time.perf_counter() - start,
        "records": records,
        "upgd_learner_note": (
            "UPGDLearner is not used because its public update owns supervised "
            "prediction heads. This experiment keeps the token readout linear "
            "and applies utility tracking/perturbation only to FFN features."
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

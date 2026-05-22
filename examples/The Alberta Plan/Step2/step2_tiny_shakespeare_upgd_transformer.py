#!/usr/bin/env python3
"""Tiny Shakespeare demo: UPGD transformer vs MLP transformer.

This script implements a small char-level online next-token prediction demo on
Tiny Shakespeare.  It compares:

1. ``mlp_transformer``: a standard tiny one-block causal transformer with a
   trainable MLP/FFN and linear softmax head, trained online with cross-entropy.
2. ``upgd_transformer``: the same kind of trainable one-head causal attention
   front-end, but the MLP/readout learner is replaced by ``UPGDLearner``.  The
   attention front-end receives a normal gradient through the current UPGD
   predictor; the UPGD learner itself updates online with its own Step 2
   utility/perturbation rule.

The implementation is deliberately compact and research-oriented.  It is a
demonstration that the Step 2 learner can be dropped into a transformer-shaped
language-modeling loop, not a claim that this tiny setup is competitive with
modern language models.
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

from alberta_framework import UPGDLearner

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class DemoConfig:
    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    upgd_hidden: int
    eval_steps: int
    train_fraction: float
    mlp_lr: float
    upgd_lr: float
    attention_lr: float
    grad_clip: float
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
    metadata = {
        "vocab_size": len(chars),
        "itos": chars,
    }
    return tokens, metadata


def make_examples(
    tokens: jax.Array,
    *,
    steps: int,
    block_size: int,
    offset: int,
) -> tuple[jax.Array, jax.Array]:
    max_start = int(tokens.shape[0]) - block_size - 1
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
    key, emb_key, q_key, k_key, v_key, o_key = jr.split(key, 6)
    scale = 1.0 / math.sqrt(d_model)
    return {
        "token_embed": 0.02 * jr.normal(emb_key, (vocab_size, d_model)),
        "pos_embed": 0.02 * _sinusoidal_positions(block_size, d_model),
        "wq": scale * jr.normal(q_key, (d_model, d_model)),
        "wk": scale * jr.normal(k_key, (d_model, d_model)),
        "wv": scale * jr.normal(v_key, (d_model, d_model)),
        "wo": scale * jr.normal(o_key, (d_model, d_model)),
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


def attention_observation(attn: dict[str, jax.Array], context: jax.Array) -> jax.Array:
    h = causal_attention_sequence(attn, context)
    return jnp.concatenate([h[-1], jnp.mean(h, axis=0)])


def init_mlp_transformer(
    key: jax.Array,
    vocab_size: int,
    block_size: int,
    d_model: int,
    mlp_hidden: int,
) -> dict[str, Any]:
    key, attn_key, ffn1_key, ffn2_key, out_key = jr.split(key, 5)
    return {
        "attn": init_attention_params(attn_key, vocab_size, block_size, d_model),
        "ffn": {
            "w1": jr.normal(ffn1_key, (d_model, mlp_hidden)) / math.sqrt(d_model),
            "b1": jnp.zeros((mlp_hidden,), dtype=jnp.float32),
            "w2": jr.normal(ffn2_key, (mlp_hidden, d_model)) / math.sqrt(mlp_hidden),
            "b2": jnp.zeros((d_model,), dtype=jnp.float32),
        },
        "readout": {
            "w": jr.normal(out_key, (d_model, vocab_size)) / math.sqrt(d_model),
            "b": jnp.zeros((vocab_size,), dtype=jnp.float32),
        },
    }


def mlp_transformer_logits(params: dict[str, Any], context: jax.Array) -> jax.Array:
    h = causal_attention_sequence(params["attn"], context)[-1]
    ffn = params["ffn"]
    z = jax.nn.gelu(h @ ffn["w1"] + ffn["b1"])
    h = h + z @ ffn["w2"] + ffn["b2"]
    readout = params["readout"]
    return h @ readout["w"] + readout["b"]


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


def run_mlp_transformer(
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
            logits = mlp_transformer_logits(carry, context)
            loss = cross_entropy_from_logits(logits, label)
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            grads = jax.grad(
                lambda p: cross_entropy_from_logits(
                    mlp_transformer_logits(p, context),
                    label,
                )
            )(carry)
            grads = clip_grads(grads, grad_clip)
            new_params = sgd_step(carry, grads, step_size)
            return new_params, jnp.stack([loss, acc])

        return jax.lax.scan(step, params, (contexts, labels))

    final_params, metrics = scan(params)
    metrics.block_until_ready()
    return final_params, np.asarray(metrics)


def make_upgd_lm_learner(
    vocab_size: int,
    hidden_size: int,
    step_size: float,
) -> UPGDLearner:
    return UPGDLearner.step2_default(
        n_heads=vocab_size,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        readout_mode="softmax_ce",
    )


def run_upgd_transformer(
    attn: dict[str, jax.Array],
    upgd_state: Any,
    learner: UPGDLearner,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    attention_lr: float,
    grad_clip: float,
) -> tuple[dict[str, jax.Array], Any, np.ndarray]:
    vocab_size = learner.n_heads

    @jax.jit
    def scan(
        attn: dict[str, jax.Array],
        upgd_state: Any,
    ) -> tuple[tuple[dict[str, jax.Array], Any], jax.Array]:
        def step(
            carry: tuple[dict[str, jax.Array], Any],
            inputs: tuple[jax.Array, jax.Array],
        ):
            attn_params, state = carry
            context, label = inputs
            obs = attention_observation(attn_params, context)
            probs = learner.predict(state, obs)
            loss = -jnp.log(probs[label] + 1e-8)
            acc = (jnp.argmax(probs) == label).astype(jnp.float32)
            target = jax.nn.one_hot(label, vocab_size, dtype=jnp.float32)

            def attn_loss(params: dict[str, jax.Array]) -> jax.Array:
                candidate_obs = attention_observation(params, context)
                candidate_probs = learner.predict(state, candidate_obs)
                return -jnp.log(candidate_probs[label] + 1e-8)

            attn_grads = clip_grads(jax.grad(attn_loss)(attn_params), grad_clip)
            new_attn = sgd_step(attn_params, attn_grads, attention_lr)
            upgd_result = learner.update(state, obs, target)
            return (new_attn, upgd_result.state), jnp.stack([loss, acc])

        return jax.lax.scan(step, (attn, upgd_state), (contexts, labels))

    (final_attn, final_upgd), metrics = scan(attn, upgd_state)
    metrics.block_until_ready()
    return final_attn, final_upgd, np.asarray(metrics)


def eval_mlp_transformer(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(lambda ctx: mlp_transformer_logits(params, ctx))(contexts)
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


def eval_upgd_transformer(
    attn: dict[str, jax.Array],
    upgd_state: Any,
    learner: UPGDLearner,
    contexts: jax.Array,
    labels: jax.Array,
) -> dict[str, float]:
    @jax.jit
    def run() -> tuple[jax.Array, jax.Array]:
        def one(ctx: jax.Array, label: jax.Array) -> tuple[jax.Array, jax.Array]:
            probs = learner.predict(upgd_state, attention_observation(attn, ctx))
            loss = -jnp.log(probs[label] + 1e-8)
            acc = (jnp.argmax(probs) == label).astype(jnp.float32)
            return loss, acc

        return jax.vmap(one)(contexts, labels)

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


def stderr(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    methods = ["mlp_transformer", "upgd_transformer"]
    metrics = [
        "final_window_nll",
        "final_window_accuracy",
        "final_window_perplexity",
        "eval_nll",
        "eval_accuracy",
        "eval_perplexity",
    ]
    lower_is_better = {
        "final_window_nll",
        "final_window_perplexity",
        "eval_nll",
        "eval_perplexity",
    }
    lines = [
        "# Tiny Shakespeare UPGD Transformer Demo",
        "",
        "Character-level online next-token prediction on Tiny Shakespeare.",
        "The MLP baseline is a tiny one-block causal transformer with an MLP/FFN.",
        "The UPGD model uses trainable causal attention plus a UPGD next-token learner.",
        "",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`.",
        f"Block size: `{payload['config']['block_size']}`. Vocab: `{payload['vocab_size']}`.",
        "",
        "| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        values = {
            method: np.asarray(
                [row["summary"][metric] for row in records if row["method"] == method],
                dtype=np.float64,
            )
            for method in methods
        }
        mlp = values["mlp_transformer"]
        upgd = values["upgd_transformer"]
        diff = mlp - upgd if metric in lower_is_better else upgd - mlp
        lines.append(
            f"| `{metric}` | {np.mean(mlp):.4f} +/- {stderr(mlp):.4f} | "
            f"{np.mean(upgd):.4f} +/- {stderr(upgd):.4f} | "
            f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive diffs favor UPGD. This is a small online demo, not a "
            "modern language-modeling benchmark.",
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
    parser.add_argument("--upgd-hidden", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--mlp-lr", type=float, default=0.03)
    parser.add_argument("--upgd-lr", type=float, default=0.03)
    parser.add_argument("--attention-lr", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/data/tinyshakespeare/input.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step2_tiny_shakespeare_upgd_transformer"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
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
    config = DemoConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        upgd_hidden=args.upgd_hidden,
        eval_steps=args.eval_steps,
        train_fraction=args.train_fraction,
        mlp_lr=args.mlp_lr,
        upgd_lr=args.upgd_lr,
        attention_lr=args.attention_lr,
        grad_clip=args.grad_clip,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        root, mlp_key, upgd_key, upgd_attn_key = jr.split(root, 4)
        train_offset = int(
            jr.randint(
                jr.fold_in(root, seed_idx),
                (),
                0,
                max(1, int(train_tokens.shape[0]) - args.block_size - args.steps - 1),
            )
        )
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

        mlp_params = init_mlp_transformer(
            mlp_key,
            vocab_size,
            args.block_size,
            args.d_model,
            args.mlp_hidden,
        )
        final_mlp, mlp_metrics = run_mlp_transformer(
            mlp_params,
            contexts,
            labels,
            step_size=args.mlp_lr,
            grad_clip=args.grad_clip,
        )
        mlp_summary = {
            **summarize_online(mlp_metrics, args.eval_steps),
            **eval_mlp_transformer(final_mlp, eval_contexts, eval_labels),
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "mlp_transformer",
                "summary": mlp_summary,
            }
        )
        print(
            f"seed={seed_idx} mlp_transformer: "
            f"fw_nll={mlp_summary['final_window_nll']:.3f}, "
            f"eval_ppl={mlp_summary['eval_perplexity']:.2f}"
        )

        learner = make_upgd_lm_learner(vocab_size, args.upgd_hidden, args.upgd_lr)
        upgd_attn = init_attention_params(
            upgd_attn_key,
            vocab_size,
            args.block_size,
            args.d_model,
        )
        upgd_state = learner.init(2 * args.d_model, upgd_key)
        final_attn, final_upgd_state, upgd_metrics = run_upgd_transformer(
            upgd_attn,
            upgd_state,
            learner,
            contexts,
            labels,
            attention_lr=args.attention_lr,
            grad_clip=args.grad_clip,
        )
        upgd_summary = {
            **summarize_online(upgd_metrics, args.eval_steps),
            **eval_upgd_transformer(
                final_attn,
                final_upgd_state,
                learner,
                eval_contexts,
                eval_labels,
            ),
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "upgd_transformer",
                "summary": upgd_summary,
            }
        )
        print(
            f"seed={seed_idx} upgd_transformer: "
            f"fw_nll={upgd_summary['final_window_nll']:.3f}, "
            f"eval_ppl={upgd_summary['eval_perplexity']:.2f}"
        )

    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
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

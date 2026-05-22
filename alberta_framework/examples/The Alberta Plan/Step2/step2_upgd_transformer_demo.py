#!/usr/bin/env python3
"""Tiny attention + UPGD demo for Alberta Plan Step 2.

This is a deliberately small demonstration, not a production transformer.  A
single frozen attention stem turns a token sequence into contextual features,
and the usual MLP readout learner is replaced by the current Step 2
``UPGDLearner``.  The baseline sees exactly the same attention features and
uses ``MultiHeadMLPLearner``.

The online task is non-stationary sequence matching.  Given a sequence, the
query token is position 0.  Even phases ask whether that query token appears
later in the sequence.  Odd phases ask whether it appears an odd number of
times.  Targets are two-head one-hot vectors, so this exercises the
target-structure UPGD path used by the Step 2 learner.
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

from alberta_framework import MultiHeadMLPLearner, ObGDBounding, UPGDLearner


@dataclass(frozen=True)
class DemoConfig:
    steps: int
    seeds: int
    final_window: int
    seq_len: int
    vocab_size: int
    d_model: int
    phase_length: int
    step_size: float
    output_dir: str
    seed: int


def _sinusoidal_positions(seq_len: int, d_model: int) -> jax.Array:
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    dims = jnp.arange(d_model, dtype=jnp.float32)[None, :]
    rates = 1.0 / (10000.0 ** (2.0 * jnp.floor(dims / 2.0) / d_model))
    angles = positions * rates
    return jnp.where((dims.astype(jnp.int32) % 2) == 0, jnp.sin(angles), jnp.cos(angles))


def make_attention_stem(
    vocab_size: int,
    d_model: int,
    seq_len: int,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Create a tiny deterministic attention stem.

    The token embedding is close to one-hot when possible, with a small random
    tail when ``d_model > vocab_size``.  Identity query/key/value projections
    make content matches visible to the readout learner without training the
    attention parameters.
    """
    base_dim = min(vocab_size, d_model)
    token_eye = jnp.eye(vocab_size, base_dim, dtype=jnp.float32)
    if d_model > base_dim:
        key, tail_key = jr.split(key)
        tail = 0.05 * jr.normal(tail_key, (vocab_size, d_model - base_dim))
        token_embed = jnp.concatenate([token_eye, tail], axis=1)
    else:
        token_embed = token_eye
    return {
        "token_embed": token_embed,
        "pos_embed": 0.05 * _sinusoidal_positions(seq_len, d_model),
        "wq": jnp.eye(d_model, dtype=jnp.float32),
        "wk": jnp.eye(d_model, dtype=jnp.float32),
        "wv": jnp.eye(d_model, dtype=jnp.float32),
        "wo": jnp.eye(d_model, dtype=jnp.float32),
    }


@jax.jit
def attention_features(tokens: jax.Array, stem: dict[str, jax.Array]) -> jax.Array:
    """Return contextual features from a one-head attention stem."""
    x = stem["token_embed"][tokens] + stem["pos_embed"]
    q = x @ stem["wq"]
    k = x @ stem["wk"]
    v = x @ stem["wv"]

    # Query position 0 attends to positions 1..end.  Excluding self prevents
    # the match detector from solving the task by always attending to itself.
    scores = (q[0] @ k[1:].T) / jnp.sqrt(jnp.asarray(x.shape[1], dtype=jnp.float32))
    attn = jax.nn.softmax(scores)
    context = (attn @ v[1:]) @ stem["wo"]
    mean_state = jnp.mean(x, axis=0)
    max_attn = jnp.max(attn)
    entropy = -jnp.sum(attn * jnp.log(attn + 1e-8))
    return jnp.concatenate(
        [
            x[0],
            context,
            mean_state,
            jnp.asarray([max_attn, entropy], dtype=jnp.float32),
        ]
    )


def make_stream(
    key: jax.Array,
    *,
    steps: int,
    seq_len: int,
    vocab_size: int,
    stem: dict[str, jax.Array],
    phase_length: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens = jr.randint(key, (steps, seq_len), minval=0, maxval=vocab_size)
    query = tokens[:, 0]
    rest = tokens[:, 1:]
    match_count = jnp.sum(rest == query[:, None], axis=1)
    any_match = match_count > 0
    odd_match = (match_count % 2) == 1
    phase = (jnp.arange(steps) // phase_length) % 2
    labels = jnp.where(phase == 0, any_match, odd_match).astype(jnp.int32)
    targets = jax.nn.one_hot(labels, 2, dtype=jnp.float32)
    observations = jax.vmap(lambda row: attention_features(row, stem))(tokens)
    return observations, targets, labels


def make_mlp(n_heads: int, hidden_size: int, step_size: float) -> MultiHeadMLPLearner:
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=0.5),
        sparsity=0.5,
        use_layer_norm=True,
    )


def run_upgd(
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    step_size: float,
) -> np.ndarray:
    learner = UPGDLearner.step2_default(n_heads=2)
    if step_size != 0.03:
        learner = UPGDLearner(
            n_heads=2,
            hidden_sizes=(32,),
            step_size=step_size,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=0.5,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            perturbation_interval=16,
            loss_normalization="target_structure",
            track_unit_utilities=False,
            track_gradient_history=False,
        )
    state = learner.init(observations.shape[1], key)

    @jax.jit
    def scan(state: Any) -> tuple[Any, jax.Array]:
        def step(carry: Any, inputs: tuple[jax.Array, jax.Array, jax.Array]):
            obs, tgt, label = inputs
            result = learner.update(carry, obs, tgt)
            mse = jnp.mean((result.predictions - tgt) ** 2)
            acc = (jnp.argmax(result.predictions) == label).astype(jnp.float32)
            return result.state, jnp.stack([mse, acc])

        return jax.lax.scan(step, state, (observations, targets, labels))

    _, metrics = scan(state)
    metrics.block_until_ready()
    return np.asarray(metrics)


def run_mlp(
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    step_size: float,
) -> np.ndarray:
    learner = make_mlp(n_heads=2, hidden_size=32, step_size=step_size)
    state = learner.init(observations.shape[1], key)

    @jax.jit
    def scan(state: Any) -> tuple[Any, jax.Array]:
        def step(carry: Any, inputs: tuple[jax.Array, jax.Array, jax.Array]):
            obs, tgt, label = inputs
            result = learner.update(carry, obs, tgt)
            mse = jnp.mean((result.predictions - tgt) ** 2)
            acc = (jnp.argmax(result.predictions) == label).astype(jnp.float32)
            return result.state, jnp.stack([mse, acc])

        return jax.lax.scan(step, state, (observations, targets, labels))

    _, metrics = scan(state)
    metrics.block_until_ready()
    return np.asarray(metrics)


def summarize(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    window = metrics[-final_window:]
    return {
        "online_mse": float(np.mean(metrics[:, 0])),
        "online_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(window[:, 0])),
        "final_window_accuracy": float(np.mean(window[:, 1])),
    }


def stderr(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    methods = ["mlp_attention_readout", "upgd_attention_readout"]
    metrics = ["final_window_mse", "final_window_accuracy", "online_mse", "online_accuracy"]
    lines = [
        "# Tiny Attention + UPGD Demo",
        "",
        "A frozen one-head attention stem creates contextual sequence features.",
        "The Step 2 UPGD learner replaces the usual MLP readout learner.",
        "",
        f"Seeds: `{payload['config']['seeds']}`. Steps: `{payload['config']['steps']}`.",
        f"Final window: `{payload['config']['final_window']}`.",
        "",
        "| Metric | MLP readout | UPGD readout | UPGD - MLP |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        by_method = {
            method: np.asarray(
                [row["summary"][metric] for row in records if row["method"] == method],
                dtype=np.float64,
            )
            for method in methods
        }
        mlp = by_method["mlp_attention_readout"]
        upgd = by_method["upgd_attention_readout"]
        diff = upgd - mlp if "accuracy" in metric else mlp - upgd
        lines.append(
            f"| `{metric}` | {np.mean(mlp):.4f} +/- {stderr(mlp):.4f} | "
            f"{np.mean(upgd):.4f} +/- {stderr(upgd):.4f} | "
            f"{np.mean(diff):+.4f} +/- {stderr(diff):.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive diffs favor UPGD: for MSE this is `MLP - UPGD`; for "
            "accuracy this is `UPGD - MLP`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--phase-length", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step2_upgd_transformer_demo"),
    )
    args = parser.parse_args()
    if args.seq_len < 3:
        raise ValueError("--seq-len must be at least 3")
    if args.vocab_size < 2 or args.d_model < 2:
        raise ValueError("--vocab-size and --d-model must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DemoConfig(
        steps=args.steps,
        seeds=args.seeds,
        final_window=args.final_window,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        phase_length=args.phase_length,
        step_size=args.step_size,
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    root, stem_key = jr.split(root)
    stem = make_attention_stem(args.vocab_size, args.d_model, args.seq_len, stem_key)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        root, stream_key, mlp_key, upgd_key = jr.split(root, 4)
        observations, targets, labels = make_stream(
            stream_key,
            steps=args.steps,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            stem=stem,
            phase_length=args.phase_length,
        )
        for method, runner, key in [
            ("mlp_attention_readout", run_mlp, mlp_key),
            ("upgd_attention_readout", run_upgd, upgd_key),
        ]:
            metrics = runner(observations, targets, labels, key, args.step_size)
            summary = summarize(metrics, args.final_window)
            records.append(
                {
                    "seed": seed_idx,
                    "method": method,
                    "summary": summary,
                }
            )
            print(
                f"seed={seed_idx} {method}: "
                f"fw_mse={summary['final_window_mse']:.4f}, "
                f"fw_acc={summary['final_window_accuracy']:.4f}"
            )

    payload = {
        "config": asdict(config),
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

#!/usr/bin/env python3
"""Exp05 fast/slow dual-memory learner for Step 2 streams.

This is a standalone moonshot prototype.  It compares exactly three
fast/slow revisions against fair Step 2 MLP baselines and UPGD on shared
materialized stream arrays:

Rev A: fast hidden/output weights update online; slow weights are an EMA copy.
Rev B: Rev A plus an elastic slow anchor, loosened when recent loss spikes.
Rev C: Rev A plus a scalar recent-loss gate interpolating fast/slow predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
    run_multi_head_learning_loop,
    run_upgd_arrays,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/moonshots/exp05_fast_slow_memory"
METHODS = (
    "mlp_64",
    "mlp_64_64",
    "upgd",
    "rev_a_ema_shadow",
    "rev_b_elastic_anchor",
    "rev_c_loss_gate",
)


@dataclass(frozen=True)
class StreamSpec:
    name: str
    feature_dim: int
    n_tasks: int
    n_contexts: int
    context_length: int


@dataclass(frozen=True)
class FastSlowConfig:
    revision: str
    hidden_size: int = 64
    step_size: float = 0.01
    slow_ema: float = 0.995
    sparsity: float = 0.5
    grad_clip: float = 2.0
    anchor_lambda: float = 0.0
    loosened_anchor_fraction: float = 0.1
    drift_threshold: float = 1.25
    loss_fast_decay: float = 0.95
    loss_ref_decay: float = 0.995
    gate_eta: float = 8.0
    gate_discount: float = 0.995
    gate_clip: float = 6.0


class Params(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array


class FastSlowState(NamedTuple):
    fast: Params
    slow: Params
    loss_fast_ema: jax.Array
    loss_ref_ema: jax.Array
    gate_logit: jax.Array


def stream_specs(context_length: int) -> list[tuple[StreamSpec, Any]]:
    return [
        (
            StreamSpec(
                name="out_of_class_polynomial",
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=context_length,
            ),
            lambda: OutOfClassPolynomialStream(
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=context_length,
                active_triples_per_context=2,
                noise_std=0.05,
            ),
        ),
        (
            StreamSpec(
                name="frequency_mismatch",
                feature_dim=4,
                n_tasks=2,
                n_contexts=4,
                context_length=context_length,
            ),
            lambda: FrequencyMismatchStream(
                feature_dim=4,
                n_tasks=2,
                n_contexts=4,
                context_length=context_length,
                n_components_per_task=3,
                noise_std=0.05,
            ),
        ),
    ]


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(step_fn, state, jnp.arange(num_steps))
    return observations, targets


def _mlp_curve(metrics: jax.Array) -> jax.Array:
    return jnp.nanmean(metrics[..., 0], axis=-1)


def run_mlp_all_seeds(
    hidden_sizes: tuple[int, ...],
    n_tasks: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    keys: jax.Array,
    step_size: float,
    sparsity: float,
    obgd_kappa: float,
) -> jax.Array:
    learner = MultiHeadMLPLearner(
        n_heads=n_tasks,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=obgd_kappa),
        sparsity=sparsity,
        use_layer_norm=True,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_multi_head_learning_loop(learner, state, obs, tgt)
        return _mlp_curve(result.per_head_metrics)

    return jax.vmap(single, in_axes=(0, 0, 0))(keys, observations, targets)


def run_upgd_all_seeds(
    n_tasks: int,
    feature_dim: int,
    observations: jax.Array,
    targets: jax.Array,
    keys: jax.Array,
    step_size: float,
    sparsity: float,
    obgd_kappa: float,
) -> jax.Array:
    learner = UPGDLearner(
        n_heads=n_tasks,
        hidden_sizes=(64,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=obgd_kappa),
        sparsity=sparsity,
        use_layer_norm=True,
        utility_decay=0.995,
        perturbation_sigma=3e-4,
        perturbation_beta=2.0,
        perturbation_interval=1,
    )

    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        state = learner.init(feature_dim, key)
        result = run_upgd_arrays(learner, state, obs, tgt)
        return result.metrics[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(keys, observations, targets)


def init_params(
    feature_dim: int,
    n_tasks: int,
    config: FastSlowConfig,
    key: jax.Array,
) -> Params:
    k1, k2, kmask = jr.split(key, 3)
    keep_prob = 1.0 - config.sparsity
    w1_scale = math.sqrt(2.0 / max(feature_dim * keep_prob, 1.0))
    w2_scale = math.sqrt(1.0 / config.hidden_size)
    w1 = w1_scale * jr.normal(k1, (feature_dim, config.hidden_size), dtype=jnp.float32)
    mask = jr.bernoulli(kmask, p=keep_prob, shape=w1.shape).astype(jnp.float32)
    w1 = w1 * mask
    w2 = w2_scale * jr.normal(k2, (config.hidden_size, n_tasks), dtype=jnp.float32)
    return Params(
        w1=w1,
        b1=jnp.zeros((config.hidden_size,), dtype=jnp.float32),
        w2=w2,
        b2=jnp.zeros((n_tasks,), dtype=jnp.float32),
    )


def forward(params: Params, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    z = x @ params.w1 + params.b1
    z = (z - jnp.mean(z)) / jnp.sqrt(jnp.var(z) + 1e-5)
    hidden = jnp.tanh(z)
    pred = hidden @ params.w2 + params.b2
    return pred, hidden


def mse_loss(params: Params, x: jax.Array, y: jax.Array) -> jax.Array:
    pred, _ = forward(params, x)
    return jnp.mean((pred - y) ** 2)


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = jtu.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))


def clip_tree(tree: Any, max_norm: float) -> tuple[Any, jax.Array]:
    norm = tree_l2_norm(tree)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jtu.tree_map(lambda x: x * scale, tree), norm


def add_anchor_grad(grads: Params, fast: Params, slow: Params, coeff: jax.Array) -> Params:
    return jtu.tree_map(lambda g, f, s: g + coeff * (f - s), grads, fast, slow)


def param_distance(fast: Params, slow: Params) -> jax.Array:
    diff = jtu.tree_map(lambda f, s: f - s, fast, slow)
    denom = tree_l2_norm(slow) + 1e-8
    return tree_l2_norm(diff) / denom


def run_fast_slow_single(
    config: FastSlowConfig,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    feature_dim = int(observations.shape[1])
    n_tasks = int(targets.shape[1])
    params = init_params(feature_dim, n_tasks, config, key)
    initial_loss = mse_loss(params, observations[0], targets[0])
    state = FastSlowState(
        fast=params,
        slow=params,
        loss_fast_ema=initial_loss,
        loss_ref_ema=initial_loss,
        gate_logit=jnp.array(0.0, dtype=jnp.float32),
    )

    revision_id = {
        "A": jnp.array(0, dtype=jnp.int32),
        "B": jnp.array(1, dtype=jnp.int32),
        "C": jnp.array(2, dtype=jnp.int32),
    }[config.revision]

    def step_fn(
        carry: FastSlowState, xy: tuple[jax.Array, jax.Array]
    ) -> tuple[FastSlowState, jax.Array]:
        x, y = xy
        fast_pred, _ = forward(carry.fast, x)
        slow_pred, _ = forward(carry.slow, x)
        fast_loss = jnp.mean((fast_pred - y) ** 2)
        slow_loss = jnp.mean((slow_pred - y) ** 2)

        fast_w = jax.nn.sigmoid(carry.gate_logit)
        gated_pred = (1.0 - fast_w) * slow_pred + fast_w * fast_pred
        gated_loss = jnp.mean((gated_pred - y) ** 2)
        pred_loss = jnp.where(revision_id == 2, gated_loss, fast_loss)

        loss_value, grads = jax.value_and_grad(mse_loss)(carry.fast, x, y)
        del loss_value

        new_fast_loss_ema = (
            config.loss_fast_decay * carry.loss_fast_ema
            + (1.0 - config.loss_fast_decay) * fast_loss
        )
        new_loss_ref_ema = (
            config.loss_ref_decay * carry.loss_ref_ema + (1.0 - config.loss_ref_decay) * fast_loss
        )
        drift = new_fast_loss_ema > config.drift_threshold * jnp.maximum(new_loss_ref_ema, 1e-8)
        anchor_coeff = jnp.array(config.anchor_lambda, dtype=jnp.float32)
        anchor_coeff = jnp.where(
            revision_id == 1,
            jnp.where(drift, anchor_coeff * config.loosened_anchor_fraction, anchor_coeff),
            0.0,
        )
        anchored_grads = add_anchor_grad(grads, carry.fast, carry.slow, anchor_coeff)
        clipped_grads, grad_norm = clip_tree(anchored_grads, config.grad_clip)
        new_fast = jtu.tree_map(
            lambda p, g: p - config.step_size * g,
            carry.fast,
            clipped_grads,
        )
        new_slow = jtu.tree_map(
            lambda s, f: config.slow_ema * s + (1.0 - config.slow_ema) * f,
            carry.slow,
            new_fast,
        )
        new_logit = config.gate_discount * carry.gate_logit + config.gate_eta * (
            slow_loss - fast_loss
        )
        new_logit = jnp.clip(new_logit, -config.gate_clip, config.gate_clip)
        metrics = jnp.array(
            [
                pred_loss,
                fast_loss,
                slow_loss,
                gated_loss,
                fast_w,
                anchor_coeff,
                drift.astype(jnp.float32),
                param_distance(carry.fast, carry.slow),
                grad_norm,
            ],
            dtype=jnp.float32,
        )
        return (
            FastSlowState(
                fast=new_fast,
                slow=new_slow,
                loss_fast_ema=new_fast_loss_ema,
                loss_ref_ema=new_loss_ref_ema,
                gate_logit=new_logit,
            ),
            metrics,
        )

    _, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    return metrics


def run_fast_slow_all_seeds(
    config: FastSlowConfig,
    observations: jax.Array,
    targets: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    def single(key: jax.Array, obs: jax.Array, tgt: jax.Array) -> jax.Array:
        return run_fast_slow_single(config, key, obs, tgt)[:, 0]

    return jax.vmap(single, in_axes=(0, 0, 0))(keys, observations, targets)


def stderr(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def cohens_d(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0.0 else 0.0


def summarize_curves(curves: dict[str, np.ndarray], final_window: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    window = min(final_window, next(iter(curves.values())).shape[1])
    for method, arr in curves.items():
        final = np.mean(arr[:, -window:], axis=1)
        total = np.mean(arr, axis=1)
        interim = np.mean(arr[:, :window], axis=1)
        summary[method] = {
            "mean_final_window_mse": float(np.mean(final)),
            "stderr_final_window_mse": stderr(final),
            "mean_total_mse": float(np.mean(total)),
            "stderr_total_mse": stderr(total),
            "mean_interim_mse": float(np.mean(interim)),
            "stderr_interim_mse": stderr(interim),
            "median_final_window_mse": float(np.median(final)),
            "min_final_window_mse": float(np.min(final)),
            "max_final_window_mse": float(np.max(final)),
        }
    return summary


def paired_vs_best_mlp(
    curves: dict[str, np.ndarray],
    aggregate: dict[str, Any],
    final_window: int,
) -> dict[str, Any]:
    mlp_names = ("mlp_64", "mlp_64_64")
    best = min(mlp_names, key=lambda name: aggregate[name]["mean_final_window_mse"])
    best_final = np.mean(curves[best][:, -final_window:], axis=1)
    paired: dict[str, Any] = {"best_mlp": best}
    for method, arr in curves.items():
        method_final = np.mean(arr[:, -final_window:], axis=1)
        diff = best_final - method_final
        paired[method] = {
            "best_mlp_minus_method_mean": float(np.mean(diff)),
            "best_mlp_minus_method_stderr": stderr(diff),
            "wins_for_method": int(np.sum(diff > 0.0)),
            "wins_for_best_mlp": int(np.sum(diff < 0.0)),
            "ties": int(np.sum(diff == 0.0)),
            "n": int(diff.shape[0]),
            "cohens_d": cohens_d(diff),
        }
    return paired


def build_records(
    stream_name: str,
    curves: dict[str, np.ndarray],
    final_window: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    window = min(final_window, next(iter(curves.values())).shape[1])
    for method, arr in curves.items():
        for seed, curve in enumerate(arr):
            records.append(
                {
                    "stream": stream_name,
                    "method": method,
                    "seed": int(seed),
                    "final_window_mse": float(np.mean(curve[-window:])),
                    "total_mean_mse": float(np.mean(curve)),
                    "interim_mse": float(np.mean(curve[:window])),
                    "loss_curve": curve.astype(np.float64).tolist(),
                }
            )
    return records


def run_one_stream(
    spec: StreamSpec,
    factory: Any,
    args: argparse.Namespace,
    revisions: dict[str, FastSlowConfig],
) -> dict[str, Any]:
    print(f"\n=== {spec.name}: seeds={args.n_seeds}, steps={args.steps} ===")
    stream = factory()
    observations_list: list[jax.Array] = []
    targets_list: list[jax.Array] = []
    stream_keys: list[jax.Array] = []
    method_keys: dict[str, list[jax.Array]] = {name: [] for name in METHODS}
    for seed in range(args.n_seeds):
        split = jr.split(jr.key(args.seed + seed), len(METHODS) + 1)
        stream_keys.append(split[0])
        for idx, method in enumerate(METHODS):
            method_keys[method].append(split[idx + 1])
    t0 = time.time()
    for key in stream_keys:
        obs, tgt = collect_stream_arrays(stream, args.steps, key)
        observations_list.append(obs)
        targets_list.append(tgt)
    observations = jnp.stack(observations_list)
    targets = jnp.stack(targets_list)
    observations.block_until_ready()
    targets.block_until_ready()
    print(f"  materialized arrays in {time.time() - t0:.1f}s")

    curves: dict[str, np.ndarray] = {}

    t = time.time()
    result = run_mlp_all_seeds(
        (64,),
        spec.n_tasks,
        spec.feature_dim,
        observations,
        targets,
        jnp.stack(method_keys["mlp_64"]),
        args.mlp_step_size,
        args.sparsity,
        args.obgd_kappa,
    )
    result.block_until_ready()
    curves["mlp_64"] = np.asarray(result)
    final_mse = curves["mlp_64"][:, -args.final_window :].mean()
    print(f"  mlp_64 final={final_mse:.4f} ({time.time() - t:.1f}s)")

    t = time.time()
    result = run_mlp_all_seeds(
        (64, 64),
        spec.n_tasks,
        spec.feature_dim,
        observations,
        targets,
        jnp.stack(method_keys["mlp_64_64"]),
        args.mlp_step_size,
        args.sparsity,
        args.obgd_kappa,
    )
    result.block_until_ready()
    curves["mlp_64_64"] = np.asarray(result)
    final_mse = curves["mlp_64_64"][:, -args.final_window :].mean()
    print(f"  mlp_64_64 final={final_mse:.4f} ({time.time() - t:.1f}s)")

    t = time.time()
    result = run_upgd_all_seeds(
        spec.n_tasks,
        spec.feature_dim,
        observations,
        targets,
        jnp.stack(method_keys["upgd"]),
        args.mlp_step_size,
        args.sparsity,
        args.obgd_kappa,
    )
    result.block_until_ready()
    curves["upgd"] = np.asarray(result)
    final_mse = curves["upgd"][:, -args.final_window :].mean()
    print(f"  upgd final={final_mse:.4f} ({time.time() - t:.1f}s)")

    for method_name, config in revisions.items():
        t = time.time()
        result = run_fast_slow_all_seeds(
            config,
            observations,
            targets,
            jnp.stack(method_keys[method_name]),
        )
        result.block_until_ready()
        curves[method_name] = np.asarray(result)
        final_mse = curves[method_name][:, -args.final_window :].mean()
        print(f"  {method_name} final={final_mse:.4f} ({time.time() - t:.1f}s)")

    aggregate = summarize_curves(curves, args.final_window)
    paired = paired_vs_best_mlp(curves, aggregate, args.final_window)
    return {
        "stream": spec.name,
        "stream_meta": asdict(spec),
        "aggregate": aggregate,
        "paired_vs_best_mlp": paired,
        "records": build_records(spec.name, curves, args.final_window),
    }


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["stream", "method", "seed", "final_window_mse", "total_mean_mse", "interim_mse"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})


def build_conclusion(stream_results: dict[str, Any]) -> dict[str, Any]:
    revision_names = ("rev_a_ema_shadow", "rev_b_elastic_anchor", "rev_c_loss_gate")
    wins: dict[str, list[str]] = {name: [] for name in revision_names}
    for stream_name, result in stream_results.items():
        paired = result["paired_vs_best_mlp"]
        for name in revision_names:
            p = paired[name]
            mean_beats = p["best_mlp_minus_method_mean"] > 0.0
            paired_beats = p["wins_for_method"] > p["wins_for_best_mlp"]
            if mean_beats and paired_beats:
                wins[name].append(stream_name)
    any_win = any(wins.values())
    return {
        "revisions_that_beat_best_mlp": wins,
        "worth_scaling": bool(any_win),
        "reason": (
            "At least one revision beats the best MLP by mean and paired seed wins."
            if any_win
            else "No fast/slow revision beats the best MLP by both mean and paired seed wins."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--context-length", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--fs-step-size", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--obgd-kappa", type=float, default=2.0)
    parser.add_argument("--slow-ema", type=float, default=0.995)
    parser.add_argument("--anchor-lambda", type=float, default=0.01)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0 or args.final_window > args.steps:
        raise ValueError("--final-window must be in [1, steps]")
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in [0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    revisions = {
        "rev_a_ema_shadow": FastSlowConfig(
            revision="A",
            step_size=args.fs_step_size,
            slow_ema=args.slow_ema,
            sparsity=args.sparsity,
        ),
        "rev_b_elastic_anchor": FastSlowConfig(
            revision="B",
            step_size=args.fs_step_size,
            slow_ema=args.slow_ema,
            sparsity=args.sparsity,
            anchor_lambda=args.anchor_lambda,
        ),
        "rev_c_loss_gate": FastSlowConfig(
            revision="C",
            step_size=args.fs_step_size,
            slow_ema=args.slow_ema,
            sparsity=args.sparsity,
        ),
    }

    t0 = time.time()
    stream_results: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for spec, factory in stream_specs(args.context_length):
        result = run_one_stream(spec, factory, args, revisions)
        stream_results[spec.name] = {
            key: value for key, value in result.items() if key != "records"
        }
        all_records.extend(result["records"])

    results = {
        "experiment": "exp05_fast_slow_memory",
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "context_length": args.context_length,
            "mlp_step_size": args.mlp_step_size,
            "fs_step_size": args.fs_step_size,
            "sparsity": args.sparsity,
            "obgd_kappa": args.obgd_kappa,
            "slow_ema": args.slow_ema,
            "anchor_lambda": args.anchor_lambda,
            "methods": METHODS,
            "revision_configs": {name: asdict(cfg) for name, cfg in revisions.items()},
            "wall_clock_s": time.time() - t0,
        },
        "stream_results": stream_results,
        "records": all_records,
    }
    results["conclusion"] = build_conclusion(stream_results)

    json_path = output_dir / "results.json"
    csv_path = output_dir / "records.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_records_csv(csv_path, all_records)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(json.dumps(results["conclusion"], indent=2))


if __name__ == "__main__":
    main()

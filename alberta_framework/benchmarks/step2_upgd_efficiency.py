#!/usr/bin/env python3
"""Throughput and state-size benchmark for Step 2 UPGD vs fair MLP.

The benchmark intentionally separates compile time from steady-state update
time. It scans over pre-generated arrays with a single JIT-compiled loop,
blocks on the result, and reports examples/second, trainable parameters, and
float state size.
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
class BenchSpec:
    name: str
    learner_type: str
    hidden_size: int
    loss_normalization: str = "target_structure"
    perturbation_sigma: float = 0.0
    perturbation_interval: int = 1
    perturbation_noise: str = "normal"
    step_size_multiplier: float = 1.0
    adaptive: bool = False
    meta_readout: bool = False
    repetition_multiplier: float = 0.0
    lean_tracking: bool = False


def make_targets(key: jax.Array, steps: int, n_heads: int, mode: str) -> jax.Array:
    if mode == "onehot":
        labels = jr.randint(key, (steps,), minval=0, maxval=n_heads)
        return jax.nn.one_hot(labels, n_heads, dtype=jnp.float32)
    if mode == "dense":
        raw = jr.normal(key, (steps, n_heads), dtype=jnp.float32)
        return jnp.tanh(raw)
    msg = f"unknown target mode {mode!r}"
    raise ValueError(msg)


def make_mlp(spec: BenchSpec, n_heads: int, step_size: float, sparsity: float):
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(spec.hidden_size,),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )


def make_upgd(spec: BenchSpec, n_heads: int, step_size: float, sparsity: float):
    return UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=(spec.hidden_size,),
        step_size=step_size * spec.step_size_multiplier,
        bounder=ObGDBounding(kappa=0.5),
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=spec.perturbation_sigma,
        perturbation_interval=spec.perturbation_interval,
        perturbation_noise=spec.perturbation_noise,
        loss_normalization=spec.loss_normalization,
        adaptive_kappa_mode="loss_ratio" if spec.adaptive else "none",
        adaptive_kappa_base=0.5,
        adaptive_kappa_min=0.35,
        adaptive_kappa_max=0.65,
        adaptive_kappa_exponent=0.5,
        adaptive_kappa_warmup_steps=120,
        meta_plasticity_mode="gradient_alignment" if spec.meta_readout else "none",
        meta_plasticity_step_size=0.003 if spec.meta_readout else 0.0,
        meta_plasticity_min_multiplier=0.5,
        meta_plasticity_max_multiplier=2.0,
        meta_plasticity_warmup_steps=30,
        meta_plasticity_trunk_enabled=False,
        head_repetition_multiplier=spec.repetition_multiplier,
        track_unit_utilities=not spec.lean_tracking,
        track_gradient_history=not spec.lean_tracking,
    )


def float_state_size(tree: Any) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(
        sum(
            leaf.size
            for leaf in leaves
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
        )
    )


def trainable_param_count(state: Any) -> int:
    params = [
        *state.trunk_params.weights,
        *state.trunk_params.biases,
        *state.head_params.weights,
        *state.head_params.biases,
    ]
    return int(sum(param.size for param in params))


def block_until_ready(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        leaves[0].block_until_ready()


def make_runner(learner: Any, learner_type: str):
    if learner_type == "mlp":

        @jax.jit
        def run(state: Any, observations: jax.Array, targets: jax.Array):
            def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]):
                obs, tgt = inputs
                result = learner.update(carry, obs, tgt)
                return result.state, result.per_head_metrics

            return jax.lax.scan(step_fn, state, (observations, targets))

        return run

    @jax.jit
    def run(state: Any, observations: jax.Array, targets: jax.Array):
        def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]):
            obs, tgt = inputs
            result = learner.update(carry, obs, tgt)
            return result.state, result.metrics

        return jax.lax.scan(step_fn, state, (observations, targets))

    return run


def bench_one(
    spec: BenchSpec,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    repeats: int,
    step_size: float,
    sparsity: float,
) -> dict[str, Any]:
    n_heads = targets.shape[1]
    if spec.learner_type == "mlp":
        learner = make_mlp(spec, n_heads, step_size, sparsity)
    else:
        learner = make_upgd(spec, n_heads, step_size, sparsity)
    state = learner.init(observations.shape[1], key)
    runner = make_runner(learner, spec.learner_type)

    warm_start = time.perf_counter()
    warm_final, warm_metrics = runner(state, observations, targets)
    block_until_ready((warm_final, warm_metrics))
    compile_plus_first_s = time.perf_counter() - warm_start

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        final_state, metrics = runner(state, observations, targets)
        block_until_ready((final_state, metrics))
        times.append(time.perf_counter() - start)

    arr = np.asarray(times, dtype=np.float64)
    mean_s = float(np.mean(arr))
    stderr_s = (
        float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))
        if arr.shape[0] > 1
        else 0.0
    )
    steps = int(observations.shape[0])
    return {
        **asdict(spec),
        "steps": steps,
        "compile_plus_first_s": compile_plus_first_s,
        "mean_s": mean_s,
        "stderr_s": stderr_s,
        "steps_per_second": steps / mean_s,
        "trainable_params": trainable_param_count(state),
        "float_state_size": float_state_size(state),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = sorted(
        payload["results"],
        key=lambda r: (r["target_mode"], -r["steps_per_second"]),
    )
    lines = [
        "# Step 2 UPGD Efficiency Benchmark",
        "",
        f"Steps per run: `{payload['config']['steps']}`.",
        f"Repeats: `{payload['config']['repeats']}`.",
        "Times exclude compilation after one warmup run.",
        "",
        "| Target mode | Method | Hidden | Steps/s | Time s | Trainable params | Float state |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target_mode} | `{name}` | {hidden_size} | {steps_per_second:.1f} | "
            "{mean_s:.4f} +/- {stderr_s:.4f} | {trainable_params} | "
            "{float_state_size} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_specs() -> list[BenchSpec]:
    return [
        BenchSpec("mlp16", "mlp", 16),
        BenchSpec("mlp32", "mlp", 32),
        BenchSpec("mlp64", "mlp", 64),
        BenchSpec(
            "upgd16_structure_simple",
            "upgd",
            16,
            perturbation_sigma=1e-4,
        ),
        BenchSpec(
            "upgd16_structure_rademacher_lean_interval16",
            "upgd",
            16,
            perturbation_sigma=1e-4,
            perturbation_interval=16,
            perturbation_noise="rademacher",
            lean_tracking=True,
        ),
        BenchSpec(
            "upgd32_structure_simple",
            "upgd",
            32,
            perturbation_sigma=1e-4,
        ),
        BenchSpec(
            "upgd32_structure_rademacher_lean_interval16",
            "upgd",
            32,
            perturbation_sigma=1e-4,
            perturbation_interval=16,
            perturbation_noise="rademacher",
            lean_tracking=True,
        ),
        BenchSpec(
            "upgd64_structure_sigma0",
            "upgd",
            64,
            perturbation_sigma=0.0,
        ),
        BenchSpec(
            "upgd64_structure_simple",
            "upgd",
            64,
            perturbation_sigma=1e-4,
        ),
        BenchSpec(
            "upgd64_structure_simple_interval4",
            "upgd",
            64,
            perturbation_sigma=1e-4,
            perturbation_interval=4,
        ),
        BenchSpec(
            "upgd64_structure_simple_interval16",
            "upgd",
            64,
            perturbation_sigma=1e-4,
            perturbation_interval=16,
        ),
        BenchSpec(
            "upgd64_structure_lean_interval16",
            "upgd",
            64,
            perturbation_sigma=1e-4,
            perturbation_interval=16,
            lean_tracking=True,
        ),
        BenchSpec(
            "upgd64_structure_rademacher_lean_interval16",
            "upgd",
            64,
            perturbation_sigma=1e-4,
            perturbation_interval=16,
            perturbation_noise="rademacher",
            lean_tracking=True,
        ),
        BenchSpec(
            "upgd64_structure_meta",
            "upgd",
            64,
            perturbation_sigma=1e-4,
            step_size_multiplier=0.6,
            adaptive=True,
            meta_readout=True,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--target-modes",
        default="onehot,dense",
        help="Comma-separated target modes: onehot,dense.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/benchmarks/step2_upgd_efficiency"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0 or args.feature_dim <= 0 or args.n_heads <= 0:
        raise ValueError("--steps, --feature-dim, and --n-heads must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = jr.key(args.seed)
    key, obs_key = jr.split(key)
    observations = jr.normal(
        obs_key,
        (args.steps, args.feature_dim),
        dtype=jnp.float32,
    )

    results: list[dict[str, Any]] = []
    for target_mode in [m.strip() for m in args.target_modes.split(",") if m.strip()]:
        key, target_key = jr.split(key)
        targets = make_targets(target_key, args.steps, args.n_heads, target_mode)
        for spec_idx, spec in enumerate(default_specs()):
            key, init_key = jr.split(key)
            result = bench_one(
                spec,
                observations,
                targets,
                jr.fold_in(init_key, spec_idx),
                args.repeats,
                args.step_size,
                args.sparsity,
            )
            result["target_mode"] = target_mode
            results.append(result)
            print(
                f"{target_mode} {spec.name}: "
                f"{result['steps_per_second']:.1f} steps/s"
            )

    payload = {
        "config": {
            "steps": args.steps,
            "feature_dim": args.feature_dim,
            "n_heads": args.n_heads,
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "repeats": args.repeats,
            "seed": args.seed,
            "target_modes": args.target_modes,
        },
        "results": results,
    }
    json_path = args.output_dir / "efficiency_results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe computational tradeoffs for continuous DiffEML circuits."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.continuous_diffeml import (
    compile_sparse_continuous_eml_circuit,
    compiled_sparse_continuous_eml_circuit_forward,
    compiled_sparse_continuous_eml_circuit_forward_approx,
    compiled_sparse_continuous_eml_parameter_count,
    continuous_diffeml_forward,
    init_continuous_diffeml_params,
    init_sparse_continuous_eml_circuit_params,
    sparse_continuous_eml_circuit_forward,
    sparse_continuous_eml_parameter_count,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a warmed-JIT inference benchmark."""

    batch_size: int
    input_dim: int
    output_dim: int
    width: int
    depth: int
    repeats: int
    warmups: int
    seed: int
    output_dir: str


class MLPParams(NamedTuple):
    """Simple GELU MLP parameters used as a dense baseline."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]


class BenchmarkState(NamedTuple):
    """Initialized inputs and model parameters for the probe."""

    inputs: jax.Array
    dense_params: Any
    sparse_params: Any
    compiled_sparse: Any
    mlp_params: MLPParams


APPROX_BENCHMARK_NAMES = (
    "sparse_compiled_approx_lut",
    "sparse_compiled_approx_lut_fast_tanh",
    "sparse_compiled_approx_poly",
    "sparse_compiled_approx_poly_fast_tanh",
)


def tree_scalar_count(tree: object) -> int:
    """Count scalar leaves in a JAX PyTree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(int(getattr(leaf, "size", 0)) for leaf in leaves))


def init_mlp_params(
    key: jax.Array,
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
) -> MLPParams:
    """Initialize a depth-matched GELU MLP baseline."""
    sizes = (input_dim, *([width] * depth), output_dim)
    keys = jr.split(key, 2 * (len(sizes) - 1))
    weights: list[jax.Array] = []
    biases: list[jax.Array] = []
    for idx, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        scale = 1.0 / jnp.sqrt(jnp.asarray(max(in_dim, 1), dtype=jnp.float32))
        weights.append(scale * jr.normal(keys[2 * idx], (in_dim, out_dim), dtype=jnp.float32))
        biases.append(jnp.zeros(out_dim, dtype=jnp.float32))
    return MLPParams(weights=tuple(weights), biases=tuple(biases))


def mlp_forward(params: MLPParams, inputs: jax.Array) -> jax.Array:
    """Run a dense GELU MLP baseline."""
    x = inputs
    for weight, bias in zip(params.weights[:-1], params.biases[:-1], strict=True):
        x = jax.nn.gelu(x @ weight + bias)
    return x @ params.weights[-1] + params.biases[-1]


def init_benchmark_state(config: BenchmarkConfig) -> BenchmarkState:
    """Initialize all models on the same random input batch."""
    if min(config.batch_size, config.input_dim, config.output_dim, config.width) < 1:
        raise ValueError("batch_size, input_dim, output_dim, and width must be positive")
    if config.depth < 1:
        raise ValueError("depth must be positive")

    input_key, dense_key, sparse_key, mlp_key = jr.split(jr.key(config.seed), 4)
    inputs = jr.normal(
        input_key,
        (config.batch_size, config.input_dim),
        dtype=jnp.float32,
    )
    dense_params = init_continuous_diffeml_params(
        dense_key,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_sizes=(config.width,) * config.depth,
    )
    sparse_params = init_sparse_continuous_eml_circuit_params(
        sparse_key,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        depth=config.depth,
        width=config.width,
    )
    compiled_sparse = compile_sparse_continuous_eml_circuit(
        sparse_params,
        input_dim=config.input_dim,
    )
    mlp_params = init_mlp_params(
        mlp_key,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        width=config.width,
        depth=config.depth,
    )
    return BenchmarkState(
        inputs=inputs,
        dense_params=dense_params,
        sparse_params=sparse_params,
        compiled_sparse=compiled_sparse,
        mlp_params=mlp_params,
    )


def parameter_summary(state: BenchmarkState) -> dict[str, float | int]:
    """Return trainable and compiled storage counts."""
    sparse_soft = sparse_continuous_eml_parameter_count(state.sparse_params)
    compiled_with_indices = compiled_sparse_continuous_eml_parameter_count(
        state.compiled_sparse,
        count_indices=True,
    )
    compiled_scalars = compiled_sparse_continuous_eml_parameter_count(
        state.compiled_sparse,
        count_indices=False,
    )
    return {
        "dense_continuous_trainable": tree_scalar_count(state.dense_params),
        "sparse_soft_trainable": sparse_soft,
        "sparse_compiled_with_indices": compiled_with_indices,
        "sparse_compiled_scalars_only": compiled_scalars,
        "mlp_trainable": tree_scalar_count(state.mlp_params),
        "sparse_soft_to_compiled_ratio": float(sparse_soft / compiled_with_indices),
    }


def operation_estimates(config: BenchmarkConfig) -> dict[str, int]:
    """Estimate core per-example work from topology, ignoring activation kernels."""
    dense_matmul_muls = 0
    in_dim = config.input_dim
    for _ in range(config.depth):
        dense_matmul_muls += 4 * in_dim * config.width
        in_dim = config.width
    dense_matmul_muls += config.width * config.output_dim

    mlp_muls = config.input_dim * config.width
    mlp_muls += max(config.depth - 1, 0) * config.width * config.width
    mlp_muls += config.width * config.output_dim

    soft_selector_muls = 0
    for layer_idx in range(config.depth):
        source_dim = (
            config.input_dim + 1
            if layer_idx == 0
            else config.input_dim + 1 + config.width
        )
        soft_selector_muls += 2 * config.width * source_dim

    eml_nodes = config.width * config.depth
    return {
        "dense_continuous_matmul_muls_per_example": dense_matmul_muls,
        "mlp_matmul_muls_per_example": mlp_muls,
        "sparse_soft_selector_muls_per_example": soft_selector_muls,
        "sparse_compiled_gathers_per_example": 2 * eml_nodes,
        "eml_nodes_per_example": eml_nodes,
        "exp_log_tanh_triplets_per_example": eml_nodes,
        "approx_lut_entries": 2 * 257,
        "approx_poly_sqrt_per_example": eml_nodes,
    }


def benchmark_jitted(
    name: str,
    function: Callable[..., jax.Array],
    args: tuple[Any, ...],
    *,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    """Measure warmed JIT latency for one forward function."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")

    compiled = jax.jit(function)
    first_start = time.perf_counter()
    output = compiled(*args)
    output.block_until_ready()
    first_ms = (time.perf_counter() - first_start) * 1000.0

    for _ in range(warmups):
        output = compiled(*args)
        output.block_until_ready()

    start = time.perf_counter()
    for _ in range(repeats):
        output = compiled(*args)
        output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_np = np.asarray(output)
    return {
        "name": name,
        "mean_ms": elapsed_ms / repeats,
        "compile_plus_first_ms": first_ms,
        "output_shape": list(output_np.shape),
        "output_mean": float(np.mean(output_np)),
        "output_std": float(np.std(output_np)),
    }


def run_forward_benchmarks(
    state: BenchmarkState,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    """Run warmed-JIT forward benchmarks for dense, soft, hard, and compiled paths."""
    specs: list[tuple[str, Callable[..., jax.Array], tuple[Any, ...]]] = [
        (
            "dense_continuous_eml",
            continuous_diffeml_forward,
            (state.dense_params, state.inputs),
        ),
        (
            "sparse_soft_training_relaxation",
            sparse_continuous_eml_circuit_forward,
            (state.sparse_params, state.inputs),
        ),
        (
            "sparse_hard_full_bank",
            lambda params, inputs: sparse_continuous_eml_circuit_forward(
                params,
                inputs,
                hard=True,
            ),
            (state.sparse_params, state.inputs),
        ),
        (
            "sparse_compiled_direct_gather",
            compiled_sparse_continuous_eml_circuit_forward,
            (state.compiled_sparse, state.inputs),
        ),
        (
            "sparse_compiled_approx_lut",
            compiled_sparse_continuous_eml_circuit_forward_approx,
            (state.compiled_sparse, state.inputs),
        ),
        (
            "sparse_compiled_approx_lut_fast_tanh",
            lambda circuit, inputs: compiled_sparse_continuous_eml_circuit_forward_approx(
                circuit,
                inputs,
                approximation="lut",
                approximate_tanh=True,
            ),
            (state.compiled_sparse, state.inputs),
        ),
        (
            "sparse_compiled_approx_poly",
            lambda circuit, inputs: compiled_sparse_continuous_eml_circuit_forward_approx(
                circuit,
                inputs,
                approximation="poly",
            ),
            (state.compiled_sparse, state.inputs),
        ),
        (
            "sparse_compiled_approx_poly_fast_tanh",
            lambda circuit, inputs: compiled_sparse_continuous_eml_circuit_forward_approx(
                circuit,
                inputs,
                approximation="poly",
                approximate_tanh=True,
            ),
            (state.compiled_sparse, state.inputs),
        ),
        ("mlp_gelu", mlp_forward, (state.mlp_params, state.inputs)),
    ]
    return [
        benchmark_jitted(
            name,
            function,
            args,
            repeats=config.repeats,
            warmups=config.warmups,
        )
        for name, function, args in specs
    ]


def derived_metrics(
    counts: dict[str, float | int],
    benchmarks: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute headline compression and latency ratios."""
    rows = {row["name"]: row for row in benchmarks}
    compiled_ms = float(rows["sparse_compiled_direct_gather"]["mean_ms"])
    soft_ms = float(rows["sparse_soft_training_relaxation"]["mean_ms"])
    hard_bank_ms = float(rows["sparse_hard_full_bank"]["mean_ms"])
    mlp_ms = float(rows["mlp_gelu"]["mean_ms"])
    metrics = {
        "compiled_storage_compression_vs_soft_sparse": float(
            counts["sparse_soft_to_compiled_ratio"]
        ),
        "compiled_latency_speedup_vs_soft_sparse": soft_ms / compiled_ms,
        "compiled_latency_speedup_vs_hard_full_bank": hard_bank_ms / compiled_ms,
        "compiled_latency_ratio_vs_mlp": compiled_ms / mlp_ms,
    }
    for name in APPROX_BENCHMARK_NAMES:
        method_ms = float(rows[name]["mean_ms"])
        metric_prefix = name.removeprefix("sparse_compiled_")
        metrics[f"{metric_prefix}_latency_speedup_vs_exact_compiled"] = (
            compiled_ms / method_ms
        )
        metrics[f"{metric_prefix}_latency_ratio_vs_mlp"] = method_ms / mlp_ms
    return metrics


def approximation_error_summary(state: BenchmarkState) -> dict[str, dict[str, float]]:
    """Compare approximate compiled kernels against exact compiled logits."""
    exact = compiled_sparse_continuous_eml_circuit_forward(
        state.compiled_sparse,
        state.inputs,
    )
    candidates = {
        "sparse_compiled_approx_lut": compiled_sparse_continuous_eml_circuit_forward_approx(
            state.compiled_sparse,
            state.inputs,
            approximation="lut",
        ),
        "sparse_compiled_approx_lut_fast_tanh": (
            compiled_sparse_continuous_eml_circuit_forward_approx(
                state.compiled_sparse,
                state.inputs,
                approximation="lut",
                approximate_tanh=True,
            )
        ),
        "sparse_compiled_approx_poly": compiled_sparse_continuous_eml_circuit_forward_approx(
            state.compiled_sparse,
            state.inputs,
            approximation="poly",
        ),
        "sparse_compiled_approx_poly_fast_tanh": (
            compiled_sparse_continuous_eml_circuit_forward_approx(
                state.compiled_sparse,
                state.inputs,
                approximation="poly",
                approximate_tanh=True,
            )
        ),
    }
    errors: dict[str, dict[str, float]] = {}
    exact_top1 = jnp.argmax(exact, axis=-1)
    for name, logits in candidates.items():
        delta = logits - exact
        errors[name] = {
            "mean_abs_logit_error": float(jnp.mean(jnp.abs(delta))),
            "max_abs_logit_error": float(jnp.max(jnp.abs(delta))),
            "rms_logit_error": float(jnp.sqrt(jnp.mean(delta * delta))),
            "top1_agreement": float(jnp.mean(jnp.argmax(logits, axis=-1) == exact_top1)),
        }
    return errors


def build_benchmark_payload(config: BenchmarkConfig) -> dict[str, Any]:
    """Build the full benchmark result payload."""
    state = init_benchmark_state(config)
    counts = parameter_summary(state)
    operations = operation_estimates(config)
    benchmarks = run_forward_benchmarks(state, config)
    derived = derived_metrics(counts, benchmarks)
    approximation_errors = approximation_error_summary(state)
    return {
        "schema_version": "continuous_diffeml_performance.v1",
        "config": asdict(config),
        "parameter_counts": counts,
        "operation_estimates": operations,
        "derived_metrics": derived,
        "approximation_errors": approximation_errors,
        "forward_benchmarks": benchmarks,
        "observations": [
            "Soft sparse EML is a training relaxation; it still pays dense selector cost.",
            "Compiled sparse EML replaces selector matrices with fixed source indices.",
            "Approximate compiled kernels trade exact EML semantics for cheaper inference.",
            "The main plausible advantage is compact hardened inference, not faster soft training.",
            (
                "Runtime gains need fused gather plus EML kernels; "
                "plain JAX gather can hide the storage win."
            ),
        ],
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary next to the JSON artifact."""
    counts = payload["parameter_counts"]
    derived = payload["derived_metrics"]
    rows = {row["name"]: row for row in payload["forward_benchmarks"]}
    param_by_name = {
        "dense_continuous_eml": counts["dense_continuous_trainable"],
        "sparse_soft_training_relaxation": counts["sparse_soft_trainable"],
        "sparse_hard_full_bank": counts["sparse_soft_trainable"],
        "sparse_compiled_direct_gather": counts["sparse_compiled_with_indices"],
        "mlp_gelu": counts["mlp_trainable"],
    }
    for name in APPROX_BENCHMARK_NAMES:
        param_by_name[name] = counts["sparse_compiled_with_indices"]
    lines = [
        "# Continuous DiffEML Performance Probe",
        "",
        "Warmed-JIT forward latency and storage counts for continuous EML variants.",
        "",
        "| Method | Mean forward ms | Compile + first ms | Stored scalars/indices |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "dense_continuous_eml",
        "sparse_soft_training_relaxation",
        "sparse_hard_full_bank",
        "sparse_compiled_direct_gather",
        *APPROX_BENCHMARK_NAMES,
        "mlp_gelu",
    ):
        row = rows[name]
        lines.append(
            f"| `{name}` | {row['mean_ms']:.4f} | "
            f"{row['compile_plus_first_ms']:.2f} | {param_by_name[name]} |"
        )
    lines.extend(
        [
            "",
            "## Storage Compression",
            "",
            (
                "`sparse_soft_trainable / sparse_compiled_with_indices` = "
                f"{counts['sparse_soft_to_compiled_ratio']:.2f}x"
            ),
            "",
            "## Latency Ratios",
            "",
            (
                "`sparse_compiled_direct_gather` speedup vs soft sparse = "
                f"{derived['compiled_latency_speedup_vs_soft_sparse']:.2f}x"
            ),
            (
                "`sparse_compiled_direct_gather` speedup vs hard full bank = "
                f"{derived['compiled_latency_speedup_vs_hard_full_bank']:.2f}x"
            ),
            (
                "`sparse_compiled_direct_gather / mlp_gelu` latency ratio = "
                f"{derived['compiled_latency_ratio_vs_mlp']:.2f}x"
            ),
        ]
    )
    for name in APPROX_BENCHMARK_NAMES:
        metric_prefix = name.removeprefix("sparse_compiled_")
        lines.append(
            f"`{name}` speedup vs exact compiled = "
            f"{derived[f'{metric_prefix}_latency_speedup_vs_exact_compiled']:.2f}x"
        )
    lines.extend(
        [
            "",
            "## Approximation Error",
            "",
            (
                "| Method | Mean abs logit error | Max abs logit error | "
                "RMS logit error | Top-1 agreement |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, error in payload["approximation_errors"].items():
        lines.append(
            f"| `{name}` | {error['mean_abs_logit_error']:.6f} | "
            f"{error['max_abs_logit_error']:.6f} | {error['rms_logit_error']:.6f} | "
            f"{error['top1_agreement']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["observations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diffeml_continuous_demo/performance_probe"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        width=args.width,
        depth=args.depth,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
        output_dir=str(args.output_dir),
    )
    payload = build_benchmark_payload(config)
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

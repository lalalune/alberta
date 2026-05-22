"""CPU throughput benchmark for the Horde learner.

Measures steps/sec across configurations of `n_demons`, `hidden_sizes`,
trace decay (on/off), and feature normalization (on/off). Required for
Step 3 DoD-10: real-time bound (>= 50 demons, traces enabled, >= 1000
steps/sec on CPU).

Each configuration is benchmarked independently — a failure in one
configuration is logged and reporting continues for the rest.

Usage
-----
```bash
python benchmarks/horde_throughput.py
python benchmarks/horde_throughput.py --n-steps 5000
python benchmarks/horde_throughput.py --feature-dim 64 --output-dir output/step3_throughput
```

Output
------
Prints a results table to stdout and saves a CSV to
``output/step3_throughput/results_<timestamp>.csv`` with columns:
``n_demons, hidden_sizes_str, traces, normalizer, steps_per_sec,
total_seconds, jit_warmup_seconds``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Force CPU even if a GPU/TPU is present so the baseline is reproducible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from alberta_framework import (  # noqa: E402
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeLearner,
    create_horde_spec,
    run_horde_learning_loop_final_state,
)
from alberta_framework.utils.timing import format_duration  # noqa: E402

# =============================================================================
# Config
# =============================================================================

DEFAULT_FEATURE_DIM = 20
DEFAULT_N_STEPS = 10_000
DEFAULT_OUTPUT_DIR = Path("output/step3_throughput")

N_DEMONS_GRID = [5, 25, 100]
HIDDEN_SIZES_GRID: list[tuple[int, ...]] = [(), (32,), (64, 64)]
TRACES_GRID = [False, True]
NORMALIZER_GRID = [None, "EMA"]


@dataclass
class BenchResult:
    n_demons: int
    hidden_sizes: tuple[int, ...]
    traces_on: bool
    normalizer: str  # "none" or "EMA"
    steps_per_sec: float
    total_seconds: float
    jit_warmup_seconds: float
    n_steps: int
    feature_dim: int
    error: str = ""

    @property
    def hidden_sizes_str(self) -> str:
        if not self.hidden_sizes:
            return "()"
        return "x".join(str(h) for h in self.hidden_sizes)


# =============================================================================
# Benchmark
# =============================================================================


def _make_horde(
    n_demons: int,
    hidden_sizes: tuple[int, ...],
    traces_on: bool,
    normalizer_name: str | None,
) -> HordeLearner:
    """Build a HordeLearner for the given configuration."""
    if traces_on:
        gamma, lamda = 0.9, 0.5
    else:
        gamma, lamda = 0.0, 0.0

    demons = [
        GVFSpec(
            name=f"d{i}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamda,
            cumulant_index=i,
        )
        for i in range(n_demons)
    ]
    spec = create_horde_spec(demons)

    normalizer = EMANormalizer() if normalizer_name == "EMA" else None

    return HordeLearner(
        horde_spec=spec,
        hidden_sizes=hidden_sizes,
        step_size=0.01,
        sparsity=0.0,
        normalizer=normalizer,
    )


def _make_synthetic_data(
    n_steps: int,
    feature_dim: int,
    n_demons: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate synthetic (obs, cumulants, next_obs) arrays."""
    k_obs, k_cum, k_next = jr.split(key, 3)
    observations = jr.normal(k_obs, (n_steps, feature_dim), dtype=jnp.float32)
    cumulants = jr.normal(k_cum, (n_steps, n_demons), dtype=jnp.float32) * 0.1
    next_observations = jr.normal(k_next, (n_steps, feature_dim), dtype=jnp.float32)
    return observations, cumulants, next_observations


def _benchmark_one(
    n_demons: int,
    hidden_sizes: tuple[int, ...],
    traces_on: bool,
    normalizer_name: str | None,
    n_steps: int,
    feature_dim: int,
    seed: int = 0,
) -> BenchResult:
    """Benchmark one Horde configuration. Times JIT warmup and a hot run."""
    norm_label = normalizer_name if normalizer_name is not None else "none"
    label = (
        f"n_demons={n_demons:3d} hidden={hidden_sizes!s:>10s} "
        f"traces={traces_on!s:5s} norm={norm_label:4s}"
    )
    print(f"  Running {label} ...", flush=True)

    horde = _make_horde(n_demons, hidden_sizes, traces_on, normalizer_name)

    key = jr.key(seed)
    k_init, k_data = jr.split(key, 2)
    observations, cumulants, next_observations = _make_synthetic_data(
        n_steps, feature_dim, n_demons, k_data
    )

    # JIT warmup: run the loop once at the same shapes as the hot run so
    # the compile cache is hit on the second pass. Use a separate init state
    # so warmup state is discarded.
    warmup_state = horde.init(feature_dim, k_init)
    t_warm0 = time.perf_counter()
    warm_state = run_horde_learning_loop_final_state(
        horde, warmup_state, observations, cumulants, next_observations
    )
    jax.block_until_ready(warm_state)  # type: ignore[no-untyped-call]
    t_warm = time.perf_counter() - t_warm0

    # Hot run: re-init so we don't carry warmup state, but the JIT cache is hot
    # and there's no recompile overhead.
    state = horde.init(feature_dim, k_init)
    t0 = time.perf_counter()
    result_state = run_horde_learning_loop_final_state(
        horde, state, observations, cumulants, next_observations
    )
    jax.block_until_ready(result_state)  # type: ignore[no-untyped-call]
    total = time.perf_counter() - t0

    steps_per_sec = n_steps / total if total > 0 else float("inf")

    print(
        f"    -> {steps_per_sec:>10.1f} steps/sec  "
        f"(total {format_duration(total)}, warmup {format_duration(t_warm)})",
        flush=True,
    )

    return BenchResult(
        n_demons=n_demons,
        hidden_sizes=hidden_sizes,
        traces_on=traces_on,
        normalizer=norm_label,
        steps_per_sec=steps_per_sec,
        total_seconds=total,
        jit_warmup_seconds=t_warm,
        n_steps=n_steps,
        feature_dim=feature_dim,
    )


def run_all_benchmarks(
    n_steps: int,
    feature_dim: int,
    seed: int,
) -> list[BenchResult]:
    """Run every configuration in the grid. Failures are caught and logged."""
    configs = list(
        itertools.product(N_DEMONS_GRID, HIDDEN_SIZES_GRID, TRACES_GRID, NORMALIZER_GRID)
    )
    print(
        f"Horde throughput benchmark: {len(configs)} configurations, "
        f"n_steps={n_steps}, feature_dim={feature_dim}",
        flush=True,
    )
    print(f"Devices: {jax.devices()}", flush=True)

    results: list[BenchResult] = []
    for i, (n_demons, hidden_sizes, traces_on, norm_name) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", flush=True)
        try:
            result = _benchmark_one(
                n_demons=n_demons,
                hidden_sizes=hidden_sizes,
                traces_on=traces_on,
                normalizer_name=norm_name,
                n_steps=n_steps,
                feature_dim=feature_dim,
                seed=seed,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {err_msg}", flush=True)
            traceback.print_exc()
            results.append(
                BenchResult(
                    n_demons=n_demons,
                    hidden_sizes=hidden_sizes,
                    traces_on=traces_on,
                    normalizer=norm_name if norm_name is not None else "none",
                    steps_per_sec=float("nan"),
                    total_seconds=float("nan"),
                    jit_warmup_seconds=float("nan"),
                    n_steps=n_steps,
                    feature_dim=feature_dim,
                    error=err_msg,
                )
            )
    return results


# =============================================================================
# Output
# =============================================================================


def print_results_table(results: list[BenchResult]) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n" + "=" * 92, flush=True)
    print("Horde Throughput Results", flush=True)
    print("=" * 92, flush=True)
    header = (
        f"{'n_demons':>8} {'hidden':>10} {'traces':>6} {'norm':>5} "
        f"{'steps/sec':>12} {'total (s)':>10} {'warmup (s)':>11} {'status':>8}"
    )
    print(header, flush=True)
    print("-" * 92, flush=True)

    for r in results:
        status = "OK" if not r.error else "FAILED"
        sps = f"{r.steps_per_sec:>12.1f}" if r.error == "" else f"{'-':>12s}"
        total = f"{r.total_seconds:>10.2f}" if r.error == "" else f"{'-':>10s}"
        warmup = f"{r.jit_warmup_seconds:>11.2f}" if r.error == "" else f"{'-':>11s}"
        print(
            f"{r.n_demons:>8d} {r.hidden_sizes_str:>10s} "
            f"{str(r.traces_on):>6s} {r.normalizer:>5s} "
            f"{sps} {total} {warmup} {status:>8s}",
            flush=True,
        )
    print("=" * 92, flush=True)

    # DoD check: >= 50 demons, traces enabled, >= 1000 steps/sec
    dod_relevant = [
        r for r in results if r.n_demons >= 50 and r.traces_on and r.error == ""
    ]
    if dod_relevant:
        passing = [r for r in dod_relevant if r.steps_per_sec >= 1000.0]
        failing = [r for r in dod_relevant if r.steps_per_sec < 1000.0]
        print(
            f"\nDoD-10 (>= 50 demons, traces ON, >= 1000 steps/sec): "
            f"{len(passing)}/{len(dod_relevant)} passing",
            flush=True,
        )
        if failing:
            print("  Below 1000 steps/sec target:", flush=True)
            for r in failing:
                print(
                    f"    n_demons={r.n_demons} hidden={r.hidden_sizes_str} "
                    f"norm={r.normalizer}: {r.steps_per_sec:.1f} steps/sec",
                    flush=True,
                )


def save_results_csv(results: list[BenchResult], output_dir: Path) -> Path:
    """Save results to a timestamped CSV. Creates output_dir if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"horde_throughput_{timestamp}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_demons",
                "hidden_sizes_str",
                "traces",
                "normalizer",
                "steps_per_sec",
                "total_seconds",
                "jit_warmup_seconds",
                "n_steps",
                "feature_dim",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n_demons,
                    r.hidden_sizes_str,
                    r.traces_on,
                    r.normalizer,
                    f"{r.steps_per_sec:.4f}",
                    f"{r.total_seconds:.6f}",
                    f"{r.jit_warmup_seconds:.6f}",
                    r.n_steps,
                    r.feature_dim,
                    r.error,
                ]
            )
    print(f"\nResults saved to: {csv_path}", flush=True)
    return csv_path


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Horde throughput benchmark (CPU steps/sec)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Steps per scan (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=DEFAULT_FEATURE_DIM,
        help=f"Feature dimension (default: {DEFAULT_FEATURE_DIM})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing CSV output",
    )
    args = parser.parse_args(argv)

    results = run_all_benchmarks(
        n_steps=args.n_steps,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    print_results_table(results)

    if not args.no_csv:
        save_results_csv(results, args.output_dir)

    # Non-zero exit if any config failed
    failures = [r for r in results if r.error]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

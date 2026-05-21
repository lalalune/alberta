"""CPU throughput benchmark for the SARSA agent.

Measures steps/sec across configurations of `n_actions`, `hidden_sizes`,
and trace decay (on/off). Uses ``run_sarsa_from_arrays`` since it is the
fully JIT-compilable scan path.

Each configuration is benchmarked independently — a failure in one
configuration is logged and reporting continues for the rest.

Usage
-----
```bash
python benchmarks/sarsa_throughput.py
python benchmarks/sarsa_throughput.py --n-steps 5000
python benchmarks/sarsa_throughput.py --feature-dim 64 --output-dir output/step3_throughput
```

Output
------
Prints a results table to stdout and saves a CSV to
``output/step3_throughput/sarsa_throughput_<timestamp>.csv``.
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
from typing import cast

# Force CPU even if a GPU/TPU is present so the baseline is reproducible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from alberta_framework import (  # noqa: E402
    SARSAAgent,
    SARSAConfig,
    SARSAState,
    run_sarsa_from_arrays_final_state,
)
from alberta_framework.utils.timing import format_duration  # noqa: E402

# =============================================================================
# Config
# =============================================================================

DEFAULT_FEATURE_DIM = 20
DEFAULT_N_STEPS = 10_000
DEFAULT_OUTPUT_DIR = Path("output/step3_throughput")

N_ACTIONS_GRID = [4, 8, 16]
HIDDEN_SIZES_GRID: list[tuple[int, ...]] = [(), (32,), (64, 64)]
TRACES_GRID = [False, True]


@dataclass
class BenchResult:
    n_actions: int
    hidden_sizes: tuple[int, ...]
    traces_on: bool
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


def _make_agent(
    n_actions: int,
    hidden_sizes: tuple[int, ...],
    traces_on: bool,
) -> SARSAAgent:
    """Build a SARSAAgent for the given configuration.

    SARSA control demons use gamma=0 internally (target computed externally),
    so trace decay is controlled by the head ``lamda`` parameter — when
    ``traces_on`` we use lamda=0.5 against gamma=0.0, but the
    ``MultiHeadMLPLearner`` per-head decay is the product gamma*lamda.
    To get a non-trivial trace, we need both > 0. So when traces_on we set
    the per-action demon ``lamda`` to a value paired with a non-zero gamma
    on the head; SARSAAgent doesn't expose this directly, so we construct
    the agent and override per-head gammas via prediction demons that share
    the same trace setup. For an honest CPU-throughput baseline we keep
    the standard SARSA setup and toggle ``lamda`` only — the work done per
    step is still representative of trace bookkeeping when lamda > 0.
    """
    config = SARSAConfig(
        n_actions=n_actions,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay_steps=0,
    )
    lamda = 0.5 if traces_on else 0.0

    return SARSAAgent(
        sarsa_config=config,
        hidden_sizes=hidden_sizes,
        step_size=0.01,
        sparsity=0.0,
        lamda=lamda,
    )


def _make_synthetic_data(
    n_steps: int,
    feature_dim: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generate synthetic (obs, rewards, terminated, next_obs) arrays."""
    k_obs, k_rew, k_term, k_next = jr.split(key, 4)
    observations = jr.normal(k_obs, (n_steps, feature_dim), dtype=jnp.float32)
    rewards = jr.normal(k_rew, (n_steps,), dtype=jnp.float32) * 0.1
    # Random terminations every ~100 steps on average
    terminated = (jr.uniform(k_term, (n_steps,)) < 0.01).astype(jnp.float32)
    next_observations = jr.normal(k_next, (n_steps, feature_dim), dtype=jnp.float32)
    return observations, rewards, terminated, next_observations


def _benchmark_one(
    n_actions: int,
    hidden_sizes: tuple[int, ...],
    traces_on: bool,
    n_steps: int,
    feature_dim: int,
    seed: int = 0,
) -> BenchResult:
    """Benchmark one SARSA configuration. Times JIT warmup and a hot run."""
    label = (
        f"n_actions={n_actions:3d} hidden={hidden_sizes!s:>10s} "
        f"traces={traces_on!s:5s}"
    )
    print(f"  Running {label} ...", flush=True)

    agent = _make_agent(n_actions, hidden_sizes, traces_on)

    key = jr.key(seed)
    k_init, k_data = jr.split(key, 2)

    observations, rewards, terminated, next_observations = _make_synthetic_data(
        n_steps, feature_dim, k_data
    )

    def _seeded_state() -> SARSAState:
        s = agent.init(feature_dim, k_init)
        return cast(
            SARSAState,
            s.replace(
                last_action=jnp.array(0, dtype=jnp.int32),
                last_observation=jnp.zeros(feature_dim, dtype=jnp.float32),
            ),
        )

    # JIT warmup: run at the same shapes as the hot run so the second pass
    # hits the compile cache.
    warmup_state = _seeded_state()
    t_warm0 = time.perf_counter()
    warm_state = run_sarsa_from_arrays_final_state(
        agent, warmup_state, observations, rewards, terminated, next_observations
    )
    jax.block_until_ready(warm_state)  # type: ignore[no-untyped-call]
    t_warm = time.perf_counter() - t_warm0

    # Hot run.
    state = _seeded_state()
    t0 = time.perf_counter()
    result_state = run_sarsa_from_arrays_final_state(
        agent, state, observations, rewards, terminated, next_observations
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
        n_actions=n_actions,
        hidden_sizes=hidden_sizes,
        traces_on=traces_on,
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
        itertools.product(N_ACTIONS_GRID, HIDDEN_SIZES_GRID, TRACES_GRID)
    )
    print(
        f"SARSA throughput benchmark: {len(configs)} configurations, "
        f"n_steps={n_steps}, feature_dim={feature_dim}",
        flush=True,
    )
    print(f"Devices: {jax.devices()}", flush=True)

    results: list[BenchResult] = []
    for i, (n_actions, hidden_sizes, traces_on) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", flush=True)
        try:
            result = _benchmark_one(
                n_actions=n_actions,
                hidden_sizes=hidden_sizes,
                traces_on=traces_on,
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
                    n_actions=n_actions,
                    hidden_sizes=hidden_sizes,
                    traces_on=traces_on,
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
    print("\n" + "=" * 88, flush=True)
    print("SARSA Throughput Results", flush=True)
    print("=" * 88, flush=True)
    header = (
        f"{'n_actions':>10} {'hidden':>10} {'traces':>6} "
        f"{'steps/sec':>12} {'total (s)':>10} {'warmup (s)':>11} {'status':>8}"
    )
    print(header, flush=True)
    print("-" * 88, flush=True)

    for r in results:
        status = "OK" if not r.error else "FAILED"
        sps = f"{r.steps_per_sec:>12.1f}" if r.error == "" else f"{'-':>12s}"
        total = f"{r.total_seconds:>10.2f}" if r.error == "" else f"{'-':>10s}"
        warmup = f"{r.jit_warmup_seconds:>11.2f}" if r.error == "" else f"{'-':>11s}"
        print(
            f"{r.n_actions:>10d} {r.hidden_sizes_str:>10s} {str(r.traces_on):>6s} "
            f"{sps} {total} {warmup} {status:>8s}",
            flush=True,
        )
    print("=" * 88, flush=True)

    # 1000 steps/sec target
    ok_results = [r for r in results if r.error == ""]
    if ok_results:
        passing = [r for r in ok_results if r.steps_per_sec >= 1000.0]
        failing = [r for r in ok_results if r.steps_per_sec < 1000.0]
        print(
            f"\n>= 1000 steps/sec target: {len(passing)}/{len(ok_results)} passing",
            flush=True,
        )
        if failing:
            print("  Below 1000 steps/sec:", flush=True)
            for r in failing:
                print(
                    f"    n_actions={r.n_actions} hidden={r.hidden_sizes_str} "
                    f"traces={r.traces_on}: {r.steps_per_sec:.1f} steps/sec",
                    flush=True,
                )


def save_results_csv(results: list[BenchResult], output_dir: Path) -> Path:
    """Save results to a timestamped CSV. Creates output_dir if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sarsa_throughput_{timestamp}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_actions",
                "hidden_sizes_str",
                "traces",
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
                    r.n_actions,
                    r.hidden_sizes_str,
                    r.traces_on,
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
        description="SARSA throughput benchmark (CPU steps/sec)"
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

    failures = [r for r in results if r.error]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

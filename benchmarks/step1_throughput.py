"""CPU throughput benchmark for Step 1 linear supervised learners.

Measures scan-loop and batched scan-loop steps/sec for the Step 1 learner
stack: ``LinearLearner`` plus public Step 1 optimizers and online normalizers.

Usage
-----
```bash
python benchmarks/step1_throughput.py
python benchmarks/step1_throughput.py --n-steps 20000 --feature-dim 20
python benchmarks/step1_throughput.py --output-dir outputs/step1_canonical/throughput
```

Output
------
Writes timestamped CSV, JSON, and Markdown files with one row per
optimizer-normalizer-mode configuration. ``warmup_seconds`` is the first run at
the benchmark shape and includes compile/dispatch warmup. ``run_seconds`` is a
second hot run from a fresh learner state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Force CPU so the daemon-readiness baseline is reproducible.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.random as jr  # noqa: E402

from alberta_framework import (  # noqa: E402
    IDBD,
    LMS,
    Autostep,
    EMANormalizer,
    LinearLearner,
    StreamingBatchNormalizer,
    WelfordNormalizer,
    run_learning_loop,
    run_learning_loop_batched,
)
from alberta_framework.core.baseline_optimizers import (  # noqa: E402
    NADALINE,
    Adam,
    RMSprop,
)
from alberta_framework.streams.alberta_plan_step1 import (  # noqa: E402
    AlbertaPlanStep1Stream,
)
from alberta_framework.utils.timing import format_duration  # noqa: E402

DEFAULT_FEATURE_DIM = 20
DEFAULT_N_STEPS = 20_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_DIR = Path("outputs/step1_canonical/throughput")

OPTIMIZER_FACTORIES: dict[str, Callable[[], Any]] = {
    "LMS": lambda: LMS(step_size=0.01),
    "IDBD": lambda: IDBD(initial_step_size=0.01, meta_step_size=0.01),
    "Autostep": lambda: Autostep(initial_step_size=0.01, meta_step_size=0.01),
    "Adam": lambda: Adam(step_size=0.001),
    "RMSprop": lambda: RMSprop(step_size=0.001),
    "NADALINE": lambda: NADALINE(step_size=0.01),
}

NORMALIZER_FACTORIES: dict[str, Callable[[], Any | None]] = {
    "none": lambda: None,
    "EMA": lambda: EMANormalizer(decay=0.99),
    "Welford": lambda: WelfordNormalizer(),
    "StreamingBatch": lambda: StreamingBatchNormalizer(momentum=0.99),
}


@dataclass
class BenchResult:
    optimizer: str
    normalizer: str
    mode: str
    steps_per_sec: float
    learner_updates_per_sec: float
    warmup_seconds: float
    run_seconds: float
    n_steps: int
    feature_dim: int
    batch_size: int
    stream: str
    device: str
    error: str = ""


def _make_stream(feature_dim: int) -> AlbertaPlanStep1Stream:
    return AlbertaPlanStep1Stream(
        feature_dim=feature_dim,
        num_relevant=min(5, feature_dim),
        drift_rate_w=0.001,
        drift_rate_b=0.001,
        noise_std=1.0,
        feature_std=1.0,
    )


def _block_until_ready(result: Any) -> None:
    if hasattr(result, "metrics"):
        jax.block_until_ready(result.metrics)  # type: ignore[no-untyped-call]
        return
    _, metrics = result
    jax.block_until_ready(metrics)  # type: ignore[no-untyped-call]


def _run_once(
    learner: LinearLearner,
    feature_dim: int,
    n_steps: int,
    seed: int,
    mode: str,
    batch_size: int,
) -> Any:
    stream = _make_stream(feature_dim)
    if mode == "scan":
        return run_learning_loop(learner, stream, n_steps, jr.key(seed))
    if mode == "batched":
        keys = jr.split(jr.key(seed), batch_size)
        return run_learning_loop_batched(learner, stream, n_steps, keys)
    raise ValueError(f"unknown mode: {mode}")


def _benchmark_one(
    optimizer_name: str,
    normalizer_name: str,
    mode: str,
    n_steps: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
) -> BenchResult:
    label = (
        f"optimizer={optimizer_name:9s} normalizer={normalizer_name:14s} "
        f"mode={mode:7s}"
    )
    print(f"  Running {label} ...", flush=True)

    learner = LinearLearner(
        optimizer=OPTIMIZER_FACTORIES[optimizer_name](),
        normalizer=NORMALIZER_FACTORIES[normalizer_name](),
    )

    t0 = time.perf_counter()
    warm_result = _run_once(learner, feature_dim, n_steps, seed, mode, batch_size)
    _block_until_ready(warm_result)
    warmup_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    hot_result = _run_once(learner, feature_dim, n_steps, seed, mode, batch_size)
    _block_until_ready(hot_result)
    run_seconds = time.perf_counter() - t1

    stream_steps_per_sec = n_steps / run_seconds if run_seconds > 0 else float("inf")
    updates = n_steps * (batch_size if mode == "batched" else 1)
    learner_updates_per_sec = updates / run_seconds if run_seconds > 0 else float("inf")

    print(
        f"    -> {learner_updates_per_sec:>10.1f} learner updates/sec "
        f"({stream_steps_per_sec:>10.1f} stream steps/sec, "
        f"run {format_duration(run_seconds)}, warmup {format_duration(warmup_seconds)})",
        flush=True,
    )

    return BenchResult(
        optimizer=optimizer_name,
        normalizer=normalizer_name,
        mode=mode,
        steps_per_sec=stream_steps_per_sec,
        learner_updates_per_sec=learner_updates_per_sec,
        warmup_seconds=warmup_seconds,
        run_seconds=run_seconds,
        n_steps=n_steps,
        feature_dim=feature_dim,
        batch_size=batch_size if mode == "batched" else 1,
        stream="AlbertaPlanStep1Stream",
        device=str(jax.devices()[0]),
    )


def run_all_benchmarks(
    n_steps: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
    optimizers: list[str] | None = None,
    normalizers: list[str] | None = None,
    modes: list[str] | None = None,
) -> list[BenchResult]:
    """Run the Step 1 throughput grid. Failures become result rows."""
    optimizers = optimizers or list(OPTIMIZER_FACTORIES)
    normalizers = normalizers or list(NORMALIZER_FACTORIES)
    modes = modes or ["scan", "batched"]
    configs = [(o, n, m) for o in optimizers for n in normalizers for m in modes]

    print(
        f"Step 1 throughput benchmark: {len(configs)} configurations, "
        f"n_steps={n_steps}, feature_dim={feature_dim}, batch_size={batch_size}",
        flush=True,
    )
    print(f"Devices: {jax.devices()}", flush=True)

    results: list[BenchResult] = []
    for idx, (optimizer_name, normalizer_name, mode) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}]", flush=True)
        try:
            results.append(
                _benchmark_one(
                    optimizer_name=optimizer_name,
                    normalizer_name=normalizer_name,
                    mode=mode,
                    n_steps=n_steps,
                    feature_dim=feature_dim,
                    batch_size=batch_size,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {err_msg}", flush=True)
            traceback.print_exc()
            results.append(
                BenchResult(
                    optimizer=optimizer_name,
                    normalizer=normalizer_name,
                    mode=mode,
                    steps_per_sec=float("nan"),
                    learner_updates_per_sec=float("nan"),
                    warmup_seconds=float("nan"),
                    run_seconds=float("nan"),
                    n_steps=n_steps,
                    feature_dim=feature_dim,
                    batch_size=batch_size if mode == "batched" else 1,
                    stream="AlbertaPlanStep1Stream",
                    device=str(jax.devices()[0]),
                    error=err_msg,
                )
            )
    return results


def print_results_table(results: list[BenchResult]) -> None:
    print("\n" + "=" * 116, flush=True)
    print("Step 1 Throughput Results", flush=True)
    print("=" * 116, flush=True)
    print(
        f"{'optimizer':>9} {'normalizer':>14} {'mode':>7} {'updates/sec':>13} "
        f"{'steps/sec':>12} {'run (s)':>9} {'warmup (s)':>10} {'status':>8}",
        flush=True,
    )
    print("-" * 116, flush=True)
    for r in results:
        status = "OK" if not r.error else "FAILED"
        updates = f"{r.learner_updates_per_sec:>13.1f}" if not r.error else f"{'-':>13s}"
        steps = f"{r.steps_per_sec:>12.1f}" if not r.error else f"{'-':>12s}"
        run = f"{r.run_seconds:>9.3f}" if not r.error else f"{'-':>9s}"
        warm = f"{r.warmup_seconds:>10.3f}" if not r.error else f"{'-':>10s}"
        print(
            f"{r.optimizer:>9s} {r.normalizer:>14s} {r.mode:>7s} "
            f"{updates} {steps} {run} {warm} {status:>8s}",
            flush=True,
        )
    print("=" * 116, flush=True)

    ok_results = [r for r in results if not r.error]
    if ok_results:
        passing = [r for r in ok_results if r.learner_updates_per_sec >= 1000.0]
        print(
            f"\n>= 1000 learner updates/sec target: "
            f"{len(passing)}/{len(ok_results)} passing",
            flush=True,
        )


def _rows(results: list[BenchResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def save_results_csv(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"step1_throughput_{timestamp}.csv"
    rows = _rows(results)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {csv_path}", flush=True)
    return csv_path


def save_results_json(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"step1_throughput_{timestamp}.json"
    payload = {
        "benchmark": "step1_throughput",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_learner_updates_per_sec": 1000.0,
        "results": _rows(results),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"JSON saved to: {json_path}", flush=True)
    return json_path


def save_results_markdown(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"step1_throughput_{timestamp}.md"
    lines = [
        "# Step 1 CPU Throughput",
        "",
        "| Optimizer | Normalizer | Mode | Learner updates/sec | "
        "Stream steps/sec | Warmup s | Run s | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        status = "OK" if not r.error else f"FAILED: {r.error}"
        lines.append(
            f"| {r.optimizer} | {r.normalizer} | {r.mode} | "
            f"{r.learner_updates_per_sec:.1f} | {r.steps_per_sec:.1f} | "
            f"{r.warmup_seconds:.6f} | {r.run_seconds:.6f} | {status} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Markdown saved to: {md_path}", flush=True)
    return md_path


def save_all_outputs(results: list[BenchResult], output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        save_results_csv(results, output_dir),
        save_results_json(results, output_dir),
        save_results_markdown(results, output_dir),
    )


def _parse_csv_list(value: str | None, allowed: set[str]) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown values {unknown}; allowed: {sorted(allowed)}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1 CPU throughput benchmark")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated optimizer subset",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default=None,
        help="Comma-separated normalizer subset",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated mode subset: scan,batched",
    )
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args(argv)

    try:
        optimizers = _parse_csv_list(args.optimizers, set(OPTIMIZER_FACTORIES))
        normalizers = _parse_csv_list(args.normalizers, set(NORMALIZER_FACTORIES))
        modes = _parse_csv_list(args.modes, {"scan", "batched"})
    except ValueError as exc:
        parser.error(str(exc))

    results = run_all_benchmarks(
        n_steps=args.n_steps,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        optimizers=optimizers,
        normalizers=normalizers,
        modes=modes,
    )
    print_results_table(results)

    if not args.no_output:
        save_all_outputs(results, args.output_dir)

    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

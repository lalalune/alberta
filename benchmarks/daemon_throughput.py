"""End-to-end daemon throughput benchmark.

Closes the "daemon end-to-end throughput open" boundary identified in
``docs/research/step3_results.md`` (DoD-10). Local-core JAX scan
benchmarks (``horde_throughput.py``, ``sarsa_throughput.py``) clear the
>=1000 steps/sec gate, but they bypass the per-step daemon overhead:
JSON parse, single-step predict + update (one round-trip per step, not a
fused scan), JSON serialize, and periodic checkpointing.

This benchmark drives the production :class:`AlbertaPipeline` through a
synthetic JSON-line stream simulating a daemon transport. The input is
generated in-memory rather than read from real ``stdin`` so the benchmark
is deterministic and reproducible. Each phase is timed independently and
the slowest phase is reported, so the bottleneck is identifiable.

Configuration grid (8 cells):

* ``features`` ∈ {identity, full Step 2 features}
* ``n_demons`` ∈ {5, 25}
* ``hidden_sizes`` ∈ {(), (32,)}

Acceptance threshold: >=500 steps/sec end-to-end (relaxed from local-core's
1000 because of overhead).

Usage
-----
```bash
python benchmarks/daemon_throughput.py
python benchmarks/daemon_throughput.py --n-steps 5000
python benchmarks/daemon_throughput.py --output-dir outputs/daemon_throughput
```
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import sys
import tempfile
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

from alberta_framework import save_checkpoint  # noqa: E402
from alberta_framework.pipeline import (  # noqa: E402
    AlbertaPipeline,
    AlbertaPipelineConfig,
    Step2FeatureConfig,
)
from alberta_framework.steps import Step3HordeConfig, Step4SARSAConfig  # noqa: E402
from alberta_framework.utils.timing import format_duration  # noqa: E402

# =============================================================================
# Config
# =============================================================================

DEFAULT_OBSERVATION_DIM = 8
DEFAULT_N_STEPS = 5_000
DEFAULT_OUTPUT_DIR = Path("outputs/daemon_throughput")
DEFAULT_CHECKPOINT_INTERVAL = 100

FEATURES_GRID = ["no_features", "features"]
N_DEMONS_GRID = [5, 25]
HIDDEN_SIZES_GRID: list[tuple[int, ...]] = [(), (32,)]


@dataclass
class BenchResult:
    """Per-configuration timing and throughput record."""

    features: str
    n_demons: int
    hidden_sizes: tuple[int, ...]
    n_steps: int
    feature_dim: int
    # Aggregate
    total_seconds: float
    steps_per_sec: float
    # Per-phase total seconds
    parse_total_s: float
    predict_total_s: float
    update_total_s: float
    serialize_total_s: float
    checkpoint_total_s: float
    # Per-phase mean ms
    parse_mean_ms: float
    predict_mean_ms: float
    update_mean_ms: float
    serialize_mean_ms: float
    checkpoint_mean_ms: float
    # Counts
    n_checkpoints: int
    # JIT warmup
    warmup_seconds: float
    error: str = ""

    @property
    def hidden_sizes_str(self) -> str:
        if not self.hidden_sizes:
            return "()"
        return "x".join(str(h) for h in self.hidden_sizes)

    def slowest_phase(self) -> tuple[str, float]:
        """Return (phase_name, total_seconds) for the slowest phase."""
        phases = {
            "parse": self.parse_total_s,
            "predict": self.predict_total_s,
            "update": self.update_total_s,
            "serialize": self.serialize_total_s,
            "checkpoint": self.checkpoint_total_s,
        }
        name = max(phases, key=lambda k: phases[k])
        return name, phases[name]

    def to_csv_row(self) -> list[object]:
        slowest_name, slowest_s = self.slowest_phase()
        return [
            self.features,
            self.n_demons,
            self.hidden_sizes_str,
            self.n_steps,
            self.feature_dim,
            f"{self.steps_per_sec:.4f}",
            f"{self.total_seconds:.6f}",
            f"{self.parse_total_s:.6f}",
            f"{self.predict_total_s:.6f}",
            f"{self.update_total_s:.6f}",
            f"{self.serialize_total_s:.6f}",
            f"{self.checkpoint_total_s:.6f}",
            f"{self.parse_mean_ms:.6f}",
            f"{self.predict_mean_ms:.6f}",
            f"{self.update_mean_ms:.6f}",
            f"{self.serialize_mean_ms:.6f}",
            f"{self.checkpoint_mean_ms:.6f}",
            self.n_checkpoints,
            f"{self.warmup_seconds:.6f}",
            slowest_name,
            f"{slowest_s:.6f}",
            self.error,
        ]


# =============================================================================
# Benchmark machinery
# =============================================================================


def _make_pipeline_config(
    features_mode: str,
    n_demons: int,
    hidden_sizes: tuple[int, ...],
    *,
    observation_dim: int,
) -> AlbertaPipelineConfig:
    """Build an AlbertaPipelineConfig for the given grid cell."""
    if features_mode == "no_features":
        feature_cfg = Step2FeatureConfig.identity(observation_dim=observation_dim)
    elif features_mode == "features":
        feature_cfg = Step2FeatureConfig(
            observation_dim=observation_dim,
            include_raw=True,
            include_ema=True,
            include_delta=True,
            include_phase_products=False,
            ema_decay=0.95,
            periods=(32.0, 64.0),
        )
    else:
        raise ValueError(f"unknown features_mode {features_mode!r}")

    # Spread gammas/lamdas over n_demons.
    gammas = tuple(min(0.99, 0.5 + 0.5 * (i / max(n_demons - 1, 1))) for i in range(n_demons))
    lamdas = tuple(0.5 for _ in range(n_demons))

    horde_cfg = Step3HordeConfig(
        gammas=gammas,
        lamdas=lamdas,
        hidden_sizes=hidden_sizes,
        step_size=0.05,
        use_obgd=True,
        obgd_kappa=2.0,
        normalizer="none",
        sparsity=0.0,
        use_layer_norm=True,
        trace_mode="accumulating",
    )
    control_cfg = Step4SARSAConfig(
        n_actions=4,
        hidden_sizes=hidden_sizes,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay_steps=0,
        lamda=0.0,
        optimizer="lms",
        bounder="obgd",
        step_size=0.03,
        meta_step_size=0.01,
        bounder_kappa=0.5,
        sparsity=0.0,
        use_layer_norm=True,
        trace_mode="accumulating",
    )
    return AlbertaPipelineConfig(
        features=feature_cfg,
        horde=horde_cfg,
        control=control_cfg,
    )


def _make_synthetic_jsonlines(
    n_steps: int,
    observation_dim: int,
    n_demons: int,
    seed: int,
) -> list[bytes]:
    """Generate ``n_steps`` JSON-line byte messages.

    Each line is a UTF-8 encoded JSON object representing a single
    transition message a daemon would receive on stdin.
    """
    rng = jr.key(seed)
    obs_key, rew_key, term_key, cum_key = jr.split(rng, 4)
    observations = jr.normal(obs_key, (n_steps, observation_dim), dtype=jnp.float32)
    rewards = jr.normal(rew_key, (n_steps,), dtype=jnp.float32) * 0.1
    terminated = (jr.uniform(term_key, (n_steps,)) < 0.01).astype(jnp.int32)
    cumulants = jr.normal(cum_key, (n_steps, n_demons), dtype=jnp.float32) * 0.1

    obs_list = observations.tolist()
    rew_list = rewards.tolist()
    term_list = terminated.tolist()
    cum_list = cumulants.tolist()

    lines: list[bytes] = []
    for i in range(n_steps):
        msg = {
            "step": i,
            "observation": obs_list[i],
            "reward": rew_list[i],
            "terminated": bool(term_list[i]),
            "cumulants": cum_list[i],
        }
        lines.append((json.dumps(msg) + "\n").encode("utf-8"))
    return lines


def _benchmark_one(
    features_mode: str,
    n_demons: int,
    hidden_sizes: tuple[int, ...],
    n_steps: int,
    observation_dim: int,
    checkpoint_interval: int,
    seed: int,
) -> BenchResult:
    """Benchmark one daemon configuration end-to-end."""
    label = (
        f"features={features_mode:>11s} n_demons={n_demons:3d} "
        f"hidden={str(hidden_sizes):>10s}"
    )
    print(f"  Running {label} ...", flush=True)

    config = _make_pipeline_config(
        features_mode,
        n_demons,
        hidden_sizes,
        observation_dim=observation_dim,
    )
    pipeline = AlbertaPipeline(config)

    # Build the synthetic JSON-line input pipe in memory.
    lines = _make_synthetic_jsonlines(n_steps, observation_dim, n_demons, seed)
    pipe = io.BytesIO(b"".join(lines))

    # Initialize pipeline state from a primer observation.
    init_key = jr.key(seed + 1)
    primer_obs = jnp.zeros(observation_dim, dtype=jnp.float32)
    state = pipeline.init(init_key, primer_obs)

    # JIT warmup: run a single predict + update so subsequent calls hit the
    # compile cache. This warmup is counted separately, not in the hot run.
    warmup_obs = jnp.asarray([0.0] * observation_dim, dtype=jnp.float32)
    warmup_reward = jnp.asarray(0.0, dtype=jnp.float32)
    warmup_term = jnp.asarray(0.0, dtype=jnp.float32)
    warmup_cum = jnp.zeros(n_demons, dtype=jnp.float32)
    t_warm0 = time.perf_counter()
    _, _ = pipeline.predict(state)
    warmup_result = pipeline.update(
        state,
        warmup_obs,
        warmup_reward,
        warmup_term,
        warmup_cum,
    )
    jax.block_until_ready(warmup_result.q_values)  # type: ignore[no-untyped-call]
    t_warm = time.perf_counter() - t_warm0

    # Re-init so warmup state is discarded and the state observed in the hot
    # loop matches a freshly-started daemon.
    state = pipeline.init(init_key, primer_obs)

    # Phase accumulators
    parse_total = 0.0
    predict_total = 0.0
    update_total = 0.0
    serialize_total = 0.0
    checkpoint_total = 0.0
    n_checkpoints = 0

    # Use a single tempdir for all checkpoints so the per-checkpoint cost is
    # measured consistently. Different paths per checkpoint avoid cross-step
    # contention on the same directory tree.
    with tempfile.TemporaryDirectory(prefix="daemon_bench_ckpt_") as ckpt_dir:
        ckpt_root = Path(ckpt_dir)

        out_buffer = io.BytesIO()  # simulated stdout sink

        # Tight daemon loop. Reset pipe to the start so we read every line.
        pipe.seek(0)
        t0 = time.perf_counter()
        for step_idx in range(n_steps):
            # ----- parse -----
            t_phase = time.perf_counter()
            line = pipe.readline()
            if not line:
                raise RuntimeError(
                    f"input pipe exhausted at step {step_idx}/{n_steps}"
                )
            msg = json.loads(line)
            obs = jnp.asarray(msg["observation"], dtype=jnp.float32)
            reward = jnp.asarray(msg["reward"], dtype=jnp.float32)
            terminated = jnp.asarray(
                1.0 if msg["terminated"] else 0.0, dtype=jnp.float32
            )
            cumulants = jnp.asarray(msg["cumulants"], dtype=jnp.float32)
            parse_total += time.perf_counter() - t_phase

            # ----- predict -----
            t_phase = time.perf_counter()
            horde_predictions, q_values = pipeline.predict(state)
            jax.block_until_ready(q_values)  # type: ignore[no-untyped-call]
            predict_total += time.perf_counter() - t_phase

            # ----- update -----
            t_phase = time.perf_counter()
            result = pipeline.update(state, obs, reward, terminated, cumulants)
            jax.block_until_ready(result.q_values)  # type: ignore[no-untyped-call]
            state = result.state
            update_total += time.perf_counter() - t_phase

            # ----- serialize -----
            t_phase = time.perf_counter()
            action = int(result.action)
            response = {
                "step": step_idx,
                "action": action,
                "q_values": [float(q) for q in q_values],
                "horde_predictions": [float(p) for p in horde_predictions],
                "control_td_error": float(result.control_td_error),
            }
            out_buffer.write((json.dumps(response) + "\n").encode("utf-8"))
            serialize_total += time.perf_counter() - t_phase

            # ----- checkpoint (every checkpoint_interval steps) -----
            if (step_idx + 1) % checkpoint_interval == 0:
                t_phase = time.perf_counter()
                ckpt_path = ckpt_root / f"ckpt_{step_idx + 1:08d}"
                save_checkpoint(
                    state,
                    ckpt_path,
                    metadata={"step": step_idx + 1},
                )
                checkpoint_total += time.perf_counter() - t_phase
                n_checkpoints += 1

        total = time.perf_counter() - t0

    steps_per_sec = n_steps / total if total > 0 else float("inf")

    print(
        f"    -> {steps_per_sec:>10.1f} steps/sec  "
        f"(total {format_duration(total)}, warmup {format_duration(t_warm)})",
        flush=True,
    )

    return BenchResult(
        features=features_mode,
        n_demons=n_demons,
        hidden_sizes=hidden_sizes,
        n_steps=n_steps,
        feature_dim=config.feature_dim(),
        total_seconds=total,
        steps_per_sec=steps_per_sec,
        parse_total_s=parse_total,
        predict_total_s=predict_total,
        update_total_s=update_total,
        serialize_total_s=serialize_total,
        checkpoint_total_s=checkpoint_total,
        parse_mean_ms=parse_total / n_steps * 1000.0,
        predict_mean_ms=predict_total / n_steps * 1000.0,
        update_mean_ms=update_total / n_steps * 1000.0,
        serialize_mean_ms=serialize_total / n_steps * 1000.0,
        checkpoint_mean_ms=(
            checkpoint_total / n_checkpoints * 1000.0 if n_checkpoints > 0 else 0.0
        ),
        n_checkpoints=n_checkpoints,
        warmup_seconds=t_warm,
    )


def run_all_benchmarks(
    n_steps: int,
    observation_dim: int,
    checkpoint_interval: int,
    seed: int,
) -> list[BenchResult]:
    """Run the full configuration grid. Failures do not stop the sweep."""
    configs = list(itertools.product(FEATURES_GRID, N_DEMONS_GRID, HIDDEN_SIZES_GRID))
    print(
        f"Daemon throughput benchmark: {len(configs)} configurations, "
        f"n_steps={n_steps}, observation_dim={observation_dim}, "
        f"checkpoint every {checkpoint_interval} steps",
        flush=True,
    )
    print(f"Devices: {jax.devices()}", flush=True)

    results: list[BenchResult] = []
    for i, (features_mode, n_demons, hidden_sizes) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", flush=True)
        try:
            result = _benchmark_one(
                features_mode=features_mode,
                n_demons=n_demons,
                hidden_sizes=hidden_sizes,
                n_steps=n_steps,
                observation_dim=observation_dim,
                checkpoint_interval=checkpoint_interval,
                seed=seed,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {err_msg}", flush=True)
            traceback.print_exc()
            results.append(
                BenchResult(
                    features=features_mode,
                    n_demons=n_demons,
                    hidden_sizes=hidden_sizes,
                    n_steps=n_steps,
                    feature_dim=-1,
                    total_seconds=float("nan"),
                    steps_per_sec=float("nan"),
                    parse_total_s=float("nan"),
                    predict_total_s=float("nan"),
                    update_total_s=float("nan"),
                    serialize_total_s=float("nan"),
                    checkpoint_total_s=float("nan"),
                    parse_mean_ms=float("nan"),
                    predict_mean_ms=float("nan"),
                    update_mean_ms=float("nan"),
                    serialize_mean_ms=float("nan"),
                    checkpoint_mean_ms=float("nan"),
                    n_checkpoints=0,
                    warmup_seconds=float("nan"),
                    error=err_msg,
                )
            )
    return results


# =============================================================================
# Output
# =============================================================================


def print_results_table(results: list[BenchResult], threshold: float) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 116, flush=True)
    print("Daemon End-to-End Throughput Results", flush=True)
    print("=" * 116, flush=True)
    header = (
        f"{'features':>11} {'demons':>6} {'hidden':>8} {'feat_dim':>8} "
        f"{'steps/sec':>10} {'parse_ms':>9} {'pred_ms':>8} {'upd_ms':>8} "
        f"{'ser_ms':>8} {'ckpt_ms':>9} {'slowest':>10} {'status':>8}"
    )
    print(header, flush=True)
    print("-" * 116, flush=True)
    for r in results:
        status = "OK" if not r.error else "FAILED"
        if r.error:
            print(
                f"{r.features:>11s} {r.n_demons:>6d} {r.hidden_sizes_str:>8s} "
                f"{'-':>8s} {'-':>10s} {'-':>9s} {'-':>8s} {'-':>8s} "
                f"{'-':>8s} {'-':>9s} {'-':>10s} {status:>8s}",
                flush=True,
            )
            continue
        slowest_name, _ = r.slowest_phase()
        print(
            f"{r.features:>11s} {r.n_demons:>6d} {r.hidden_sizes_str:>8s} "
            f"{r.feature_dim:>8d} {r.steps_per_sec:>10.1f} {r.parse_mean_ms:>9.3f} "
            f"{r.predict_mean_ms:>8.3f} {r.update_mean_ms:>8.3f} "
            f"{r.serialize_mean_ms:>8.3f} {r.checkpoint_mean_ms:>9.3f} "
            f"{slowest_name:>10s} {status:>8s}",
            flush=True,
        )
    print("=" * 116, flush=True)

    ok = [r for r in results if r.error == ""]
    if ok:
        passing = [r for r in ok if r.steps_per_sec >= threshold]
        failing = [r for r in ok if r.steps_per_sec < threshold]
        print(
            f"\nDaemon E2E gate (>= {threshold:.0f} steps/sec): "
            f"{len(passing)}/{len(ok)} passing",
            flush=True,
        )
        if failing:
            print("  Below threshold:", flush=True)
            for r in failing:
                slowest_name, slowest_s = r.slowest_phase()
                print(
                    f"    features={r.features} n_demons={r.n_demons} "
                    f"hidden={r.hidden_sizes_str}: {r.steps_per_sec:.1f} steps/sec "
                    f"(slowest phase: {slowest_name}, {slowest_s:.2f}s)",
                    flush=True,
                )


def save_results_csv(results: list[BenchResult], output_dir: Path) -> Path:
    """Save the timing grid as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "daemon_throughput_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "features",
                "n_demons",
                "hidden_sizes",
                "n_steps",
                "feature_dim",
                "steps_per_sec",
                "total_seconds",
                "parse_total_s",
                "predict_total_s",
                "update_total_s",
                "serialize_total_s",
                "checkpoint_total_s",
                "parse_mean_ms",
                "predict_mean_ms",
                "update_mean_ms",
                "serialize_mean_ms",
                "checkpoint_mean_ms",
                "n_checkpoints",
                "warmup_seconds",
                "slowest_phase",
                "slowest_phase_total_s",
                "error",
            ]
        )
        for r in results:
            writer.writerow(r.to_csv_row())
    print(f"\nResults saved to: {csv_path}", flush=True)
    return csv_path


def save_results_json(
    results: list[BenchResult],
    output_dir: Path,
    threshold: float,
) -> Path:
    """Save the timing grid as a structured JSON document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"

    ok = [r for r in results if r.error == ""]
    passing = [r for r in ok if r.steps_per_sec >= threshold]
    failing = [r for r in ok if r.steps_per_sec < threshold]
    if ok:
        slowest_phase_counts: dict[str, int] = {}
        for r in ok:
            slowest_name, _ = r.slowest_phase()
            slowest_phase_counts[slowest_name] = (
                slowest_phase_counts.get(slowest_name, 0) + 1
            )
        most_common_slowest_phase = max(
            slowest_phase_counts, key=lambda k: slowest_phase_counts[k]
        )
        slowest_overall = min(ok, key=lambda r: r.steps_per_sec)
    else:
        slowest_phase_counts = {}
        most_common_slowest_phase = ""
        slowest_overall = None

    body = {
        "generated_at": datetime.now().isoformat(),
        "threshold_steps_per_sec": threshold,
        "n_configurations": len(results),
        "n_passing": len(passing),
        "n_failing": len(failing),
        "n_errored": len(results) - len(ok),
        "slowest_phase_counts": slowest_phase_counts,
        "most_common_slowest_phase": most_common_slowest_phase,
        "slowest_configuration": (
            {
                "features": slowest_overall.features,
                "n_demons": slowest_overall.n_demons,
                "hidden_sizes": list(slowest_overall.hidden_sizes),
                "steps_per_sec": slowest_overall.steps_per_sec,
                "slowest_phase": slowest_overall.slowest_phase()[0],
            }
            if slowest_overall is not None
            else None
        ),
        "results": [
            {
                "features": r.features,
                "n_demons": r.n_demons,
                "hidden_sizes": list(r.hidden_sizes),
                "feature_dim": r.feature_dim,
                "n_steps": r.n_steps,
                "steps_per_sec": r.steps_per_sec,
                "total_seconds": r.total_seconds,
                "warmup_seconds": r.warmup_seconds,
                "phase_total_seconds": {
                    "parse": r.parse_total_s,
                    "predict": r.predict_total_s,
                    "update": r.update_total_s,
                    "serialize": r.serialize_total_s,
                    "checkpoint": r.checkpoint_total_s,
                },
                "phase_mean_ms": {
                    "parse": r.parse_mean_ms,
                    "predict": r.predict_mean_ms,
                    "update": r.update_mean_ms,
                    "serialize": r.serialize_mean_ms,
                    "checkpoint": r.checkpoint_mean_ms,
                },
                "n_checkpoints": r.n_checkpoints,
                "slowest_phase": r.slowest_phase()[0],
                "passes_gate": r.steps_per_sec >= threshold,
                "error": r.error,
            }
            for r in results
        ],
    }
    with json_path.open("w") as f:
        json.dump(body, f, indent=2)
    print(f"Summary saved to: {json_path}", flush=True)
    return json_path


def save_results_markdown(
    results: list[BenchResult],
    output_dir: Path,
    threshold: float,
) -> Path:
    """Save a human-readable markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "SUMMARY.md"

    ok = [r for r in results if r.error == ""]
    passing = [r for r in ok if r.steps_per_sec >= threshold]
    failing = [r for r in ok if r.steps_per_sec < threshold]
    slowest_phase_counts: dict[str, int] = {}
    for r in ok:
        slowest_name, _ = r.slowest_phase()
        slowest_phase_counts[slowest_name] = (
            slowest_phase_counts.get(slowest_name, 0) + 1
        )
    if slowest_phase_counts:
        most_common_slowest = max(slowest_phase_counts, key=lambda k: slowest_phase_counts[k])
    else:
        most_common_slowest = "n/a"

    lines: list[str] = []
    lines.append("# Daemon End-to-End Throughput\n")
    lines.append(
        "Closes the daemon end-to-end throughput boundary identified in "
        "DoD-10. Measures the production `AlbertaPipeline` driven by a "
        "synthetic JSON-line transport with parse, predict, update, "
        "serialize, and periodic checkpoint phases.\n"
    )
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Acceptance threshold: >={threshold:.0f} steps/sec.\n")
    lines.append("")
    lines.append("## Results\n")
    lines.append(
        "| features | n_demons | hidden | feat_dim | steps/sec | parse ms | "
        "predict ms | update ms | serialize ms | ckpt ms | slowest | gate |"
    )
    lines.append(
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )
    for r in results:
        if r.error:
            lines.append(
                f"| {r.features} | {r.n_demons} | {r.hidden_sizes_str} | - | "
                f"FAILED | - | - | - | - | - | - | FAIL |"
            )
            continue
        slowest_name, _ = r.slowest_phase()
        gate = "PASS" if r.steps_per_sec >= threshold else "FAIL"
        lines.append(
            f"| {r.features} | {r.n_demons} | {r.hidden_sizes_str} | "
            f"{r.feature_dim} | {r.steps_per_sec:.1f} | {r.parse_mean_ms:.3f} | "
            f"{r.predict_mean_ms:.3f} | {r.update_mean_ms:.3f} | "
            f"{r.serialize_mean_ms:.3f} | {r.checkpoint_mean_ms:.3f} | "
            f"{slowest_name} | {gate} |"
        )
    lines.append("")
    lines.append(
        f"**Gate summary**: {len(passing)}/{len(ok)} configurations clear the "
        f">={threshold:.0f} steps/sec target."
    )
    lines.append("")
    lines.append(f"**Most common slowest phase**: `{most_common_slowest}`.")
    lines.append("")
    if ok:
        slowest_overall = min(ok, key=lambda r: r.steps_per_sec)
        slowest_overall_phase, _ = slowest_overall.slowest_phase()
        lines.append(
            f"**Slowest configuration**: "
            f"features={slowest_overall.features}, "
            f"n_demons={slowest_overall.n_demons}, "
            f"hidden_sizes={slowest_overall.hidden_sizes_str} at "
            f"{slowest_overall.steps_per_sec:.1f} steps/sec; bottleneck "
            f"phase `{slowest_overall_phase}`."
        )
    lines.append("")
    if failing:
        lines.append("## Failing configurations\n")
        for r in failing:
            slowest_name, slowest_s = r.slowest_phase()
            lines.append(
                f"- features={r.features}, n_demons={r.n_demons}, "
                f"hidden_sizes={r.hidden_sizes_str}: {r.steps_per_sec:.1f} "
                f"steps/sec, slowest phase `{slowest_name}` ({slowest_s:.2f}s "
                f"across {r.n_steps} steps)."
            )
    lines.append("")

    with md_path.open("w") as f:
        f.write("\n".join(lines))
    print(f"Markdown summary saved to: {md_path}", flush=True)
    return md_path


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Daemon end-to-end throughput benchmark"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Steps per configuration (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--observation-dim",
        type=int,
        default=DEFAULT_OBSERVATION_DIM,
        help=f"Raw observation dimension (default: {DEFAULT_OBSERVATION_DIM})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=(
            "Save a checkpoint every N steps "
            f"(default: {DEFAULT_CHECKPOINT_INTERVAL})"
        ),
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
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=500.0,
        help="Acceptance threshold in steps/sec (default: 500)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing CSV/JSON/Markdown output",
    )
    args = parser.parse_args(argv)

    results = run_all_benchmarks(
        n_steps=args.n_steps,
        observation_dim=args.observation_dim,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    print_results_table(results, threshold=args.threshold)

    if not args.no_write:
        save_results_csv(results, args.output_dir)
        save_results_json(results, args.output_dir, threshold=args.threshold)
        save_results_markdown(results, args.output_dir, threshold=args.threshold)

    failures = [r for r in results if r.error]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

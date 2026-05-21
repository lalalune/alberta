#!/usr/bin/env python3
"""Compact sieve/universal-approximation probe for target-structure UPGD.

This benchmark is evidence-oriented, not a proof.  It asks whether a single
target-structure UPGD learner improves as a plain capacity budget grows on
families that are friendly to universal approximation arguments: low-degree
polynomials, Fourier functions, radial bumps, sparse interactions, and a
piecewise threshold stressor.  The comparison is deliberately simple: UPGD
versus a capacity-matched MLP trained on the same prequential stream.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner  # noqa: E402
from alberta_framework.core.optimizers import ObGDBounding  # noqa: E402
from alberta_framework.core.upgd import UPGDLearner  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("output/step2_universal_sieve_probe")
FAMILY_NAMES = (
    "polynomial",
    "fourier",
    "radial_bumps",
    "sparse_interactions",
    "piecewise_threshold",
)


@dataclass(frozen=True)
class ProbeConfig:
    """Experimental scale and learner hyperparameters."""

    steps: int = 2_000
    n_seeds: int = 5
    final_window: int = 500
    feature_dim: int = 6
    capacities: tuple[int, ...] = (8, 16, 32, 64)
    step_size: float = 0.03
    sparsity: float = 0.5
    obgd_kappa: float = 0.5
    noise_std: float = 0.01
    material_improvement_ratio: float = 0.10
    monotone_tolerance_ratio: float = 0.03


@dataclass(frozen=True)
class MethodSpec:
    """A capacity-scaled learner specification."""

    name: str
    learner_type: str
    capacity: int


def stable_key(seed: int, *parts: str) -> jax.Array:
    """Build a deterministic JAX key from a seed and labels."""
    checksum = 0
    for part in parts:
        checksum = zlib.crc32(part.encode("utf-8"), checksum)
    return jr.fold_in(jr.key(seed), checksum & 0x7FFFFFFF)


def make_inputs(seed: int, steps: int, feature_dim: int) -> np.ndarray:
    """Sample the shared online input stream."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(steps, feature_dim)).astype(np.float32)


def raw_targets(family: str, x: np.ndarray) -> np.ndarray:
    """Evaluate one synthetic target family on a batch of inputs."""
    if family == "polynomial":
        y = (
            0.80 * x[:, 0]
            - 0.55 * x[:, 1] ** 2
            + 0.45 * x[:, 2] * x[:, 3]
            + 0.30 * x[:, 4] ** 3
        )
    elif family == "fourier":
        y = (
            np.sin(math.pi * x[:, 0])
            + 0.50 * np.cos(2.0 * math.pi * x[:, 1])
            + 0.35 * np.sin(math.pi * (x[:, 2] + x[:, 3]))
        )
    elif family == "radial_bumps":
        centers = np.asarray(
            [
                [-0.55, -0.30, 0.10, 0.45, 0.00, 0.20],
                [0.35, 0.50, -0.45, -0.15, 0.30, -0.25],
                [0.05, -0.65, 0.55, 0.15, -0.50, 0.40],
            ],
            dtype=np.float32,
        )
        widths = np.asarray([0.32, 0.42, 0.36], dtype=np.float32)
        weights = np.asarray([1.00, -0.75, 0.60], dtype=np.float32)
        sq_dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        y = np.sum(weights[None, :] * np.exp(-sq_dist / (2.0 * widths[None, :] ** 2)), axis=1)
    elif family == "sparse_interactions":
        y = (
            0.90 * x[:, 0] * x[:, 2]
            - 0.65 * x[:, 1] * x[:, 3] * x[:, 4]
            + 0.35 * x[:, 5]
            + 0.25 * np.sin(math.pi * x[:, 0] * x[:, 4])
        )
    elif family == "piecewise_threshold":
        gate = (x[:, 0] + 0.35 * x[:, 1] - 0.20 * x[:, 2]) > 0.0
        left = -0.75 + 0.45 * x[:, 3] - 0.35 * x[:, 4] ** 2
        right = 0.70 + 0.50 * np.sin(math.pi * x[:, 1]) + 0.25 * x[:, 5]
        y = np.where(gate, right, left)
    else:
        raise ValueError(f"unknown target family {family!r}")
    return np.asarray(y, dtype=np.float32)


def make_stream(
    family: str,
    seed: int,
    steps: int,
    feature_dim: int,
    noise_std: float,
) -> tuple[jax.Array, jax.Array]:
    """Create normalized observations and scalar targets for one run."""
    x = make_inputs(seed, steps, feature_dim)
    y = raw_targets(family, x)
    y = (y - np.mean(y)) / max(float(np.std(y)), 1e-6)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed + 10_000)
        y = y + rng.normal(0.0, noise_std, size=y.shape).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y[:, None].astype(np.float32))


def make_method(spec: MethodSpec, config: ProbeConfig) -> Any:
    """Construct one learner from a method specification."""
    hidden_sizes = (spec.capacity,)
    if spec.learner_type == "upgd":
        learner = UPGDLearner.step2_default(
            n_heads=1,
            hidden_sizes=hidden_sizes,
            loss_normalization="target_structure",
        )
        if config.step_size == 0.03 and config.sparsity == 0.5 and config.obgd_kappa == 0.5:
            return learner
        return UPGDLearner(
            n_heads=1,
            hidden_sizes=hidden_sizes,
            step_size=config.step_size,
            bounder=ObGDBounding(kappa=config.obgd_kappa),
            sparsity=config.sparsity,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            utility_decay=0.995,
            perturbation_beta=2.0,
            perturbation_interval=16,
            loss_normalization="target_structure",
            track_unit_utilities=False,
            track_gradient_history=False,
        )
    if spec.learner_type == "mlp":
        return MultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=hidden_sizes,
            step_size=config.step_size,
            bounder=ObGDBounding(kappa=config.obgd_kappa),
            sparsity=config.sparsity,
            use_layer_norm=True,
        )
    raise ValueError(f"unknown learner type {spec.learner_type!r}")


def prequential_mse(
    learner: Any,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
) -> np.ndarray:
    """Run online supervised learning and return the per-step MSE curve."""
    state = learner.init(observations.shape[1], key)

    def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        obs, target = inputs
        result = learner.update(carry, obs, target)
        mse = jnp.mean((result.predictions - target) ** 2)
        return result.state, mse

    _, curve = jax.lax.scan(step_fn, state, (observations, targets))
    curve.block_until_ready()
    return np.asarray(curve, dtype=np.float32)


def method_specs(capacities: tuple[int, ...]) -> tuple[MethodSpec, ...]:
    """Return UPGD and capacity-matched MLP method specs."""
    specs: list[MethodSpec] = []
    for capacity in capacities:
        specs.append(MethodSpec(f"upgd_h{capacity}", "upgd", capacity))
        specs.append(MethodSpec(f"mlp_h{capacity}", "mlp", capacity))
    return tuple(specs)


def summarize_capacity_trend(
    rows: list[dict[str, Any]],
    capacities: tuple[int, ...],
    config: ProbeConfig,
) -> dict[str, Any]:
    """Assess monotone or material improvement as capacity grows."""
    by_capacity: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_capacity[int(row["capacity"])].append(float(row["final_window_mse"]))
    means = {capacity: float(np.mean(by_capacity[capacity])) for capacity in capacities}
    ordered = [means[capacity] for capacity in capacities]
    tolerance = config.monotone_tolerance_ratio
    monotone_with_tolerance = all(
        nxt <= prev * (1.0 + tolerance) for prev, nxt in zip(ordered, ordered[1:])
    )
    first = ordered[0]
    best = min(ordered)
    material_improvement = (first - best) / max(first, 1e-12)
    return {
        "capacity_mean_final_window_mse": means,
        "best_capacity": capacities[int(np.argmin(np.asarray(ordered)))],
        "monotone_with_tolerance": bool(monotone_with_tolerance),
        "material_improvement_ratio": float(material_improvement),
        "materially_improves": bool(material_improvement >= config.material_improvement_ratio),
    }


def aggregate_records(records: list[dict[str, Any]], config: ProbeConfig) -> dict[str, Any]:
    """Aggregate records by family, learner type, and capacity trend."""
    aggregate: dict[str, Any] = {}
    for family in FAMILY_NAMES:
        aggregate[family] = {}
        for learner_type in ("upgd", "mlp"):
            rows = [
                r
                for r in records
                if r["family"] == family and r["learner_type"] == learner_type
            ]
            trend = summarize_capacity_trend(rows, config.capacities, config)
            aggregate[family][learner_type] = trend
    upgd_support_count = sum(
        int(
            aggregate[family]["upgd"]["monotone_with_tolerance"]
            or aggregate[family]["upgd"]["materially_improves"]
        )
        for family in FAMILY_NAMES
    )
    aggregate["support_summary"] = {
        "upgd_supporting_families": upgd_support_count,
        "total_families": len(FAMILY_NAMES),
        "supports_universal_representation_learning_assumptions": (
            upgd_support_count >= 4
        ),
    }
    return aggregate


def format_table(aggregate: dict[str, Any], capacities: tuple[int, ...]) -> str:
    """Format the capacity trend table for Markdown and stdout."""
    header = "| family | learner | " + " | ".join(f"h={c}" for c in capacities)
    header += " | monotone/tol | material gain |"
    sep = "|" + "---|" * (len(capacities) + 4)
    lines = [header, sep]
    for family in FAMILY_NAMES:
        for learner_type in ("upgd", "mlp"):
            row = aggregate[family][learner_type]
            means = row["capacity_mean_final_window_mse"]
            values = " | ".join(f"{means[c]:.4f}" for c in capacities)
            lines.append(
                f"| {family} | {learner_type} | {values} | "
                f"{row['monotone_with_tolerance']} | "
                f"{100.0 * row['material_improvement_ratio']:.1f}% |"
            )
    return "\n".join(lines)


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    """Run the full probe and return serializable results."""
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    specs = method_specs(config.capacities)
    for family in FAMILY_NAMES:
        for seed in range(config.n_seeds):
            observations, targets = make_stream(
                family,
                seed,
                config.steps,
                config.feature_dim,
                config.noise_std,
            )
            for spec in specs:
                learner = make_method(spec, config)
                key = stable_key(seed, family, spec.name)
                curve = prequential_mse(learner, key, observations, targets)
                window = min(config.final_window, curve.shape[0])
                records.append(
                    {
                        "family": family,
                        "seed": seed,
                        "method": spec.name,
                        "learner_type": spec.learner_type,
                        "capacity": spec.capacity,
                        "online_mean_mse": float(np.mean(curve)),
                        "final_window_mse": float(np.mean(curve[-window:])),
                    }
                )
    aggregate = aggregate_records(records, config)
    return {
        "config": asdict(config),
        "records": records,
        "aggregate": aggregate,
        "result_table": format_table(aggregate, config.capacities),
        "runtime_s": time.perf_counter() - start,
    }


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    """Write JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    support = results["aggregate"]["support_summary"]
    verdict = (
        "supports the representation-learning assumptions as empirical evidence"
        if support["supports_universal_representation_learning_assumptions"]
        else "does not support the assumptions at this scale"
    )
    summary = "\n".join(
        [
            "# Step 2 Universal Sieve Probe",
            "",
            results["result_table"],
            "",
            "## Interpretation",
            "",
            (
                "This is evidence about whether the proof assumptions look "
                "plausible under a compact online benchmark; it is not a proof."
            ),
            "",
            (
                f"UPGD showed monotone-with-tolerance or material capacity "
                f"improvement on {support['upgd_supporting_families']} of "
                f"{support['total_families']} families, so this run {verdict}."
            ),
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=ProbeConfig.steps)
    parser.add_argument("--n-seeds", type=int, default=ProbeConfig.n_seeds)
    parser.add_argument("--final-window", type=int, default=ProbeConfig.final_window)
    parser.add_argument("--capacities", type=str, default="8,16,32,64")
    parser.add_argument("--noise-std", type=float, default=ProbeConfig.noise_std)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ProbeConfig:
    """Create a config from parsed CLI arguments."""
    capacities = tuple(int(part) for part in args.capacities.split(",") if part)
    if args.smoke:
        return ProbeConfig(
            steps=600,
            n_seeds=2,
            final_window=150,
            capacities=(8, 16, 32),
            noise_std=args.noise_std,
        )
    if args.moderate:
        return ProbeConfig(
            steps=2_500,
            n_seeds=5,
            final_window=600,
            capacities=(8, 16, 32, 64),
            noise_std=args.noise_std,
        )
    return ProbeConfig(
        steps=args.steps,
        n_seeds=args.n_seeds,
        final_window=args.final_window,
        capacities=capacities,
        noise_std=args.noise_std,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the probe from the command line."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    results = run_probe(config)
    write_outputs(results, args.output_dir)
    print(results["result_table"])
    print(json.dumps(results["aggregate"]["support_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

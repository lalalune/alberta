#!/usr/bin/env python3
"""Compact falsification probes for the Step 2 representation-learning claim.

The runner is intentionally small: each scenario is a stream-level counterexample
candidate, not a tuned benchmark.  Positive results here do not prove the theory;
negative rows identify assumptions that the theory must state explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import dataclass
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

from alberta_framework import (  # noqa: E402
    MultiHeadMLPLearner,
    ObGDBounding,
    TemporalContextConfig,
    TemporalContextFeaturizer,
    UPGDLearner,
    transform_temporal_context_arrays,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_theory_falsification")
DEFAULT_NOTE_PATH = DEFAULT_OUTPUT_DIR / "SUMMARY.md"
SCENARIOS: tuple[str, ...] = (
    "delayed_parity",
    "hidden_context_aliasing",
    "adversarial_nonstationary_oos",
    "rotating_relevant_subspace",
    "class_blocked_discontinuous_shift",
    "sparse_rare_feature_utility",
)


@dataclass(frozen=True)
class StreamCase:
    """Generated online supervised stream."""

    name: str
    observations: np.ndarray
    targets: np.ndarray
    limitation: str
    hidden_assumption: str
    oracle_floor: float | None = None


@dataclass(frozen=True)
class MethodSpec:
    """Small learner specification used by the falsification runner."""

    name: str
    kind: str
    hidden_sizes: tuple[int, ...]


def stable_key(seed: int, text: str) -> jax.Array:
    """Return a reproducible PRNG key folded by text."""
    checksum = zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF
    return jr.fold_in(jr.key(seed), checksum)


def _rng(seed: int, scenario: str) -> np.random.Generator:
    return np.random.default_rng(seed + (zlib.crc32(scenario.encode("utf-8")) & 0xFFFF))


def _standardize(x: np.ndarray) -> np.ndarray:
    scale = np.std(x, axis=0, keepdims=True) + 1e-6
    standardized: np.ndarray = (
        (x - np.mean(x, axis=0, keepdims=True)) / scale
    ).astype(np.float32)
    return standardized


def generate_stream(scenario: str, steps: int, seed: int) -> StreamCase:
    """Generate one adversarial Step 2 stream."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = _rng(seed, scenario)

    if scenario == "delayed_parity":
        x = _standardize(rng.normal(size=(steps, 8)))
        delay = 5
        signs = x[:, :4] > 0.0
        parity = np.logical_xor.reduce(signs, axis=1)
        delayed = np.roll(parity, delay)
        delayed[:delay] = False
        y = np.where(delayed, 1.0, -1.0).reshape(-1, 1).astype(np.float32)
        return StreamCase(
            scenario,
            x,
            y,
            (
                "The target is a delayed high-order parity of past observations, "
                "not a smooth current-state function."
            ),
            (
                "Inputs include the needed finite history and the generator can "
                "express parity-like compositions."
            ),
            float(np.mean((y[:, 0] - 0.0) ** 2)),
        )

    if scenario == "hidden_context_aliasing":
        x = _standardize(rng.normal(size=(steps, 6)))
        hidden_context = np.where((np.arange(steps) // 8) % 2 == 0, 1.0, -1.0)
        y = (hidden_context * np.sign(x[:, 0])).reshape(-1, 1).astype(np.float32)
        floor = float(np.mean((y[:, 0] - 0.0) ** 2))
        return StreamCase(
            scenario,
            x,
            y,
            "The same observation requires incompatible targets.",
            "The observation contains enough state to make the target function Markov.",
            floor,
        )

    if scenario == "adversarial_nonstationary_oos":
        x = _standardize(rng.normal(size=(steps, 10)))
        y = np.zeros((steps, 1), dtype=np.float32)
        thirds = np.array_split(np.arange(steps), 3)
        if len(thirds[0]):
            idx = thirds[0]
            y[idx, 0] = np.sin(11.0 * x[idx, 0] + 7.0 * x[idx, 1])
        if len(thirds[1]):
            idx = thirds[1]
            bits = x[idx, :6] > 0.0
            y[idx, 0] = np.where(np.logical_xor.reduce(bits, axis=1), 1.0, -1.0)
        if len(thirds[2]):
            idx = thirds[2]
            y[idx, 0] = np.sign((x[idx, 2] * x[idx, 3]) - (x[idx, 4] * x[idx, 5]))
        return StreamCase(
            scenario,
            x,
            y,
            (
                "The stream switches among high-frequency, parity, and discontinuous "
                "targets outside a fixed smooth dictionary."
            ),
            (
                "The generator family is closed over the target sequence and drift is "
                "slow enough for bounded adaptation."
            ),
        )

    if scenario == "rotating_relevant_subspace":
        x = _standardize(rng.normal(size=(steps, 12)))
        theta = np.arange(steps, dtype=np.float32) * 0.08
        first_axis = np.cos(theta)[:, None] * x[:, :4]
        second_axis = np.sin(theta)[:, None] * x[:, 4:8]
        moving_projection = np.sum(first_axis + second_axis, axis=1) / 2.0
        y = np.tanh(moving_projection).reshape(-1, 1).astype(np.float32)
        return StreamCase(
            scenario,
            x,
            y,
            (
                "The relevant subspace rotates continuously, so stale utility can "
                "point at yesterday's features."
            ),
            (
                "The utility/adaptation time scale is faster than subspace drift or "
                "time/context is observable."
            ),
        )

    if scenario == "class_blocked_discontinuous_shift":
        x = _standardize(rng.normal(size=(steps, 10)))
        y = np.zeros((steps, 4), dtype=np.float32)
        base = (x[:, 0] > 0.0).astype(int) + 2 * (x[:, 1] > 0.0).astype(int)
        block_len = max(steps // 4, 1)
        permutation_bank = np.asarray(
            [
                [0, 1, 2, 3],
                [2, 3, 1, 0],
                [1, 0, 3, 2],
                [3, 2, 0, 1],
            ],
            dtype=np.int32,
        )
        block = np.minimum(np.arange(steps) // block_len, 3)
        cls = permutation_bank[block, base]
        y[np.arange(steps), cls] = 1.0
        return StreamCase(
            scenario,
            x,
            y,
            "Class meanings change in discontinuous blocks without a context variable.",
            (
                "Task identity or block context is observable, or evaluation only asks "
                "for fast tracking after each jump."
            ),
        )

    x = _standardize(rng.normal(size=(steps, 12)))
    y = np.zeros((steps, 4), dtype=np.float32)
    early = np.tanh(2.0 * x[:, 0])
    y[:, 0] = early
    rare = rng.random(steps) < 0.04
    late = np.arange(steps) > int(0.55 * steps)
    event = rare & late
    y[event, :] = 0.0
    rare_bit = (x[event, 9] * x[event, 10] + 0.5 * x[event, 11]) > 0.0
    rare_head = np.where(rare_bit, 2, 3)
    y[np.flatnonzero(event), rare_head] = 6.0
    return StreamCase(
        scenario,
        x,
        y,
        "A dormant rare feature has no early utility, then dominates rare high-value events.",
        (
            "Useful rare features receive sufficient excitation or the learner "
            "preserves option value for dormant features."
        ),
    )


def active_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Return MSE over non-NaN targets."""
    active = ~jnp.isnan(targets)
    safe_targets = jnp.where(active, targets, 0.0)
    squared = jnp.where(active, (predictions - safe_targets) ** 2, 0.0)
    return jnp.sum(squared) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)


CONTEXT_PERIODS_DEFAULT = (50.0, 100.0, 200.0)
CONTEXT_PERIODS_DENSE = (
    32.0,
    40.0,
    48.0,
    56.0,
    64.0,
    72.0,
    80.0,
    88.0,
    96.0,
    112.0,
    128.0,
    160.0,
    192.0,
)


def make_learner(spec: MethodSpec, n_heads: int) -> Any:
    """Construct one falsification learner."""
    if spec.kind.startswith("upgd_context") or spec.kind == "upgd":
        return UPGDLearner.step2_default(n_heads=n_heads, hidden_sizes=spec.hidden_sizes)
    if spec.kind == "upgd_fastslow":
        return UPGDLearner(
            n_heads=n_heads,
            hidden_sizes=spec.hidden_sizes,
            step_size=0.03,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=0.5,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            utility_decay=0.995,
            perturbation_beta=2.0,
            perturbation_interval=16,
            loss_normalization="target_structure",
            track_unit_utilities=False,
            track_gradient_history=False,
            adaptive_kappa_mode="loss_ratio",
            adaptive_kappa_base=0.5,
            adaptive_kappa_min=0.2,
            adaptive_kappa_max=1.0,
            adaptive_kappa_exponent=0.7,
            head_loss_pressure_gate_ratio=1.05,
            head_loss_pressure_multiplier=1.0,
            head_loss_pressure_warmup_steps=50,
        )
    if spec.kind == "mlp":
        return MultiHeadMLPLearner(
            n_heads=n_heads,
            hidden_sizes=spec.hidden_sizes,
            step_size=0.03,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=0.5,
            use_layer_norm=True,
        )
    raise ValueError(f"unknown method kind {spec.kind!r}")


def run_method(spec: MethodSpec, stream: StreamCase, seed: int) -> dict[str, float]:
    """Run one learner online and return aggregate prequential losses."""
    obs = jnp.asarray(stream.observations)
    tgt = jnp.asarray(stream.targets)
    if spec.kind.startswith("upgd_context"):
        dense = spec.kind in {"upgd_context_dense", "upgd_context_phase_only"}
        phase_only = spec.kind == "upgd_context_phase_only"
        featurizer = TemporalContextFeaturizer(
            TemporalContextConfig(
                input_dim=stream.observations.shape[1],
                include_raw=True,
                include_ema=not phase_only,
                include_delta=not phase_only,
                ema_decay=0.96,
                include_phase_products=True,
                periods=CONTEXT_PERIODS_DENSE if dense else CONTEXT_PERIODS_DEFAULT,
            )
        )
        _context_state, obs = transform_temporal_context_arrays(featurizer, obs)
    learner = make_learner(spec, n_heads=stream.targets.shape[1])
    state = learner.init(obs.shape[1], stable_key(seed, spec.name))

    def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        observation, target = inputs
        result = learner.update(carry, observation, target)
        return result.state, active_mse(result.predictions, target)

    started = time.time()
    _, losses = jax.lax.scan(step_fn, state, (obs, tgt))
    losses.block_until_ready()
    arr = np.asarray(losses)
    return {
        "mean_mse": float(np.mean(arr)),
        "final_window_mse": float(np.mean(arr[-max(1, len(arr) // 4) :])),
        "elapsed_s": time.time() - started,
    }


def method_specs(names: str, upgd_width: int, mlp_width: int) -> tuple[MethodSpec, ...]:
    """Parse comma-separated method names."""
    catalog = {
        "upgd": MethodSpec("target_structure_upgd", "upgd", (upgd_width,)),
        "upgd_context": MethodSpec(
            "target_structure_upgd_context",
            "upgd_context",
            (upgd_width,),
        ),
        "upgd_context_dense": MethodSpec(
            "target_structure_upgd_context_dense",
            "upgd_context_dense",
            (upgd_width,),
        ),
        "upgd_context_phase_only": MethodSpec(
            "target_structure_upgd_context_phase_only",
            "upgd_context_phase_only",
            (upgd_width,),
        ),
        "upgd_fastslow": MethodSpec(
            "target_structure_upgd_fastslow",
            "upgd_fastslow",
            (upgd_width,),
        ),
        "mlp": MethodSpec(f"mlp{mlp_width}", "mlp", (mlp_width,)),
        "mlp_deep": MethodSpec(f"mlp{mlp_width}_{mlp_width}", "mlp", (mlp_width, mlp_width)),
    }
    selected = []
    for name in names.split(","):
        key = name.strip()
        if key:
            if key not in catalog:
                raise ValueError(f"unknown method {key!r}; expected one of {sorted(catalog)}")
            selected.append(catalog[key])
    return tuple(selected)


def expand_scenarios(value: str) -> tuple[str, ...]:
    """Expand a scenario selector."""
    if value == "all":
        return SCENARIOS
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return selected


def paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize UPGD versus the best non-UPGD method."""
    by_scenario: dict[str, list[float]] = {name: [] for name in SCENARIOS}
    for record in records:
        methods = record["methods"]
        upgd = methods["target_structure_upgd"]["final_window_mse"]
        baselines = [
            metrics["final_window_mse"]
            for name, metrics in methods.items()
            if name != "target_structure_upgd"
        ]
        if baselines:
            by_scenario[record["scenario"]].append(min(baselines) - upgd)

    summary = {}
    for scenario, diffs in by_scenario.items():
        if not diffs:
            continue
        arr = np.asarray(diffs, dtype=np.float64)
        summary[scenario] = {
            "diff_positive_favors_upgd": float(np.mean(arr)),
            "wins": int(np.sum(arr > 0.0)),
            "losses": int(np.sum(arr < 0.0)),
            "ties": int(np.sum(arr == 0.0)),
        }
    return summary


def run_suite(
    scenarios: tuple[str, ...],
    methods: tuple[MethodSpec, ...],
    steps: int,
    seeds: int,
) -> dict[str, Any]:
    """Run the requested falsification suite."""
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in range(seeds):
            stream = generate_stream(scenario, steps=steps, seed=seed)
            methods_payload = {
                spec.name: run_method(spec, stream, seed=seed) for spec in methods
            }
            records.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "limitation": stream.limitation,
                    "hidden_assumption": stream.hidden_assumption,
                    "oracle_floor": stream.oracle_floor,
                    "methods": methods_payload,
                }
            )
    return {"records": records, "summary": paired_summary(records)}


def write_note(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown note for the run."""
    lines = [
        "# Step 2 Theory Falsification Results",
        "",
        "This file is generated by `step2_theory_falsification.py`. It records",
        "counterexample probes for target-structure UPGD; failures are evidence",
        "that the theory needs narrower assumptions, not proof of a better learner.",
        "",
        "## Run Summary",
        "",
        "| Scenario | UPGD diff vs best baseline | Wins/losses/ties | Hidden assumption |",
        "|---|---:|---:|---|",
    ]
    records_by_scenario = {record["scenario"]: record for record in payload["records"]}
    for scenario, row in payload["summary"].items():
        record = records_by_scenario[scenario]
        lines.append(
            f"| `{scenario}` | {row['diff_positive_favors_upgd']:+.6f} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | "
            f"{record['hidden_assumption']} |"
        )
    lines.extend(["", "## Raw JSON", "", "See the sibling JSON artifact for per-seed losses."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--methods", default="upgd,mlp,mlp_deep")
    parser.add_argument("--upgd-width", type=int, default=32)
    parser.add_argument("--mlp-width", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the CLI."""
    args = parse_args()
    scenarios = expand_scenarios(args.scenarios)
    methods = method_specs(args.methods, args.upgd_width, args.mlp_width)
    payload = run_suite(scenarios, methods, steps=args.steps, seeds=args.n_seeds)
    payload["config"] = {
        "scenarios": scenarios,
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "methods": [spec.name for spec in methods],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_note(args.note_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

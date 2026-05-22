#!/usr/bin/env python3
"""D26: packaged UPGD-memory learner on D18 digit stressors.

This runner checks whether the production UPGD-memory learner transfers from
compact OPMNIST to the D18 digit regimes that exposed retention and mask-noise
weaknesses.  It compares same-run fair MLP baselines against a small set of
UPGD-memory variants.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import d18_simple_universal_resource_basis as d18  # noqa: E402

from alberta_framework.core.upgd_memory import (  # noqa: E402
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    UPGDMemoryState,
    run_upgd_memory_arrays,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d26_upgd_memory_digit_stress")
DEFAULT_RESULT_PREFIX = "d26_upgd_memory_digit_stress"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d26_upgd_memory_digit_stress.md"
)


@dataclass(frozen=True)
class UPGDMemoryVariant:
    """One packaged UPGD-memory variant."""

    name: str
    hidden_sizes: tuple[int, ...] = (64,)
    upgd_step_size: float = 0.03
    slots_per_class: int = 20
    memory_update_rate: float = 0.3
    memory_bandwidth: float = 0.01
    initial_memory_logit: float = -2.0
    memory_logit_step_size: float = 0.25
    confidence_logit_scale: float = 2.0
    reliability_logit_scale: float = 8.0
    target_trace_blend_scale: float = 0.0
    target_trace_pressure_threshold: float = 0.75
    target_allocation_rate: float = 0.18
    upgd_head_step_size_multiplier: float = 1.0
    upgd_head_bias_step_size_multiplier: float = 1.0
    upgd_head_loss_pressure_gate_ratio: float = 0.0
    upgd_head_loss_pressure_multiplier: float = 0.0
    upgd_head_loss_pressure_warmup_steps: int = 0
    upgd_head_repetition_multiplier: float = 0.0
    upgd_head_repetition_decay: float = 0.9
    upgd_head_repetition_delta_threshold: float = 0.05
    upgd_head_repetition_pressure_threshold: float = 0.0
    upgd_head_repetition_warmup_steps: int = 0


def core_variants() -> tuple[UPGDMemoryVariant, ...]:
    """Return the small historical D26 variant set."""
    return (
        UPGDMemoryVariant("upgdmem_s20_alloc18"),
        UPGDMemoryVariant("upgdmem_s10_alloc18", slots_per_class=10),
        UPGDMemoryVariant("upgdmem_s20_alloc18_mem0", initial_memory_logit=0.0),
    )


def head_plasticity_variants() -> tuple[UPGDMemoryVariant, ...]:
    """Return variants that test output-head plasticity."""
    return (
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_head2",
            upgd_head_step_size_multiplier=2.0,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_bias4",
            upgd_head_bias_step_size_multiplier=4.0,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_rep2",
            upgd_head_repetition_multiplier=2.0,
            upgd_head_repetition_pressure_threshold=0.25,
            upgd_head_repetition_warmup_steps=20,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_rep2",
            initial_memory_logit=0.0,
            upgd_head_repetition_multiplier=2.0,
            upgd_head_repetition_pressure_threshold=0.25,
            upgd_head_repetition_warmup_steps=20,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_loss2",
            upgd_head_loss_pressure_gate_ratio=1.2,
            upgd_head_loss_pressure_multiplier=2.0,
            upgd_head_loss_pressure_warmup_steps=50,
        ),
    )


def scale_plasticity_variants() -> tuple[UPGDMemoryVariant, ...]:
    """Return variants that test UPGD width/depth and base step-size."""
    return (
        UPGDMemoryVariant("upgdmem_h128_s20_alloc18", hidden_sizes=(128,)),
        UPGDMemoryVariant("upgdmem_h64_64_s20_alloc18", hidden_sizes=(64, 64)),
        UPGDMemoryVariant("upgdmem_s20_alloc18_eta06", upgd_step_size=0.06),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_eta06",
            initial_memory_logit=0.0,
            upgd_step_size=0.06,
        ),
        UPGDMemoryVariant(
            "upgdmem_h128_s20_alloc18_mem0",
            hidden_sizes=(128,),
            initial_memory_logit=0.0,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_eta06_rep2",
            initial_memory_logit=0.0,
            upgd_step_size=0.06,
            upgd_head_repetition_multiplier=2.0,
            upgd_head_repetition_pressure_threshold=0.25,
            upgd_head_repetition_warmup_steps=20,
        ),
    )


def memory_variants() -> tuple[UPGDMemoryVariant, ...]:
    """Return variants that test prototype allocation and matching sharpness."""
    return (
        UPGDMemoryVariant(
            "upgdmem_s20_alloc40_mem0",
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_mem0",
            slots_per_class=50,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc80_mem0",
            slots_per_class=50,
            initial_memory_logit=0.0,
            target_allocation_rate=0.80,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_bw005_mem0",
            slots_per_class=50,
            memory_bandwidth=0.005,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_bw02_mem0",
            slots_per_class=50,
            memory_bandwidth=0.02,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_mu05_mem0",
            slots_per_class=50,
            memory_update_rate=0.5,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
        ),
    )


def trace_variants() -> tuple[UPGDMemoryVariant, ...]:
    """Return variants that test causal previous-target trace blending."""
    return (
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace20",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.20,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace35",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.35,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace50",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.50,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace80",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.80,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_mem0_trace20",
            slots_per_class=50,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
            target_trace_blend_scale=0.20,
        ),
        UPGDMemoryVariant(
            "upgdmem_s50_alloc40_mem0_trace35",
            slots_per_class=50,
            initial_memory_logit=0.0,
            target_allocation_rate=0.40,
            target_trace_blend_scale=0.35,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace35_thr50",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.35,
            target_trace_pressure_threshold=0.50,
        ),
        UPGDMemoryVariant(
            "upgdmem_s20_alloc18_mem0_trace80_thr50",
            initial_memory_logit=0.0,
            target_trace_blend_scale=0.80,
            target_trace_pressure_threshold=0.50,
        ),
    )


def default_variants(variant_set: str = "all") -> tuple[UPGDMemoryVariant, ...]:
    """Return the requested compact UPGD-memory sweep set."""
    groups = {
        "core": core_variants(),
        "head": (*core_variants(), *head_plasticity_variants()),
        "scale": (*core_variants(), *scale_plasticity_variants()),
        "memory": (*core_variants(), *memory_variants()),
        "trace": (*core_variants(), *trace_variants()),
        "all": (
            *core_variants(),
            *head_plasticity_variants(),
            *scale_plasticity_variants(),
            *memory_variants(),
            *trace_variants(),
        ),
    }
    return groups[variant_set]


def make_upgd_memory(
    variant: UPGDMemoryVariant,
    feature_dim: int,
    n_heads: int,
) -> UPGDMemoryLearner:
    """Construct one UPGD-memory variant."""
    return UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=feature_dim,
            n_heads=n_heads,
            hidden_sizes=variant.hidden_sizes,
            upgd_step_size=variant.upgd_step_size,
            slots_per_class=variant.slots_per_class,
            memory_update_rate=variant.memory_update_rate,
            memory_bandwidth=variant.memory_bandwidth,
            initial_memory_logit=variant.initial_memory_logit,
            memory_logit_step_size=variant.memory_logit_step_size,
            confidence_logit_scale=variant.confidence_logit_scale,
            reliability_logit_scale=variant.reliability_logit_scale,
            target_trace_blend_scale=variant.target_trace_blend_scale,
            target_trace_pressure_threshold=variant.target_trace_pressure_threshold,
            target_allocation_rate=variant.target_allocation_rate,
            upgd_head_step_size_multiplier=(
                variant.upgd_head_step_size_multiplier
            ),
            upgd_head_bias_step_size_multiplier=(
                variant.upgd_head_bias_step_size_multiplier
            ),
            upgd_head_loss_pressure_gate_ratio=(
                variant.upgd_head_loss_pressure_gate_ratio
            ),
            upgd_head_loss_pressure_multiplier=(
                variant.upgd_head_loss_pressure_multiplier
            ),
            upgd_head_loss_pressure_warmup_steps=(
                variant.upgd_head_loss_pressure_warmup_steps
            ),
            upgd_head_repetition_multiplier=(
                variant.upgd_head_repetition_multiplier
            ),
            upgd_head_repetition_decay=variant.upgd_head_repetition_decay,
            upgd_head_repetition_delta_threshold=(
                variant.upgd_head_repetition_delta_threshold
            ),
            upgd_head_repetition_pressure_threshold=(
                variant.upgd_head_repetition_pressure_threshold
            ),
            upgd_head_repetition_warmup_steps=(
                variant.upgd_head_repetition_warmup_steps
            ),
        )
    )


def summarize_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    final_window: int,
) -> dict[str, float]:
    """Summarize prequential classification predictions."""
    losses = np.mean((predictions - targets) ** 2, axis=1)
    correct = np.argmax(predictions, axis=1) == labels
    window = min(final_window, predictions.shape[0])
    return {
        "online_mean_mse": float(np.mean(losses)),
        "final_window_mse": float(np.mean(losses[-window:])),
        "online_mean_accuracy": float(np.mean(correct)),
        "final_window_accuracy": float(np.mean(correct[-window:])),
    }


def evaluate_upgd_memory_classifier(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate final UPGD-memory classifier on held-out digits."""
    targets = np.eye(d18.N_DIGIT_CLASSES, dtype=np.float32)[y_test]
    predictions = jax.vmap(lambda obs: learner.predict(state, obs))(
        jnp.asarray(x_test, dtype=jnp.float32)
    )
    predictions.block_until_ready()
    preds = np.asarray(predictions)
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def run_upgd_memory_stream(
    learner: UPGDMemoryLearner,
    observations: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
) -> tuple[UPGDMemoryState, np.ndarray, np.ndarray, float]:
    """Run UPGD-memory with one JIT scan."""
    state = learner.init(key)
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)

    @jax.jit  # type: ignore[untyped-decorator]
    def run(initial_state: UPGDMemoryState):
        result = run_upgd_memory_arrays(
            learner,
            initial_state,
            observations_jax,
            targets_jax,
        )
        return result.state, result.predictions, result.metrics

    t0 = time.time()
    final_state, predictions, metrics = run(state)
    predictions.block_until_ready()
    return final_state, np.asarray(predictions), np.asarray(metrics), time.time() - t0


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    variants: tuple[UPGDMemoryVariant, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all baselines and UPGD-memory variants for one seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = d18.make_dataset(
        dataset_name,
        seed,
        args,
    )
    assert labels is not None
    assert x_test is not None
    assert y_test is not None
    methods: dict[str, dict[str, float]] = {}

    for method in d18.MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp = d18.make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = d18.run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 30_000 + d18.MLP_METHODS.index(method)),
        )
        methods[method] = d18.summarize_prequential(
            metrics,
            args.final_window,
            labels,
        )
        methods[method].update(d18.evaluate_mlp_classifier(mlp, state, x_test, y_test))
        methods[method]["runtime_s"] = float(time.time() - t0)

    for offset, variant in enumerate(variants):
        print(f"{dataset_name} seed={seed}: running {variant.name}")
        learner = make_upgd_memory(
            variant,
            feature_dim=int(observations.shape[1]),
            n_heads=int(targets.shape[1]),
        )
        state, predictions, metrics, runtime = run_upgd_memory_stream(
            learner,
            observations,
            targets,
            jr.key(seed + 91_000 + offset),
        )
        methods[variant.name] = summarize_predictions(
            predictions,
            targets,
            labels,
            args.final_window,
        )
        methods[variant.name].update(
            evaluate_upgd_memory_classifier(learner, state, x_test, y_test)
        )
        methods[variant.name].update(
            {
                "runtime_s": float(runtime),
                "active_prototypes": float(metrics[-1, 7]),
                "mean_gate": float(np.mean(metrics[:, 3])),
            }
        )

    return (
        {
            "dataset_name": dataset_name,
            "seed": int(seed),
            "methods": methods,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.6f} +/- {row[metric]['stderr']:.6f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary."""
    cfg = results["config"]
    lines = [
        "# D26 UPGD-Memory Digit Stress",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}."
        ),
        "",
        (
            "| Dataset | Method | Final MSE | Final Acc | Test MSE | Test Acc | "
            "Protos | Gate | Runtime s |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, aggregate in results["aggregate"].items():
        for method, row in aggregate.items():
            if method == "comparisons":
                continue
            lines.append(
                f"| `{dataset}` | `{method}` | "
                f"{metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_mse')} | "
                f"{metric_cell(row, 'test_accuracy')} | "
                f"{metric_cell(row, 'active_prototypes')} | "
                f"{metric_cell(row, 'mean_gate')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.extend(["", f"## {dataset} Comparisons", ""])
        comparisons = aggregate["comparisons"]
        for metric in (
            "final_window_mse",
            "final_window_accuracy",
            "test_mse",
            "test_accuracy",
        ):
            if metric not in comparisons:
                continue
            best = comparisons[metric]["best_kernel_vs_best_mlp"]
            lines.append(
                f"- `{metric}`: diff="
                f"{best['paired_diff_mean_positive_favors_kernel']:+.6f}, "
                f"wins={best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}, "
                f"best={best['best_kernel_counts']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="digits_mask_noise,digits_class_blocked")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=150)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--phase-length", type=int, default=200)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.1)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--variant-set",
        choices=("core", "head", "scale", "memory", "trace", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < args.mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if args.mask_noise_std < 0.0:
        raise ValueError("--mask-noise-std must be non-negative")
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")
    if not 0.0 <= args.mlp_sparsity < 1.0:
        raise ValueError("--mlp-sparsity must be in [0, 1)")


def main() -> None:
    """Run D26."""
    args = parse_args()
    validate_args(args)
    datasets = d18.expand_dataset_names(args.datasets)
    unknown = sorted(set(datasets) - set(d18.DIGITS_REGIMES))
    if unknown:
        raise ValueError(f"D26 only supports digit regimes; got {unknown}")
    variants = default_variants(args.variant_set)
    candidate_methods = tuple(variant.name for variant in variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, meta = run_one_dataset_seed(dataset_name, seed, variants, args)
            records.append(record)
            datasets_meta[dataset_name] = meta

    results = {
        "config": {
            "runner": "d26_upgd_memory_digit_stress",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "datasets": datasets,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "variant_set": args.variant_set,
            "candidate_methods": list(candidate_methods),
            "variants": [variant.__dict__ for variant in variants],
            "elapsed_s": float(time.time() - t0),
        },
        "datasets": datasets_meta,
        "records": records,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(d18.MLP_METHODS),
        "aggregate": d18.aggregate_records(records, candidate_methods),
    }
    output_path = args.output_dir / f"{args.result_prefix}_results.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    write_summary(summary_path, results)
    write_summary(args.note_path, results)
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

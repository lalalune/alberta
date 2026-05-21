#!/usr/bin/env python3
"""D23: lean UPGD compute-knob sweep on OPMNIST.

This runner isolates the UPGD efficiency knobs that matter after D21:
perturbation interval, noise family, perturbation removal, hidden width, and
state tracking removal.  It uses the same compact OPMNIST protocol and fair MLP
baselines as D21/D22.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import d18_opmnist_bridge as bridge  # noqa: E402
from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    aggregate_records,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)
from d21_upgd_opmnist_efficiency import (  # noqa: E402
    evaluate_upgd_views,
    run_upgd_stream,
    summarize_predictions,
    trainable_param_count,
    tree_float_size,
)

from alberta_framework.core.optimizers import ObGDBounding  # noqa: E402
from alberta_framework.core.upgd import UPGDLearner  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d23_upgd_lean_sweep")
DEFAULT_RESULT_PREFIX = "d23_upgd_lean_sweep"
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d23_upgd_lean_sweep.md")
DATASET_NAME = "opmnist_bridge"
N_CLASSES = bridge.N_CLASSES


@dataclass(frozen=True)
class UPGDLeanVariant:
    """One UPGD compute-knob variant."""

    name: str
    hidden_size: int = 64
    perturbation_sigma: float = 1e-4
    perturbation_interval: int = 16
    perturbation_noise: str = "rademacher"
    track_unit_utilities: bool = False
    track_gradient_history: bool = False


def default_variants() -> tuple[UPGDLeanVariant, ...]:
    """Return the compact compute-knob sweep."""
    return (
        UPGDLeanVariant("upgd_h16_rademacher_i16_lean", hidden_size=16),
        UPGDLeanVariant("upgd_h32_rademacher_i16_lean", hidden_size=32),
        UPGDLeanVariant("upgd_h64_sigma0_lean", perturbation_sigma=0.0),
        UPGDLeanVariant("upgd_h64_rademacher_i4_lean", perturbation_interval=4),
        UPGDLeanVariant("upgd_h64_rademacher_i16_lean"),
        UPGDLeanVariant("upgd_h64_rademacher_i32_lean", perturbation_interval=32),
        UPGDLeanVariant("upgd_h64_normal_i16_lean", perturbation_noise="normal"),
        UPGDLeanVariant(
            "upgd_h64_rademacher_i16_fulltrack",
            track_unit_utilities=True,
            track_gradient_history=True,
        ),
    )


def make_upgd(variant: UPGDLeanVariant) -> UPGDLearner:
    """Construct one UPGD compute-knob variant."""
    return UPGDLearner(
        n_heads=N_CLASSES,
        hidden_sizes=(variant.hidden_size,),
        step_size=0.03,
        bounder=ObGDBounding(kappa=0.5),
        sparsity=0.5,
        use_layer_norm=True,
        perturbation_sigma=variant.perturbation_sigma,
        perturbation_noise=variant.perturbation_noise,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=variant.perturbation_interval,
        loss_normalization="target_structure",
        readout_mode="softmax_ce",
        track_unit_utilities=variant.track_unit_utilities,
        track_gradient_history=variant.track_gradient_history,
    )


def parse_optional_positive_int(value: str) -> int | None:
    """Parse bridge-compatible optional integer values."""
    parsed = bridge.parse_optional_positive_int(value)
    if parsed is None:
        return None
    return int(parsed)


def run_one_seed(
    seed: int,
    variants: tuple[UPGDLeanVariant, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one paired OPMNIST seed."""
    dataset, stream = bridge.make_stream(seed, args)
    observations = np.asarray(stream.observations, dtype=np.float32)
    targets = np.asarray(stream.targets, dtype=np.float32)
    labels = np.asarray(stream.labels, dtype=np.int32)
    methods: dict[str, dict[str, float]] = {}

    for method in MLP_METHODS:
        print(f"opmnist seed={seed}: running {method}")
        mlp = make_mlp(
            method=method,
            n_heads=N_CLASSES,
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 90_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method].update(
            bridge.evaluate_mlp_classifier_views(
                learner=mlp,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.mlp_deployment_transform,
            )
        )
        runtime = float(time.time() - t0)
        methods[method]["runtime_s"] = runtime
        methods[method]["steps_per_second"] = float(args.steps / runtime)

    for variant in variants:
        print(f"opmnist seed={seed}: running {variant.name}")
        learner = make_upgd(variant)
        state = learner.init(observations.shape[1], jr.key(seed + 91_000 + variant.hidden_size))
        initial_params = trainable_param_count(state)
        initial_float_state = tree_float_size(state)
        final_state, predictions, _metrics, runtime = run_upgd_stream(
            learner,
            state,
            observations,
            targets,
        )
        methods[variant.name] = summarize_predictions(
            predictions,
            targets,
            labels,
            args.final_window,
        )
        methods[variant.name].update(
            evaluate_upgd_views(learner, final_state, stream.test_views, stream.test_labels)
        )
        methods[variant.name].update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "trainable_params": float(initial_params),
                "float_state_size": float(initial_float_state),
            }
        )

    meta = dict(dataset.metadata)
    meta.update(stream.metadata)
    return (
        {
            "dataset_name": DATASET_NAME,
            "seed": int(seed),
            "dataset": meta,
            "methods": methods,
        },
        meta,
    )


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one metric cell."""
    if metric not in row:
        return ""
    value = row[metric]
    return f"{value['mean']:.6f} +/- {value['stderr']:.6f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    aggregate = results["aggregate"][DATASET_NAME]
    comparisons = aggregate["comparisons"]
    lines = [
        "# D23 UPGD Lean Sweep OPMNIST",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks."
        ),
        "",
        "| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s | "
        "Steps/s | Params | Float State |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in aggregate.items():
        if method == "comparisons":
            continue
        lines.append(
            f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
            f"{metric_cell(row, 'final_window_accuracy')} | "
            f"{metric_cell(row, 'test_mse')} | "
            f"{metric_cell(row, 'test_accuracy')} | "
            f"{metric_cell(row, 'runtime_s')} | "
            f"{metric_cell(row, 'steps_per_second')} | "
            f"{metric_cell(row, 'trainable_params')} | "
            f"{metric_cell(row, 'float_state_size')} |"
        )
    lines.extend(["", "## Best UPGD Variant vs Best MLP", ""])
    for metric in (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "deployment_test_mse",
        "deployment_test_accuracy",
    ):
        if metric not in comparisons:
            continue
        comparison = comparisons[metric]["best_kernel_vs_best_mlp"]
        lines.append(f"- `{metric}`: {bridge.comparison_cell(comparison)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument(
        "--mnist-source",
        choices=(
            "auto",
            "openml",
            "torchvision",
            "sklearn_digits_28x28",
            "sklearn_digits_8x8",
        ),
        default="sklearn_digits_28x28",
    )
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--allow-torchvision-download", action="store_true")
    parser.add_argument("--mnist-split", choices=("stratified", "canonical"), default="stratified")
    parser.add_argument("--openml-data-home", type=Path, default=None)
    parser.add_argument("--torchvision-data-home", type=Path, default=None)
    parser.add_argument("--openml-n-retries", type=int, default=2)
    parser.add_argument("--openml-retry-delay", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=parse_optional_positive_int, default=1000)
    parser.add_argument("--max-test-examples", type=parse_optional_positive_int, default=400)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--task-block-size", type=int, default=200)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--include-identity-permutation", action="store_true")
    parser.add_argument(
        "--task-sampling",
        choices=("random", "sequential_epoch"),
        default="sequential_epoch",
    )
    parser.add_argument(
        "--max-test-permutation-views",
        type=parse_optional_positive_int,
        default=None,
    )
    parser.add_argument("--evaluate-all-permutation-views", action="store_true")
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--mlp-deployment-transform",
        choices=bridge.DEPLOYMENT_TRANSFORMS,
        default="raw",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def apply_smoke(args: argparse.Namespace) -> None:
    """Apply quick smoke settings."""
    if not args.smoke:
        return
    args.steps = 120
    args.n_seeds = 1
    args.final_window = 40
    args.max_train_examples = 200
    args.max_test_examples = 80
    args.n_permutations = 2
    args.task_block_size = 40


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.n_permutations < 2:
        raise ValueError("--n-permutations must be at least 2")
    if args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive or 'all'")
    if args.max_test_examples is not None and args.max_test_examples <= 0:
        raise ValueError("--max-test-examples must be positive or 'all'")
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")
    if not 0.0 <= args.mlp_sparsity < 1.0:
        raise ValueError("--mlp-sparsity must be in [0, 1)")


def main() -> None:
    """Run D23."""
    args = parse_args()
    apply_smoke(args)
    validate_args(args)
    variants = default_variants()
    candidate_methods = tuple(variant.name for variant in variants)

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, variants, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta

    results = {
        "config": {
            "runner": "d23_upgd_lean_sweep_opmnist",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "mnist_source": args.mnist_source,
            "max_train_examples": args.max_train_examples,
            "max_test_examples": args.max_test_examples,
            "n_permutations": args.n_permutations,
            "task_block_size": args.task_block_size,
            "task_sampling": args.task_sampling,
            "sample_with_replacement": args.sample_with_replacement,
            "variants": [variant.__dict__ for variant in variants],
        },
        "datasets": datasets_meta,
        "records": records,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "aggregate": bridge.add_deployment_comparisons(
            aggregate_records(records, candidate_methods),
            records,
            candidate_methods,
        ),
        "wall_clock_s": float(time.time() - t0),
        "evidence_level": "upgd_lean_sweep_opmnist",
    }
    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    bridge.atomic_write_json(json_path, results)
    write_summary(summary_path, results)
    if args.note_path is not None:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path is not None:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

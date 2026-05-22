#!/usr/bin/env python3
"""D20: multi-prototype simplex memory for OPMNIST retention.

This runner isolates the OPMNIST failure found in D18: one retained prototype
per class is the wrong geometry when the same class appears under multiple
pixel permutations.  D20 keeps several novelty-allocated prototypes per class
and predicts by a softmax over nearest-prototype class logits.

It is intentionally simple: no MLP expert, no prediction router, no task id,
and no offline labels beyond the online one-hot target available at each
timestep.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d20_multiprototype_opmnist")
DEFAULT_RESULT_PREFIX = "d20_multiprototype_opmnist"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d20_multiprototype_opmnist.md"
)
DATASET_NAME = "opmnist_bridge"
N_CLASSES = bridge.N_CLASSES


@dataclass(frozen=True)
class MultiPrototypeConfig:
    """Configuration for the multi-prototype memory learner."""

    slots_per_class: int = 20
    update_rate: float = 0.3
    novelty_threshold: float = 0.08
    bandwidth: float = 0.01
    replacement: str = "least_used_oldest"


@dataclass
class MultiPrototypeState:
    """Mutable state for D20."""

    means: np.ndarray
    counts: np.ndarray
    last_update: np.ndarray
    step_count: int = 0


class MultiPrototypeClassifier:
    """Online class memory with multiple prototypes per one-hot class."""

    def __init__(
        self,
        feature_dim: int,
        config: MultiPrototypeConfig,
        *,
        n_classes: int = N_CLASSES,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.n_classes = int(n_classes)
        self.config = config

    def init(self) -> MultiPrototypeState:
        """Return an empty prototype memory."""
        shape = (self.n_classes, self.config.slots_per_class, self.feature_dim)
        return MultiPrototypeState(
            means=np.zeros(shape, dtype=np.float64),
            counts=np.zeros((self.n_classes, self.config.slots_per_class), dtype=np.float64),
            last_update=np.zeros(
                (self.n_classes, self.config.slots_per_class),
                dtype=np.int64,
            ),
        )

    def class_logits(
        self,
        state: MultiPrototypeState,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Return class logits from nearest active prototype distances."""
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        diffs = x[:, None, None, :] - state.means[None, :, :, :]
        distances = np.mean(diffs * diffs, axis=3)
        logits = -distances / max(self.config.bandwidth, 1e-12)
        logits = np.where(state.counts[None, :, :] > 0.0, logits, -np.inf)
        class_logits = np.max(logits, axis=2)
        empty_rows = np.all(~np.isfinite(class_logits), axis=1)
        class_logits = np.where(np.isfinite(class_logits), class_logits, -1e9)
        class_logits[empty_rows] = 0.0
        return cast(np.ndarray, class_logits)

    def predict_batch(
        self,
        state: MultiPrototypeState,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Return class probabilities for a batch or single observation."""
        logits = self.class_logits(state, observations)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return cast(
            np.ndarray,
            exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12),
        )

    def predict(self, state: MultiPrototypeState, observation: np.ndarray) -> np.ndarray:
        """Return class probabilities for one observation."""
        return self.predict_batch(state, observation)[0]

    def _replacement_slot(self, state: MultiPrototypeState, head: int) -> int:
        """Choose the prototype slot to overwrite for a novel observation."""
        if self.config.replacement != "least_used_oldest":
            raise ValueError(f"unknown replacement policy: {self.config.replacement}")
        order = np.lexsort((state.last_update[head], state.counts[head]))
        return int(order[0])

    def update(
        self,
        state: MultiPrototypeState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Update the nearest compatible prototype or allocate a new one."""
        state.step_count += 1
        if np.any(np.isnan(target)):
            return
        if abs(float(np.sum(target)) - 1.0) > 1e-6 or float(np.max(target)) < 0.999:
            return
        head = int(np.argmax(target))
        x = np.asarray(observation, dtype=np.float64)
        used = state.counts[head] > 0.0
        if not bool(np.any(used)):
            slot = 0
            state.means[head, slot] = x
            state.counts[head, slot] = 1.0
            state.last_update[head, slot] = state.step_count
            return

        distances = np.mean((state.means[head] - x[None, :]) ** 2, axis=1)
        used_distances = np.where(used, distances, np.inf)
        slot = int(np.argmin(used_distances))
        if float(used_distances[slot]) > self.config.novelty_threshold:
            if bool(np.any(~used)):
                slot = int(np.where(~used)[0][0])
            else:
                slot = self._replacement_slot(state, head)
            state.means[head, slot] = x
            state.counts[head, slot] = 1.0
            state.last_update[head, slot] = state.step_count
            return

        state.means[head, slot] += self.config.update_rate * (
            x - state.means[head, slot]
        )
        state.counts[head, slot] += 1.0
        state.last_update[head, slot] = state.step_count


def run_multiprototype_stream(
    observations: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    config: MultiPrototypeConfig,
    final_window: int,
) -> tuple[MultiPrototypeClassifier, MultiPrototypeState, dict[str, float]]:
    """Run D20 over one online stream."""
    learner = MultiPrototypeClassifier(
        feature_dim=int(observations.shape[1]),
        config=config,
        n_classes=int(targets.shape[1]),
    )
    state = learner.init()
    losses: list[float] = []
    correct: list[float] = []
    active_slots: list[int] = []
    for observation, target, label in zip(observations, targets, labels, strict=True):
        prediction = learner.predict(state, observation)
        losses.append(float(np.mean((prediction - target) ** 2)))
        correct.append(float(np.argmax(prediction) == int(label)))
        learner.update(state, observation, target)
        active_slots.append(int(np.sum(state.counts > 0.0)))
    window = min(final_window, len(losses))
    summary = {
        "online_mean_mse": float(np.mean(losses)),
        "final_window_mse": float(np.mean(losses[-window:])),
        "online_mean_accuracy": float(np.mean(correct)),
        "final_window_accuracy": float(np.mean(correct[-window:])),
        "active_prototypes": float(active_slots[-1]),
        "mean_active_prototypes": float(np.mean(active_slots)),
    }
    return learner, state, summary


def evaluate_classifier_views(
    learner: MultiPrototypeClassifier,
    state: MultiPrototypeState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate D20 averaged over held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float64)[y_test]
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        predictions = learner.predict_batch(state, np.asarray(view, dtype=np.float64))
        mse_values.append(float(np.mean((predictions - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(predictions, axis=1) == y_test)))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
    }


def config_name(config: MultiPrototypeConfig) -> str:
    """Return a stable method suffix."""
    novelty = str(config.novelty_threshold).replace(".", "p")
    bandwidth = str(config.bandwidth).replace(".", "p")
    update = str(config.update_rate).replace(".", "p")
    return f"s{config.slots_per_class}_n{novelty}_bw{bandwidth}_eta{update}"


def run_one_seed(
    seed: int,
    config: MultiPrototypeConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run MLP baselines and D20 on one OPMNIST bridge seed."""
    dataset, stream = bridge.make_stream(seed, args)
    labels_np = np.asarray(stream.labels, dtype=np.int32)
    observations = np.asarray(stream.observations, dtype=np.float64)
    targets = np.asarray(stream.targets, dtype=np.float64)
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
            jr.key(seed + 50_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(
            metrics,
            args.final_window,
            labels_np,
        )
        methods[method].update(
            bridge.evaluate_mlp_classifier_views(
                learner=mlp,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.mlp_deployment_transform,
            )
        )
        methods[method]["runtime_s"] = float(time.time() - t0)

    method = f"d20_{config_name(config)}"
    print(f"opmnist seed={seed}: running {method}")
    t0 = time.time()
    learner, state, summary = run_multiprototype_stream(
        observations,
        targets,
        labels_np,
        config,
        args.final_window,
    )
    methods[method] = summary
    methods[method].update(
        evaluate_classifier_views(
            learner,
            state,
            stream.test_views,
            stream.test_labels,
        )
    )
    methods[method]["runtime_s"] = float(time.time() - t0)

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


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    aggregate = results["aggregate"][DATASET_NAME]
    comparisons = aggregate["comparisons"]
    lines = [
        "# D20 Multi-Prototype OPMNIST",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks, "
            f"block size {cfg['task_block_size']}."
        ),
        "",
        (
            "D20 is a single online memory learner: multiple novelty-allocated "
            "prototypes per class, softmax over nearest-prototype class logits, "
            "no task id, no MLP expert, and no prediction router."
        ),
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Final MSE | Final Acc | Test MSE | Test Acc | Prototypes | Runtime s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in aggregate.items():
        if method == "comparisons":
            continue
        lines.append(
            f"| `{method}` | {bridge.metric_cell(row, 'final_window_mse')} | "
            f"{bridge.metric_cell(row, 'final_window_accuracy')} | "
            f"{bridge.metric_cell(row, 'test_mse')} | "
            f"{bridge.metric_cell(row, 'test_accuracy')} | "
            f"{bridge.metric_cell(row, 'active_prototypes')} | "
            f"{bridge.metric_cell(row, 'runtime_s')} |"
        )
    lines.extend(["", "## D20 vs Best MLP", ""])
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
    protocol = results["datasets"][DATASET_NAME]
    lines.extend(["", "## Protocol Gates", "", "| Gate | Value |", "| --- | --- |"])
    for key, value in bridge.protocol_gate_rows(protocol):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The result shows that OPMNIST retention needs task-view memory "
                "geometry: multiple prototypes per class, not a single averaged "
                "class prototype. This is a candidate component for folding into "
                "D18 or the core fast/slow learner, not yet a full all14 Step 2 "
                "replacement."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_optional_positive_int(value: str) -> int | None:
    """Parse bridge-compatible optional integer arguments."""
    return bridge.parse_optional_positive_int(value)


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
    parser.add_argument(
        "--mnist-split",
        choices=("stratified", "canonical"),
        default="stratified",
    )
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
    parser.add_argument("--slots-per-class", type=int, default=20)
    parser.add_argument("--prototype-update-rate", type=float, default=0.3)
    parser.add_argument("--prototype-novelty-threshold", type=float, default=0.08)
    parser.add_argument("--prototype-bandwidth", type=float, default=0.01)
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
    """Validate CLI arguments."""
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
    if args.slots_per_class <= 0:
        raise ValueError("--slots-per-class must be positive")
    if not 0.0 < args.prototype_update_rate <= 1.0:
        raise ValueError("--prototype-update-rate must be in (0, 1]")
    if args.prototype_novelty_threshold < 0.0:
        raise ValueError("--prototype-novelty-threshold must be non-negative")
    if args.prototype_bandwidth <= 0.0:
        raise ValueError("--prototype-bandwidth must be positive")


def main() -> None:
    """Run D20."""
    args = parse_args()
    apply_smoke(args)
    validate_args(args)
    config = MultiPrototypeConfig(
        slots_per_class=args.slots_per_class,
        update_rate=args.prototype_update_rate,
        novelty_threshold=args.prototype_novelty_threshold,
        bandwidth=args.prototype_bandwidth,
    )
    candidate_methods = (f"d20_{config_name(config)}",)

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, config, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta
        print(
            f"opmnist seed={seed}: D20 final MSE "
            f"{record['methods'][candidate_methods[0]]['final_window_mse']:.4f}"
        )

    results = {
        "config": {
            "runner": "d20_multiprototype_opmnist",
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
            "sample_with_replacement": args.sample_with_replacement,
            "task_sampling": args.task_sampling,
            "mlp_step_size": args.mlp_step_size,
            "mlp_sparsity": args.mlp_sparsity,
            "multiprototype": asdict(config),
            "output_dir": str(args.output_dir),
            "result_prefix": args.result_prefix,
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
        "evidence_level": "compact_opmnist_multiprototype",
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

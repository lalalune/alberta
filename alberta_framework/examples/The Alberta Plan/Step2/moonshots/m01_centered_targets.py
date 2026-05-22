#!/usr/bin/env python3
"""M01 centered target-code smoke experiment for Step 2 moonshots.

The experiment compares two architecture-identical ``MultiHeadMLPLearner``
classifiers on sklearn's bundled digits data:

* ``one_hot`` trains against the standard one-hot target vector.
* ``centered`` trains against a zero-sum code with +1 on the true class and
  ``-1 / (K - 1)`` on every other class.

Both learners see the same online stream and start from identical initial
parameters for each paired seed. Predictions are evaluated by argmax over the
10 heads. MSE is reported against each learner's own target code, so paired
accuracy is the primary code-invariant metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import MultiHeadMLPLearner, ObGDBounding  # noqa: E402

N_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("outputs/step2_moonshots/m01_centered_targets")
VALID_REGIMES = ("iid", "label_drift", "permuted_pixels", "class_blocked")
DEFAULT_REGIMES = ("iid", "label_drift")


@dataclass(frozen=True)
class DigitsSplit:
    """Standardized train/test arrays for sklearn digits."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class RegimeStream:
    """Materialized online stream plus final-phase held-out view."""

    observations: jax.Array
    labels: jax.Array
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def one_hot_targets(labels: np.ndarray) -> np.ndarray:
    """Return one-hot targets for integer class labels."""
    return np.eye(N_CLASSES, dtype=np.float32)[labels]


def centered_targets(labels: np.ndarray) -> np.ndarray:
    """Return +1 / negative-zero-sum centered targets for integer labels."""
    targets = np.full((labels.shape[0], N_CLASSES), -1.0 / (N_CLASSES - 1), dtype=np.float32)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return targets


def targets_for(labels: np.ndarray, code: str) -> np.ndarray:
    """Build targets for a named target code."""
    if code == "one_hot":
        return one_hot_targets(labels)
    if code == "centered":
        return centered_targets(labels)
    raise ValueError(f"unknown target code {code!r}")


def load_digits_split(seed: int, train_fraction: float) -> DigitsSplit:
    """Load sklearn digits and make a stratified standardized split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")

    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for this smoke experiment. Install the "
            "external extra with `pip install -e '.[external]'`."
        )
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for cls in range(N_CLASSES):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        n_train = max(1, min(len(cls_idx) - 1, n_train))
        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    train_idx = np.asarray(train_indices, dtype=np.int32)
    test_idx = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    return DigitsSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        meta={
            "dataset": "sklearn.datasets.load_digits",
            "dataset_description": "8x8 handwritten digit images, 1797 examples, 10 classes",
            "n_total": int(x.shape[0]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "n_classes": N_CLASSES,
            "train_fraction": train_fraction,
            "split_seed": seed,
        },
    )


def make_base_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    class_blocked: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample shuffled train-set epochs into a finite online sequence."""
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_CLASSES)]

    while total < steps:
        if class_blocked:
            epoch_parts: list[np.ndarray] = []
            for cls in rng.permutation(N_CLASSES):
                cls_idx = class_indices[int(cls)].copy()
                rng.shuffle(cls_idx)
                epoch_parts.append(cls_idx)
            indices = np.concatenate(epoch_parts)
        else:
            indices = rng.permutation(len(y_train))
        chunks_x.append(x_train[indices])
        chunks_y.append(y_train[indices])
        total += len(indices)

    observations = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    return observations, labels


def phase_count(steps: int, phase_length: int) -> int:
    """Return number of nonstationary phases spanned by the stream."""
    return int(math.ceil(steps / phase_length))


def label_permutations(n_phases: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Identity first phase, random class-head remappings thereafter."""
    permutations = [np.arange(N_CLASSES, dtype=np.int32)]
    for _ in range(1, n_phases):
        permutations.append(rng.permutation(N_CLASSES).astype(np.int32))
    return permutations


def pixel_permutations(
    feature_dim: int,
    n_phases: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Identity first phase, random pixel orderings thereafter."""
    permutations = [np.arange(feature_dim, dtype=np.int32)]
    for _ in range(1, n_phases):
        permutations.append(rng.permutation(feature_dim).astype(np.int32))
    return permutations


def make_regime_stream(
    regime: str,
    split: DigitsSplit,
    steps: int,
    seed: int,
    phase_length: int,
) -> RegimeStream:
    """Create an IID or nonstationary online digits stream."""
    if regime not in VALID_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {VALID_REGIMES}")
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if phase_length <= 0:
        raise ValueError("--phase-length must be positive")

    observations, labels = make_base_sequence(
        x_train=split.x_train,
        y_train=split.y_train,
        steps=steps,
        seed=seed,
        class_blocked=regime == "class_blocked",
    )
    x_test = split.x_test.copy()
    y_test = split.y_test.copy()
    phase_ids = np.arange(steps, dtype=np.int32) // phase_length
    n_phases = phase_count(steps, phase_length)
    final_phase = int(phase_ids[-1])
    rng = np.random.default_rng(seed + 50_000)

    meta: dict[str, Any] = {
        "regime": regime,
        "phase_length": phase_length,
        "n_phases": n_phases,
        "final_phase": final_phase,
    }

    if regime == "label_drift":
        permutations = label_permutations(n_phases, rng)
        for phase, perm in enumerate(permutations):
            phase_mask = phase_ids == phase
            labels[phase_mask] = perm[labels[phase_mask]]
        y_test = permutations[final_phase][y_test]
        meta["description"] = (
            "Class-head meanings are permuted by phase; held-out uses the final phase."
        )
    elif regime == "permuted_pixels":
        permutations = pixel_permutations(observations.shape[1], n_phases, rng)
        for phase, perm in enumerate(permutations):
            phase_mask = phase_ids == phase
            observations[phase_mask] = observations[phase_mask][:, perm]
        x_test = x_test[:, permutations[final_phase]]
        meta["description"] = "Pixel order is permuted by phase; held-out uses the final phase."
    elif regime == "class_blocked":
        meta["description"] = (
            "Training stream is grouped into digit-class blocks within each epoch."
        )
    else:
        meta["description"] = "Stationary IID shuffled-epoch control."

    return RegimeStream(
        observations=jnp.asarray(observations),
        labels=jnp.asarray(labels),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
        meta=meta,
    )


def make_learner(
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    bounder_kappa: float,
    use_layer_norm: bool,
) -> MultiHeadMLPLearner:
    """Build the architecture used by both target-code treatments."""
    return MultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=bounder_kappa),
        sparsity=sparsity,
        use_layer_norm=use_layer_norm,
    )


def run_online_stream(
    learner: MultiHeadMLPLearner,
    key: jax.Array,
    observations: jax.Array,
    labels: jax.Array,
    target_code: str,
) -> tuple[Any, np.ndarray]:
    """Run one learner online and return final state plus [code_mse, correct]."""
    labels_np = np.asarray(labels)
    targets = jnp.asarray(targets_for(labels_np, target_code))
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        mse = jnp.mean((result.predictions - target) ** 2)
        correct = jnp.argmax(result.predictions) == label
        return result.state, jnp.stack([mse, correct.astype(jnp.float32)])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def evaluate_classifier(
    learner: MultiHeadMLPLearner,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_code: str,
) -> dict[str, float]:
    """Evaluate final learner state on the held-out final-phase split."""
    observations = jnp.asarray(x_test.astype(np.float32))
    labels = jnp.asarray(y_test.astype(np.int32))
    targets = jnp.asarray(targets_for(y_test, target_code))
    predictions = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((predictions - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(predictions, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def summarize_curve(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize prequential online metrics."""
    if final_window <= 0:
        raise ValueError("--final-window must be positive")
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def stderr(values: np.ndarray) -> float:
    """Return standard error with a zero fallback for one seed."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def paired_lower_summary(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, float | int | str]:
    """Summarize paired one-hot minus centered differences for lower-is-better metrics."""
    one_hot = np.asarray([r["methods"]["one_hot"][metric] for r in records], dtype=np.float64)
    centered = np.asarray([r["methods"]["centered"][metric] for r in records], dtype=np.float64)
    diff = one_hot - centered
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "metric": metric,
        "one_hot_mean": float(np.mean(one_hot)),
        "one_hot_stderr": stderr(one_hot),
        "centered_mean": float(np.mean(centered)),
        "centered_stderr": stderr(centered),
        "one_hot_minus_centered_mean": float(np.mean(diff)),
        "one_hot_minus_centered_stderr": stderr(diff),
        "wins_for_centered": int(np.sum(diff > 0.0)),
        "wins_for_one_hot": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n_pairs": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def paired_higher_summary(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, float | int | str]:
    """Summarize paired centered minus one-hot differences for higher-is-better metrics."""
    one_hot = np.asarray([r["methods"]["one_hot"][metric] for r in records], dtype=np.float64)
    centered = np.asarray([r["methods"]["centered"][metric] for r in records], dtype=np.float64)
    diff = centered - one_hot
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "metric": metric,
        "one_hot_mean": float(np.mean(one_hot)),
        "one_hot_stderr": stderr(one_hot),
        "centered_mean": float(np.mean(centered)),
        "centered_stderr": stderr(centered),
        "centered_minus_one_hot_mean": float(np.mean(diff)),
        "centered_minus_one_hot_stderr": stderr(diff),
        "wins_for_centered": int(np.sum(diff > 0.0)),
        "wins_for_one_hot": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n_pairs": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int | str]]:
    """Aggregate all metrics over paired records."""
    return {
        "online_mean_mse": paired_lower_summary(records, "online_mean_mse"),
        "final_window_mse": paired_lower_summary(records, "final_window_mse"),
        "test_mse": paired_lower_summary(records, "test_mse"),
        "online_mean_accuracy": paired_higher_summary(records, "online_mean_accuracy"),
        "final_window_accuracy": paired_higher_summary(records, "final_window_accuracy"),
        "test_accuracy": paired_higher_summary(records, "test_accuracy"),
    }


def scaling_decision(overall: dict[str, dict[str, float | int | str]]) -> tuple[bool, str]:
    """Apply the M01 smoke criterion for whether this is worth scaling."""
    acc_diff = float(overall["final_window_accuracy"]["centered_minus_one_hot_mean"])
    mse_diff = float(overall["final_window_mse"]["one_hot_minus_centered_mean"])
    if acc_diff > 0.0:
        return True, "centered target code improved paired final-window accuracy."
    if mse_diff > 0.0:
        return True, "centered target code improved paired final-window code MSE."
    return False, "centered target code did not improve paired final-window accuracy or code MSE."


def format_mean(row: dict[str, float | int | str], method: str) -> str:
    """Format mean +/- stderr for a method in a summary row."""
    mean = float(row[f"{method}_mean"])
    err = float(row[f"{method}_stderr"])
    return f"{mean:.4f} +/- {err:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown result note."""
    cfg = results["config"]
    lines = [
        "# M01 Centered Targets",
        "",
        "Hypothesis: one-hot MSE on 10-class digits wastes gradient mass on negative classes. "
        "A zero-sum centered target code may improve online conditioning without changing the "
        "MultiHeadMLPLearner architecture.",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} online steps, "
            f"last-{cfg['final_window']} final-window metrics, phase_length={cfg['phase_length']}."
        ),
        f"Regimes: {', '.join(cfg['regimes'])}.",
        (
            "Target codes: `one_hot` uses 1 for the true class and 0 otherwise; "
            "`centered` uses +1 for the true class and -1/(K-1) otherwise."
        ),
        "",
    ]

    for regime, regime_result in results["regime_results"].items():
        lines.extend(
            [
                f"## Regime: {regime}",
                "",
                regime_result["regime_meta"]["description"],
                "",
                "| Metric | One-hot | Centered | Paired diff | Centered wins | One-hot wins |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        aggregate = regime_result["aggregate"]
        for key in ("final_window_accuracy", "final_window_mse", "test_accuracy", "test_mse"):
            row = aggregate[key]
            diff_key = (
                "one_hot_minus_centered_mean"
                if key.endswith("mse")
                else "centered_minus_one_hot_mean"
            )
            lines.append(
                f"| {key} | {format_mean(row, 'one_hot')} | "
                f"{format_mean(row, 'centered')} | {float(row[diff_key]):+.4f} | "
                f"{row['wins_for_centered']}/{row['n_pairs']} | "
                f"{row['wins_for_one_hot']}/{row['n_pairs']} |"
            )
        lines.append("")

    overall = results["overall_aggregate"]
    lines.extend(
        [
            "## Overall Decision",
            "",
            f"Decision: {'worth scaling' if results['worth_scaling'] else 'not worth scaling'}.",
            results["decision_reason"],
            "",
            "| Metric | One-hot | Centered | Paired diff | Centered wins | One-hot wins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("final_window_accuracy", "final_window_mse", "test_accuracy", "test_mse"):
        row = overall[key]
        diff_key = (
            "one_hot_minus_centered_mean" if key.endswith("mse") else "centered_minus_one_hot_mean"
        )
        lines.append(
            f"| {key} | {format_mean(row, 'one_hot')} | "
            f"{format_mean(row, 'centered')} | {float(row[diff_key]):+.4f} | "
            f"{row['wins_for_centered']}/{row['n_pairs']} | "
            f"{row['wins_for_one_hot']}/{row['n_pairs']} |"
        )

    lines.extend(
        [
            "",
            "Note: MSE is measured against each method's own target code; accuracy is the primary "
            "code-invariant metric.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--phase-length", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--bounder-kappa", type=float, default=2.0)
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=VALID_REGIMES,
        default=list(DEFAULT_REGIMES),
        help="Online stream regimes. Defaults to iid and label_drift.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the smoke experiment and write JSON/Markdown artifacts."""
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.phase_length <= 0:
        raise ValueError("--phase-length must be positive")

    start_time = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes = list(dict.fromkeys(args.regimes))
    hidden_sizes = (args.hidden_size,)
    use_layer_norm = not args.no_layer_norm

    regime_records: dict[str, list[dict[str, Any]]] = {regime: [] for regime in regimes}
    regime_meta: dict[str, dict[str, Any]] = {}
    dataset_meta: dict[str, Any] | None = None

    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        split = load_digits_split(seed=seed, train_fraction=args.train_fraction)
        dataset_meta = split.meta

        for regime_idx, regime in enumerate(regimes):
            stream = make_regime_stream(
                regime=regime,
                split=split,
                steps=args.steps,
                seed=seed + 10_000 + 1_000 * regime_idx,
                phase_length=args.phase_length,
            )
            regime_meta.setdefault(regime, stream.meta)

            learner = make_learner(
                hidden_sizes=hidden_sizes,
                step_size=args.step_size,
                sparsity=args.sparsity,
                bounder_kappa=args.bounder_kappa,
                use_layer_norm=use_layer_norm,
            )
            init_key = jr.key(seed + 100_000 * (regime_idx + 1))

            print(f"regime={regime} seed={seed}: running one_hot")
            one_hot_state, one_hot_metrics = run_online_stream(
                learner=learner,
                key=init_key,
                observations=stream.observations,
                labels=stream.labels,
                target_code="one_hot",
            )
            print(f"regime={regime} seed={seed}: running centered")
            centered_state, centered_metrics = run_online_stream(
                learner=learner,
                key=init_key,
                observations=stream.observations,
                labels=stream.labels,
                target_code="centered",
            )

            one_hot_summary = summarize_curve(one_hot_metrics, args.final_window)
            one_hot_summary.update(
                evaluate_classifier(learner, one_hot_state, stream.x_test, stream.y_test, "one_hot")
            )
            centered_summary = summarize_curve(centered_metrics, args.final_window)
            centered_summary.update(
                evaluate_classifier(
                    learner,
                    centered_state,
                    stream.x_test,
                    stream.y_test,
                    "centered",
                )
            )

            regime_records[regime].append(
                {
                    "seed": seed,
                    "methods": {
                        "one_hot": one_hot_summary,
                        "centered": centered_summary,
                    },
                }
            )
            print(
                f"regime={regime} seed={seed}: "
                f"final-window acc one_hot={one_hot_summary['final_window_accuracy']:.3f}, "
                f"centered={centered_summary['final_window_accuracy']:.3f}; "
                f"test acc one_hot={one_hot_summary['test_accuracy']:.3f}, "
                f"centered={centered_summary['test_accuracy']:.3f}"
            )

    all_records = [record for records in regime_records.values() for record in records]
    regime_results: dict[str, dict[str, Any]] = {}
    for regime, records in regime_records.items():
        regime_results[regime] = {
            "regime_meta": regime_meta[regime],
            "records": records,
            "aggregate": aggregate_records(records),
        }
    overall_aggregate = aggregate_records(all_records)
    worth_scaling, decision_reason = scaling_decision(overall_aggregate)

    results = {
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "final_window": args.final_window,
            "phase_length": args.phase_length,
            "hidden_sizes": list(hidden_sizes),
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "bounder_kappa": args.bounder_kappa,
            "use_layer_norm": use_layer_norm,
            "regimes": regimes,
        },
        "dataset": dataset_meta,
        "target_codes": {
            "one_hot": "true class=1.0, other classes=0.0",
            "centered": f"true class=1.0, other classes={-1.0 / (N_CLASSES - 1):.8f}",
        },
        "regime_results": regime_results,
        "overall_aggregate": overall_aggregate,
        "worth_scaling": worth_scaling,
        "decision_reason": decision_reason,
        "wall_clock_s": time.time() - start_time,
        "evidence_level": "quick_smoke_sklearn_digits_online",
    }

    json_path = output_dir / "m01_centered_targets_results.json"
    md_path = output_dir / "m01_centered_targets_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

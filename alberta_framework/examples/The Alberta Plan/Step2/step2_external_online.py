#!/usr/bin/env python3
"""Externally grounded Step 2 online supervised benchmark.

This benchmark uses scikit-learn's bundled handwritten ``digits`` dataset
instead of a synthetic stream.  The protocol converts the fixed dataset into
an online prequential classification stream: at each step the learner predicts
the current digit from its 8x8 image features, records pre-update loss and
accuracy, then updates on the one-hot target.

The comparison is intentionally narrow and fair:

* ``mlp``: ``MultiHeadMLPLearner`` with a shared hidden trunk and 10 heads.
* ``upgd``: same hidden trunk, step size, sparsity, layer norm, and ObGD
  bounding, plus UPGD's utility-scaled perturbations.

Outputs are written under ``output/direction4_external_nonstationary/`` by default.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import MultiHeadMLPLearner, ObGDBounding, UPGDLearner  # noqa: E402

N_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("output/direction4_external_nonstationary")
VALID_REGIMES = ("iid", "class_blocked", "permuted_pixels", "mask_noise", "label_drift")


@dataclass(frozen=True)
class RegimeStream:
    """Materialized stream plus final-phase held-out evaluation view."""

    observations: jax.Array
    targets: jax.Array
    labels: jax.Array
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def load_digits_arrays(
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and standardize sklearn digits using only train-split statistics."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for this externally grounded benchmark. "
            "Install sklearn or run this script in an environment that includes it. "
            "No network access is needed at runtime because load_digits is bundled."
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
        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    train_indices_arr = np.asarray(train_indices, dtype=np.int32)
    test_indices_arr = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_indices_arr)
    rng.shuffle(test_indices_arr)

    x_train = x[train_indices_arr]
    y_train = y[train_indices_arr]
    x_test = x[test_indices_arr]
    y_test = y[test_indices_arr]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    meta = {
        "dataset": "sklearn.datasets.load_digits",
        "dataset_description": "8x8 handwritten digit images, 1797 examples, 10 classes",
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_CLASSES,
        "train_fraction": train_fraction,
        "split_seed": seed,
    }
    return x_train, y_train, x_test, y_test, meta


def make_base_online_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    class_blocked: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a finite online stream by sampling shuffled train-set epochs."""
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0

    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_CLASSES)]
    while total < steps:
        if class_blocked:
            order = rng.permutation(N_CLASSES)
            epoch_parts: list[np.ndarray] = []
            for cls in order:
                cls_idx = class_indices[int(cls)].copy()
                rng.shuffle(cls_idx)
                epoch_parts.append(cls_idx)
            indices = np.concatenate(epoch_parts)
        else:
            indices = rng.permutation(len(y_train))
        chunks_x.append(x_train[indices])
        chunks_y.append(y_train[indices])
        total += len(indices)

    obs = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    return obs, labels


def _phase_count(steps: int, phase_length: int) -> int:
    return int(math.ceil(steps / phase_length))


def _label_permutations(n_phases: int, rng: np.random.Generator) -> list[np.ndarray]:
    perms = [np.arange(N_CLASSES, dtype=np.int32)]
    for _ in range(1, n_phases):
        perms.append(rng.permutation(N_CLASSES).astype(np.int32))
    return perms


def _pixel_permutations(
    feature_dim: int,
    n_phases: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    perms = [np.arange(feature_dim, dtype=np.int32)]
    for _ in range(1, n_phases):
        perms.append(rng.permutation(feature_dim).astype(np.int32))
    return perms


def _feature_masks(
    feature_dim: int,
    n_phases: int,
    keep_fraction: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_keep = max(1, min(feature_dim, int(round(feature_dim * keep_fraction))))
    masks: list[np.ndarray] = []
    for _ in range(n_phases):
        keep = rng.choice(feature_dim, size=n_keep, replace=False)
        mask = np.zeros(feature_dim, dtype=np.float32)
        mask[keep] = 1.0
        masks.append(mask)
    return masks


def _apply_phasewise_pixel_permutation(
    observations: np.ndarray,
    phase_ids: np.ndarray,
    permutations: list[np.ndarray],
) -> np.ndarray:
    out = observations.copy()
    for phase, perm in enumerate(permutations):
        phase_mask = phase_ids == phase
        out[phase_mask] = out[phase_mask][:, perm]
    return out.astype(np.float32)


def _apply_phasewise_mask_noise(
    observations: np.ndarray,
    phase_ids: np.ndarray,
    masks: list[np.ndarray],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = observations.copy()
    if noise_std > 0.0:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    for phase, mask in enumerate(masks):
        phase_mask = phase_ids == phase
        out[phase_mask] = out[phase_mask] * mask
    return out.astype(np.float32)


def make_regime_stream(
    regime: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    steps: int,
    seed: int,
    phase_length: int,
    mask_keep_fraction: float,
    noise_std: float,
) -> RegimeStream:
    """Create one stationary or nonstationary online digits stream."""
    if regime not in VALID_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {VALID_REGIMES}")
    if phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative")

    base_x, base_labels = make_base_online_sequence(
        x_train=x_train,
        y_train=y_train,
        steps=steps,
        seed=seed,
        class_blocked=regime == "class_blocked",
    )
    observations = base_x.copy()
    labels = base_labels.copy()
    phase_ids = np.arange(steps, dtype=np.int32) // phase_length
    n_phases = _phase_count(steps, phase_length)
    final_phase = int(phase_ids[-1])
    rng = np.random.default_rng(seed + 50_000)

    test_x = x_test.astype(np.float32).copy()
    test_y = y_test.astype(np.int32).copy()
    meta: dict[str, Any] = {
        "regime": regime,
        "phase_length": phase_length,
        "n_phases": n_phases,
        "final_phase": final_phase,
        "class_blocked": regime == "class_blocked",
    }

    if regime == "permuted_pixels":
        permutations = _pixel_permutations(observations.shape[1], n_phases, rng)
        observations = _apply_phasewise_pixel_permutation(observations, phase_ids, permutations)
        test_x = test_x[:, permutations[final_phase]]
        meta["description"] = "Pixel order is changed by phase; held-out uses final phase."
    elif regime == "mask_noise":
        masks = _feature_masks(observations.shape[1], n_phases, mask_keep_fraction, rng)
        observations = _apply_phasewise_mask_noise(observations, phase_ids, masks, noise_std, rng)
        test_rng = np.random.default_rng(seed + 60_000 + final_phase)
        if noise_std > 0.0:
            test_x = test_x + test_rng.normal(0.0, noise_std, size=test_x.shape).astype(np.float32)
        test_x = test_x * masks[final_phase]
        meta.update({
            "description": "Visible feature mask rotates by phase with Gaussian feature noise.",
            "mask_keep_fraction": mask_keep_fraction,
            "noise_std": noise_std,
        })
    elif regime == "label_drift":
        label_perms = _label_permutations(n_phases, rng)
        for phase, perm in enumerate(label_perms):
            phase_mask = phase_ids == phase
            labels[phase_mask] = perm[labels[phase_mask]]
        test_y = label_perms[final_phase][test_y]
        meta["description"] = (
            "Class-head meanings are permuted by phase; held-out uses final phase."
        )
    elif regime == "class_blocked":
        meta["description"] = (
            "Training stream is grouped into digit-class blocks within each epoch."
        )
    else:
        meta["description"] = "Stationary IID shuffled-epoch control."

    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return RegimeStream(
        observations=jnp.asarray(observations),
        targets=jnp.asarray(targets),
        labels=jnp.asarray(labels),
        x_test=test_x.astype(np.float32),
        y_test=test_y.astype(np.int32),
        meta=meta,
    )


def make_mlp(
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
) -> MultiHeadMLPLearner:
    """Architecture-matched non-perturbed MLP baseline."""
    return MultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )


def make_upgd(
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    perturbation_sigma: float,
) -> UPGDLearner:
    """Architecture-matched UPGD learner."""
    return UPGDLearner(
        n_heads=N_CLASSES,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
        perturbation_sigma=perturbation_sigma,
    )


def run_mlp_stream(
    learner: MultiHeadMLPLearner,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run MLP online and return final state plus per-step [mse, correct]."""
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        mse = jnp.mean((result.predictions - tgt) ** 2)
        correct = jnp.argmax(result.predictions) == label
        return result.state, jnp.stack([mse, correct.astype(jnp.float32)])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def run_upgd_stream(
    learner: UPGDLearner,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run UPGD online and return final state plus per-step [mse, correct]."""
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        mse = jnp.mean((result.predictions - tgt) ** 2)
        correct = jnp.argmax(result.predictions) == label
        return result.state, jnp.stack([mse, correct.astype(jnp.float32)])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def evaluate_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate final state on the held-out split."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def summarize_curve(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize prequential online metrics."""
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def stderr(values: np.ndarray) -> float:
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def paired_summary(records: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    """Summarize paired UPGD-vs-MLP differences for a lower-is-better metric."""
    mlp = np.asarray([r["methods"]["mlp"][metric] for r in records], dtype=np.float64)
    upgd = np.asarray([r["methods"]["upgd"][metric] for r in records], dtype=np.float64)
    diff = mlp - upgd
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "metric": metric,
        "mlp_mean": float(np.mean(mlp)),
        "mlp_stderr": stderr(mlp),
        "upgd_mean": float(np.mean(upgd)),
        "upgd_stderr": stderr(upgd),
        "mlp_minus_upgd_mean": float(np.mean(diff)),
        "mlp_minus_upgd_stderr": stderr(diff),
        "wins_for_upgd": int(np.sum(diff > 0.0)),
        "wins_for_mlp": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n_seeds": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def accuracy_summary(records: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    """Summarize paired UPGD-vs-MLP differences for a higher-is-better metric."""
    mlp = np.asarray([r["methods"]["mlp"][metric] for r in records], dtype=np.float64)
    upgd = np.asarray([r["methods"]["upgd"][metric] for r in records], dtype=np.float64)
    diff = upgd - mlp
    sd = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 0.0
    return {
        "metric": metric,
        "mlp_mean": float(np.mean(mlp)),
        "mlp_stderr": stderr(mlp),
        "upgd_mean": float(np.mean(upgd)),
        "upgd_stderr": stderr(upgd),
        "upgd_minus_mlp_mean": float(np.mean(diff)),
        "upgd_minus_mlp_stderr": stderr(diff),
        "wins_for_upgd": int(np.sum(diff > 0.0)),
        "wins_for_mlp": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n_seeds": int(diff.shape[0]),
        "cohens_d": float(np.mean(diff) / sd) if sd > 0.0 else 0.0,
    }


def regime_winners(results: dict[str, Any], metric: str) -> list[str]:
    winners: list[str] = []
    for regime, regime_result in results["regime_results"].items():
        row = regime_result["aggregate"][metric]
        diff_key = "mlp_minus_upgd_mean" if metric.endswith("mse") else "upgd_minus_mlp_mean"
        if row[diff_key] > 0.0:
            winners.append(regime)
    return winners


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    cfg = results["config"]
    lines = [
        "# Direction 4 Step 2 External Nonstationary Benchmark",
        "",
        "Dataset: `sklearn.datasets.load_digits` (bundled 8x8 handwritten digits; no network).",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} online training steps, "
            f"last-{cfg['final_window']} final window, "
            f"phase_length={cfg['phase_length']}."
        ),
        f"Regimes: {', '.join(cfg['regimes'])}.",
        "",
        "Methods are architecture-matched: same hidden sizes, step size, sparsity, "
        "layer norm, and ObGD bounding. UPGD adds utility-scaled perturbations.",
        "",
    ]

    for regime, regime_result in results["regime_results"].items():
        agg = regime_result["aggregate"]
        lines.extend([
            f"## Regime: {regime}",
            "",
            regime_result["regime_meta"]["description"],
            "",
            "| Metric | MLP | UPGD | Paired diff | UPGD wins | MLP wins |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for key in [
            "final_window_mse",
            "test_mse",
            "final_window_accuracy",
            "test_accuracy",
        ]:
            row = agg[key]
            diff_key = "mlp_minus_upgd_mean" if key.endswith("mse") else "upgd_minus_mlp_mean"
            lines.append(
                f"| {key} | "
                f"{row['mlp_mean']:.4f} +/- {row['mlp_stderr']:.4f} | "
                f"{row['upgd_mean']:.4f} +/- {row['upgd_stderr']:.4f} | "
                f"{row[diff_key]:+.4f} | "
                f"{row['wins_for_upgd']}/{row['n_seeds']} | "
                f"{row['wins_for_mlp']}/{row['n_seeds']} |"
            )
        lines.append("")

    final_acc_winners = regime_winners(results, "final_window_accuracy")
    test_acc_winners = regime_winners(results, "test_accuracy")
    lines.extend([
        "## Interpretation",
        "",
        "UPGD mean final-window accuracy beats MLP in: "
        + (", ".join(final_acc_winners) if final_acc_winners else "none")
        + ".",
        "UPGD mean held-out final-phase accuracy beats MLP in: "
        + (", ".join(test_acc_winners) if test_acc_winners else "none")
        + ".",
        "",
        "This weakens any broad universality claim if UPGD does not win across "
        "most regimes. Isolated wins are better read as conditional plasticity "
        "evidence on specific nonstationary streams, not as a general MLP "
        "replacement claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--phase-length", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-3)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.65)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=VALID_REGIMES,
        default=list(VALID_REGIMES),
        help="Regimes to run. Defaults to IID plus all nonstationary regimes.",
    )
    parser.add_argument(
        "--class-blocked",
        action="store_true",
        help="Backward-compatible alias that ensures the class_blocked regime is included.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.phase_length <= 0:
        raise ValueError("--phase-length must be positive")

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = list(dict.fromkeys(args.regimes))
    if args.class_blocked and "class_blocked" not in regimes:
        regimes.append("class_blocked")

    hidden_sizes = (args.hidden_size,)
    regime_records: dict[str, list[dict[str, Any]]] = {regime: [] for regime in regimes}
    regime_meta: dict[str, dict[str, Any]] = {}
    dataset_meta: dict[str, Any] | None = None

    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
            seed=seed,
            train_fraction=args.train_fraction,
        )

        for regime_idx, regime in enumerate(regimes):
            stream = make_regime_stream(
                regime=regime,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                steps=args.steps,
                seed=seed + 10_000 + 1_000 * regime_idx,
                phase_length=args.phase_length,
                mask_keep_fraction=args.mask_keep_fraction,
                noise_std=args.noise_std,
            )
            regime_meta.setdefault(regime, stream.meta)

            key = jr.key(seed + 100_000 * (regime_idx + 1))
            mlp_key, upgd_key = jr.split(key)
            mlp = make_mlp(hidden_sizes, args.step_size, args.sparsity)
            upgd = make_upgd(hidden_sizes, args.step_size, args.sparsity, args.perturbation_sigma)

            print(f"regime={regime} seed={seed}: running MLP")
            mlp_state, mlp_metrics = run_mlp_stream(
                mlp,
                mlp_key,
                stream.observations,
                stream.targets,
                stream.labels,
            )
            print(f"regime={regime} seed={seed}: running UPGD")
            upgd_state, upgd_metrics = run_upgd_stream(
                upgd,
                upgd_key,
                stream.observations,
                stream.targets,
                stream.labels,
            )

            mlp_summary = summarize_curve(mlp_metrics, args.final_window)
            mlp_summary.update(evaluate_classifier(mlp, mlp_state, stream.x_test, stream.y_test))
            upgd_summary = summarize_curve(upgd_metrics, args.final_window)
            upgd_summary.update(evaluate_classifier(upgd, upgd_state, stream.x_test, stream.y_test))

            regime_records[regime].append({
                "seed": seed,
                "methods": {
                    "mlp": mlp_summary,
                    "upgd": upgd_summary,
                },
            })
            print(
                f"regime={regime} seed={seed}: "
                f"final-window acc mlp={mlp_summary['final_window_accuracy']:.3f}, "
                f"upgd={upgd_summary['final_window_accuracy']:.3f}; "
                f"test acc mlp={mlp_summary['test_accuracy']:.3f}, "
                f"upgd={upgd_summary['test_accuracy']:.3f}"
            )

    regime_results: dict[str, dict[str, Any]] = {}
    for regime, records in regime_records.items():
        regime_results[regime] = {
            "regime_meta": regime_meta[regime],
            "records": records,
            "aggregate": {
                "online_mean_mse": paired_summary(records, "online_mean_mse"),
                "final_window_mse": paired_summary(records, "final_window_mse"),
                "test_mse": paired_summary(records, "test_mse"),
                "online_mean_accuracy": accuracy_summary(records, "online_mean_accuracy"),
                "final_window_accuracy": accuracy_summary(records, "final_window_accuracy"),
                "test_accuracy": accuracy_summary(records, "test_accuracy"),
            },
        }

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
            "perturbation_sigma": args.perturbation_sigma,
            "mask_keep_fraction": args.mask_keep_fraction,
            "noise_std": args.noise_std,
            "regimes": regimes,
        },
        "dataset": dataset_meta,
        "regime_results": regime_results,
        "wall_clock_s": time.time() - t0,
        "evidence_level": "real_external_nonstationary_dataset_evidence",
    }

    json_path = output_dir / "digits_nonstationary_results.json"
    md_path = output_dir / "digits_nonstationary_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

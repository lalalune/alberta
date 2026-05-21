#!/usr/bin/env python3
"""D15: groupwise preconditioned basis learner for Step 2.

The D14 exact basis specialists showed useful non-MLP mechanisms but naive
concatenation caused interference.  This runner keeps one summed prediction and
one residual, while each basis family receives its own normalized LMS
preconditioner.  Every family updates at every timestep; no route or expert is
selected.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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

from step2_expert_mixture import DIGITS_REGIMES, N_DIGIT_CLASSES  # noqa: E402

from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    aggregate_records,
    evaluate_mlp_classifier,
    expand_dataset_names,
    make_dataset,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d15_groupwise_basis_lms")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d15_groupwise_basis_lms.md"
)


@dataclass(frozen=True)
class GroupwiseConfig:
    """Configuration for a block-preconditioned additive basis learner."""

    name: str
    input_clip: float
    poly_max_dim: int
    fourier_max_dim: int
    fourier_frequencies: tuple[float, ...]
    tanh_width: int
    tanh_weight_scale: float
    poly_step_size: float
    fourier_step_size: float
    tanh_step_size: float
    poly_scale: float
    fourier_scale: float
    tanh_scale: float
    include_poly: bool
    include_fourier: bool
    include_tanh: bool
    weight_decay: float = 1.0
    simplex_weight_decay: float | None = None


@dataclass
class GroupwiseState:
    """Mutable state for the groupwise learner."""

    poly_weights: np.ndarray
    fourier_weights: np.ndarray
    tanh_weights_out: np.ndarray
    tanh_weights: np.ndarray
    tanh_biases: np.ndarray
    steps: int
    finite_failures: int
    mean_poly_norm: float
    mean_fourier_norm: float
    mean_tanh_norm: float


class GroupwiseBasisLMS:
    """One additive predictor with blockwise normalized LMS updates."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: GroupwiseConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        rng = np.random.default_rng(seed)
        scale = config.tanh_weight_scale / math.sqrt(max(feature_dim, 1))
        self.tanh_weights = rng.normal(
            0.0,
            scale,
            size=(config.tanh_width, feature_dim),
        ).astype(np.float64)
        self.tanh_biases = rng.uniform(
            -config.tanh_weight_scale,
            config.tanh_weight_scale,
            size=config.tanh_width,
        ).astype(np.float64)
        self.poly_dim = self._poly_dim()
        self.fourier_dim = self._fourier_dim()
        self.tanh_dim = 1 + config.tanh_width

    def _poly_dim(self) -> int:
        d = min(self.feature_dim, self.config.poly_max_dim)
        return 1 + d + (d * (d + 1)) // 2 + (d * (d + 1) * (d + 2)) // 6

    def _fourier_dim(self) -> int:
        d = min(self.feature_dim, self.config.fourier_max_dim)
        return 1 + d + 2 * d * len(self.config.fourier_frequencies)

    @property
    def total_dim(self) -> int:
        """Total represented feature count across active groups."""
        dim = 0
        if self.config.include_poly:
            dim += self.poly_dim
        if self.config.include_fourier:
            dim += self.fourier_dim
        if self.config.include_tanh:
            dim += self.tanh_dim
        return dim

    def init(self) -> GroupwiseState:
        """Return initial state."""
        return GroupwiseState(
            poly_weights=np.zeros((self.poly_dim, self.n_heads), dtype=np.float64),
            fourier_weights=np.zeros(
                (self.fourier_dim, self.n_heads),
                dtype=np.float64,
            ),
            tanh_weights_out=np.zeros(
                (self.tanh_dim, self.n_heads),
                dtype=np.float64,
            ),
            tanh_weights=self.tanh_weights.copy(),
            tanh_biases=self.tanh_biases.copy(),
            steps=0,
            finite_failures=0,
            mean_poly_norm=0.0,
            mean_fourier_norm=0.0,
            mean_tanh_norm=0.0,
        )

    def _clip(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        return np.clip(x, -self.config.input_clip, self.config.input_clip)

    def _poly_features(self, x: np.ndarray) -> np.ndarray:
        p = x[: self.config.poly_max_dim]
        terms: list[float] = [1.0]
        terms.extend(float(value) for value in p)
        terms.extend(
            float(p[i] * p[j])
            for i in range(p.shape[0])
            for j in range(i, p.shape[0])
        )
        terms.extend(
            float(p[i] * p[j] * p[k])
            for i in range(p.shape[0])
            for j in range(i, p.shape[0])
            for k in range(j, p.shape[0])
        )
        return self.config.poly_scale * np.asarray(terms, dtype=np.float64)

    def _fourier_features(self, x: np.ndarray) -> np.ndarray:
        f = x[: self.config.fourier_max_dim]
        terms: list[np.ndarray] = [np.ones(1, dtype=np.float64), f]
        for frequency in self.config.fourier_frequencies:
            phase = float(frequency) * f
            terms.append(np.sin(phase))
            terms.append(np.cos(phase))
        return self.config.fourier_scale * np.concatenate(terms, axis=0)

    def _tanh_features(self, state: GroupwiseState, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(state.tanh_weights @ x + state.tanh_biases)
        return self.config.tanh_scale * np.concatenate(
            [np.ones(1, dtype=np.float64), hidden],
            axis=0,
        )

    def _group_predictions(
        self,
        state: GroupwiseState,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self._clip(observation)
        poly = self._poly_features(x)
        fourier = self._fourier_features(x)
        tanh = self._tanh_features(state, x)
        prediction = np.zeros(self.n_heads, dtype=np.float64)
        if self.config.include_poly:
            prediction += poly @ state.poly_weights
        if self.config.include_fourier:
            prediction += fourier @ state.fourier_weights
        if self.config.include_tanh:
            prediction += tanh @ state.tanh_weights_out
        return prediction, poly, fourier, tanh

    @staticmethod
    def _one_hot_target(target: np.ndarray) -> bool:
        """Return whether a target is an observed one-hot simplex vector."""
        active = ~np.isnan(target)
        if target.shape[0] <= 1 or not bool(np.all(active)):
            return False
        values = np.asarray(target, dtype=np.float64)
        return (
            abs(float(np.sum(values)) - 1.0) <= 1e-4
            and abs(float(np.max(values)) - 1.0) <= 1e-4
            and float(np.min(values)) >= -1e-4
            and float(np.max(np.minimum(np.abs(values), np.abs(values - 1.0))))
            <= 1e-4
        )

    def predict(self, state: GroupwiseState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads."""
        prediction, _, _, _ = self._group_predictions(state, observation)
        return prediction

    def step(
        self,
        state: GroupwiseState,
        observation: np.ndarray,
        target: np.ndarray,
        decay_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict once, then update every active basis family from one residual."""
        prediction, poly, fourier, tanh = self._group_predictions(state, observation)
        active = ~np.isnan(target)
        errors = np.where(active, target - prediction, 0.0)
        weight_decay = self.config.weight_decay
        decay_source = target if decay_target is None else decay_target
        if (
            self.config.simplex_weight_decay is not None
            and self._one_hot_target(decay_source)
        ):
            weight_decay = self.config.simplex_weight_decay
        if weight_decay < 1.0:
            state.poly_weights *= weight_decay
            state.fourier_weights *= weight_decay
            state.tanh_weights_out *= weight_decay
        if self.config.include_poly:
            state.poly_weights += (
                self.config.poly_step_size
                * np.outer(poly, errors)
                / (1.0 + float(poly @ poly))
            )
        if self.config.include_fourier:
            state.fourier_weights += (
                self.config.fourier_step_size
                * np.outer(fourier, errors)
                / (1.0 + float(fourier @ fourier))
            )
        if self.config.include_tanh:
            state.tanh_weights_out += (
                self.config.tanh_step_size
                * np.outer(tanh, errors)
                / (1.0 + float(tanh @ tanh))
            )
        finite = (
            np.all(np.isfinite(state.poly_weights))
            and np.all(np.isfinite(state.fourier_weights))
            and np.all(np.isfinite(state.tanh_weights_out))
        )
        if not finite:
            state.finite_failures += 1
            state.poly_weights = np.nan_to_num(state.poly_weights, copy=False)
            state.fourier_weights = np.nan_to_num(state.fourier_weights, copy=False)
            state.tanh_weights_out = np.nan_to_num(state.tanh_weights_out, copy=False)
        state.steps += 1
        decay = 0.99
        poly_norm = float(np.linalg.norm(poly))
        fourier_norm = float(np.linalg.norm(fourier))
        tanh_norm = float(np.linalg.norm(tanh))
        state.mean_poly_norm = decay * state.mean_poly_norm + (1.0 - decay) * poly_norm
        state.mean_fourier_norm = decay * state.mean_fourier_norm + (
            1.0 - decay
        ) * fourier_norm
        state.mean_tanh_norm = decay * state.mean_tanh_norm + (1.0 - decay) * tanh_norm
        return prediction, {
            "poly_norm": poly_norm,
            "fourier_norm": fourier_norm,
            "tanh_norm": tanh_norm,
            "finite_failures": float(state.finite_failures),
        }


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_groupwise_stream(
    observations: Any,
    targets: Any,
    config: GroupwiseConfig,
    seed: int,
) -> tuple[GroupwiseBasisLMS, GroupwiseState, np.ndarray]:
    """Run one groupwise learner."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = GroupwiseBasisLMS(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 6), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["poly_norm"]
        metrics[idx, 3] = diagnostics["fourier_norm"]
        metrics[idx, 4] = diagnostics["tanh_norm"]
        metrics[idx, 5] = diagnostics["finite_failures"]
    return learner, state, metrics


def evaluate_groupwise_classifier(
    learner: GroupwiseBasisLMS,
    state: GroupwiseState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate final classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def groupwise_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one groupwise run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_poly_norm": float(np.mean(metrics[:, 2])),
            "mean_fourier_norm": float(np.mean(metrics[:, 3])),
            "mean_tanh_norm": float(np.mean(metrics[:, 4])),
            "finite_failures": float(metrics[-1, 5]),
        }
    )
    return entry


def make_configs(args: argparse.Namespace) -> list[GroupwiseConfig]:
    """Return named groupwise configurations."""
    freqs = tuple(float(item) for item in args.fourier_frequencies)
    common = {
        "input_clip": args.input_clip,
        "poly_max_dim": args.poly_max_dim,
        "fourier_max_dim": args.fourier_max_dim,
        "fourier_frequencies": freqs,
        "tanh_width": args.tanh_width,
        "tanh_weight_scale": args.tanh_weight_scale,
    }
    configs = {
        "canonical": GroupwiseConfig(
            name="canonical",
            **common,
            poly_step_size=0.5,
            fourier_step_size=0.3,
            tanh_step_size=0.4,
            poly_scale=1.0,
            fourier_scale=1.0,
            tanh_scale=1.0,
            include_poly=True,
            include_fourier=True,
            include_tanh=True,
        ),
        "no_poly": GroupwiseConfig(
            name="no_poly",
            **common,
            poly_step_size=0.0,
            fourier_step_size=0.3,
            tanh_step_size=0.4,
            poly_scale=1.0,
            fourier_scale=1.0,
            tanh_scale=1.0,
            include_poly=False,
            include_fourier=True,
            include_tanh=True,
        ),
        "slow_poly": GroupwiseConfig(
            name="slow_poly",
            **common,
            poly_step_size=0.15,
            fourier_step_size=0.3,
            tanh_step_size=0.4,
            poly_scale=1.0,
            fourier_scale=1.0,
            tanh_scale=1.0,
            include_poly=True,
            include_fourier=True,
            include_tanh=True,
        ),
        "tanh_fourier_fast": GroupwiseConfig(
            name="tanh_fourier_fast",
            **common,
            poly_step_size=0.3,
            fourier_step_size=0.5,
            tanh_step_size=0.5,
            poly_scale=1.0,
            fourier_scale=1.0,
            tanh_scale=1.0,
            include_poly=True,
            include_fourier=True,
            include_tanh=True,
        ),
    }
    if args.configs == "all":
        return list(configs.values())
    selected: list[GroupwiseConfig] = []
    for raw in args.configs.split(","):
        name = raw.strip()
        if name:
            selected.append(configs[name])
    return selected


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    lines = [
        "# D15 Groupwise Basis LMS Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "Each candidate is one additive predictor with blockwise normalized LMS. "
        "Every included basis family updates from the same residual at every "
        "step; there is no prediction router or expert selection.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Feature dim | Runtime s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in dataset_agg.items():
            if method == "comparisons":
                continue
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} | "
                f"{metric_cell(row, 'feature_dim')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-groupwise-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-groupwise counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-groupwise-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    configs: list[GroupwiseConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all methods for one dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    for method in MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_mlp_classifier(mlp, state, x_test, y_test))

    for offset, config in enumerate(configs):
        method = f"groupwise_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_groupwise_stream(
            observations,
            targets,
            config,
            seed=seed + 100_000 + offset,
        )
        methods[method] = groupwise_summary(metrics, args.final_window, labels)
        methods[method].update(
            {
                "runtime_s": float(time.time() - t0),
                "feature_dim": float(learner.total_dim),
                "finite_failures": float(state.finite_failures),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_groupwise_classifier(learner, state, x_test, y_test)
            )

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "methods": methods,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="synthetic")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument("--input-clip", type=float, default=3.0)
    parser.add_argument("--poly-max-dim", type=int, default=8)
    parser.add_argument("--fourier-max-dim", type=int, default=8)
    parser.add_argument(
        "--fourier-frequencies",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0, 4.0),
    )
    parser.add_argument("--tanh-width", type=int, default=256)
    parser.add_argument("--tanh-weight-scale", type=float, default=1.0)
    parser.add_argument("--configs", default="canonical")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.input_clip <= 0.0:
        raise ValueError("--input-clip must be positive")
    if args.poly_max_dim <= 0 or args.fourier_max_dim <= 0:
        raise ValueError("basis max dimensions must be positive")
    if args.tanh_width <= 0:
        raise ValueError("--tanh-width must be positive")
    if args.tanh_weight_scale <= 0.0:
        raise ValueError("--tanh-weight-scale must be positive")


def main() -> None:
    """Run D15 experiments."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.configs = "canonical"
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_configs(args)
    candidate_methods = tuple(f"groupwise_{config.name}" for config in configs)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(dataset_name, seed, configs, args)
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta
    results = {
        "config": {
            "datasets": datasets,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "feature_dim": args.feature_dim,
            "noise_std": args.noise_std,
            "rare_period": args.rare_period,
            "train_fraction": args.train_fraction,
            "phase_length": args.phase_length,
            "mask_keep_fraction": args.mask_keep_fraction,
            "mask_noise_std": args.mask_noise_std,
            "mlp_step_size": args.mlp_step_size,
            "mlp_sparsity": args.mlp_sparsity,
            "input_clip": args.input_clip,
            "poly_max_dim": args.poly_max_dim,
            "fourier_max_dim": args.fourier_max_dim,
            "fourier_frequencies": list(args.fourier_frequencies),
            "tanh_width": args.tanh_width,
            "tanh_weight_scale": args.tanh_weight_scale,
            "configs": args.configs,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "single_predictor_groupwise_basis_lms_probe",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    if args.note_path:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

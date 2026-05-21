#!/usr/bin/env python3
"""D14: unified fixed-basis LMS learner for Step 2.

This probe turns the strongest non-router observations from prior runs into a
single learner: one linear readout over a concatenated basis bank containing
raw inputs, low-dimensional polynomial chaos, Fourier coordinates, and fixed
random tanh observables.  There is no prediction routing and no MLP trunk; all
features share one normalized LMS update at every timestep.
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

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d14_unified_basis_lms")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d14_unified_basis_lms.md")


@dataclass(frozen=True)
class BasisConfig:
    """Configuration for one unified basis learner."""

    name: str
    step_size: float
    input_clip: float
    poly_max_dim: int
    poly_degree: int
    fourier_max_dim: int
    fourier_frequencies: tuple[float, ...]
    tanh_width: int
    tanh_weight_scale: float
    raw_scale: float
    poly_scale: float
    fourier_scale: float
    tanh_scale: float


@dataclass
class BasisState:
    """Mutable state for one basis learner."""

    weights: np.ndarray
    tanh_weights: np.ndarray
    tanh_biases: np.ndarray
    steps: int
    finite_failures: int
    mean_feature_norm: float


class UnifiedBasisLMS:
    """Single normalized LMS readout over complementary fixed bases."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: BasisConfig,
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
        self.feature_dim_out = self._feature_dim()

    def _feature_dim(self) -> int:
        poly_dim = min(self.feature_dim, self.config.poly_max_dim)
        fourier_dim = min(self.feature_dim, self.config.fourier_max_dim)
        dim = 1 + self.feature_dim
        if self.config.poly_degree >= 2:
            dim += (poly_dim * (poly_dim + 1)) // 2
        if self.config.poly_degree >= 3:
            dim += (poly_dim * (poly_dim + 1) * (poly_dim + 2)) // 6
        dim += 2 * fourier_dim * len(self.config.fourier_frequencies)
        dim += self.config.tanh_width
        return dim

    def init(self) -> BasisState:
        """Return initial learner state."""
        return BasisState(
            weights=np.zeros((self.feature_dim_out, self.n_heads), dtype=np.float64),
            tanh_weights=self.tanh_weights.copy(),
            tanh_biases=self.tanh_biases.copy(),
            steps=0,
            finite_failures=0,
            mean_feature_norm=0.0,
        )

    def _features(self, state: BasisState, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        x = np.clip(x, -self.config.input_clip, self.config.input_clip)
        parts: list[np.ndarray] = [
            np.ones(1, dtype=np.float64),
            self.config.raw_scale * x / math.sqrt(max(self.feature_dim, 1)),
        ]

        p = x[: self.config.poly_max_dim]
        poly_terms: list[float] = []
        if self.config.poly_degree >= 2:
            poly_terms.extend(
                float(p[i] * p[j])
                for i in range(p.shape[0])
                for j in range(i, p.shape[0])
            )
        if self.config.poly_degree >= 3:
            poly_terms.extend(
                float(p[i] * p[j] * p[k])
                for i in range(p.shape[0])
                for j in range(i, p.shape[0])
                for k in range(j, p.shape[0])
            )
        if poly_terms:
            poly = np.asarray(poly_terms, dtype=np.float64)
            parts.append(
                self.config.poly_scale
                * poly
                / math.sqrt(max(poly.shape[0], 1))
            )

        f = x[: self.config.fourier_max_dim]
        fourier_terms: list[np.ndarray] = []
        for frequency in self.config.fourier_frequencies:
            phase = float(frequency) * f
            fourier_terms.append(np.sin(phase))
            fourier_terms.append(np.cos(phase))
        if fourier_terms:
            fourier = np.concatenate(fourier_terms, axis=0)
            parts.append(
                self.config.fourier_scale
                * fourier
                / math.sqrt(max(fourier.shape[0], 1))
            )

        if self.config.tanh_width:
            tanh = np.tanh(state.tanh_weights @ x + state.tanh_biases)
            parts.append(
                self.config.tanh_scale
                * tanh
                / math.sqrt(max(self.config.tanh_width, 1))
            )
        phi = np.concatenate(parts, axis=0)
        return np.asarray(np.nan_to_num(phi, copy=False), dtype=np.float64)

    def predict(self, state: BasisState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads."""
        return np.asarray(self._features(state, observation) @ state.weights)

    def step(
        self,
        state: BasisState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict then update active target heads."""
        phi = self._features(state, observation)
        prediction = np.asarray(phi @ state.weights)
        active = ~np.isnan(target)
        errors = np.where(active, target - prediction, 0.0)
        normalizer = 1.0 + float(phi @ phi)
        state.weights += self.config.step_size * np.outer(phi, errors) / normalizer
        if not np.all(np.isfinite(state.weights)):
            state.finite_failures += 1
            state.weights = np.nan_to_num(state.weights, copy=False)
        state.steps += 1
        norm = float(np.linalg.norm(phi))
        state.mean_feature_norm = 0.99 * state.mean_feature_norm + 0.01 * norm
        return prediction, {
            "feature_norm": norm,
            "finite_failures": float(state.finite_failures),
        }


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_basis_stream(
    observations: Any,
    targets: Any,
    config: BasisConfig,
    seed: int,
) -> tuple[UnifiedBasisLMS, BasisState, np.ndarray]:
    """Run one unified basis learner."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = UnifiedBasisLMS(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 4), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["feature_norm"]
        metrics[idx, 3] = diagnostics["finite_failures"]
    return learner, state, metrics


def evaluate_basis_classifier(
    learner: UnifiedBasisLMS,
    state: BasisState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def basis_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one basis run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_feature_norm": float(np.mean(metrics[:, 2])),
            "finite_failures": float(metrics[-1, 3]),
        }
    )
    return entry


def make_basis_configs(args: argparse.Namespace) -> list[BasisConfig]:
    """Return named basis configurations."""
    freqs = tuple(float(item) for item in args.fourier_frequencies)
    configs = {
        "tanh_exact": BasisConfig(
            name="tanh_exact",
            step_size=args.step_size,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=0.0,
            poly_scale=0.0,
            fourier_scale=0.0,
            tanh_scale=math.sqrt(max(args.tanh_width, 1)),
        ),
        "fourier_exact": BasisConfig(
            name="fourier_exact",
            step_size=0.3,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=0.0,
            poly_scale=0.0,
            fourier_scale=math.sqrt(
                max(2 * args.fourier_max_dim * len(freqs), 1)
            ),
            tanh_scale=0.0,
        ),
        "poly_exact": BasisConfig(
            name="poly_exact",
            step_size=0.5,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=math.sqrt(max(args.poly_max_dim, 1)),
            poly_scale=math.sqrt(
                max(
                    (args.poly_max_dim * (args.poly_max_dim + 1)) // 2
                    + (
                        args.poly_max_dim
                        * (args.poly_max_dim + 1)
                        * (args.poly_max_dim + 2)
                    )
                    // 6,
                    1,
                )
            ),
            fourier_scale=0.0,
            tanh_scale=0.0,
        ),
        "exact_union": BasisConfig(
            name="exact_union",
            step_size=args.step_size,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=math.sqrt(max(args.poly_max_dim, 1)),
            poly_scale=math.sqrt(
                max(
                    (args.poly_max_dim * (args.poly_max_dim + 1)) // 2
                    + (
                        args.poly_max_dim
                        * (args.poly_max_dim + 1)
                        * (args.poly_max_dim + 2)
                    )
                    // 6,
                    1,
                )
            ),
            fourier_scale=math.sqrt(
                max(2 * args.fourier_max_dim * len(freqs), 1)
            ),
            tanh_scale=math.sqrt(max(args.tanh_width, 1)),
        ),
        "balanced": BasisConfig(
            name="balanced",
            step_size=args.step_size,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=1.0,
            poly_scale=1.0,
            fourier_scale=1.0,
            tanh_scale=1.0,
        ),
        "tanh_heavy": BasisConfig(
            name="tanh_heavy",
            step_size=args.step_size,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=0.75,
            poly_scale=0.5,
            fourier_scale=0.5,
            tanh_scale=2.0,
        ),
        "structure_heavy": BasisConfig(
            name="structure_heavy",
            step_size=args.step_size,
            input_clip=args.input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=1.0,
            poly_scale=2.0,
            fourier_scale=2.0,
            tanh_scale=0.75,
        ),
    }
    if args.configs == "all":
        return list(configs.values())
    selected: list[BasisConfig] = []
    for raw in args.configs.split(","):
        name = raw.strip()
        if name:
            selected.append(configs[name])
    return selected


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell for Markdown."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_d14_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary."""
    cfg = results["config"]
    lines = [
        "# D14 Unified Basis LMS Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "This is one normalized LMS predictor over a concatenated basis bank. "
        "It is not a route selector and does not include an MLP baseline inside "
        "the candidate prediction.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Basis dim | Runtime s |",
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
                f"{metric_cell(row, 'basis_dim')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-basis-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-basis counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-basis-vs-best-MLP diff: "
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
    configs: list[BasisConfig],
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
        method = f"basis_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_basis_stream(
            observations,
            targets,
            config,
            seed=seed + 90_000 + offset,
        )
        methods[method] = basis_summary(metrics, args.final_window, labels)
        methods[method].update(
            {
                "runtime_s": float(time.time() - t0),
                "basis_dim": float(learner.feature_dim_out),
                "mean_feature_norm_ema": float(state.mean_feature_norm),
                "finite_failures": float(state.finite_failures),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_basis_classifier(learner, state, x_test, y_test))

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
    parser.add_argument("--step-size", type=float, default=0.4)
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
    parser.add_argument("--configs", default="balanced")
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
    if args.step_size < 0.0:
        raise ValueError("--step-size must be non-negative")
    if args.input_clip <= 0.0:
        raise ValueError("--input-clip must be positive")
    if args.poly_max_dim <= 0 or args.fourier_max_dim <= 0:
        raise ValueError("basis max dimensions must be positive")
    if args.tanh_width <= 0:
        raise ValueError("--tanh-width must be positive")
    if args.tanh_weight_scale <= 0.0:
        raise ValueError("--tanh-weight-scale must be positive")


def main() -> None:
    """Run the D14 sweep."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.configs = "balanced"
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_basis_configs(args)
    candidate_methods = tuple(f"basis_{config.name}" for config in configs)
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
            "step_size": args.step_size,
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
        "evidence_level": "single_predictor_unified_fixed_basis_lms_probe",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_d14_summary(md_path, results)
    if args.note_path:
        write_d14_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

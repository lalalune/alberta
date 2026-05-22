#!/usr/bin/env python3
"""D13: Koopman/reservoir random-feature learner for Step 2.

This probe tests a single non-MLP predictor: fixed nonlinear observables plus
an optional random recurrent reservoir, with a multi-head recursive least
squares readout.  The mechanism is closer to Koopman/reservoir computing than
to a trained neural trunk: representation dynamics are fixed after
initialization; only the linear readout adapts online.
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

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d13_koopman_reservoir_rls")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d13_koopman_reservoir_rls.md"
)


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for one fixed-observable RLS learner."""

    name: str
    reservoir_size: int
    tanh_features: int
    fourier_features: int
    product_features: int
    chebyshev_degree: int
    input_scale: float
    recurrent_scale: float
    leak: float
    feature_scale: float
    input_clip: float
    rho: float
    rls_delta: float
    include_raw: bool
    include_time: bool


@dataclass
class ReservoirState:
    """Mutable online state for the fixed-observable learner."""

    reservoir: np.ndarray
    weights: np.ndarray
    covariance: np.ndarray
    steps: int
    finite_failures: int
    mean_feature_norm: float
    mean_gain_norm: float


class KoopmanReservoirRLS:
    """Fixed Koopman/reservoir observables with an online RLS readout."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: ReservoirConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        rng = np.random.default_rng(seed)
        self.input_weights = rng.normal(
            0.0,
            config.input_scale / math.sqrt(max(feature_dim, 1)),
            size=(config.reservoir_size, feature_dim),
        )
        recurrent = rng.normal(
            0.0,
            1.0 / math.sqrt(max(config.reservoir_size, 1)),
            size=(config.reservoir_size, config.reservoir_size),
        )
        if config.reservoir_size > 0:
            radius = max(abs(np.linalg.eigvals(recurrent)).max().real, 1e-6)
            recurrent = recurrent * (config.recurrent_scale / radius)
        self.recurrent_weights = recurrent
        self.reservoir_bias = rng.normal(0.0, 0.1, size=config.reservoir_size)
        self.tanh_weights = rng.normal(
            0.0,
            config.feature_scale / math.sqrt(max(feature_dim, 1)),
            size=(config.tanh_features, feature_dim),
        )
        self.tanh_bias = rng.uniform(-math.pi, math.pi, size=config.tanh_features)
        self.fourier_weights = rng.normal(
            0.0,
            config.feature_scale / math.sqrt(max(feature_dim, 1)),
            size=(config.fourier_features, feature_dim),
        )
        self.fourier_bias = rng.uniform(-math.pi, math.pi, size=config.fourier_features)
        self.product_indices = rng.integers(
            0,
            max(feature_dim, 1),
            size=(config.product_features, 3),
            endpoint=False,
        )
        self.readout_dim = self._feature_dim()

    def _feature_dim(self) -> int:
        dim = 1
        if self.config.include_raw:
            dim += self.feature_dim
        if self.config.include_time:
            dim += 4
        dim += self.config.reservoir_size
        dim += self.config.tanh_features
        dim += 2 * self.config.fourier_features
        dim += self.config.product_features
        if self.config.chebyshev_degree > 1:
            dim += self.feature_dim * (self.config.chebyshev_degree - 1)
        return dim

    def init(self) -> ReservoirState:
        """Return the initial online state."""
        return ReservoirState(
            reservoir=np.zeros(self.config.reservoir_size, dtype=np.float64),
            weights=np.zeros((self.readout_dim, self.n_heads), dtype=np.float64),
            covariance=np.eye(self.readout_dim, dtype=np.float64)
            * self.config.rls_delta,
            steps=0,
            finite_failures=0,
            mean_feature_norm=0.0,
            mean_gain_norm=0.0,
        )

    def _transform_input(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        if self.config.input_clip > 0.0:
            x = np.clip(x, -self.config.input_clip, self.config.input_clip)
        return x

    def _advance_reservoir(self, state: ReservoirState, x: np.ndarray) -> None:
        if self.config.reservoir_size == 0:
            return
        drive = (
            self.input_weights @ x
            + self.recurrent_weights @ state.reservoir
            + self.reservoir_bias
        )
        candidate = np.tanh(drive)
        leak = float(np.clip(self.config.leak, 0.0, 1.0))
        state.reservoir = (1.0 - leak) * state.reservoir + leak * candidate

    def _features(self, state: ReservoirState, x: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = [np.ones(1, dtype=np.float64)]
        if self.config.include_raw:
            parts.append(x / math.sqrt(max(self.feature_dim, 1)))
        if self.config.include_time:
            phase = 2.0 * math.pi * (state.steps % 512) / 512.0
            slow_phase = 2.0 * math.pi * (state.steps % 4096) / 4096.0
            parts.append(
                np.asarray(
                    [
                        math.sin(phase),
                        math.cos(phase),
                        math.sin(slow_phase),
                        math.cos(slow_phase),
                    ],
                    dtype=np.float64,
                )
            )
        if self.config.reservoir_size:
            parts.append(
                state.reservoir / math.sqrt(max(self.config.reservoir_size, 1))
            )
        if self.config.tanh_features:
            tanh = np.tanh(self.tanh_weights @ x + self.tanh_bias)
            parts.append(tanh / math.sqrt(self.config.tanh_features))
        if self.config.fourier_features:
            phase = self.fourier_weights @ x + self.fourier_bias
            scale = math.sqrt(2.0 * self.config.fourier_features)
            parts.append(np.sin(phase) / scale)
            parts.append(np.cos(phase) / scale)
        if self.config.product_features:
            idx = self.product_indices
            products = x[idx[:, 0]] * x[idx[:, 1]] * x[idx[:, 2]]
            parts.append(products / math.sqrt(self.config.product_features))
        if self.config.chebyshev_degree > 1:
            clipped = np.clip(x / max(self.config.input_clip, 1.0), -1.0, 1.0)
            cheb_parts = []
            t_prev = np.ones_like(clipped)
            t_curr = clipped
            for _degree in range(2, self.config.chebyshev_degree + 1):
                t_next = 2.0 * clipped * t_curr - t_prev
                cheb_parts.append(t_next)
                t_prev, t_curr = t_curr, t_next
            parts.append(
                np.concatenate(cheb_parts, axis=0)
                / math.sqrt(max(len(cheb_parts) * self.feature_dim, 1))
            )
        phi = np.concatenate(parts, axis=0)
        return np.asarray(np.nan_to_num(phi, copy=False), dtype=np.float64)

    def predict_with_features(
        self,
        state: ReservoirState,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance observation-only state and return prediction/features."""
        x = self._transform_input(observation)
        self._advance_reservoir(state, x)
        phi = self._features(state, x)
        return np.asarray(phi @ state.weights), phi

    def step(
        self,
        state: ReservoirState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict before target update, then update the RLS readout."""
        prediction, phi = self.predict_with_features(state, observation)
        active = ~np.isnan(target)
        errors = np.where(active, target - prediction, 0.0)
        p_phi = state.covariance @ phi
        denom = self.config.rho + float(phi @ p_phi)
        gain_norm = 0.0
        if denom <= 1e-12 or not np.isfinite(denom):
            state.finite_failures += 1
        else:
            gain = p_phi / denom
            state.weights += np.outer(gain, errors)
            next_cov = (
                state.covariance - np.outer(gain, phi @ state.covariance)
            ) / self.config.rho
            state.covariance = 0.5 * (next_cov + next_cov.T)
            gain_norm = float(np.linalg.norm(gain))
        if not np.all(np.isfinite(state.weights)):
            state.finite_failures += 1
            state.weights = np.nan_to_num(state.weights, copy=False)
        state.steps += 1
        decay = 0.99
        feature_norm = float(np.linalg.norm(phi))
        state.mean_feature_norm = decay * state.mean_feature_norm + (
            1.0 - decay
        ) * feature_norm
        state.mean_gain_norm = decay * state.mean_gain_norm + (1.0 - decay) * gain_norm
        return prediction, {
            "feature_norm": feature_norm,
            "gain_norm": gain_norm,
            "finite_failures": float(state.finite_failures),
        }


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_reservoir_stream(
    observations: Any,
    targets: Any,
    config: ReservoirConfig,
    seed: int,
) -> tuple[KoopmanReservoirRLS, ReservoirState, np.ndarray]:
    """Run one reservoir configuration on a materialized stream."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = KoopmanReservoirRLS(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 5), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["feature_norm"]
        metrics[idx, 3] = diagnostics["gain_norm"]
        metrics[idx, 4] = diagnostics["finite_failures"]
    return learner, state, metrics


def evaluate_reservoir_classifier(
    learner: KoopmanReservoirRLS,
    state: ReservoirState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final readout on held-out digits with frozen reservoir state."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    saved_reservoir = state.reservoir.copy()
    saved_steps = state.steps
    preds = []
    for obs in x_test.astype(np.float64):
        pred, _ = learner.predict_with_features(state, obs)
        preds.append(pred)
    state.reservoir = saved_reservoir
    state.steps = saved_steps
    pred_arr = np.stack(preds, axis=0)
    return {
        "test_mse": float(np.mean((pred_arr - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(pred_arr, axis=1) == y_test)),
    }


def reservoir_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one reservoir run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_feature_norm": float(np.mean(metrics[:, 2])),
            "mean_gain_norm": float(np.mean(metrics[:, 3])),
            "finite_failures": float(metrics[-1, 4]),
        }
    )
    return entry


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell for Markdown."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_d13_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown summary for the reservoir probe."""
    cfg = results["config"]
    lines = [
        "# D13 Koopman/Reservoir RLS Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "This is a single-predictor test. The candidate methods use fixed "
        "nonlinear observables and an online RLS readout; they do not route "
        "over MLP predictions or train an MLP trunk. Positive candidate-vs-MLP "
        "paired differences favor the reservoir learner.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Readout dim | Runtime s |",
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
                f"{metric_cell(row, 'readout_dim')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-candidate counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Bar",
            "",
            "A positive row here means the fixed-observable mechanism has "
            "headroom on that benchmark. A Step 2 closure claim still requires "
            "one canonical configuration to beat the best fair MLP across the "
            "full benchmark set without per-dataset selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_reservoir_configs(args: argparse.Namespace) -> list[ReservoirConfig]:
    """Return named learner configurations."""
    configs: dict[str, ReservoirConfig] = {
        "tanh256": ReservoirConfig(
            name="tanh256",
            reservoir_size=0,
            tanh_features=256,
            fourier_features=0,
            product_features=0,
            chebyshev_degree=1,
            input_scale=0.7,
            recurrent_scale=0.0,
            leak=1.0,
            feature_scale=0.7,
            input_clip=args.input_clip,
            rho=args.rho,
            rls_delta=args.rls_delta,
            include_raw=True,
            include_time=False,
        ),
        "tanh512": ReservoirConfig(
            name="tanh512",
            reservoir_size=0,
            tanh_features=512,
            fourier_features=0,
            product_features=0,
            chebyshev_degree=1,
            input_scale=0.7,
            recurrent_scale=0.0,
            leak=1.0,
            feature_scale=0.7,
            input_clip=args.input_clip,
            rho=args.rho,
            rls_delta=args.rls_delta,
            include_raw=True,
            include_time=False,
        ),
        "mixed384": ReservoirConfig(
            name="mixed384",
            reservoir_size=96,
            tanh_features=192,
            fourier_features=48,
            product_features=96,
            chebyshev_degree=3,
            input_scale=0.8,
            recurrent_scale=0.85,
            leak=0.35,
            feature_scale=0.8,
            input_clip=args.input_clip,
            rho=args.rho,
            rls_delta=args.rls_delta,
            include_raw=True,
            include_time=True,
        ),
        "reservoir512": ReservoirConfig(
            name="reservoir512",
            reservoir_size=256,
            tanh_features=256,
            fourier_features=0,
            product_features=0,
            chebyshev_degree=1,
            input_scale=0.65,
            recurrent_scale=0.9,
            leak=0.25,
            feature_scale=0.65,
            input_clip=args.input_clip,
            rho=args.rho,
            rls_delta=args.rls_delta,
            include_raw=True,
            include_time=True,
        ),
    }
    if args.configs == "all":
        return list(configs.values())
    selected: list[ReservoirConfig] = []
    for name in args.configs.split(","):
        key = name.strip()
        if key:
            selected.append(configs[key])
    return selected


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    configs: list[ReservoirConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all methods for one paired dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    for method in MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp_learner = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            mlp_learner,
            observations,
            targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_mlp_classifier(mlp_learner, state, x_test, y_test)
            )

    for offset, config in enumerate(configs):
        method = f"koopman_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        reservoir_learner, reservoir_state, metrics = run_reservoir_stream(
            observations,
            targets,
            config,
            seed=seed + 80_000 + offset,
        )
        methods[method] = reservoir_summary(metrics, args.final_window, labels)
        methods[method].update(
            {
                "runtime_s": float(time.time() - t0),
                "readout_dim": float(reservoir_learner.readout_dim),
                "mean_feature_norm_ema": float(reservoir_state.mean_feature_norm),
                "mean_gain_norm_ema": float(reservoir_state.mean_gain_norm),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_reservoir_classifier(
                    reservoir_learner,
                    reservoir_state,
                    x_test,
                    y_test,
                )
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
    parser.add_argument("--datasets", default="synthetic_compositional")
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
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--rls-delta", type=float, default=25.0)
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument(
        "--configs",
        default="tanh256,mixed384",
        help="Comma-separated config names, or all.",
    )
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
    if not 0.0 < args.rho <= 1.0:
        raise ValueError("--rho must be in (0, 1]")
    if args.rls_delta <= 0.0:
        raise ValueError("--rls-delta must be positive")
    if args.input_clip <= 0.0:
        raise ValueError("--input-clip must be positive")


def main() -> None:
    """Run the D13 sweep."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.configs = "tanh256"
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_reservoir_configs(args)
    candidate_methods = tuple(f"koopman_{config.name}" for config in configs)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name,
                seed,
                configs,
                args,
            )
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
            "rho": args.rho,
            "rls_delta": args.rls_delta,
            "input_clip": args.input_clip,
            "configs": args.configs,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "single_predictor_koopman_reservoir_rls_probe",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_d13_summary(md_path, results)
    if args.note_path:
        write_d13_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

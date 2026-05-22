#!/usr/bin/env python3
"""D16: additive universal learner candidate for Step 2.

This runner integrates the useful non-router mechanisms found in D07/D09/D15:

* groupwise random-tanh + Fourier LMS for compositional/frequency structure,
* throttled algebraic-Green KRLS for stateful digit retention and interactions,
* optional raw polynomial KRLS for exact low-dimensional cubic structure.

The prediction is a single sum of all active blocks.  At each timestep all
blocks predict before the target is consumed, the global residual is computed,
and every block updates from that same residual.  No block is selected as an
expert output and no MLP prediction is part of the candidate.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
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
    BudgetedDiffusionKRLS,
    BudgetedKRLSState,
    KernelConfig,
    aggregate_records,
    evaluate_mlp_classifier,
    expand_dataset_names,
    kernel_method_name,
    make_dataset,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)
from d15_groupwise_basis_lms import (  # noqa: E402
    GroupwiseBasisLMS,
    GroupwiseConfig,
    GroupwiseState,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d16_additive_universal_learner")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d16_additive_universal_learner.md"
)


@dataclass(frozen=True)
class AdditiveConfig:
    """Configuration for one additive learner candidate."""

    name: str
    groupwise: GroupwiseConfig
    memory_kernel: KernelConfig | None
    poly_kernel: KernelConfig | None
    residual_clip: float
    group_residual_gain: float
    memory_residual_gain: float
    poly_residual_gain: float


@dataclass
class AdditiveState:
    """Mutable state for the additive learner."""

    groupwise_state: GroupwiseState
    memory_state: BudgetedKRLSState | None
    poly_state: BudgetedKRLSState | None
    steps: int
    finite_failures: int
    mean_residual_norm: float


class AdditiveUniversalLearner:
    """Single additive model with block-coordinate online residual updates."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: AdditiveConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.groupwise = GroupwiseBasisLMS(
            n_heads=n_heads,
            feature_dim=feature_dim,
            config=config.groupwise,
            seed=seed,
        )
        self.memory = (
            BudgetedDiffusionKRLS(n_heads, feature_dim, config.memory_kernel)
            if config.memory_kernel is not None
            else None
        )
        self.poly = (
            BudgetedDiffusionKRLS(n_heads, feature_dim, config.poly_kernel)
            if config.poly_kernel is not None
            else None
        )

    def init(self) -> AdditiveState:
        """Return initial state."""
        return AdditiveState(
            groupwise_state=self.groupwise.init(),
            memory_state=self.memory.init() if self.memory is not None else None,
            poly_state=self.poly.init() if self.poly is not None else None,
            steps=0,
            finite_failures=0,
            mean_residual_norm=0.0,
        )

    def predict(self, state: AdditiveState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads."""
        prediction = self.groupwise.predict(state.groupwise_state, observation)
        if self.memory is not None and state.memory_state is not None:
            prediction = prediction + self.memory.predict(state.memory_state, observation)
        if self.poly is not None and state.poly_state is not None:
            prediction = prediction + self.poly.predict(state.poly_state, observation)
        return np.asarray(prediction)

    def step(
        self,
        state: AdditiveState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict once, then update every block from the global residual."""
        group_pred = self.groupwise.predict(state.groupwise_state, observation)
        memory_pred = (
            self.memory.predict(state.memory_state, observation)
            if self.memory is not None and state.memory_state is not None
            else np.zeros(self.n_heads, dtype=np.float64)
        )
        poly_pred = (
            self.poly.predict(state.poly_state, observation)
            if self.poly is not None and state.poly_state is not None
            else np.zeros(self.n_heads, dtype=np.float64)
        )
        prediction = np.asarray(group_pred + memory_pred + poly_pred)
        active = ~np.isnan(target)
        residual = np.where(active, target - prediction, 0.0)
        if self.config.residual_clip > 0.0:
            residual = np.clip(
                residual,
                -self.config.residual_clip,
                self.config.residual_clip,
            )

        group_target = np.where(
            active,
            group_pred + self.config.group_residual_gain * residual,
            np.nan,
        )
        _, group_diag = self.groupwise.step(
            state.groupwise_state,
            observation,
            group_target,
        )
        memory_diag: dict[str, float] = {}
        if self.memory is not None and state.memory_state is not None:
            memory_target = np.where(
                active,
                memory_pred + self.config.memory_residual_gain * residual,
                np.nan,
            )
            _, memory_diag = self.memory.step(
                state.memory_state,
                observation,
                memory_target,
            )
        poly_diag: dict[str, float] = {}
        if self.poly is not None and state.poly_state is not None:
            poly_target = np.where(
                active,
                poly_pred + self.config.poly_residual_gain * residual,
                np.nan,
            )
            _, poly_diag = self.poly.step(
                state.poly_state,
                observation,
                poly_target,
            )

        state.steps += 1
        residual_norm = float(np.linalg.norm(residual))
        state.mean_residual_norm = 0.99 * state.mean_residual_norm + (
            1.0 - 0.99
        ) * residual_norm
        finite_failures = float(group_diag.get("finite_failures", 0.0))
        finite_failures += float(memory_diag.get("finite_failures", 0.0))
        finite_failures += float(poly_diag.get("finite_failures", 0.0))
        state.finite_failures = int(finite_failures)
        diagnostics = {
            "residual_norm": residual_norm,
            "finite_failures": float(state.finite_failures),
            "memory_centers": float(
                state.memory_state.active_count if state.memory_state is not None else 0
            ),
            "poly_centers": float(
                state.poly_state.active_count if state.poly_state is not None else 0
            ),
            "memory_additions": float(
                state.memory_state.additions if state.memory_state is not None else 0
            ),
            "poly_additions": float(
                state.poly_state.additions if state.poly_state is not None else 0
            ),
        }
        return prediction, diagnostics


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_additive_stream(
    observations: Any,
    targets: Any,
    config: AdditiveConfig,
    seed: int,
) -> tuple[AdditiveUniversalLearner, AdditiveState, np.ndarray]:
    """Run one additive configuration."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = AdditiveUniversalLearner(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 7), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["residual_norm"]
        metrics[idx, 3] = diagnostics["finite_failures"]
        metrics[idx, 4] = diagnostics["memory_centers"]
        metrics[idx, 5] = diagnostics["poly_centers"]
        metrics[idx, 6] = diagnostics["memory_additions"] + diagnostics["poly_additions"]
    return learner, state, metrics


def evaluate_additive_classifier(
    learner: AdditiveUniversalLearner,
    state: AdditiveState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate final additive classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def additive_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one additive run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_residual_norm": float(np.mean(metrics[:, 2])),
            "finite_failures": float(metrics[-1, 3]),
            "memory_centers": float(metrics[-1, 4]),
            "poly_centers": float(metrics[-1, 5]),
            "kernel_additions": float(metrics[-1, 6]),
        }
    )
    return entry


def make_groupwise(args: argparse.Namespace) -> GroupwiseConfig:
    """Return the D15 no-poly block that won compositional/frequency."""
    return GroupwiseConfig(
        name="tanh_fourier",
        input_clip=args.input_clip,
        poly_max_dim=args.poly_max_dim,
        fourier_max_dim=args.fourier_max_dim,
        fourier_frequencies=tuple(float(item) for item in args.fourier_frequencies),
        tanh_width=args.tanh_width,
        tanh_weight_scale=args.tanh_weight_scale,
        poly_step_size=0.0,
        fourier_step_size=args.fourier_step_size,
        tanh_step_size=args.tanh_step_size,
        poly_scale=1.0,
        fourier_scale=1.0,
        tanh_scale=1.0,
        include_poly=False,
        include_fourier=True,
        include_tanh=True,
    )


def make_memory_kernel(args: argparse.Namespace) -> KernelConfig:
    """Return throttled algebraic-Green memory kernel."""
    return KernelConfig(
        budget=args.memory_budget,
        sigma=args.memory_sigma,
        rho=args.memory_rho,
        novelty_threshold=args.memory_novelty_threshold,
        ridge=args.kernel_ridge,
        rls_delta=args.kernel_rls_delta,
        utility_decay=0.99,
        min_center_age=50,
        input_clip=args.input_clip,
        kernel="algebraic_green",
        bandwidth_multipliers=(0.5, 1.0, 2.0),
        polynomial_degree=3,
        algebraic_weight=0.75,
        normalize_polynomial=True,
        arccosine_depth=2,
        kernel_weight_variance=2.0,
        kernel_bias_variance=0.1,
        coefficient_update="rls",
        lms_step_size=0.5,
        replace_when_full=True,
        center_add_interval=args.memory_center_interval,
    )


def make_poly_kernel(args: argparse.Namespace) -> KernelConfig:
    """Return raw polynomial KRLS block."""
    return KernelConfig(
        budget=args.poly_budget,
        sigma=1.0,
        rho=args.poly_rho,
        novelty_threshold=args.poly_novelty_threshold,
        ridge=args.kernel_ridge,
        rls_delta=args.kernel_rls_delta,
        utility_decay=0.99,
        min_center_age=50,
        input_clip=args.input_clip,
        kernel="polynomial",
        bandwidth_multipliers=(0.5, 1.0, 2.0),
        polynomial_degree=3,
        algebraic_weight=0.5,
        normalize_polynomial=False,
        arccosine_depth=2,
        kernel_weight_variance=2.0,
        kernel_bias_variance=0.1,
        coefficient_update="rls",
        lms_step_size=0.5,
        replace_when_full=True,
        center_add_interval=args.poly_center_interval,
    )


def make_configs(args: argparse.Namespace) -> list[AdditiveConfig]:
    """Return additive configurations."""
    groupwise = make_groupwise(args)
    memory = make_memory_kernel(args)
    poly = make_poly_kernel(args)
    configs = {
        "group_memory": AdditiveConfig(
            name="group_memory",
            groupwise=groupwise,
            memory_kernel=memory,
            poly_kernel=None,
            residual_clip=args.residual_clip,
            group_residual_gain=args.group_residual_gain,
            memory_residual_gain=args.memory_residual_gain,
            poly_residual_gain=args.poly_residual_gain,
        ),
        "group_memory_poly": AdditiveConfig(
            name="group_memory_poly",
            groupwise=groupwise,
            memory_kernel=memory,
            poly_kernel=poly,
            residual_clip=args.residual_clip,
            group_residual_gain=args.group_residual_gain,
            memory_residual_gain=args.memory_residual_gain,
            poly_residual_gain=args.poly_residual_gain,
        ),
        "group_poly": AdditiveConfig(
            name="group_poly",
            groupwise=groupwise,
            memory_kernel=None,
            poly_kernel=poly,
            residual_clip=args.residual_clip,
            group_residual_gain=args.group_residual_gain,
            memory_residual_gain=args.memory_residual_gain,
            poly_residual_gain=args.poly_residual_gain,
        ),
        "group_only": AdditiveConfig(
            name="group_only",
            groupwise=groupwise,
            memory_kernel=None,
            poly_kernel=None,
            residual_clip=args.residual_clip,
            group_residual_gain=args.group_residual_gain,
            memory_residual_gain=args.memory_residual_gain,
            poly_residual_gain=args.poly_residual_gain,
        ),
    }
    if args.configs == "all":
        return list(configs.values())
    selected: list[AdditiveConfig] = []
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
        "# D16 Additive Universal Learner Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "Each candidate is one additive predictor updated from one global "
        "residual at every step. Positive candidate-vs-MLP differences favor "
        "the additive learner.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Memory centers | Poly centers | Runtime s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{metric_cell(row, 'memory_centers')} | "
                f"{metric_cell(row, 'poly_centers')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-additive-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-additive counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-additive-vs-best-MLP diff: "
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
    configs: list[AdditiveConfig],
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
        method = f"additive_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_additive_stream(
            observations,
            targets,
            config,
            seed=seed + 110_000 + offset,
        )
        methods[method] = additive_summary(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_additive_classifier(learner, state, x_test, y_test))

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
    parser.add_argument("--tanh-width", type=int, default=512)
    parser.add_argument("--tanh-weight-scale", type=float, default=1.0)
    parser.add_argument("--fourier-step-size", type=float, default=0.3)
    parser.add_argument("--tanh-step-size", type=float, default=0.4)
    parser.add_argument("--memory-budget", type=int, default=128)
    parser.add_argument("--memory-sigma", type=float, default=0.5)
    parser.add_argument("--memory-rho", type=float, default=0.99)
    parser.add_argument("--memory-novelty-threshold", type=float, default=1e-3)
    parser.add_argument("--memory-center-interval", type=int, default=4)
    parser.add_argument("--poly-budget", type=int, default=64)
    parser.add_argument("--poly-rho", type=float, default=0.99)
    parser.add_argument("--poly-novelty-threshold", type=float, default=1e-3)
    parser.add_argument("--poly-center-interval", type=int, default=1)
    parser.add_argument("--kernel-ridge", type=float, default=1e-3)
    parser.add_argument("--kernel-rls-delta", type=float, default=100.0)
    parser.add_argument("--residual-clip", type=float, default=5.0)
    parser.add_argument("--group-residual-gain", type=float, default=1.0)
    parser.add_argument("--memory-residual-gain", type=float, default=0.05)
    parser.add_argument("--poly-residual-gain", type=float, default=0.02)
    parser.add_argument("--configs", default="group_memory")
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
    if args.tanh_width <= 0:
        raise ValueError("--tanh-width must be positive")
    if not 0.0 < args.memory_rho <= 1.0:
        raise ValueError("--memory-rho must be in (0, 1]")
    if not 0.0 < args.poly_rho <= 1.0:
        raise ValueError("--poly-rho must be in (0, 1]")
    if args.memory_center_interval <= 0 or args.poly_center_interval <= 0:
        raise ValueError("center intervals must be positive")
    if args.group_residual_gain < 0.0:
        raise ValueError("--group-residual-gain must be non-negative")
    if args.memory_residual_gain < 0.0:
        raise ValueError("--memory-residual-gain must be non-negative")
    if args.poly_residual_gain < 0.0:
        raise ValueError("--poly-residual-gain must be non-negative")


def main() -> None:
    """Run D16 experiments."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.configs = "group_memory"
        args.tanh_width = 64
        args.memory_budget = 16
        args.poly_budget = 16
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_configs(args)
    candidate_methods = tuple(f"additive_{config.name}" for config in configs)
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
            "groupwise": make_groupwise(args).__dict__,
            "memory_kernel": kernel_method_name(make_memory_kernel(args)),
            "poly_kernel": kernel_method_name(make_poly_kernel(args)),
            "residual_clip": args.residual_clip,
            "group_residual_gain": args.group_residual_gain,
            "memory_residual_gain": args.memory_residual_gain,
            "poly_residual_gain": args.poly_residual_gain,
            "configs": args.configs,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "single_additive_residual_block_coordinate_probe",
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

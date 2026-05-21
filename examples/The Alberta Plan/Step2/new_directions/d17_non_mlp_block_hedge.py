#!/usr/bin/env python3
"""D17: learned non-MLP block gate for Step 2.

This runner tests the smallest learned manager that directly addresses the
D07/D15 conflict.  It maintains several non-MLP blocks, predicts with a causal
discounted-Hedge convex combination of their predictions, then updates every
block on the same target at every timestep.  The gate never sees an MLP
prediction and never selects a single route after the current target is known.
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

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d17_non_mlp_block_hedge")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d17_non_mlp_block_hedge.md")


@dataclass(frozen=True)
class BlockHedgeConfig:
    """Configuration for one learned non-MLP block gate."""

    name: str
    groupwise: GroupwiseConfig
    memory_kernel: KernelConfig
    poly_kernel: KernelConfig | None
    hedge_eta: float
    hedge_discount: float
    initial_log_weight_group: float
    initial_log_weight_memory: float
    initial_log_weight_poly: float


@dataclass
class BlockHedgeState:
    """Mutable state for the learned gate."""

    group_state: GroupwiseState
    memory_state: BudgetedKRLSState
    poly_state: BudgetedKRLSState | None
    log_weights: np.ndarray
    steps: int
    finite_failures: int


class NonMLPBlockHedge:
    """Causal Hedge over non-MLP blocks, with temporally uniform block updates."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: BlockHedgeConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.group = GroupwiseBasisLMS(n_heads, feature_dim, config.groupwise, seed)
        self.memory = BudgetedDiffusionKRLS(n_heads, feature_dim, config.memory_kernel)
        self.poly = (
            BudgetedDiffusionKRLS(n_heads, feature_dim, config.poly_kernel)
            if config.poly_kernel is not None
            else None
        )

    @property
    def n_blocks(self) -> int:
        """Return active block count."""
        return 3 if self.poly is not None else 2

    def init(self) -> BlockHedgeState:
        """Return initial state."""
        log_weights = [
            self.config.initial_log_weight_group,
            self.config.initial_log_weight_memory,
        ]
        if self.poly is not None:
            log_weights.append(self.config.initial_log_weight_poly)
        return BlockHedgeState(
            group_state=self.group.init(),
            memory_state=self.memory.init(),
            poly_state=self.poly.init() if self.poly is not None else None,
            log_weights=np.asarray(log_weights, dtype=np.float64),
            steps=0,
            finite_failures=0,
        )

    def _weights(self, state: BlockHedgeState) -> np.ndarray:
        shifted = state.log_weights - float(np.max(state.log_weights))
        exp = np.exp(shifted)
        return np.asarray(exp / np.maximum(np.sum(exp), 1e-12), dtype=np.float64)

    def predict_blocks(
        self,
        state: BlockHedgeState,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return convex and per-block predictions."""
        weights = self._weights(state)
        group_pred = self.group.predict(state.group_state, observation)
        memory_pred = self.memory.predict(state.memory_state, observation)
        preds = [group_pred, memory_pred]
        poly_pred: np.ndarray | None = None
        if self.poly is not None and state.poly_state is not None:
            poly_pred = self.poly.predict(state.poly_state, observation)
            preds.append(poly_pred)
        stacked = np.stack(preds, axis=0)
        prediction = np.asarray(weights @ stacked)
        return prediction, group_pred, memory_pred, poly_pred

    def predict(self, state: BlockHedgeState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads."""
        prediction, _, _, _ = self.predict_blocks(state, observation)
        return prediction

    def step(
        self,
        state: BlockHedgeState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, update causal block weights, then update all blocks."""
        prediction, group_pred, memory_pred, poly_pred = self.predict_blocks(
            state,
            observation,
        )
        active = ~np.isnan(target)
        safe_target = np.where(active, target, 0.0)
        active_count = max(float(np.sum(active)), 1.0)
        block_preds = [group_pred, memory_pred]
        if poly_pred is not None:
            block_preds.append(poly_pred)
        block_losses = np.asarray(
            [
                float(
                    np.sum(np.where(active, (pred - safe_target) ** 2, 0.0))
                    / active_count
                )
                for pred in block_preds
            ],
            dtype=np.float64,
        )
        state.log_weights = (
            self.config.hedge_discount * state.log_weights
            - self.config.hedge_eta * block_losses
        )
        state.log_weights = state.log_weights - float(np.max(state.log_weights))

        self.group.step(state.group_state, observation, target)
        _, memory_diag = self.memory.step(state.memory_state, observation, target)
        poly_diag: dict[str, float] = {}
        if self.poly is not None and state.poly_state is not None:
            _, poly_diag = self.poly.step(state.poly_state, observation, target)

        state.steps += 1
        state.finite_failures = int(
            float(memory_diag.get("finite_failures", 0.0))
            + float(poly_diag.get("finite_failures", 0.0))
        )
        weights = self._weights(state)
        diagnostics = {
            "group_weight": float(weights[0]),
            "memory_weight": float(weights[1]),
            "poly_weight": float(weights[2]) if weights.shape[0] > 2 else 0.0,
            "group_loss": float(block_losses[0]),
            "memory_loss": float(block_losses[1]),
            "poly_loss": float(block_losses[2]) if block_losses.shape[0] > 2 else 0.0,
            "memory_centers": float(state.memory_state.active_count),
            "poly_centers": float(
                state.poly_state.active_count if state.poly_state is not None else 0
            ),
            "finite_failures": float(state.finite_failures),
        }
        return prediction, diagnostics


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_block_hedge_stream(
    observations: Any,
    targets: Any,
    config: BlockHedgeConfig,
    seed: int,
) -> tuple[NonMLPBlockHedge, BlockHedgeState, np.ndarray]:
    """Run one learned-gate configuration."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = NonMLPBlockHedge(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 9), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["group_weight"]
        metrics[idx, 3] = diagnostics["memory_weight"]
        metrics[idx, 4] = diagnostics["poly_weight"]
        metrics[idx, 5] = diagnostics["memory_centers"]
        metrics[idx, 6] = diagnostics["poly_centers"]
        metrics[idx, 7] = diagnostics["finite_failures"]
        metrics[idx, 8] = diagnostics["group_loss"] - diagnostics["memory_loss"]
    return learner, state, metrics


def evaluate_block_hedge_classifier(
    learner: NonMLPBlockHedge,
    state: BlockHedgeState,
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


def block_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one block-hedge run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_group_weight": float(np.mean(metrics[:, 2])),
            "mean_memory_weight": float(np.mean(metrics[:, 3])),
            "mean_poly_weight": float(np.mean(metrics[:, 4])),
            "final_group_weight": float(metrics[-1, 2]),
            "final_memory_weight": float(metrics[-1, 3]),
            "final_poly_weight": float(metrics[-1, 4]),
            "memory_centers": float(metrics[-1, 5]),
            "poly_centers": float(metrics[-1, 6]),
            "finite_failures": float(metrics[-1, 7]),
        }
    )
    return entry


def make_groupwise(args: argparse.Namespace) -> GroupwiseConfig:
    """Return groupwise tanh+Fourier block."""
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
    """Return D07-style algebraic-Green memory kernel."""
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


def make_configs(args: argparse.Namespace) -> list[BlockHedgeConfig]:
    """Return learned-gate configs."""
    groupwise = make_groupwise(args)
    memory = make_memory_kernel(args)
    poly = make_poly_kernel(args)
    common = {
        "groupwise": groupwise,
        "memory_kernel": memory,
        "hedge_eta": args.hedge_eta,
        "hedge_discount": args.hedge_discount,
        "initial_log_weight_group": args.initial_log_weight_group,
        "initial_log_weight_memory": args.initial_log_weight_memory,
        "initial_log_weight_poly": args.initial_log_weight_poly,
    }
    configs = {
        "group_memory": BlockHedgeConfig(
            name="group_memory",
            poly_kernel=None,
            **common,
        ),
        "group_memory_poly": BlockHedgeConfig(
            name="group_memory_poly",
            poly_kernel=poly,
            **common,
        ),
    }
    if args.configs == "all":
        return list(configs.values())
    selected: list[BlockHedgeConfig] = []
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
        "# D17 Non-MLP Block Hedge Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "The candidate gate only combines non-MLP mathematical blocks. All "
        "blocks update every timestep; weights are causal discounted-Hedge "
        "weights computed before each current target update.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Group w | Memory w | Poly w | Runtime s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                f"{metric_cell(row, 'final_group_weight')} | "
                f"{metric_cell(row, 'final_memory_weight')} | "
                f"{metric_cell(row, 'final_poly_weight')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-block-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-block counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-block-vs-best-MLP diff: "
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
    configs: list[BlockHedgeConfig],
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
        method = f"blockhedge_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_block_hedge_stream(
            observations,
            targets,
            config,
            seed=seed + 120_000 + offset,
        )
        methods[method] = block_summary(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_block_hedge_classifier(learner, state, x_test, y_test)
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
    parser.add_argument("--hedge-eta", type=float, default=4.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument("--initial-log-weight-group", type=float, default=0.0)
    parser.add_argument("--initial-log-weight-memory", type=float, default=0.0)
    parser.add_argument("--initial-log-weight-poly", type=float, default=-1.0)
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
    if not 0.0 < args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in (0, 1]")
    if args.hedge_eta < 0.0:
        raise ValueError("--hedge-eta must be non-negative")


def main() -> None:
    """Run D17 experiments."""
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
    candidate_methods = tuple(f"blockhedge_{config.name}" for config in configs)
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
            "hedge_eta": args.hedge_eta,
            "hedge_discount": args.hedge_discount,
            "configs": args.configs,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "causal_learned_gate_over_non_mlp_blocks",
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

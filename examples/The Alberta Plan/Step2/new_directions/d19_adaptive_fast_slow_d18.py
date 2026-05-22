#!/usr/bin/env python3
"""D19: adaptive fast/slow controls for D18.

This is a Worker-A prototype for replacing two static D18 memory knobs without
adding an MLP expert or prediction router:

* the target-persistence threshold used by D18's fast target/prototype/simplex
  gates;
* the basis readout decay and one-hot/simplex readout decay used by the
  groupwise tanh/Fourier basis.

The learner is still one additive D18 model.  A causal controller updates every
step from online loss and one-hot target-persistence statistics, materializes a
fresh D18 config for the next step, and then D18 updates all active components.

Smoke:

```bash
source .venv/bin/activate
python -m py_compile "examples/The Alberta Plan/Step2/new_directions/d19_adaptive_fast_slow_d18.py"
python "examples/The Alberta Plan/Step2/new_directions/d19_adaptive_fast_slow_d18.py" \
  --smoke --datasets digits_class_blocked --configs step2_canonical \
  --adaptive-policies learned,fixed
python "examples/The Alberta Plan/Step2/new_directions/d19_adaptive_fast_slow_d18.py" \
  --smoke --datasets opmnist --configs step2_canonical \
  --adaptive-policies learned,fixed --n-permutations 3 --task-block-size 40
```
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
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

import d18_simple_universal_resource_basis as d18  # noqa: E402
import step2_published_stressors as stressors  # noqa: E402

D18_ANY: Any = d18
DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d19_adaptive_fast_slow")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d19_adaptive_fast_slow.md"
)
OPMNIST_NAMES = {"opmnist", "permuted_mnist_like"}


@dataclass(frozen=True)
class AdaptiveFastSlowConfig:
    """Controller configuration wrapped around one D18 candidate."""

    name: str
    base_config: d18.UniversalConfig
    learn_threshold: bool
    learn_decay: bool
    fast_loss_decay: float
    slow_loss_decay: float
    persistence_decay: float
    gate_learning_rate: float
    threshold_learning_rate: float
    threshold_tracking_rate: float
    threshold_target_gate: float
    threshold_margin: float
    loss_rise_weight: float
    persistence_weight: float
    gate_temperature: float
    initial_gate_logit: float
    initial_threshold: float | None
    min_threshold: float
    max_threshold: float
    fast_basis_decay: float
    slow_basis_decay: float
    fast_simplex_decay: float
    slow_simplex_decay: float


@dataclass
class AdaptiveFastSlowState:
    """Mutable D19 state: D18 plus the causal control variables."""

    base_state: d18.UniversalState
    fast_loss_ema: float
    slow_loss_ema: float
    persistence_ema: float
    gate_logit: float
    gate: float
    threshold: float
    basis_decay: float
    simplex_decay: float
    steps: int


def sigmoid(value: float) -> float:
    """Stable scalar sigmoid."""
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    return float(D18_ANY.masked_mse_np(prediction, target))


class AdaptiveFastSlowD18:
    """D18 wrapped in a causal adaptive fast/slow memory controller."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: AdaptiveFastSlowConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.base = d18.SimpleUniversalResourceBasis(
            n_heads=n_heads,
            feature_dim=feature_dim,
            config=config.base_config,
            seed=seed,
        )

    def init(self) -> AdaptiveFastSlowState:
        """Return initial adaptive state."""
        base_state = self.base.init()
        initial_threshold = (
            self.config.base_config.target_persistence_threshold
            if self.config.initial_threshold is None
            else self.config.initial_threshold
        )
        initial_threshold = float(
            np.clip(
                initial_threshold,
                self.config.min_threshold,
                self.config.max_threshold,
            )
        )
        gate = sigmoid(
            self.config.initial_gate_logit / max(self.config.gate_temperature, 1e-12)
        )
        basis_decay, simplex_decay = self._decays_from_gate(gate)
        return AdaptiveFastSlowState(
            base_state=base_state,
            fast_loss_ema=0.0,
            slow_loss_ema=0.0,
            persistence_ema=0.0,
            gate_logit=float(self.config.initial_gate_logit),
            gate=gate,
            threshold=initial_threshold,
            basis_decay=basis_decay,
            simplex_decay=simplex_decay,
            steps=0,
        )

    def _decays_from_gate(self, gate: float) -> tuple[float, float]:
        """Map the slow-memory gate to basis and simplex readout decays."""
        gate = float(np.clip(gate, 0.0, 1.0))
        if not self.config.learn_decay:
            basis_decay = self.config.base_config.basis_config.weight_decay
            simplex_decay = self.config.base_config.basis_config.simplex_weight_decay
            if simplex_decay is None:
                simplex_decay = basis_decay
            return float(basis_decay), float(simplex_decay)
        basis_decay = self.config.fast_basis_decay + gate * (
            self.config.slow_basis_decay - self.config.fast_basis_decay
        )
        simplex_decay = self.config.fast_simplex_decay + gate * (
            self.config.slow_simplex_decay - self.config.fast_simplex_decay
        )
        return (
            float(np.clip(basis_decay, 0.0, 1.0)),
            float(np.clip(simplex_decay, 0.0, 1.0)),
        )

    def _causal_persistence_strength(self, state: AdaptiveFastSlowState) -> float:
        """Return D18's one-hot persistence strength before thresholding."""
        base_state = state.base_state
        if self.n_heads <= 1 or not self.base._target_simplex_active(base_state):
            return 0.0
        chance = 1.0 / max(float(self.n_heads), 1.0)
        strength = (base_state.target_persistence - chance) / max(1.0 - chance, 1e-12)
        return float(np.clip(strength, 0.0, 1.0))

    def _materialized_base_config(
        self,
        state: AdaptiveFastSlowState,
    ) -> d18.UniversalConfig:
        """Return a D18 config with current adaptive memory controls."""
        basis_config = replace(
            self.config.base_config.basis_config,
            weight_decay=state.basis_decay,
            simplex_weight_decay=state.simplex_decay,
        )
        threshold = (
            state.threshold
            if self.config.learn_threshold
            else self.config.base_config.target_persistence_threshold
        )
        return replace(
            self.config.base_config,
            basis_config=basis_config,
            target_persistence_threshold=float(threshold),
        )

    def _install_base_config(self, state: AdaptiveFastSlowState) -> None:
        """Install the current adaptive config into the wrapped D18 learner."""
        base_config = self._materialized_base_config(state)
        self.base.config = base_config
        self.base.basis.config = base_config.basis_config

    def _update_controller(
        self,
        state: AdaptiveFastSlowState,
        loss: float,
    ) -> None:
        """Update causal fast/slow statistics for the next prediction."""
        if state.steps <= 0:
            state.fast_loss_ema = loss
            state.slow_loss_ema = loss
        else:
            state.fast_loss_ema = (
                (1.0 - self.config.fast_loss_decay) * state.fast_loss_ema
                + self.config.fast_loss_decay * loss
            )
            state.slow_loss_ema = (
                (1.0 - self.config.slow_loss_decay) * state.slow_loss_ema
                + self.config.slow_loss_decay * loss
            )
        persistence = self._causal_persistence_strength(state)
        state.persistence_ema = (
            (1.0 - self.config.persistence_decay) * state.persistence_ema
            + self.config.persistence_decay * persistence
        )
        loss_rise = max(
            0.0,
            (state.fast_loss_ema - state.slow_loss_ema)
            / max(abs(state.slow_loss_ema), 1e-12),
        )
        loss_rise = float(np.tanh(loss_rise))
        persistence_surplus = state.persistence_ema - state.threshold
        signal = (
            self.config.persistence_weight * persistence_surplus
            - self.config.loss_rise_weight * loss_rise
        )
        state.gate_logit += self.config.gate_learning_rate * signal
        state.gate_logit = float(np.clip(state.gate_logit, -20.0, 20.0))
        state.gate = sigmoid(
            state.gate_logit / max(self.config.gate_temperature, 1e-12)
        )

        if self.config.learn_threshold:
            dual_update = self.config.threshold_learning_rate * (
                state.gate - self.config.threshold_target_gate
            )
            tracking_target = state.persistence_ema - self.config.threshold_margin
            tracking_update = self.config.threshold_tracking_rate * (
                tracking_target - state.threshold
            )
            state.threshold = float(
                np.clip(
                    state.threshold + dual_update + tracking_update,
                    self.config.min_threshold,
                    self.config.max_threshold,
                )
            )
        else:
            state.threshold = float(self.config.base_config.target_persistence_threshold)

        state.basis_decay, state.simplex_decay = self._decays_from_gate(state.gate)
        state.steps += 1

    def predict(self, state: AdaptiveFastSlowState, observation: np.ndarray) -> np.ndarray:
        """Predict without fast temporal context, matching D18 deployment prediction."""
        self._install_base_config(state)
        return cast(np.ndarray, self.base.predict(state.base_state, observation))

    def step(
        self,
        state: AdaptiveFastSlowState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict once, update D18, then update causal controller statistics."""
        self._install_base_config(state)
        prediction, diagnostics = self.base.step(state.base_state, observation, target)
        loss = masked_mse_np(prediction, target)
        self._update_controller(state, loss)
        diagnostics.update(
            {
                "adaptive_loss": float(loss),
                "adaptive_gate": float(state.gate),
                "adaptive_threshold": float(state.threshold),
                "adaptive_basis_decay": float(state.basis_decay),
                "adaptive_simplex_decay": float(state.simplex_decay),
                "adaptive_fast_loss_ema": float(state.fast_loss_ema),
                "adaptive_slow_loss_ema": float(state.slow_loss_ema),
                "adaptive_persistence_ema": float(state.persistence_ema),
            }
        )
        return prediction, diagnostics


def run_adaptive_stream(
    observations: Any,
    targets: Any,
    config: AdaptiveFastSlowConfig,
    seed: int,
) -> tuple[AdaptiveFastSlowD18, AdaptiveFastSlowState, np.ndarray]:
    """Run one D19 adaptive candidate."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = AdaptiveFastSlowD18(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 24), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["residual_norm"]
        metrics[idx, 3] = diagnostics["finite_failures"]
        metrics[idx, 4] = diagnostics["active_centers"]
        metrics[idx, 5] = diagnostics["raw_poly_centers"]
        metrics[idx, 6] = diagnostics["algebraic_green_centers"]
        metrics[idx, 7] = diagnostics["arccosine_centers"]
        metrics[idx, 8] = diagnostics["raw_poly_weight"]
        metrics[idx, 9] = diagnostics["algebraic_green_weight"]
        metrics[idx, 10] = diagnostics["poly_feature_norm"]
        metrics[idx, 11] = diagnostics["unified_feature_norm"]
        metrics[idx, 12] = diagnostics["core_gain"]
        metrics[idx, 13] = diagnostics["basis_gain"]
        metrics[idx, 14] = diagnostics["poly_gain"]
        metrics[idx, 15] = diagnostics["unified_gain"]
        metrics[idx, 16] = diagnostics["adaptive_gate"]
        metrics[idx, 17] = diagnostics["adaptive_threshold"]
        metrics[idx, 18] = diagnostics["adaptive_basis_decay"]
        metrics[idx, 19] = diagnostics["adaptive_simplex_decay"]
        metrics[idx, 20] = diagnostics["adaptive_fast_loss_ema"]
        metrics[idx, 21] = diagnostics["adaptive_slow_loss_ema"]
        metrics[idx, 22] = diagnostics["adaptive_persistence_ema"]
        metrics[idx, 23] = diagnostics["adaptive_loss"]
    return learner, state, metrics


def adaptive_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one D19 run."""
    entry = d18.universal_summary(metrics, final_window, labels)
    entry.update(
        {
            "mean_adaptive_gate": float(np.mean(metrics[:, 16])),
            "final_adaptive_gate": float(metrics[-1, 16]),
            "mean_adaptive_threshold": float(np.mean(metrics[:, 17])),
            "final_adaptive_threshold": float(metrics[-1, 17]),
            "mean_adaptive_basis_decay": float(np.mean(metrics[:, 18])),
            "final_adaptive_basis_decay": float(metrics[-1, 18]),
            "mean_adaptive_simplex_decay": float(np.mean(metrics[:, 19])),
            "final_adaptive_simplex_decay": float(metrics[-1, 19]),
            "mean_adaptive_persistence_ema": float(np.mean(metrics[:, 22])),
            "final_adaptive_persistence_ema": float(metrics[-1, 22]),
        }
    )
    return cast(dict[str, float], entry)


def evaluate_adaptive_classifier_views(
    learner: AdaptiveFastSlowD18,
    state: AdaptiveFastSlowState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a D19 classifier over selected OPMNIST permutation views."""
    targets = np.eye(stressors.N_CLASSES, dtype=np.float64)[y_test]
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        preds = np.stack(
            [learner.predict(state, obs) for obs in view.astype(np.float64)]
        )
        mse_values.append(float(np.mean((preds - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(preds, axis=1) == y_test)))
    return {
        "test_mse": float(np.mean(mse_values)),
        "test_accuracy": float(np.mean(accuracy_values)),
    }


def evaluate_adaptive_classifier(
    learner: AdaptiveFastSlowD18,
    state: AdaptiveFastSlowState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a D19 classifier on one fixed heldout view."""
    targets = np.eye(int(D18_ANY.N_DIGIT_CLASSES), dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def make_adaptive_configs(
    base_configs: list[d18.UniversalConfig],
    args: argparse.Namespace,
) -> list[AdaptiveFastSlowConfig]:
    """Return selected adaptive wrappers around D18 configs."""
    policies = [item.strip() for item in args.adaptive_policies.split(",") if item.strip()]
    if not policies:
        raise ValueError("--adaptive-policies cannot be empty")
    valid = {"learned", "fixed", "threshold_only", "decay_only"}
    unknown = sorted(set(policies) - valid)
    if unknown:
        raise ValueError(f"unknown adaptive policy: {', '.join(unknown)}")

    configs: list[AdaptiveFastSlowConfig] = []
    for base in base_configs:
        for policy in policies:
            configs.append(
                AdaptiveFastSlowConfig(
                    name=f"{policy}_{base.name}",
                    base_config=base,
                    learn_threshold=policy in {"learned", "threshold_only"},
                    learn_decay=policy in {"learned", "decay_only"},
                    fast_loss_decay=args.adaptive_fast_loss_decay,
                    slow_loss_decay=args.adaptive_slow_loss_decay,
                    persistence_decay=args.adaptive_persistence_decay,
                    gate_learning_rate=args.adaptive_gate_learning_rate,
                    threshold_learning_rate=args.adaptive_threshold_learning_rate,
                    threshold_tracking_rate=args.adaptive_threshold_tracking_rate,
                    threshold_target_gate=args.adaptive_threshold_target_gate,
                    threshold_margin=args.adaptive_threshold_margin,
                    loss_rise_weight=args.adaptive_loss_rise_weight,
                    persistence_weight=args.adaptive_persistence_weight,
                    gate_temperature=args.adaptive_gate_temperature,
                    initial_gate_logit=args.adaptive_initial_gate_logit,
                    initial_threshold=args.adaptive_initial_threshold,
                    min_threshold=args.adaptive_min_threshold,
                    max_threshold=args.adaptive_max_threshold,
                    fast_basis_decay=args.adaptive_fast_basis_decay,
                    slow_basis_decay=args.adaptive_slow_basis_decay,
                    fast_simplex_decay=args.adaptive_fast_simplex_decay,
                    slow_simplex_decay=args.adaptive_slow_simplex_decay,
                )
            )
    return configs


def run_d18_dataset_seed(
    dataset_name: str,
    seed: int,
    configs: list[AdaptiveFastSlowConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all D19 methods for one standard D18 dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = D18_ANY.make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    mlp_methods = tuple(D18_ANY.MLP_METHODS)
    for method in mlp_methods:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp = D18_ANY.make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = D18_ANY.run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 30_000 + mlp_methods.index(method)),
        )
        methods[method] = D18_ANY.summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in D18_ANY.DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(D18_ANY.evaluate_mlp_classifier(mlp, state, x_test, y_test))

    for offset, config in enumerate(configs):
        method = f"d19_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_adaptive_stream(
            observations,
            targets,
            config,
            seed=seed + 230_000 + offset,
        )
        methods[method] = adaptive_summary(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in D18_ANY.DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_adaptive_classifier(learner, state, x_test, y_test))

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "methods": methods,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def make_opmnist_args(
    run_args: argparse.Namespace,
    d18_args: argparse.Namespace,
) -> argparse.Namespace:
    """Return the stressor namespace needed to create compact OPMNIST streams."""
    return argparse.Namespace(
        mnist_source=run_args.mnist_source,
        allow_openml_download=run_args.allow_openml_download,
        allow_torchvision_download=run_args.allow_torchvision_download,
        train_fraction=d18_args.train_fraction,
        max_train_examples=run_args.max_train_examples,
        max_test_examples=run_args.max_test_examples,
        mnist_split=run_args.mnist_split,
        openml_data_home=run_args.openml_data_home,
        openml_n_retries=run_args.openml_n_retries,
        openml_retry_delay=run_args.openml_retry_delay,
        torchvision_data_home=run_args.torchvision_data_home,
    )


def run_opmnist_seed(
    seed: int,
    configs: list[AdaptiveFastSlowConfig],
    d18_args: argparse.Namespace,
    run_args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run compact OPMNIST/permuted-MNIST-like classification for one seed."""
    opmnist_args = make_opmnist_args(run_args, d18_args)
    dataset = stressors.load_mnist_like_source(opmnist_args, seed)
    stream = stressors.make_permuted_classification_stream(
        dataset=dataset,
        steps=d18_args.steps,
        seed=seed,
        n_permutations=run_args.n_permutations,
        task_block_size=run_args.task_block_size,
        sample_with_replacement=run_args.sample_with_replacement,
        task_sampling=run_args.task_sampling,
        include_identity_permutation=run_args.include_identity_permutation,
        max_test_permutation_views=run_args.max_test_permutation_views,
        evaluate_all_permutation_views=run_args.evaluate_all_permutation_views,
    )
    observations = np.asarray(stream.observations, dtype=np.float64)
    targets = np.asarray(stream.targets, dtype=np.float64)
    labels = np.asarray(stream.labels, dtype=np.int32)

    methods: dict[str, dict[str, float]] = {}
    mlp_methods = tuple(D18_ANY.MLP_METHODS)
    for method in mlp_methods:
        print(f"opmnist seed={seed}: running {method}")
        mlp = D18_ANY.make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=d18_args.mlp_step_size,
            sparsity=d18_args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = D18_ANY.run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 40_000 + mlp_methods.index(method)),
        )
        methods[method] = D18_ANY.summarize_prequential(
            metrics,
            d18_args.final_window,
            labels,
        )
        methods[method]["runtime_s"] = float(time.time() - t0)
        methods[method].update(
            stressors.evaluate_classifier_views(
                mlp,
                state,
                stream.test_views,
                stream.test_labels,
            )
        )

    for offset, config in enumerate(configs):
        method = f"d19_{config.name}"
        print(f"opmnist seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_adaptive_stream(
            observations,
            targets,
            config,
            seed=seed + 240_000 + offset,
        )
        methods[method] = adaptive_summary(metrics, d18_args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        methods[method].update(
            evaluate_adaptive_classifier_views(
                learner,
                state,
                stream.test_views,
                stream.test_labels,
            )
        )

    return (
        {
            "dataset_name": "opmnist",
            "seed": seed,
            "methods": methods,
            "dataset": stream.metadata,
        },
        stream.metadata,
    )


def expand_datasets(value: str) -> list[str]:
    """Expand D18 dataset aliases while preserving D19 OPMNIST aliases."""
    raw_names = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_names:
        raise ValueError("--datasets cannot be empty")
    names: list[str] = []
    for name in raw_names:
        if name in OPMNIST_NAMES:
            names.append("opmnist")
        else:
            names.extend(D18_ANY.expand_dataset_names(name))
    return names


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    return str(D18_ANY.metric_cell(row, metric))


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a compact D19 Markdown summary."""
    cfg = results["config"]
    lines = [
        "# D19 Adaptive Fast/Slow D18 Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "D19 is one D18 additive learner with a causal fast/slow controller over "
        "target-persistence threshold and basis readout decay. There is no MLP "
        "expert and no prediction router.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Gate | "
                "Threshold | Basis Decay | Runtime s |",
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
                f"{metric_cell(row, 'final_adaptive_gate')} | "
                f"{metric_cell(row, 'final_adaptive_threshold')} | "
                f"{metric_cell(row, 'final_adaptive_basis_decay')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-D19-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.6f} +/- "
                f"{best['paired_diff_stderr']:.6f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-D19 counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-D19-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.6f} +/- "
                f"{best['paired_diff_stderr']:.6f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Return the D19-specific parser; D18 args are parsed separately."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "--adaptive-policies",
        default="learned",
        help="Comma list: learned,fixed,threshold_only,decay_only.",
    )
    parser.add_argument("--adaptive-fast-loss-decay", type=float, default=0.20)
    parser.add_argument("--adaptive-slow-loss-decay", type=float, default=0.01)
    parser.add_argument("--adaptive-persistence-decay", type=float, default=0.05)
    parser.add_argument("--adaptive-gate-learning-rate", type=float, default=0.15)
    parser.add_argument("--adaptive-threshold-learning-rate", type=float, default=0.02)
    parser.add_argument("--adaptive-threshold-tracking-rate", type=float, default=0.01)
    parser.add_argument("--adaptive-threshold-target-gate", type=float, default=0.35)
    parser.add_argument("--adaptive-threshold-margin", type=float, default=0.05)
    parser.add_argument("--adaptive-loss-rise-weight", type=float, default=0.75)
    parser.add_argument("--adaptive-persistence-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-gate-temperature", type=float, default=1.0)
    parser.add_argument("--adaptive-initial-gate-logit", type=float, default=0.0)
    parser.add_argument("--adaptive-initial-threshold", type=float, default=None)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.4)
    parser.add_argument("--adaptive-max-threshold", type=float, default=0.95)
    parser.add_argument("--adaptive-fast-basis-decay", type=float, default=0.9975)
    parser.add_argument("--adaptive-slow-basis-decay", type=float, default=1.0)
    parser.add_argument("--adaptive-fast-simplex-decay", type=float, default=0.9975)
    parser.add_argument("--adaptive-slow-simplex-decay", type=float, default=1.0)
    parser.add_argument("--mnist-source", default="sklearn_digits_28x28")
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--allow-torchvision-download", action="store_true")
    parser.add_argument("--mnist-split", default="stratified")
    parser.add_argument("--openml-data-home", type=Path, default=None)
    parser.add_argument("--torchvision-data-home", type=Path, default=None)
    parser.add_argument("--openml-n-retries", type=int, default=3)
    parser.add_argument("--openml-retry-delay", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=stressors.optional_positive_int, default=600)
    parser.add_argument("--max-test-examples", type=stressors.optional_positive_int, default=200)
    parser.add_argument("--n-permutations", type=int, default=3)
    parser.add_argument("--task-block-size", type=int, default=200)
    parser.add_argument("--task-sampling", choices=("random", "sequential_epoch"), default="random")
    parser.add_argument(
        "--sample-with-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-identity-permutation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-test-permutation-views",
        type=stressors.optional_positive_int,
        default=3,
    )
    parser.add_argument(
        "--evaluate-all-permutation-views",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse D19-specific args and pass remaining args through D18's parser."""
    argv = sys.argv[1:] if argv is None else argv
    run_args, d18_argv = build_parser().parse_known_args(argv)

    old_argv = sys.argv
    try:
        sys.argv = ["d18_simple_universal_resource_basis.py", *d18_argv]
        d18_args = d18.parse_args()
    finally:
        sys.argv = old_argv

    if "--output-dir" not in d18_argv:
        d18_args.output_dir = DEFAULT_OUTPUT_DIR
    if "--note-path" not in d18_argv:
        d18_args.note_path = DEFAULT_NOTE_PATH
    if "--configs" not in d18_argv:
        d18_args.configs = "step2_canonical"
    if d18_args.smoke:
        d18_args.steps = min(d18_args.steps, 120)
        d18_args.n_seeds = 1
        d18_args.final_window = min(d18_args.final_window, 40)
        d18_args.raw_poly_budget = min(d18_args.raw_poly_budget, 12)
        d18_args.algebraic_budget = min(d18_args.algebraic_budget, 12)
        d18_args.arccosine_budget = min(d18_args.arccosine_budget, 12)
        d18_args.total_center_budget = min(d18_args.total_center_budget, 24)
        d18_args.tanh_width = min(d18_args.tanh_width, 64)
        if "--datasets" not in d18_argv:
            d18_args.datasets = "digits_class_blocked"
    return run_args, d18_args


def validate_args(run_args: argparse.Namespace, d18_args: argparse.Namespace) -> None:
    """Validate D19 and delegated D18 args."""
    d18.validate_args(d18_args)
    rates = [
        run_args.adaptive_fast_loss_decay,
        run_args.adaptive_slow_loss_decay,
        run_args.adaptive_persistence_decay,
        run_args.adaptive_gate_learning_rate,
        run_args.adaptive_threshold_learning_rate,
        run_args.adaptive_threshold_tracking_rate,
    ]
    if any(rate < 0.0 for rate in rates):
        raise ValueError("adaptive rates must be non-negative")
    if not 0.0 <= run_args.adaptive_threshold_target_gate <= 1.0:
        raise ValueError("--adaptive-threshold-target-gate must be in [0, 1]")
    if run_args.adaptive_gate_temperature <= 0.0:
        raise ValueError("--adaptive-gate-temperature must be positive")
    if not 0.0 <= run_args.adaptive_min_threshold <= run_args.adaptive_max_threshold <= 1.0:
        raise ValueError("adaptive threshold bounds must satisfy 0 <= min <= max <= 1")
    decays = [
        run_args.adaptive_fast_basis_decay,
        run_args.adaptive_slow_basis_decay,
        run_args.adaptive_fast_simplex_decay,
        run_args.adaptive_slow_simplex_decay,
    ]
    if any(not 0.0 <= decay <= 1.0 for decay in decays):
        raise ValueError("adaptive decay values must be in [0, 1]")
    if run_args.n_permutations < 2:
        raise ValueError("--n-permutations must be at least 2")
    if run_args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")


def main() -> None:
    """Run D19."""
    run_args, d18_args = parse_args()
    validate_args(run_args, d18_args)
    datasets = expand_datasets(d18_args.datasets)
    base_configs = d18.make_configs(d18_args)
    configs = make_adaptive_configs(base_configs, run_args)
    candidate_methods = tuple(f"d19_{config.name}" for config in configs)

    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(d18_args.n_seeds):
            seed = d18_args.seed + offset
            if dataset_name == "opmnist":
                record, dataset_meta = run_opmnist_seed(
                    seed,
                    configs,
                    d18_args,
                    run_args,
                )
            else:
                record, dataset_meta = run_d18_dataset_seed(
                    dataset_name,
                    seed,
                    configs,
                    d18_args,
                )
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta

    results = {
        "config": {
            "datasets": datasets,
            "steps": d18_args.steps,
            "n_seeds": d18_args.n_seeds,
            "seed": d18_args.seed,
            "final_window": d18_args.final_window,
            "configs": d18_args.configs,
            "adaptive_policies": run_args.adaptive_policies,
            "adaptive": {
                "fast_loss_decay": run_args.adaptive_fast_loss_decay,
                "slow_loss_decay": run_args.adaptive_slow_loss_decay,
                "persistence_decay": run_args.adaptive_persistence_decay,
                "gate_learning_rate": run_args.adaptive_gate_learning_rate,
                "threshold_learning_rate": run_args.adaptive_threshold_learning_rate,
                "threshold_tracking_rate": run_args.adaptive_threshold_tracking_rate,
                "threshold_target_gate": run_args.adaptive_threshold_target_gate,
                "threshold_margin": run_args.adaptive_threshold_margin,
                "loss_rise_weight": run_args.adaptive_loss_rise_weight,
                "persistence_weight": run_args.adaptive_persistence_weight,
                "fast_basis_decay": run_args.adaptive_fast_basis_decay,
                "slow_basis_decay": run_args.adaptive_slow_basis_decay,
                "fast_simplex_decay": run_args.adaptive_fast_simplex_decay,
                "slow_simplex_decay": run_args.adaptive_slow_simplex_decay,
            },
            "opmnist": {
                "mnist_source": run_args.mnist_source,
                "n_permutations": run_args.n_permutations,
                "task_block_size": run_args.task_block_size,
                "task_sampling": run_args.task_sampling,
                "sample_with_replacement": run_args.sample_with_replacement,
                "max_train_examples": run_args.max_train_examples,
                "max_test_examples": run_args.max_test_examples,
            },
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(D18_ANY.MLP_METHODS),
        "records": records,
        "aggregate": D18_ANY.aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "adaptive_fast_slow_d18_no_router",
    }
    output_dir = Path(d18_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(Path(d18_args.note_path), results)
    print(f"Wrote {result_path}")
    print(f"Wrote {d18_args.note_path}")


if __name__ == "__main__":
    main()

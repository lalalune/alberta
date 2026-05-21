#!/usr/bin/env python3
"""D07: budgeted diffusion-kernel recursive learner for Step 2.

This is a standalone research runner for a single non-MLP learner.  The model
owns the prediction: it grows a bounded kernel dictionary by an approximate
linear-dependency (ALD) Schur-complement test and updates the shared dictionary
coefficients with recursive least squares (RLS).  It is not a router, stacker,
or correction on top of an MLP.

The default kernel is a small diffusion/Green-style mixture of heat kernels:
instead of committing to one bandwidth, it averages Gaussian heat kernels over
fixed bandwidth multipliers.  Distances are normalized by input dimension, which
keeps the same sigma range usable for low-dimensional synthetic streams and
standardized sklearn digits.
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

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
for path in (SRC_DIR, STEP2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step2_expert_mixture import (  # noqa: E402
    DIGITS_REGIMES,
    N_DIGIT_CLASSES,
    SYNTHETIC_REGIMES,
    load_digits_arrays,
    make_digits_regime_sequence,
    make_synthetic_stream,
)
from step2_recursive_feature_utility_probe import (  # noqa: E402
    SUITE_TASKS,
)
from step2_recursive_feature_utility_probe import (
    make_data as make_controlled_data,
)

from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner  # noqa: E402
from alberta_framework.core.optimizers import ObGDBounding  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d07_budgeted_kernel_recursive")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d07_results.md")

CONTROLLED_DATASETS = tuple(f"controlled_{task}" for task in SUITE_TASKS)
VALID_DATASETS = (*CONTROLLED_DATASETS, *SYNTHETIC_REGIMES, *DIGITS_REGIMES)
MLP_METHODS = ("mlp_h64", "mlp_h128", "mlp_h64_64")
DatasetTuple = tuple[
    jax.Array,
    jax.Array,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]


@dataclass(frozen=True)
class KernelConfig:
    """Configuration for one budgeted kernel learner."""

    budget: int
    sigma: float
    rho: float
    novelty_threshold: float
    ridge: float
    rls_delta: float
    utility_decay: float
    min_center_age: int
    input_clip: float
    kernel: str
    bandwidth_multipliers: tuple[float, ...]
    polynomial_degree: int
    algebraic_weight: float
    normalize_polynomial: bool
    arccosine_depth: int
    kernel_weight_variance: float
    kernel_bias_variance: float
    coefficient_update: str
    lms_step_size: float
    replace_when_full: bool
    center_add_interval: int


@dataclass
class BudgetedKRLSState:
    """Mutable NumPy state for the experiment-local KRLS learner."""

    centers: np.ndarray
    alpha: np.ndarray
    p_matrix: np.ndarray
    k_inv: np.ndarray
    activation_ema: np.ndarray
    coefficient_ema: np.ndarray
    ages: np.ndarray
    active_count: int
    additions: int
    replacements: int
    skipped_novel: int
    throttled_novel: int
    finite_failures: int
    novelty_sum: float
    leverage_sum: float
    steps: int
    last_center_step: int


def stderr(values: np.ndarray) -> float:
    """Return the standard error of a one-dimensional array."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over active target heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def transform_observation(observation: np.ndarray, input_clip: float) -> np.ndarray:
    """Apply the fixed, temporally valid input transform used by the kernel."""
    x = np.asarray(observation, dtype=np.float64)
    if input_clip > 0.0:
        x = np.clip(x, -input_clip, input_clip)
    return x


class BudgetedDiffusionKRLS:
    """Budgeted ALD dictionary with RLS coefficient updates."""

    def __init__(self, n_heads: int, feature_dim: int, config: KernelConfig) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config

    def init(self) -> BudgetedKRLSState:
        """Return an empty dictionary state."""
        budget = self.config.budget
        return BudgetedKRLSState(
            centers=np.zeros((budget, self.feature_dim), dtype=np.float64),
            alpha=np.zeros((budget, self.n_heads), dtype=np.float64),
            p_matrix=np.eye(budget, dtype=np.float64) * self.config.rls_delta,
            k_inv=np.zeros((budget, budget), dtype=np.float64),
            activation_ema=np.zeros(budget, dtype=np.float64),
            coefficient_ema=np.zeros(budget, dtype=np.float64),
            ages=np.zeros(budget, dtype=np.int64),
            active_count=0,
            additions=0,
            replacements=0,
            skipped_novel=0,
            throttled_novel=0,
            finite_failures=0,
            novelty_sum=0.0,
            leverage_sum=0.0,
            steps=0,
            last_center_step=-self.config.center_add_interval,
        )

    def _heat_kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return a normalized Gaussian or diffusion-Green kernel matrix."""
        left_2d = np.atleast_2d(left).astype(np.float64)
        right_2d = np.atleast_2d(right).astype(np.float64)
        diff = left_2d[:, None, :] - right_2d[None, :, :]
        mean_sq = np.mean(diff * diff, axis=2)
        sigma = max(float(self.config.sigma), 1e-8)
        if self.config.kernel == "gaussian":
            return np.asarray(np.exp(-mean_sq / (2.0 * sigma * sigma)))
        kernels: list[np.ndarray] = []
        for multiplier in self.config.bandwidth_multipliers:
            width = max(sigma * float(multiplier), 1e-8)
            kernels.append(np.asarray(np.exp(-mean_sq / (2.0 * width * width))))
        return np.asarray(np.mean(np.stack(kernels, axis=0), axis=0))

    def _polynomial_kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return a self-normalized inhomogeneous polynomial kernel."""
        left_2d = np.atleast_2d(left).astype(np.float64)
        right_2d = np.atleast_2d(right).astype(np.float64)
        degree = max(int(self.config.polynomial_degree), 1)
        dim = max(float(left_2d.shape[1]), 1.0)
        offset = 1.0
        dot = (left_2d @ right_2d.T) / dim
        left_self = offset + np.sum(left_2d * left_2d, axis=1) / dim
        right_self = offset + np.sum(right_2d * right_2d, axis=1) / dim
        raw = np.asarray(np.power(offset + dot, degree))
        if not self.config.normalize_polynomial:
            return raw
        normalizer = np.sqrt(
            np.power(left_self[:, None], degree) * np.power(right_self[None, :], degree)
        )
        return np.asarray(raw / np.maximum(normalizer, 1e-12))

    def _arccosine_kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return a finite-depth ReLU NNGP / arc-cosine kernel."""
        left_2d = np.atleast_2d(left).astype(np.float64)
        right_2d = np.atleast_2d(right).astype(np.float64)
        dim = max(float(left_2d.shape[1]), 1.0)
        weight_var = float(self.config.kernel_weight_variance)
        bias_var = float(self.config.kernel_bias_variance)
        k_xy = weight_var * (left_2d @ right_2d.T) / dim + bias_var
        k_xx = weight_var * np.sum(left_2d * left_2d, axis=1) / dim + bias_var
        k_yy = weight_var * np.sum(right_2d * right_2d, axis=1) / dim + bias_var
        for _ in range(max(int(self.config.arccosine_depth), 1)):
            denom = np.sqrt(np.maximum(k_xx[:, None] * k_yy[None, :], 1e-12))
            cos_theta = np.clip(k_xy / denom, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            relu_cov = denom * (
                np.sin(theta) + (math.pi - theta) * cos_theta
            ) / (2.0 * math.pi)
            k_xy = weight_var * relu_cov + bias_var
            k_xx = weight_var * k_xx / 2.0 + bias_var
            k_yy = weight_var * k_yy / 2.0 + bias_var
        return np.asarray(k_xy)

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the configured RKHS kernel matrix."""
        if self.config.kernel == "polynomial":
            return self._polynomial_kernel(left, right)
        if self.config.kernel == "arccosine":
            return self._arccosine_kernel(left, right)
        heat = self._heat_kernel(left, right)
        if self.config.kernel == "algebraic_green":
            algebraic_weight = float(np.clip(self.config.algebraic_weight, 0.0, 1.0))
            poly = self._polynomial_kernel(left, right)
            return algebraic_weight * poly + (1.0 - algebraic_weight) * heat
        if self.config.kernel == "algebraic_arccosine":
            algebraic_weight = float(np.clip(self.config.algebraic_weight, 0.0, 1.0))
            poly = self._polynomial_kernel(left, right)
            arc = self._arccosine_kernel(left, right)
            return algebraic_weight * poly + (1.0 - algebraic_weight) * arc
        return heat

    def _rebuild_k_inv(self, state: BudgetedKRLSState) -> None:
        """Recompute the active dictionary inverse with ridge jitter."""
        m = state.active_count
        state.k_inv.fill(0.0)
        if m <= 0:
            return
        k_dd = self._kernel(state.centers[:m], state.centers[:m])
        k_dd = k_dd + self.config.ridge * np.eye(m, dtype=np.float64)
        try:
            state.k_inv[:m, :m] = np.linalg.inv(k_dd)
        except np.linalg.LinAlgError:
            state.k_inv[:m, :m] = np.linalg.pinv(k_dd)
            state.finite_failures += 1

    def _novelty(self, state: BudgetedKRLSState, z: np.ndarray) -> float:
        """Return the ALD Schur-complement residual variance for ``z``."""
        m = state.active_count
        self_kernel = float(self._kernel(z, z).reshape(())) + self.config.ridge
        if m == 0:
            return self_kernel
        phi = self._kernel(z, state.centers[:m]).reshape(m)
        residual = self_kernel - float(
            phi @ state.k_inv[:m, :m] @ phi
        )
        return max(residual, 0.0)

    def _replacement_index(self, state: BudgetedKRLSState) -> int:
        """Choose the weakest mature center for replacement."""
        m = state.active_count
        score = state.activation_ema[:m] + 0.05 * state.coefficient_ema[:m]
        mature = state.ages[:m] >= self.config.min_center_age
        if np.any(mature):
            masked = np.where(mature, score, np.inf)
            return int(np.argmin(masked))
        return int(np.argmin(score))

    def _add_or_replace_center(
        self,
        state: BudgetedKRLSState,
        z: np.ndarray,
        novelty: float,
    ) -> None:
        """Use the ALD test to allocate a center before the coefficient update."""
        if novelty <= self.config.novelty_threshold:
            return
        can_allocate = (
            state.steps - state.last_center_step
        ) >= self.config.center_add_interval
        if not can_allocate:
            state.throttled_novel += 1
            return
        if state.active_count < self.config.budget:
            idx = state.active_count
            state.active_count += 1
            state.additions += 1
            reset_covariance = False
        elif self.config.replace_when_full:
            idx = self._replacement_index(state)
            state.replacements += 1
            reset_covariance = True
        else:
            state.skipped_novel += 1
            return

        state.centers[idx] = z
        state.alpha[idx] = 0.0
        state.activation_ema[idx] = 0.0
        state.coefficient_ema[idx] = 0.0
        state.ages[idx] = 0
        state.last_center_step = state.steps
        if reset_covariance:
            state.p_matrix[: state.active_count, : state.active_count] = (
                np.eye(state.active_count, dtype=np.float64) * self.config.rls_delta
            )
        else:
            state.p_matrix[idx, :] = 0.0
            state.p_matrix[:, idx] = 0.0
            state.p_matrix[idx, idx] = self.config.rls_delta
        self._rebuild_k_inv(state)

    def predict(self, state: BudgetedKRLSState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads without changing state."""
        if state.active_count == 0:
            return np.zeros(self.n_heads, dtype=np.float64)
        z = transform_observation(observation, self.config.input_clip)
        phi = self._kernel(z, state.centers[: state.active_count]).reshape(
            state.active_count
        )
        return np.asarray(phi @ state.alpha[: state.active_count])

    def step(
        self,
        state: BudgetedKRLSState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, grow the dictionary if needed, then update coefficients."""
        z = transform_observation(observation, self.config.input_clip)
        prediction = self.predict(state, observation)
        novelty = self._novelty(state, z)
        state.novelty_sum += novelty
        self._add_or_replace_center(state, z, novelty)

        m = state.active_count
        leverage = 0.0
        if m > 0:
            phi = self._kernel(z, state.centers[:m]).reshape(m)
            active = ~np.isnan(target)
            safe_target = np.where(active, target, 0.0)
            update_prediction = phi @ state.alpha[:m]
            errors = np.where(active, safe_target - update_prediction, 0.0)
            if self.config.coefficient_update == "rls":
                p_active = state.p_matrix[:m, :m]
                p_phi = p_active @ phi
                denom = self.config.rho + float(phi @ p_phi)
                if denom <= 1e-12 or not np.isfinite(denom):
                    state.finite_failures += 1
                else:
                    gain = p_phi / denom
                    state.alpha[:m] += np.outer(gain, errors)
                    next_p = (p_active - np.outer(gain, phi @ p_active)) / self.config.rho
                    state.p_matrix[:m, :m] = 0.5 * (next_p + next_p.T)
                    leverage = float(phi @ p_phi)
            else:
                normalizer = 1.0 + float(phi @ phi)
                state.alpha[:m] += (
                    self.config.lms_step_size * np.outer(phi, errors) / normalizer
                )
                leverage = float(phi @ phi)

            decay = self.config.utility_decay
            state.activation_ema[:m] = decay * state.activation_ema[:m] + (
                1.0 - decay
            ) * np.abs(phi)
            state.coefficient_ema[:m] = decay * state.coefficient_ema[:m] + (
                1.0 - decay
            ) * np.mean(np.abs(state.alpha[:m]), axis=1)
            state.ages[:m] += 1

        if not np.all(np.isfinite(state.alpha[:m])):
            state.finite_failures += 1
            state.alpha[:m] = np.nan_to_num(state.alpha[:m], copy=False)
        state.leverage_sum += leverage
        state.steps += 1
        diagnostics = {
            "active_centers": float(state.active_count),
            "novelty": float(novelty),
            "leverage": float(leverage),
            "additions": float(state.additions),
            "replacements": float(state.replacements),
            "skipped_novel": float(state.skipped_novel),
            "throttled_novel": float(state.throttled_novel),
            "finite_failures": float(state.finite_failures),
        }
        return prediction, diagnostics


def run_kernel_stream(
    observations: jax.Array,
    targets: jax.Array,
    config: KernelConfig,
) -> tuple[BudgetedDiffusionKRLS, BudgetedKRLSState, np.ndarray]:
    """Run one kernel configuration on a materialized stream."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = BudgetedDiffusionKRLS(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 8), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["active_centers"]
        metrics[idx, 3] = diagnostics["novelty"]
        metrics[idx, 4] = diagnostics["leverage"]
        metrics[idx, 5] = diagnostics["additions"]
        metrics[idx, 6] = diagnostics["replacements"]
        metrics[idx, 7] = diagnostics["finite_failures"]
    return learner, state, metrics


def make_mlp(method: str, n_heads: int, step_size: float, sparsity: float) -> MultiHeadMLPLearner:
    """Create one fair MLP baseline from the local grid."""
    hidden_sizes = {
        "mlp_h64": (64,),
        "mlp_h128": (128,),
        "mlp_h64_64": (64, 64),
    }[method]
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=True,
    )


def run_mlp_stream(
    learner: MultiHeadMLPLearner,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run a fair MLP baseline and record prequential predictions."""
    state = learner.init(feature_dim=int(observations.shape[1]), key=key)

    def step_fn(carry: Any, sample: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        obs, tgt = sample
        pred = learner.predict(carry, obs)
        active = ~jnp.isnan(tgt)
        safe_target = jnp.where(active, tgt, 0.0)
        active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        loss = jnp.sum(jnp.where(active, (pred - safe_target) ** 2, 0.0)) / active_count
        result = learner.update(carry, obs, tgt)
        metric = jnp.asarray([loss, jnp.argmax(pred).astype(jnp.float32)])
        return result.state, metric

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def hybrid_method_name(mlp_method: str, config: KernelConfig) -> str:
    """Return a method name for an additive MLP + residual kernel learner."""
    return f"hybrid_{mlp_method}_{kernel_method_name(config)}"


def run_residual_hybrid_stream(
    mlp_method: str,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: KernelConfig,
    mlp_step_size: float,
    mlp_sparsity: float,
) -> tuple[
    MultiHeadMLPLearner,
    Any,
    BudgetedDiffusionKRLS,
    BudgetedKRLSState,
    np.ndarray,
]:
    """Run one additive predictor: fair MLP plus KRLS residual memory."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    mlp = make_mlp(
        method=mlp_method,
        n_heads=int(targets.shape[1]),
        step_size=mlp_step_size,
        sparsity=mlp_sparsity,
    )
    mlp_state = mlp.init(feature_dim=int(observations.shape[1]), key=key)
    kernel = BudgetedDiffusionKRLS(
        n_heads=int(targets.shape[1]),
        feature_dim=int(observations.shape[1]),
        config=config,
    )
    kernel_state = kernel.init()
    metrics = np.zeros((obs_np.shape[0], 8), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        obs_jax = jnp.asarray(obs.astype(np.float32))
        target_jax = jnp.asarray(target.astype(np.float32))
        mlp_pred = np.asarray(mlp.predict(mlp_state, obs_jax), dtype=np.float64)
        kernel_pred = kernel.predict(kernel_state, obs)
        prediction = mlp_pred + kernel_pred
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        active = ~np.isnan(target)
        residual_target = np.where(active, target - mlp_pred, np.nan)
        _, diagnostics = kernel.step(kernel_state, obs, residual_target)
        result = mlp.update(mlp_state, obs_jax, target_jax)
        mlp_state = result.state
        metrics[idx, 2] = diagnostics["active_centers"]
        metrics[idx, 3] = diagnostics["novelty"]
        metrics[idx, 4] = diagnostics["leverage"]
        metrics[idx, 5] = diagnostics["additions"]
        metrics[idx, 6] = diagnostics["replacements"]
        metrics[idx, 7] = diagnostics["finite_failures"]
    return mlp, mlp_state, kernel, kernel_state, metrics


def summarize_prequential(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
    loss_col: int = 0,
    pred_col: int = 1,
) -> dict[str, float]:
    """Summarize one method's prequential stream metrics."""
    window = min(final_window, metrics.shape[0])
    entry = {
        "online_mean_mse": float(np.mean(metrics[:, loss_col])),
        "final_window_mse": float(np.mean(metrics[-window:, loss_col])),
    }
    if labels is not None:
        predicted = metrics[:, pred_col].astype(np.int32)
        correct = predicted == labels
        entry["online_mean_accuracy"] = float(np.mean(correct))
        entry["final_window_accuracy"] = float(np.mean(correct[-window:]))
    return entry


def evaluate_kernel_classifier(
    learner: BudgetedDiffusionKRLS,
    state: BudgetedKRLSState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final kernel classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def evaluate_mlp_classifier(
    learner: MultiHeadMLPLearner,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final MLP classifier on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def evaluate_hybrid_classifier(
    mlp: MultiHeadMLPLearner,
    mlp_state: Any,
    kernel: BudgetedDiffusionKRLS,
    kernel_state: BudgetedKRLSState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final additive MLP + kernel classifier."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    mlp_preds = np.asarray(
        jax.vmap(lambda obs: mlp.predict(mlp_state, obs))(observations),
        dtype=np.float64,
    )
    kernel_preds = np.stack(
        [kernel.predict(kernel_state, obs) for obs in x_test.astype(np.float64)]
    )
    preds = mlp_preds + kernel_preds
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def kernel_method_name(config: KernelConfig) -> str:
    """Return a compact stable name for a kernel configuration."""
    threshold = f"{config.novelty_threshold:g}".replace("-", "m").replace(".", "p")
    sigma = f"{config.sigma:g}".replace(".", "p")
    rho = f"{config.rho:g}".replace(".", "p")
    suffix = ""
    if config.kernel in {"polynomial", "algebraic_green", "algebraic_arccosine"}:
        weight = f"{config.algebraic_weight:g}".replace(".", "p")
        suffix = f"_d{config.polynomial_degree}_aw{weight}"
        if not config.normalize_polynomial:
            suffix = f"{suffix}_rawpoly"
    if config.kernel in {"arccosine", "algebraic_arccosine"}:
        suffix = f"{suffix}_arc{config.arccosine_depth}"
    if config.center_add_interval > 1:
        suffix = f"{suffix}_ai{config.center_add_interval}"
    return (
        f"{config.kernel}_{config.coefficient_update}_b{config.budget}_"
        f"s{sigma}_r{rho}_n{threshold}{suffix}"
    )


def make_kernel_configs(args: argparse.Namespace) -> list[KernelConfig]:
    """Expand the CLI sweep into concrete kernel configurations."""
    configs: list[KernelConfig] = []
    multipliers = tuple(float(item) for item in args.bandwidth_multipliers)
    add_intervals = tuple(
        int(item)
        for item in (
            args.center_add_intervals
            if args.center_add_intervals is not None
            else (args.center_add_interval,)
        )
    )
    for budget in args.budgets:
        for sigma in args.sigmas:
            for rho in args.rhos:
                for novelty in args.novelty_thresholds:
                    for coefficient_update in args.coefficient_updates:
                        for add_interval in add_intervals:
                            configs.append(
                                KernelConfig(
                                    budget=int(budget),
                                    sigma=float(sigma),
                                    rho=float(rho),
                                    novelty_threshold=float(novelty),
                                    ridge=float(args.ridge),
                                    rls_delta=float(args.rls_delta),
                                    utility_decay=float(args.utility_decay),
                                    min_center_age=int(args.min_center_age),
                                    input_clip=float(args.input_clip),
                                    kernel=args.kernel,
                                    bandwidth_multipliers=multipliers,
                                    polynomial_degree=int(args.polynomial_degree),
                                    algebraic_weight=float(args.algebraic_weight),
                                    normalize_polynomial=bool(args.normalize_polynomial),
                                    arccosine_depth=int(args.arccosine_depth),
                                    kernel_weight_variance=float(
                                        args.kernel_weight_variance
                                    ),
                                    kernel_bias_variance=float(args.kernel_bias_variance),
                                    coefficient_update=coefficient_update,
                                    lms_step_size=float(args.kernel_lms_step_size),
                                    replace_when_full=bool(args.replace_when_full),
                                    center_add_interval=int(add_interval),
                                )
                            )
    return configs


def controlled_dataset_name(task_mode: str) -> str:
    """Return canonical name for a controlled recursive task."""
    return f"controlled_{task_mode}"


def expand_dataset_names(spec: str) -> list[str]:
    """Expand dataset aliases into concrete benchmark names."""
    aliases = {
        "all": list(VALID_DATASETS),
        "controlled": list(CONTROLLED_DATASETS),
        "synthetic": list(SYNTHETIC_REGIMES),
        "digits": list(DIGITS_REGIMES),
        "universal": list((*SYNTHETIC_REGIMES, *DIGITS_REGIMES)),
    }
    names: list[str] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in aliases:
            names.extend(aliases[item])
        elif item in SUITE_TASKS:
            names.append(controlled_dataset_name(item))
        else:
            names.append(item)
    unknown = sorted({name for name in names if name not in VALID_DATASETS})
    if unknown:
        valid = ", ".join(("all", "controlled", "synthetic", "digits", *VALID_DATASETS))
        raise ValueError(f"unknown datasets {unknown}; valid entries: {valid}")
    return list(dict.fromkeys(names))


def split_hybrid_mlp_methods(spec: str) -> tuple[str, ...]:
    """Parse and validate hybrid MLP backbone names."""
    methods = tuple(item.strip() for item in spec.split(",") if item.strip())
    unknown = sorted(set(methods).difference(MLP_METHODS))
    if unknown:
        raise ValueError(
            f"unknown --hybrid-mlp-methods entries {unknown}; valid: {MLP_METHODS}"
        )
    return methods


def make_dataset(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> DatasetTuple:
    """Materialize one paired stream and optional held-out classifier split."""
    labels_np: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if dataset_name.startswith("controlled_"):
        task_mode = dataset_name.removeprefix("controlled_")
        observations, targets = make_controlled_data(
            seed=seed,
            num_steps=args.steps,
            feature_dim=args.feature_dim,
            noise_std=args.noise_std,
            task_mode=task_mode,
            rare_period=args.rare_period,
        )
        meta: dict[str, Any] = {
            "benchmark_family": "controlled_recursive_suite",
            "task_mode": task_mode,
            "feature_dim": args.feature_dim,
        }
    elif dataset_name in SYNTHETIC_REGIMES:
        observations, targets, meta = make_synthetic_stream(
            steps=args.steps,
            seed=seed + 20_000,
            regime=dataset_name,
        )
    elif dataset_name in DIGITS_REGIMES:
        x_train, y_train, x_test, y_test, meta = load_digits_arrays(
            seed=seed,
            train_fraction=args.train_fraction,
        )
        observations, targets, labels, x_test, y_test, stream_meta = make_digits_regime_sequence(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            steps=args.steps,
            seed=seed + 10_000,
            regime=dataset_name,
            phase_length=args.phase_length,
            mask_keep_fraction=args.mask_keep_fraction,
            mask_noise_std=args.mask_noise_std,
        )
        meta.update(stream_meta)
        labels_np = np.asarray(labels, dtype=np.int32)
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")
    return observations, targets, labels_np, x_test, y_test, meta


def method_metric_keys(methods: dict[str, dict[str, float]]) -> list[str]:
    """Return all scalar metrics present on any method."""
    keys: set[str] = set()
    for metrics in methods.values():
        keys.update(metrics)
    return sorted(keys)


def is_higher_better(metric: str) -> bool:
    """Return whether larger values are better for this metric."""
    return metric.endswith("accuracy")


def paired_diff(candidate: float, baseline: float, metric: str) -> float:
    """Positive paired difference favors the candidate."""
    if is_higher_better(metric):
        return candidate - baseline
    return baseline - candidate


def compare_to_group(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
    group: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one method with the per-seed best member of a baseline group."""
    diffs: list[float] = []
    best_methods: list[str] = []
    for record in records:
        methods = record["methods"]
        if metric not in methods[method]:
            continue
        group_values = {
            name: float(methods[name][metric])
            for name in group
            if name in methods and metric in methods[name]
        }
        if not group_values:
            continue
        if is_higher_better(metric):
            best_name = max(group_values, key=group_values.__getitem__)
        else:
            best_name = min(group_values, key=group_values.__getitem__)
        best_methods.append(best_name)
        diffs.append(paired_diff(float(methods[method][metric]), group_values[best_name], metric))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "method": method,
        "metric": metric,
        "paired_diff_mean_positive_favors_method": float(np.mean(diff_arr))
        if diff_arr.size
        else 0.0,
        "paired_diff_stderr": stderr(diff_arr) if diff_arr.size else 0.0,
        "wins_for_method": int(np.sum(diff_arr > 0.0)),
        "wins_for_baseline": int(np.sum(diff_arr < 0.0)),
        "ties": int(np.sum(diff_arr == 0.0)),
        "n": int(diff_arr.shape[0]),
        "diffs": diff_arr.tolist(),
        "best_baseline_counts": dict(
            sorted((name, best_methods.count(name)) for name in set(best_methods))
        ),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    kernel_methods: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate seed records by dataset/method and add MLP comparisons."""
    aggregate: dict[str, Any] = {}
    for dataset in sorted({record["dataset_name"] for record in records}):
        dataset_records = [r for r in records if r["dataset_name"] == dataset]
        method_names = list(dataset_records[0]["methods"])
        dataset_agg: dict[str, Any] = {}
        for method in method_names:
            metric_rows: dict[str, Any] = {}
            for metric in sorted(dataset_records[0]["methods"][method]):
                values = np.asarray(
                    [r["methods"][method][metric] for r in dataset_records],
                    dtype=np.float64,
                )
                metric_rows[metric] = {
                    "mean": float(np.mean(values)),
                    "stderr": stderr(values),
                    "values": values.tolist(),
                }
            dataset_agg[method] = metric_rows
        primary_metrics = [
            metric
            for metric in (
                "final_window_mse",
                "online_mean_mse",
                "test_mse",
                "final_window_accuracy",
                "online_mean_accuracy",
                "test_accuracy",
            )
            if metric in dataset_records[0]["methods"][method_names[0]]
        ]
        comparisons: dict[str, Any] = {}
        for metric in primary_metrics:
            comparisons[metric] = {
                method: compare_to_group(dataset_records, method, metric, MLP_METHODS)
                for method in kernel_methods
            }
            best_kernel_by_seed: list[str] = []
            diffs: list[float] = []
            for record in dataset_records:
                methods = record["methods"]
                kernel_values = {
                    method: methods[method][metric]
                    for method in kernel_methods
                    if metric in methods[method]
                }
                if is_higher_better(metric):
                    best_kernel = max(kernel_values, key=kernel_values.__getitem__)
                    best_mlp = max(
                        MLP_METHODS,
                        key=lambda name: methods[name][metric],
                    )
                else:
                    best_kernel = min(kernel_values, key=kernel_values.__getitem__)
                    best_mlp = min(
                        MLP_METHODS,
                        key=lambda name: methods[name][metric],
                    )
                best_kernel_by_seed.append(best_kernel)
                diffs.append(
                    paired_diff(
                        float(methods[best_kernel][metric]),
                        float(methods[best_mlp][metric]),
                        metric,
                    )
                )
            diff_arr = np.asarray(diffs, dtype=np.float64)
            comparisons[metric]["best_kernel_vs_best_mlp"] = {
                "paired_diff_mean_positive_favors_kernel": float(np.mean(diff_arr)),
                "paired_diff_stderr": stderr(diff_arr),
                "wins_for_kernel": int(np.sum(diff_arr > 0.0)),
                "wins_for_mlp": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n": int(diff_arr.shape[0]),
                "diffs": diff_arr.tolist(),
                "best_kernel_counts": dict(
                    sorted(
                        (name, best_kernel_by_seed.count(name))
                        for name in set(best_kernel_by_seed)
                    )
                ),
            }
        dataset_agg["comparisons"] = comparisons
        aggregate[dataset] = dataset_agg
    return aggregate


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell for Markdown."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a detailed but compact Markdown report."""
    cfg = results["config"]
    lines = [
        "# D07 Budgeted Diffusion-KRLS Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Kernel={cfg['kernel']}, "
            f"updates={', '.join(cfg['coefficient_updates'])}, budgets="
            f"{cfg['budgets']}, sigmas={cfg['sigmas']}, rhos={cfg['rhos']}, "
            f"novelty thresholds={cfg['novelty_thresholds']}."
        ),
        "",
        "This is a single-learner test. The kernel methods do not consume MLP "
        "predictions, routes, stacker weights, or offline labels. Positive "
        "kernel-vs-MLP paired differences favor the kernel method.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Active centers | Runtime s |",
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
                f"{metric_cell(row, 'active_centers')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-kernel-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-kernel counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-kernel-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Bar",
            "",
            "A positive result here is still a search result unless one fixed "
            "kernel configuration beats the best fair MLP across the broad suite. "
            "The `best_kernel_vs_best_mlp` rows are useful for detecting whether "
            "the mathematical mechanism has headroom; a universal learner claim "
            "requires promoting one canonical configuration and rerunning it "
            "without per-dataset selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="controlled",
        help=(
            "Comma-separated regimes or aliases: all, controlled, synthetic, "
            "digits, universal, controlled task names, or concrete dataset names."
        ),
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=5)
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
    parser.add_argument("--budgets", type=int, nargs="+", default=(64,))
    parser.add_argument("--sigmas", type=float, nargs="+", default=(1.0,))
    parser.add_argument("--rhos", type=float, nargs="+", default=(0.99,))
    parser.add_argument(
        "--novelty-thresholds",
        type=float,
        nargs="+",
        default=(1e-3,),
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rls-delta", type=float, default=100.0)
    parser.add_argument("--utility-decay", type=float, default=0.99)
    parser.add_argument("--min-center-age", type=int, default=50)
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument(
        "--kernel",
        choices=(
            "green",
            "gaussian",
            "polynomial",
            "algebraic_green",
            "arccosine",
            "algebraic_arccosine",
        ),
        default="green",
        help=(
            "`green` averages heat kernels; `algebraic_green` adds a normalized "
            "polynomial RKHS component."
        ),
    )
    parser.add_argument("--arccosine-depth", type=int, default=2)
    parser.add_argument("--kernel-weight-variance", type=float, default=2.0)
    parser.add_argument("--kernel-bias-variance", type=float, default=0.1)
    parser.add_argument(
        "--bandwidth-multipliers",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
    )
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--algebraic-weight", type=float, default=0.5)
    parser.add_argument(
        "--normalize-polynomial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Self-normalize the polynomial component. Disable for exact finite "
            "degree polynomial RKHS tests on low-dimensional controlled streams."
        ),
    )
    parser.add_argument(
        "--coefficient-updates",
        choices=("rls", "lms"),
        nargs="+",
        default=("rls",),
    )
    parser.add_argument("--kernel-lms-step-size", type=float, default=0.5)
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Also run additive MLP + KRLS residual learners as single predictors.",
    )
    parser.add_argument(
        "--hybrid-mlp-methods",
        default="mlp_h128",
        help=f"Comma-separated hybrid backbones from: {', '.join(MLP_METHODS)}.",
    )
    parser.add_argument(
        "--center-add-interval",
        type=int,
        default=1,
        help=(
            "Minimum online steps between accepted dictionary insertions. Values "
            ">1 reserve capacity for later regimes instead of filling the budget "
            "immediately."
        ),
    )
    parser.add_argument(
        "--center-add-intervals",
        type=int,
        nargs="+",
        default=None,
        help="Optional sweep values overriding --center-add-interval.",
    )
    parser.add_argument(
        "--replace-when-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny harness check.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments with explicit messages."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("--budgets must be positive")
    if any(sigma <= 0.0 for sigma in args.sigmas):
        raise ValueError("--sigmas must be positive")
    if any(rho <= 0.0 or rho > 1.0 for rho in args.rhos):
        raise ValueError("--rhos must be in (0, 1]")
    if any(threshold < 0.0 for threshold in args.novelty_thresholds):
        raise ValueError("--novelty-thresholds must be non-negative")
    if args.ridge <= 0.0:
        raise ValueError("--ridge must be positive")
    if args.rls_delta <= 0.0:
        raise ValueError("--rls-delta must be positive")
    if args.center_add_interval <= 0:
        raise ValueError("--center-add-interval must be positive")
    if args.center_add_intervals is not None and any(
        interval <= 0 for interval in args.center_add_intervals
    ):
        raise ValueError("--center-add-intervals entries must be positive")
    if args.polynomial_degree <= 0:
        raise ValueError("--polynomial-degree must be positive")
    if not 0.0 <= args.algebraic_weight <= 1.0:
        raise ValueError("--algebraic-weight must be in [0, 1]")
    if args.arccosine_depth <= 0:
        raise ValueError("--arccosine-depth must be positive")
    if args.kernel_weight_variance <= 0.0:
        raise ValueError("--kernel-weight-variance must be positive")
    if args.kernel_bias_variance < 0.0:
        raise ValueError("--kernel-bias-variance must be non-negative")


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    kernel_configs: list[KernelConfig],
    hybrid_mlp_methods: tuple[str, ...],
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
        learner = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            learner,
            observations,
            targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_mlp_classifier(learner, state, x_test, y_test))

    for config in kernel_configs:
        method = kernel_method_name(config)
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        kernel_learner, kernel_state, metrics = run_kernel_stream(
            observations,
            targets,
            config,
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method].update(
            {
                "runtime_s": float(time.time() - t0),
                "active_centers": float(kernel_state.active_count),
                "additions": float(kernel_state.additions),
                "replacements": float(kernel_state.replacements),
                "skipped_novel": float(kernel_state.skipped_novel),
                "throttled_novel": float(kernel_state.throttled_novel),
                "mean_novelty": float(
                    kernel_state.novelty_sum / max(kernel_state.steps, 1)
                ),
                "mean_leverage": float(
                    kernel_state.leverage_sum / max(kernel_state.steps, 1)
                ),
                "finite_failures": float(kernel_state.finite_failures),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_kernel_classifier(
                    kernel_learner,
                    kernel_state,
                    x_test,
                    y_test,
                )
            )

    if args.include_hybrid:
        for mlp_method in hybrid_mlp_methods:
            for config in kernel_configs:
                method = hybrid_method_name(mlp_method, config)
                print(f"{dataset_name} seed={seed}: running {method}")
                t0 = time.time()
                mlp, mlp_state, kernel, kernel_state, metrics = run_residual_hybrid_stream(
                    mlp_method=mlp_method,
                    observations=observations,
                    targets=targets,
                    key=jr.key(seed + 50_000 + MLP_METHODS.index(mlp_method)),
                    config=config,
                    mlp_step_size=args.mlp_step_size,
                    mlp_sparsity=args.mlp_sparsity,
                )
                methods[method] = summarize_prequential(metrics, args.final_window, labels)
                methods[method].update(
                    {
                        "runtime_s": float(time.time() - t0),
                        "active_centers": float(kernel_state.active_count),
                        "additions": float(kernel_state.additions),
                        "replacements": float(kernel_state.replacements),
                        "skipped_novel": float(kernel_state.skipped_novel),
                        "throttled_novel": float(kernel_state.throttled_novel),
                        "mean_novelty": float(
                            kernel_state.novelty_sum / max(kernel_state.steps, 1)
                        ),
                        "mean_leverage": float(
                            kernel_state.leverage_sum / max(kernel_state.steps, 1)
                        ),
                        "finite_failures": float(kernel_state.finite_failures),
                    }
                )
                if dataset_name in DIGITS_REGIMES:
                    assert x_test is not None and y_test is not None
                    methods[method].update(
                        evaluate_hybrid_classifier(
                            mlp,
                            mlp_state,
                            kernel,
                            kernel_state,
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


def main() -> None:
    """Run the D07 sweep and write JSON/Markdown outputs."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.budgets = (16,)
        args.sigmas = (1.0,)
        args.rhos = (0.99,)
        args.novelty_thresholds = (1e-3,)
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    kernel_configs = make_kernel_configs(args)
    hybrid_mlp_methods = split_hybrid_mlp_methods(args.hybrid_mlp_methods)
    kernel_methods = tuple(kernel_method_name(config) for config in kernel_configs)
    hybrid_methods = (
        tuple(
            hybrid_method_name(mlp_method, config)
            for mlp_method in hybrid_mlp_methods
            for config in kernel_configs
        )
        if args.include_hybrid
        else ()
    )
    candidate_methods = (*kernel_methods, *hybrid_methods)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name,
                seed,
                kernel_configs,
                hybrid_mlp_methods,
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
            "budgets": list(args.budgets),
            "sigmas": list(args.sigmas),
            "rhos": list(args.rhos),
            "novelty_thresholds": list(args.novelty_thresholds),
            "ridge": args.ridge,
            "rls_delta": args.rls_delta,
            "utility_decay": args.utility_decay,
            "min_center_age": args.min_center_age,
            "input_clip": args.input_clip,
            "kernel": args.kernel,
            "bandwidth_multipliers": list(args.bandwidth_multipliers),
            "polynomial_degree": args.polynomial_degree,
            "algebraic_weight": args.algebraic_weight,
            "normalize_polynomial": args.normalize_polynomial,
            "arccosine_depth": args.arccosine_depth,
            "kernel_weight_variance": args.kernel_weight_variance,
            "kernel_bias_variance": args.kernel_bias_variance,
            "coefficient_updates": list(args.coefficient_updates),
            "kernel_lms_step_size": args.kernel_lms_step_size,
            "include_hybrid": args.include_hybrid,
            "hybrid_mlp_methods": list(hybrid_mlp_methods),
            "center_add_interval": args.center_add_interval,
            "center_add_intervals": (
                list(args.center_add_intervals)
                if args.center_add_intervals is not None
                else None
            ),
            "replace_when_full": args.replace_when_full,
        },
        "datasets": datasets_meta,
        "kernel_methods": list(kernel_methods),
        "hybrid_methods": list(hybrid_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "standalone_single_learner_budgeted_diffusion_krls_probe",
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    md_path = output_dir / "SUMMARY.md"
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

#!/usr/bin/env python3
"""D18: simple universal resource-basis learner for Step 2.

This is the direct integration candidate after D09/D10/D15-D17:

* a D10-style learned resource manager over RKHS banks for retained memory and
  algebraic structure,
* a fixed tanh/Fourier basis block for the compositional and frequency regimes,
* an optional finite polynomial block for degree-3 recursive products,
* one additive prediction and one global residual at every timestep.

There is no prediction router, no MLP expert, and no expert-selection gate.  All
blocks update every timestep; the only learned manager allocates RKHS resource.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
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
from d10_learned_kernel_resource_manager import (  # noqa: E402
    BankSpec,
    ManagedMultiBankKRLS,
    ManagerConfig,
    MultiBankState,
    make_bank_specs,
)
from d14_unified_basis_lms import (  # noqa: E402
    BasisConfig as UnifiedBasisConfig,
    BasisState as UnifiedBasisState,
    UnifiedBasisLMS,
)
from d15_groupwise_basis_lms import (  # noqa: E402
    GroupwiseBasisLMS,
    GroupwiseConfig,
    GroupwiseState,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d18_simple_universal")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d18_simple_universal.md")


@dataclass(frozen=True)
class UniversalConfig:
    """Configuration for one simple universal candidate."""

    name: str
    bank_specs: tuple[BankSpec, ...]
    manager_config: ManagerConfig
    basis_config: GroupwiseConfig
    poly_config: PolynomialConfig
    unified_config: UnifiedBasisConfig | None
    core_residual_gain: float
    basis_residual_gain: float
    poly_residual_gain: float
    unified_residual_gain: float
    prediction_core_scale: float
    prediction_basis_scale: float
    prediction_poly_scale: float
    prediction_unified_scale: float
    prediction_residual_trace_scale: float
    residual_trace_decay: float
    residual_trace_clip: float
    residual_trace_simplex_only: bool
    prediction_target_trace_scale: float
    target_trace_decay: float
    target_trace_clip: float
    target_trace_contextual: bool
    target_trace_context_power: float
    target_trace_persistence_gate: bool
    target_persistence_decay: float
    target_persistence_threshold: float
    target_persistence_power: float
    simplex_output: bool
    simplex_project_update: bool
    simplex_min_observations: int
    simplex_tolerance: float
    simplex_persistence_gate: bool
    learned_gains: bool
    gain_step_size: float
    gain_l2: float
    gain_min: float
    gain_max: float
    gain_learn_mask: tuple[bool, ...]
    component_clip: float
    residual_clip: float
    prototype_scale: float
    prototype_bandwidth: float
    prototype_min_count: int
    prototype_update_rate: float
    prototype_online: bool
    prototype_persistence_gate: bool


@dataclass
class UniversalState:
    """Mutable state for D18."""

    core_state: MultiBankState
    basis_state: GroupwiseState
    poly_state: PolynomialState
    unified_state: UnifiedBasisState | None
    residual_trace: np.ndarray
    target_trace: np.ndarray
    previous_target: np.ndarray
    target_persistence: float
    simplex_observations: int
    simplex_violations: int
    block_gains: np.ndarray | None
    prototype_counts: np.ndarray
    prototype_means: np.ndarray
    steps: int
    finite_failures: int
    mean_residual_norm: float


@dataclass(frozen=True)
class PolynomialConfig:
    """Configuration for a finite normalized polynomial residual block."""

    mode: str
    strict_degree3: bool
    max_dim: int
    degree: int
    step_size: float
    weight_decay: float
    rls_delta: float
    rls_forgetting: float
    input_clip: float
    raw_scale: float
    degree2_scale: float
    degree3_scale: float


@dataclass
class PolynomialState:
    """Mutable state for the finite polynomial block."""

    weights: np.ndarray
    covariance: np.ndarray
    steps: int
    finite_failures: int
    mean_feature_norm: float


class PolynomialResidualLMS:
    """Normalized LMS over low-degree products with optional temporal decay."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: PolynomialConfig,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.active_dim = min(self.feature_dim, self.config.max_dim)
        self.feature_dim_out = self._feature_dim()

    def _feature_dim(self) -> int:
        dim = 1 + self.active_dim
        if self.config.degree >= 2:
            dim += (self.active_dim * (self.active_dim + 1)) // 2
        if self.config.degree >= 3:
            if self.config.strict_degree3:
                dim += (
                    self.active_dim
                    * (self.active_dim - 1)
                    * (self.active_dim - 2)
                ) // 6
            else:
                dim += (
                    self.active_dim
                    * (self.active_dim + 1)
                    * (self.active_dim + 2)
                ) // 6
        return dim

    def init(self) -> PolynomialState:
        """Return initial state."""
        return PolynomialState(
            weights=np.zeros((self.feature_dim_out, self.n_heads), dtype=np.float64),
            covariance=self.config.rls_delta
            * np.eye(self.feature_dim_out, dtype=np.float64),
            steps=0,
            finite_failures=0,
            mean_feature_norm=0.0,
        )

    def _features(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        x = np.clip(x, -self.config.input_clip, self.config.input_clip)
        p = x[: self.active_dim]
        parts: list[np.ndarray] = [
            np.ones(1, dtype=np.float64),
            self.config.raw_scale * p / np.sqrt(max(self.active_dim, 1)),
        ]
        if self.config.degree >= 2:
            terms2 = np.asarray(
                [
                    float(p[i] * p[j])
                    for i in range(self.active_dim)
                    for j in range(i, self.active_dim)
                ],
                dtype=np.float64,
            )
            parts.append(
                self.config.degree2_scale
                * terms2
                / np.sqrt(max(terms2.shape[0], 1))
            )
        if self.config.degree >= 3:
            if self.config.strict_degree3:
                terms3 = np.asarray(
                    [
                        float(p[i] * p[j] * p[k])
                        for i in range(self.active_dim)
                        for j in range(i + 1, self.active_dim)
                        for k in range(j + 1, self.active_dim)
                    ],
                    dtype=np.float64,
                )
            else:
                terms3 = np.asarray(
                    [
                        float(p[i] * p[j] * p[k])
                        for i in range(self.active_dim)
                        for j in range(i, self.active_dim)
                        for k in range(j, self.active_dim)
                    ],
                    dtype=np.float64,
                )
            parts.append(
                self.config.degree3_scale
                * terms3
                / np.sqrt(max(terms3.shape[0], 1))
            )
        return np.asarray(
            np.nan_to_num(np.concatenate(parts, axis=0), copy=False),
            dtype=np.float64,
        )

    def _update_lms(
        self,
        state: PolynomialState,
        phi: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        """Apply normalized LMS update."""
        if self.config.weight_decay < 1.0:
            state.weights *= self.config.weight_decay
        state.weights += (
            self.config.step_size
            * np.outer(phi, errors)
            / (1.0 + float(phi @ phi))
        )

    def _update_rls(
        self,
        state: PolynomialState,
        phi: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        """Apply one shared-covariance RLS update."""
        p_phi = state.covariance @ phi
        denom = self.config.rls_forgetting + float(phi @ p_phi)
        if denom <= 0.0 or not np.isfinite(denom):
            state.finite_failures += 1
            return
        gain = p_phi / denom
        state.weights += self.config.step_size * np.outer(gain, errors)
        state.covariance = (
            state.covariance - np.outer(gain, p_phi)
        ) / self.config.rls_forgetting

    def predict(self, state: PolynomialState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads."""
        if self.config.step_size == 0.0:
            return np.zeros(self.n_heads, dtype=np.float64)
        return np.asarray(self._features(observation) @ state.weights)

    def step(
        self,
        state: PolynomialState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict once, then update active target heads."""
        if self.config.step_size == 0.0:
            return np.zeros(self.n_heads, dtype=np.float64), {
                "poly_feature_norm": 0.0,
                "poly_finite_failures": float(state.finite_failures),
            }
        phi = self._features(observation)
        prediction = np.asarray(phi @ state.weights)
        active = ~np.isnan(target)
        errors = np.where(active, target - prediction, 0.0)
        if self.config.mode == "lms":
            self._update_lms(state, phi, errors)
        elif self.config.mode == "rls":
            self._update_rls(state, phi, errors)
        else:
            raise ValueError(f"unknown polynomial mode {self.config.mode!r}")
        if (
            not np.all(np.isfinite(state.weights))
            or not np.all(np.isfinite(state.covariance))
        ):
            state.finite_failures += 1
            state.weights = np.nan_to_num(state.weights, copy=False)
            state.covariance = np.nan_to_num(state.covariance, copy=False)
        state.steps += 1
        norm = float(np.linalg.norm(phi))
        state.mean_feature_norm = 0.99 * state.mean_feature_norm + 0.01 * norm
        return prediction, {
            "poly_feature_norm": norm,
            "poly_finite_failures": float(state.finite_failures),
        }


class SimpleUniversalResourceBasis:
    """One additive predictor with resource-managed RKHS and tanh/Fourier basis."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: UniversalConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.core = ManagedMultiBankKRLS(
            n_heads=n_heads,
            feature_dim=feature_dim,
            bank_specs=config.bank_specs,
            manager_config=config.manager_config,
        )
        self.basis = GroupwiseBasisLMS(
            n_heads=n_heads,
            feature_dim=feature_dim,
            config=config.basis_config,
            seed=seed,
        )
        self.poly = PolynomialResidualLMS(
            n_heads=n_heads,
            feature_dim=feature_dim,
            config=config.poly_config,
        )
        self.unified = (
            None
            if config.unified_config is None
            else UnifiedBasisLMS(
                n_heads=n_heads,
                feature_dim=feature_dim,
                config=config.unified_config,
                seed=seed + 19_000,
            )
        )

    def init(self) -> UniversalState:
        """Return initial state."""
        block_gains = None
        if self.config.learned_gains:
            base = np.asarray(
                [
                    self.config.prediction_core_scale,
                    self.config.prediction_basis_scale,
                    self.config.prediction_poly_scale,
                    self.config.prediction_unified_scale,
                ],
                dtype=np.float64,
            )
            block_gains = np.repeat(base[:, None], self.n_heads, axis=1)
        return UniversalState(
            core_state=self.core.init(),
            basis_state=self.basis.init(),
            poly_state=self.poly.init(),
            unified_state=None if self.unified is None else self.unified.init(),
            residual_trace=np.zeros(self.n_heads, dtype=np.float64),
            target_trace=np.zeros(self.n_heads, dtype=np.float64),
            previous_target=np.zeros(self.n_heads, dtype=np.float64),
            target_persistence=1.0 / max(float(self.n_heads), 1.0),
            simplex_observations=0,
            simplex_violations=0,
            block_gains=block_gains,
            prototype_counts=np.zeros(self.n_heads, dtype=np.float64),
            prototype_means=np.zeros((self.n_heads, self.feature_dim), dtype=np.float64),
            steps=0,
            finite_failures=0,
            mean_residual_norm=0.0,
        )

    def _target_simplex_active(self, state: UniversalState) -> bool:
        """Return whether prior targets identify one-hot simplex geometry."""
        return (
            self.n_heads > 1
            and state.simplex_observations >= self.config.simplex_min_observations
            and state.simplex_violations == 0
        )

    def _simplex_active(self, state: UniversalState) -> bool:
        """Return whether prediction projection should use simplex geometry."""
        if not self.config.simplex_output or not self._target_simplex_active(state):
            return False
        if (
            self.config.simplex_persistence_gate
            and self._target_trace_strength(state) <= 1e-6
        ):
            return False
        return True

    def _project_output(self, state: UniversalState, prediction: np.ndarray) -> np.ndarray:
        """Apply an online-discovered simplex readout when target geometry warrants it."""
        if not self._simplex_active(state):
            return prediction
        projected = np.zeros_like(prediction)
        projected[int(np.argmax(prediction))] = 1.0
        return projected

    def _target_trace_strength(self, state: UniversalState) -> float:
        """Return a causal concentration gate for target-context memory."""
        if self.config.target_trace_persistence_gate:
            if not self._target_simplex_active(state):
                return 0.0
            chance = 1.0 / max(float(self.n_heads), 1.0)
            strength = (state.target_persistence - chance) / max(1.0 - chance, 1e-12)
            strength = float(np.clip(strength, 0.0, 1.0))
            if strength < self.config.target_persistence_threshold:
                return 0.0
            return float(strength**self.config.target_persistence_power)
        if not self.config.target_trace_contextual:
            return 1.0
        if not self._target_simplex_active(state):
            return 0.0
        total = float(np.sum(np.maximum(state.target_trace, 0.0)))
        if total <= 1e-12:
            return 0.0
        probs = np.maximum(state.target_trace, 0.0) / total
        uniform = 1.0 / max(float(self.n_heads), 1.0)
        strength = (float(np.max(probs)) - uniform) / max(1.0 - uniform, 1e-12)
        strength = float(np.clip(strength, 0.0, 1.0))
        return float(strength ** self.config.target_trace_context_power)

    def _update_simplex_detector(self, state: UniversalState, target: np.ndarray) -> None:
        """Track whether observed targets are full one-hot vectors."""
        if self.n_heads <= 1:
            return
        if not (
            self.config.simplex_output
            or self.config.target_trace_contextual
            or self.config.residual_trace_simplex_only
            or self.config.prototype_scale > 0.0
        ):
            return
        active = ~np.isnan(target)
        if not bool(np.all(active)):
            state.simplex_violations += 1
            return
        values = np.asarray(target, dtype=np.float64)
        tol = self.config.simplex_tolerance
        is_simplex = (
            abs(float(np.sum(values)) - 1.0) <= tol
            and abs(float(np.max(values)) - 1.0) <= tol
            and float(np.min(values)) >= -tol
            and float(np.max(np.minimum(np.abs(values), np.abs(values - 1.0)))) <= tol
        )
        state.simplex_observations += 1
        if not is_simplex:
            state.simplex_violations += 1

    def _update_target_persistence(self, state: UniversalState, target: np.ndarray) -> None:
        """Track one-step persistence of discovered one-hot targets."""
        if self.n_heads <= 1:
            return
        if not self.config.target_trace_persistence_gate:
            return
        active = ~np.isnan(target)
        if not bool(np.all(active)):
            return
        values = np.asarray(target, dtype=np.float64)
        tol = self.config.simplex_tolerance
        is_simplex = (
            abs(float(np.sum(values)) - 1.0) <= tol
            and abs(float(np.max(values)) - 1.0) <= tol
            and float(np.min(values)) >= -tol
            and float(np.max(np.minimum(np.abs(values), np.abs(values - 1.0)))) <= tol
        )
        if not is_simplex:
            return
        if float(np.sum(np.abs(state.previous_target))) <= tol:
            same = 1.0 / max(float(self.n_heads), 1.0)
        else:
            same = float(np.dot(state.previous_target, values))
        decay = self.config.target_persistence_decay
        state.target_persistence = decay * state.target_persistence + (1.0 - decay) * same
        state.previous_target = values.copy()

    def _prototype_prediction(
        self,
        state: UniversalState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return retained one-hot prototype scores for discovered simplex targets."""
        if self.config.prototype_scale <= 0.0 or self.n_heads <= 1:
            return np.zeros(self.n_heads, dtype=np.float64)
        if not self._target_simplex_active(state):
            return np.zeros(self.n_heads, dtype=np.float64)
        if (
            self.config.prototype_persistence_gate
            and self._target_trace_strength(state) <= 1e-6
        ):
            return np.zeros(self.n_heads, dtype=np.float64)
        ready = state.prototype_counts >= float(self.config.prototype_min_count)
        if not bool(np.any(ready)):
            return np.zeros(self.n_heads, dtype=np.float64)
        x = np.asarray(observation, dtype=np.float64)
        diffs = state.prototype_means - x[None, :]
        distances = np.mean(diffs * diffs, axis=1)
        bandwidth = max(self.config.prototype_bandwidth, 1e-12)
        logits = -distances / bandwidth
        logits = np.where(ready, logits, -np.inf)
        finite_logits = logits[np.isfinite(logits)]
        if finite_logits.size == 0:
            return np.zeros(self.n_heads, dtype=np.float64)
        shifted = logits - float(np.max(finite_logits))
        scores = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
        total = float(np.sum(scores))
        if total <= 1e-12:
            return np.zeros(self.n_heads, dtype=np.float64)
        return scores / total

    def _update_prototypes(
        self,
        state: UniversalState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Update retained class prototypes from full one-hot targets."""
        if self.config.prototype_scale <= 0.0 or self.n_heads <= 1:
            return
        if not GroupwiseBasisLMS._one_hot_target(target):
            return
        head = int(np.argmax(target))
        count = state.prototype_counts[head]
        x = np.asarray(observation, dtype=np.float64)
        state.prototype_counts[head] = count + 1.0
        rate = (
            1.0 / state.prototype_counts[head]
            if self.config.prototype_update_rate <= 0.0
            else (1.0 if count <= 0.0 else self.config.prototype_update_rate)
        )
        state.prototype_means[head] += rate * (x - state.prototype_means[head])

    def predict_parts(
        self,
        state: UniversalState,
        observation: np.ndarray,
        *,
        project: bool = True,
        include_fast_context: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return total and component predictions."""
        core_pred = self.core.predict(state.core_state, observation)
        basis_pred = self.basis.predict(state.basis_state, observation)
        poly_pred = self.poly.predict(state.poly_state, observation)
        if self.unified is None:
            unified_pred = np.zeros(self.n_heads, dtype=np.float64)
        else:
            assert state.unified_state is not None
            unified_pred = self.unified.predict(state.unified_state, observation)
        if self.config.component_clip > 0.0:
            poly_pred = np.clip(
                poly_pred,
                -self.config.component_clip,
                self.config.component_clip,
            )
            unified_pred = np.clip(
                unified_pred,
                -self.config.component_clip,
                self.config.component_clip,
            )
        components = np.stack(
            [
                np.asarray(core_pred, dtype=np.float64),
                np.asarray(basis_pred, dtype=np.float64),
                np.asarray(poly_pred, dtype=np.float64),
                np.asarray(unified_pred, dtype=np.float64),
            ],
            axis=0,
        )
        if self.config.learned_gains:
            assert state.block_gains is not None
            total = np.sum(state.block_gains * components, axis=0)
        else:
            scales = np.asarray(
                [
                    self.config.prediction_core_scale,
                    self.config.prediction_basis_scale,
                    self.config.prediction_poly_scale,
                    self.config.prediction_unified_scale,
                ],
                dtype=np.float64,
            )
            total = np.sum(scales[:, None] * components, axis=0)
        if self.config.prototype_scale > 0.0 and (
            self.config.prototype_online or not include_fast_context
        ):
            total = total + (
                self.config.prototype_scale
                * self._prototype_prediction(state, observation)
            )
        if include_fast_context:
            total = total + (
                self.config.prediction_target_trace_scale
                * self._target_trace_strength(state)
                * state.target_trace
            )
        if (
            include_fast_context
            and
            self.config.prediction_residual_trace_scale > 0.0
            and (
                not self.config.residual_trace_simplex_only
                or self._simplex_active(state)
            )
        ):
            total = total + (
                self.config.prediction_residual_trace_scale * state.residual_trace
            )
        total = np.asarray(total)
        if project:
            total = self._project_output(state, total)
        return (
            np.asarray(total),
            components[0],
            components[1],
            components[2],
            components[3],
        )

    def predict(self, state: UniversalState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads without fast temporal target/residual context."""
        total, _, _, _, _ = self.predict_parts(
            state,
            observation,
            include_fast_context=False,
        )
        return total

    def step(
        self,
        state: UniversalState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict once, then update all blocks from one global residual."""
        raw_prediction, core_pred, basis_pred, poly_pred, unified_pred = self.predict_parts(
            state,
            observation,
            project=False,
        )
        prediction = self._project_output(state, raw_prediction)
        update_prediction = prediction if self.config.simplex_project_update else raw_prediction
        active = ~np.isnan(target)
        residual = np.where(active, target - update_prediction, 0.0)
        if self.config.residual_clip > 0.0:
            residual = np.clip(
                residual,
                -self.config.residual_clip,
                self.config.residual_clip,
            )
        if (
            self.config.prediction_residual_trace_scale > 0.0
            and (
                not self.config.residual_trace_simplex_only
                or self._simplex_active(state)
            )
        ):
            state.residual_trace *= self.config.residual_trace_decay
            state.residual_trace = np.where(
                active,
                state.residual_trace
                + (1.0 - self.config.residual_trace_decay) * residual,
                state.residual_trace,
            )
            if self.config.residual_trace_clip > 0.0:
                state.residual_trace = np.clip(
                    state.residual_trace,
                    -self.config.residual_trace_clip,
                    self.config.residual_trace_clip,
                )
        if self.config.prediction_target_trace_scale > 0.0:
            state.target_trace *= self.config.target_trace_decay
            safe_target = np.where(active, target, 0.0)
            state.target_trace = np.where(
                active,
                state.target_trace
                + (1.0 - self.config.target_trace_decay) * safe_target,
                state.target_trace,
            )
            if self.config.target_trace_clip > 0.0:
                state.target_trace = np.clip(
                    state.target_trace,
                    -self.config.target_trace_clip,
                    self.config.target_trace_clip,
                )
        self._update_target_persistence(state, target)
        self._update_simplex_detector(state, target)
        self._update_prototypes(state, observation, target)
        if self.config.learned_gains:
            assert state.block_gains is not None
            components = np.stack(
                [core_pred, basis_pred, poly_pred, unified_pred],
                axis=0,
            )
            denom = 1.0 + np.sum(components * components, axis=0)
            base = np.asarray(
                [
                    self.config.prediction_core_scale,
                    self.config.prediction_basis_scale,
                    self.config.prediction_poly_scale,
                    self.config.prediction_unified_scale,
                ],
                dtype=np.float64,
            )[:, None]
            delta = (
                self.config.gain_step_size
                * (
                    components * residual[None, :] / denom[None, :]
                    - self.config.gain_l2 * (state.block_gains - base)
                )
            )
            learn_mask = np.asarray(self.config.gain_learn_mask, dtype=bool)[:, None]
            delta = np.where(learn_mask & active[None, :], delta, 0.0)
            state.block_gains += delta
            state.block_gains = np.clip(
                state.block_gains,
                self.config.gain_min,
                self.config.gain_max,
            )

        core_target = np.where(
            active,
            core_pred + self.config.core_residual_gain * residual,
            np.nan,
        )
        basis_target = np.where(
            active,
            basis_pred + self.config.basis_residual_gain * residual,
            np.nan,
        )
        poly_target = np.where(
            active,
            poly_pred + self.config.poly_residual_gain * residual,
            np.nan,
        )
        unified_target = np.where(
            active,
            unified_pred + self.config.unified_residual_gain * residual,
            np.nan,
        )
        _, core_diag = self.core.step(state.core_state, observation, core_target)
        _, basis_diag = self.basis.step(
            state.basis_state,
            observation,
            basis_target,
            decay_target=target,
        )
        _, poly_diag = self.poly.step(state.poly_state, observation, poly_target)
        if self.unified is None:
            unified_diag: dict[str, float] = {"feature_norm": 0.0}
        else:
            assert state.unified_state is not None
            _, unified_diag = self.unified.step(
                state.unified_state,
                observation,
                unified_target,
            )

        state.steps += 1
        residual_norm = float(np.linalg.norm(residual))
        state.mean_residual_norm = 0.99 * state.mean_residual_norm + 0.01 * residual_norm
        state.finite_failures = int(
            float(core_diag.get("finite_failures", 0.0))
            + float(basis_diag.get("finite_failures", 0.0))
            + float(poly_diag.get("poly_finite_failures", 0.0))
        )
        return prediction, {
            "residual_norm": residual_norm,
            "finite_failures": float(state.finite_failures),
            "active_centers": float(core_diag.get("active_centers", 0.0)),
            "raw_poly_centers": float(core_diag.get("raw_poly_centers", 0.0)),
            "algebraic_green_centers": float(
                core_diag.get("algebraic_green_centers", 0.0)
            ),
            "arccosine_centers": float(core_diag.get("arccosine_centers", 0.0)),
            "raw_poly_weight": float(core_diag.get("raw_poly_weight", 0.0)),
            "algebraic_green_weight": float(
                core_diag.get("algebraic_green_weight", 0.0)
            ),
            "arccosine_weight": float(core_diag.get("arccosine_weight", 0.0)),
            "poly_feature_norm": float(poly_diag.get("poly_feature_norm", 0.0)),
            "unified_feature_norm": float(unified_diag.get("feature_norm", 0.0)),
            "core_gain": float(
                self.config.prediction_core_scale
                if state.block_gains is None
                else np.nanmean(state.block_gains[0])
            ),
            "basis_gain": float(
                self.config.prediction_basis_scale
                if state.block_gains is None
                else np.nanmean(state.block_gains[1])
            ),
            "poly_gain": float(
                self.config.prediction_poly_scale
                if state.block_gains is None
                else np.nanmean(state.block_gains[2])
            ),
            "unified_gain": float(
                self.config.prediction_unified_scale
                if state.block_gains is None
                else np.nanmean(state.block_gains[3])
            ),
        }


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over non-NaN heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_universal_stream(
    observations: Any,
    targets: Any,
    config: UniversalConfig,
    seed: int,
) -> tuple[SimpleUniversalResourceBasis, UniversalState, np.ndarray]:
    """Run one D18 candidate."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = SimpleUniversalResourceBasis(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 16), dtype=np.float64)
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
    return learner, state, metrics


def evaluate_universal_classifier(
    learner: SimpleUniversalResourceBasis,
    state: UniversalState,
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


def universal_summary(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one D18 run."""
    entry = summarize_prequential(metrics, final_window, labels)
    entry.update(
        {
            "mean_residual_norm": float(np.mean(metrics[:, 2])),
            "finite_failures": float(metrics[-1, 3]),
            "active_centers": float(metrics[-1, 4]),
            "raw_poly_centers": float(metrics[-1, 5]),
            "algebraic_green_centers": float(metrics[-1, 6]),
            "arccosine_centers": float(metrics[-1, 7]),
            "final_raw_poly_weight": float(metrics[-1, 8]),
            "final_algebraic_green_weight": float(metrics[-1, 9]),
            "mean_poly_feature_norm": float(np.mean(metrics[:, 10])),
            "mean_unified_feature_norm": float(np.mean(metrics[:, 11])),
            "final_core_gain": float(metrics[-1, 12]),
            "final_basis_gain": float(metrics[-1, 13]),
            "final_poly_gain": float(metrics[-1, 14]),
            "final_unified_gain": float(metrics[-1, 15]),
        }
    )
    return entry


def make_basis_config(args: argparse.Namespace) -> GroupwiseConfig:
    """Return the tanh/Fourier basis block."""
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
        weight_decay=args.basis_weight_decay,
        simplex_weight_decay=args.basis_simplex_weight_decay,
    )


def make_poly_config(
    args: argparse.Namespace,
    *,
    mode: str,
    strict_degree3: bool,
    step_size: float,
    weight_decay: float,
    rls_delta: float,
    rls_forgetting: float,
    raw_scale: float,
    degree2_scale: float,
    degree3_scale: float,
) -> PolynomialConfig:
    """Return a finite normalized polynomial block config."""
    return PolynomialConfig(
        mode=mode,
        strict_degree3=strict_degree3,
        max_dim=args.poly_max_dim,
        degree=args.finite_poly_degree,
        step_size=step_size,
        weight_decay=weight_decay,
        rls_delta=rls_delta,
        rls_forgetting=rls_forgetting,
        input_clip=args.input_clip,
        raw_scale=raw_scale,
        degree2_scale=degree2_scale,
        degree3_scale=degree3_scale,
    )


def make_unified_config(args: argparse.Namespace, name: str) -> UnifiedBasisConfig:
    """Return a D14-style unified fixed-basis residual block."""
    freqs = tuple(float(item) for item in args.fourier_frequencies)
    poly_terms = (args.poly_max_dim * (args.poly_max_dim + 1)) // 2 + (
        args.poly_max_dim * (args.poly_max_dim + 1) * (args.poly_max_dim + 2)
    ) // 6
    if name == "exact_union":
        return UnifiedBasisConfig(
            name="exact_union_residual",
            step_size=args.unified_step_size,
            input_clip=args.unified_input_clip,
            poly_max_dim=args.poly_max_dim,
            poly_degree=3,
            fourier_max_dim=args.fourier_max_dim,
            fourier_frequencies=freqs,
            tanh_width=args.tanh_width,
            tanh_weight_scale=args.tanh_weight_scale,
            raw_scale=math.sqrt(max(args.poly_max_dim, 1)),
            poly_scale=math.sqrt(max(poly_terms, 1)),
            fourier_scale=math.sqrt(
                max(2 * args.fourier_max_dim * len(freqs), 1)
            ),
            tanh_scale=math.sqrt(max(args.tanh_width, 1)),
        )
    if name == "tanh_heavy":
        return UnifiedBasisConfig(
            name="tanh_heavy_residual",
            step_size=args.unified_step_size,
            input_clip=args.unified_input_clip,
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
        )
    raise ValueError(f"unknown unified basis config {name!r}")


def make_manager_config(args: argparse.Namespace) -> ManagerConfig:
    """Return the D10 manager config used by the RKHS core."""
    return ManagerConfig(
        method=args.manager_method,
        learning_rate=args.manager_learning_rate,
        discount=args.manager_discount,
        exploration=args.manager_exploration,
        utility_decay=args.manager_utility_decay,
        cost_weight=args.manager_cost_weight,
        advantage_clip=args.manager_advantage_clip,
        ucb_bonus=args.manager_ucb_bonus,
        residual_power=args.manager_residual_power,
        novelty_power=args.manager_novelty_power,
        actual_gain_weight=args.manager_actual_gain_weight,
        total_center_budget=args.total_center_budget,
        allow_rebalance=args.allow_rebalance,
        rho_span=args.manager_rho_span,
        min_rho=args.manager_min_rho,
    )


def make_configs(args: argparse.Namespace) -> list[UniversalConfig]:
    """Return candidate variants."""
    bank_specs = make_bank_specs(args)
    manager = make_manager_config(args)
    basis = make_basis_config(args)
    disabled_poly = make_poly_config(
        args,
        mode="lms",
        strict_degree3=False,
        step_size=0.0,
        weight_decay=1.0,
        rls_delta=args.finite_poly_rls_delta,
        rls_forgetting=args.finite_poly_rls_forgetting,
        raw_scale=0.0,
        degree2_scale=0.0,
        degree3_scale=0.0,
    )

    def config(
        name: str,
        core_scale: float,
        basis_scale: float,
        poly_scale: float = 0.0,
        unified_scale: float = 0.0,
        unified_name: str = "exact_union",
        residual_trace_scale: float | None = None,
        residual_trace_decay: float | None = None,
        residual_trace_clip: float | None = None,
        residual_trace_simplex_only: bool | None = None,
        target_trace_scale: float | None = None,
        target_trace_decay: float | None = None,
        target_trace_clip: float | None = None,
        target_trace_contextual: bool | None = None,
        target_trace_context_power: float | None = None,
        target_trace_persistence_gate: bool | None = None,
        target_persistence_decay: float | None = None,
        target_persistence_threshold: float | None = None,
        target_persistence_power: float | None = None,
        simplex_output: bool | None = None,
        simplex_project_update: bool | None = None,
        simplex_min_observations: int | None = None,
        simplex_tolerance: float | None = None,
        simplex_persistence_gate: bool | None = None,
        learned_gains: bool = False,
        gain_step_size: float | None = None,
        gain_l2: float | None = None,
        gain_min: float | None = None,
        gain_max: float | None = None,
        gain_learn_mask: tuple[bool, ...] | None = None,
        component_clip: float | None = None,
        poly_mode: str = "lms",
        poly_strict_degree3: bool = False,
        poly_step_size: float = 0.0,
        poly_weight_decay: float = 1.0,
        poly_rls_delta: float | None = None,
        poly_rls_forgetting: float | None = None,
        poly_raw_scale: float = 0.0,
        poly_degree2_scale: float = 0.0,
        poly_degree3_scale: float = 0.0,
        basis_tanh_width: int | None = None,
        basis_tanh_step_size: float | None = None,
        basis_weight_decay: float | None = None,
        basis_simplex_weight_decay: float | None = None,
        prototype_scale: float | None = None,
        prototype_bandwidth: float | None = None,
        prototype_min_count: int | None = None,
        prototype_update_rate: float | None = None,
        prototype_online: bool | None = None,
        prototype_persistence_gate: bool | None = None,
    ) -> UniversalConfig:
        basis_for_config = replace(
            basis,
            tanh_width=basis.tanh_width if basis_tanh_width is None else basis_tanh_width,
            tanh_step_size=(
                basis.tanh_step_size
                if basis_tanh_step_size is None
                else basis_tanh_step_size
            ),
            weight_decay=(
                basis.weight_decay if basis_weight_decay is None else basis_weight_decay
            ),
            simplex_weight_decay=(
                basis.simplex_weight_decay
                if basis_simplex_weight_decay is None
                else basis_simplex_weight_decay
            ),
        )
        poly = (
            disabled_poly
            if poly_step_size == 0.0 or poly_scale == 0.0
            else make_poly_config(
                args,
                mode=poly_mode,
                strict_degree3=poly_strict_degree3,
                step_size=poly_step_size,
                weight_decay=poly_weight_decay,
                rls_delta=(
                    args.finite_poly_rls_delta
                    if poly_rls_delta is None
                    else poly_rls_delta
                ),
                rls_forgetting=(
                    args.finite_poly_rls_forgetting
                    if poly_rls_forgetting is None
                    else poly_rls_forgetting
                ),
                raw_scale=poly_raw_scale,
                degree2_scale=poly_degree2_scale,
                degree3_scale=poly_degree3_scale,
            )
        )
        unified = (
            None
            if unified_scale == 0.0
            else make_unified_config(args, unified_name)
        )
        return UniversalConfig(
            name=name,
            bank_specs=bank_specs,
            manager_config=manager,
            basis_config=basis_for_config,
            poly_config=poly,
            unified_config=unified,
            core_residual_gain=args.core_residual_gain,
            basis_residual_gain=args.basis_residual_gain,
            poly_residual_gain=args.poly_residual_gain,
            unified_residual_gain=args.unified_residual_gain,
            prediction_core_scale=core_scale,
            prediction_basis_scale=basis_scale,
            prediction_poly_scale=poly_scale,
            prediction_unified_scale=unified_scale,
            prediction_residual_trace_scale=(
                args.residual_trace_scale
                if residual_trace_scale is None
                else residual_trace_scale
            ),
            residual_trace_decay=(
                args.residual_trace_decay
                if residual_trace_decay is None
                else residual_trace_decay
            ),
            residual_trace_clip=(
                args.residual_trace_clip
                if residual_trace_clip is None
                else residual_trace_clip
            ),
            residual_trace_simplex_only=(
                args.residual_trace_simplex_only
                if residual_trace_simplex_only is None
                else residual_trace_simplex_only
            ),
            prediction_target_trace_scale=(
                args.target_trace_scale
                if target_trace_scale is None
                else target_trace_scale
            ),
            target_trace_decay=(
                args.target_trace_decay
                if target_trace_decay is None
                else target_trace_decay
            ),
            target_trace_clip=(
                args.target_trace_clip
                if target_trace_clip is None
                else target_trace_clip
            ),
            target_trace_contextual=(
                args.target_trace_contextual
                if target_trace_contextual is None
                else target_trace_contextual
            ),
            target_trace_context_power=(
                args.target_trace_context_power
                if target_trace_context_power is None
                else target_trace_context_power
            ),
            target_trace_persistence_gate=(
                args.target_trace_persistence_gate
                if target_trace_persistence_gate is None
                else target_trace_persistence_gate
            ),
            target_persistence_decay=(
                args.target_persistence_decay
                if target_persistence_decay is None
                else target_persistence_decay
            ),
            target_persistence_threshold=(
                args.target_persistence_threshold
                if target_persistence_threshold is None
                else target_persistence_threshold
            ),
            target_persistence_power=(
                args.target_persistence_power
                if target_persistence_power is None
                else target_persistence_power
            ),
            simplex_output=args.simplex_output if simplex_output is None else simplex_output,
            simplex_project_update=(
                args.simplex_project_update
                if simplex_project_update is None
                else simplex_project_update
            ),
            simplex_min_observations=(
                args.simplex_min_observations
                if simplex_min_observations is None
                else simplex_min_observations
            ),
            simplex_tolerance=(
                args.simplex_tolerance
                if simplex_tolerance is None
                else simplex_tolerance
            ),
            simplex_persistence_gate=(
                args.simplex_persistence_gate
                if simplex_persistence_gate is None
                else simplex_persistence_gate
            ),
            learned_gains=learned_gains,
            gain_step_size=(
                args.gain_step_size if gain_step_size is None else gain_step_size
            ),
            gain_l2=args.gain_l2 if gain_l2 is None else gain_l2,
            gain_min=args.gain_min if gain_min is None else gain_min,
            gain_max=args.gain_max if gain_max is None else gain_max,
            gain_learn_mask=(
                (True, True, True, True)
                if gain_learn_mask is None
                else gain_learn_mask
            ),
            component_clip=args.component_clip if component_clip is None else component_clip,
            residual_clip=args.residual_clip,
            prototype_scale=(
                args.prototype_scale if prototype_scale is None else prototype_scale
            ),
            prototype_bandwidth=(
                args.prototype_bandwidth
                if prototype_bandwidth is None
                else prototype_bandwidth
            ),
            prototype_min_count=(
                args.prototype_min_count
                if prototype_min_count is None
                else prototype_min_count
            ),
            prototype_update_rate=(
                args.prototype_update_rate
                if prototype_update_rate is None
                else prototype_update_rate
            ),
            prototype_online=(
                args.prototype_online
                if prototype_online is None
                else prototype_online
            ),
            prototype_persistence_gate=(
                args.prototype_persistence_gate
                if prototype_persistence_gate is None
                else prototype_persistence_gate
            ),
        )

    variants = {
        "simple": config("simple", 1.0, 1.0),
        "basis_half": config("basis_half", 1.0, 0.5),
        "basis_quarter": config("basis_quarter", 1.0, 0.25),
        "basis_third": config("basis_third", 1.0, 1.0 / 3.0),
        "basis_0p4": config("basis_0p4", 1.0, 0.4),
        "core_0p75_basis_0p4": config("core_0p75_basis_0p4", 0.75, 0.4),
        "core_0p5_basis_0p4": config("core_0p5_basis_0p4", 0.5, 0.4),
        "core_0p25_basis_0p6": config("core_0p25_basis_0p6", 0.25, 0.6),
        "core_0p25_basis_0p8": config("core_0p25_basis_0p8", 0.25, 0.8),
        "core_0p1_basis_1p0": config("core_0p1_basis_1p0", 0.1, 1.0),
        "basis_only": config("basis_only", 0.0, 1.0),
        "core_0p5_basis_0p4_poly_0p25": config(
            "core_0p5_basis_0p4_poly_0p25",
            0.5,
            0.4,
            poly_scale=0.25,
            poly_step_size=args.finite_poly_step_size,
            poly_weight_decay=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.5,
            poly_degree3_scale=2.0,
        ),
        "core_0p5_basis_0p4_poly_0p4": config(
            "core_0p5_basis_0p4_poly_0p4",
            0.5,
            0.4,
            poly_scale=0.4,
            poly_step_size=args.finite_poly_step_size,
            poly_weight_decay=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.5,
            poly_degree3_scale=2.0,
        ),
        "core_0p5_basis_0p4_poly_0p4_decay": config(
            "core_0p5_basis_0p4_poly_0p4_decay",
            0.5,
            0.4,
            poly_scale=0.4,
            poly_step_size=args.finite_poly_step_size,
            poly_weight_decay=args.finite_poly_weight_decay,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.5,
            poly_degree3_scale=2.0,
        ),
        "core_0p5_basis_0p4_poly_0p6_decay": config(
            "core_0p5_basis_0p4_poly_0p6_decay",
            0.5,
            0.4,
            poly_scale=0.6,
            poly_step_size=args.finite_poly_step_size,
            poly_weight_decay=args.finite_poly_weight_decay,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.5,
            poly_degree3_scale=2.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p25": config(
            "core_0p5_basis_0p4_poly_rls_0p25",
            0.5,
            0.4,
            poly_scale=0.25,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p05": config(
            "core_0p5_basis_0p4_poly_rls_0p05",
            0.5,
            0.4,
            poly_scale=0.05,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p1": config(
            "core_0p5_basis_0p4_poly_rls_0p1",
            0.5,
            0.4,
            poly_scale=0.1,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p15": config(
            "core_0p5_basis_0p4_poly_rls_0p15",
            0.5,
            0.4,
            poly_scale=0.15,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p4": config(
            "core_0p5_basis_0p4_poly_rls_0p4",
            0.5,
            0.4,
            poly_scale=0.4,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p5_basis_0p4_poly_rls_0p6": config(
            "core_0p5_basis_0p4_poly_rls_0p6",
            0.5,
            0.4,
            poly_scale=0.6,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p25_basis_0p6_poly_rls_0p05": config(
            "core_0p25_basis_0p6_poly_rls_0p05",
            0.25,
            0.6,
            poly_scale=0.05,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p25_basis_0p6_poly_rls_0p1": config(
            "core_0p25_basis_0p6_poly_rls_0p1",
            0.25,
            0.6,
            poly_scale=0.1,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p25_basis_0p6_poly_rls_0p15": config(
            "core_0p25_basis_0p6_poly_rls_0p15",
            0.25,
            0.6,
            poly_scale=0.15,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "core_0p25_basis_0p6_unified_0p05": config(
            "core_0p25_basis_0p6_unified_0p05",
            0.25,
            0.6,
            unified_scale=0.05,
        ),
        "core_0p25_basis_0p6_unified_0p1": config(
            "core_0p25_basis_0p6_unified_0p1",
            0.25,
            0.6,
            unified_scale=0.1,
        ),
        "core_0p25_basis_0p6_unified_0p2": config(
            "core_0p25_basis_0p6_unified_0p2",
            0.25,
            0.6,
            unified_scale=0.2,
        ),
        "core_0p5_basis_0p4_unified_0p1": config(
            "core_0p5_basis_0p4_unified_0p1",
            0.5,
            0.4,
            unified_scale=0.1,
        ),
        "gain_lowcore_unified": config(
            "gain_lowcore_unified",
            0.25,
            0.6,
            unified_scale=0.05,
            learned_gains=True,
        ),
        "gain_lowcore_poly_unified": config(
            "gain_lowcore_poly_unified",
            0.25,
            0.6,
            poly_scale=0.05,
            unified_scale=0.05,
            learned_gains=True,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_safe_digits": config(
            "gain_safe_digits",
            0.5,
            0.4,
            unified_scale=0.05,
            learned_gains=True,
            gain_l2=0.02,
        ),
        "gain_fixedcore_poly_unified_0p005": config(
            "gain_fixedcore_poly_unified_0p005",
            0.5,
            0.4,
            poly_scale=0.005,
            unified_scale=0.005,
            learned_gains=True,
            gain_l2=0.001,
            gain_learn_mask=(False, False, True, True),
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_fixedcore_poly_unified_0p01": config(
            "gain_fixedcore_poly_unified_0p01",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_l2=0.001,
            gain_learn_mask=(False, False, True, True),
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_fixedcore_poly_unified_0p02": config(
            "gain_fixedcore_poly_unified_0p02",
            0.5,
            0.4,
            poly_scale=0.02,
            unified_scale=0.02,
            learned_gains=True,
            gain_l2=0.001,
            gain_learn_mask=(False, False, True, True),
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_safecore_poly_unified_0p005": config(
            "gain_safecore_poly_unified_0p005",
            0.5,
            0.4,
            poly_scale=0.005,
            unified_scale=0.005,
            learned_gains=True,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_safecore_poly_unified_0p01": config(
            "gain_safecore_poly_unified_0p01",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "gain_safecore_poly_unified_0p02": config(
            "gain_safecore_poly_unified_0p02",
            0.5,
            0.4,
            poly_scale=0.02,
            unified_scale=0.02,
            learned_gains=True,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_canonical": config(
            "step2_canonical",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.075,
            component_clip=1.0,
            target_trace_scale=4.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            target_trace_persistence_gate=True,
            target_persistence_decay=0.95,
            target_persistence_threshold=0.5,
            target_persistence_power=3.0,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_persistence_gate=True,
            basis_tanh_width=128,
            basis_tanh_step_size=0.62,
            basis_weight_decay=0.9975,
            basis_simplex_weight_decay=1.0,
            prototype_scale=4.0,
            prototype_bandwidth=0.1,
            prototype_min_count=1,
            prototype_update_rate=0.05,
            prototype_online=False,
            prototype_persistence_gate=True,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_basis_0p5": config(
            "step2_basis_0p5",
            0.5,
            0.5,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_basis_0p6": config(
            "step2_basis_0p6",
            0.5,
            0.6,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_core_0p4_basis_0p5": config(
            "step2_core_0p4_basis_0p5",
            0.4,
            0.5,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_gain_l2_0p1": config(
            "step2_gain_l2_0p1",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.1,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_no_unified": config(
            "step2_no_unified",
            0.5,
            0.4,
            poly_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_no_poly": config(
            "step2_no_poly",
            0.5,
            0.4,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
        ),
        "step2_distilled_memory": config(
            "step2_distilled_memory",
            0.5,
            0.4,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.075,
            component_clip=1.0,
            target_trace_scale=4.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            target_trace_persistence_gate=True,
            target_persistence_decay=0.95,
            target_persistence_threshold=0.5,
            target_persistence_power=3.0,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_persistence_gate=True,
            basis_tanh_width=128,
            basis_tanh_step_size=0.62,
            basis_weight_decay=0.9975,
            basis_simplex_weight_decay=1.0,
            prototype_scale=4.0,
            prototype_bandwidth=0.1,
            prototype_min_count=1,
            prototype_update_rate=0.05,
            prototype_online=False,
            prototype_persistence_gate=True,
        ),
        "step2_distilled_memory_nogates": config(
            "step2_distilled_memory_nogates",
            0.5,
            0.4,
            unified_scale=0.01,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.075,
            component_clip=1.0,
            target_trace_scale=4.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            target_trace_persistence_gate=False,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_persistence_gate=False,
            basis_tanh_width=128,
            basis_tanh_step_size=0.62,
            basis_weight_decay=0.9975,
            basis_simplex_weight_decay=1.0,
            prototype_scale=4.0,
            prototype_bandwidth=0.1,
            prototype_min_count=1,
            prototype_update_rate=0.05,
            prototype_online=True,
            prototype_persistence_gate=False,
        ),
        "step2_fast_residual_trace": config(
            "step2_fast_residual_trace",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            residual_trace_scale=1.0,
            residual_trace_decay=0.95,
            residual_trace_clip=0.5,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_target_trace": config(
            "step2_target_trace",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            target_trace_scale=1.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_context_trace": config(
            "step2_context_trace",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            target_trace_scale=2.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            target_trace_contextual=True,
            target_trace_context_power=2.0,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_tolerance=1e-4,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_persistent_trace": config(
            "step2_persistent_trace",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            target_trace_scale=4.0,
            target_trace_decay=0.95,
            target_trace_clip=1.0,
            target_trace_persistence_gate=True,
            target_persistence_decay=0.95,
            target_persistence_power=6.0,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_tolerance=1e-4,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.1,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_simplex": config(
            "step2_simplex",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_tolerance=1e-4,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
        "step2_simplex_trace": config(
            "step2_simplex_trace",
            0.5,
            0.4,
            poly_scale=0.01,
            unified_scale=0.01,
            residual_trace_scale=12.0,
            residual_trace_decay=0.98,
            residual_trace_clip=2.0,
            residual_trace_simplex_only=True,
            simplex_output=True,
            simplex_project_update=False,
            simplex_min_observations=20,
            simplex_tolerance=1e-4,
            learned_gains=True,
            gain_step_size=0.2,
            gain_l2=0.05,
            component_clip=1.0,
            poly_mode="rls",
            poly_strict_degree3=True,
            poly_step_size=1.0,
            poly_raw_scale=1.0,
            poly_degree2_scale=0.0,
            poly_degree3_scale=1.0,
        ),
    }
    if args.configs == "all":
        return list(variants.values())
    selected: list[UniversalConfig] = []
    for raw in args.configs.split(","):
        name = raw.strip()
        if name:
            selected.append(variants[name])
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
        "# D18 Simple Universal Resource-Basis Results",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Candidate configs: "
            f"{', '.join(cfg['candidate_methods'])}."
        ),
        "",
        "Candidate prediction is one additive model: resource-managed RKHS core "
        "plus tanh/Fourier and optional finite polynomial bases. There is no "
        "output router and no MLP expert.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Centers | Runtime s |",
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
                "`final_window_mse` best-D18-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-D18 counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-D18-vs-best-MLP diff: "
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
    configs: list[UniversalConfig],
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
        method = f"d18_{config.name}"
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        learner, state, metrics = run_universal_stream(
            observations,
            targets,
            config,
            seed=seed + 130_000 + offset,
        )
        methods[method] = universal_summary(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(evaluate_universal_classifier(learner, state, x_test, y_test))

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
    parser.add_argument("--datasets", default="synthetic_compositional,digits_label_drift")
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
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rls-delta", type=float, default=100.0)
    parser.add_argument("--kernel-utility-decay", type=float, default=0.99)
    parser.add_argument("--min-center-age", type=int, default=50)
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument("--bandwidth-multipliers", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--kernel-weight-variance", type=float, default=2.0)
    parser.add_argument("--kernel-bias-variance", type=float, default=0.1)
    parser.add_argument("--kernel-lms-step-size", type=float, default=0.5)
    parser.add_argument("--replace-when-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--raw-poly-budget", type=int, default=64)
    parser.add_argument("--algebraic-budget", type=int, default=128)
    parser.add_argument("--arccosine-budget", type=int, default=128)
    parser.add_argument("--raw-poly-novelty", type=float, default=1e-5)
    parser.add_argument("--algebraic-novelty", type=float, default=1e-3)
    parser.add_argument("--arccosine-novelty", type=float, default=1e-3)
    parser.add_argument("--raw-poly-add-interval", type=int, default=1)
    parser.add_argument("--algebraic-add-interval", type=int, default=8)
    parser.add_argument("--arccosine-add-interval", type=int, default=2)
    parser.add_argument("--raw-poly-update-scale", type=float, default=0.55)
    parser.add_argument("--algebraic-update-scale", type=float, default=0.45)
    parser.add_argument("--arccosine-update-scale", type=float, default=0.45)
    parser.add_argument("--raw-poly-cost", type=float, default=1.0)
    parser.add_argument("--algebraic-cost", type=float, default=1.4)
    parser.add_argument("--arccosine-cost", type=float, default=1.8)
    parser.add_argument("--raw-poly-utility-scale", type=float, default=1.0)
    parser.add_argument("--algebraic-utility-scale", type=float, default=1.05)
    parser.add_argument("--arccosine-utility-scale", type=float, default=1.15)
    parser.add_argument("--algebraic-weight", type=float, default=0.75)
    parser.add_argument("--arccosine-depth", type=int, default=1)
    parser.add_argument("--total-center-budget", type=int, default=320)
    parser.add_argument("--allow-rebalance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--manager-method", default="learned_softmax")
    parser.add_argument("--manager-learning-rate", type=float, default=4.0)
    parser.add_argument("--manager-discount", type=float, default=0.995)
    parser.add_argument("--manager-exploration", type=float, default=0.05)
    parser.add_argument("--manager-utility-decay", type=float, default=0.98)
    parser.add_argument("--manager-cost-weight", type=float, default=0.01)
    parser.add_argument("--manager-advantage-clip", type=float, default=5.0)
    parser.add_argument("--manager-ucb-bonus", type=float, default=0.02)
    parser.add_argument("--manager-residual-power", type=float, default=1.0)
    parser.add_argument("--manager-novelty-power", type=float, default=1.0)
    parser.add_argument("--manager-actual-gain-weight", type=float, default=2.0)
    parser.add_argument("--manager-rho-span", type=float, default=0.0)
    parser.add_argument("--manager-min-rho", type=float, default=0.97)
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
    parser.add_argument("--basis-weight-decay", type=float, default=1.0)
    parser.add_argument("--basis-simplex-weight-decay", type=float, default=None)
    parser.add_argument("--finite-poly-degree", type=int, default=3)
    parser.add_argument("--finite-poly-step-size", type=float, default=0.5)
    parser.add_argument("--finite-poly-weight-decay", type=float, default=0.995)
    parser.add_argument("--finite-poly-rls-delta", type=float, default=100.0)
    parser.add_argument("--finite-poly-rls-forgetting", type=float, default=0.99)
    parser.add_argument("--unified-step-size", type=float, default=0.4)
    parser.add_argument("--unified-input-clip", type=float, default=3.0)
    parser.add_argument("--residual-trace-scale", type=float, default=0.0)
    parser.add_argument("--residual-trace-decay", type=float, default=0.95)
    parser.add_argument("--residual-trace-clip", type=float, default=0.5)
    parser.add_argument(
        "--residual-trace-simplex-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-trace-scale", type=float, default=0.0)
    parser.add_argument("--target-trace-decay", type=float, default=0.95)
    parser.add_argument("--target-trace-clip", type=float, default=1.0)
    parser.add_argument(
        "--target-trace-contextual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-trace-context-power", type=float, default=2.0)
    parser.add_argument(
        "--target-trace-persistence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target-persistence-decay", type=float, default=0.95)
    parser.add_argument("--target-persistence-threshold", type=float, default=0.0)
    parser.add_argument("--target-persistence-power", type=float, default=3.0)
    parser.add_argument("--simplex-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--simplex-project-update",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--simplex-min-observations", type=int, default=20)
    parser.add_argument("--simplex-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--simplex-persistence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gain-step-size", type=float, default=0.2)
    parser.add_argument("--gain-l2", type=float, default=0.005)
    parser.add_argument("--gain-min", type=float, default=0.0)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--component-clip", type=float, default=5.0)
    parser.add_argument("--core-residual-gain", type=float, default=1.0)
    parser.add_argument("--basis-residual-gain", type=float, default=1.0)
    parser.add_argument("--poly-residual-gain", type=float, default=1.0)
    parser.add_argument("--unified-residual-gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=5.0)
    parser.add_argument("--prototype-scale", type=float, default=0.0)
    parser.add_argument("--prototype-bandwidth", type=float, default=1.0)
    parser.add_argument("--prototype-min-count", type=int, default=1)
    parser.add_argument("--prototype-update-rate", type=float, default=0.0)
    parser.add_argument(
        "--prototype-online",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prototype-persistence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--configs", default="simple")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments."""
    if args.steps <= 0 or args.n_seeds <= 0 or args.final_window <= 0:
        raise ValueError("steps, n-seeds, and final-window must be positive")
    if args.tanh_width <= 0:
        raise ValueError("--tanh-width must be positive")
    if not 0.0 < args.basis_weight_decay <= 1.0:
        raise ValueError("--basis-weight-decay must be in (0, 1]")
    if (
        args.basis_simplex_weight_decay is not None
        and not 0.0 < args.basis_simplex_weight_decay <= 1.0
    ):
        raise ValueError("--basis-simplex-weight-decay must be in (0, 1]")
    if args.finite_poly_degree < 1:
        raise ValueError("--finite-poly-degree must be positive")
    if args.finite_poly_step_size < 0.0:
        raise ValueError("--finite-poly-step-size must be non-negative")
    if not 0.0 < args.finite_poly_weight_decay <= 1.0:
        raise ValueError("--finite-poly-weight-decay must be in (0, 1]")
    if args.finite_poly_rls_delta <= 0.0:
        raise ValueError("--finite-poly-rls-delta must be positive")
    if not 0.0 < args.finite_poly_rls_forgetting <= 1.0:
        raise ValueError("--finite-poly-rls-forgetting must be in (0, 1]")
    if args.unified_step_size < 0.0:
        raise ValueError("--unified-step-size must be non-negative")
    if args.unified_input_clip <= 0.0:
        raise ValueError("--unified-input-clip must be positive")
    if args.residual_trace_scale < 0.0:
        raise ValueError("--residual-trace-scale must be non-negative")
    if not 0.0 <= args.residual_trace_decay < 1.0:
        raise ValueError("--residual-trace-decay must be in [0, 1)")
    if args.residual_trace_clip < 0.0:
        raise ValueError("--residual-trace-clip must be non-negative")
    if args.target_trace_scale < 0.0:
        raise ValueError("--target-trace-scale must be non-negative")
    if not 0.0 <= args.target_trace_decay < 1.0:
        raise ValueError("--target-trace-decay must be in [0, 1)")
    if args.target_trace_clip < 0.0:
        raise ValueError("--target-trace-clip must be non-negative")
    if args.target_trace_context_power < 0.0:
        raise ValueError("--target-trace-context-power must be non-negative")
    if not 0.0 <= args.target_persistence_decay < 1.0:
        raise ValueError("--target-persistence-decay must be in [0, 1)")
    if not 0.0 <= args.target_persistence_threshold <= 1.0:
        raise ValueError("--target-persistence-threshold must be in [0, 1]")
    if args.target_persistence_power < 0.0:
        raise ValueError("--target-persistence-power must be non-negative")
    if args.simplex_min_observations < 0:
        raise ValueError("--simplex-min-observations must be non-negative")
    if args.simplex_tolerance < 0.0:
        raise ValueError("--simplex-tolerance must be non-negative")
    if args.gain_step_size < 0.0:
        raise ValueError("--gain-step-size must be non-negative")
    if args.gain_l2 < 0.0:
        raise ValueError("--gain-l2 must be non-negative")
    if args.gain_min > args.gain_max:
        raise ValueError("--gain-min cannot exceed --gain-max")
    if args.component_clip < 0.0:
        raise ValueError("--component-clip must be non-negative")
    if args.poly_residual_gain < 0.0:
        raise ValueError("--poly-residual-gain must be non-negative")
    if args.unified_residual_gain < 0.0:
        raise ValueError("--unified-residual-gain must be non-negative")
    if args.prototype_scale < 0.0:
        raise ValueError("--prototype-scale must be non-negative")
    if args.prototype_bandwidth <= 0.0:
        raise ValueError("--prototype-bandwidth must be positive")
    if args.prototype_min_count <= 0:
        raise ValueError("--prototype-min-count must be positive")
    if not 0.0 <= args.prototype_update_rate <= 1.0:
        raise ValueError("--prototype-update-rate must be in [0, 1]")
    if args.total_center_budget > (
        args.raw_poly_budget + args.algebraic_budget + args.arccosine_budget
    ):
        raise ValueError("--total-center-budget cannot exceed summed per-bank budgets")


def main() -> None:
    """Run D18."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.raw_poly_budget = 12
        args.algebraic_budget = 12
        args.arccosine_budget = 12
        args.total_center_budget = 24
        args.tanh_width = 64
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_configs(args)
    candidate_methods = tuple(f"d18_{config.name}" for config in configs)
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
            "core_residual_gain": args.core_residual_gain,
            "basis_residual_gain": args.basis_residual_gain,
            "poly_residual_gain": args.poly_residual_gain,
            "unified_residual_gain": args.unified_residual_gain,
            "finite_poly_degree": args.finite_poly_degree,
            "basis_weight_decay": args.basis_weight_decay,
            "basis_simplex_weight_decay": args.basis_simplex_weight_decay,
            "finite_poly_step_size": args.finite_poly_step_size,
            "finite_poly_weight_decay": args.finite_poly_weight_decay,
            "finite_poly_rls_delta": args.finite_poly_rls_delta,
            "finite_poly_rls_forgetting": args.finite_poly_rls_forgetting,
            "unified_step_size": args.unified_step_size,
            "unified_input_clip": args.unified_input_clip,
            "residual_trace_scale": args.residual_trace_scale,
            "residual_trace_decay": args.residual_trace_decay,
            "residual_trace_clip": args.residual_trace_clip,
            "residual_trace_simplex_only": args.residual_trace_simplex_only,
            "target_trace_scale": args.target_trace_scale,
            "target_trace_decay": args.target_trace_decay,
            "target_trace_clip": args.target_trace_clip,
            "target_trace_contextual": args.target_trace_contextual,
            "target_trace_context_power": args.target_trace_context_power,
            "target_trace_persistence_gate": args.target_trace_persistence_gate,
            "target_persistence_decay": args.target_persistence_decay,
            "target_persistence_threshold": args.target_persistence_threshold,
            "target_persistence_power": args.target_persistence_power,
            "simplex_output": args.simplex_output,
            "simplex_project_update": args.simplex_project_update,
            "simplex_min_observations": args.simplex_min_observations,
            "simplex_tolerance": args.simplex_tolerance,
            "simplex_persistence_gate": args.simplex_persistence_gate,
            "gain_step_size": args.gain_step_size,
            "gain_l2": args.gain_l2,
            "gain_min": args.gain_min,
            "gain_max": args.gain_max,
            "component_clip": args.component_clip,
            "prototype_scale": args.prototype_scale,
            "prototype_bandwidth": args.prototype_bandwidth,
            "prototype_min_count": args.prototype_min_count,
            "prototype_update_rate": args.prototype_update_rate,
            "prototype_online": args.prototype_online,
            "prototype_persistence_gate": args.prototype_persistence_gate,
            "configs": args.configs,
            "candidate_methods": list(candidate_methods),
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "simple_additive_resource_basis_no_router",
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

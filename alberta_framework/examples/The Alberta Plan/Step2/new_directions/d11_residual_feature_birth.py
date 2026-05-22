#!/usr/bin/env python3
"""D11: residual-driven feature birth for Step 2.

This is a standalone non-router learner probe.  The model owns a single
prediction: a linear/RLS readout over a bounded feature library.  At every
online step it predicts and updates the readout.  At scheduled allocation
events it proposes candidate features and admits the candidates whose recent
activation is most correlated with recent prequential residuals.

The candidate families are deliberately heterogeneous:

* random tanh ridges;
* random Fourier features;
* polynomial interaction monomials;
* local radial bumps centered on high-residual samples;
* compositions of already useful features.

This tests whether residual evidence can construct useful recursive features
without delegating prediction to an MLP router or stacker.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
NEW_DIRECTIONS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, NEW_DIRECTIONS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from d07_budgeted_kernel_recursive import (  # type: ignore[attr-defined]  # noqa: E402
    DIGITS_REGIMES,
    MLP_METHODS,
    aggregate_records,
    evaluate_mlp_classifier,
    expand_dataset_names,
    make_dataset,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d11_residual_feature_birth")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d11_residual_feature_birth.md"
)

FeatureFamilies = tuple[str, ...]


@dataclass(frozen=True)
class BirthConfig:
    """Configuration for one residual feature-birth learner."""

    name: str
    budget: int
    families: FeatureFamilies
    base_degree: int
    max_base_pair_dim: int
    allocate_interval: int
    births_per_event: int
    candidates_per_event: int
    buffer_size: int
    rho: float
    rls_delta: float
    utility_decay: float
    min_feature_age: int
    input_clip: float
    feature_clip: float
    tanh_weight_scale: float
    fourier_weight_scale: float
    bump_sigma: float
    deep_tanh_hidden: int
    composition_top_k: int
    random_seed_offset: int
    replace_when_full: bool


@dataclass
class FeatureSpec:
    """One feature slot in the bounded feature library."""

    kind: str
    birth_step: int
    mean: float = 0.0
    scale: float = 1.0
    weight: np.ndarray | None = None
    matrix: np.ndarray | None = None
    inner_bias: np.ndarray | None = None
    bias: float = 0.0
    indices: tuple[int, ...] = ()
    center: np.ndarray | None = None
    sigma: float = 1.0
    parents: tuple[int, ...] = ()
    age: int = 0
    activation_ema: float = 0.0
    coefficient_ema: float = 0.0
    score: float = 0.0

    def signature(self) -> str:
        """Return a stable coarse duplicate key for this feature."""
        if self.kind in {"bias"}:
            return self.kind
        if self.kind in {"raw", "square", "monomial"}:
            return f"{self.kind}:{self.indices}"
        if self.kind.startswith("compose"):
            return f"{self.kind}:{self.parents}:{self.bias:.3f}"
        if self.weight is not None:
            quant = tuple(np.round(self.weight, 3).reshape(-1).tolist())
            if self.matrix is not None:
                matrix_quant = tuple(np.round(self.matrix, 3).reshape(-1).tolist())
                return f"{self.kind}:{matrix_quant}:{quant}:{self.bias:.3f}"
            return f"{self.kind}:{quant}:{self.bias:.3f}"
        if self.center is not None:
            quant = tuple(np.round(self.center, 2).tolist())
            return f"{self.kind}:{quant}:{self.sigma:.3f}"
        return f"{self.kind}:{self.birth_step}:{self.score:.6f}"


@dataclass
class ResidualFeatureBirthState:
    """Mutable NumPy state for one residual feature-birth learner."""

    specs: list[FeatureSpec]
    weights: np.ndarray
    p_matrix: np.ndarray
    active_count: int
    protected_slots: set[int]
    buffer_x: np.ndarray
    buffer_phi: np.ndarray
    buffer_residual: np.ndarray
    buffer_active: np.ndarray
    buffer_count: int
    buffer_pos: int
    steps: int
    births: int
    replacements: int
    skipped_births: int
    finite_failures: int
    allocation_events: int
    score_sum: float
    residual_norm_sum: float


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over active target heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def sanitize_observation(observation: np.ndarray, input_clip: float) -> np.ndarray:
    """Convert an observation to clipped float64 for feature evaluation."""
    x = np.asarray(observation, dtype=np.float64)
    if input_clip > 0.0:
        x = np.clip(x, -input_clip, input_clip)
    return x


def normalize_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return normalized candidate values and the frozen normalizer."""
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1e-6 or not np.isfinite(scale):
        scale = 1.0
    normalized = (values - mean) / scale
    return normalized, mean, scale


def clipped_feature(value: float | np.ndarray, feature_clip: float) -> float | np.ndarray:
    """Clip feature values when requested."""
    if feature_clip <= 0.0:
        return value
    return np.clip(value, -feature_clip, feature_clip)


class ResidualFeatureBirthLearner:
    """Linear/RLS learner with online residual-driven feature construction."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: BirthConfig,
        seed: int,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.rng = np.random.default_rng(seed + config.random_seed_offset)

    def init(self) -> ResidualFeatureBirthState:
        """Create state with fixed base features and empty birth capacity."""
        specs = self._base_specs()
        if len(specs) >= self.config.budget:
            specs = specs[: self.config.budget]
        active_count = len(specs)
        return ResidualFeatureBirthState(
            specs=specs,
            weights=np.zeros((self.config.budget, self.n_heads), dtype=np.float64),
            p_matrix=np.eye(self.config.budget, dtype=np.float64)
            * self.config.rls_delta,
            active_count=active_count,
            protected_slots=set(range(active_count)),
            buffer_x=np.zeros(
                (self.config.buffer_size, self.feature_dim), dtype=np.float64
            ),
            buffer_phi=np.zeros(
                (self.config.buffer_size, self.config.budget), dtype=np.float64
            ),
            buffer_residual=np.zeros(
                (self.config.buffer_size, self.n_heads), dtype=np.float64
            ),
            buffer_active=np.zeros(
                (self.config.buffer_size, self.n_heads), dtype=bool
            ),
            buffer_count=0,
            buffer_pos=0,
            steps=0,
            births=0,
            replacements=0,
            skipped_births=0,
            finite_failures=0,
            allocation_events=0,
            score_sum=0.0,
            residual_norm_sum=0.0,
        )

    def _base_specs(self) -> list[FeatureSpec]:
        """Return fixed base features available from the first timestep."""
        specs = [FeatureSpec(kind="bias", birth_step=0, scale=1.0)]
        specs.extend(
            FeatureSpec(kind="raw", birth_step=0, indices=(idx,))
            for idx in range(self.feature_dim)
        )
        if self.config.base_degree >= 2:
            specs.extend(
                FeatureSpec(kind="square", birth_step=0, indices=(idx,))
                for idx in range(self.feature_dim)
            )
            if self.feature_dim <= self.config.max_base_pair_dim:
                specs.extend(
                    FeatureSpec(kind="monomial", birth_step=0, indices=(i, j))
                    for i in range(self.feature_dim)
                    for j in range(i + 1, self.feature_dim)
                )
        return specs

    def _feature_value_from_phi(
        self,
        spec: FeatureSpec,
        x: np.ndarray,
        phi: np.ndarray,
    ) -> float:
        """Evaluate one feature for a single observation."""
        if spec.kind == "bias":
            raw = 1.0
        elif spec.kind == "raw":
            raw = float(x[spec.indices[0]])
        elif spec.kind == "square":
            raw = float(x[spec.indices[0]] ** 2)
        elif spec.kind == "monomial":
            raw = float(np.prod(x[list(spec.indices)]))
        elif spec.kind == "tanh":
            assert spec.weight is not None
            raw = float(np.tanh(spec.weight @ x + spec.bias))
        elif spec.kind == "fourier_sin":
            assert spec.weight is not None
            raw = float(np.sin(spec.weight @ x + spec.bias))
        elif spec.kind == "fourier_cos":
            assert spec.weight is not None
            raw = float(np.cos(spec.weight @ x + spec.bias))
        elif spec.kind == "bump":
            assert spec.center is not None
            mean_sq = float(np.mean((x - spec.center) ** 2))
            raw = float(np.exp(-mean_sq / (2.0 * max(spec.sigma, 1e-6) ** 2)))
        elif spec.kind == "deep_tanh":
            assert spec.matrix is not None
            assert spec.inner_bias is not None
            assert spec.weight is not None
            inner = np.tanh(spec.matrix @ x + spec.inner_bias)
            raw = float(np.tanh(spec.weight @ inner + spec.bias))
        elif spec.kind == "compose_product":
            raw = float(np.prod(phi[list(spec.parents)]))
        elif spec.kind == "compose_tanh":
            assert spec.weight is not None
            parent_values = phi[list(spec.parents)]
            raw = float(np.tanh(spec.weight[: len(parent_values)] @ parent_values + spec.bias))
        else:
            raise ValueError(f"unknown feature kind {spec.kind!r}")
        normalized = (raw - spec.mean) / max(spec.scale, 1e-6)
        return float(clipped_feature(normalized, self.config.feature_clip))

    def _feature_matrix_for_specs(
        self,
        specs: list[FeatureSpec],
        x_matrix: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a list of ordered feature specs on a batch of observations."""
        phi_matrix = np.zeros((x_matrix.shape[0], len(specs)), dtype=np.float64)
        for row_idx, x in enumerate(x_matrix):
            for col_idx, spec in enumerate(specs):
                phi_matrix[row_idx, col_idx] = self._feature_value_from_phi(
                    spec,
                    x,
                    phi_matrix[row_idx],
                )
        return phi_matrix

    def features(
        self,
        state: ResidualFeatureBirthState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all active features for a single observation."""
        x = sanitize_observation(observation, self.config.input_clip)
        phi = np.zeros(state.active_count, dtype=np.float64)
        for idx, spec in enumerate(state.specs[: state.active_count]):
            phi[idx] = self._feature_value_from_phi(spec, x, phi)
        return phi

    def predict(
        self,
        state: ResidualFeatureBirthState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Predict all heads without modifying state."""
        phi = self.features(state, observation)
        return np.asarray(phi @ state.weights[: state.active_count])

    def step(
        self,
        state: ResidualFeatureBirthState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, update RLS, record residuals, and allocate features online."""
        x = sanitize_observation(observation, self.config.input_clip)
        phi = self.features(state, x)
        prediction = np.asarray(phi @ state.weights[: state.active_count])
        active = ~np.isnan(target)
        safe_target = np.where(active, target, 0.0)
        residual = np.where(active, safe_target - prediction, 0.0)

        if state.active_count > 0:
            p_active = state.p_matrix[: state.active_count, : state.active_count]
            p_phi = p_active @ phi
            denom = self.config.rho + float(phi @ p_phi)
            if denom <= 1e-12 or not np.isfinite(denom):
                state.finite_failures += 1
            else:
                gain = p_phi / denom
                state.weights[: state.active_count] += np.outer(gain, residual)
                next_p = (
                    p_active - np.outer(gain, phi @ p_active)
                ) / self.config.rho
                state.p_matrix[: state.active_count, : state.active_count] = (
                    0.5 * (next_p + next_p.T)
                )

        if not np.all(np.isfinite(state.weights[: state.active_count])):
            state.finite_failures += 1
            state.weights[: state.active_count] = np.nan_to_num(
                state.weights[: state.active_count],
                copy=False,
            )

        self._update_feature_utilities(state, phi)
        self._record_residual(state, x, phi, residual, active)
        if (
            state.steps > 0
            and state.steps % self.config.allocate_interval == 0
            and state.buffer_count >= max(12, min(32, self.config.buffer_size // 2))
        ):
            self._allocate_from_residuals(state)
        state.residual_norm_sum += float(np.linalg.norm(residual[active]))
        state.steps += 1
        diagnostics = {
            "active_features": float(state.active_count),
            "births": float(state.births),
            "replacements": float(state.replacements),
            "skipped_births": float(state.skipped_births),
            "finite_failures": float(state.finite_failures),
            "allocation_events": float(state.allocation_events),
            "mean_birth_score": state.score_sum / max(state.births, 1),
            "mean_residual_norm": state.residual_norm_sum / max(state.steps + 1, 1),
        }
        return prediction, diagnostics

    def _update_feature_utilities(
        self,
        state: ResidualFeatureBirthState,
        phi: np.ndarray,
    ) -> None:
        """Update feature age, activation, and coefficient traces."""
        decay = self.config.utility_decay
        coeff = np.mean(np.abs(state.weights[: state.active_count]), axis=1)
        for idx, spec in enumerate(state.specs[: state.active_count]):
            spec.age += 1
            spec.activation_ema = decay * spec.activation_ema + (1.0 - decay) * abs(
                float(phi[idx])
            )
            spec.coefficient_ema = decay * spec.coefficient_ema + (1.0 - decay) * float(
                coeff[idx]
            )

    def _record_residual(
        self,
        state: ResidualFeatureBirthState,
        x: np.ndarray,
        phi: np.ndarray,
        residual: np.ndarray,
        active: np.ndarray,
    ) -> None:
        """Push one residual record into the circular evidence buffer."""
        pos = state.buffer_pos
        state.buffer_x[pos] = x
        state.buffer_phi[pos] = 0.0
        state.buffer_phi[pos, : phi.shape[0]] = phi
        state.buffer_residual[pos] = residual
        state.buffer_active[pos] = active
        state.buffer_pos = (pos + 1) % self.config.buffer_size
        state.buffer_count = min(state.buffer_count + 1, self.config.buffer_size)

    def _buffer_view(
        self,
        state: ResidualFeatureBirthState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return residual buffer contents in chronological order."""
        count = state.buffer_count
        if count < self.config.buffer_size:
            return (
                state.buffer_x[:count],
                state.buffer_phi[:count],
                state.buffer_residual[:count],
                state.buffer_active[:count],
            )
        order = np.concatenate(
            [
                np.arange(state.buffer_pos, self.config.buffer_size),
                np.arange(0, state.buffer_pos),
            ]
        )
        return (
            state.buffer_x[order],
            state.buffer_phi[order],
            state.buffer_residual[order],
            state.buffer_active[order],
        )

    def _candidate_score(
        self,
        values: np.ndarray,
        residual: np.ndarray,
        active: np.ndarray,
    ) -> float:
        """Score a candidate by normalized recent residual covariance."""
        valid_rows = np.any(active, axis=1)
        if np.sum(valid_rows) < 4:
            return 0.0
        v = values[valid_rows].astype(np.float64)
        r = residual[valid_rows].astype(np.float64)
        a = active[valid_rows]
        v = v - np.mean(v)
        v_norm = float(np.sqrt(np.sum(v * v))) + 1e-12
        score_sq = 0.0
        for head in range(self.n_heads):
            mask = a[:, head]
            if np.sum(mask) < 4:
                continue
            vh = v[mask]
            rh = r[mask, head] - np.mean(r[mask, head])
            denom = v_norm * float(np.sqrt(np.sum(rh * rh)) + 1e-12)
            corr = abs(float(np.sum(vh * rh)) / denom)
            rms = float(np.sqrt(np.mean(r[mask, head] ** 2)))
            score_sq += (corr * rms) ** 2
        return float(np.sqrt(score_sq))

    def _allocate_from_residuals(self, state: ResidualFeatureBirthState) -> None:
        """Create feature candidates and admit the highest residual scores."""
        state.allocation_events += 1
        x_buf, phi_buf, residual, active = self._buffer_view(state)
        existing_matrix = phi_buf[:, : state.active_count]
        candidates = self._propose_candidates(
            state,
            x_buf,
            residual,
            active,
            existing_matrix,
        )
        seen = {spec.signature() for spec in state.specs[: state.active_count]}
        scored: list[tuple[float, FeatureSpec]] = []
        for spec, raw_values in candidates:
            values, mean, scale = normalize_values(raw_values)
            score = self._candidate_score(values, residual, active)
            if score <= 1e-8 or not np.isfinite(score):
                continue
            next_spec = replace(spec, mean=mean, scale=scale, score=score)
            signature = next_spec.signature()
            if signature in seen:
                continue
            seen.add(signature)
            scored.append((score, next_spec))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, spec in scored[: self.config.births_per_event]:
            if self._install_feature(state, spec):
                state.score_sum += score

    def _propose_candidates(
        self,
        state: ResidualFeatureBirthState,
        x_buf: np.ndarray,
        residual: np.ndarray,
        active: np.ndarray,
        feature_matrix: np.ndarray,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Generate candidate features from all enabled families."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        per_family = max(
            1,
            self.config.candidates_per_event // max(len(self.config.families), 1),
        )
        if "poly" in self.config.families:
            candidates.extend(self._poly_candidates(x_buf, per_family))
        if "imprint" in self.config.families:
            candidates.extend(self._imprint_candidates(x_buf, residual, active, per_family))
        if "tanh" in self.config.families:
            candidates.extend(self._tanh_candidates(x_buf, per_family))
        if "fourier" in self.config.families:
            candidates.extend(self._fourier_candidates(x_buf, per_family))
        if "bump" in self.config.families:
            candidates.extend(self._bump_candidates(x_buf, residual, per_family))
        if "deep" in self.config.families:
            candidates.extend(self._deep_tanh_candidates(x_buf, per_family))
        if "compose" in self.config.families and state.active_count >= 2:
            candidates.extend(
                self._compose_candidates(state, feature_matrix, per_family)
            )
        self.rng.shuffle(candidates)
        return candidates[: self.config.candidates_per_event]

    def _poly_candidates(
        self,
        x_buf: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose deterministic and random polynomial interactions."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        if self.feature_dim <= 8:
            combos: list[tuple[int, ...]] = []
            for degree in (2, 3, 4):
                combos.extend(combinations_with_replacement(range(self.feature_dim), degree))
            self.rng.shuffle(combos)
            selected = combos[: max(count, min(len(combos), count * 2))]
        else:
            selected = [
                tuple(
                    sorted(
                        self.rng.integers(
                            0,
                            self.feature_dim,
                            size=int(self.rng.choice((2, 3))),
                        ).tolist()
                    )
                )
                for _ in range(count)
            ]
        for indices in selected:
            values = np.prod(x_buf[:, list(indices)], axis=1)
            spec = FeatureSpec(
                kind="monomial",
                birth_step=0,
                indices=tuple(int(idx) for idx in indices),
            )
            candidates.append((spec, values))
        return candidates

    def _tanh_candidates(
        self,
        x_buf: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose random tanh ridge features."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        scale = self.config.tanh_weight_scale / math.sqrt(max(self.feature_dim, 1))
        for _ in range(count):
            weight = self.rng.normal(0.0, scale, size=self.feature_dim)
            bias = float(self.rng.normal(0.0, 0.75))
            values = np.tanh(x_buf @ weight + bias)
            candidates.append(
                (
                    FeatureSpec(
                        kind="tanh",
                        birth_step=0,
                        weight=weight.astype(np.float64),
                        bias=bias,
                    ),
                    values,
                )
            )
        return candidates

    def _imprint_candidates(
        self,
        x_buf: np.ndarray,
        residual: np.ndarray,
        active: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose tanh/Fourier units aligned with recent residual gradients."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        if x_buf.shape[0] < 8:
            return candidates
        x_centered = x_buf - np.mean(x_buf, axis=0, keepdims=True)
        per_head: list[np.ndarray] = []
        for head in range(self.n_heads):
            mask = active[:, head]
            if np.sum(mask) < 8:
                continue
            r = residual[mask, head]
            r = r - np.mean(r)
            w = x_centered[mask].T @ r
            norm = float(np.linalg.norm(w))
            if norm > 1e-8 and np.isfinite(norm):
                per_head.append(w / norm)
        self.rng.shuffle(per_head)
        for direction in per_head[: max(1, count // 4)]:
            projection = x_buf @ direction
            quantiles = np.quantile(projection, [0.2, 0.5, 0.8])
            for scale in (0.75, 1.5, 3.0):
                weight = (
                    scale
                    * self.config.tanh_weight_scale
                    * direction
                    / math.sqrt(max(self.feature_dim, 1))
                )
                for quantile in quantiles:
                    bias = -float(scale * quantile)
                    tanh_values = np.tanh(x_buf @ weight + bias)
                    candidates.append(
                        (
                            FeatureSpec(
                                kind="tanh",
                                birth_step=0,
                                weight=weight.astype(np.float64),
                                bias=bias,
                            ),
                            tanh_values,
                        )
                    )
                phase = x_buf @ (scale * self.config.fourier_weight_scale * direction)
                candidates.append(
                    (
                        FeatureSpec(
                            kind="fourier_sin",
                            birth_step=0,
                            weight=(
                                scale * self.config.fourier_weight_scale * direction
                            ).astype(np.float64),
                            bias=0.0,
                        ),
                        np.sin(phase),
                    )
                )
        return candidates[:count]

    def _fourier_candidates(
        self,
        x_buf: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose random Fourier features."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        scale = self.config.fourier_weight_scale / math.sqrt(max(self.feature_dim, 1))
        for _ in range(max(1, count // 2)):
            weight = self.rng.normal(0.0, scale, size=self.feature_dim)
            bias = float(self.rng.uniform(-math.pi, math.pi))
            phase = x_buf @ weight + bias
            candidates.append(
                (
                    FeatureSpec(
                        kind="fourier_sin",
                        birth_step=0,
                        weight=weight.astype(np.float64),
                        bias=bias,
                    ),
                    np.sin(phase),
                )
            )
            candidates.append(
                (
                    FeatureSpec(
                        kind="fourier_cos",
                        birth_step=0,
                        weight=weight.astype(np.float64),
                        bias=bias,
                    ),
                    np.cos(phase),
                )
            )
        return candidates

    def _bump_candidates(
        self,
        x_buf: np.ndarray,
        residual: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose local radial bumps centered on high-residual samples."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        residual_norm = np.linalg.norm(residual, axis=1)
        if residual_norm.size == 0:
            return candidates
        top_k = min(max(count, 1), residual_norm.shape[0])
        top_indices = np.argpartition(residual_norm, -top_k)[-top_k:]
        self.rng.shuffle(top_indices)
        for idx in top_indices[:count]:
            center = x_buf[int(idx)].copy()
            sigma = self.config.bump_sigma
            if sigma <= 0.0 and x_buf.shape[0] > 1:
                distances = np.sqrt(np.mean((x_buf - center) ** 2, axis=1))
                sigma = float(np.median(distances[distances > 1e-6]))
            sigma = max(float(sigma), 0.25)
            mean_sq = np.mean((x_buf - center) ** 2, axis=1)
            values = np.exp(-mean_sq / (2.0 * sigma * sigma))
            candidates.append(
                (
                    FeatureSpec(
                        kind="bump",
                        birth_step=0,
                        center=center,
                        sigma=sigma,
                    ),
                    values,
                )
            )
        return candidates

    def _deep_tanh_candidates(
        self,
        x_buf: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose small two-layer tanh operator features."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        hidden = max(int(self.config.deep_tanh_hidden), 2)
        inner_scale = self.config.tanh_weight_scale / math.sqrt(max(self.feature_dim, 1))
        outer_scale = 1.0 / math.sqrt(float(hidden))
        for _ in range(count):
            matrix = self.rng.normal(0.0, inner_scale, size=(hidden, self.feature_dim))
            inner_bias = self.rng.normal(0.0, 0.35, size=hidden)
            weight = self.rng.normal(0.0, outer_scale, size=hidden)
            bias = float(self.rng.normal(0.0, 0.35))
            inner = np.tanh(x_buf @ matrix.T + inner_bias)
            values = np.tanh(inner @ weight + bias)
            candidates.append(
                (
                    FeatureSpec(
                        kind="deep_tanh",
                        birth_step=0,
                        matrix=matrix.astype(np.float64),
                        inner_bias=inner_bias.astype(np.float64),
                        weight=weight.astype(np.float64),
                        bias=bias,
                    ),
                    values,
                )
            )
        return candidates

    def _compose_candidates(
        self,
        state: ResidualFeatureBirthState,
        feature_matrix: np.ndarray,
        count: int,
    ) -> list[tuple[FeatureSpec, np.ndarray]]:
        """Propose products and tanh transforms of useful existing features."""
        candidates: list[tuple[FeatureSpec, np.ndarray]] = []
        utilities = np.asarray(
            [
                spec.activation_ema * spec.coefficient_ema
                for spec in state.specs[: state.active_count]
            ],
            dtype=np.float64,
        )
        if utilities.size == 0:
            return candidates
        top_k = min(self.config.composition_top_k, utilities.shape[0])
        parent_slots = np.argsort(utilities)[-top_k:]
        parent_slots = parent_slots[parent_slots < state.active_count]
        if parent_slots.shape[0] < 2:
            return candidates
        for _ in range(count):
            left, right = self.rng.choice(parent_slots, size=2, replace=False)
            if left == right:
                continue
            parents = tuple(sorted((int(left), int(right))))
            if self.rng.random() < 0.5:
                values = feature_matrix[:, parents[0]] * feature_matrix[:, parents[1]]
                candidates.append(
                    (
                        FeatureSpec(
                            kind="compose_product",
                            birth_step=0,
                            parents=parents,
                        ),
                        values,
                    )
                )
            else:
                weight = self.rng.normal(0.0, 1.0, size=2)
                bias = float(self.rng.normal(0.0, 0.25))
                values = np.tanh(feature_matrix[:, list(parents)] @ weight + bias)
                candidates.append(
                    (
                        FeatureSpec(
                            kind="compose_tanh",
                            birth_step=0,
                            parents=parents,
                            weight=weight.astype(np.float64),
                            bias=bias,
                        ),
                        values,
                    )
                )
        return candidates

    def _install_feature(
        self,
        state: ResidualFeatureBirthState,
        spec: FeatureSpec,
    ) -> bool:
        """Install one born feature, replacing the weakest mature slot if needed."""
        installed = replace(spec, birth_step=state.steps, age=0)
        if state.active_count < self.config.budget:
            idx = state.active_count
            state.specs.append(installed)
            state.active_count += 1
            state.births += 1
        elif self.config.replace_when_full:
            replace_idx = self._replacement_index(state)
            if replace_idx is None:
                state.skipped_births += 1
                return False
            idx = replace_idx
            state.specs[idx] = installed
            state.replacements += 1
            state.births += 1
        else:
            state.skipped_births += 1
            return False

        for parent in installed.parents:
            state.protected_slots.add(parent)
        state.weights[idx] = 0.0
        state.p_matrix[idx, :] = 0.0
        state.p_matrix[:, idx] = 0.0
        state.p_matrix[idx, idx] = self.config.rls_delta
        return True

    def _replacement_index(self, state: ResidualFeatureBirthState) -> int | None:
        """Choose a weak mature born feature that is not a composition parent."""
        candidates: list[tuple[float, int]] = []
        for idx, spec in enumerate(state.specs[: state.active_count]):
            if idx in state.protected_slots:
                continue
            if spec.age < self.config.min_feature_age:
                continue
            utility = spec.activation_ema * spec.coefficient_ema + 0.1 * spec.score
            candidates.append((float(utility), idx))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]


def run_birth_stream(
    observations: jax.Array,
    targets: jax.Array,
    config: BirthConfig,
    seed: int,
) -> tuple[ResidualFeatureBirthLearner, ResidualFeatureBirthState, np.ndarray]:
    """Run one residual feature-birth configuration on a materialized stream."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = ResidualFeatureBirthLearner(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
        seed=seed,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 8), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["active_features"]
        metrics[idx, 3] = diagnostics["births"]
        metrics[idx, 4] = diagnostics["replacements"]
        metrics[idx, 5] = diagnostics["mean_birth_score"]
        metrics[idx, 6] = diagnostics["allocation_events"]
        metrics[idx, 7] = diagnostics["finite_failures"]
    return learner, state, metrics


def evaluate_birth_classifier(
    learner: ResidualFeatureBirthLearner,
    state: ResidualFeatureBirthState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final residual-birth classifier on held-out digits."""
    n_classes = int(np.max(y_test)) + 1
    targets = np.eye(n_classes, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def method_name(config: BirthConfig) -> str:
    """Return a compact stable method name."""
    fam = "".join(family[0] for family in config.families)
    return (
        f"rfb_{config.name}_{fam}_b{config.budget}_i{config.allocate_interval}_"
        f"add{config.births_per_event}"
    )


def named_birth_config(name: str) -> BirthConfig:
    """Return one named ablation/configuration."""
    base = BirthConfig(
        name=name,
        budget=256,
        families=("poly", "tanh", "fourier", "bump", "compose"),
        base_degree=2,
        max_base_pair_dim=10,
        allocate_interval=20,
        births_per_event=8,
        candidates_per_event=240,
        buffer_size=160,
        rho=0.995,
        rls_delta=50.0,
        utility_decay=0.995,
        min_feature_age=80,
        input_clip=5.0,
        feature_clip=5.0,
        tanh_weight_scale=2.0,
        fourier_weight_scale=2.5,
        bump_sigma=0.0,
        deep_tanh_hidden=4,
        composition_top_k=48,
        random_seed_offset=91_000,
        replace_when_full=True,
    )
    if name == "canonical":
        return base
    if name == "tanh_comp":
        return replace(
            base,
            families=("tanh", "compose"),
            candidates_per_event=220,
            births_per_event=10,
            tanh_weight_scale=2.25,
        )
    if name == "wide_tanh_comp":
        return replace(
            base,
            name=name,
            budget=384,
            families=("tanh", "compose"),
            candidates_per_event=320,
            births_per_event=12,
            allocate_interval=15,
            tanh_weight_scale=2.25,
            composition_top_k=80,
            random_seed_offset=92_000,
        )
    if name == "no_compose":
        return replace(
            base,
            families=("poly", "tanh", "fourier", "bump"),
            random_seed_offset=93_000,
        )
    if name == "poly_only":
        return replace(
            base,
            budget=192,
            families=("poly",),
            births_per_event=8,
            candidates_per_event=220,
            random_seed_offset=94_000,
        )
    if name == "fourier_tanh":
        return replace(
            base,
            families=("tanh", "fourier"),
            candidates_per_event=240,
            births_per_event=10,
            random_seed_offset=95_000,
        )
    if name == "stable_tanh":
        return replace(
            base,
            name=name,
            budget=256,
            families=("tanh", "fourier"),
            allocate_interval=30,
            births_per_event=4,
            candidates_per_event=320,
            buffer_size=192,
            rho=1.0,
            rls_delta=1.0,
            feature_clip=3.0,
            tanh_weight_scale=1.75,
            fourier_weight_scale=1.75,
            replace_when_full=False,
            random_seed_offset=96_000,
        )
    if name == "stable_comp":
        return replace(
            base,
            name=name,
            budget=256,
            families=("tanh", "fourier", "compose"),
            allocate_interval=30,
            births_per_event=4,
            candidates_per_event=320,
            buffer_size=192,
            rho=1.0,
            rls_delta=1.0,
            feature_clip=3.0,
            tanh_weight_scale=1.75,
            fourier_weight_scale=1.75,
            composition_top_k=48,
            replace_when_full=False,
            random_seed_offset=97_000,
        )
    if name == "conservative":
        return replace(
            base,
            name=name,
            budget=192,
            families=("poly", "tanh", "fourier", "compose"),
            allocate_interval=50,
            births_per_event=3,
            candidates_per_event=300,
            buffer_size=192,
            rho=1.0,
            rls_delta=0.5,
            feature_clip=3.0,
            tanh_weight_scale=1.5,
            fourier_weight_scale=1.5,
            composition_top_k=32,
            replace_when_full=False,
            random_seed_offset=98_000,
        )
    if name == "deep_birth":
        return replace(
            base,
            name=name,
            budget=384,
            families=("deep", "tanh", "fourier"),
            allocate_interval=15,
            births_per_event=8,
            candidates_per_event=480,
            buffer_size=240,
            rho=1.0,
            rls_delta=0.5,
            feature_clip=3.0,
            tanh_weight_scale=1.75,
            fourier_weight_scale=1.75,
            deep_tanh_hidden=4,
            replace_when_full=False,
            random_seed_offset=99_000,
        )
    if name == "deep_comp":
        return replace(
            base,
            name=name,
            budget=384,
            families=("deep", "tanh", "fourier", "compose"),
            allocate_interval=15,
            births_per_event=8,
            candidates_per_event=480,
            buffer_size=240,
            rho=1.0,
            rls_delta=0.5,
            feature_clip=3.0,
            tanh_weight_scale=1.75,
            fourier_weight_scale=1.75,
            deep_tanh_hidden=4,
            composition_top_k=64,
            replace_when_full=False,
            random_seed_offset=100_000,
        )
    if name == "imprint":
        return replace(
            base,
            name=name,
            budget=256,
            families=("imprint", "tanh", "fourier", "compose"),
            allocate_interval=20,
            births_per_event=5,
            candidates_per_event=260,
            buffer_size=192,
            rho=1.0,
            rls_delta=0.75,
            feature_clip=3.0,
            tanh_weight_scale=2.0,
            fourier_weight_scale=1.5,
            composition_top_k=48,
            replace_when_full=False,
            random_seed_offset=101_000,
        )
    if name == "imprint_fast":
        return replace(
            base,
            name=name,
            budget=320,
            families=("imprint", "tanh", "fourier"),
            allocate_interval=15,
            births_per_event=6,
            candidates_per_event=320,
            buffer_size=192,
            rho=1.0,
            rls_delta=0.75,
            feature_clip=3.0,
            tanh_weight_scale=2.0,
            fourier_weight_scale=1.5,
            replace_when_full=False,
            random_seed_offset=102_000,
        )
    raise ValueError(f"unknown config name {name!r}")


def parse_config_names(spec: str) -> tuple[str, ...]:
    """Parse named configuration spec."""
    names = tuple(item.strip() for item in spec.split(",") if item.strip())
    valid = {
        "canonical",
        "tanh_comp",
        "wide_tanh_comp",
        "no_compose",
        "poly_only",
        "fourier_tanh",
        "stable_tanh",
        "stable_comp",
        "conservative",
        "deep_birth",
        "deep_comp",
        "imprint",
        "imprint_fast",
    }
    unknown = sorted(set(names).difference(valid))
    if unknown:
        raise ValueError(f"unknown --configs entries {unknown}; valid={sorted(valid)}")
    return names


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    birth_configs: list[BirthConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all baselines and residual-birth learners for one paired seed."""
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
            methods[method].update(
                evaluate_mlp_classifier(learner, state, x_test, y_test)
            )

    for config in birth_configs:
        name = method_name(config)
        print(f"{dataset_name} seed={seed}: running {name}")
        t0 = time.time()
        birth_learner, birth_state, metrics = run_birth_stream(
            observations,
            targets,
            config,
            seed + 70_000,
        )
        methods[name] = summarize_prequential(metrics, args.final_window, labels)
        methods[name].update(
            {
                "runtime_s": float(time.time() - t0),
                "active_features": float(birth_state.active_count),
                "births": float(birth_state.births),
                "replacements": float(birth_state.replacements),
                "skipped_births": float(birth_state.skipped_births),
                "allocation_events": float(birth_state.allocation_events),
                "mean_birth_score": float(
                    birth_state.score_sum / max(birth_state.births, 1)
                ),
                "mean_residual_norm": float(
                    birth_state.residual_norm_sum / max(birth_state.steps, 1)
                ),
                "finite_failures": float(birth_state.finite_failures),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[name].update(
                evaluate_birth_classifier(birth_learner, birth_state, x_test, y_test)
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


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a detailed Markdown assessment."""
    cfg = results["config"]
    lines = [
        "# D11 Residual Feature Birth Results",
        "",
        "## Algorithm",
        "",
        (
            "D11 is a single predictor: a bounded linear/RLS readout over a "
            "feature library.  It predicts before updating, updates every "
            "timestep, stores recent residuals, and periodically creates new "
            "features whose activations have high normalized covariance with "
            "the residual buffer.  There is no MLP router, no stacker, and no "
            "offline target access beyond the online residuals generated by the "
            "current learner."
        ),
        "",
        "Feature families tested: random tanh ridges, Fourier features, "
        "polynomial monomials, high-residual radial bumps, and compositions of "
        "currently useful features.  Positive paired differences below favor "
        "the residual-birth learner.",
        "",
        (
            f"Protocol: datasets={cfg['datasets']}, seeds={cfg['n_seeds']}, "
            f"steps={cfg['steps']}, final_window={cfg['final_window']}, "
            f"configs={cfg['configs']}."
        ),
        "",
        "## Results",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Active features | Births | Replacements | Runtime s |",
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
                f"{metric_cell(row, 'active_features')} | "
                f"{metric_cell(row, 'births')} | "
                f"{metric_cell(row, 'replacements')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-residual-birth-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best residual-birth counts {best['best_kernel_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_kernel_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-residual-birth-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_kernel']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_kernel']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Assessment",
            "",
            (
                "This result should be read as a mechanism test.  Promotion to "
                "canonical Step 2 would require one fixed D11 configuration to "
                "beat the best fair MLP on the broad benchmark suite, without "
                "per-dataset selection.  If the best-config row wins but no "
                "single configuration wins broadly, the evidence is only that "
                "residual-guided feature birth has useful headroom."
            ),
            "",
            "## Output Files",
            "",
            f"- JSON: `{results['output_json']}`",
            f"- Summary: `{results['output_summary']}`",
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
        default="synthetic_compositional",
        help="Comma-separated datasets or D07 aliases.",
    )
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
    parser.add_argument(
        "--configs",
        default="canonical,tanh_comp,no_compose,poly_only,fourier_tanh",
        help=(
            "Comma-separated named D11 configs: canonical, tanh_comp, "
            "wide_tanh_comp, no_compose, poly_only, fourier_tanh, "
            "stable_tanh, stable_comp, conservative, deep_birth, deep_comp, "
            "imprint, imprint_fast."
        ),
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
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")


def main() -> None:
    """Run D11 and write JSON/Markdown outputs."""
    args = parse_args()
    if args.smoke:
        args.datasets = "synthetic_compositional"
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.configs = "canonical"
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    config_names = parse_config_names(args.configs)
    birth_configs = [named_birth_config(name) for name in config_names]
    candidate_methods = tuple(method_name(config) for config in birth_configs)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name,
                seed,
                birth_configs,
                args,
            )
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta

    aggregate = aggregate_records(records, candidate_methods)
    results: dict[str, Any] = {
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
            "configs": list(config_names),
            "birth_configs": [
                {
                    "method": method_name(config),
                    "name": config.name,
                    "budget": config.budget,
                    "families": list(config.families),
                    "allocate_interval": config.allocate_interval,
                    "births_per_event": config.births_per_event,
                    "candidates_per_event": config.candidates_per_event,
                    "buffer_size": config.buffer_size,
                    "rho": config.rho,
                    "rls_delta": config.rls_delta,
                }
                for config in birth_configs
            ],
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate,
        "wall_clock_s": time.time() - t0,
        "evidence_level": "standalone_single_predictor_residual_feature_birth_probe",
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    summary_path = output_dir / "SUMMARY.md"
    results["output_json"] = str(json_path)
    results["output_summary"] = str(summary_path)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(summary_path, results)
    if args.note_path:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

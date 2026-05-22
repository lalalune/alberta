#!/usr/bin/env python3
"""D12: budgeted spectral/tensor learner for Step 2.

This runner tests a single non-MLP learner whose representation is a compact
online basis rather than a neural trunk or router.  The learner builds a fixed
pool of spectral/tensor candidate features, maintains a budgeted active subset,
and fits the active representation with RLS, normalized LMS, or an Autostep-like
per-feature update.

The feature pool is deliberately heterogeneous but still one predictor:

* Chebyshev/power polynomial chaos terms.
* Sparse ANOVA products over low-order monomials.
* Fourier and random Fourier features.
* TensorSketch-style compressed interaction blocks.
* Random Chebyshev polynomial-chaos products.
* Ridgelet/Hermite spectral features and tensor-CP interaction ridges.
* Fixed two-level compositional ridgelets with a linear online readout.
* Optional local Gaussian wavelet bumps.

At every online step all candidate utility traces are updated, active features
are fitted, and the resource manager may promote or replace features under the
same loss.  There is no dataset router, MLP residual, or per-regime branch.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import combinations, product
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

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d12_spectral_tensor_universal")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d12_spectral_tensor_universal.md"
)

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
class SpectralTensorConfig:
    """Configuration for one budgeted spectral/tensor learner."""

    basis_family: str
    budget: int
    candidate_count: int
    degree: int
    interaction_order: int
    update_rule: str
    rho: float
    rls_delta: float
    nlms_step_size: float
    autostep_mu: float
    autostep_tau: float
    autostep_init_step_size: float
    utility_decay: float
    min_feature_age: int
    replacement_interval: int
    replace_margin: float
    promotions_per_step: int
    score_threshold: float
    input_clip: float
    input_scale: float
    feature_clip: float
    rff_count: int
    rff_scale: float
    anova_count: int
    tensor_sketch_dim: int
    chaos_count: int
    ridgelet_count: int
    ridgelet_scale: float
    tensor_cp_count: int
    tensor_cp_scale: float
    deep_ridgelet_count: int
    deep_ridgelet_inner: int
    deep_ridgelet_scale: float
    wavelet_count: int
    wavelet_scales: tuple[float, ...]
    fourier_frequencies: tuple[float, ...]
    feature_seed: int


@dataclass
class SpectralTensorState:
    """Mutable state for one budgeted spectral/tensor learner."""

    active_indices: np.ndarray
    active_mask: np.ndarray
    alpha: np.ndarray
    p_matrix: np.ndarray
    utility: np.ndarray
    ages: np.ndarray
    candidate_corr: np.ndarray
    candidate_energy: np.ndarray
    log_step_size: np.ndarray
    h_trace: np.ndarray
    v_trace: np.ndarray
    active_count: int
    additions: int
    replacements: int
    skipped_promotions: int
    finite_failures: int
    steps: int


class SpectralFeatureMap:
    """Deterministic compact spectral/tensor candidate feature map."""

    def __init__(self, feature_dim: int, config: SpectralTensorConfig) -> None:
        self.feature_dim = int(feature_dim)
        self.config = config
        self.rng = np.random.default_rng(config.feature_seed)
        self.basis_family = config.basis_family

        self.use_power = config.basis_family in {
            "chebyshev",
            "cheb_anova",
            "spectral_tensor",
            "all",
        }
        self.use_chebyshev = config.basis_family in {
            "chebyshev",
            "cheb_anova",
            "spectral_tensor",
            "all",
        }
        self.use_fourier = config.basis_family in {
            "fourier",
            "fourier_wavelet",
            "spectral_tensor",
            "all",
        }
        self.use_rff = config.basis_family in {
            "rff_tensor",
            "fourier_wavelet",
            "spectral_tensor",
            "all",
        }
        self.use_anova = config.basis_family in {
            "anova",
            "cheb_anova",
            "rff_tensor",
            "spectral_tensor",
            "all",
        }
        self.use_tensor_sketch = config.basis_family in {
            "tensor_sketch",
            "rff_tensor",
            "spectral_tensor",
            "all",
        }
        self.use_wavelet = config.basis_family in {
            "wavelet",
            "fourier_wavelet",
            "spectral_tensor",
            "ridgelet_chaos",
            "all",
        }
        self.use_chaos = config.basis_family in {
            "chaos",
            "cheb_anova",
            "ridgelet_chaos",
            "spectral_tensor",
            "all",
        }
        self.use_ridgelet = config.basis_family in {
            "ridgelet",
            "ridgelet_chaos",
            "spectral_tensor",
            "all",
        }
        self.use_tensor_cp = config.basis_family in {
            "tensor_cp",
            "ridgelet_chaos",
            "spectral_tensor",
            "all",
        }
        self.use_deep_ridgelet = config.basis_family in {
            "deep_ridgelet",
            "ridgelet_chaos",
            "spectral_tensor",
            "all",
        }

        rff_count = config.rff_count if self.use_rff else 0
        self.rff_weights = self.rng.normal(
            0.0,
            config.rff_scale / math.sqrt(max(self.feature_dim, 1)),
            size=(rff_count, self.feature_dim),
        ).astype(np.float64)
        self.rff_bias = self.rng.uniform(0.0, 2.0 * math.pi, size=rff_count).astype(
            np.float64
        )

        self.anova_specs = self._make_anova_specs(config.anova_count)
        self.wavelet_dims, self.wavelet_centers, self.wavelet_scale_ids = (
            self._make_wavelet_specs(config.wavelet_count)
        )
        self.chaos_specs = self._make_chaos_specs(config.chaos_count)
        self.ridgelet_weights, self.ridgelet_bias = self._make_ridgelet_specs(
            config.ridgelet_count,
        )
        self.tensor_cp_weights, self.tensor_cp_bias = self._make_tensor_cp_specs(
            config.tensor_cp_count,
        )
        (
            self.deep_inner_weights,
            self.deep_inner_bias,
            self.deep_outer_weights,
            self.deep_outer_bias,
        ) = self._make_deep_ridgelet_specs(config.deep_ridgelet_count)
        self.tensor_hashes, self.tensor_signs = self._make_tensor_sketch_specs()
        self.selected_indices = self._make_candidate_selection()
        self.block_sizes = self._compute_block_sizes()

    def _make_anova_specs(self, requested: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return deterministic sparse ANOVA monomial products."""
        if not self.use_anova or requested <= 0:
            return []

        max_order = max(2, min(self.config.interaction_order, 4, self.feature_dim))
        max_degree = max(1, min(self.config.degree, 5))
        specs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        low_dim = min(self.feature_dim, 8)
        for order in range(2, max_order + 1):
            for dims in combinations(range(low_dim), order):
                specs.append((tuple(dims), tuple(1 for _ in dims)))
                if len(specs) >= requested:
                    return specs

        low_powers = tuple(range(1, min(max_degree, 3) + 1))
        for order in range(2, max_order + 1):
            for dims in combinations(range(low_dim), order):
                for powers in product(low_powers, repeat=order):
                    if sum(powers) <= max_degree + order - 1:
                        specs.append((tuple(dims), tuple(int(p) for p in powers)))
                        if len(specs) >= requested:
                            return specs

        while len(specs) < requested:
            order = int(self.rng.integers(2, max_order + 1))
            dims_arr = self.rng.choice(self.feature_dim, size=order, replace=False)
            powers_arr = self.rng.integers(1, max_degree + 1, size=order)
            specs.append(
                (
                    tuple(int(dim) for dim in np.sort(dims_arr)),
                    tuple(int(power) for power in powers_arr),
                )
            )
        return specs

    def _make_wavelet_specs(self, requested: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local Gaussian bump coordinates."""
        if not self.use_wavelet or requested <= 0:
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float64)
            return empty_i, empty_f, empty_i
        scales = self.config.wavelet_scales or (0.35,)
        base_centers = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        dims: list[int] = []
        centers: list[float] = []
        scale_ids: list[int] = []
        for dim in range(self.feature_dim):
            for scale_id, _ in enumerate(scales):
                for center in base_centers:
                    dims.append(dim)
                    centers.append(float(center))
                    scale_ids.append(scale_id)
                    if len(dims) >= requested:
                        return (
                            np.asarray(dims, dtype=np.int32),
                            np.asarray(centers, dtype=np.float64),
                            np.asarray(scale_ids, dtype=np.int32),
                        )
        while len(dims) < requested:
            dims.append(int(self.rng.integers(0, self.feature_dim)))
            centers.append(float(self.rng.uniform(-1.0, 1.0)))
            scale_ids.append(int(self.rng.integers(0, len(scales))))
        return (
            np.asarray(dims, dtype=np.int32),
            np.asarray(centers, dtype=np.float64),
            np.asarray(scale_ids, dtype=np.int32),
        )

    def _make_chaos_specs(
        self,
        requested: int,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return sparse Chebyshev polynomial-chaos interaction specs."""
        if not self.use_chaos or requested <= 0:
            return []

        max_order = max(1, min(self.config.interaction_order, 5, self.feature_dim))
        max_degree = max(1, min(self.config.degree, 8))
        specs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        low_dim = min(self.feature_dim, 10)
        low_powers = tuple(range(1, min(max_degree, 4) + 1))
        for order in range(1, max_order + 1):
            for dims in combinations(range(low_dim), order):
                for powers in product(low_powers, repeat=order):
                    if sum(powers) > max_degree + order - 1:
                        continue
                    spec = (tuple(dims), tuple(int(power) for power in powers))
                    if spec in seen:
                        continue
                    seen.add(spec)
                    specs.append(spec)
                    if len(specs) >= requested:
                        return specs

        while len(specs) < requested:
            order = int(self.rng.integers(1, max_order + 1))
            dims_arr = np.sort(
                self.rng.choice(self.feature_dim, size=order, replace=False),
            )
            powers_arr = self.rng.integers(1, max_degree + 1, size=order)
            spec = (
                tuple(int(dim) for dim in dims_arr),
                tuple(int(power) for power in powers_arr),
            )
            if spec in seen:
                continue
            seen.add(spec)
            specs.append(spec)
        return specs

    def _make_ridgelet_specs(self, requested: int) -> tuple[np.ndarray, np.ndarray]:
        """Return random spectral ridgelet projection parameters."""
        if not self.use_ridgelet or requested <= 0:
            return (
                np.zeros((0, self.feature_dim), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        n_bases = int(math.ceil(requested / 5))
        weights = self.rng.normal(
            0.0,
            self.config.ridgelet_scale / math.sqrt(max(self.feature_dim, 1)),
            size=(n_bases, self.feature_dim),
        ).astype(np.float64)
        bias = self.rng.uniform(-math.pi, math.pi, size=n_bases).astype(np.float64)
        return weights, bias

    def _make_tensor_cp_specs(self, requested: int) -> tuple[np.ndarray, np.ndarray]:
        """Return compressed CP-style random tensor interaction parameters."""
        if not self.use_tensor_cp or requested <= 0:
            return (
                np.zeros((0, 0, self.feature_dim), dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
            )
        max_order = max(2, min(self.config.interaction_order, 5))
        weights = self.rng.normal(
            0.0,
            self.config.tensor_cp_scale / math.sqrt(max(self.feature_dim, 1)),
            size=(requested, max_order, self.feature_dim),
        ).astype(np.float64)
        bias = self.rng.uniform(
            -0.5 * math.pi,
            0.5 * math.pi,
            size=(requested, max_order),
        ).astype(np.float64)
        return weights, bias

    def _make_deep_ridgelet_specs(
        self,
        requested: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return fixed two-level random ridgelet parameters."""
        inner_width = max(1, int(self.config.deep_ridgelet_inner))
        if not self.use_deep_ridgelet or requested <= 0:
            return (
                np.zeros((0, inner_width, self.feature_dim), dtype=np.float64),
                np.zeros((0, inner_width), dtype=np.float64),
                np.zeros((0, inner_width), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        inner_weights = self.rng.normal(
            0.0,
            self.config.deep_ridgelet_scale / math.sqrt(max(self.feature_dim, 1)),
            size=(requested, inner_width, self.feature_dim),
        ).astype(np.float64)
        inner_bias = (
            0.25
            * self.rng.normal(0.0, 1.0, size=(requested, inner_width)).astype(
                np.float64,
            )
        )
        outer_weights = self.rng.normal(
            0.0,
            self.config.deep_ridgelet_scale / math.sqrt(inner_width),
            size=(requested, inner_width),
        ).astype(np.float64)
        outer_bias = (
            0.25 * self.rng.normal(0.0, 1.0, size=requested).astype(np.float64)
        )
        return inner_weights, inner_bias, outer_weights, outer_bias

    def _make_tensor_sketch_specs(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Return CountSketch hashes/signs for TensorSketch blocks."""
        hashes: dict[int, np.ndarray] = {}
        signs: dict[int, np.ndarray] = {}
        if not self.use_tensor_sketch or self.config.tensor_sketch_dim <= 0:
            return hashes, signs
        sketch_dim = int(self.config.tensor_sketch_dim)
        for order in range(2, max(2, self.config.interaction_order) + 1):
            hashes[order] = self.rng.integers(
                0,
                sketch_dim,
                size=(order, self.feature_dim),
                dtype=np.int32,
            )
            signs[order] = self.rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float64),
                size=(order, self.feature_dim),
            )
        return hashes, signs

    def _raw_full_features(self, observation: np.ndarray) -> np.ndarray:
        """Return the untruncated candidate vector."""
        x = np.asarray(observation, dtype=np.float64).reshape(-1)
        if self.config.input_clip > 0.0:
            x = np.clip(x, -self.config.input_clip, self.config.input_clip)
        scale = max(float(self.config.input_scale), 1e-8)
        x_unit = np.clip(x / max(self.config.input_clip, scale), -1.0, 1.0)
        x_squashed = np.tanh(x / scale)

        parts: list[np.ndarray] = [np.ones(1, dtype=np.float64)]
        parts.append(x_unit / math.sqrt(max(self.feature_dim, 1)))

        if self.use_power:
            power_terms = [
                np.power(x_unit, degree)
                / math.sqrt(max(self.feature_dim * degree, 1))
                for degree in range(2, self.config.degree + 1)
            ]
            if power_terms:
                parts.append(np.concatenate(power_terms))

        if self.use_chebyshev:
            cheb_terms = self._chebyshev_terms(x_squashed)
            if cheb_terms.size:
                parts.append(cheb_terms)

        if self.use_fourier:
            fourier_terms = self._fourier_terms(x_squashed)
            if fourier_terms.size:
                parts.append(fourier_terms)

        if self.use_rff and self.rff_weights.size:
            projected = self.rff_weights @ x_squashed + self.rff_bias
            parts.append(
                math.sqrt(2.0 / max(self.rff_weights.shape[0], 1)) * np.cos(projected)
            )

        if self.use_wavelet and self.wavelet_dims.size:
            parts.append(self._wavelet_terms(x_squashed))

        if self.use_anova and self.anova_specs:
            parts.append(self._anova_terms(x_unit))

        if self.use_chaos and self.chaos_specs:
            parts.append(self._chaos_terms(x_squashed))

        if self.use_ridgelet and self.ridgelet_weights.size:
            parts.append(self._ridgelet_terms(x_squashed))

        if self.use_tensor_cp and self.tensor_cp_weights.size:
            parts.append(self._tensor_cp_terms(x_squashed))

        if self.use_deep_ridgelet and self.deep_inner_weights.size:
            parts.append(self._deep_ridgelet_terms(x_squashed))

        if self.use_tensor_sketch and self.tensor_hashes:
            parts.append(self._tensor_sketch_terms(x_unit))

        features = np.concatenate(parts).astype(np.float64, copy=False)
        if self.config.feature_clip > 0.0:
            features = np.clip(
                features,
                -self.config.feature_clip,
                self.config.feature_clip,
            )
        return np.nan_to_num(features, copy=False)

    def _chebyshev_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return Chebyshev terms T_2 through T_degree for each coordinate."""
        if self.config.degree < 2:
            return np.zeros(0, dtype=np.float64)
        terms: list[np.ndarray] = []
        t_prev = np.ones_like(x_squashed)
        t_cur = x_squashed.copy()
        for _degree in range(2, self.config.degree + 1):
            t_next = 2.0 * x_squashed * t_cur - t_prev
            terms.append(t_next / math.sqrt(max(self.feature_dim, 1)))
            t_prev, t_cur = t_cur, t_next
        return np.concatenate(terms) if terms else np.zeros(0, dtype=np.float64)

    def _fourier_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return coordinate-wise sine/cosine spectral features."""
        terms: list[np.ndarray] = []
        for frequency in self.config.fourier_frequencies:
            phase = math.pi * float(frequency) * x_squashed
            norm = math.sqrt(max(2 * self.feature_dim, 1))
            terms.append(np.sin(phase) / norm)
            terms.append(np.cos(phase) / norm)
        return np.concatenate(terms) if terms else np.zeros(0, dtype=np.float64)

    def _wavelet_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return local Gaussian multiresolution bump features."""
        scales = np.asarray(self.config.wavelet_scales or (0.35,), dtype=np.float64)
        dim_values = x_squashed[self.wavelet_dims]
        selected_scales = np.maximum(scales[self.wavelet_scale_ids], 1e-4)
        z = (dim_values - self.wavelet_centers) / selected_scales
        values = np.exp(-0.5 * z * z)
        return np.asarray(values / math.sqrt(max(values.shape[0], 1)), dtype=np.float64)

    def _anova_terms(self, x_unit: np.ndarray) -> np.ndarray:
        """Return sparse low-order ANOVA monomial products."""
        values = np.empty(len(self.anova_specs), dtype=np.float64)
        for idx, (dims, powers) in enumerate(self.anova_specs):
            term = 1.0
            for dim, power in zip(dims, powers, strict=True):
                term *= float(x_unit[dim]) ** power
            values[idx] = term
        return values / math.sqrt(max(values.shape[0], 1))

    @staticmethod
    def _chebyshev_value(x_value: float, degree: int) -> float:
        """Return one scalar Chebyshev polynomial value."""
        if degree <= 0:
            return 1.0
        if degree == 1:
            return x_value
        t_prev = 1.0
        t_cur = x_value
        for _ in range(2, degree + 1):
            t_prev, t_cur = t_cur, 2.0 * x_value * t_cur - t_prev
        return t_cur

    def _chaos_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return random sparse Chebyshev polynomial-chaos products."""
        values = np.empty(len(self.chaos_specs), dtype=np.float64)
        for idx, (dims, powers) in enumerate(self.chaos_specs):
            term = 1.0
            for dim, power in zip(dims, powers, strict=True):
                term *= self._chebyshev_value(float(x_squashed[dim]), int(power))
            values[idx] = term
        return values / math.sqrt(max(values.shape[0], 1))

    def _ridgelet_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return ridgelet/Hermite spectral features from random projections."""
        z = self.ridgelet_weights @ x_squashed + self.ridgelet_bias
        clipped_z = np.clip(z, -8.0, 8.0)
        envelope = np.exp(-0.5 * clipped_z * clipped_z)
        values = np.concatenate(
            [
                np.tanh(z),
                np.sin(z),
                np.cos(z),
                clipped_z * envelope,
                (clipped_z * clipped_z - 1.0) * envelope,
            ],
        )
        values = values[: self.config.ridgelet_count]
        return values / math.sqrt(max(values.shape[0], 1))

    def _tensor_cp_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return compressed tensor-CP products of nonlinear ridge factors."""
        projections = np.einsum("cof,f->co", self.tensor_cp_weights, x_squashed)
        factors = np.tanh(projections + self.tensor_cp_bias)
        values: list[np.ndarray] = []
        for order in range(2, factors.shape[1] + 1):
            values.append(np.prod(factors[:, :order], axis=1))
        concatenated = np.concatenate(values)[: self.config.tensor_cp_count]
        return concatenated / math.sqrt(max(concatenated.shape[0], 1))

    def _deep_ridgelet_terms(self, x_squashed: np.ndarray) -> np.ndarray:
        """Return fixed two-level compositional ridgelet features."""
        inner_pre = (
            np.einsum("chf,f->ch", self.deep_inner_weights, x_squashed)
            + self.deep_inner_bias
        )
        inner = np.tanh(inner_pre)
        outer_pre = np.sum(self.deep_outer_weights * inner, axis=1) + self.deep_outer_bias
        values = np.tanh(outer_pre)
        return np.asarray(values / math.sqrt(max(values.shape[0], 1)), dtype=np.float64)

    def _tensor_sketch_terms(self, x_unit: np.ndarray) -> np.ndarray:
        """Return TensorSketch compressed polynomial interaction features."""
        sketch_dim = int(self.config.tensor_sketch_dim)
        blocks: list[np.ndarray] = []
        for order, hashes in self.tensor_hashes.items():
            signs = self.tensor_signs[order]
            sketches = []
            for slot in range(order):
                sketch = np.zeros(sketch_dim, dtype=np.float64)
                np.add.at(sketch, hashes[slot], signs[slot] * x_unit)
                sketches.append(np.fft.fft(sketch))
            convolved = sketches[0]
            for slot_fft in sketches[1:]:
                convolved = convolved * slot_fft
            block = np.fft.ifft(convolved).real
            blocks.append(block / math.sqrt(max(sketch_dim, 1)))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def _make_candidate_selection(self) -> np.ndarray:
        """Choose a stable candidate subset when the raw feature pool is larger."""
        full_dim = int(self._raw_full_features(np.zeros(self.feature_dim)).shape[0])
        target = min(max(self.config.candidate_count, 1), full_dim)
        if full_dim <= target:
            return np.arange(full_dim, dtype=np.int32)

        always = list(range(min(full_dim, 1 + self.feature_dim)))
        remaining = np.asarray(
            [idx for idx in range(full_dim) if idx not in set(always)],
            dtype=np.int32,
        )
        self.rng.shuffle(remaining)
        selected = np.concatenate(
            [
                np.asarray(always, dtype=np.int32),
                remaining[: max(target - len(always), 0)],
            ]
        )
        return np.sort(selected.astype(np.int32))

    def _compute_block_sizes(self) -> dict[str, int]:
        """Return rough diagnostic block sizes for the configured map."""
        sizes = {
            "selected": int(self.selected_indices.shape[0]),
            "raw_full": int(self._raw_full_features(np.zeros(self.feature_dim)).shape[0]),
            "rff": int(self.rff_weights.shape[0]),
            "anova": int(len(self.anova_specs)),
            "tensor_sketch": int(
                self.config.tensor_sketch_dim * len(self.tensor_hashes)
            ),
            "chaos": int(len(self.chaos_specs)),
            "ridgelet": int(self.config.ridgelet_count if self.use_ridgelet else 0),
            "tensor_cp": int(self.config.tensor_cp_count if self.use_tensor_cp else 0),
            "deep_ridgelet": int(
                self.config.deep_ridgelet_count if self.use_deep_ridgelet else 0
            ),
            "wavelet": int(self.wavelet_dims.shape[0]),
        }
        return sizes

    def transform(self, observation: np.ndarray) -> np.ndarray:
        """Return the fixed-size selected candidate vector."""
        return np.asarray(
            self._raw_full_features(observation)[self.selected_indices],
            dtype=np.float64,
        )


class BudgetedSpectralTensorLearner:
    """One online predictor over a budgeted spectral/tensor feature bank."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: SpectralTensorConfig,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.feature_map = SpectralFeatureMap(feature_dim, config)
        self.candidate_dim = int(self.feature_map.selected_indices.shape[0])

    def init(self) -> SpectralTensorState:
        """Return an empty active bank."""
        budget = min(self.config.budget, self.candidate_dim)
        return SpectralTensorState(
            active_indices=np.full(budget, -1, dtype=np.int32),
            active_mask=np.zeros(self.candidate_dim, dtype=bool),
            alpha=np.zeros((budget, self.n_heads), dtype=np.float64),
            p_matrix=np.eye(budget, dtype=np.float64) * self.config.rls_delta,
            utility=np.zeros(budget, dtype=np.float64),
            ages=np.zeros(budget, dtype=np.int64),
            candidate_corr=np.zeros((self.candidate_dim, self.n_heads), dtype=np.float64),
            candidate_energy=np.full(self.candidate_dim, 1e-8, dtype=np.float64),
            log_step_size=np.full(
                (budget, self.n_heads),
                math.log(max(self.config.autostep_init_step_size, 1e-8)),
                dtype=np.float64,
            ),
            h_trace=np.zeros((budget, self.n_heads), dtype=np.float64),
            v_trace=np.zeros((budget, self.n_heads), dtype=np.float64),
            active_count=0,
            additions=0,
            replacements=0,
            skipped_promotions=0,
            finite_failures=0,
            steps=0,
        )

    def predict(self, state: SpectralTensorState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads from active features only."""
        if state.active_count == 0:
            return np.zeros(self.n_heads, dtype=np.float64)
        candidates = self.feature_map.transform(observation)
        phi = candidates[state.active_indices[: state.active_count]]
        return np.asarray(phi @ state.alpha[: state.active_count])

    def _candidate_scores(
        self,
        state: SpectralTensorState,
        candidates: np.ndarray,
        residual: np.ndarray,
        active_heads: np.ndarray,
    ) -> np.ndarray:
        """Update and return normalized utility scores for all candidates."""
        decay = self.config.utility_decay
        state.candidate_energy = decay * state.candidate_energy + (
            1.0 - decay
        ) * (candidates * candidates)
        if np.any(active_heads):
            state.candidate_corr[:, active_heads] = (
                decay * state.candidate_corr[:, active_heads]
                + (1.0 - decay)
                * candidates[:, None]
                * residual[active_heads][None, :]
            )
        score = np.mean(np.abs(state.candidate_corr), axis=1)
        score = score / np.sqrt(np.maximum(state.candidate_energy, 1e-8))
        return np.asarray(np.nan_to_num(score, copy=False), dtype=np.float64)

    def _promote_candidate(
        self,
        state: SpectralTensorState,
        candidate_idx: int,
        replace_slot: int | None,
    ) -> None:
        """Add or replace one feature in the active budget."""
        if replace_slot is None:
            slot = state.active_count
            state.active_count += 1
            state.additions += 1
        else:
            slot = replace_slot
            old_idx = int(state.active_indices[slot])
            if old_idx >= 0:
                state.active_mask[old_idx] = False
            state.replacements += 1

        state.active_indices[slot] = int(candidate_idx)
        state.active_mask[candidate_idx] = True
        state.alpha[slot] = 0.0
        state.utility[slot] = 0.0
        state.ages[slot] = 0
        state.log_step_size[slot] = math.log(
            max(self.config.autostep_init_step_size, 1e-8)
        )
        state.h_trace[slot] = 0.0
        state.v_trace[slot] = 0.0
        state.p_matrix[slot, :] = 0.0
        state.p_matrix[:, slot] = 0.0
        state.p_matrix[slot, slot] = self.config.rls_delta

    def _maybe_update_active_set(
        self,
        state: SpectralTensorState,
        scores: np.ndarray,
    ) -> None:
        """Promote high-scoring candidates under the active feature budget."""
        if self.config.promotions_per_step <= 0:
            return
        if state.steps % max(self.config.replacement_interval, 1) != 0:
            can_fill = state.active_count < state.active_indices.shape[0]
            if not can_fill:
                return

        score_view = scores.copy()
        score_view[state.active_mask] = -np.inf
        for _ in range(self.config.promotions_per_step):
            candidate_idx = int(np.argmax(score_view))
            best_score = float(score_view[candidate_idx])
            if not np.isfinite(best_score) or best_score < self.config.score_threshold:
                state.skipped_promotions += 1
                return
            if state.active_count < state.active_indices.shape[0]:
                self._promote_candidate(state, candidate_idx, None)
                score_view[candidate_idx] = -np.inf
                continue

            mature = state.ages[: state.active_count] >= self.config.min_feature_age
            if not np.any(mature):
                state.skipped_promotions += 1
                return
            masked_utility = np.where(mature, state.utility[: state.active_count], np.inf)
            replace_slot = int(np.argmin(masked_utility))
            weakest = float(masked_utility[replace_slot])
            if best_score <= self.config.replace_margin * max(weakest, 1e-8):
                state.skipped_promotions += 1
                return
            self._promote_candidate(state, candidate_idx, replace_slot)
            score_view[candidate_idx] = -np.inf

    def _rls_update(
        self,
        state: SpectralTensorState,
        phi: np.ndarray,
        errors: np.ndarray,
    ) -> float:
        """Run one multi-head RLS coefficient update."""
        m = state.active_count
        p_active = state.p_matrix[:m, :m]
        p_phi = p_active @ phi
        denom = self.config.rho + float(phi @ p_phi)
        if denom <= 1e-12 or not np.isfinite(denom):
            state.finite_failures += 1
            return 0.0
        gain = p_phi / denom
        state.alpha[:m] += np.outer(gain, errors)
        next_p = (p_active - np.outer(gain, phi @ p_active)) / self.config.rho
        state.p_matrix[:m, :m] = 0.5 * (next_p + next_p.T)
        return float(phi @ p_phi)

    def _nlms_update(
        self,
        state: SpectralTensorState,
        phi: np.ndarray,
        errors: np.ndarray,
    ) -> float:
        """Run one normalized LMS update."""
        normalizer = 1.0 + float(phi @ phi)
        state.alpha[: state.active_count] += (
            self.config.nlms_step_size * np.outer(phi, errors) / normalizer
        )
        return float(phi @ phi)

    def _autostep_update(
        self,
        state: SpectralTensorState,
        phi: np.ndarray,
        errors: np.ndarray,
        active_heads: np.ndarray,
    ) -> float:
        """Run a compact Autostep-style per-feature coefficient update."""
        m = state.active_count
        if not np.any(active_heads):
            return 0.0
        tau_rate = 1.0 / max(self.config.autostep_tau, 1.0)
        phi_col = phi[:, None]
        meta_grad = phi_col * errors[None, :] * state.h_trace[:m]
        abs_meta = np.abs(meta_grad)
        state.v_trace[:m] = np.maximum(
            abs_meta,
            (1.0 - tau_rate) * state.v_trace[:m] + tau_rate * abs_meta,
        )
        normalized_meta = meta_grad / np.maximum(state.v_trace[:m], 1e-8)
        state.log_step_size[:m, active_heads] += (
            self.config.autostep_mu * normalized_meta[:, active_heads]
        )
        state.log_step_size[:m] = np.clip(state.log_step_size[:m], -18.0, 2.0)
        step_sizes = np.exp(state.log_step_size[:m])
        overshoot = max(float(np.sum(step_sizes * (phi_col * phi_col))), 1.0)
        normalized_step = step_sizes / overshoot
        state.alpha[:m] += normalized_step * phi_col * errors[None, :]
        state.h_trace[:m] = state.h_trace[:m] * (
            1.0 - normalized_step * (phi_col * phi_col)
        ) + normalized_step * phi_col * errors[None, :]
        return float(overshoot)

    def step(
        self,
        state: SpectralTensorState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, select features, then update coefficients."""
        candidates = self.feature_map.transform(observation)
        if state.active_count > 0:
            active_indices = state.active_indices[: state.active_count]
            phi_pre = candidates[active_indices]
            prediction = np.asarray(phi_pre @ state.alpha[: state.active_count])
        else:
            prediction = np.zeros(self.n_heads, dtype=np.float64)

        active_heads = ~np.isnan(target)
        safe_target = np.where(active_heads, target, 0.0)
        residual = np.where(active_heads, safe_target - prediction, 0.0)
        scores = self._candidate_scores(state, candidates, residual, active_heads)
        self._maybe_update_active_set(state, scores)

        leverage = 0.0
        if state.active_count > 0 and np.any(active_heads):
            phi = candidates[state.active_indices[: state.active_count]]
            update_prediction = np.asarray(phi @ state.alpha[: state.active_count])
            update_errors = np.where(active_heads, safe_target - update_prediction, 0.0)
            if self.config.update_rule == "rls":
                leverage = self._rls_update(state, phi, update_errors)
            elif self.config.update_rule == "autostep":
                leverage = self._autostep_update(
                    state,
                    phi,
                    update_errors,
                    active_heads,
                )
            else:
                leverage = self._nlms_update(state, phi, update_errors)
            contribution = np.mean(
                np.abs(state.alpha[: state.active_count] * phi[:, None]),
                axis=1,
            )
            state.utility[: state.active_count] = (
                self.config.utility_decay * state.utility[: state.active_count]
                + (1.0 - self.config.utility_decay) * contribution
            )
            state.ages[: state.active_count] += 1

        if not np.all(np.isfinite(state.alpha[: state.active_count])):
            state.finite_failures += 1
            state.alpha[: state.active_count] = np.nan_to_num(
                state.alpha[: state.active_count],
                copy=False,
            )
        state.steps += 1
        diagnostics = {
            "active_features": float(state.active_count),
            "additions": float(state.additions),
            "replacements": float(state.replacements),
            "skipped_promotions": float(state.skipped_promotions),
            "mean_utility": float(
                np.mean(state.utility[: state.active_count])
                if state.active_count > 0
                else 0.0
            ),
            "max_candidate_score": float(np.max(scores)),
            "leverage": float(leverage),
            "finite_failures": float(state.finite_failures),
        }
        return prediction, diagnostics


def stderr(values: np.ndarray) -> float:
    """Return the standard error of a vector."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def masked_mse_np(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return MSE over non-NaN target heads."""
    active = ~np.isnan(target)
    if not np.any(active):
        return 0.0
    diff = prediction[active] - target[active]
    return float(np.mean(diff * diff))


def run_spectral_stream(
    observations: jax.Array,
    targets: jax.Array,
    config: SpectralTensorConfig,
) -> tuple[BudgetedSpectralTensorLearner, SpectralTensorState, np.ndarray]:
    """Run one spectral/tensor learner on a materialized stream."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = BudgetedSpectralTensorLearner(
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
        metrics[idx, 2] = diagnostics["active_features"]
        metrics[idx, 3] = diagnostics["additions"]
        metrics[idx, 4] = diagnostics["replacements"]
        metrics[idx, 5] = diagnostics["mean_utility"]
        metrics[idx, 6] = diagnostics["max_candidate_score"]
        metrics[idx, 7] = diagnostics["finite_failures"]
    return learner, state, metrics


def make_mlp(
    method: str,
    n_heads: int,
    step_size: float,
    sparsity: float,
) -> MultiHeadMLPLearner:
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


def summarize_prequential(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
    loss_col: int = 0,
    pred_col: int = 1,
) -> dict[str, float]:
    """Summarize one method's online metrics."""
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


def evaluate_spectral_classifier(
    learner: BudgetedSpectralTensorLearner,
    state: SpectralTensorState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final spectral/tensor classifier on held-out digits."""
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
    """Evaluate a final MLP classifier on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def controlled_dataset_name(task_mode: str) -> str:
    """Return canonical controlled task name."""
    return f"controlled_{task_mode}"


def expand_dataset_names(spec: str) -> list[str]:
    """Expand dataset aliases into concrete benchmark names."""
    aliases = {
        "all": list(VALID_DATASETS),
        "controlled": list(CONTROLLED_DATASETS),
        "synthetic": list(SYNTHETIC_REGIMES),
        "digits": list(DIGITS_REGIMES),
        "blockers": [
            "synthetic_compositional",
            "controlled_frequency",
            "synthetic_frequency",
            "controlled_polynomial",
            "digits_label_drift",
        ],
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
        (
            observations,
            targets,
            labels,
            x_test,
            y_test,
            stream_meta,
        ) = make_digits_regime_sequence(
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


def method_name(config: SpectralTensorConfig) -> str:
    """Return a stable compact method name."""
    rho = f"{config.rho:g}".replace(".", "p")
    step = f"{config.nlms_step_size:g}".replace(".", "p")
    seed = f"s{config.feature_seed}" if config.feature_seed else "s0"
    return (
        f"spec_{config.basis_family}_{config.update_rule}_b{config.budget}_"
        f"c{config.candidate_count}_d{config.degree}_o{config.interaction_order}_"
        f"rho{rho}_eta{step}_{seed}"
    )


def make_spectral_configs(args: argparse.Namespace) -> list[SpectralTensorConfig]:
    """Expand CLI sweep values into concrete spectral/tensor configs."""
    configs: list[SpectralTensorConfig] = []
    for basis_family in args.basis_families:
        for budget in args.budgets:
            for candidate_count in args.candidate_counts:
                for degree in args.degrees:
                    for interaction_order in args.interaction_orders:
                        for update_rule in args.update_rules:
                            for rho in args.rhos:
                                configs.append(
                                    SpectralTensorConfig(
                                        basis_family=str(basis_family),
                                        budget=int(budget),
                                        candidate_count=int(candidate_count),
                                        degree=int(degree),
                                        interaction_order=int(interaction_order),
                                        update_rule=str(update_rule),
                                        rho=float(rho),
                                        rls_delta=float(args.rls_delta),
                                        nlms_step_size=float(args.nlms_step_size),
                                        autostep_mu=float(args.autostep_mu),
                                        autostep_tau=float(args.autostep_tau),
                                        autostep_init_step_size=float(
                                            args.autostep_init_step_size
                                        ),
                                        utility_decay=float(args.utility_decay),
                                        min_feature_age=int(args.min_feature_age),
                                        replacement_interval=int(
                                            args.replacement_interval
                                        ),
                                        replace_margin=float(args.replace_margin),
                                        promotions_per_step=int(args.promotions_per_step),
                                        score_threshold=float(args.score_threshold),
                                        input_clip=float(args.input_clip),
                                        input_scale=float(args.input_scale),
                                        feature_clip=float(args.feature_clip),
                                        rff_count=int(args.rff_count),
                                        rff_scale=float(args.rff_scale),
                                        anova_count=int(args.anova_count),
                                        tensor_sketch_dim=int(args.tensor_sketch_dim),
                                        chaos_count=int(args.chaos_count),
                                        ridgelet_count=int(args.ridgelet_count),
                                        ridgelet_scale=float(args.ridgelet_scale),
                                        tensor_cp_count=int(args.tensor_cp_count),
                                        tensor_cp_scale=float(args.tensor_cp_scale),
                                        deep_ridgelet_count=int(
                                            args.deep_ridgelet_count
                                        ),
                                        deep_ridgelet_inner=int(
                                            args.deep_ridgelet_inner
                                        ),
                                        deep_ridgelet_scale=float(
                                            args.deep_ridgelet_scale
                                        ),
                                        wavelet_count=int(args.wavelet_count),
                                        wavelet_scales=tuple(
                                            float(value) for value in args.wavelet_scales
                                        ),
                                        fourier_frequencies=tuple(
                                            float(value)
                                            for value in args.fourier_frequencies
                                        ),
                                        feature_seed=int(args.feature_seed),
                                    )
                                )
    return configs


def is_higher_better(metric: str) -> bool:
    """Return whether larger metric values are better."""
    return metric.endswith("accuracy")


def paired_diff(candidate: float, baseline: float, metric: str) -> float:
    """Return a paired difference where positive favors the candidate."""
    if is_higher_better(metric):
        return candidate - baseline
    return baseline - candidate


def compare_to_group(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
    group: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one candidate method with the per-seed best MLP baseline."""
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
    candidate_methods: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate seed records by dataset and method."""
    aggregate: dict[str, Any] = {}
    for dataset in sorted({record["dataset_name"] for record in records}):
        dataset_records = [record for record in records if record["dataset_name"] == dataset]
        method_names = list(dataset_records[0]["methods"])
        dataset_agg: dict[str, Any] = {}
        for method in method_names:
            metric_rows: dict[str, Any] = {}
            for metric in sorted(dataset_records[0]["methods"][method]):
                values = np.asarray(
                    [record["methods"][method][metric] for record in dataset_records],
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
                for method in candidate_methods
            }
            best_candidate_by_seed: list[str] = []
            diffs: list[float] = []
            for record in dataset_records:
                methods = record["methods"]
                candidate_values = {
                    method: methods[method][metric]
                    for method in candidate_methods
                    if metric in methods[method]
                }
                if not candidate_values:
                    continue
                if is_higher_better(metric):
                    best_candidate = max(
                        candidate_values,
                        key=candidate_values.__getitem__,
                    )
                    best_mlp = max(MLP_METHODS, key=lambda name: methods[name][metric])
                else:
                    best_candidate = min(
                        candidate_values,
                        key=candidate_values.__getitem__,
                    )
                    best_mlp = min(MLP_METHODS, key=lambda name: methods[name][metric])
                best_candidate_by_seed.append(best_candidate)
                diffs.append(
                    paired_diff(
                        float(methods[best_candidate][metric]),
                        float(methods[best_mlp][metric]),
                        metric,
                    )
                )
            diff_arr = np.asarray(diffs, dtype=np.float64)
            comparisons[metric]["best_candidate_vs_best_mlp"] = {
                "paired_diff_mean_positive_favors_candidate": float(np.mean(diff_arr))
                if diff_arr.size
                else 0.0,
                "paired_diff_stderr": stderr(diff_arr) if diff_arr.size else 0.0,
                "wins_for_candidate": int(np.sum(diff_arr > 0.0)),
                "wins_for_mlp": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n": int(diff_arr.shape[0]),
                "diffs": diff_arr.tolist(),
                "best_candidate_counts": dict(
                    sorted(
                        (name, best_candidate_by_seed.count(name))
                        for name in set(best_candidate_by_seed)
                    )
                ),
            }
        dataset_agg["comparisons"] = comparisons
        aggregate[dataset] = dataset_agg
    return aggregate


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown assessment for this run."""
    cfg = results["config"]
    lines = [
        "# D12 Spectral/Tensor Universal Learner",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Basis families="
            f"{cfg['basis_families']}, budgets={cfg['budgets']}, "
            f"candidate counts={cfg['candidate_counts']}, degrees={cfg['degrees']}, "
            f"orders={cfg['interaction_orders']}, updates={cfg['update_rules']}."
        ),
        "",
        "The candidate is one online predictor: candidate feature utilities, active "
        "feature selection, and coefficients are all updated under the same loss. "
        "There is no MLP residual, expert router, or dataset-specific switch.",
        "",
    ]
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Active feats | Replacements | Runtime s |",
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
                f"{metric_cell(row, 'active_features')} | "
                f"{metric_cell(row, 'replacements')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        comparisons = dataset_agg["comparisons"]
        if "final_window_mse" in comparisons:
            best = comparisons["final_window_mse"]["best_candidate_vs_best_mlp"]
            lines.append(
                "`final_window_mse` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_candidate']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_candidate']}/{best['wins_for_mlp']}/"
                f"{best['ties']}; best-candidate counts "
                f"{best['best_candidate_counts']}."
            )
        if "test_accuracy" in comparisons:
            best = comparisons["test_accuracy"]["best_candidate_vs_best_mlp"]
            lines.append(
                "`test_accuracy` best-candidate-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_candidate']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_candidate']}/{best['wins_for_mlp']}/{best['ties']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Bar",
            "",
            "A successful D12 result would require a single promoted configuration "
            "to beat the best fair MLP on the blocker suite without choosing a "
            "basis family per dataset. A sweep win is useful headroom evidence, "
            "but it should feed the canonical multi-bank/resource-manager design "
            "unless one fixed configuration is robust across all paired seeds.",
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
        help=(
            "Comma-separated regimes or aliases: all, controlled, synthetic, "
            "digits, blockers, controlled task names, or concrete dataset names."
        ),
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
        "--basis-families",
        nargs="+",
        default=("spectral_tensor",),
        choices=(
            "chebyshev",
            "anova",
            "cheb_anova",
            "fourier",
            "wavelet",
            "fourier_wavelet",
            "tensor_sketch",
            "chaos",
            "ridgelet",
            "ridgelet_chaos",
            "tensor_cp",
            "deep_ridgelet",
            "rff_tensor",
            "spectral_tensor",
            "all",
        ),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=(256,))
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=(768,))
    parser.add_argument("--degrees", type=int, nargs="+", default=(5,))
    parser.add_argument("--interaction-orders", type=int, nargs="+", default=(3,))
    parser.add_argument(
        "--update-rules",
        nargs="+",
        default=("rls",),
        choices=("rls", "nlms", "autostep"),
    )
    parser.add_argument("--rhos", type=float, nargs="+", default=(0.995,))
    parser.add_argument("--rls-delta", type=float, default=50.0)
    parser.add_argument("--nlms-step-size", type=float, default=0.2)
    parser.add_argument("--autostep-mu", type=float, default=0.001)
    parser.add_argument("--autostep-tau", type=float, default=1000.0)
    parser.add_argument("--autostep-init-step-size", type=float, default=0.03)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--min-feature-age", type=int, default=80)
    parser.add_argument("--replacement-interval", type=int, default=20)
    parser.add_argument("--replace-margin", type=float, default=1.25)
    parser.add_argument("--promotions-per-step", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--feature-clip", type=float, default=5.0)
    parser.add_argument("--rff-count", type=int, default=256)
    parser.add_argument("--rff-scale", type=float, default=2.0)
    parser.add_argument("--anova-count", type=int, default=256)
    parser.add_argument("--tensor-sketch-dim", type=int, default=128)
    parser.add_argument("--chaos-count", type=int, default=256)
    parser.add_argument("--ridgelet-count", type=int, default=512)
    parser.add_argument("--ridgelet-scale", type=float, default=2.5)
    parser.add_argument("--tensor-cp-count", type=int, default=256)
    parser.add_argument("--tensor-cp-scale", type=float, default=2.0)
    parser.add_argument("--deep-ridgelet-count", type=int, default=512)
    parser.add_argument("--deep-ridgelet-inner", type=int, default=6)
    parser.add_argument("--deep-ridgelet-scale", type=float, default=1.5)
    parser.add_argument("--wavelet-count", type=int, default=128)
    parser.add_argument(
        "--wavelet-scales",
        type=float,
        nargs="+",
        default=(0.25, 0.5),
    )
    parser.add_argument(
        "--fourier-frequencies",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0, 5.0),
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true", help="Tiny harness check.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("--budgets must be positive")
    if any(count <= 0 for count in args.candidate_counts):
        raise ValueError("--candidate-counts must be positive")
    if any(degree <= 0 for degree in args.degrees):
        raise ValueError("--degrees must be positive")
    if any(order <= 0 for order in args.interaction_orders):
        raise ValueError("--interaction-orders must be positive")
    if any(rho <= 0.0 or rho > 1.0 for rho in args.rhos):
        raise ValueError("--rhos must be in (0, 1]")
    if args.rls_delta <= 0.0:
        raise ValueError("--rls-delta must be positive")
    if args.nlms_step_size <= 0.0:
        raise ValueError("--nlms-step-size must be positive")
    if args.input_clip <= 0.0:
        raise ValueError("--input-clip must be positive")
    if args.input_scale <= 0.0:
        raise ValueError("--input-scale must be positive")
    if args.rff_count < 0 or args.anova_count < 0:
        raise ValueError("--rff-count and --anova-count must be non-negative")
    if args.tensor_sketch_dim < 0 or args.wavelet_count < 0:
        raise ValueError("--tensor-sketch-dim and --wavelet-count must be non-negative")
    if (
        args.chaos_count < 0
        or args.ridgelet_count < 0
        or args.tensor_cp_count < 0
        or args.deep_ridgelet_count < 0
    ):
        raise ValueError(
            "--chaos-count, --ridgelet-count, --tensor-cp-count, and "
            "--deep-ridgelet-count must be non-negative"
        )
    if args.deep_ridgelet_inner <= 0:
        raise ValueError("--deep-ridgelet-inner must be positive")
    if (
        args.ridgelet_scale <= 0.0
        or args.tensor_cp_scale <= 0.0
        or args.deep_ridgelet_scale <= 0.0
    ):
        raise ValueError(
            "--ridgelet-scale, --tensor-cp-scale, and --deep-ridgelet-scale "
            "must be positive"
        )


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    configs: list[SpectralTensorConfig],
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
        mlp_state, metrics = run_mlp_stream(
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
                evaluate_mlp_classifier(mlp_learner, mlp_state, x_test, y_test)
            )

    for config in configs:
        method = method_name(config)
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        spectral_learner, spectral_state, metrics = run_spectral_stream(
            observations,
            targets,
            config,
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method].update(
            {
                "runtime_s": float(time.time() - t0),
                "active_features": float(spectral_state.active_count),
                "candidate_dim": float(spectral_learner.candidate_dim),
                "raw_feature_dim": float(
                    spectral_learner.feature_map.block_sizes["raw_full"]
                ),
                "additions": float(spectral_state.additions),
                "replacements": float(spectral_state.replacements),
                "skipped_promotions": float(spectral_state.skipped_promotions),
                "mean_utility": float(
                    np.mean(spectral_state.utility[: spectral_state.active_count])
                    if spectral_state.active_count > 0
                    else 0.0
                ),
                "finite_failures": float(spectral_state.finite_failures),
            }
        )
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_spectral_classifier(
                    spectral_learner,
                    spectral_state,
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
    """Run the D12 sweep and write JSON/Markdown outputs."""
    args = parse_args()
    if args.smoke:
        args.steps = 80
        args.n_seeds = 1
        args.final_window = 20
        args.datasets = "controlled_frequency"
        args.basis_families = ("cheb_anova",)
        args.budgets = (24,)
        args.candidate_counts = (80,)
        args.degrees = (3,)
        args.interaction_orders = (3,)
        args.rff_count = 16
        args.anova_count = 32
        args.tensor_sketch_dim = 16
        args.chaos_count = 32
        args.ridgelet_count = 32
        args.tensor_cp_count = 32
        args.deep_ridgelet_count = 32
        args.deep_ridgelet_inner = 4
        args.wavelet_count = 16
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    configs = make_spectral_configs(args)
    candidate_methods = tuple(method_name(config) for config in configs)
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
            "basis_families": list(args.basis_families),
            "budgets": list(args.budgets),
            "candidate_counts": list(args.candidate_counts),
            "degrees": list(args.degrees),
            "interaction_orders": list(args.interaction_orders),
            "update_rules": list(args.update_rules),
            "rhos": list(args.rhos),
            "rls_delta": args.rls_delta,
            "nlms_step_size": args.nlms_step_size,
            "autostep_mu": args.autostep_mu,
            "autostep_tau": args.autostep_tau,
            "autostep_init_step_size": args.autostep_init_step_size,
            "utility_decay": args.utility_decay,
            "min_feature_age": args.min_feature_age,
            "replacement_interval": args.replacement_interval,
            "replace_margin": args.replace_margin,
            "promotions_per_step": args.promotions_per_step,
            "score_threshold": args.score_threshold,
            "input_clip": args.input_clip,
            "input_scale": args.input_scale,
            "feature_clip": args.feature_clip,
            "rff_count": args.rff_count,
            "rff_scale": args.rff_scale,
            "anova_count": args.anova_count,
            "tensor_sketch_dim": args.tensor_sketch_dim,
            "chaos_count": args.chaos_count,
            "ridgelet_count": args.ridgelet_count,
            "ridgelet_scale": args.ridgelet_scale,
            "tensor_cp_count": args.tensor_cp_count,
            "tensor_cp_scale": args.tensor_cp_scale,
            "deep_ridgelet_count": args.deep_ridgelet_count,
            "deep_ridgelet_inner": args.deep_ridgelet_inner,
            "deep_ridgelet_scale": args.deep_ridgelet_scale,
            "wavelet_count": args.wavelet_count,
            "wavelet_scales": list(args.wavelet_scales),
            "fourier_frequencies": list(args.fourier_frequencies),
            "feature_seed": args.feature_seed,
        },
        "datasets": datasets_meta,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "standalone_single_learner_spectral_tensor_probe",
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

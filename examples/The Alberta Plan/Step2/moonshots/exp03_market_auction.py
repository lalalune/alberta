#!/usr/bin/env python3
"""Exp03 moonshot: market auction feature economy for Step 2.

This is a deliberately small online supervised prototype.  A fixed-size
feature economy competes for slots in a linear multi-head learner.  Candidate
features are simple constructed transforms of the raw observation; recent
residual reduction is the bid signal.

Revisions:

* Rev A: tasks bid for features using gradient credit / recent residual
  reduction; low-revenue features are replaced.
* Rev B: bids are task-balanced so high-variance/easy tasks do not monopolize
  the feature budget.
* Rev C: features pay rent, save banked utility, and bankrupt features are
  replaced while historically useful features are protected by savings.

The comparators are the local fair MLP baselines and one UPGD setting used by
nearby Step 2 scripts.  Outputs are written only under
``output/moonshots/exp03_market_auction`` by default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    CompositionalStream,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
)

DEFAULT_OUTPUT_DIR = Path("output/moonshots/exp03_market_auction")
OBGD_KAPPA = 2.0


FeatureKind = Literal["raw", "pair", "triple", "sin", "cos", "tanh", "rtanh"]


@dataclass(frozen=True)
class FeatureSpec:
    """One constructed feature definition."""

    kind: FeatureKind
    indices: tuple[int, ...]
    freq: float = 1.0
    weights: tuple[float, ...] = ()
    bias: float = 0.0


@dataclass(frozen=True)
class AuctionConfig:
    """Auction learner hyperparameters."""

    name: str
    revision: Literal["A", "B", "C"]
    n_slots: int = 64
    candidate_count: int = 256
    replacement_interval: int = 80
    replacement_count: int = 4
    buffer_size: int = 240
    step_size: float = 0.25
    weight_decay: float = 1e-5
    revenue_decay: float = 0.985
    ridge: float = 1e-4
    feature_clip: float = 5.0
    weight_clip: float = 5.0
    min_age: int = 120
    rent: float = 0.0
    bank_decay: float = 0.995
    include_tanh_combos: bool = False
    cost_scale: float = 0.0
    trained_tanh_candidates: int = 0
    trained_tanh_steps: int = 4
    trained_tanh_lr: float = 0.05
    adapt_rtanh_lr: float = 0.0
    trained_tanh_bundles: int = 0
    trained_tanh_bundle_size: int = 4
    trained_tanh_bundle_steps: int = 8
    trained_tanh_bundle_lr: float = 0.03
    bundle_diversity_cost: float = 0.01
    nursery_size: int = 0
    nursery_offer_count: int = 0
    nursery_maturity: int = 120
    nursery_lr: float = 0.01
    nursery_readout_lr: float = 0.05
    nursery_score_decay: float = 0.99
    nursery_reset_interval: int = 240
    nursery_reset_fraction: float = 0.1
    meta_constructor: bool = False
    meta_constructor_lr: float = 0.01
    meta_constructor_temperature: float = 0.25
    meta_constructor_floor: float = 0.05
    meta_constructor_survival_weight: float = 1.0


@dataclass(frozen=True)
class MethodConfig:
    """Baseline method metadata."""

    name: str
    method_type: Literal["mlp", "upgd"]
    hidden_sizes: tuple[int, ...] = (64,)
    step_size: float = 0.03
    sparsity: float = 0.5
    use_layer_norm: bool = True
    perturbation_sigma: float = 3e-4
    utility_decay: float = 0.995
    perturbation_beta: float = 2.0
    perturbation_interval: int = 1


BASELINES: tuple[MethodConfig, ...] = (
    MethodConfig("mlp64", "mlp", (64,)),
    MethodConfig("mlp64_64", "mlp", (64, 64)),
    MethodConfig("upgd64_sigma3e_4", "upgd", (64,)),
)

AUCTION_REVISIONS: tuple[AuctionConfig, ...] = (
    AuctionConfig("auction_revA_revenue", "A"),
    AuctionConfig("auction_revB_task_balanced", "B"),
    AuctionConfig(
        "auction_revC_rent_bankruptcy",
        "C",
        rent=0.002,
        bank_decay=0.997,
        revenue_decay=0.99,
    ),
)


def stable_key(seed: int, label: str) -> jax.Array:
    """Stable JAX key for a seed/label pair."""
    checksum = zlib.crc32(label.encode("utf-8")) & 0x7FFFFFFF
    return jr.fold_in(jr.key(seed), checksum)


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a Step 2 stream into NumPy arrays."""
    state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(step_fn, state, jnp.arange(num_steps))
    observations.block_until_ready()
    targets.block_until_ready()
    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def synthetic_stream_factories() -> dict[str, tuple[Any, int, int]]:
    """Return the out-of-class Step 2 stream factories used in this probe."""
    return {
        "polynomial": (
            lambda: OutOfClassPolynomialStream(
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=400,
                active_triples_per_context=2,
                noise_std=0.05,
            ),
            8,
            3,
        ),
        "frequency": (
            lambda: FrequencyMismatchStream(
                feature_dim=4,
                n_tasks=2,
                n_components_per_task=3,
                n_contexts=4,
                context_length=400,
                noise_std=0.05,
            ),
            4,
            2,
        ),
        "compositional": (
            lambda: CompositionalStream(
                feature_dim=6,
                n_tasks=3,
                inner_hidden=4,
                outer_components=5,
                n_contexts=4,
                context_length=400,
                noise_std=0.05,
            ),
            6,
            3,
        ),
    }


def make_mlp(config: MethodConfig, n_heads: int) -> MultiHeadMLPLearner:
    """Create a local fair MLP comparator."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=config.hidden_sizes,
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=config.sparsity,
        use_layer_norm=config.use_layer_norm,
    )


def make_upgd(config: MethodConfig, n_heads: int) -> UPGDLearner:
    """Create the UPGD comparator."""
    return UPGDLearner(
        n_heads=n_heads,
        hidden_sizes=config.hidden_sizes,
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=OBGD_KAPPA),
        sparsity=config.sparsity,
        use_layer_norm=config.use_layer_norm,
        perturbation_sigma=config.perturbation_sigma,
        utility_decay=config.utility_decay,
        perturbation_beta=config.perturbation_beta,
        perturbation_interval=config.perturbation_interval,
    )


def active_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error over active targets."""
    active = ~jnp.isnan(targets)
    safe_targets = jnp.where(active, targets, 0.0)
    se = jnp.where(active, (predictions - safe_targets) ** 2, 0.0)
    return jnp.sum(se) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)


def run_jax_baseline(
    learner: Any,
    key: jax.Array,
    observations_np: np.ndarray,
    targets_np: np.ndarray,
) -> np.ndarray:
    """Run a JAX learner and return one MSE value per step."""
    observations = jnp.asarray(observations_np)
    targets = jnp.asarray(targets_np)
    state = learner.init(observations.shape[1], key)

    def step_fn(carry: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, jax.Array]:
        obs, tgt = inputs
        result = learner.update(carry, obs, tgt)
        return result.state, active_mse(result.predictions, tgt)

    _, curve = jax.lax.scan(step_fn, state, (observations, targets))
    curve.block_until_ready()
    return np.asarray(curve, dtype=np.float64)


def all_feature_specs(
    feature_dim: int,
    include_tanh_combos: bool = False,
) -> list[FeatureSpec]:
    """Build a finite constructed-feature catalogue."""
    specs: list[FeatureSpec] = []
    for i in range(feature_dim):
        specs.append(FeatureSpec("raw", (i,)))
    for i in range(feature_dim):
        for j in range(i + 1, feature_dim):
            specs.append(FeatureSpec("pair", (i, j)))
    for i in range(feature_dim):
        for j in range(i + 1, feature_dim):
            for k in range(j + 1, feature_dim):
                specs.append(FeatureSpec("triple", (i, j, k)))
    for freq in (1.0, 2.0, 3.0, 5.0):
        for i in range(feature_dim):
            specs.append(FeatureSpec("sin", (i,), freq))
            specs.append(FeatureSpec("cos", (i,), freq))
    if include_tanh_combos:
        for scale in (0.5, 1.0, 2.0):
            for i in range(feature_dim):
                specs.append(FeatureSpec("tanh", (i,), scale))
            for i in range(feature_dim):
                for j in range(i + 1, feature_dim):
                    for signs in itertools.product((-1, 1), repeat=2):
                        signed = tuple(
                            sign * (idx + 1)
                            for sign, idx in zip(signs, (i, j), strict=True)
                        )
                        specs.append(FeatureSpec("tanh", signed, scale))
            for i in range(feature_dim):
                for j in range(i + 1, feature_dim):
                    for k in range(j + 1, feature_dim):
                        for signs in itertools.product((-1, 1), repeat=3):
                            signed = tuple(
                                sign * (idx + 1)
                                for sign, idx in zip(signs, (i, j, k), strict=True)
                            )
                            specs.append(FeatureSpec("tanh", signed, scale))
    return specs


def eval_feature(spec: FeatureSpec, x: np.ndarray, clip: float) -> float:
    """Evaluate one feature on one observation."""
    if spec.kind == "raw":
        value = x[spec.indices[0]]
    elif spec.kind == "pair":
        i, j = spec.indices
        value = x[i] * x[j]
    elif spec.kind == "triple":
        i, j, k = spec.indices
        value = x[i] * x[j] * x[k]
    elif spec.kind == "sin":
        value = math.sin(spec.freq * x[spec.indices[0]])
    elif spec.kind == "cos":
        value = math.cos(spec.freq * x[spec.indices[0]])
    elif spec.kind == "tanh":
        values = [math.copysign(float(x[abs(i) - 1]), float(i)) for i in spec.indices]
        value = math.tanh(spec.freq * float(np.sum(values)) / math.sqrt(len(values)))
    elif spec.kind == "rtanh":
        value = math.tanh(float(np.dot(np.asarray(spec.weights, dtype=np.float32), x)) + spec.bias)
    else:  # pragma: no cover - exhaustive Literal guard
        raise ValueError(f"unknown feature kind: {spec.kind}")
    return float(np.clip(value, -clip, clip))


def eval_feature_matrix(
    specs: list[FeatureSpec],
    x: np.ndarray,
    clip: float,
) -> np.ndarray:
    """Evaluate specs on a batch of observations."""
    out = np.empty((x.shape[0], len(specs)), dtype=np.float32)
    for idx, spec in enumerate(specs):
        if spec.kind == "raw":
            out[:, idx] = x[:, spec.indices[0]]
        elif spec.kind == "pair":
            i, j = spec.indices
            out[:, idx] = x[:, i] * x[:, j]
        elif spec.kind == "triple":
            i, j, k = spec.indices
            out[:, idx] = x[:, i] * x[:, j] * x[:, k]
        elif spec.kind == "sin":
            out[:, idx] = np.sin(spec.freq * x[:, spec.indices[0]])
        elif spec.kind == "cos":
            out[:, idx] = np.cos(spec.freq * x[:, spec.indices[0]])
        elif spec.kind == "tanh":
            vals = np.zeros(x.shape[0], dtype=np.float32)
            for signed_idx in spec.indices:
                sign = 1.0 if signed_idx > 0 else -1.0
                vals += sign * x[:, abs(signed_idx) - 1]
            vals = vals / math.sqrt(len(spec.indices))
            out[:, idx] = np.tanh(spec.freq * vals)
        elif spec.kind == "rtanh":
            weights = np.asarray(spec.weights, dtype=np.float32)
            out[:, idx] = np.tanh(x @ weights + spec.bias)
    np.clip(out, -clip, clip, out=out)
    return out


def feature_cost(spec: FeatureSpec) -> float:
    """Small complexity price used only when cost scaling is enabled."""
    if spec.kind == "raw":
        return 0.0
    if spec.kind in ("sin", "cos"):
        return 1.0
    if spec.kind == "rtanh":
        return 4.0
    return float(len(spec.indices))


class MarketAuctionLearner:
    """Budgeted online feature economy with task bids and slot replacement."""

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int,
        config: AuctionConfig,
        seed: int,
    ):
        self.feature_dim = feature_dim
        self.n_tasks = n_tasks
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.catalogue = all_feature_specs(
            feature_dim,
            include_tanh_combos=config.include_tanh_combos,
        )
        self.specs = self._initial_specs()
        self.protected = np.asarray([spec.kind == "raw" for spec in self.specs], dtype=bool)
        self.weights = np.zeros((n_tasks, config.n_slots), dtype=np.float32)
        self.bias = np.zeros(n_tasks, dtype=np.float32)
        self.revenue = np.zeros(config.n_slots, dtype=np.float32)
        self.bank = np.ones(config.n_slots, dtype=np.float32) * 0.01
        self.age = np.zeros(config.n_slots, dtype=np.int32)
        self.task_bid_ema = np.ones(n_tasks, dtype=np.float32)
        self.buffer_x: list[np.ndarray] = []
        self.buffer_y: list[np.ndarray] = []
        self.replacements = 0
        self.auctions = 0
        self.bundle_admissions = 0
        self.constructor_score: dict[str, float] = {
            "pair": 0.0,
            "triple": 0.0,
            "sin": 0.0,
            "cos": 0.0,
            "tanh": 0.0,
            "rtanh": 0.0,
        }
        self.constructor_survival_score: dict[str, float] = dict(self.constructor_score)
        self.constructor_arm_score: dict[str, float] = {}
        self.constructor_arm_survival_score: dict[str, float] = {}
        self.nursery_w = np.empty((0, feature_dim), dtype=np.float32)
        self.nursery_b = np.empty(0, dtype=np.float32)
        self.nursery_readout = np.empty((0, n_tasks), dtype=np.float32)
        self.nursery_score = np.empty(0, dtype=np.float32)
        self.nursery_age = np.empty(0, dtype=np.int32)
        self.nursery_resets = 0
        if config.nursery_size > 0:
            self._init_nursery(config.nursery_size)

    def _initial_specs(self) -> list[FeatureSpec]:
        raw = [FeatureSpec("raw", (i,)) for i in range(self.feature_dim)]
        remaining = self.config.n_slots - len(raw)
        non_raw = [spec for spec in self.catalogue if spec.kind != "raw"]
        chosen = self.rng.choice(
            len(non_raw),
            size=remaining,
            replace=remaining > len(non_raw),
        )
        return [*raw, *[non_raw[int(idx)] for idx in chosen]]

    def _feature_vector(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            [eval_feature(spec, x, self.config.feature_clip) for spec in self.specs],
            dtype=np.float32,
        )

    def _append_buffer(self, x: np.ndarray, y: np.ndarray) -> None:
        self.buffer_x.append(np.asarray(x, dtype=np.float32).copy())
        self.buffer_y.append(np.asarray(y, dtype=np.float32).copy())
        excess = len(self.buffer_x) - self.config.buffer_size
        if excess > 0:
            del self.buffer_x[:excess]
            del self.buffer_y[:excess]

    def _init_nursery(self, count: int) -> None:
        """Initialize persistent residual-trained candidate units."""
        self.nursery_w = self.rng.normal(size=(count, self.feature_dim)).astype(np.float32)
        norms = np.maximum(np.linalg.norm(self.nursery_w, axis=1, keepdims=True), 1e-6)
        self.nursery_w = self.nursery_w / norms
        self.nursery_b = self.rng.normal(scale=0.1, size=count).astype(np.float32)
        self.nursery_readout = np.zeros((count, self.n_tasks), dtype=np.float32)
        self.nursery_score = np.zeros(count, dtype=np.float32)
        self.nursery_age = np.zeros(count, dtype=np.int32)

    def _reset_nursery_units(self, indices: np.ndarray) -> None:
        """Re-seed weak nursery units so the constructor keeps exploring."""
        if indices.size == 0:
            return
        w = self.rng.normal(size=(indices.size, self.feature_dim)).astype(np.float32)
        w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-6)
        self.nursery_w[indices] = w
        self.nursery_b[indices] = self.rng.normal(scale=0.1, size=indices.size).astype(np.float32)
        self.nursery_readout[indices] = 0.0
        self.nursery_score[indices] = 0.0
        self.nursery_age[indices] = 0
        self.nursery_resets += int(indices.size)

    def _update_nursery(self, x: np.ndarray, err: np.ndarray, step: int) -> None:
        """Train persistent candidate features against current residuals."""
        if self.nursery_w.shape[0] == 0:
            return
        h = np.tanh(self.nursery_w @ x.astype(np.float32) + self.nursery_b)
        pred_residual = h[:, None] * self.nursery_readout
        local_err = err[None, :] - pred_residual
        before = float(np.mean(err**2))
        after = np.mean(local_err**2, axis=1)
        reduction = np.maximum(0.0, before - after)
        decay = float(np.clip(self.config.nursery_score_decay, 0.0, 1.0))
        self.nursery_score = decay * self.nursery_score + (1.0 - decay) * reduction.astype(
            np.float32
        )

        readout_lr = self.config.nursery_readout_lr / (1e-3 + h**2)
        self.nursery_readout += readout_lr[:, None].astype(np.float32) * local_err * h[:, None]
        credit = np.sum(local_err * self.nursery_readout, axis=1)
        dz = credit * (1.0 - h * h)
        self.nursery_w += self.config.nursery_lr * dz[:, None].astype(np.float32) * x[None, :]
        self.nursery_b += self.config.nursery_lr * dz.astype(np.float32)
        norms = np.maximum(np.linalg.norm(self.nursery_w, axis=1, keepdims=True), 1e-6)
        too_large = norms > 4.0
        self.nursery_w = np.where(too_large, self.nursery_w * (4.0 / norms), self.nursery_w)
        self.nursery_readout = np.nan_to_num(
            self.nursery_readout,
            nan=0.0,
            posinf=self.config.weight_clip,
            neginf=-self.config.weight_clip,
        )
        np.clip(
            self.nursery_readout,
            -self.config.weight_clip,
            self.config.weight_clip,
            out=self.nursery_readout,
        )
        self.nursery_age += 1

        interval = max(1, self.config.nursery_reset_interval)
        if (step + 1) % interval == 0:
            mature = np.flatnonzero(self.nursery_age >= self.config.nursery_maturity)
            n_reset = int(round(self.config.nursery_reset_fraction * self.nursery_w.shape[0]))
            n_reset = min(max(0, n_reset), mature.size)
            if n_reset > 0:
                weak = mature[np.argsort(self.nursery_score[mature])[:n_reset]]
                self._reset_nursery_units(weak)

    def _nursery_candidates(self) -> list[FeatureSpec]:
        """Offer the best persistent candidate units to the auction."""
        if self.nursery_w.shape[0] == 0 or self.config.nursery_offer_count <= 0:
            return []
        mature = np.flatnonzero(self.nursery_age >= self.config.nursery_maturity)
        if mature.size == 0:
            return []
        count = min(self.config.nursery_offer_count, mature.size)
        chosen = mature[np.argsort(self.nursery_score[mature])[-count:]]
        specs = []
        for idx in chosen:
            specs.append(
                FeatureSpec(
                    "rtanh",
                    (),
                    1.0,
                    tuple(float(v) for v in np.round(self.nursery_w[int(idx)], 5)),
                    float(round(float(self.nursery_b[int(idx)]), 5)),
                )
            )
        return specs

    def _adapt_rtanh_features(self, x: np.ndarray, err: np.ndarray, z: np.ndarray) -> None:
        """Continue adapting admitted residual-tanh features with local gradients."""
        if self.config.adapt_rtanh_lr <= 0.0:
            return
        updated = False
        specs = list(self.specs)
        for slot, spec in enumerate(specs):
            if spec.kind != "rtanh":
                continue
            h = float(z[slot])
            credit = float(np.mean(err * self.weights[:, slot]))
            grad_factor = credit * (1.0 - h * h)
            weights = np.asarray(spec.weights, dtype=np.float32)
            weights += self.config.adapt_rtanh_lr * grad_factor * x.astype(np.float32)
            bias = float(spec.bias + self.config.adapt_rtanh_lr * grad_factor)
            norm = max(float(np.linalg.norm(weights)), 1e-6)
            if norm > 4.0:
                weights *= 4.0 / norm
            specs[slot] = FeatureSpec(
                "rtanh",
                (),
                1.0,
                tuple(float(v) for v in np.round(weights, 5)),
                float(round(bias, 5)),
            )
            updated = True
        if updated:
            self.specs = specs

    def _batch_prediction(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = eval_feature_matrix(self.specs, x, self.config.feature_clip)
        weights = np.nan_to_num(
            self.weights.astype(np.float64),
            nan=0.0,
            posinf=self.config.weight_clip,
            neginf=-self.config.weight_clip,
        )
        weights = np.clip(weights, -self.config.weight_clip, self.config.weight_clip)
        bias = np.nan_to_num(self.bias.astype(np.float64), nan=0.0)
        bias = np.clip(bias, -self.config.weight_clip, self.config.weight_clip)
        z64 = np.nan_to_num(
            z.astype(np.float64),
            nan=0.0,
            posinf=self.config.feature_clip,
            neginf=-self.config.feature_clip,
        )
        z64 = np.clip(z64, -self.config.feature_clip, self.config.feature_clip)
        pred = np.sum(z64[:, :, None] * weights.T[None, :, :], axis=1) + bias[None, :]
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        return pred.astype(np.float32), z

    def _task_scale(self, residual: np.ndarray) -> np.ndarray:
        if self.config.revision == "A":
            return np.ones(self.n_tasks, dtype=np.float32)
        variance = np.mean(residual**2, axis=0).astype(np.float32)
        self.task_bid_ema = 0.95 * self.task_bid_ema + 0.05 * np.maximum(variance, 1e-4)
        return np.sqrt(self.task_bid_ema + 1e-4)

    def _score_candidates(
        self,
        candidate_specs: list[FeatureSpec],
    ) -> tuple[np.ndarray, np.ndarray]:
        x_buf = np.stack(self.buffer_x, axis=0)
        y_buf = np.stack(self.buffer_y, axis=0)
        pred, _ = self._batch_prediction(x_buf)
        residual = np.nan_to_num(
            y_buf.astype(np.float64) - pred.astype(np.float64),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        scale = self._task_scale(residual)
        cmat = eval_feature_matrix(candidate_specs, x_buf, self.config.feature_clip)
        cmat = cmat.astype(np.float64)
        cmat = cmat - np.mean(cmat, axis=0, keepdims=True)
        var = np.mean(cmat**2, axis=0) + self.config.ridge
        corr = np.sum(cmat[:, :, None] * residual[:, None, :], axis=0) / residual.shape[0]
        per_task_reduction = (corr**2) / var[:, None]
        balanced = per_task_reduction / (scale[None, :] ** 2)
        balanced = np.nan_to_num(balanced, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.mean(balanced, axis=1)
        if self.config.cost_scale > 0.0:
            costs = np.asarray([feature_cost(spec) for spec in candidate_specs], dtype=np.float64)
            scores = scores - self.config.cost_scale * costs
        init_weights = (0.25 * corr / var[:, None]).T.astype(np.float32)
        init_weights = np.nan_to_num(init_weights, nan=0.0, posinf=0.0, neginf=0.0)
        init_weights = np.clip(
            init_weights,
            -self.config.weight_clip,
            self.config.weight_clip,
        )
        return scores.astype(np.float32), init_weights

    def _sample_catalogue_candidates(self) -> list[FeatureSpec]:
        """Sample candidates, optionally using a learned constructor prior."""
        n = min(self.config.candidate_count, len(self.catalogue))
        if n <= 0:
            return []
        if not self.config.meta_constructor:
            candidate_indices = self.rng.choice(len(self.catalogue), size=n, replace=False)
            return [self.catalogue[int(idx)] for idx in candidate_indices]

        logits = np.asarray(
            [
                self.constructor_arm_score.get(
                    self._constructor_label(spec),
                    self.constructor_score.get(spec.kind, 0.0),
                )
                + self.config.meta_constructor_survival_weight
                * self.constructor_arm_survival_score.get(
                    self._constructor_label(spec),
                    self.constructor_survival_score.get(spec.kind, 0.0),
                )
                - self.config.cost_scale * feature_cost(spec)
                for spec in self.catalogue
            ],
            dtype=np.float64,
        )
        temp = max(float(self.config.meta_constructor_temperature), 1e-6)
        logits = np.clip(logits / temp, -20.0, 20.0)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        floor = float(np.clip(self.config.meta_constructor_floor, 0.0, 1.0))
        probs = (1.0 - floor) * probs + floor / len(probs)
        probs = probs / np.sum(probs)
        candidate_indices = self.rng.choice(
            len(self.catalogue),
            size=n,
            replace=False,
            p=probs,
        )
        return [self.catalogue[int(idx)] for idx in candidate_indices]

    def _constructor_label(self, spec: FeatureSpec) -> str:
        """Return the constructor arm that produced a feature."""
        if spec.kind in ("raw", "pair", "triple", "rtanh"):
            return spec.kind
        if spec.kind in ("sin", "cos"):
            return f"{spec.kind}:f{spec.freq:g}"
        if spec.kind == "tanh":
            return f"tanh:a{len(spec.indices)}:s{spec.freq:g}"
        return spec.kind

    def _update_constructor_score(
        self,
        scores: dict[str, float],
        kind: str,
        target: float,
    ) -> None:
        """EMA update for a constructor family score."""
        if kind == "raw":
            return
        lr = float(np.clip(self.config.meta_constructor_lr, 0.0, 1.0))
        old = scores.get(kind, 0.0)
        scores[kind] = (1.0 - lr) * old + lr * float(max(target, 0.0))

    def _slot_survival_credit(self, slot: int) -> float:
        """Delayed credit for a slot based on current utility after survival pressure."""
        spec = self.specs[int(slot)]
        if spec.kind == "raw":
            return 0.0
        pressure = float(max(self.revenue[int(slot)], 0.0))
        if self.config.revision == "C":
            pressure = float(max(pressure, self.bank[int(slot)], 0.0))
        mature = min(float(self.age[int(slot)]) / float(max(1, self.config.min_age)), 1.0)
        age_bonus = 1.0 + 0.05 * math.log1p(
            float(self.age[int(slot)]) / float(max(1, self.config.replacement_interval))
        )
        return mature * age_bonus * pressure / (1.0 + feature_cost(spec))

    def _credit_constructor_survival(
        self,
        slots: np.ndarray | list[int],
        discount: float = 1.0,
    ) -> None:
        """Credit constructors from features that reached a later auction."""
        if not self.config.meta_constructor:
            return
        for slot in slots:
            spec = self.specs[int(slot)]
            target = discount * self._slot_survival_credit(int(slot))
            self._update_constructor_score(self.constructor_survival_score, spec.kind, target)
            self._update_constructor_score(
                self.constructor_arm_survival_score,
                self._constructor_label(spec),
                target,
            )

    def _trained_tanh_candidates(self) -> list[FeatureSpec]:
        """Generate tanh units briefly adapted to recent residuals."""
        if self.config.trained_tanh_candidates <= 0 or len(self.buffer_x) < 32:
            return []
        x_buf = np.stack(self.buffer_x, axis=0).astype(np.float32)
        y_buf = np.stack(self.buffer_y, axis=0).astype(np.float32)
        pred, _ = self._batch_prediction(x_buf)
        residual = y_buf - pred
        candidates: list[FeatureSpec] = []
        for _ in range(self.config.trained_tanh_candidates):
            w = self.rng.normal(size=self.feature_dim).astype(np.float32)
            w /= max(float(np.linalg.norm(w)), 1e-6)
            b = float(self.rng.normal(scale=0.1))
            readout = np.zeros(self.n_tasks, dtype=np.float32)
            for _step in range(max(0, self.config.trained_tanh_steps)):
                h = np.tanh(x_buf @ w + b)
                denom = float(np.dot(h, h) + self.config.ridge)
                readout = (residual.T @ h / denom).astype(np.float32)
                err = residual - h[:, None] * readout[None, :]
                dh = -2.0 * (err @ readout) / max(1, x_buf.shape[0])
                dz = dh * (1.0 - h * h)
                grad_w = x_buf.T @ dz
                grad_b = float(np.sum(dz))
                w -= self.config.trained_tanh_lr * grad_w.astype(np.float32)
                b -= self.config.trained_tanh_lr * grad_b
                norm = max(float(np.linalg.norm(w)), 1e-6)
                if norm > 4.0:
                    w *= 4.0 / norm
            rounded = tuple(float(v) for v in np.round(w, 5))
            candidates.append(FeatureSpec("rtanh", (), 1.0, rounded, float(round(b, 5))))
        return candidates

    def _fit_rtanh_bundle(
        self,
        x_buf: np.ndarray,
        residual: np.ndarray,
    ) -> tuple[list[FeatureSpec], np.ndarray, float]:
        """Fit a small tanh feature bundle to residuals on the buffer."""
        bundle_size = max(1, self.config.trained_tanh_bundle_size)
        w = self.rng.normal(size=(bundle_size, self.feature_dim)).astype(np.float32)
        w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-6)
        b = self.rng.normal(scale=0.1, size=bundle_size).astype(np.float32)
        readout = np.zeros((self.n_tasks, bundle_size), dtype=np.float32)
        n = max(1, x_buf.shape[0])
        base_loss = float(np.mean(residual**2))

        for _step in range(max(0, self.config.trained_tanh_bundle_steps)):
            h = np.tanh(x_buf @ w.T + b[None, :]).astype(np.float32)
            gram = h.T @ h + self.config.ridge * np.eye(bundle_size, dtype=np.float32)
            rhs = h.T @ residual
            try:
                readout = np.linalg.solve(gram, rhs).T.astype(np.float32)
            except np.linalg.LinAlgError:
                readout = (np.linalg.pinv(gram) @ rhs).T.astype(np.float32)
            err = residual - h @ readout.T
            dz = ((err @ readout) * (1.0 - h * h)) / float(n)
            w += self.config.trained_tanh_bundle_lr * (dz.T @ x_buf).astype(np.float32)
            b += self.config.trained_tanh_bundle_lr * np.sum(dz, axis=0).astype(np.float32)
            norms = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-6)
            too_large = norms > 4.0
            w = np.where(too_large, w * (4.0 / norms), w)

        h = np.tanh(x_buf @ w.T + b[None, :]).astype(np.float32)
        gram = h.T @ h + self.config.ridge * np.eye(bundle_size, dtype=np.float32)
        rhs = h.T @ residual
        try:
            readout = np.linalg.solve(gram, rhs).T.astype(np.float32)
        except np.linalg.LinAlgError:
            readout = (np.linalg.pinv(gram) @ rhs).T.astype(np.float32)
        err = residual - h @ readout.T
        final_loss = float(np.mean(err**2))
        centered = h - np.mean(h, axis=0, keepdims=True)
        denom = np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), 1e-6)
        corr = (centered / denom).T @ (centered / denom)
        off_diag = corr - np.eye(bundle_size, dtype=np.float32)
        diversity_penalty = float(np.mean(np.abs(off_diag)))
        score = base_loss - final_loss
        score -= self.config.bundle_diversity_cost * diversity_penalty
        zero_params = (0.0,) * self.feature_dim
        score -= self.config.cost_scale * feature_cost(
            FeatureSpec("rtanh", (), 1.0, zero_params)
        )

        specs = [
            FeatureSpec(
                "rtanh",
                (),
                1.0,
                tuple(float(v) for v in np.round(w[idx], 5)),
                float(round(float(b[idx]), 5)),
            )
            for idx in range(bundle_size)
        ]
        return specs, readout, float(score)

    def _bundle_auction(self) -> bool:
        """Admit jointly trained residual-tanh bundles if net revenue clears cost."""
        if self.config.trained_tanh_bundles <= 0 or len(self.buffer_x) < 32:
            return False
        x_buf = np.stack(self.buffer_x, axis=0).astype(np.float32)
        y_buf = np.stack(self.buffer_y, axis=0).astype(np.float32)
        pred, _ = self._batch_prediction(x_buf)
        residual = y_buf - pred
        best_specs: list[FeatureSpec] | None = None
        best_readout: np.ndarray | None = None
        best_score = -np.inf
        for _ in range(self.config.trained_tanh_bundles):
            specs, readout, score = self._fit_rtanh_bundle(x_buf, residual)
            if score > best_score:
                best_specs = specs
                best_readout = readout
                best_score = score
        if best_specs is None or best_readout is None:
            return False

        available = np.flatnonzero(~self.protected)
        eligible = available[self.age[available] >= self.config.min_age]
        if eligible.size < len(best_specs):
            eligible = available
        if eligible.size < len(best_specs):
            return False
        pressure = self.revenue[eligible] + 0.2 * self.bank[eligible]
        slot_order = eligible[np.argsort(pressure)[: len(best_specs)]]
        incumbent = float(np.mean(np.maximum(self.revenue[slot_order], self.bank[slot_order])))
        score_per_slot = best_score / float(len(best_specs))
        if score_per_slot <= incumbent:
            return False
        for idx, slot in enumerate(slot_order):
            self._credit_constructor_survival([int(slot)], discount=0.25)
            self.specs[int(slot)] = best_specs[idx]
            self.weights[:, int(slot)] = best_readout[:, idx]
            self.revenue[int(slot)] = score_per_slot
            self.bank[int(slot)] = max(score_per_slot, 0.01)
            self.age[int(slot)] = 0
            self.replacements += 1
        self.bundle_admissions += 1
        return True

    def _auction(self) -> None:
        if len(self.buffer_x) < max(24, self.config.replacement_interval // 2):
            return
        self.auctions += 1
        self._credit_constructor_survival(np.flatnonzero(~self.protected))
        self._bundle_auction()

        candidate_specs = self._sample_catalogue_candidates()
        candidate_specs.extend(self._trained_tanh_candidates())
        candidate_specs.extend(self._nursery_candidates())
        existing_keys = set(self.specs)
        candidate_specs = [spec for spec in candidate_specs if spec not in existing_keys]
        if not candidate_specs:
            return

        candidate_scores, candidate_weights = self._score_candidates(candidate_specs)
        order = np.argsort(candidate_scores)[::-1]
        available = np.flatnonzero(~self.protected)
        if self.config.revision == "C":
            bankrupt = available[
                (self.bank[available] < 0.0) & (self.age[available] >= self.config.min_age)
            ]
            if bankrupt.size:
                slot_order = bankrupt[np.argsort(self.bank[bankrupt])]
            else:
                pressure = self.revenue[available] + 0.2 * self.bank[available]
                slot_order = available[np.argsort(pressure)]
        else:
            eligible = available[self.age[available] >= self.config.min_age]
            if eligible.size == 0:
                return
            slot_order = eligible[np.argsort(self.revenue[eligible])]

        n_try = min(self.config.replacement_count, len(order), len(slot_order))
        for pos in range(n_try):
            cand_idx = int(order[pos])
            slot = int(slot_order[pos])
            score = float(candidate_scores[cand_idx])
            incumbent = float(self.revenue[slot])
            if self.config.revision == "C":
                incumbent = float(max(self.bank[slot], self.revenue[slot]))
            if score <= incumbent and not (self.config.revision == "C" and self.bank[slot] < 0.0):
                continue
            self._credit_constructor_survival([slot], discount=0.25)
            self.specs[slot] = candidate_specs[cand_idx]
            self.weights[:, slot] = candidate_weights[:, cand_idx]
            self.revenue[slot] = score
            self.bank[slot] = max(score, 0.01)
            self.age[slot] = 0
            self.replacements += 1

    def update(self, x: np.ndarray, y: np.ndarray, step: int) -> float:
        """One online supervised update; returns pre-update MSE."""
        z = self._feature_vector(x)
        self.weights = np.nan_to_num(
            self.weights,
            nan=0.0,
            posinf=self.config.weight_clip,
            neginf=-self.config.weight_clip,
        )
        np.clip(self.weights, -self.config.weight_clip, self.config.weight_clip, out=self.weights)
        np.clip(self.bias, -self.config.weight_clip, self.config.weight_clip, out=self.bias)
        z64 = np.nan_to_num(
            z.astype(np.float64),
            nan=0.0,
            posinf=self.config.feature_clip,
            neginf=-self.config.feature_clip,
        )
        z64 = np.clip(z64, -self.config.feature_clip, self.config.feature_clip)
        pred = np.sum(self.weights.astype(np.float64) * z64[None, :], axis=1) + self.bias.astype(
            np.float64
        )
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        err = np.nan_to_num(y - pred, nan=0.0, posinf=100.0, neginf=-100.0)
        err = np.clip(err, -100.0, 100.0)
        mse = float(np.mean(err**2))
        self._update_nursery(x, err, step)
        self._adapt_rtanh_features(x, err, z)

        # Online task bids from counterfactual loss reduction and gradient credit.
        contribution = self.weights * z[None, :]
        counterfactual_err = err[:, None] + contribution
        reduction = 0.5 * (counterfactual_err**2 - err[:, None] ** 2)
        grad_credit = np.abs(err[:, None] * contribution)
        bids = np.maximum(reduction, 0.0) + 0.05 * grad_credit
        if self.config.revision in ("B", "C"):
            denom = np.sqrt(self.task_bid_ema[:, None] + 1e-4)
            bids = bids / denom
            self.task_bid_ema = 0.995 * self.task_bid_ema + 0.005 * np.maximum(err**2, 1e-4)
        bid_total = np.mean(bids, axis=0)
        if self.config.meta_constructor:
            for spec, credit in zip(self.specs, bid_total, strict=True):
                normalized_credit = float(credit) / (1.0 + feature_cost(spec))
                self._update_constructor_score(
                    self.constructor_score,
                    spec.kind,
                    normalized_credit,
                )
                self._update_constructor_score(
                    self.constructor_arm_score,
                    self._constructor_label(spec),
                    normalized_credit,
                )
        self.revenue = self.config.revenue_decay * self.revenue + (
            1.0 - self.config.revenue_decay
        ) * bid_total.astype(np.float32)
        if self.config.revision == "C":
            self.bank = (
                self.config.bank_decay * self.bank + bid_total.astype(np.float32) - self.config.rent
            )

        self.weights *= 1.0 - self.config.weight_decay
        normalizer = 1e-3 + float(np.dot(z, z))
        self.weights += self.config.step_size * err[:, None] * z[None, :] / normalizer
        self.bias += 0.05 * self.config.step_size * err
        np.clip(self.weights, -self.config.weight_clip, self.config.weight_clip, out=self.weights)
        np.clip(self.bias, -self.config.weight_clip, self.config.weight_clip, out=self.bias)
        self.weights = np.nan_to_num(
            self.weights,
            nan=0.0,
            posinf=self.config.weight_clip,
            neginf=-self.config.weight_clip,
        )
        self.bias = np.nan_to_num(
            self.bias,
            nan=0.0,
            posinf=self.config.weight_clip,
            neginf=-self.config.weight_clip,
        )
        self.age += 1
        self._append_buffer(x, y)
        if (step + 1) % self.config.replacement_interval == 0:
            self._auction()
        return mse

    def diagnostics(self) -> dict[str, Any]:
        """Return final economy diagnostics."""
        kinds: dict[str, int] = defaultdict(int)
        for spec in self.specs:
            kinds[spec.kind] += 1
        return {
            "feature_kinds": dict(sorted(kinds.items())),
            "replacements": int(self.replacements),
            "auctions": int(self.auctions),
            "mean_revenue": float(np.mean(self.revenue)),
            "min_revenue": float(np.min(self.revenue)),
            "mean_bank": float(np.mean(self.bank)),
            "bankrupt_slots": int(np.sum(self.bank < 0.0)),
            "bundle_admissions": int(self.bundle_admissions),
            "nursery_resets": int(self.nursery_resets),
            "nursery_mean_score": (
                float(np.mean(self.nursery_score)) if self.nursery_score.size else 0.0
            ),
            "nursery_max_score": (
                float(np.max(self.nursery_score)) if self.nursery_score.size else 0.0
            ),
            "constructor_score": dict(self.constructor_score),
            "constructor_survival_score": dict(self.constructor_survival_score),
            "constructor_arm_score": dict(self.constructor_arm_score),
            "constructor_arm_survival_score": dict(self.constructor_arm_survival_score),
            "final_specs": [asdict(spec) for spec in self.specs],
        }


def run_auction(
    config: AuctionConfig,
    seed: int,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one auction revision on one materialized stream."""
    learner = MarketAuctionLearner(
        feature_dim=observations.shape[1],
        n_tasks=targets.shape[1],
        config=config,
        seed=seed,
    )
    curve = np.empty(observations.shape[0], dtype=np.float64)
    for step, (x, y) in enumerate(zip(observations, targets, strict=True)):
        curve[step] = learner.update(x, y, step)
    return curve, learner.diagnostics()


def summarize_curve(curve: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize an online MSE curve."""
    window = min(final_window, curve.shape[0])
    return {
        "online_mean_mse": float(np.mean(curve)),
        "final_window_mse": float(np.mean(curve[-window:])),
        "first_half_mse": float(np.mean(curve[: max(1, curve.shape[0] // 2)])),
    }


def stderr(values: list[float] | np.ndarray) -> float:
    """Sample standard error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate flat run records by scenario and method."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], record["method"])].append(record)
    aggregate: dict[str, dict[str, Any]] = defaultdict(dict)
    for (scenario, method), group in sorted(grouped.items()):
        metric_names = sorted(group[0]["metrics"].keys())
        row: dict[str, Any] = {"n_seeds": len(group)}
        for metric in metric_names:
            vals = [float(record["metrics"][metric]) for record in group]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_stderr"] = stderr(vals)
            row[f"{metric}_min"] = float(np.min(vals))
            row[f"{metric}_max"] = float(np.max(vals))
        aggregate[scenario][method] = row
    return dict(aggregate)


def paired_vs_best_mlp(
    records: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build paired final-window MSE comparisons against best MLP."""
    scenario_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_records[record["scenario"]].append(record)
    paired: dict[str, dict[str, Any]] = {}
    for scenario, group in sorted(scenario_records.items()):
        best_mlp = min(
            ("mlp64", "mlp64_64"),
            key=lambda name: aggregate[scenario][name]["final_window_mse_mean"],
        )
        by_key = {(record["method"], int(record["seed"])): record for record in group}
        seeds = sorted(int(record["seed"]) for record in group if record["method"] == best_mlp)
        scenario_paired: dict[str, Any] = {"_best_mlp": best_mlp}
        methods = sorted({record["method"] for record in group})
        for method in methods:
            if method == best_mlp:
                continue
            diffs = []
            for seed in seeds:
                baseline = by_key[(best_mlp, seed)]["metrics"]["final_window_mse"]
                value = by_key[(method, seed)]["metrics"]["final_window_mse"]
                diffs.append(float(baseline - value))
            diff_arr = np.asarray(diffs, dtype=np.float64)
            sd = float(np.std(diff_arr, ddof=1)) if diff_arr.shape[0] > 1 else 0.0
            scenario_paired[method] = {
                "metric": "final_window_mse",
                "positive_means_method_beats_best_mlp": True,
                "mean_diff": float(np.mean(diff_arr)),
                "stderr": stderr(diff_arr),
                "wins": int(np.sum(diff_arr > 0.0)),
                "losses": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n_seeds": int(diff_arr.shape[0]),
                "cohens_d": float(np.mean(diff_arr) / sd) if sd > 0.0 else 0.0,
            }
        paired[scenario] = scenario_paired
    return paired


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested streams, baselines, and the three auction revisions."""
    t0 = time.time()
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    stream_names = [item.strip() for item in args.streams.split(",") if item.strip()]
    factories = synthetic_stream_factories()
    unknown = sorted(set(stream_names) - set(factories))
    if unknown:
        raise ValueError(f"unknown streams: {unknown}")

    for stream_name in stream_names:
        factory, feature_dim, n_tasks = factories[stream_name]
        scenario = f"synthetic_{stream_name}"
        print(f"\n=== {scenario}: seeds={args.n_seeds}, steps={args.steps} ===")
        for seed_offset in range(args.n_seeds):
            seed = args.seed + seed_offset
            observations, targets = collect_stream_arrays(
                factory(),
                args.steps,
                stable_key(seed, f"{stream_name}_stream"),
            )

            for method in BASELINES:
                learner = (
                    make_mlp(method, n_tasks)
                    if method.method_type == "mlp"
                    else make_upgd(method, n_tasks)
                )
                print(f"  seed={seed} stream={stream_name} method={method.name}")
                curve = run_jax_baseline(
                    learner,
                    stable_key(seed, f"{stream_name}_{method.name}"),
                    observations,
                    targets,
                )
                records.append(
                    {
                        "suite": "synthetic",
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": method.name,
                        "method_config": asdict(method),
                        "metrics": summarize_curve(curve, args.final_window),
                    }
                )

            auction_revisions = []
            for base_config in AUCTION_REVISIONS:
                config_kwargs = asdict(base_config)
                if args.n_slots is not None:
                    config_kwargs["n_slots"] = args.n_slots
                if args.candidate_count is not None:
                    config_kwargs["candidate_count"] = args.candidate_count
                if args.replacement_count is not None:
                    config_kwargs["replacement_count"] = args.replacement_count
                if args.replacement_interval is not None:
                    config_kwargs["replacement_interval"] = args.replacement_interval
                if args.auction_step_size is not None:
                    config_kwargs["step_size"] = args.auction_step_size
                if args.rent_multiplier is not None:
                    config_kwargs["rent"] = float(config_kwargs["rent"]) * args.rent_multiplier
                config_kwargs["include_tanh_combos"] = args.include_tanh_combos
                if args.cost_scale is not None:
                    config_kwargs["cost_scale"] = args.cost_scale
                if args.trained_tanh_candidates is not None:
                    config_kwargs["trained_tanh_candidates"] = args.trained_tanh_candidates
                if args.trained_tanh_steps is not None:
                    config_kwargs["trained_tanh_steps"] = args.trained_tanh_steps
                if args.trained_tanh_lr is not None:
                    config_kwargs["trained_tanh_lr"] = args.trained_tanh_lr
                if args.adapt_rtanh_lr is not None:
                    config_kwargs["adapt_rtanh_lr"] = args.adapt_rtanh_lr
                if args.trained_tanh_bundles is not None:
                    config_kwargs["trained_tanh_bundles"] = args.trained_tanh_bundles
                if args.trained_tanh_bundle_size is not None:
                    config_kwargs["trained_tanh_bundle_size"] = args.trained_tanh_bundle_size
                if args.trained_tanh_bundle_steps is not None:
                    config_kwargs["trained_tanh_bundle_steps"] = args.trained_tanh_bundle_steps
                if args.trained_tanh_bundle_lr is not None:
                    config_kwargs["trained_tanh_bundle_lr"] = args.trained_tanh_bundle_lr
                if args.bundle_diversity_cost is not None:
                    config_kwargs["bundle_diversity_cost"] = args.bundle_diversity_cost
                if args.nursery_size is not None:
                    config_kwargs["nursery_size"] = args.nursery_size
                if args.nursery_offer_count is not None:
                    config_kwargs["nursery_offer_count"] = args.nursery_offer_count
                if args.nursery_maturity is not None:
                    config_kwargs["nursery_maturity"] = args.nursery_maturity
                if args.nursery_lr is not None:
                    config_kwargs["nursery_lr"] = args.nursery_lr
                if args.nursery_readout_lr is not None:
                    config_kwargs["nursery_readout_lr"] = args.nursery_readout_lr
                if args.nursery_score_decay is not None:
                    config_kwargs["nursery_score_decay"] = args.nursery_score_decay
                if args.nursery_reset_interval is not None:
                    config_kwargs["nursery_reset_interval"] = args.nursery_reset_interval
                if args.nursery_reset_fraction is not None:
                    config_kwargs["nursery_reset_fraction"] = args.nursery_reset_fraction
                config_kwargs["meta_constructor"] = args.meta_constructor
                if args.meta_constructor_lr is not None:
                    config_kwargs["meta_constructor_lr"] = args.meta_constructor_lr
                if args.meta_constructor_temperature is not None:
                    config_kwargs["meta_constructor_temperature"] = (
                        args.meta_constructor_temperature
                    )
                if args.meta_constructor_floor is not None:
                    config_kwargs["meta_constructor_floor"] = args.meta_constructor_floor
                if args.meta_constructor_survival_weight is not None:
                    config_kwargs["meta_constructor_survival_weight"] = (
                        args.meta_constructor_survival_weight
                    )
                auction_revisions.append(AuctionConfig(**config_kwargs))

            for config in auction_revisions:
                print(f"  seed={seed} stream={stream_name} method={config.name}")
                curve, diag = run_auction(config, seed + 100_000, observations, targets)
                records.append(
                    {
                        "suite": "synthetic",
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": config.name,
                        "method_config": asdict(config),
                        "metrics": summarize_curve(curve, args.final_window),
                    }
                )
                diagnostics.append(
                    {
                        "scenario": scenario,
                        "stream": stream_name,
                        "seed": seed,
                        "method": config.name,
                        **diag,
                    }
                )

    aggregate = aggregate_records(records)
    paired = paired_vs_best_mlp(records, aggregate)
    winners = []
    for scenario, methods in paired.items():
        for method, row in methods.items():
            if method == "_best_mlp":
                continue
            if (
                method.startswith("auction_")
                and row["mean_diff"] > 0.0
                and row["wins"] > row["losses"]
            ):
                winners.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "mean_diff": row["mean_diff"],
                        "wins": row["wins"],
                        "n_seeds": row["n_seeds"],
                    }
                )
    return {
        "experiment": "exp03_market_auction",
        "hypothesis": (
            "A supervised market over constructed features can allocate a small "
            "feature budget better than a fair online MLP on at least one "
            "out-of-class Step 2 stream."
        ),
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "streams": stream_names,
            "auction_revisions": [asdict(config) for config in AUCTION_REVISIONS],
            "baselines": [asdict(config) for config in BASELINES],
        },
        "records": records,
        "aggregate": aggregate,
        "paired_vs_best_mlp": {"final_window_mse": paired},
        "auction_diagnostics": diagnostics,
        "auction_revisions_that_beat_mlp": winners,
        "deserves_scaling": bool(winners),
        "scaling_rule": (
            "Scale only if an auction revision has positive paired final-window "
            "MSE diff vs best MLP and wins a majority of paired seeds on at "
            "least one out-of-class stream."
        ),
        "wall_clock_s": time.time() - t0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a short local Markdown summary under the output directory."""
    lines = [
        "# Exp03 Market Auction",
        "",
        (
            f"Steps: {payload['config']['steps']}; "
            f"seeds: {payload['config']['n_seeds']}; "
            f"final window: {payload['config']['final_window']}; "
            f"wall clock: {payload['wall_clock_s']:.1f}s."
        ),
        "",
        "Positive paired diff means lower final-window MSE than the best fair MLP.",
        "",
    ]
    aggregate = payload["aggregate"]
    paired = payload["paired_vs_best_mlp"]["final_window_mse"]
    for scenario, per_method in aggregate.items():
        lines.append(f"## {scenario}")
        lines.append("")
        lines.append("| Method | Final-window MSE | Online mean MSE |")
        lines.append("|---|---:|---:|")
        for method in sorted(
            per_method, key=lambda name: per_method[name]["final_window_mse_mean"]
        ):
            row = per_method[method]
            lines.append(
                f"| `{method}` | "
                f"{row['final_window_mse_mean']:.6f} +/- {row['final_window_mse_stderr']:.6f} | "
                f"{row['online_mean_mse_mean']:.6f} +/- {row['online_mean_mse_stderr']:.6f} |"
            )
        lines.append("")
        lines.append(f"Best MLP: `{paired[scenario]['_best_mlp']}`")
        lines.append("")
        lines.append("| Method | best MLP - method | Wins | Losses | d |")
        lines.append("|---|---:|---:|---:|---:|")
        for method, row in paired[scenario].items():
            if method == "_best_mlp":
                continue
            lines.append(
                f"| `{method}` | {row['mean_diff']:+.6f} +/- {row['stderr']:.6f} | "
                f"{row['wins']}/{row['n_seeds']} | {row['losses']} | {row['cohens_d']:+.3f} |"
            )
        lines.append("")
    winners = payload["auction_revisions_that_beat_mlp"]
    lines.append("## Verdict")
    lines.append("")
    if winners:
        labels = ", ".join(f"`{w['method']}` on `{w['scenario']}`" for w in winners)
        lines.append(f"Auction revisions beating MLP by the decision rule: {labels}.")
    else:
        lines.append("No auction revision beat the best fair MLP by the decision rule.")
    lines.append(f"Deserves scaling: `{payload['deserves_scaling']}`.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=800)
    parser.add_argument(
        "--streams",
        type=str,
        default="polynomial,frequency",
        help="Comma-separated subset of polynomial,frequency,compositional.",
    )
    parser.add_argument("--n-slots", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--replacement-count", type=int, default=None)
    parser.add_argument("--replacement-interval", type=int, default=None)
    parser.add_argument("--auction-step-size", type=float, default=None)
    parser.add_argument("--rent-multiplier", type=float, default=None)
    parser.add_argument("--include-tanh-combos", action="store_true")
    parser.add_argument("--cost-scale", type=float, default=None)
    parser.add_argument("--trained-tanh-candidates", type=int, default=None)
    parser.add_argument("--trained-tanh-steps", type=int, default=None)
    parser.add_argument("--trained-tanh-lr", type=float, default=None)
    parser.add_argument("--adapt-rtanh-lr", type=float, default=None)
    parser.add_argument("--trained-tanh-bundles", type=int, default=None)
    parser.add_argument("--trained-tanh-bundle-size", type=int, default=None)
    parser.add_argument("--trained-tanh-bundle-steps", type=int, default=None)
    parser.add_argument("--trained-tanh-bundle-lr", type=float, default=None)
    parser.add_argument("--bundle-diversity-cost", type=float, default=None)
    parser.add_argument("--nursery-size", type=int, default=None)
    parser.add_argument("--nursery-offer-count", type=int, default=None)
    parser.add_argument("--nursery-maturity", type=int, default=None)
    parser.add_argument("--nursery-lr", type=float, default=None)
    parser.add_argument("--nursery-readout-lr", type=float, default=None)
    parser.add_argument("--nursery-score-decay", type=float, default=None)
    parser.add_argument("--nursery-reset-interval", type=int, default=None)
    parser.add_argument("--nursery-reset-fraction", type=float, default=None)
    parser.add_argument("--meta-constructor", action="store_true")
    parser.add_argument("--meta-constructor-lr", type=float, default=None)
    parser.add_argument("--meta-constructor-temperature", type=float, default=None)
    parser.add_argument("--meta-constructor-floor", type=float, default=None)
    parser.add_argument("--meta-constructor-survival-weight", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write artifacts."""
    args = parse_args()
    payload = run_suite(args)
    output_dir = args.output_dir
    results_path = output_dir / "results.json"
    summary_path = output_dir / "SUMMARY.md"
    write_json(results_path, payload)
    write_summary(summary_path, payload)
    print(f"\nwrote {results_path}")
    print(f"wrote {summary_path}")
    winners = payload["auction_revisions_that_beat_mlp"]
    print(f"deserves_scaling={payload['deserves_scaling']} auction_winners={winners}")


if __name__ == "__main__":
    main()

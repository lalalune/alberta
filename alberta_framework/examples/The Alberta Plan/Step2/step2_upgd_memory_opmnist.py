#!/usr/bin/env python3
"""Resumable single-learner OPMNIST benchmark for Step 2 UPGD-memory.

This runner is intentionally narrower than ``step2_published_stressors.py``:
it evaluates the packaged single UPGD-memory trace learner against fair MLP
baselines on chunked Online Permuted MNIST style streams.  The full Dohare et
al. OPMNIST protocol is 800 tasks x 60,000 examples, so the runner checkpoints
after every chunk and writes status/ETA sidecars.
"""

from __future__ import annotations

import argparse
import errno
import functools
import hashlib
import importlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step2_published_stressors as pub  # noqa: E402

from alberta_framework import (  # noqa: E402
    LMS,
    MultiHeadMLPLearner,
    ObGDBounding,
    Step2HybridConfig,
    Step2MemoryConfig,
    make_step2_hybrid_learner,
    make_step2_memory_learner,
)
from alberta_framework.core.upgd import UPGDLearner, UPGDState  # noqa: E402

N_CLASSES = 10
PRIMARY_METHOD = "step2_hybrid_memory_trace"
PRIMARY_RLS_CALIBRATED_METHOD = "step2_hybrid_memory_trace_rls_cal"
PRIMARY_SHARP_METHOD = "step2_hybrid_memory_trace_sharp"
PRIMARY_ADAPTIVE_SHARP_METHOD = "step2_hybrid_memory_trace_adaptive_sharp"
PRIMARY_DREAM_METHOD = "step2_hybrid_memory_trace_dream_surprise"
MLP_METHODS = ("mlp_h64", "mlp_h128")
SHARPENED_MLP_METHODS = ("mlp_h64_sharp", "mlp_h128_sharp")
CENTROID_METHOD = "centroid_hysteretic64_center_c030"
PROTO_MEMORY_METHODS = ("proto_mem_s20", "proto_mem_s32")
PROTO_MEMORY_METHOD = "proto_mem_s32"
SINGLE_UPGD_METHODS = (
    "upgd_structure_linear_h64",
    "upgd_structure_softmax_h64",
    "upgd_structure_linear_h128",
    "upgd_structure_softmax_h128",
)
SMOOTHED_SINGLE_UPGD_METHODS = (
    "upgd_structure_softmax_h64_smooth30",
    "upgd_structure_softmax_h64_smooth40",
    "upgd_structure_softmax_h128_smooth30",
    "upgd_structure_softmax_h128_smooth40",
)
BRIER_SINGLE_UPGD_METHODS = (
    "upgd_structure_brier_h64",
    "upgd_structure_brier_h128",
)
TEMPERATURE_SINGLE_UPGD_METHODS = (
    "upgd_structure_softmax_h256_temp4p0",
    "upgd_structure_softmax_h256_temp4p25",
    "upgd_structure_softmax_h256_temp4p5",
    "upgd_structure_softmax_h256_temp4p75",
    "upgd_structure_softmax_h256_temp5",
    "upgd_structure_softmax_h256_temp5p0",
)
DREAM_SINGLE_UPGD_METHODS = (
    "upgd_structure_softmax_h64_dream_surprise",
    "upgd_structure_softmax_h64_dream_progress",
    "upgd_structure_softmax_h64_dream_random",
    "upgd_structure_softmax_h64_dream_hybrid",
    "upgd_structure_softmax_h64_dream_surprise2",
    "upgd_structure_softmax_h64_dream_surprise_c256",
)
DELIGHT_METHODS = (
    "step2_hybrid_memory_trace_delight_gate15",
    "step2_hybrid_memory_trace_delight_gate30",
    "step2_hybrid_memory_trace_delight_gate50",
    "upgd_structure_softmax_h64_delight_gate30",
    "upgd_structure_softmax_h128_delight_gate30",
)
RLS_CALIBRATED_METHOD = "upgd_structure_softmax_h64_rls_cal"
DEFAULT_OUTPUT_DIR = Path("outputs/step2_upgd_memory_opmnist")
DEFAULT_NOTE_PATH = Path("docs/research/step2_upgd_memory_opmnist.md")
CHECKPOINT_VERSION = 1
CORE_SOLUTION_METRICS = (
    "online_mean_mse",
    "online_mean_accuracy",
    "final_window_mse",
    "final_window_accuracy",
    "test_mse",
    "test_accuracy",
)


@dataclass(frozen=True)
class MethodAccumulator:
    """Streaming metric accumulator for one method."""

    n_steps: int
    loss_sum: float
    correct_sum: float
    final_losses: np.ndarray
    final_correct: np.ndarray


class GenericUpdateResult(NamedTuple):
    """Minimal update result shape used by local candidate adapters."""

    state: Any
    predictions: jax.Array


@dataclass(frozen=True)
class PartialRunStop:
    """Metadata emitted when a bounded chunk-advance run stops early."""

    seed: int
    completed_steps: int
    requested_steps: int
    checkpoint_path: Path
    status_path: Path | None


class PartialRunCompleteError(RuntimeError):
    """Raised internally after a bounded chunk-advance checkpoint is written."""

    def __init__(self, stop: PartialRunStop) -> None:
        super().__init__(
            f"stopped seed {stop.seed} at "
            f"{stop.completed_steps}/{stop.requested_steps} steps"
        )
        self.stop = stop


class CentroidHystereticState(NamedTuple):
    """State for the one-centroid hysteretic residual candidate."""

    base_state: Any
    prototypes: jax.Array
    counts: jax.Array
    centroid_advantage: jax.Array
    centroid_gate: jax.Array


class AdaptiveSharpenedState(NamedTuple):
    """State for the causal raw-vs-sharpened deployment gate."""

    base_state: Any
    sharp_advantage: jax.Array
    sharp_gate: jax.Array


class RLSCalibratedState(NamedTuple):
    """State for prediction-space RLS calibration."""

    base_state: Any
    weights: jax.Array
    covariance: jax.Array
    loss_advantage: jax.Array
    calibration_gate: jax.Array


class DreamReplayState(NamedTuple):
    """State for causal high-surprise dream replay."""

    base_state: Any
    observations: jax.Array
    targets: jax.Array
    priorities: jax.Array
    valid: jax.Array
    write_index: jax.Array
    key: jax.Array
    dream_loss_ema: jax.Array
    dream_count: jax.Array
    step_count: jax.Array


class DelightGatedState(NamedTuple):
    """State for a supervised Kondo-style update gate."""

    base_state: Any
    price: jax.Array
    key: jax.Array
    delight_ema: jax.Array
    gate_rate_ema: jax.Array
    update_count: jax.Array
    step_count: jax.Array


class SmoothedSimplexLearner:
    """Apply a fixed uniform floor to simplex predictions.

    This is deliberately not a route, gate, or learned selector.  The base
    learner updates exactly as usual; the deployed prediction is always the
    same convex combination of the base simplex output and the uniform class
    distribution.
    """

    def __init__(self, base: Any, *, smoothing: float):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._base = base
        self._smoothing = float(smoothing)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        if hasattr(self._base, "config") and hasattr(self._base.config, "n_heads"):
            return int(self._base.config.n_heads)
        return int(self._base.n_heads)

    def _smooth(self, predictions: jax.Array) -> jax.Array:
        """Return fixed-floor simplex predictions."""
        smoothing = jnp.asarray(self._smoothing, dtype=jnp.float32)
        uniform = jnp.ones_like(predictions) / predictions.shape[0]
        return (1.0 - smoothing) * predictions + smoothing * uniform

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the base learner."""
        return self._base.init(feature_dim, key)

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict with fixed simplex smoothing."""
        return self._smooth(self._base.predict(state, observation))

    def update(
        self,
        state: Any,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update the base learner and return smoothed deployment predictions."""
        result = self._base.update(state, observation, targets)
        return GenericUpdateResult(
            state=result.state,
            predictions=self._smooth(result.predictions),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "SmoothedSimplexLearner",
            "smoothing": self._smoothing,
            "base": self._base.to_config(),
        }


class TemperatureScaledSimplexLearner:
    """Apply a fixed probability-temperature transform to simplex predictions."""

    def __init__(self, base: Any, *, temperature: float):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self._base = base
        self._temperature = float(temperature)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        if hasattr(self._base, "config") and hasattr(self._base.config, "n_heads"):
            return int(self._base.config.n_heads)
        return int(self._base.n_heads)

    def _scale(self, predictions: jax.Array) -> jax.Array:
        """Return fixed-temperature simplex predictions."""
        if self._temperature == 1.0:
            return predictions
        inverse_temperature = jnp.asarray(1.0 / self._temperature, dtype=jnp.float32)
        powered = jnp.power(jnp.maximum(predictions, 1e-8), inverse_temperature)
        return powered / jnp.sum(powered)

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the base learner."""
        return self._base.init(feature_dim, key)

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict with fixed temperature scaling."""
        return self._scale(self._base.predict(state, observation))

    def update(
        self,
        state: Any,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update the base learner and return temperature-scaled predictions."""
        result = self._base.update(state, observation, targets)
        return GenericUpdateResult(
            state=result.state,
            predictions=self._scale(result.predictions),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "TemperatureScaledSimplexLearner",
            "temperature": self._temperature,
            "base": self._base.to_config(),
        }


class RLSCalibratedLearner:
    """Prediction-space RLS calibrator around an online classifier.

    The calibrator is deliberately small: it sees only ``[1, base_prediction]``
    rather than raw pixels. That keeps the per-step cost roughly
    ``O(n_classes * (n_classes + 1)^2)`` instead of a pixel-space covariance.
    """

    def __init__(
        self,
        base: Any,
        *,
        ridge: float = 10.0,
        forgetting: float = 0.995,
        utility_decay: float = 0.99,
        utility_off_threshold: float = 0.0001,
        utility_on_threshold: float = 0.0002,
        mix_scale: float = 1.0,
        prediction_mode: str = "simplex",
        identity_init: bool = True,
        init_requires_feature_dim: bool = True,
    ):
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        if mix_scale < 0.0:
            raise ValueError("mix_scale must be non-negative")
        if prediction_mode not in {"simplex", "softmax"}:
            raise ValueError("prediction_mode must be 'simplex' or 'softmax'")
        self._base = base
        self._ridge = float(ridge)
        self._forgetting = float(forgetting)
        self._utility_decay = float(utility_decay)
        self._utility_off_threshold = float(utility_off_threshold)
        self._utility_on_threshold = float(utility_on_threshold)
        self._mix_scale = float(mix_scale)
        self._prediction_mode = prediction_mode
        self._identity_init = bool(identity_init)
        self._init_requires_feature_dim = bool(init_requires_feature_dim)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        if hasattr(self._base, "config") and hasattr(self._base.config, "n_heads"):
            return int(self._base.config.n_heads)
        return int(self._base.n_heads)

    @property
    def calibration_dim(self) -> int:
        """Number of RLS calibration features."""
        return self.n_heads + 1

    def init(self, feature_dim: int, key: jax.Array) -> RLSCalibratedState:
        """Initialize base learner and tiny per-class RLS calibrator."""
        if self._init_requires_feature_dim:
            base_state = self._base.init(feature_dim, key)
        else:
            base_state = self._base.init(key)
        cal_dim = self.calibration_dim
        weights = jnp.zeros((self.n_heads, cal_dim), dtype=jnp.float32)
        if self._identity_init:
            weights = weights.at[:, 1:].set(jnp.eye(self.n_heads, dtype=jnp.float32))
        return RLSCalibratedState(
            base_state=base_state,
            weights=weights,
            covariance=jnp.tile(
                (jnp.eye(cal_dim, dtype=jnp.float32) / self._ridge)[None, :, :],
                (self.n_heads, 1, 1),
            ),
            loss_advantage=jnp.array(0.0, dtype=jnp.float32),
            calibration_gate=jnp.array(0.0, dtype=jnp.float32),
        )

    def _features(self, base_prediction: jax.Array) -> jax.Array:
        """Return bias-augmented base prediction features."""
        return jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.float32), base_prediction],
            axis=0,
        )

    def _calibrated_prediction(
        self,
        weights: jax.Array,
        features: jax.Array,
    ) -> jax.Array:
        """Return normalized calibrated class probabilities."""
        raw = weights @ features
        if self._prediction_mode == "softmax":
            return jax.nn.softmax(raw)
        clipped = jnp.maximum(raw, 0.0)
        return clipped / jnp.maximum(jnp.sum(clipped), 1e-12)

    def _mix(
        self,
        base_prediction: jax.Array,
        calibrated_prediction: jax.Array,
        gate: jax.Array,
    ) -> jax.Array:
        """Mix base and calibrated predictions with a utility gate."""
        mix = jnp.clip(gate * jnp.asarray(self._mix_scale, dtype=jnp.float32), 0.0, 1.0)
        mixed = (1.0 - mix) * base_prediction + mix * calibrated_prediction
        return mixed / jnp.maximum(jnp.sum(mixed), 1e-12)

    def predict(
        self,
        state: RLSCalibratedState,
        observation: jax.Array,
    ) -> jax.Array:
        """Predict with current base-vs-calibrated utility gate."""
        base_prediction = self._base.predict(state.base_state, observation)
        features = self._features(base_prediction)
        calibrated = self._calibrated_prediction(state.weights, features)
        return self._mix(base_prediction, calibrated, state.calibration_gate)

    def _update_rls(
        self,
        weights: jax.Array,
        covariance: jax.Array,
        features: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Update one RLS scalar target model per output class."""
        covariance_features = jnp.einsum("hij,j->hi", covariance, features)
        forgetting = jnp.asarray(self._forgetting, dtype=jnp.float32)
        denominator = forgetting + jnp.einsum("j,hj->h", features, covariance_features)
        gain = covariance_features / denominator[:, None]
        prediction = jnp.einsum("hj,j->h", weights, features)
        error = targets - prediction
        next_weights = weights + gain * error[:, None]
        next_covariance = (
            covariance - jnp.einsum("hi,hj->hij", gain, covariance_features)
        ) / forgetting
        return next_weights, next_covariance

    def update(
        self,
        state: RLSCalibratedState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update the base learner and calibrator, returning gated predictions."""
        result = self._base.update(state.base_state, observation, targets)
        base_prediction = result.predictions
        features = self._features(base_prediction)
        calibrated = self._calibrated_prediction(state.weights, features)
        prediction = self._mix(
            base_prediction,
            calibrated,
            state.calibration_gate,
        )
        base_loss = active_mse(base_prediction, targets)
        calibrated_loss = active_mse(calibrated, targets)
        next_advantage = (
            jnp.asarray(self._utility_decay, dtype=jnp.float32) * state.loss_advantage
            + (1.0 - jnp.asarray(self._utility_decay, dtype=jnp.float32))
            * (base_loss - calibrated_loss)
        )
        next_gate = hysteretic_utility_gate(
            state.calibration_gate,
            next_advantage,
            jnp.asarray(self._utility_off_threshold, dtype=jnp.float32),
            jnp.asarray(self._utility_on_threshold, dtype=jnp.float32),
        )
        next_weights, next_covariance = self._update_rls(
            state.weights,
            state.covariance,
            features,
            targets,
        )
        return GenericUpdateResult(
            state=RLSCalibratedState(
                base_state=result.state,
                weights=next_weights,
                covariance=next_covariance,
                loss_advantage=next_advantage,
                calibration_gate=next_gate,
            ),
            predictions=prediction,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "RLSCalibratedLearner",
            "ridge": self._ridge,
            "forgetting": self._forgetting,
            "utility_decay": self._utility_decay,
            "utility_off_threshold": self._utility_off_threshold,
            "utility_on_threshold": self._utility_on_threshold,
            "mix_scale": self._mix_scale,
            "prediction_mode": self._prediction_mode,
            "identity_init": self._identity_init,
            "init_requires_feature_dim": self._init_requires_feature_dim,
            "base": self._base.to_config(),
        }


class DreamReplayLearner:
    """Causal dream replay around an online classifier.

    The dreamer stores already-seen examples, prioritizes them from current
    prediction loss and learning progress, and applies extra updates only after
    the current real prediction has been scored. It is therefore a conservative
    supervised analogue of dreaming: no target can affect its own deployed
    prediction, and replay pressure decays as replayed examples become less
    surprising.
    """

    def __init__(
        self,
        base: Any,
        *,
        capacity: int = 128,
        dreams_per_step: int = 1,
        warmup_steps: int = 64,
        mode: str = "surprise",
        learning_progress_weight: float = 1.0,
        dream_loss_decay: float = 0.99,
        priority_floor: float = 1e-6,
        init_requires_feature_dim: bool = True,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dreams_per_step < 0:
            raise ValueError("dreams_per_step must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if mode not in {"surprise", "progress", "hybrid", "random"}:
            raise ValueError("mode must be surprise, progress, hybrid, or random")
        if learning_progress_weight < 0.0:
            raise ValueError("learning_progress_weight must be non-negative")
        if not 0.0 <= dream_loss_decay < 1.0:
            raise ValueError("dream_loss_decay must be in [0, 1)")
        if priority_floor < 0.0:
            raise ValueError("priority_floor must be non-negative")
        self._base = base
        self._capacity = int(capacity)
        self._dreams_per_step = int(dreams_per_step)
        self._warmup_steps = int(warmup_steps)
        self._mode = mode
        self._learning_progress_weight = float(learning_progress_weight)
        self._dream_loss_decay = float(dream_loss_decay)
        self._priority_floor = float(priority_floor)
        self._init_requires_feature_dim = bool(init_requires_feature_dim)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        if hasattr(self._base, "config") and hasattr(self._base.config, "n_heads"):
            return int(self._base.config.n_heads)
        return int(self._base.n_heads)

    def init(self, feature_dim: int, key: jax.Array) -> DreamReplayState:
        """Initialize the wrapped learner and fixed-size dream buffer."""
        base_key, dream_key = jr.split(key)
        if self._init_requires_feature_dim:
            base_state = self._base.init(feature_dim, base_key)
        else:
            base_state = self._base.init(base_key)
        return DreamReplayState(
            base_state=base_state,
            observations=jnp.zeros(
                (self._capacity, feature_dim),
                dtype=jnp.float32,
            ),
            targets=jnp.zeros((self._capacity, self.n_heads), dtype=jnp.float32),
            priorities=jnp.zeros((self._capacity,), dtype=jnp.float32),
            valid=jnp.zeros((self._capacity,), dtype=bool),
            write_index=jnp.array(0, dtype=jnp.int32),
            key=dream_key,
            dream_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            dream_count=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def predict(self, state: DreamReplayState, observation: jax.Array) -> jax.Array:
        """Predict with the wrapped learner only."""
        return cast(jax.Array, self._base.predict(state.base_state, observation))

    def _priority(
        self,
        pre_loss: jax.Array,
        post_loss: jax.Array,
    ) -> jax.Array:
        """Compute dream priority from surprise and learning progress."""
        progress = jnp.maximum(pre_loss - post_loss, 0.0)
        floor = jnp.asarray(self._priority_floor, dtype=jnp.float32)
        if self._mode == "surprise":
            return jnp.maximum(pre_loss, floor)
        if self._mode == "progress":
            return jnp.maximum(progress, floor)
        if self._mode == "random":
            return jnp.array(1.0, dtype=jnp.float32)
        return jnp.maximum(
            pre_loss
            + jnp.asarray(self._learning_progress_weight, dtype=jnp.float32)
            * progress,
            floor,
        )

    def _sample_index(
        self,
        state: DreamReplayState,
        key: jax.Array,
    ) -> jax.Array:
        """Select one dream buffer index."""
        if self._mode == "random":
            sample_size = jnp.minimum(state.step_count + 1, self._capacity)
            raw_idx = jr.randint(key, (), 0, self._capacity).astype(jnp.int32)
            return raw_idx % jnp.maximum(sample_size, 1)
        scores = jnp.where(state.valid, state.priorities, -jnp.inf)
        return jnp.argmax(scores).astype(jnp.int32)

    def update(
        self,
        state: DreamReplayState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update from one real example and optional after-score dream replay."""
        real_result = self._base.update(state.base_state, observation, targets)
        real_prediction = real_result.predictions
        real_pre_loss = active_mse(real_prediction, targets)
        real_post_prediction = self._base.predict(real_result.state, observation)
        real_post_loss = active_mse(real_post_prediction, targets)
        priority = self._priority(real_pre_loss, real_post_loss)

        next_observations = state.observations.at[state.write_index].set(observation)
        next_targets = state.targets.at[state.write_index].set(targets)
        next_priorities = state.priorities.at[state.write_index].set(priority)
        next_valid = state.valid.at[state.write_index].set(True)
        next_index = ((state.write_index + 1) % self._capacity).astype(jnp.int32)

        dream_state = DreamReplayState(
            base_state=real_result.state,
            observations=next_observations,
            targets=next_targets,
            priorities=next_priorities,
            valid=next_valid,
                write_index=next_index,
            key=state.key,
            dream_loss_ema=state.dream_loss_ema,
            dream_count=state.dream_count,
            step_count=state.step_count + 1,
        )

        can_dream = dream_state.step_count >= self._warmup_steps
        for _ in range(self._dreams_per_step):
            dream_key, next_key = jr.split(dream_state.key)
            dream_idx = self._sample_index(dream_state, dream_key)
            dream_obs = dream_state.observations[dream_idx]
            dream_target = dream_state.targets[dream_idx]
            dream_result = self._base.update(
                dream_state.base_state,
                dream_obs,
                dream_target,
            )
            dream_prediction = self._base.predict(dream_result.state, dream_obs)
            dream_loss = active_mse(dream_prediction, dream_target)
            next_base_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(can_dream, new, old),
                dream_result.state,
                dream_state.base_state,
            )
            next_priorities = dream_state.priorities.at[dream_idx].set(
                jnp.where(
                    can_dream,
                    jnp.maximum(dream_loss, self._priority_floor),
                    dream_state.priorities[dream_idx],
                )
            )
            next_loss_ema = jnp.where(
                dream_state.dream_count == 0,
                dream_loss,
                jnp.asarray(self._dream_loss_decay, dtype=jnp.float32)
                * dream_state.dream_loss_ema
                + (1.0 - jnp.asarray(self._dream_loss_decay, dtype=jnp.float32))
                * dream_loss,
            )
            dream_state = DreamReplayState(
                base_state=next_base_state,
                observations=dream_state.observations,
                targets=dream_state.targets,
                priorities=next_priorities,
                valid=dream_state.valid,
                write_index=dream_state.write_index,
                key=next_key,
                dream_loss_ema=jnp.where(
                    can_dream,
                    next_loss_ema,
                    dream_state.dream_loss_ema,
                ),
                dream_count=(
                    dream_state.dream_count
                    + jnp.where(
                        can_dream,
                        jnp.array(1, dtype=jnp.int32),
                        jnp.array(0, dtype=jnp.int32),
                    )
                ),
                step_count=dream_state.step_count,
            )

        return GenericUpdateResult(
            state=dream_state,
            predictions=real_prediction,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "DreamReplayLearner",
            "capacity": self._capacity,
            "dreams_per_step": self._dreams_per_step,
            "warmup_steps": self._warmup_steps,
            "mode": self._mode,
            "learning_progress_weight": self._learning_progress_weight,
            "dream_loss_decay": self._dream_loss_decay,
            "priority_floor": self._priority_floor,
            "init_requires_feature_dim": self._init_requires_feature_dim,
            "base": self._base.to_config(),
        }


class DelightGatedLearner:
    """Kondo-style update gate for supervised online classification.

    The true label is treated as the sampled "good" action. The gate uses
    ``delight = (1 - p_true) * -log(p_true)``: high when the correct label is
    valuable and surprising, low when the learner already predicts it well.
    This is a supervised analogue of Delightful Policy Gradient/Kondo gating,
    not a replacement for policy-gradient control.
    """

    def __init__(
        self,
        base: Any,
        *,
        target_rate: float = 0.30,
        temperature: float = 0.05,
        price_step_size: float = 0.01,
        diagnostic_decay: float = 0.99,
        init_requires_feature_dim: bool = True,
    ):
        if not 0.0 < target_rate <= 1.0:
            raise ValueError("target_rate must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if price_step_size < 0.0:
            raise ValueError("price_step_size must be non-negative")
        if not 0.0 <= diagnostic_decay < 1.0:
            raise ValueError("diagnostic_decay must be in [0, 1)")
        self._base = base
        self._target_rate = float(target_rate)
        self._temperature = float(temperature)
        self._price_step_size = float(price_step_size)
        self._diagnostic_decay = float(diagnostic_decay)
        self._init_requires_feature_dim = bool(init_requires_feature_dim)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        if hasattr(self._base, "config") and hasattr(self._base.config, "n_heads"):
            return int(self._base.config.n_heads)
        return int(self._base.n_heads)

    def init(self, feature_dim: int, key: jax.Array) -> DelightGatedState:
        """Initialize wrapped learner and gate state."""
        base_key, gate_key = jr.split(key)
        if self._init_requires_feature_dim:
            base_state = self._base.init(feature_dim, base_key)
        else:
            base_state = self._base.init(base_key)
        return DelightGatedState(
            base_state=base_state,
            price=jnp.array(0.0, dtype=jnp.float32),
            key=gate_key,
            delight_ema=jnp.array(0.0, dtype=jnp.float32),
            gate_rate_ema=jnp.array(0.0, dtype=jnp.float32),
            update_count=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def predict(self, state: DelightGatedState, observation: jax.Array) -> jax.Array:
        """Predict with the wrapped learner."""
        return cast(jax.Array, self._base.predict(state.base_state, observation))

    def update(
        self,
        state: DelightGatedState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update only when the current example clears the delight gate."""
        prediction = self._base.predict(state.base_state, observation)
        label = jnp.argmax(targets).astype(jnp.int32)
        true_probability = jnp.clip(prediction[label], 1.0e-6, 1.0)
        advantage = 1.0 - true_probability
        surprisal = -jnp.log(true_probability)
        delight = advantage * surprisal
        gate_probability = jax.nn.sigmoid(
            (delight - state.price) / jnp.asarray(self._temperature, dtype=jnp.float32)
        )
        gate_key, next_key = jr.split(state.key)
        gate = jr.uniform(gate_key, ()) < gate_probability
        next_base_state = jax.lax.cond(
            gate,
            lambda _: self._base.update(state.base_state, observation, targets).state,
            lambda _: state.base_state,
            operand=None,
        )
        decay = jnp.asarray(self._diagnostic_decay, dtype=jnp.float32)
        first = state.step_count == 0
        gate_float = gate.astype(jnp.float32)
        next_delight_ema = jnp.where(
            first,
            delight,
            decay * state.delight_ema + (1.0 - decay) * delight,
        )
        next_gate_rate_ema = jnp.where(
            first,
            gate_float,
            decay * state.gate_rate_ema + (1.0 - decay) * gate_float,
        )
        next_price = jnp.maximum(
            state.price
            + jnp.asarray(self._price_step_size, dtype=jnp.float32)
            * (gate_probability - jnp.asarray(self._target_rate, dtype=jnp.float32)),
            0.0,
        )
        return GenericUpdateResult(
            state=DelightGatedState(
                base_state=next_base_state,
                price=next_price,
                key=next_key,
                delight_ema=next_delight_ema,
                gate_rate_ema=next_gate_rate_ema,
                update_count=state.update_count + gate.astype(jnp.int32),
                step_count=state.step_count + 1,
            ),
            predictions=prediction,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "DelightGatedLearner",
            "target_rate": self._target_rate,
            "temperature": self._temperature,
            "price_step_size": self._price_step_size,
            "diagnostic_decay": self._diagnostic_decay,
            "init_requires_feature_dim": self._init_requires_feature_dim,
            "base": self._base.to_config(),
        }


class FixedPrototypeMemoryCandidate:
    """Adapt standalone Step 2 prototype memory to this runner's learner API."""

    def __init__(self, feature_dim: int, slots_per_class: int = 32):
        self._feature_dim = int(feature_dim)
        self._base = make_step2_memory_learner(
            Step2MemoryConfig(
                feature_dim=feature_dim,
                n_classes=N_CLASSES,
                slots_per_class=slots_per_class,
                update_rate=0.3,
                novelty_threshold=0.08,
                bandwidth=0.01,
            )
        )

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self._base.config.n_classes

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize after checking the fixed image feature dimension."""
        del key
        if int(feature_dim) != self._feature_dim:
            raise ValueError(
                f"expected feature_dim {self._feature_dim}, got {feature_dim}"
            )
        return self._base.init()

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict one one-hot class vector."""
        return cast(jax.Array, self._base.predict(state, observation))

    def update(self, state: Any, observation: jax.Array, target: jax.Array) -> Any:
        """Run one causal prototype-memory update."""
        return self._base.update(state, observation, target)

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapped prototype-memory config."""
        payload = self._base.to_config()
        payload["fixed_feature_dim"] = self._feature_dim
        return payload


def init_accumulator(final_window: int) -> MethodAccumulator:
    """Create an empty accumulator with a fixed final-window capacity."""
    return MethodAccumulator(
        n_steps=0,
        loss_sum=0.0,
        correct_sum=0.0,
        final_losses=np.zeros(final_window, dtype=np.float64),
        final_correct=np.zeros(final_window, dtype=np.float64),
    )


def update_accumulator(
    accumulator: MethodAccumulator,
    metrics: np.ndarray,
    final_window: int,
) -> MethodAccumulator:
    """Add one metrics chunk to an accumulator."""
    losses = np.asarray(metrics[:, 0], dtype=np.float64)
    correct = np.asarray(metrics[:, 1], dtype=np.float64)
    previous_count = min(accumulator.n_steps, final_window)
    loss_tail = np.concatenate([accumulator.final_losses[:previous_count], losses])
    correct_tail = np.concatenate([accumulator.final_correct[:previous_count], correct])
    new_losses = np.zeros(final_window, dtype=np.float64)
    new_correct = np.zeros(final_window, dtype=np.float64)
    keep_losses = loss_tail[-final_window:]
    keep_correct = correct_tail[-final_window:]
    new_losses[: keep_losses.shape[0]] = keep_losses
    new_correct[: keep_correct.shape[0]] = keep_correct
    return MethodAccumulator(
        n_steps=int(accumulator.n_steps + losses.shape[0]),
        loss_sum=float(accumulator.loss_sum + np.sum(losses)),
        correct_sum=float(accumulator.correct_sum + np.sum(correct)),
        final_losses=new_losses,
        final_correct=new_correct,
    )


def summarize_accumulator(accumulator: MethodAccumulator) -> dict[str, float]:
    """Return online and final-window metrics for one method."""
    if accumulator.n_steps <= 0:
        raise RuntimeError("cannot summarize an empty accumulator")
    final_count = min(accumulator.n_steps, accumulator.final_losses.shape[0])
    return {
        "online_mean_mse": float(accumulator.loss_sum / accumulator.n_steps),
        "online_mean_accuracy": float(accumulator.correct_sum / accumulator.n_steps),
        "final_window_mse": float(np.mean(accumulator.final_losses[:final_count])),
        "final_window_accuracy": float(np.mean(accumulator.final_correct[:final_count])),
    }


def make_mlp(method: str) -> MultiHeadMLPLearner:
    """Create one fair MLP comparator."""
    hidden_sizes_by_method = {
        "mlp_h64": (64,),
        "mlp_h128": (128,),
        "mlp_h64_sharp": (64,),
        "mlp_h128_sharp": (128,),
    }
    return MultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=hidden_sizes_by_method[method],
        optimizer=LMS(step_size=0.03),
        bounder=ObGDBounding(kappa=0.5),
        sparsity=0.5,
        use_layer_norm=True,
    )


def sharpen_if_confident(
    predictions: jax.Array,
    threshold: jax.Array | float = 0.10,
    blend: jax.Array | float = 1.0,
) -> jax.Array:
    """Blend confident class predictions toward their top-1 one-hot vector."""
    top_values, top_indices = jax.lax.top_k(predictions, 2)
    margin = top_values[0] - top_values[1]
    one_hot = jax.nn.one_hot(top_indices[0], predictions.shape[0])
    blend_j = jnp.asarray(blend, dtype=jnp.float32)
    sharpened = (1.0 - blend_j) * predictions + blend_j * one_hot
    return jnp.where(
        (blend_j > 0.0) & (margin >= jnp.asarray(threshold, dtype=jnp.float32)),
        sharpened,
        predictions,
    )


class SharpenedLearner:
    """Score and deploy an existing learner with causal confidence sharpening."""

    def __init__(
        self,
        base: Any,
        *,
        threshold: float = 0.10,
        blend: float = 1.0,
        init_requires_feature_dim: bool = True,
    ):
        self._base = base
        self._threshold = float(threshold)
        self._blend = float(blend)
        self._init_requires_feature_dim = bool(init_requires_feature_dim)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return int(self._base.n_heads)

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the wrapped learner."""
        if not self._init_requires_feature_dim:
            return self._base.init(key)
        return self._base.init(feature_dim, key)

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict with sharpened deployment readout."""
        return sharpen_if_confident(
            self._base.predict(state, observation),
            self._threshold,
            self._blend,
        )

    def update(
        self,
        state: Any,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update the base learner and score the sharpened prediction."""
        result = self._base.update(state, observation, targets)
        return GenericUpdateResult(
            state=result.state,
            predictions=sharpen_if_confident(
                result.predictions,
                self._threshold,
                self._blend,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "SharpenedLearner",
            "threshold": self._threshold,
            "blend": self._blend,
            "init_requires_feature_dim": self._init_requires_feature_dim,
            "base": self._base.to_config(),
        }


class AdaptiveSharpenedLearner:
    """Causal utility gate between raw and sharpened deployment readouts."""

    def __init__(
        self,
        base: Any,
        *,
        threshold: float = 0.10,
        blend: float = 1.0,
        utility_decay: float = 0.99,
        utility_off_threshold: float = 0.0001,
        utility_on_threshold: float = 0.0002,
        init_requires_feature_dim: bool = True,
    ):
        self._base = base
        self._threshold = float(threshold)
        self._blend = float(blend)
        self._utility_decay = float(utility_decay)
        self._utility_off_threshold = float(utility_off_threshold)
        self._utility_on_threshold = float(utility_on_threshold)
        self._init_requires_feature_dim = bool(init_requires_feature_dim)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return int(self._base.n_heads)

    def init(self, feature_dim: int, key: jax.Array) -> AdaptiveSharpenedState:
        """Initialize the wrapped learner and readout gate."""
        if self._init_requires_feature_dim:
            base_state = self._base.init(feature_dim, key)
        else:
            base_state = self._base.init(key)
        return AdaptiveSharpenedState(
            base_state=base_state,
            sharp_advantage=jnp.array(0.0, dtype=jnp.float32),
            sharp_gate=jnp.array(0.0, dtype=jnp.float32),
        )

    def _mix(self, raw: jax.Array, sharp: jax.Array, gate: jax.Array) -> jax.Array:
        """Mix raw and sharpened predictions according to the scalar gate."""
        return (1.0 - gate) * raw + gate * sharp

    def predict(
        self,
        state: AdaptiveSharpenedState,
        observation: jax.Array,
    ) -> jax.Array:
        """Predict with the current raw-vs-sharpened deployment gate."""
        raw = self._base.predict(state.base_state, observation)
        sharp = sharpen_if_confident(raw, self._threshold, self._blend)
        return self._mix(raw, sharp, state.sharp_gate)

    def update(
        self,
        state: AdaptiveSharpenedState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update the base learner and causal readout utility."""
        result = self._base.update(state.base_state, observation, targets)
        raw = result.predictions
        sharp = sharpen_if_confident(raw, self._threshold, self._blend)
        next_gate = hysteretic_utility_gate(
            state.sharp_gate,
            state.sharp_advantage,
            jnp.asarray(self._utility_off_threshold, dtype=jnp.float32),
            jnp.asarray(self._utility_on_threshold, dtype=jnp.float32),
        )
        prediction = self._mix(raw, sharp, next_gate)
        raw_loss = active_mse(raw, targets)
        sharp_loss = active_mse(sharp, targets)
        next_advantage = (
            jnp.asarray(self._utility_decay, dtype=jnp.float32)
            * state.sharp_advantage
            + (1.0 - jnp.asarray(self._utility_decay, dtype=jnp.float32))
            * (raw_loss - sharp_loss)
        )
        return GenericUpdateResult(
            state=AdaptiveSharpenedState(
                base_state=result.state,
                sharp_advantage=next_advantage,
                sharp_gate=next_gate,
            ),
            predictions=prediction,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper."""
        return {
            "type": "AdaptiveSharpenedLearner",
            "threshold": self._threshold,
            "blend": self._blend,
            "utility_decay": self._utility_decay,
            "utility_off_threshold": self._utility_off_threshold,
            "utility_on_threshold": self._utility_on_threshold,
            "init_requires_feature_dim": self._init_requires_feature_dim,
            "base": self._base.to_config(),
        }


def prototype_probs(
    prototypes: jax.Array,
    counts: jax.Array,
    observation: jax.Array,
    temperature: jax.Array,
) -> jax.Array:
    """Return softmaxed cosine similarity to causal class centroids."""
    obs = observation / (jnp.linalg.norm(observation) + 1e-8)
    proto_norms = jnp.linalg.norm(prototypes, axis=1)
    normalized_prototypes = prototypes / (proto_norms[:, None] + 1e-8)
    cosine = normalized_prototypes @ obs
    seen = counts > 0.0
    scores = jnp.where(seen, cosine / temperature, -20.0)
    return jax.nn.softmax(scores)


def update_prototype(
    prototypes: jax.Array,
    counts: jax.Array,
    observation: jax.Array,
    label: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """EMA-update only the observed class centroid."""
    obs = observation / (jnp.linalg.norm(observation) + 1e-8)
    old = prototypes[label]
    new = (1.0 - alpha) * old + alpha * obs
    new = new / (jnp.linalg.norm(new) + 1e-8)
    return prototypes.at[label].set(new), counts.at[label].add(1.0)


def active_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Mean squared error over a dense class target."""
    return jnp.mean(jnp.square(prediction - target))


def hysteretic_utility_gate(
    previous_gate: jax.Array,
    advantage: jax.Array,
    off_threshold: jax.Array,
    on_threshold: jax.Array,
) -> jax.Array:
    """Update a binary utility gate with hysteresis."""
    was_on = previous_gate > 0.5
    stay_on = advantage >= -off_threshold
    turn_on = advantage >= on_threshold
    return jnp.where(was_on, stay_on, turn_on).astype(jnp.float32)


class CentroidHystereticLearner:
    """One-centroid-per-class residual readout with hysteretic utility."""

    def __init__(
        self,
        base: Any,
        *,
        n_heads: int = N_CLASSES,
        alpha: float = 0.05,
        temperature: float = 0.05,
        centroid_mix: float = 0.30,
        utility_decay: float = 0.99,
        utility_off_threshold: float = 0.005,
        utility_on_threshold: float = 0.001,
        sharpen_threshold: float = 0.10,
        sharpen_blend: float = 1.0,
    ):
        self._base = base
        self._n_heads = int(n_heads)
        self._alpha = float(alpha)
        self._temperature = float(temperature)
        self._centroid_mix = float(centroid_mix)
        self._utility_decay = float(utility_decay)
        self._utility_off_threshold = float(utility_off_threshold)
        self._utility_on_threshold = float(utility_on_threshold)
        self._sharpen_threshold = float(sharpen_threshold)
        self._sharpen_blend = float(sharpen_blend)

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self._n_heads

    def init(self, feature_dim: int, key: jax.Array) -> CentroidHystereticState:
        """Initialize base learner and centroid state."""
        return CentroidHystereticState(
            base_state=self._base.init(feature_dim, key),
            prototypes=jnp.zeros((self._n_heads, feature_dim), dtype=jnp.float32),
            counts=jnp.zeros((self._n_heads,), dtype=jnp.float32),
            centroid_advantage=jnp.array(0.0, dtype=jnp.float32),
            centroid_gate=jnp.array(1.0, dtype=jnp.float32),
        )

    def _predict_from_parts(
        self,
        base_prediction: jax.Array,
        centroid_prediction: jax.Array,
        centroid_gate: jax.Array,
    ) -> jax.Array:
        gated_mix = jnp.asarray(self._centroid_mix, dtype=jnp.float32) * centroid_gate
        mixed = (1.0 - gated_mix) * base_prediction + gated_mix * centroid_prediction
        return sharpen_if_confident(
            mixed,
            self._sharpen_threshold,
            self._sharpen_blend,
        )

    def predict(
        self,
        state: CentroidHystereticState,
        observation: jax.Array,
    ) -> jax.Array:
        """Predict from the base learner plus final centroid readout."""
        base_prediction = self._base.predict(state.base_state, observation)
        centroid_prediction = prototype_probs(
            state.prototypes,
            state.counts,
            observation,
            jnp.asarray(self._temperature, dtype=jnp.float32),
        )
        return self._predict_from_parts(
            base_prediction,
            centroid_prediction,
            state.centroid_gate,
        )

    def update(
        self,
        state: CentroidHystereticState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update base learner, centroid utility, and centroid state."""
        temperature = jnp.asarray(self._temperature, dtype=jnp.float32)
        base_prediction = self._base.predict(state.base_state, observation)
        centroid_prediction = prototype_probs(
            state.prototypes,
            state.counts,
            observation,
            temperature,
        )
        full_centroid_prediction = (
            (1.0 - self._centroid_mix) * base_prediction
            + self._centroid_mix * centroid_prediction
        )
        next_gate = hysteretic_utility_gate(
            state.centroid_gate,
            state.centroid_advantage,
            jnp.asarray(self._utility_off_threshold, dtype=jnp.float32),
            jnp.asarray(self._utility_on_threshold, dtype=jnp.float32),
        )
        prediction = self._predict_from_parts(
            base_prediction,
            centroid_prediction,
            next_gate,
        )
        base_loss = active_mse(
            sharpen_if_confident(
                base_prediction,
                self._sharpen_threshold,
                self._sharpen_blend,
            ),
            targets,
        )
        centroid_loss = active_mse(
            sharpen_if_confident(
                full_centroid_prediction,
                self._sharpen_threshold,
                self._sharpen_blend,
            ),
            targets,
        )
        next_advantage = (
            jnp.asarray(self._utility_decay, dtype=jnp.float32)
            * state.centroid_advantage
            + (1.0 - jnp.asarray(self._utility_decay, dtype=jnp.float32))
            * (base_loss - centroid_loss)
        )
        result = self._base.update(state.base_state, observation, targets)
        label = jnp.argmax(targets).astype(jnp.int32)
        new_prototypes, new_counts = update_prototype(
            state.prototypes,
            state.counts,
            observation,
            label,
            jnp.asarray(self._alpha, dtype=jnp.float32),
        )
        return GenericUpdateResult(
            state=CentroidHystereticState(
                base_state=result.state,
                prototypes=new_prototypes,
                counts=new_counts,
                centroid_advantage=next_advantage,
                centroid_gate=next_gate,
            ),
            predictions=prediction,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the centroid learner."""
        return {
            "type": "CentroidHystereticLearner",
            "base": self._base.to_config(),
            "alpha": self._alpha,
            "temperature": self._temperature,
            "centroid_mix": self._centroid_mix,
            "utility_decay": self._utility_decay,
            "utility_off_threshold": self._utility_off_threshold,
            "utility_on_threshold": self._utility_on_threshold,
            "sharpen_threshold": self._sharpen_threshold,
            "sharpen_blend": self._sharpen_blend,
        }


def make_centroid_candidate() -> CentroidHystereticLearner:
    """Create the current best centroid-only anti-drift candidate."""
    ablation = importlib.import_module("step2_upgd_ablation")
    base_config = ablation.UPGD_CATALOG[
        (
            "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
            "meta001_center_notrunk_tight"
        )
    ]
    base = ablation.make_upgd(base_config, N_CLASSES)
    return CentroidHystereticLearner(base)


def make_single_upgd_candidates() -> dict[str, UPGDLearner]:
    """Create fixed-readout UPGD candidates with no gates or memory blends."""
    return {
        "upgd_structure_linear_h64": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(64,),
            readout_mode="linear_mse",
        ),
        "upgd_structure_softmax_h64": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(64,),
            readout_mode="softmax_ce",
        ),
        "upgd_structure_linear_h128": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(128,),
            readout_mode="linear_mse",
        ),
        "upgd_structure_softmax_h128": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(128,),
            readout_mode="softmax_ce",
        ),
    }


def make_smoothed_single_upgd_candidates() -> dict[str, SmoothedSimplexLearner]:
    """Create fixed-smoothed softmax UPGD candidates with no gates."""
    candidates: dict[str, SmoothedSimplexLearner] = {}
    for width in (64, 128):
        for smoothing in (0.30, 0.40):
            label = int(round(smoothing * 100))
            candidates[f"upgd_structure_softmax_h{width}_smooth{label}"] = (
                SmoothedSimplexLearner(
                    UPGDLearner.step2_default(
                        n_heads=N_CLASSES,
                        hidden_sizes=(width,),
                        readout_mode="softmax_ce",
                    ),
                    smoothing=smoothing,
                )
            )
    return candidates


def make_brier_single_upgd_candidates() -> dict[str, UPGDLearner]:
    """Create fixed softmax-Brier UPGD candidates with no gates."""
    return {
        "upgd_structure_brier_h64": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(64,),
            readout_mode="softmax_mse",
        ),
        "upgd_structure_brier_h128": UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(128,),
            readout_mode="softmax_mse",
        ),
    }


def make_temperature_single_upgd_candidates() -> dict[str, TemperatureScaledSimplexLearner]:
    """Create fixed-temperature softmax UPGD candidates with no gates."""
    def learner(temperature: float) -> TemperatureScaledSimplexLearner:
        return TemperatureScaledSimplexLearner(
            UPGDLearner.step2_default(
                n_heads=N_CLASSES,
                hidden_sizes=(256,),
                readout_mode="softmax_ce",
            ),
            temperature=temperature,
        )

    return {
        "upgd_structure_softmax_h256_temp4p0": learner(4.0),
        "upgd_structure_softmax_h256_temp4p25": learner(4.25),
        "upgd_structure_softmax_h256_temp4p5": learner(4.5),
        "upgd_structure_softmax_h256_temp4p75": learner(4.75),
        "upgd_structure_softmax_h256_temp5": learner(5.0),
        "upgd_structure_softmax_h256_temp5p0": learner(5.0),
    }


def make_rls_calibrated_candidate() -> RLSCalibratedLearner:
    """Create the compute-cheap prediction-space RLS calibration candidate."""
    return RLSCalibratedLearner(
        UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(64,),
            readout_mode="softmax_ce",
        ),
        ridge=10.0,
        forgetting=0.995,
        utility_decay=0.99,
        utility_off_threshold=0.0001,
        utility_on_threshold=0.0002,
        mix_scale=1.0,
        prediction_mode="simplex",
        identity_init=True,
    )


def make_primary_rls_calibrated_candidate(feature_dim: int) -> RLSCalibratedLearner:
    """Create RLS calibration around the packaged UPGD-memory hybrid."""
    return RLSCalibratedLearner(
        make_step2_hybrid_learner(
            Step2HybridConfig(
                feature_dim=feature_dim,
                n_heads=N_CLASSES,
                hidden_sizes=(64,),
                readout_mode="softmax_ce",
            )
        ),
        ridge=10.0,
        forgetting=0.995,
        utility_decay=0.99,
        utility_off_threshold=0.0001,
        utility_on_threshold=0.0002,
        mix_scale=1.0,
        prediction_mode="simplex",
        identity_init=True,
        init_requires_feature_dim=False,
    )


def make_dreaming_single_upgd_candidates() -> dict[str, DreamReplayLearner]:
    """Create dream-replay variants around the h64 softmax UPGD candidate."""
    def base() -> UPGDLearner:
        return UPGDLearner.step2_default(
            n_heads=N_CLASSES,
            hidden_sizes=(64,),
            readout_mode="softmax_ce",
        )

    return {
        "upgd_structure_softmax_h64_dream_surprise": DreamReplayLearner(
            base(),
            capacity=128,
            dreams_per_step=1,
            warmup_steps=64,
            mode="surprise",
        ),
        "upgd_structure_softmax_h64_dream_progress": DreamReplayLearner(
            base(),
            capacity=128,
            dreams_per_step=1,
            warmup_steps=64,
            mode="progress",
        ),
        "upgd_structure_softmax_h64_dream_random": DreamReplayLearner(
            base(),
            capacity=128,
            dreams_per_step=1,
            warmup_steps=64,
            mode="random",
        ),
        "upgd_structure_softmax_h64_dream_hybrid": DreamReplayLearner(
            base(),
            capacity=128,
            dreams_per_step=1,
            warmup_steps=64,
            mode="hybrid",
            learning_progress_weight=1.0,
        ),
        "upgd_structure_softmax_h64_dream_surprise2": DreamReplayLearner(
            base(),
            capacity=128,
            dreams_per_step=2,
            warmup_steps=64,
            mode="surprise",
        ),
        "upgd_structure_softmax_h64_dream_surprise_c256": DreamReplayLearner(
            base(),
            capacity=256,
            dreams_per_step=1,
            warmup_steps=64,
            mode="surprise",
        ),
    }


def make_primary_dreaming_candidate(feature_dim: int) -> DreamReplayLearner:
    """Create high-surprise dream replay around the packaged primary hybrid."""
    return DreamReplayLearner(
        make_step2_hybrid_learner(
            Step2HybridConfig(
                feature_dim=feature_dim,
                n_heads=N_CLASSES,
                hidden_sizes=(64,),
                readout_mode="softmax_ce",
            )
        ),
        capacity=128,
        dreams_per_step=1,
        warmup_steps=64,
        mode="surprise",
        init_requires_feature_dim=False,
    )


def make_delight_gated_candidates(feature_dim: int) -> dict[str, DelightGatedLearner]:
    """Create supervised Kondo-gated OPMNIST candidates."""

    def hybrid(rate: float) -> DelightGatedLearner:
        return DelightGatedLearner(
            make_step2_hybrid_learner(
                Step2HybridConfig(
                    feature_dim=feature_dim,
                    n_heads=N_CLASSES,
                    hidden_sizes=(64,),
                    readout_mode="softmax_ce",
                )
            ),
            target_rate=rate,
            temperature=0.05,
            price_step_size=0.01,
            init_requires_feature_dim=False,
        )

    def single_upgd(hidden_size: int, rate: float) -> DelightGatedLearner:
        return DelightGatedLearner(
            UPGDLearner.step2_default(
                n_heads=N_CLASSES,
                hidden_sizes=(hidden_size,),
                readout_mode="softmax_ce",
            ),
            target_rate=rate,
            temperature=0.05,
            price_step_size=0.01,
        )

    return {
        "step2_hybrid_memory_trace_delight_gate15": hybrid(0.15),
        "step2_hybrid_memory_trace_delight_gate30": hybrid(0.30),
        "step2_hybrid_memory_trace_delight_gate50": hybrid(0.50),
        "upgd_structure_softmax_h64_delight_gate30": single_upgd(64, 0.30),
        "upgd_structure_softmax_h128_delight_gate30": single_upgd(128, 0.30),
    }


def make_methods(
    feature_dim: int,
    *,
    include_centroid_candidates: bool = False,
    include_sharpened_mlp: bool = False,
    include_primary_sharpened: bool = False,
    include_adaptive_primary_sharpened: bool = False,
    include_prototype_memory: bool = False,
    include_single_upgd: bool = False,
    include_smoothed_single_upgd: bool = False,
    include_brier_single_upgd: bool = False,
    include_temperature_single_upgd: bool = False,
    include_rls_calibrated: bool = False,
    include_dreaming_candidates: bool = False,
    include_delight_candidates: bool = False,
) -> dict[str, Any]:
    """Create the primary hybrid and its MLP baselines."""
    methods: dict[str, Any] = {
        PRIMARY_METHOD: make_step2_hybrid_learner(
            Step2HybridConfig(
                feature_dim=feature_dim,
                n_heads=N_CLASSES,
                hidden_sizes=(64,),
                readout_mode="softmax_ce",
            )
        )
    }
    if include_primary_sharpened:
        methods[PRIMARY_SHARP_METHOD] = SharpenedLearner(
            make_step2_hybrid_learner(
                Step2HybridConfig(
                    feature_dim=feature_dim,
                    n_heads=N_CLASSES,
                    hidden_sizes=(64,),
                    readout_mode="softmax_ce",
                )
            ),
            init_requires_feature_dim=False,
        )
    if include_adaptive_primary_sharpened:
        methods[PRIMARY_ADAPTIVE_SHARP_METHOD] = AdaptiveSharpenedLearner(
            make_step2_hybrid_learner(
                Step2HybridConfig(
                    feature_dim=feature_dim,
                    n_heads=N_CLASSES,
                    hidden_sizes=(64,),
                    readout_mode="softmax_ce",
                )
            ),
            init_requires_feature_dim=False,
        )
    for method in MLP_METHODS:
        methods[method] = make_mlp(method)
    if include_sharpened_mlp or include_centroid_candidates:
        for method in SHARPENED_MLP_METHODS:
            methods[method] = SharpenedLearner(make_mlp(method))
    if include_centroid_candidates:
        methods[CENTROID_METHOD] = make_centroid_candidate()
    if include_prototype_memory:
        methods["proto_mem_s20"] = FixedPrototypeMemoryCandidate(
            feature_dim,
            slots_per_class=20,
        )
        methods["proto_mem_s32"] = FixedPrototypeMemoryCandidate(
            feature_dim,
            slots_per_class=32,
        )
    if include_single_upgd:
        methods.update(make_single_upgd_candidates())
    if include_smoothed_single_upgd:
        methods.update(make_smoothed_single_upgd_candidates())
    if include_brier_single_upgd:
        methods.update(make_brier_single_upgd_candidates())
    if include_temperature_single_upgd:
        methods.update(make_temperature_single_upgd_candidates())
    if include_rls_calibrated:
        methods[PRIMARY_RLS_CALIBRATED_METHOD] = make_primary_rls_calibrated_candidate(
            feature_dim
        )
        methods[RLS_CALIBRATED_METHOD] = make_rls_calibrated_candidate()
    if include_dreaming_candidates:
        methods[PRIMARY_DREAM_METHOD] = make_primary_dreaming_candidate(feature_dim)
        methods.update(make_dreaming_single_upgd_candidates())
    if include_delight_candidates:
        methods.update(make_delight_gated_candidates(feature_dim))
    return methods


def filter_methods(methods: dict[str, Any], only_methods: str | None) -> dict[str, Any]:
    """Restrict methods to a comma-separated subset while preserving order."""
    if only_methods is None:
        return methods
    requested = [name.strip() for name in only_methods.split(",") if name.strip()]
    if not requested:
        raise ValueError("--only-methods must name at least one method")
    unknown = [name for name in requested if name not in methods]
    if unknown:
        raise ValueError(
            "--only-methods requested unavailable methods: "
            + ", ".join(sorted(unknown))
        )
    return {name: methods[name] for name in requested}


def init_states(methods: dict[str, Any], feature_dim: int, seed: int) -> dict[str, Any]:
    """Initialize learner states."""
    states: dict[str, Any] = {}
    for idx, (name, learner) in enumerate(methods.items()):
        key = jr.key(seed + 31_337 + idx)
        if name == PRIMARY_METHOD:
            states[name] = learner.init(key)
        else:
            states[name] = learner.init(feature_dim, key)
    return states


@functools.partial(jax.jit, static_argnums=(0,))
def scan_classifier_chunk(
    learner: Any,
    state: Any,
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> tuple[Any, jax.Array]:
    """Run one learner over a chunk and return per-step MSE/accuracy."""

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        prediction = result.predictions
        mse = jnp.mean((prediction - target) ** 2)
        correct = (jnp.argmax(prediction) == label).astype(jnp.float32)
        return result.state, jnp.stack([mse, correct])

    return jax.lax.scan(step_fn, state, (observations, targets, labels))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(3):
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            with tmp_path.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(path)
            return
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.ENOSPC or attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    if last_error is not None:
        raise last_error


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def checkpoint_sidecar(path: Path) -> Path:
    """Return the JSON sidecar path for a checkpoint."""
    return path.with_suffix(f"{path.suffix}.json" if path.suffix else ".json")


def progress_log_path(path: Path) -> Path:
    """Return the JSONL progress path for a checkpoint."""
    return path.with_suffix(f"{path.suffix}.progress.jsonl" if path.suffix else ".progress.jsonl")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one progress row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _legacy_simplex_ema(state: UPGDState) -> jax.Array:
    """Infer a missing target-simplex EMA for old UPGD checkpoints."""
    previous_targets = jnp.asarray(getattr(state, "previous_targets", jnp.asarray([])))
    if previous_targets.ndim != 1 or previous_targets.shape[0] <= 1:
        return jnp.array(0.0, dtype=jnp.float32)
    target_sum = float(jnp.sum(previous_targets))
    target_min = float(jnp.min(previous_targets))
    target_max = float(jnp.max(previous_targets))
    simplex_like = (
        target_min >= -1e-6
        and target_max <= 1.0 + 1e-6
        and abs(target_sum - 1.0) <= 1e-4
    )
    return jnp.array(1.0 if simplex_like else 0.0, dtype=jnp.float32)


def _attr_or_default(state: Any, name: str, default: Any) -> Any:
    """Return an attribute unless it is missing or explicitly ``None``."""
    value = getattr(state, name, default)
    return default if value is None else value


def migrate_upgd_state(state: UPGDState) -> UPGDState:
    """Upgrade older pickled UPGD states to the current state schema."""
    has_current_schema = (
        hasattr(state, "target_simplex_ema")
        and hasattr(state, "readout_fast_head_params")
        and getattr(state, "unit_replacement_counts", None) is not None
    )
    if has_current_schema:
        return state
    unit_utilities = getattr(state, "unit_utilities", ())
    return UPGDState(  # type: ignore[call-arg]
        trunk_params=state.trunk_params,
        head_params=state.head_params,
        readout_fast_head_params=getattr(
            state,
            "readout_fast_head_params",
            state.head_params,
        ),
        readout_label_adapter=getattr(
            state,
            "readout_label_adapter",
            jnp.eye(len(state.head_params.weights), dtype=jnp.float32),
        ),
        utilities=getattr(state, "utilities", ()),
        unit_utilities=unit_utilities,
        unit_long_utilities=getattr(state, "unit_long_utilities", ()),
        unit_gradient_emas=getattr(state, "unit_gradient_emas", ()),
        unit_ages=getattr(state, "unit_ages", ()),
        unit_replacement_counts=_attr_or_default(
            state,
            "unit_replacement_counts",
            jnp.zeros(len(unit_utilities), dtype=jnp.float32),
        ),
        unit_replacement_accumulators=_attr_or_default(
            state,
            "unit_replacement_accumulators",
            jnp.zeros(len(unit_utilities), dtype=jnp.float32),
        ),
        loss_fast_ema=_attr_or_default(
            state,
            "loss_fast_ema",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        loss_slow_ema=_attr_or_default(
            state,
            "loss_slow_ema",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        previous_targets=_attr_or_default(
            state,
            "previous_targets",
            jnp.zeros(len(state.head_params.weights), dtype=jnp.float32),
        ),
        target_repeat_ema=_attr_or_default(
            state,
            "target_repeat_ema",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        target_simplex_ema=_attr_or_default(
            state,
            "target_simplex_ema",
            _legacy_simplex_ema(state),
        ),
        meta_trunk_log_scale=_attr_or_default(
            state,
            "meta_trunk_log_scale",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        meta_head_weight_log_scale=_attr_or_default(
            state,
            "meta_head_weight_log_scale",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        meta_head_bias_log_scale=_attr_or_default(
            state,
            "meta_head_bias_log_scale",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        meta_repetition_log_scale=_attr_or_default(
            state,
            "meta_repetition_log_scale",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        adaptive_kappa_log_scale=_attr_or_default(
            state,
            "adaptive_kappa_log_scale",
            jnp.array(0.0, dtype=jnp.float32),
        ),
        previous_trunk_weight_grads=getattr(state, "previous_trunk_weight_grads", ()),
        previous_trunk_bias_grads=getattr(state, "previous_trunk_bias_grads", ()),
        previous_head_weight_grads=getattr(state, "previous_head_weight_grads", ()),
        previous_head_bias_grads=getattr(state, "previous_head_bias_grads", ()),
        key=state.key,
        step_count=getattr(state, "step_count", jnp.array(0, dtype=jnp.int32)),
        birth_timestamp=float(getattr(state, "birth_timestamp", 0.0)),
        uptime_s=float(getattr(state, "uptime_s", 0.0)),
    )


def _replace_state_field(state: Any, field_name: str, value: Any) -> Any:
    """Return ``state`` with one nested field replaced."""
    if hasattr(state, "_replace"):
        return state._replace(**{field_name: value})
    if hasattr(state, "replace"):
        return state.replace(**{field_name: value})
    raise TypeError(f"cannot replace field {field_name!r} on {type(state).__name__}")


def migrate_checkpoint_state(state: Any) -> Any:
    """Recursively migrate learner states loaded from older checkpoints."""
    if isinstance(state, UPGDState):
        return migrate_upgd_state(state)
    if hasattr(state, "upgd_state"):
        migrated = migrate_checkpoint_state(state.upgd_state)
        if migrated is not state.upgd_state:
            return _replace_state_field(state, "upgd_state", migrated)
    if hasattr(state, "base_state"):
        migrated = migrate_checkpoint_state(state.base_state)
        if migrated is not state.base_state:
            return _replace_state_field(state, "base_state", migrated)
    return state


def _methods_match_or_subset(saved_methods: Any, expected_methods: Any) -> bool:
    """Return true when expected methods exactly match or are saved subset."""
    if not isinstance(saved_methods, list) or not isinstance(expected_methods, list):
        return False
    if saved_methods == expected_methods:
        return True
    return all(name in saved_methods for name in expected_methods)


def save_checkpoint(
    path: Path,
    *,
    completed_steps: int,
    states: dict[str, Any],
    accumulators: dict[str, MethodAccumulator],
    feature_orders: tuple[np.ndarray, ...],
    config: dict[str, Any],
    elapsed_s: float,
    progress_history: list[dict[str, Any]],
) -> None:
    """Persist resumable OPMNIST state."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "completed_steps": int(completed_steps),
        "states": states,
        "accumulators": accumulators,
        "feature_orders": feature_orders,
        "config": config,
        "elapsed_s": float(elapsed_s),
        "progress_history": progress_history,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    atomic_write_json(
        checkpoint_sidecar(path),
        {
            "version": CHECKPOINT_VERSION,
            "completed_steps": int(completed_steps),
            "elapsed_s": float(elapsed_s),
            "latest_progress": progress_history[-1] if progress_history else None,
            "config": config,
            "updated_at_utc": payload["updated_at_utc"],
        },
    )


def load_checkpoint(path: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    """Load a checkpoint and validate its resume-critical config."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported checkpoint version in {path}")
    saved_config = payload.get("config", {})
    for key, expected_value in expected_config.items():
        if key == "methods" and _methods_match_or_subset(
            saved_config.get(key),
            expected_value,
        ):
            continue
        if saved_config.get(key) != expected_value:
            raise RuntimeError(
                f"checkpoint config mismatch for {key}: "
                f"saved={saved_config.get(key)!r}, current={expected_value!r}"
            )
    expected_methods = list(expected_config.get("methods", []))
    migrated_states = {
        name: migrate_checkpoint_state(state)
        for name, state in payload.get("states", {}).items()
        if not expected_methods or name in expected_methods
    }
    migrated_accumulators = {
        name: accumulator
        for name, accumulator in payload.get("accumulators", {}).items()
        if not expected_methods or name in expected_methods
    }
    if expected_methods:
        payload["states"] = {name: migrated_states[name] for name in expected_methods}
        payload["accumulators"] = {
            name: migrated_accumulators[name] for name in expected_methods
        }
    else:
        payload["states"] = migrated_states
        payload["accumulators"] = migrated_accumulators
    payload["config"] = {**saved_config, "methods": expected_methods}
    return cast(dict[str, Any], payload)


def paired_diff(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Return paired candidate-vs-baseline deltas."""
    cand = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    diff = cand - base if higher_is_better else base - cand
    return {
        "diff_mean_positive_favors_candidate": float(np.mean(diff)),
        "diff_stderr": float(np.std(diff, ddof=1) / math.sqrt(diff.shape[0]))
        if diff.shape[0] > 1
        else 0.0,
        "wins_for_candidate": int(np.sum(diff > 0.0)),
        "wins_for_baseline": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "diffs": diff.tolist(),
    }


def method_names_from_records(records: list[dict[str, Any]]) -> list[str]:
    """Return the ordered method list shared by all records."""
    if not records:
        raise RuntimeError("cannot aggregate an empty record list")
    method_names = list(records[0]["methods"])
    for record in records[1:]:
        if list(record["methods"]) != method_names:
            raise RuntimeError("all records must contain the same ordered methods")
    return method_names


def mlp_method_names(method_names: Sequence[str]) -> list[str]:
    """Return method names treated as fair MLP comparators."""
    return [name for name in method_names if name.startswith("mlp_")]


def candidate_method_names(method_names: Sequence[str]) -> list[str]:
    """Return non-MLP methods compared against the best fair MLP."""
    return [name for name in method_names if not name.startswith("mlp_")]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed method records and compare candidates vs best MLP."""
    method_names = method_names_from_records(records)
    mlp_names = mlp_method_names(method_names)
    summary: dict[str, Any] = {}
    for method in method_names:
        rows = [record["methods"][method] for record in records]
        summary[method] = {}
        for metric in (
            "online_mean_mse",
            "online_mean_accuracy",
            "final_window_mse",
            "final_window_accuracy",
            "test_mse",
            "test_accuracy",
        ):
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[method][metric] = {
                "mean": float(np.mean(values)),
                "stderr": float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))
                if values.shape[0] > 1
                else 0.0,
                "per_seed": values.tolist(),
            }

    comparisons: dict[str, Any] = {}
    metric_specs = {
        "online_mean_mse": False,
        "online_mean_accuracy": True,
        "final_window_mse": False,
        "final_window_accuracy": True,
        "test_mse": False,
        "test_accuracy": True,
    }
    if mlp_names:
        for metric, higher in metric_specs.items():
            best_mlp = (max if higher else min)(
                mlp_names,
                key=lambda name: summary[name][metric]["mean"],
            )
            candidate_diffs: dict[str, Any] = {}
            for candidate in candidate_method_names(method_names):
                candidate_diffs[candidate] = paired_diff(
                    [
                        float(record["methods"][candidate][metric])
                        for record in records
                    ],
                    [
                        float(record["methods"][best_mlp][metric])
                        for record in records
                    ],
                    higher_is_better=higher,
                )
            comparison_row: dict[str, Any] = {
                "best_mlp": best_mlp,
                "candidate_vs_best_mlp": candidate_diffs,
            }
            if PRIMARY_METHOD in candidate_diffs:
                comparison_row["primary_vs_best_mlp"] = candidate_diffs[
                    PRIMARY_METHOD
                ]
            comparisons[metric] = comparison_row
    summary["comparisons"] = comparisons
    summary["method_order"] = method_names
    summary["mlp_methods"] = mlp_names
    summary["candidate_methods"] = candidate_method_names(method_names)
    return summary


def _candidate_metric_wins(
    aggregate: dict[str, Any],
    candidate: str,
) -> dict[str, bool]:
    """Return candidate-vs-best-MLP wins for all solution metrics."""
    comparisons = aggregate.get("comparisons", {})
    wins: dict[str, bool] = {}
    for metric in CORE_SOLUTION_METRICS:
        metric_comparison = comparisons.get(metric, {})
        candidate_rows = metric_comparison.get("candidate_vs_best_mlp", {})
        candidate_row = candidate_rows.get(candidate, {})
        value = candidate_row.get("diff_mean_positive_favors_candidate")
        wins[metric] = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0.0
        )
    return wins


def _valid_sha256(value: Any) -> bool:
    """Return true for a lowercase hexadecimal SHA-256 digest string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _runner_manifest_complete(manifest: dict[str, Any]) -> bool:
    """Return true when a direct runner manifest has publishable provenance."""
    git = manifest.get("git")
    environment = manifest.get("environment")
    source_sha256 = manifest.get("source_sha256")
    return bool(
        manifest.get("schema") == "alberta.step2.upgd_memory_opmnist.manifest.v1"
        and isinstance(manifest.get("argv"), list)
        and isinstance(manifest.get("methods"), list)
        and isinstance(git, dict)
        and isinstance(git.get("commit"), str)
        and isinstance(environment, dict)
        and isinstance(environment.get("jax"), str)
        and isinstance(source_sha256, dict)
        and _valid_sha256(source_sha256.get("runner"))
    )


def _merge_manifest_complete(manifest: dict[str, Any], *, min_seeds: int) -> bool:
    """Return true when a merged split manifest preserves per-seed provenance."""
    split_results = manifest.get("split_results")
    if not isinstance(split_results, list) or len(split_results) < min_seeds:
        return False
    if not (
        manifest.get("schema")
        == "alberta.step2.upgd_memory_opmnist.merge_manifest.v1"
        and isinstance(manifest.get("methods"), list)
        and isinstance(manifest.get("seeds"), list)
        and _valid_sha256(manifest.get("merge_script_sha256"))
        and _valid_sha256(manifest.get("runner_sha256"))
    ):
        return False
    for row in split_results:
        if not isinstance(row, dict):
            return False
        if not _valid_sha256(row.get("sha256")):
            return False
        split_manifest = row.get("manifest")
        if not isinstance(split_manifest, dict):
            return False
        if not _runner_manifest_complete(split_manifest):
            return False
    return True


def _mixed_method_manifest_complete(
    manifest: dict[str, Any],
    *,
    min_seeds: int,
) -> bool:
    """Return true when mixed baseline/candidate artifacts are fully traceable."""
    baseline = manifest.get("baseline_artifact")
    candidate_splits = manifest.get("candidate_split_results")
    if not (
        manifest.get("schema")
        == "alberta.step2.upgd_memory_opmnist.mixed_method_manifest.v1"
        and isinstance(manifest.get("methods"), list)
        and isinstance(manifest.get("seeds"), list)
        and _valid_sha256(manifest.get("combine_script_sha256"))
        and _valid_sha256(manifest.get("runner_sha256"))
        and isinstance(baseline, dict)
        and isinstance(candidate_splits, list)
        and len(candidate_splits) >= min_seeds
    ):
        return False

    baseline_manifest = baseline.get("manifest")
    if not (
        _valid_sha256(baseline.get("sha256"))
        and isinstance(baseline_manifest, dict)
        and (
            _runner_manifest_complete(baseline_manifest)
            or _merge_manifest_complete(baseline_manifest, min_seeds=min_seeds)
        )
    ):
        return False

    for row in candidate_splits:
        if not isinstance(row, dict):
            return False
        split_manifest = row.get("manifest")
        if not (
            _valid_sha256(row.get("sha256"))
            and isinstance(split_manifest, dict)
            and _runner_manifest_complete(split_manifest)
        ):
            return False
    return True


def opmnist_artifact_provenance_status(
    payload: dict[str, Any],
    *,
    min_seeds: int = 3,
) -> dict[str, Any]:
    """Audit whether an OPMNIST artifact carries publishable provenance."""
    manifest = payload.get("manifest")
    manifest_dict = manifest if isinstance(manifest, dict) else {}
    direct_runner_manifest_complete = _runner_manifest_complete(manifest_dict)
    merged_split_manifest_complete = _merge_manifest_complete(
        manifest_dict,
        min_seeds=min_seeds,
    )
    mixed_method_manifest_complete = _mixed_method_manifest_complete(
        manifest_dict,
        min_seeds=min_seeds,
    )
    return {
        "schema": "alberta.step2.opmnist_artifact_provenance_status.v1",
        "manifest_schema": manifest_dict.get("schema"),
        "direct_runner_manifest_complete": bool(direct_runner_manifest_complete),
        "merged_split_manifest_complete": bool(merged_split_manifest_complete),
        "mixed_method_manifest_complete": bool(mixed_method_manifest_complete),
        "provenance_complete": bool(
            direct_runner_manifest_complete
            or merged_split_manifest_complete
            or mixed_method_manifest_complete
        ),
    }


def opmnist_solution_status(
    payload: dict[str, Any],
    *,
    min_seeds: int = 3,
) -> dict[str, Any]:
    """Audit whether an OPMNIST result solves the Step 2 acceptance gate.

    This is intentionally stricter than "published scale": it requires true
    800-task MNIST protocol completion for multiple seeds and a candidate that
    beats the best fair MLP comparator on every core online/final/held-out
    metric. The function is schema-tolerant enough to audit both fresh runner
    outputs and older canonical snapshots.
    """
    config = payload.get("config")
    datasets = payload.get("datasets")
    records = payload.get("records")
    aggregate = payload.get("aggregate")
    config_dict = config if isinstance(config, dict) else {}
    datasets_dict = datasets if isinstance(datasets, dict) else {}
    dataset = datasets_dict.get("permuted_mnist_like", {})
    dataset_dict = dataset if isinstance(dataset, dict) else {}
    records_list = records if isinstance(records, list) else []
    aggregate_dict = aggregate if isinstance(aggregate, dict) else {}
    task_aggregate = aggregate_dict.get("permuted_mnist_like", {})
    task_aggregate_dict = task_aggregate if isinstance(task_aggregate, dict) else {}

    configured_seed_count = config_dict.get("n_seeds")
    if isinstance(configured_seed_count, int) and not isinstance(configured_seed_count, bool):
        n_configured_seeds = configured_seed_count
    else:
        n_configured_seeds = len(records_list)
    record_seeds = {
        int(record["seed"])
        for record in records_list
        if isinstance(record, dict) and isinstance(record.get("seed"), int)
    }
    record_steps_complete = bool(
        records_list
        and all(
            isinstance(record, dict)
            and isinstance(record.get("dataset"), dict)
            and record["dataset"].get("steps") == pub.DOHARE_OPMNIST_TOTAL_STEPS
            for record in records_list
        )
    )
    protocol_complete = bool(
        config_dict.get("mnist_published_scale") is True
        and config_dict.get("steps") == pub.DOHARE_OPMNIST_TOTAL_STEPS
        and config_dict.get("n_permutations") == pub.DOHARE_OPMNIST_TASKS
        and config_dict.get("task_block_size") == pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and dataset_dict.get("is_true_mnist") is True
        and dataset_dict.get("is_full_mnist_split") is True
        and dataset_dict.get("n_train") == pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
        and dataset_dict.get("n_test") == 10_000
        and dataset_dict.get("steps") == pub.DOHARE_OPMNIST_TOTAL_STEPS
        and dataset_dict.get("completed_full_task_blocks") == pub.DOHARE_OPMNIST_TASKS
        and dataset_dict.get("opmnist_completed_full_60000_task_blocks")
        == pub.DOHARE_OPMNIST_TASKS
        and dataset_dict.get("matches_dohare_opmnist_core_protocol") is True
        and dataset_dict.get("matches_dohare_opmnist_published_task_count") is True
        and dataset_dict.get("prediction_before_update_every_step") is True
        and dataset_dict.get("task_id_provided_to_learner") is False
        and dataset_dict.get("test_views_cover_all_permutations") is True
        and record_steps_complete
    )
    multi_seed_full_scale = bool(
        protocol_complete
        and n_configured_seeds >= min_seeds
        and len(record_seeds) >= min_seeds
        and len(record_seeds) == len(records_list)
    )
    candidates = [
        str(candidate)
        for candidate in task_aggregate_dict.get("candidate_methods", [])
        if isinstance(candidate, str)
    ]
    candidate_metric_wins = {
        candidate: _candidate_metric_wins(task_aggregate_dict, candidate)
        for candidate in candidates
    }
    candidates_winning_all_metrics = [
        candidate
        for candidate, metric_wins in candidate_metric_wins.items()
        if all(metric_wins.values())
    ]
    provenance_status = opmnist_artifact_provenance_status(
        payload,
        min_seeds=min_seeds,
    )
    solved = bool(
        multi_seed_full_scale
        and candidates_winning_all_metrics
        and provenance_status["provenance_complete"]
    )
    return {
        "schema": "alberta.step2.opmnist_solution_status.v1",
        "min_seeds": int(min_seeds),
        "protocol_complete": bool(protocol_complete),
        "configured_seed_count": int(n_configured_seeds),
        "completed_record_seed_count": int(len(record_seeds)),
        "multi_seed_full_scale": bool(multi_seed_full_scale),
        "artifact_provenance": provenance_status,
        "candidate_metric_wins": candidate_metric_wins,
        "candidates_winning_all_metrics": candidates_winning_all_metrics,
        "solved_opmnist_step2": bool(solved),
        "claim_scope": (
            "multi_seed_full_scale_opmnist_solution"
            if solved
            else "limited_opmnist_evidence_not_step2_solution"
        ),
    }


def build_resume_config(
    args: argparse.Namespace,
    dataset: pub.ClassificationDataset,
    seed: int,
    stream_seed: int,
    method_names: Sequence[str],
) -> dict[str, Any]:
    """Return fields that must match when resuming."""
    return {
        "seed": int(seed),
        "stream_seed": int(stream_seed),
        "source_kind": dataset.metadata.get("source_kind"),
        "dataset_split": dataset.metadata.get("split"),
        "dataset_n_train": dataset.metadata.get("n_train"),
        "dataset_n_test": dataset.metadata.get("n_test"),
        "dataset_is_full_mnist_split": dataset.metadata.get("is_full_mnist_split"),
        "max_train_examples": dataset.metadata.get("max_train_examples"),
        "max_test_examples": dataset.metadata.get("max_test_examples"),
        "feature_dim": int(dataset.x_train.shape[1]),
        "n_heads": N_CLASSES,
        "n_permutations": int(args.n_permutations),
        "task_block_size": int(args.task_block_size),
        "sample_with_replacement": bool(args.sample_with_replacement),
        "task_sampling": args.task_sampling,
        "include_identity_permutation": bool(args.include_identity_permutation),
        "final_window": int(args.final_window),
        "methods": list(method_names),
    }


def checkpoint_path(seed: int, args: argparse.Namespace) -> Path:
    """Return checkpoint path for one seed."""
    if args.resume_path is not None:
        path = cast(Path, args.resume_path)
        if args.n_seeds > 1:
            return path.with_name(f"{path.stem}_seed{seed}{path.suffix or '.pkl'}")
        return path
    return cast(Path, args.output_dir) / f"{args.result_prefix}_seed{seed}_resume.pkl"


def run_one_seed(seed: int, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one paired seed, resuming from checkpoint when available."""
    dataset = pub.load_mnist_like_source(args, seed)
    feature_dim = int(dataset.x_train.shape[1])
    stream_seed = seed + 10_000
    feature_orders = pub.make_feature_orders(
        seed=stream_seed,
        feature_dim=feature_dim,
        n_permutations=args.n_permutations,
        include_identity_permutation=args.include_identity_permutation,
    )
    observed_task_ids = pub.opmnist_task_ids_for_steps(
        steps=args.steps,
        n_permutations=args.n_permutations,
        task_block_size=args.task_block_size,
    )
    test_task_ids = pub.opmnist_test_task_ids(
        observed_task_ids=observed_task_ids,
        n_permutations=args.n_permutations,
        max_test_permutation_views=args.max_test_permutation_views,
        evaluate_all_permutation_views=args.evaluate_all_permutation_views,
    )
    methods = make_methods(
        feature_dim,
        include_centroid_candidates=args.include_centroid_candidates,
        include_sharpened_mlp=args.include_sharpened_mlp,
        include_primary_sharpened=args.include_primary_sharpened,
        include_adaptive_primary_sharpened=args.include_adaptive_primary_sharpened,
        include_prototype_memory=args.include_prototype_memory,
        include_single_upgd=args.include_single_upgd,
        include_smoothed_single_upgd=args.include_smoothed_single_upgd,
        include_brier_single_upgd=args.include_brier_single_upgd,
        include_temperature_single_upgd=args.include_temperature_single_upgd,
        include_rls_calibrated=args.include_rls_calibrated,
        include_dreaming_candidates=args.include_dreaming_candidates,
        include_delight_candidates=args.include_delight_candidates,
    )
    methods = filter_methods(methods, args.only_methods)
    states = init_states(methods, feature_dim, seed)
    accumulators = {name: init_accumulator(args.final_window) for name in methods}
    resume_config = build_resume_config(args, dataset, seed, stream_seed, list(methods))
    path = checkpoint_path(seed, args)
    completed_steps = 0
    elapsed_s = 0.0
    progress_history: list[dict[str, Any]] = []
    checkpoint_loaded = False
    if args.resume and path.exists() and not args.force_restart:
        payload = load_checkpoint(path, resume_config)
        states = payload["states"]
        accumulators = payload["accumulators"]
        feature_orders = payload["feature_orders"]
        completed_steps = int(payload["completed_steps"])
        elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
        progress_history = list(payload.get("progress_history", []))
        checkpoint_loaded = True
        print(f"opmnist seed={seed}: resumed {completed_steps}/{args.steps} from {path}")
    elif args.force_restart and path.exists():
        path.unlink()
        for sidecar in (checkpoint_sidecar(path), progress_log_path(path)):
            if sidecar.exists():
                sidecar.unlink()

    meta = dict(dataset.metadata)
    meta.update(
        pub.opmnist_protocol_metadata(
            dataset=dataset,
            steps=args.steps,
            seed=stream_seed,
            n_permutations=args.n_permutations,
            task_block_size=args.task_block_size,
            sample_with_replacement=args.sample_with_replacement,
            task_sampling=args.task_sampling,
            include_identity_permutation=args.include_identity_permutation,
            max_test_permutation_views=args.max_test_permutation_views,
            evaluate_all_permutation_views=args.evaluate_all_permutation_views,
            observed_task_ids=observed_task_ids,
            test_task_ids=test_task_ids,
            streaming_runner=True,
            chunk_size=args.chunk_size,
            resume_checkpoint_path=str(path),
        )
    )
    meta["methods"] = list(methods)
    meta["mlp_methods"] = [name for name in methods if name.startswith("mlp_")]
    meta["candidate_methods"] = candidate_method_names(list(methods))

    chunks_run_this_invocation = 0
    while completed_steps < args.steps:
        chunk_steps = min(args.chunk_size, args.steps - completed_steps)
        chunk_t0 = time.time()
        observations, targets, labels_np = pub.make_permuted_classification_chunk(
            dataset=dataset,
            start_step=completed_steps,
            chunk_steps=chunk_steps,
            seed=stream_seed,
            n_permutations=args.n_permutations,
            task_block_size=args.task_block_size,
            sample_with_replacement=args.sample_with_replacement,
            task_sampling=args.task_sampling,
            feature_orders=feature_orders,
        )
        labels = jnp.asarray(labels_np)
        for name, learner in methods.items():
            states[name], metrics = scan_classifier_chunk(
                learner,
                states[name],
                observations,
                targets,
                labels,
            )
            np_metrics = np.asarray(metrics)
            accumulators[name] = update_accumulator(
                accumulators[name],
                np_metrics,
                args.final_window,
            )
        completed_steps += chunk_steps
        chunks_run_this_invocation += 1
        chunk_elapsed = time.time() - chunk_t0
        elapsed_s += chunk_elapsed
        recent_sps = float(chunk_steps / max(chunk_elapsed, 1e-12))
        status = pub.opmnist_progress_status(
            completed_steps=completed_steps,
            target_steps=pub.DOHARE_OPMNIST_TOTAL_STEPS,
            task_block_size=pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE,
            elapsed_s=elapsed_s,
            recent_steps_per_second=recent_sps,
        )
        progress_row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "seed": int(seed),
            "completed_steps": int(completed_steps),
            "requested_steps": int(args.steps),
            "dohare_target_steps": pub.DOHARE_OPMNIST_TOTAL_STEPS,
            "stop_after_chunks": args.stop_after_chunks,
            "chunks_run_this_invocation": int(chunks_run_this_invocation),
            "chunk_steps": int(chunk_steps),
            "chunk_elapsed_s": float(chunk_elapsed),
            "elapsed_s": float(elapsed_s),
            "steps_per_second": recent_sps,
            "completed_full_task_blocks": int(completed_steps // args.task_block_size),
            "eta_to_dohare_800_s": status["eta_seconds"],
        }
        progress_history.append(progress_row)
        if args.resume:
            save_checkpoint(
                path,
                completed_steps=completed_steps,
                states=states,
                accumulators=accumulators,
                feature_orders=feature_orders,
                config=resume_config,
                elapsed_s=elapsed_s,
                progress_history=progress_history,
            )
            append_jsonl(progress_log_path(path), progress_row)
        if args.status_path is not None:
            atomic_write_json(
                args.status_path,
                {
                    "schema": "alberta.upgd_memory_opmnist.status.v1",
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "seed": int(seed),
                    "checkpoint_path": str(path),
                    "requested_steps": int(args.steps),
                    "dohare_target_steps": pub.DOHARE_OPMNIST_TOTAL_STEPS,
                    "completed_steps": int(completed_steps),
                    "status": status,
                    "latest_progress": progress_row,
                    "protocol": meta,
                },
            )
        print(
            f"opmnist seed={seed}: streamed {completed_steps}/{args.steps} "
            f"({completed_steps // args.task_block_size} full blocks, "
            f"{recent_sps:.1f} steps/s, 800-task ETA "
            f"{pub.format_seconds(status['eta_seconds'])})"
        )
        if (
            args.stop_after_chunks is not None
            and chunks_run_this_invocation >= args.stop_after_chunks
            and completed_steps < args.steps
        ):
            raise PartialRunCompleteError(
                PartialRunStop(
                    seed=int(seed),
                    completed_steps=int(completed_steps),
                    requested_steps=int(args.steps),
                    checkpoint_path=path,
                    status_path=args.status_path,
                )
            )

    method_rows: dict[str, dict[str, float]] = {
        name: summarize_accumulator(accumulator)
        for name, accumulator in accumulators.items()
    }
    for name, learner in methods.items():
        method_rows[name].update(
            pub.evaluate_classifier_feature_orders(
                learner=learner,
                state=states[name],
                dataset=dataset,
                feature_orders=feature_orders,
                test_task_ids=test_task_ids,
            )
        )
    meta["checkpoint_loaded"] = checkpoint_loaded
    meta["opmnist_elapsed_s"] = elapsed_s
    meta["opmnist_overall_steps_per_second"] = (
        float(completed_steps / elapsed_s) if elapsed_s > 0.0 else None
    )
    meta["opmnist_eta_to_800_tasks_s"] = pub.opmnist_progress_status(
        completed_steps=completed_steps,
        target_steps=pub.DOHARE_OPMNIST_TOTAL_STEPS,
        task_block_size=pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE,
        elapsed_s=elapsed_s,
    )["eta_seconds"]
    return (
        {
            "dataset_name": "permuted_mnist_like",
            "seed": int(seed),
            "dataset": meta,
            "methods": method_rows,
        },
        meta,
    )


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric."""
    data = row[metric]
    return f"{data['mean']:.6f} +/- {data['stderr']:.6f}"


def write_note(results: dict[str, Any], path: Path) -> None:
    """Write a compact assessment note."""
    aggregate = results["aggregate"]["permuted_mnist_like"]
    method_order = aggregate["method_order"]
    lines = [
        "# Step 2 UPGD-Memory OPMNIST",
        "",
        "This note records the resumable OPMNIST run for the packaged "
        "UPGD-memory trace learner, optional simple candidates, and fair MLP "
        "baselines.",
        "",
        f"- Primary method: `{PRIMARY_METHOD}`",
        f"- MNIST source: `{results['config']['mnist_source']}`",
        f"- Steps: `{results['config']['steps']}`",
        f"- Seeds: `{results['config']['n_seeds']}`",
        f"- Permutations: `{results['config']['n_permutations']}`",
        f"- Task block size: `{results['config']['task_block_size']}`",
        "",
        "| Method | Final MSE | Final Acc | Test MSE | Test Acc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in method_order:
        row = aggregate[method]
        lines.append(
            f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
            f"{metric_cell(row, 'final_window_accuracy')} | "
            f"{metric_cell(row, 'test_mse')} | "
            f"{metric_cell(row, 'test_accuracy')} |"
        )
    if aggregate["comparisons"]:
        lines.extend(["", "## Primary vs Best MLP", ""])
        for metric, comparison in aggregate["comparisons"].items():
            paired = comparison.get("primary_vs_best_mlp")
            if paired is None:
                continue
            lines.append(
                f"- `{metric}` vs `{comparison['best_mlp']}`: "
                f"{paired['diff_mean_positive_favors_candidate']:+.6f} +/- "
                f"{paired['diff_stderr']:.6f}; wins/losses/ties "
                f"{paired['wins_for_candidate']}/{paired['wins_for_baseline']}/"
                f"{paired['ties']}."
            )
    candidate_methods = [
        method for method in aggregate["candidate_methods"] if method != PRIMARY_METHOD
    ]
    if candidate_methods and aggregate["comparisons"]:
        lines.extend(["", "## Additional Candidate vs Best MLP", ""])
        for metric, comparison in aggregate["comparisons"].items():
            best_mlp = comparison["best_mlp"]
            for method in candidate_methods:
                paired = comparison["candidate_vs_best_mlp"][method]
                lines.append(
                    f"- `{method}` `{metric}` vs `{best_mlp}`: "
                    f"{paired['diff_mean_positive_favors_candidate']:+.6f} +/- "
                    f"{paired['diff_stderr']:.6f}; wins/losses/ties "
                    f"{paired['wins_for_candidate']}/{paired['wins_for_baseline']}/"
                    f"{paired['ties']}."
                )
    lines.extend(
        [
            "",
            "## Scale Status",
            "",
            "A full published-scale OPMNIST result requires 800 completed "
            "60,000-example task blocks, or 48,000,000 online updates. This runner "
            "reports exact completed blocks and leaves a checkpoint/status sidecar "
            "for continuation rather than treating partial runs as full closure.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_optional_positive_int(value: str) -> int | None:
    """Parse optional positive int arguments."""
    return pub.optional_positive_int(value)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-safe CLI config."""
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: Sequence[str]) -> str:
    """Run a git command, returning ``unknown`` when git metadata is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def git_metadata() -> dict[str, Any]:
    """Return reproducibility metadata for the current git worktree."""
    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "describe": run_git(["describe", "--always", "--dirty", "--tags"]),
        "dirty": bool(status and status != "unknown"),
        "status_porcelain": status.splitlines() if status != "unknown" else [],
    }


def build_manifest(
    *,
    argv: Sequence[str],
    args: argparse.Namespace,
    method_names: Sequence[str],
) -> dict[str, Any]:
    """Build a publishable reproducibility manifest for one OPMNIST run."""
    source_files = {
        "runner": Path(__file__).resolve(),
        "published_stressors": Path(pub.__file__).resolve(),
    }
    return {
        "schema": "alberta.step2.upgd_memory_opmnist.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "argv": list(argv),
        "config": jsonable_args(args),
        "methods": list(method_names),
        "git": git_metadata(),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": getattr(jax.lib, "__version__", "unknown"),
            "numpy": np.__version__,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument(
        "--mnist-source",
        choices=("auto", "openml", "torchvision", "sklearn_digits_28x28", "sklearn_digits_8x8"),
        default="sklearn_digits_28x28",
    )
    parser.add_argument("--allow-openml-download", action="store_true")
    parser.add_argument("--allow-torchvision-download", action="store_true")
    parser.add_argument("--mnist-split", choices=("stratified", "canonical"), default="stratified")
    parser.add_argument("--openml-data-home", type=Path, default=None)
    parser.add_argument("--torchvision-data-home", type=Path, default=None)
    parser.add_argument("--openml-n-retries", type=int, default=2)
    parser.add_argument("--openml-retry-delay", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=parse_optional_positive_int, default=1_000)
    parser.add_argument("--max-test-examples", type=parse_optional_positive_int, default=400)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--task-block-size", type=int, default=200)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--include-identity-permutation", action="store_true")
    parser.add_argument(
        "--task-sampling",
        choices=("random", "sequential_epoch"),
        default="sequential_epoch",
    )
    parser.add_argument(
        "--max-test-permutation-views",
        type=parse_optional_positive_int,
        default=None,
    )
    parser.add_argument("--evaluate-all-permutation-views", action="store_true")
    parser.add_argument("--mnist-published-scale", action="store_true")
    parser.add_argument(
        "--opmnist-fraction",
        type=float,
        default=None,
        help=(
            "Run this fraction of the 800-task Dohare OPMNIST protocol. "
            "For example, 0.01 runs 480,000 updates, or 8 complete tasks."
        ),
    )
    parser.add_argument("--include-sharpened-mlp", action="store_true")
    parser.add_argument("--include-primary-sharpened", action="store_true")
    parser.add_argument("--include-adaptive-primary-sharpened", action="store_true")
    parser.add_argument("--include-centroid-candidates", action="store_true")
    parser.add_argument("--include-prototype-memory", action="store_true")
    parser.add_argument(
        "--include-single-upgd",
        action="store_true",
        help="Include fixed-readout target-structure UPGD candidates with no gates.",
    )
    parser.add_argument(
        "--include-smoothed-single-upgd",
        action="store_true",
        help=(
            "Include fixed-readout softmax UPGD candidates with a constant "
            "uniform simplex floor and no gates."
        ),
    )
    parser.add_argument(
        "--include-brier-single-upgd",
        action="store_true",
        help="Include fixed softmax-Brier target-structure UPGD candidates.",
    )
    parser.add_argument(
        "--include-temperature-single-upgd",
        action="store_true",
        help="Include fixed-temperature softmax UPGD candidates with no gates.",
    )
    parser.add_argument(
        "--include-rls-calibrated",
        action="store_true",
        help=(
            "Include prediction-space RLS calibration around a softmax UPGD "
            "candidate. This is the cheap online analogue of the Step 8 "
            "calibrated reward-model result."
        ),
    )
    parser.add_argument(
        "--include-dreaming-candidates",
        action="store_true",
        help=(
            "Include causal dream-replay variants around the primary hybrid "
            "and the h64 softmax UPGD candidate."
        ),
    )
    parser.add_argument(
        "--include-delight-candidates",
        action="store_true",
        help=(
            "Include supervised Kondo-style delight-gated update candidates "
            "inspired by Delightful Policy Gradient."
        ),
    )
    parser.add_argument(
        "--only-methods",
        type=str,
        default=None,
        help=(
            "Comma-separated method subset to run. This is intended for "
            "parallel wall-clock splits from a superset checkpoint."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=6_000)
    parser.add_argument(
        "--stop-after-chunks",
        type=int,
        default=None,
        help=(
            "Checkpoint and exit after this many chunks in the current "
            "invocation. No final result JSON is written unless the configured "
            "step count is completed."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", type=str, default="upgd_memory_opmnist")
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args(argv)


def apply_presets(args: argparse.Namespace) -> None:
    """Apply published-scale OPMNIST settings."""
    if not args.mnist_published_scale:
        return
    args.mnist_source = "openml"
    args.mnist_split = "canonical"
    args.max_train_examples = None
    args.max_test_examples = None
    args.n_permutations = pub.DOHARE_OPMNIST_TASKS
    args.task_block_size = pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
    if args.opmnist_fraction is not None:
        requested_steps = int(round(pub.DOHARE_OPMNIST_TOTAL_STEPS * args.opmnist_fraction))
        block_aligned_steps = (
            requested_steps // pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
        ) * pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE
        args.steps = max(pub.DOHARE_OPMNIST_TASK_BLOCK_SIZE, block_aligned_steps)
    elif args.steps == 1_000:
        args.steps = pub.DOHARE_OPMNIST_TOTAL_STEPS


def validate_args(args: argparse.Namespace) -> None:
    """Validate run arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.stop_after_chunks is not None and args.stop_after_chunks <= 0:
        raise ValueError("--stop-after-chunks must be positive")
    if args.n_permutations < 2:
        raise ValueError("--n-permutations must be at least 2")
    if args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")
    if args.opmnist_fraction is not None:
        if not args.mnist_published_scale:
            raise ValueError("--opmnist-fraction requires --mnist-published-scale")
        if not 0.0 < args.opmnist_fraction <= 1.0:
            raise ValueError("--opmnist-fraction must be in (0, 1]")
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark."""
    manifest_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    apply_presets(args)
    validate_args(args)
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        try:
            record, meta = run_one_seed(seed, args)
        except PartialRunCompleteError as exc:
            print(
                json.dumps(
                    {
                        "partial_run": True,
                        "seed": exc.stop.seed,
                        "completed_steps": exc.stop.completed_steps,
                        "requested_steps": exc.stop.requested_steps,
                        "checkpoint_path": str(exc.stop.checkpoint_path),
                        "status_path": str(exc.stop.status_path)
                        if exc.stop.status_path is not None
                        else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        records.append(record)
        datasets["permuted_mnist_like"] = meta
    method_names = method_names_from_records(records)
    mlp_names = mlp_method_names(method_names)
    results = {
        "config": {
            **jsonable_args(args),
            "output_dir": str(args.output_dir),
            "resume_path": str(args.resume_path) if args.resume_path is not None else None,
            "status_path": str(args.status_path) if args.status_path is not None else None,
            "note_path": str(args.note_path) if args.note_path is not None else None,
            "created_at": datetime.now(UTC).isoformat(),
            "runner": "step2_upgd_memory_opmnist",
            "methods": method_names,
        },
        "datasets": datasets,
        "records": records,
        "primary_method": PRIMARY_METHOD,
        "mlp_methods": mlp_names,
        "candidate_methods": candidate_method_names(method_names),
        "aggregate": {"permuted_mnist_like": aggregate_records(records)},
        "wall_clock_s": float(time.time() - t0),
        "evidence_level": "single_upgd_memory_opmnist_resumable",
    }
    results["manifest"] = build_manifest(
        argv=manifest_argv,
        args=args,
        method_names=method_names,
    )
    results["solution_status"] = opmnist_solution_status(results)
    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    atomic_write_json(json_path, results)
    write_note(results, summary_path)
    if args.note_path is not None:
        write_note(results, args.note_path)
    print(json.dumps({"result_path": str(json_path), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

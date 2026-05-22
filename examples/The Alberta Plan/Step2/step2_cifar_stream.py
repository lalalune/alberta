#!/usr/bin/env python3
"""Continual CIFAR-10 stream benchmark for Step 2 UPGD-memory.

The benchmark compares the packaged single Step 2 UPGD-memory trace learner
against plain target-structure UPGD and fair same-run MLP baselines on
materialized supervised streams.  Real CIFAR-10 uses torchvision or the public
Python archive when available.  Without local CIFAR data, the default path falls
back to a small deterministic CIFAR-shaped smoke dataset so tests and dry runs
require no network access.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    LMS,
    MultiHeadMLPLearner,
    ObGDBounding,
    Step2HybridConfig,
    UPGDLearner,
    UPGDMemoryLearner,
    make_step2_hybrid_learner,
)
from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
)

N_CLASSES = 10
PRIMARY_METHOD = "step2_hybrid_memory_trace"
PRIMARY_SHARP_METHOD = "step2_hybrid_memory_trace_sharp"
PRIMARY_ADAPTIVE_SHARP_METHOD = "step2_hybrid_memory_trace_adaptive_sharp"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
DEFAULT_OUTPUT_DIR = Path("outputs/step2_cifar_stream")
DEFAULT_NOTE_PATH = Path("docs/research/step2_cifar_stream.md")


class VectorLearner(Protocol):
    """Minimal learner protocol shared by UPGD and MultiHeadMLP."""

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        ...

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize learner state."""
        ...

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict a vector output."""
        ...

    def update(self, state: Any, observation: jax.Array, targets: jax.Array) -> Any:
        """Run one online update."""
        ...

    def to_config(self) -> dict[str, Any]:
        """Serialize learner config."""
        ...


class GenericUpdateResult(NamedTuple):
    """Minimal update result for local deployment-readout wrappers."""

    state: Any
    predictions: jax.Array


class AdaptiveSharpenedState(NamedTuple):
    """State for the causal raw-vs-sharpened readout gate."""

    base_state: Any
    sharp_advantage: jax.Array
    sharp_gate: jax.Array


@dataclass(frozen=True)
class CifarArrays:
    """Materialized image classification arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source: str
    real_cifar: bool


@dataclass(frozen=True)
class FixedFeatureLearnerAdapter:
    """Adapt learners whose feature dimension is fixed at construction time."""

    learner: UPGDMemoryLearner
    feature_dim: int

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self.learner.config.n_heads

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the wrapped learner after checking feature dimensionality."""
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature_dim {self.feature_dim}, got {feature_dim}"
            )
        return self.learner.init(key)

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict a vector output."""
        return self.learner.predict(state, observation)

    def update(self, state: Any, observation: jax.Array, targets: jax.Array) -> Any:
        """Run one online update."""
        return self.learner.update(state, observation, targets)

    def to_config(self) -> dict[str, Any]:
        """Serialize learner config."""
        payload = dict(self.learner.to_config())
        payload["fixed_feature_dim"] = self.feature_dim
        return payload


@dataclass(frozen=True)
class FixedPrototypeMemoryAdapter:
    """Adapt fixed-budget prototype memory to the vector-learner protocol."""

    learner: PrototypeMemoryLearner
    feature_dim: int

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self.learner.config.n_classes

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the wrapped learner after checking feature dimensionality."""
        del key
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature_dim {self.feature_dim}, got {feature_dim}"
            )
        return self.learner.init()

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict a vector output."""
        return self.learner.predict(state, observation)

    def update(self, state: Any, observation: jax.Array, targets: jax.Array) -> Any:
        """Run one online update."""
        return self.learner.update(state, observation, targets)

    def to_config(self) -> dict[str, Any]:
        """Serialize learner config."""
        payload = self.learner.to_config()
        payload["fixed_feature_dim"] = self.feature_dim
        return payload


@dataclass(frozen=True)
class RunConfig:
    """Benchmark configuration."""

    steps: int = 120
    n_seeds: int = 1
    final_window: int = 40
    max_train: int = 400
    max_test: int = 200
    data_dir: str = "data"
    allow_download: bool = False
    regimes: tuple[str, ...] = ("iid", "class_blocked")
    include_primary_sharpened: bool = False
    include_adaptive_primary_sharpened: bool = False
    include_sharpened_mlp: bool = False
    include_prototype_memory: bool = False
    include_wide_mlp: bool = False
    only_methods: tuple[str, ...] | None = None


def one_hot(labels: np.ndarray) -> np.ndarray:
    """Return one-hot CIFAR class targets."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return np.asarray(targets, dtype=np.float32)


def make_synthetic_cifar_like(
    seed: int,
    n_train: int,
    n_test: int,
    feature_dim: int = 32,
) -> CifarArrays:
    """Create a deterministic CIFAR-style smoke dataset with ten classes."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0.0, 1.0, size=(N_CLASSES, feature_dim)).astype(np.float32)

    def sample(n_items: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.arange(n_items, dtype=np.int32) % N_CLASSES
        rng.shuffle(labels)
        noise = rng.normal(0.0, 0.35, size=(n_items, feature_dim)).astype(np.float32)
        return centroids[labels] + noise, labels

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return standardize_arrays(
        CifarArrays(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            source="synthetic_cifar_smoke",
            real_cifar=False,
        )
    )


def _load_torchvision_cifar10(data_dir: Path, allow_download: bool) -> Any:
    """Load torchvision CIFAR-10 dynamically to keep the dependency optional."""
    datasets = importlib.import_module("torchvision.datasets")
    return datasets.CIFAR10(root=str(data_dir), train=True, download=allow_download)


def _download_cifar10_archive(data_dir: Path) -> Path:
    """Download the CIFAR-10 Python archive with the standard library."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / CIFAR10_ARCHIVE
    if archive_path.exists():
        return archive_path
    tmp_path = archive_path.with_suffix(".tmp")
    urllib.request.urlretrieve(CIFAR10_URL, tmp_path)  # noqa: S310
    tmp_path.replace(archive_path)
    return archive_path


def _ensure_cifar10_extracted(data_dir: Path, allow_download: bool) -> Path:
    """Return the extracted CIFAR-10 directory, downloading when allowed."""
    root = data_dir / CIFAR10_DIR
    if root.exists():
        return root
    archive_path = data_dir / CIFAR10_ARCHIVE
    if not archive_path.exists():
        if not allow_download:
            raise FileNotFoundError("CIFAR-10 archive not available")
        archive_path = _download_cifar10_archive(data_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")
    return root


def _load_pickle_batch(path: Path) -> dict[bytes, Any]:
    """Load one CIFAR-10 Python batch file."""
    with path.open("rb") as f:
        return cast(dict[bytes, Any], pickle.load(f, encoding="bytes"))


def _load_direct_cifar10(
    data_dir: Path,
    allow_download: bool,
    max_train: int,
    max_test: int,
    seed: int,
) -> CifarArrays:
    """Load CIFAR-10 directly from the public Python archive."""
    root = _ensure_cifar10_extracted(data_dir, allow_download)
    train_batches = [_load_pickle_batch(root / f"data_batch_{i}") for i in range(1, 6)]
    test_batch = _load_pickle_batch(root / "test_batch")
    x_train = np.concatenate(
        [np.asarray(batch[b"data"], dtype=np.float32) for batch in train_batches],
        axis=0,
    )
    y_train = np.concatenate(
        [np.asarray(batch[b"labels"], dtype=np.int32) for batch in train_batches],
        axis=0,
    )
    x_test = np.asarray(test_batch[b"data"], dtype=np.float32)
    y_test = np.asarray(test_batch[b"labels"], dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(x_train.shape[0])[:max_train]
    test_idx = rng.permutation(x_test.shape[0])[:max_test]
    real = CifarArrays(
        x_train=x_train[train_idx] / 255.0,
        y_train=y_train[train_idx],
        x_test=x_test[test_idx] / 255.0,
        y_test=y_test[test_idx],
        source="cifar10_python_archive",
        real_cifar=True,
    )
    return standardize_arrays(real)


def load_cifar_or_smoke(
    data_dir: Path,
    allow_download: bool,
    max_train: int,
    max_test: int,
    seed: int,
) -> CifarArrays:
    """Load real CIFAR-10 when available, otherwise use synthetic smoke data."""
    try:
        train_ds = _load_torchvision_cifar10(data_dir, allow_download)
        datasets = importlib.import_module("torchvision.datasets")
        test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=allow_download)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        try:
            return _load_direct_cifar10(
                data_dir=data_dir,
                allow_download=allow_download,
                max_train=max_train,
                max_test=max_test,
                seed=seed,
            )
        except (OSError, pickle.UnpicklingError, tarfile.TarError):
            return make_synthetic_cifar_like(
                seed=seed,
                n_train=max(max_train, N_CLASSES * 2),
                n_test=max(max_test, N_CLASSES),
            )

    x_train = np.asarray(train_ds.data, dtype=np.float32) / 255.0
    y_train = np.asarray(train_ds.targets, dtype=np.int32)
    x_test = np.asarray(test_ds.data, dtype=np.float32) / 255.0
    y_test = np.asarray(test_ds.targets, dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(x_train.shape[0])[:max_train]
    test_idx = rng.permutation(x_test.shape[0])[:max_test]
    real = CifarArrays(
        x_train=x_train[train_idx].reshape(len(train_idx), -1),
        y_train=y_train[train_idx],
        x_test=x_test[test_idx].reshape(len(test_idx), -1),
        y_test=y_test[test_idx],
        source="torchvision_cifar10",
        real_cifar=True,
    )
    return standardize_arrays(real)


def standardize_arrays(arrays: CifarArrays) -> CifarArrays:
    """Standardize features by training-set statistics."""
    x_train = arrays.x_train.reshape(arrays.x_train.shape[0], -1).astype(np.float32)
    x_test = arrays.x_test.reshape(arrays.x_test.shape[0], -1).astype(np.float32)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    safe_std = np.where(std < 1e-6, 1.0, std)
    return CifarArrays(
        x_train=((x_train - mean) / safe_std).astype(np.float32),
        y_train=arrays.y_train.astype(np.int32),
        x_test=((x_test - mean) / safe_std).astype(np.float32),
        y_test=arrays.y_test.astype(np.int32),
        source=arrays.source,
        real_cifar=arrays.real_cifar,
    )


def stream_indices(labels: np.ndarray, steps: int, regime: str, seed: int) -> np.ndarray:
    """Return deterministic online stream indices for one regime."""
    if regime == "iid":
        rng = np.random.default_rng(seed)
        return rng.integers(0, labels.shape[0], size=steps, dtype=np.int32)
    if regime == "class_blocked":
        blocks: list[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for cls in range(N_CLASSES):
            cls_idx = np.flatnonzero(labels == cls).astype(np.int32)
            if cls_idx.shape[0] == 0:
                continue
            rng.shuffle(cls_idx)
            blocks.append(cls_idx)
        if not blocks:
            raise ValueError("class_blocked regime requires at least one class")
        ordered = np.concatenate(blocks)
        repeats = int(math.ceil(steps / ordered.shape[0]))
        return np.tile(ordered, repeats)[:steps].astype(np.int32)
    raise ValueError(f"unknown regime {regime!r}")


def make_upgd() -> UPGDLearner:
    """Return the promoted Step 2 non-router target-structure UPGD learner."""
    return UPGDLearner.step2_default(n_heads=N_CLASSES)


def make_step2_hybrid(feature_dim: int) -> FixedFeatureLearnerAdapter:
    """Return the packaged single Step 2 UPGD-memory trace learner."""
    return FixedFeatureLearnerAdapter(
        make_step2_hybrid_learner(
            Step2HybridConfig(
                feature_dim=feature_dim,
                n_heads=N_CLASSES,
                hidden_sizes=(64,),
                readout_mode="softmax_ce",
            )
        ),
        feature_dim=feature_dim,
    )


def make_prototype_memory(
    feature_dim: int,
    slots_per_class: int,
) -> FixedPrototypeMemoryAdapter:
    """Return a standalone fixed-budget prototype-memory block."""
    return FixedPrototypeMemoryAdapter(
        PrototypeMemoryLearner(
            PrototypeMemoryConfig(
                feature_dim=feature_dim,
                n_classes=N_CLASSES,
                slots_per_class=slots_per_class,
                update_rate=0.3,
                novelty_threshold=0.08,
                bandwidth=0.01,
            )
        ),
        feature_dim=feature_dim,
    )


def sharpen_if_confident(
    predictions: jax.Array,
    threshold: jax.Array | float = 0.10,
    blend: jax.Array | float = 1.0,
) -> jax.Array:
    """Blend confident predictions toward their top-1 one-hot class."""
    top_values, top_indices = jax.lax.top_k(predictions, 2)
    margin = top_values[0] - top_values[1]
    one_hot_prediction = jax.nn.one_hot(top_indices[0], predictions.shape[0])
    blend_j = jnp.asarray(blend, dtype=jnp.float32)
    sharpened = (1.0 - blend_j) * predictions + blend_j * one_hot_prediction
    return jnp.where(
        (blend_j > 0.0) & (margin >= jnp.asarray(threshold, dtype=jnp.float32)),
        sharpened,
        predictions,
    )


def dense_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Dense one-hot MSE."""
    return jnp.mean(jnp.square(predictions - targets))


def hysteretic_gate(
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


@dataclass(frozen=True)
class SharpenedVectorLearner:
    """Causal confidence-sharpened deployment wrapper."""

    learner: VectorLearner
    threshold: float = 0.10
    blend: float = 1.0

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self.learner.n_heads

    def init(self, feature_dim: int, key: jax.Array) -> Any:
        """Initialize the wrapped learner."""
        return self.learner.init(feature_dim, key)

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        """Predict with sharpened deployment readout."""
        return sharpen_if_confident(
            self.learner.predict(state, observation),
            self.threshold,
            self.blend,
        )

    def update(self, state: Any, observation: jax.Array, targets: jax.Array) -> Any:
        """Update the base learner and return sharpened scoring predictions."""
        result = self.learner.update(state, observation, targets)
        return GenericUpdateResult(
            state=result.state,
            predictions=sharpen_if_confident(
                result.predictions,
                self.threshold,
                self.blend,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the wrapper config."""
        return {
            "type": "SharpenedVectorLearner",
            "threshold": self.threshold,
            "blend": self.blend,
            "base": self.learner.to_config(),
        }


@dataclass(frozen=True)
class AdaptiveSharpenedVectorLearner:
    """Causal utility gate between raw and sharpened deployment readouts."""

    learner: VectorLearner
    threshold: float = 0.10
    blend: float = 1.0
    utility_decay: float = 0.99
    utility_off_threshold: float = 0.0001
    utility_on_threshold: float = 0.0002

    @property
    def n_heads(self) -> int:
        """Number of output heads."""
        return self.learner.n_heads

    def init(self, feature_dim: int, key: jax.Array) -> AdaptiveSharpenedState:
        """Initialize the wrapped learner and readout gate."""
        return AdaptiveSharpenedState(
            base_state=self.learner.init(feature_dim, key),
            sharp_advantage=jnp.array(0.0, dtype=jnp.float32),
            sharp_gate=jnp.array(0.0, dtype=jnp.float32),
        )

    def _mix(self, raw: jax.Array, sharp: jax.Array, gate: jax.Array) -> jax.Array:
        """Mix raw and sharpened readouts according to the scalar gate."""
        return (1.0 - gate) * raw + gate * sharp

    def predict(
        self,
        state: AdaptiveSharpenedState,
        observation: jax.Array,
    ) -> jax.Array:
        """Predict from the currently selected deployment readout."""
        raw = self.learner.predict(state.base_state, observation)
        sharp = sharpen_if_confident(raw, self.threshold, self.blend)
        return self._mix(raw, sharp, state.sharp_gate)

    def update(
        self,
        state: AdaptiveSharpenedState,
        observation: jax.Array,
        targets: jax.Array,
    ) -> GenericUpdateResult:
        """Update base learner and causal readout utility."""
        result = self.learner.update(state.base_state, observation, targets)
        raw = result.predictions
        sharp = sharpen_if_confident(raw, self.threshold, self.blend)
        next_gate = hysteretic_gate(
            state.sharp_gate,
            state.sharp_advantage,
            jnp.asarray(self.utility_off_threshold, dtype=jnp.float32),
            jnp.asarray(self.utility_on_threshold, dtype=jnp.float32),
        )
        prediction = self._mix(raw, sharp, next_gate)
        raw_loss = dense_mse(raw, targets)
        sharp_loss = dense_mse(sharp, targets)
        next_advantage = (
            jnp.asarray(self.utility_decay, dtype=jnp.float32) * state.sharp_advantage
            + (1.0 - jnp.asarray(self.utility_decay, dtype=jnp.float32))
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
        """Serialize the wrapper config."""
        return {
            "type": "AdaptiveSharpenedVectorLearner",
            "threshold": self.threshold,
            "blend": self.blend,
            "utility_decay": self.utility_decay,
            "utility_off_threshold": self.utility_off_threshold,
            "utility_on_threshold": self.utility_on_threshold,
            "base": self.learner.to_config(),
        }


def make_mlp(hidden_sizes: tuple[int, ...]) -> MultiHeadMLPLearner:
    """Return a fair LMS/ObGD MLP comparator."""
    return MultiHeadMLPLearner(
        n_heads=N_CLASSES,
        hidden_sizes=hidden_sizes,
        optimizer=LMS(step_size=0.03),
        bounder=ObGDBounding(kappa=0.5),
        sparsity=0.0 if hidden_sizes == () else 0.5,
        use_layer_norm=hidden_sizes != (),
    )


def method_factories(
    feature_dim: int,
    config: RunConfig,
) -> dict[str, Callable[[], VectorLearner]]:
    """Return benchmark methods."""
    factories: dict[str, Callable[[], VectorLearner]] = {
        PRIMARY_METHOD: lambda: make_step2_hybrid(feature_dim),
        "upgd_step2_default": make_upgd,
        "mlp_h32": lambda: make_mlp((32,)),
        "mlp_h64": lambda: make_mlp((64,)),
    }
    if config.include_wide_mlp:
        factories.update(
            {
                "mlp_h128": lambda: make_mlp((128,)),
                "mlp_h256": lambda: make_mlp((256,)),
                "mlp_h128_128": lambda: make_mlp((128, 128)),
            }
        )
    if config.include_prototype_memory:
        factories.update(
            {
                "proto_mem_s20": lambda: make_prototype_memory(feature_dim, 20),
                "proto_mem_s32": lambda: make_prototype_memory(feature_dim, 32),
            }
        )
    if config.include_primary_sharpened:
        factories[PRIMARY_SHARP_METHOD] = lambda: SharpenedVectorLearner(
            make_step2_hybrid(feature_dim)
        )
    if config.include_adaptive_primary_sharpened:
        factories[PRIMARY_ADAPTIVE_SHARP_METHOD] = (
            lambda: AdaptiveSharpenedVectorLearner(make_step2_hybrid(feature_dim))
        )
    if config.include_sharpened_mlp:
        factories["mlp_h32_sharp"] = lambda: SharpenedVectorLearner(make_mlp((32,)))
        factories["mlp_h64_sharp"] = lambda: SharpenedVectorLearner(make_mlp((64,)))
    if config.only_methods is not None:
        unknown = [name for name in config.only_methods if name not in factories]
        if unknown:
            raise ValueError(
                "--only-methods requested unavailable methods: "
                + ", ".join(sorted(unknown))
            )
        factories = {name: factories[name] for name in config.only_methods}
    return factories


def run_online_classifier(
    learner: VectorLearner,
    observations: jax.Array,
    labels: jax.Array,
    key: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run one online supervised stream and return final state plus metrics."""
    targets = jnp.asarray(one_hot(np.asarray(labels)))
    state = learner.init(int(observations.shape[1]), key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, label = inputs
        result = learner.update(carry, obs, target)
        predictions = result.predictions
        mse = jnp.mean((predictions - target) ** 2)
        correct = (jnp.argmax(predictions) == label).astype(jnp.float32)
        return result.state, jnp.stack([mse, correct])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def evaluate_classifier(
    learner: VectorLearner,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate held-out one-hot MSE and accuracy."""
    observations = jnp.asarray(x_test)
    labels = jnp.asarray(y_test)
    targets = jnp.asarray(one_hot(y_test))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def final_window(values: np.ndarray, window: int) -> float:
    """Return final-window mean."""
    return float(np.mean(values[-min(window, values.shape[0]) :]))


def stderr(values: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Return standard error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def mean_row(values: Sequence[float] | np.ndarray[Any, Any]) -> dict[str, Any]:
    """Summarize a metric over seeds."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "stderr": stderr(arr), "per_seed": arr.tolist()}


def paired_lower(
    candidate: Sequence[float] | np.ndarray[Any, Any],
    baseline: Sequence[float] | np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Summarize lower-is-better paired deltas; positive favors candidate."""
    diff = np.asarray(baseline, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_stderr": stderr(diff),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "diffs": diff.tolist(),
    }


def paired_higher(
    candidate: Sequence[float] | np.ndarray[Any, Any],
    baseline: Sequence[float] | np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Summarize higher-is-better paired deltas; positive favors candidate."""
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_stderr": stderr(diff),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "diffs": diff.tolist(),
    }


def run_experiment(config: RunConfig) -> dict[str, Any]:
    """Run all regimes, seeds, and methods."""
    arrays = load_cifar_or_smoke(
        data_dir=Path(config.data_dir),
        allow_download=config.allow_download,
        max_train=config.max_train,
        max_test=config.max_test,
        seed=0,
    )
    factories = method_factories(int(arrays.x_train.shape[1]), config)
    raw: dict[str, dict[str, list[dict[str, Any]]]] = {
        regime: {name: [] for name in factories} for regime in config.regimes
    }

    for regime in config.regimes:
        for seed in range(config.n_seeds):
            idx = stream_indices(arrays.y_train, config.steps, regime, seed)
            observations = jnp.asarray(arrays.x_train[idx])
            labels = jnp.asarray(arrays.y_train[idx])
            for method_name, factory in factories.items():
                learner = factory()
                state, online = run_online_classifier(
                    learner=learner,
                    observations=observations,
                    labels=labels,
                    key=jr.key(seed),
                )
                held_out = evaluate_classifier(learner, state, arrays.x_test, arrays.y_test)
                raw[regime][method_name].append(
                    {
                        "seed": seed,
                        "final_window_mse": final_window(online[:, 0], config.final_window),
                        "final_window_accuracy": final_window(
                            online[:, 1],
                            config.final_window,
                        ),
                        **held_out,
                    }
                )

    summary: dict[str, Any] = {}
    for regime, per_method in raw.items():
        summary[regime] = {}
        for method_name, rows in per_method.items():
            summary[regime][method_name] = {
                metric: mean_row([float(row[metric]) for row in rows])
                for metric in (
                    "final_window_mse",
                    "final_window_accuracy",
                    "test_mse",
                    "test_accuracy",
                )
            }

        mlp_names = [name for name in factories if name.startswith("mlp_")]
        candidate_names = [name for name in factories if not name.startswith("mlp_")]
        if not mlp_names:
            raise ValueError("at least one MLP comparator is required")
        if not candidate_names:
            raise ValueError("at least one non-MLP candidate is required")
        best_mlp_mse = min(
            mlp_names,
            key=lambda name: summary[regime][name]["final_window_mse"]["mean"],
        )
        best_mlp_final_acc = max(
            mlp_names,
            key=lambda name: summary[regime][name]["final_window_accuracy"]["mean"],
        )
        best_mlp_test_mse = min(
            mlp_names,
            key=lambda name: summary[regime][name]["test_mse"]["mean"],
        )
        best_mlp_acc = max(
            mlp_names,
            key=lambda name: summary[regime][name]["test_accuracy"]["mean"],
        )
        paired_by_method = {}
        for candidate_name in candidate_names:
            candidate_rows = per_method[candidate_name]
            paired_by_method[candidate_name] = {
                "best_mlp_final_window_mse": best_mlp_mse,
                "best_mlp_final_window_accuracy": best_mlp_final_acc,
                "best_mlp_test_mse": best_mlp_test_mse,
                "best_mlp_test_accuracy": best_mlp_acc,
                "final_window_mse": paired_lower(
                    [float(row["final_window_mse"]) for row in candidate_rows],
                    [
                        float(row["final_window_mse"])
                        for row in per_method[best_mlp_mse]
                    ],
                ),
                "final_window_accuracy": paired_higher(
                    [
                        float(row["final_window_accuracy"])
                        for row in candidate_rows
                    ],
                    [
                        float(row["final_window_accuracy"])
                        for row in per_method[best_mlp_final_acc]
                    ],
                ),
                "test_mse": paired_lower(
                    [float(row["test_mse"]) for row in candidate_rows],
                    [float(row["test_mse"]) for row in per_method[best_mlp_test_mse]],
                ),
                "test_accuracy": paired_higher(
                    [float(row["test_accuracy"]) for row in candidate_rows],
                    [float(row["test_accuracy"]) for row in per_method[best_mlp_acc]],
                ),
            }
        primary_method = (
            PRIMARY_METHOD if PRIMARY_METHOD in paired_by_method else candidate_names[0]
        )
        summary[regime]["primary_method"] = primary_method
        summary[regime]["paired_vs_best_mlp"] = paired_by_method[primary_method]
        summary[regime]["paired_vs_best_mlp_by_method"] = paired_by_method

    return {
        "config": asdict(config),
        "dataset": {
            "source": arrays.source,
            "real_cifar": arrays.real_cifar,
            "train_shape": list(arrays.x_train.shape),
            "test_shape": list(arrays.x_test.shape),
        },
        "primary_method": PRIMARY_METHOD
        if PRIMARY_METHOD in factories
        else next(name for name in factories if not name.startswith("mlp_")),
        "methods": {name: factory().to_config() for name, factory in factories.items()},
        "raw": raw,
        "summary": summary,
    }


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format a summary metric cell."""
    data = row[metric]
    return f"{data['mean']:.6f} +/- {data['stderr']:.6f}"


def write_note(results: dict[str, Any], path: Path) -> None:
    """Write a compact research note."""
    lines = [
        "# Step 2 CIFAR Stream Benchmark",
        "",
        "This note records the external CIFAR-style supervised stream probe for the "
        "current packaged single UPGD-memory trace learner.",
        "",
        f"Positive paired differences favor `{results['primary_method']}` in the "
        "primary comparison. For MSE, the difference is best MLP minus candidate; "
        "for accuracy, it is candidate minus best MLP.",
        "",
        "## Dataset",
        "",
        f"- Source: `{results['dataset']['source']}`",
        f"- Real CIFAR-10 evidence: `{results['dataset']['real_cifar']}`",
        f"- Train shape: `{results['dataset']['train_shape']}`",
        f"- Test shape: `{results['dataset']['test_shape']}`",
        "",
    ]
    if not results["dataset"]["real_cifar"]:
        lines.extend(
            [
                "This is not real CIFAR-10 evidence. The runner used the "
                "deterministic synthetic smoke fallback because local CIFAR-10 "
                "was unavailable without optional dependency/data setup.",
                "",
            ]
        )
    lines.extend(["## Results", ""])
    for regime, per_method in results["summary"].items():
        lines.extend(
            [
                f"### `{regime}`",
                "",
                "| Method | Final-window MSE | Final-window accuracy | "
                "Held-out MSE | Held-out accuracy |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in results["methods"]:
            row = per_method[method]
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_mse')} | {metric_cell(row, 'test_accuracy')} |"
            )
        paired = per_method["paired_vs_best_mlp"]
        lines.extend(
            [
                "",
                f"- Primary candidate: `{results['primary_method']}`",
                f"- Best MLP for final-window MSE: `{paired['best_mlp_final_window_mse']}`",
                f"- Best MLP for held-out accuracy: `{paired['best_mlp_test_accuracy']}`",
                "- `final_window_mse` primary-vs-best-MLP diff: "
                f"{paired['final_window_mse']['diff_mean']:+.6f} +/- "
                f"{paired['final_window_mse']['diff_stderr']:.6f}; "
                f"wins/losses/ties {paired['final_window_mse']['wins']}/"
                f"{paired['final_window_mse']['losses']}/"
                f"{paired['final_window_mse']['ties']}.",
                "- `test_accuracy` primary-vs-best-MLP diff: "
                f"{paired['test_accuracy']['diff_mean']:+.6f} +/- "
                f"{paired['test_accuracy']['diff_stderr']:.6f}; "
                f"wins/losses/ties {paired['test_accuracy']['wins']}/"
                f"{paired['test_accuracy']['losses']}/"
                f"{paired['test_accuracy']['ties']}.",
                "",
            ]
        )
        additional = {
            name: row
            for name, row in per_method["paired_vs_best_mlp_by_method"].items()
            if name != results["primary_method"]
        }
        if additional:
            lines.extend(["Additional candidate comparisons:", ""])
            for method, candidate_paired in additional.items():
                lines.extend(
                    [
                        f"- `{method}` final-window MSE diff vs best MLP: "
                        f"{candidate_paired['final_window_mse']['diff_mean']:+.6f} +/- "
                        f"{candidate_paired['final_window_mse']['diff_stderr']:.6f}; "
                        f"wins/losses/ties {candidate_paired['final_window_mse']['wins']}/"
                        f"{candidate_paired['final_window_mse']['losses']}/"
                        f"{candidate_paired['final_window_mse']['ties']}.",
                        f"- `{method}` held-out accuracy diff vs best MLP: "
                        f"{candidate_paired['test_accuracy']['diff_mean']:+.6f} +/- "
                        f"{candidate_paired['test_accuracy']['diff_stderr']:.6f}; "
                        f"wins/losses/ties {candidate_paired['test_accuracy']['wins']}/"
                        f"{candidate_paired['test_accuracy']['losses']}/"
                        f"{candidate_paired['test_accuracy']['ties']}.",
                    ]
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Synthetic smoke results only validate wiring, reproducibility, optional "
            "dependency behavior, and metric accounting. A stronger CIFAR claim "
            "requires real CIFAR-10 runs with `--allow-download`, multiple seeds, "
            "larger streams, and no test-set tuning.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=40)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["iid", "class_blocked"],
        choices=["iid", "class_blocked"],
    )
    parser.add_argument("--include-primary-sharpened", action="store_true")
    parser.add_argument("--include-adaptive-primary-sharpened", action="store_true")
    parser.add_argument("--include-sharpened-mlp", action="store_true")
    parser.add_argument("--include-prototype-memory", action="store_true")
    parser.add_argument("--include-wide-mlp", action="store_true")
    parser.add_argument(
        "--only-methods",
        type=str,
        default=None,
        help="Comma-separated method subset to run after optional methods are enabled.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", type=str, default="cifar_stream")
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark from the command line."""
    args = parse_args()
    config = RunConfig(
        steps=args.steps,
        n_seeds=args.n_seeds,
        final_window=args.final_window,
        max_train=args.max_train,
        max_test=args.max_test,
        data_dir=args.data_dir,
        allow_download=args.allow_download,
        regimes=tuple(args.regimes),
        include_primary_sharpened=args.include_primary_sharpened,
        include_adaptive_primary_sharpened=args.include_adaptive_primary_sharpened,
        include_sharpened_mlp=args.include_sharpened_mlp,
        include_prototype_memory=args.include_prototype_memory,
        include_wide_mlp=args.include_wide_mlp,
        only_methods=tuple(
            name.strip() for name in args.only_methods.split(",") if name.strip()
        )
        if args.only_methods
        else None,
    )
    results = run_experiment(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.result_prefix}_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_note(results, args.note_path)
    print(json.dumps({"result_path": str(result_path), "dataset": results["dataset"]}, indent=2))


if __name__ == "__main__":
    main()

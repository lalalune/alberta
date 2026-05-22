#!/usr/bin/env python3
# mypy: disable-error-code="call-arg"
"""D22: JAX prototype memory and UPGD+memory hybrids on OPMNIST.

This runner productionizes the D20 geometry enough to test it fairly with the
UPGD Step 2 candidate: the prototype memory is fixed-budget, JAX-native, and
scan-compatible.  It compares fair MLP baselines, standalone UPGD, standalone
prototype memory, and simple prediction-uncertainty gates between UPGD and the
memory head.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import d18_opmnist_bridge as bridge  # noqa: E402
from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    aggregate_records,
    make_mlp,
    run_mlp_stream,
    summarize_prequential,
)
from d21_upgd_opmnist_efficiency import (  # noqa: E402
    evaluate_upgd_views,
    summarize_predictions,
    trainable_param_count,
    tree_float_size,
)

from alberta_framework.core.prototype_memory import (  # noqa: E402
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    PrototypeMemoryState,
)
from alberta_framework.core.upgd import UPGDLearner, UPGDState  # noqa: E402
from alberta_framework.core.upgd_memory import (  # noqa: E402
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    UPGDMemoryState,
    run_upgd_memory_arrays,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d22_upgd_prototype_hybrid")
DEFAULT_RESULT_PREFIX = "d22_upgd_prototype_hybrid"
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d22_upgd_prototype_hybrid.md"
)
DATASET_NAME = "opmnist_bridge"
N_CLASSES = bridge.N_CLASSES


@dataclass(frozen=True)
class HybridVariant:
    """One UPGD+prototype hybrid configuration."""

    name: str
    hidden_size: int = 64
    slots_per_class: int = 20
    gate_mode: str = "memory_confidence"
    gate_threshold: float = 0.65
    gate_margin: float = 0.0
    gate_temperature: float = 0.05
    memory_weight: float = 0.95


@dataclass(frozen=True)
class PrototypeVariant:
    """One standalone prototype-memory configuration."""

    name: str
    slots_per_class: int = 20


@dataclass(frozen=True)
class CoreHybridVariant:
    """One packaged UPGD-memory learner configuration."""

    name: str
    hidden_size: int = 64
    slots_per_class: int = 10
    initial_memory_logit: float = -2.0
    memory_logit_step_size: float = 0.25
    confidence_logit_scale: float = 2.0
    reliability_logit_scale: float = 8.0
    novelty_adaptation_rate: float = 0.02
    target_allocation_rate: float = 0.02
    target_trace_blend_scale: float = 0.0
    target_trace_pressure_threshold: float = 0.75


def default_hybrid_variants() -> tuple[HybridVariant, ...]:
    """Return compact D22 hybrid variants."""
    return (
        HybridVariant("hybrid_h64_s8_memconf", slots_per_class=8),
        HybridVariant("hybrid_h64_s20_memconf", slots_per_class=20),
        HybridVariant(
            "hybrid_h64_s20_advantage",
            slots_per_class=20,
            gate_mode="confidence_advantage",
            gate_margin=0.0,
        ),
        HybridVariant(
            "hybrid_h64_s20_uncertain",
            slots_per_class=20,
            gate_mode="upgd_uncertainty",
            gate_threshold=0.55,
        ),
        HybridVariant(
            "hybrid_h64_s20_memory95",
            slots_per_class=20,
            gate_mode="memory_weight",
            memory_weight=0.95,
        ),
    )


def default_prototype_variants() -> tuple[PrototypeVariant, ...]:
    """Return standalone prototype memory variants."""
    return (
        PrototypeVariant("proto_s8", slots_per_class=8),
        PrototypeVariant("proto_s20", slots_per_class=20),
    )


def default_core_hybrid_variants() -> tuple[CoreHybridVariant, ...]:
    """Return packaged UPGD-memory variants."""
    return (
        CoreHybridVariant("core_upgdmem_h64_s10"),
        CoreHybridVariant("core_upgdmem_h64_s20", slots_per_class=20),
        CoreHybridVariant(
            "core_upgdmem_h64_s10_alloc05",
            target_allocation_rate=0.05,
        ),
        CoreHybridVariant(
            "core_upgdmem_h64_s20_alloc18",
            slots_per_class=20,
            target_allocation_rate=0.18,
        ),
        CoreHybridVariant(
            "core_upgdmem_h64_s20_alloc18_mem0",
            slots_per_class=20,
            initial_memory_logit=0.0,
            target_allocation_rate=0.18,
        ),
        CoreHybridVariant(
            "core_upgdmem_h64_s20_alloc18_mem0_trace80_thr50",
            slots_per_class=20,
            initial_memory_logit=0.0,
            target_allocation_rate=0.18,
            target_trace_blend_scale=0.8,
            target_trace_pressure_threshold=0.5,
        ),
    )


def make_memory(slots_per_class: int, feature_dim: int) -> PrototypeMemoryLearner:
    """Create the D20-equivalent fixed-budget prototype memory."""
    return PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=feature_dim,
            n_classes=N_CLASSES,
            slots_per_class=slots_per_class,
            update_rate=0.3,
            novelty_threshold=0.08,
            bandwidth=0.01,
        )
    )


def make_upgd(hidden_size: int) -> UPGDLearner:
    """Create the promoted UPGD classification learner."""
    return UPGDLearner.step2_default(
        n_heads=N_CLASSES,
        hidden_sizes=(hidden_size,),
        readout_mode="softmax_ce",
    )


def make_core_hybrid(
    variant: CoreHybridVariant,
    feature_dim: int,
) -> UPGDMemoryLearner:
    """Create one packaged UPGD-memory learner."""
    return UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=feature_dim,
            n_heads=N_CLASSES,
            hidden_sizes=(variant.hidden_size,),
            readout_mode="softmax_ce",
            slots_per_class=variant.slots_per_class,
            initial_memory_logit=variant.initial_memory_logit,
            memory_logit_step_size=variant.memory_logit_step_size,
            confidence_logit_scale=variant.confidence_logit_scale,
            reliability_logit_scale=variant.reliability_logit_scale,
            novelty_adaptation_rate=variant.novelty_adaptation_rate,
            target_allocation_rate=variant.target_allocation_rate,
            target_trace_blend_scale=variant.target_trace_blend_scale,
            target_trace_pressure_threshold=variant.target_trace_pressure_threshold,
        )
    )


def _hybrid_gate(
    variant: HybridVariant,
    upgd_prediction: jax.Array,
    memory_prediction: jax.Array,
    memory_state: PrototypeMemoryState,
) -> jax.Array:
    active = (jnp.sum(memory_state.counts > 0.0) > 0).astype(jnp.float32)
    upgd_conf = jnp.max(upgd_prediction)
    memory_conf = jnp.max(memory_prediction)
    temp = jnp.asarray(max(variant.gate_temperature, 1e-6), dtype=jnp.float32)
    if variant.gate_mode == "memory_confidence":
        gate = jax.nn.sigmoid((memory_conf - variant.gate_threshold) / temp)
    elif variant.gate_mode == "confidence_advantage":
        gate = jax.nn.sigmoid(
            (memory_conf - upgd_conf - variant.gate_margin) / temp
        )
    elif variant.gate_mode == "upgd_uncertainty":
        gate = jax.nn.sigmoid((variant.gate_threshold - upgd_conf) / temp)
        gate = gate * jax.nn.sigmoid((memory_conf - 0.55) / temp)
    elif variant.gate_mode == "memory_weight":
        gate = jnp.asarray(variant.memory_weight, dtype=jnp.float32)
    else:
        raise ValueError(f"unknown gate_mode: {variant.gate_mode!r}")
    return jnp.clip(active * gate, 0.0, 1.0)


def _hybrid_prediction(
    variant: HybridVariant,
    upgd_prediction: jax.Array,
    memory_prediction: jax.Array,
    memory_state: PrototypeMemoryState,
) -> tuple[jax.Array, jax.Array]:
    gate = _hybrid_gate(variant, upgd_prediction, memory_prediction, memory_state)
    prediction = (1.0 - gate) * upgd_prediction + gate * memory_prediction
    prediction = jnp.maximum(prediction, 0.0)
    prediction = prediction / jnp.maximum(jnp.sum(prediction), 1e-12)
    return prediction, gate


def run_prototype_stream(
    learner: PrototypeMemoryLearner,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[PrototypeMemoryState, np.ndarray, np.ndarray, float]:
    """Run standalone prototype memory with one JIT scan."""
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)
    state = learner.init()

    @jax.jit  # type: ignore[untyped-decorator]
    def run(
        initial_state: PrototypeMemoryState,
    ) -> tuple[PrototypeMemoryState, jax.Array, jax.Array]:
        def step_fn(
            carry: PrototypeMemoryState,
            batch: tuple[jax.Array, jax.Array],
        ) -> tuple[PrototypeMemoryState, tuple[jax.Array, jax.Array]]:
            observation, target = batch
            result = learner.update(carry, observation, target)
            return result.state, (result.predictions, result.metrics)

        final_state, (predictions, metrics) = jax.lax.scan(
            step_fn,
            initial_state,
            (observations_jax, targets_jax),
        )
        return final_state, predictions, metrics

    t0 = time.time()
    final_state, predictions, metrics = run(state)
    predictions.block_until_ready()
    return final_state, np.asarray(predictions), np.asarray(metrics), time.time() - t0


def evaluate_prototype_views(
    learner: PrototypeMemoryLearner,
    state: PrototypeMemoryState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate prototype memory across held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        predictions = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        predictions.block_until_ready()
        preds = np.asarray(predictions)
        mse_values.append(float(np.mean((preds - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(preds, axis=1) == labels)))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
    }


def run_hybrid_stream(
    variant: HybridVariant,
    upgd: UPGDLearner,
    memory: PrototypeMemoryLearner,
    observations: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
) -> tuple[UPGDState, PrototypeMemoryState, np.ndarray, np.ndarray, float]:
    """Run UPGD+prototype memory with one JIT scan."""
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)
    upgd_state = upgd.init(observations.shape[1], key)
    memory_state = memory.init()

    @jax.jit  # type: ignore[untyped-decorator]
    def run(
        initial_upgd_state: UPGDState,
        initial_memory_state: PrototypeMemoryState,
    ) -> tuple[UPGDState, PrototypeMemoryState, jax.Array, jax.Array]:
        def step_fn(
            carry: tuple[UPGDState, PrototypeMemoryState],
            batch: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[UPGDState, PrototypeMemoryState], tuple[jax.Array, jax.Array]]:
            upgd_state_inner, memory_state_inner = carry
            observation, target = batch
            upgd_prediction = upgd.predict(upgd_state_inner, observation)
            memory_prediction = memory.predict(memory_state_inner, observation)
            prediction, gate = _hybrid_prediction(
                variant,
                upgd_prediction,
                memory_prediction,
                memory_state_inner,
            )
            upgd_result = upgd.update(upgd_state_inner, observation, target)
            memory_result = memory.update(memory_state_inner, observation, target)
            safe_target = jnp.where(jnp.isfinite(target), target, 0.0)
            mse = jnp.mean((prediction - safe_target) ** 2)
            correct = (jnp.argmax(prediction) == jnp.argmax(safe_target)).astype(
                jnp.float32
            )
            metrics = jnp.asarray(
                [
                    mse,
                    correct,
                    gate,
                    jnp.max(upgd_prediction),
                    jnp.max(memory_prediction),
                    jnp.sum(memory_state_inner.counts > 0.0).astype(jnp.float32),
                ],
                dtype=jnp.float32,
            )
            return (
                upgd_result.state,
                memory_result.state,
            ), (prediction, metrics)

        (final_upgd_state, final_memory_state), (predictions, metrics) = jax.lax.scan(
            step_fn,
            (initial_upgd_state, initial_memory_state),
            (observations_jax, targets_jax),
        )
        return final_upgd_state, final_memory_state, predictions, metrics

    t0 = time.time()
    final_upgd, final_memory, predictions, metrics = run(upgd_state, memory_state)
    predictions.block_until_ready()
    return (
        final_upgd,
        final_memory,
        np.asarray(predictions),
        np.asarray(metrics),
        time.time() - t0,
    )


def evaluate_hybrid_views(
    variant: HybridVariant,
    upgd: UPGDLearner,
    upgd_state: UPGDState,
    memory: PrototypeMemoryLearner,
    memory_state: PrototypeMemoryState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a hybrid across held-out permutation views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    gate_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))

        def predict_one(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
            upgd_prediction = upgd.predict(upgd_state, obs)
            memory_prediction = memory.predict(memory_state, obs)
            return _hybrid_prediction(
                variant,
                upgd_prediction,
                memory_prediction,
                memory_state,
            )

        predictions, gates = jax.vmap(predict_one)(observations)
        predictions.block_until_ready()
        preds = np.asarray(predictions)
        mse_values.append(float(np.mean((preds - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(preds, axis=1) == labels)))
        gate_values.append(float(np.mean(np.asarray(gates))))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
        "mean_eval_gate": float(np.mean(gate_values)),
    }


def run_core_hybrid_stream(
    learner: UPGDMemoryLearner,
    observations: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
) -> tuple[UPGDMemoryState, np.ndarray, np.ndarray, float]:
    """Run the packaged UPGD-memory learner with one JIT scan."""
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    targets_jax = jnp.asarray(targets, dtype=jnp.float32)
    state = learner.init(key)

    @jax.jit  # type: ignore[untyped-decorator]
    def run(initial_state: UPGDMemoryState) -> tuple[UPGDMemoryState, jax.Array, jax.Array]:
        result = run_upgd_memory_arrays(
            learner,
            initial_state,
            observations_jax,
            targets_jax,
        )
        return result.state, result.predictions, result.metrics

    t0 = time.time()
    final_state, predictions, metrics = run(state)
    predictions.block_until_ready()
    return final_state, np.asarray(predictions), np.asarray(metrics), time.time() - t0


def evaluate_core_hybrid_views(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    test_views: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the packaged UPGD-memory learner across held-out views."""
    targets = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    labels = np.asarray(y_test, dtype=np.int32)
    mse_values: list[float] = []
    accuracy_values: list[float] = []
    gate_values: list[float] = []
    for view in test_views:
        observations = jnp.asarray(view.astype(np.float32))
        predictions = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
        predictions.block_until_ready()
        preds = np.asarray(predictions)
        mse_values.append(float(np.mean((preds - targets) ** 2)))
        accuracy_values.append(float(np.mean(np.argmax(preds, axis=1) == labels)))
        memory_predictions = jax.vmap(
            lambda obs: learner.memory.predict(state.memory_state, obs)
        )(observations)
        upgd_predictions = jax.vmap(
            lambda obs: learner.upgd.predict(state.upgd_state, obs)
        )(observations)
        gates = jax.vmap(
            lambda upgd_pred, mem_pred: learner._blend_gate(state, upgd_pred, mem_pred)
        )(upgd_predictions, memory_predictions)
        gate_values.append(float(np.mean(np.asarray(gates))))
    mean_mse = float(np.mean(mse_values))
    mean_accuracy = float(np.mean(accuracy_values))
    return {
        "test_mse": mean_mse,
        "test_accuracy": mean_accuracy,
        "deployment_test_mse": mean_mse,
        "deployment_test_accuracy": mean_accuracy,
        "mean_eval_gate": float(np.mean(gate_values)),
    }


def parse_optional_positive_int(value: str) -> int | None:
    """Parse bridge-compatible optional integer values."""
    parsed = bridge.parse_optional_positive_int(value)
    if parsed is None:
        return None
    return int(parsed)


def run_one_seed(
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one paired OPMNIST seed."""
    dataset, stream = bridge.make_stream(seed, args)
    observations = np.asarray(stream.observations, dtype=np.float32)
    targets = np.asarray(stream.targets, dtype=np.float32)
    labels = np.asarray(stream.labels, dtype=np.int32)
    methods: dict[str, dict[str, float]] = {}

    for method in MLP_METHODS:
        print(f"opmnist seed={seed}: running {method}")
        mlp = make_mlp(
            method=method,
            n_heads=N_CLASSES,
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
            mlp,
            observations,
            targets,
            jr.key(seed + 80_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method].update(
            bridge.evaluate_mlp_classifier_views(
                learner=mlp,
                state=state,
                test_views=stream.test_views,
                y_test=stream.test_labels,
                deployment_transform=args.mlp_deployment_transform,
            )
        )
        runtime = float(time.time() - t0)
        methods[method]["runtime_s"] = runtime
        methods[method]["steps_per_second"] = float(args.steps / runtime)

    upgd = make_upgd(args.upgd_hidden_size)
    print(f"opmnist seed={seed}: running upgd_h{args.upgd_hidden_size}_softmax_ce")
    upgd_state = upgd.init(observations.shape[1], jr.key(seed + 81_000))
    final_upgd, upgd_predictions, _upgd_metrics, runtime = d21_run_upgd_stream(
        upgd,
        upgd_state,
        observations,
        targets,
    )
    upgd_name = f"upgd_h{args.upgd_hidden_size}_softmax_ce"
    methods[upgd_name] = summarize_predictions(
        upgd_predictions,
        targets,
        labels,
        args.final_window,
    )
    methods[upgd_name].update(
        evaluate_upgd_views(upgd, final_upgd, stream.test_views, stream.test_labels)
    )
    methods[upgd_name].update(
        {
            "runtime_s": float(runtime),
            "steps_per_second": float(args.steps / runtime),
            "trainable_params": float(trainable_param_count(upgd_state)),
            "float_state_size": float(tree_float_size(upgd_state)),
        }
    )

    for proto_variant in default_prototype_variants():
        print(f"opmnist seed={seed}: running {proto_variant.name}")
        memory = make_memory(proto_variant.slots_per_class, observations.shape[1])
        state, predictions, metrics, runtime = run_prototype_stream(
            memory,
            observations,
            targets,
        )
        methods[proto_variant.name] = summarize_predictions(
            predictions,
            targets,
            labels,
            args.final_window,
        )
        methods[proto_variant.name].update(
            evaluate_prototype_views(
                memory,
                state,
                stream.test_views,
                stream.test_labels,
            )
        )
        methods[proto_variant.name].update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "active_prototypes": float(np.asarray(metrics)[-1, 3]),
                "float_state_size": float(tree_float_size(state)),
            }
        )

    for hybrid_variant in default_hybrid_variants():
        print(f"opmnist seed={seed}: running {hybrid_variant.name}")
        upgd = make_upgd(hybrid_variant.hidden_size)
        memory = make_memory(hybrid_variant.slots_per_class, observations.shape[1])
        initial_upgd = upgd.init(observations.shape[1], jr.key(seed + 82_000))
        (
            final_upgd_state,
            final_memory_state,
            predictions,
            metrics,
            runtime,
        ) = run_hybrid_stream(
            hybrid_variant,
            upgd,
            memory,
            observations,
            targets,
            jr.key(seed + 82_000 + hybrid_variant.slots_per_class),
        )
        entry = summarize_predictions(predictions, targets, labels, args.final_window)
        entry.update(
            evaluate_hybrid_views(
                hybrid_variant,
                upgd,
                final_upgd_state,
                memory,
                final_memory_state,
                stream.test_views,
                stream.test_labels,
            )
        )
        entry.update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "mean_train_gate": float(np.mean(metrics[:, 2])),
                "final_train_gate": float(np.mean(metrics[-args.final_window :, 2])),
                "active_prototypes": float(metrics[-1, 5]),
                "trainable_params": float(trainable_param_count(initial_upgd)),
                "float_state_size": float(
                    tree_float_size(initial_upgd) + tree_float_size(final_memory_state)
                ),
            }
        )
        methods[hybrid_variant.name] = entry

    for core_variant in default_core_hybrid_variants():
        print(f"opmnist seed={seed}: running {core_variant.name}")
        learner = make_core_hybrid(core_variant, observations.shape[1])
        initial_state = learner.init(jr.key(seed + 83_000 + core_variant.slots_per_class))
        final_state, predictions, metrics, runtime = run_core_hybrid_stream(
            learner,
            observations,
            targets,
            jr.key(seed + 84_000 + core_variant.slots_per_class),
        )
        entry = summarize_predictions(predictions, targets, labels, args.final_window)
        entry.update(
            evaluate_core_hybrid_views(
                learner,
                final_state,
                stream.test_views,
                stream.test_labels,
            )
        )
        entry.update(
            {
                "runtime_s": float(runtime),
                "steps_per_second": float(args.steps / runtime),
                "mean_train_gate": float(np.mean(metrics[:, 3])),
                "final_train_gate": float(np.mean(metrics[-args.final_window :, 3])),
                "active_prototypes": float(metrics[-1, 7]),
                "trainable_params": float(trainable_param_count(initial_state.upgd_state)),
                "float_state_size": float(tree_float_size(initial_state)),
            }
        )
        methods[core_variant.name] = entry

    meta = dict(dataset.metadata)
    meta.update(stream.metadata)
    return (
        {
            "dataset_name": DATASET_NAME,
            "seed": int(seed),
            "dataset": meta,
            "methods": methods,
        },
        meta,
    )


def d21_run_upgd_stream(
    learner: UPGDLearner,
    state: UPGDState,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[UPGDState, np.ndarray, np.ndarray, float]:
    """Local wrapper to keep the imported D21 function type precise."""
    from d21_upgd_opmnist_efficiency import run_upgd_stream

    final_state, predictions, metrics, runtime = run_upgd_stream(
        learner,
        state,
        observations,
        targets,
    )
    return final_state, predictions, metrics, runtime


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one metric cell."""
    if metric not in row:
        return ""
    value = row[metric]
    return f"{value['mean']:.6f} +/- {value['stderr']:.6f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary."""
    cfg = results["config"]
    aggregate = results["aggregate"][DATASET_NAME]
    comparisons = aggregate["comparisons"]
    lines = [
        "# D22 UPGD Prototype Hybrid OPMNIST",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seed(s), {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}, source "
            f"`{cfg['mnist_source']}`, {cfg['n_permutations']} permutation tasks."
        ),
        "",
        (
            "Standalone `proto_*` methods are the JAX fixed-budget D20 memory. "
            "`hybrid_*` methods update UPGD and memory every step and blend their "
            "predictions with a prediction-confidence gate."
        ),
        "",
        "| Method | Final MSE | Final Acc | Test MSE | Test Acc | Runtime s | "
        "Steps/s | Gate | Protos | Float State |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in aggregate.items():
        if method == "comparisons":
            continue
        lines.append(
            f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
            f"{metric_cell(row, 'final_window_accuracy')} | "
            f"{metric_cell(row, 'test_mse')} | "
            f"{metric_cell(row, 'test_accuracy')} | "
            f"{metric_cell(row, 'runtime_s')} | "
            f"{metric_cell(row, 'steps_per_second')} | "
            f"{metric_cell(row, 'mean_eval_gate')} | "
            f"{metric_cell(row, 'active_prototypes')} | "
            f"{metric_cell(row, 'float_state_size')} |"
        )
    lines.extend(["", "## Best Candidate vs Best MLP", ""])
    for metric in (
        "final_window_mse",
        "final_window_accuracy",
        "test_mse",
        "test_accuracy",
        "deployment_test_mse",
        "deployment_test_accuracy",
    ):
        if metric not in comparisons:
            continue
        comparison = comparisons[metric]["best_kernel_vs_best_mlp"]
        lines.append(f"- `{metric}`: {bridge.comparison_cell(comparison)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument(
        "--mnist-source",
        choices=(
            "auto",
            "openml",
            "torchvision",
            "sklearn_digits_28x28",
            "sklearn_digits_8x8",
        ),
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
    parser.add_argument("--max-train-examples", type=parse_optional_positive_int, default=1000)
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
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--mlp-deployment-transform",
        choices=bridge.DEPLOYMENT_TRANSFORMS,
        default="raw",
    )
    parser.add_argument("--upgd-hidden-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def apply_smoke(args: argparse.Namespace) -> None:
    """Apply quick smoke settings."""
    if not args.smoke:
        return
    args.steps = 120
    args.n_seeds = 1
    args.final_window = 40
    args.max_train_examples = 200
    args.max_test_examples = 80
    args.n_permutations = 2
    args.task_block_size = 40


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.n_permutations < 2:
        raise ValueError("--n-permutations must be at least 2")
    if args.task_block_size <= 0:
        raise ValueError("--task-block-size must be positive")
    if args.mnist_source == "openml" and not args.allow_openml_download:
        raise ValueError("--mnist-source openml requires --allow-openml-download")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be positive or 'all'")
    if args.max_test_examples is not None and args.max_test_examples <= 0:
        raise ValueError("--max-test-examples must be positive or 'all'")
    if args.mlp_step_size <= 0.0:
        raise ValueError("--mlp-step-size must be positive")
    if not 0.0 <= args.mlp_sparsity < 1.0:
        raise ValueError("--mlp-sparsity must be in [0, 1)")
    if args.upgd_hidden_size <= 0:
        raise ValueError("--upgd-hidden-size must be positive")


def main() -> None:
    """Run D22."""
    args = parse_args()
    apply_smoke(args)
    validate_args(args)
    candidate_methods = (
        f"upgd_h{args.upgd_hidden_size}_softmax_ce",
        *(variant.name for variant in default_prototype_variants()),
        *(variant.name for variant in default_hybrid_variants()),
        *(variant.name for variant in default_core_hybrid_variants()),
    )

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for run_idx in range(args.n_seeds):
        seed = args.seed + run_idx
        record, meta = run_one_seed(seed, args)
        records.append(record)
        datasets_meta[DATASET_NAME] = meta

    results = {
        "config": {
            "runner": "d22_upgd_prototype_hybrid_opmnist",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "mnist_source": args.mnist_source,
            "max_train_examples": args.max_train_examples,
            "max_test_examples": args.max_test_examples,
            "n_permutations": args.n_permutations,
            "task_block_size": args.task_block_size,
            "task_sampling": args.task_sampling,
            "sample_with_replacement": args.sample_with_replacement,
            "upgd_hidden_size": args.upgd_hidden_size,
            "prototype_variants": [
                variant.__dict__ for variant in default_prototype_variants()
            ],
            "hybrid_variants": [
                variant.__dict__ for variant in default_hybrid_variants()
            ],
            "core_hybrid_variants": [
                variant.__dict__ for variant in default_core_hybrid_variants()
            ],
        },
        "datasets": datasets_meta,
        "records": records,
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "aggregate": bridge.add_deployment_comparisons(
            aggregate_records(records, candidate_methods),
            records,
            candidate_methods,
        ),
        "wall_clock_s": float(time.time() - t0),
        "evidence_level": "jax_prototype_hybrid_opmnist",
    }
    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    bridge.atomic_write_json(json_path, results)
    write_summary(summary_path, results)
    if args.note_path is not None:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path is not None:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

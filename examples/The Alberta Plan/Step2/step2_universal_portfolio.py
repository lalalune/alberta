#!/usr/bin/env python3
"""Prediction-space Step 2 portfolio over MLP, UPGD, and dynamic sparse experts.

This is the current "try to close Step 2" runner.  It extends the earlier
MLP/UPGD expert mixture in two ways:

1. It compares against a small fair-MLP width grid, not a single MLP.
2. It adds the scaled D03 dynamic-sparse expert that beat MLP on the
   interaction/nonlinear pilot screen.

The portfolio is a live prediction-space discounted Hedge mixture.  Each expert
predicts before update at every time step; the mixture prediction is the convex
weighted sum of expert predictions; then every expert updates on the same
example.  The held-out digits deployment path can optionally use the same
class-imbalance retention router as ``step2_expert_mixture.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step2_dynamic_sparse import DynamicSparseMLP  # noqa: E402
from step2_expert_mixture import (  # noqa: E402
    DIGITS_REGIMES,
    N_DIGIT_CLASSES,
    RETENTION_ROUTERS,
    SYNTHETIC_REGIMES,
    VALID_DATASETS,
    class_imbalance_retention_signal,
    expand_dataset_names,
    load_digits_arrays,
    make_digits_regime_sequence,
    make_mlp,
    make_synthetic_stream,
    make_upgd,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_universal_portfolio")
DEFAULT_NOTE_PATH = Path("docs/research/step2_universal_portfolio.md")
DEFAULT_ALL_FRONTS_SUMMARY_PATH = Path(
    "docs/research/step2_all_fronts_portfolio_attempt.md"
)
EXPERT_NAMES = (
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_64",
    "upgd_low_noise",
    "dynamic_sparse",
)
METHOD_NAMES = ("mixture", *EXPERT_NAMES)
MLP_METHODS = ("mlp_h64", "mlp_h128", "mlp_h64_64")
UPGD_INDEX = EXPERT_NAMES.index("upgd_low_noise")
LOSS_START = 1
WEIGHT_START = LOSS_START + len(EXPERT_NAMES)
MLP_WEIGHT_START = WEIGHT_START + len(EXPERT_NAMES)
ACC_WEIGHT_START = MLP_WEIGHT_START + len(MLP_METHODS)
ROUTER_START = ACC_WEIGHT_START + len(EXPERT_NAMES)
ALL_SELECTOR_START = ROUTER_START + 1
MLP_SELECTOR_START = ALL_SELECTOR_START + 1
PRED_START = MLP_SELECTOR_START + 1
ROUTER_NAMES = (
    "all_convex",
    "all_selector",
    "mlp_convex",
    "mlp_selector",
)
ONLINE_RETENTION_GUARD_ROUTE_ID = len(ROUTER_NAMES)

ALL_FRONTS_ARTIFACTS = {
    "strict_supervised": Path("outputs/step2_canonical/universal_portfolio_strict_results.json"),
    "recursive_controlled": Path(
        "outputs/step2_canonical/recursive_feature_router_suite_10seed_5000/"
        "recursive_feature_utility_results.json"
    ),
    "opmnist_partial": Path(
        "outputs/step2_opmnist_scale_800task/"
        "opmnist_true_mnist_800task_partial20block_mse_results.json"
    ),
    "opmnist_10block": Path(
        "outputs/step2_opmnist_scale_10block/opmnist_true_mnist_10block_results.json"
    ),
    "scr_million": Path(
        "outputs/step2_scr_million_slow_meta_3seed/"
        "scr_million_slow_meta_3seed_results.json"
    ),
    "td_gvf_bridge": Path("output/td_gvf_ar1_squares_5seed/summary.json"),
}


def stderr(values: np.ndarray) -> float:
    """Return standard error."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.shape[0]))


def make_dynamic_sparse(
    n_heads: int,
    hidden_size: int,
    step_size: float,
    sparsity: float,
    utility_decay: float,
    rewire_interval: int,
    unit_replacement_rate: float,
) -> DynamicSparseMLP:
    """Construct the D03 dynamic sparse expert."""
    return DynamicSparseMLP(
        n_heads=n_heads,
        hidden_size=hidden_size,
        step_size=step_size,
        sparsity=sparsity,
        utility_decay=utility_decay,
        rewire_interval=rewire_interval,
        unit_replacement_rate=unit_replacement_rate,
        use_layer_norm=True,
        obgd_kappa=2.0,
    )


def run_portfolio_stream(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the live prediction-space portfolio on one materialized stream."""
    n_heads = int(targets.shape[1])
    feature_dim = int(observations.shape[1])
    keys = jr.split(key, len(EXPERT_NAMES))
    mlp64 = make_mlp(n_heads, (64,), args.step_size, args.sparsity)
    mlp128 = make_mlp(n_heads, (128,), args.step_size, args.sparsity)
    mlp6464 = make_mlp(n_heads, (64, 64), args.step_size, args.sparsity)
    upgd = make_upgd(
        n_heads=n_heads,
        hidden_sizes=(64,),
        step_size=args.step_size,
        sparsity=args.sparsity,
        perturbation_sigma=args.perturbation_sigma,
        perturbation_warmup_steps=args.perturbation_warmup_steps,
        perturbation_ramp_steps=args.perturbation_ramp_steps,
    )
    dynamic = make_dynamic_sparse(
        n_heads=n_heads,
        hidden_size=args.dynamic_hidden_size,
        step_size=args.step_size,
        sparsity=args.sparsity,
        utility_decay=args.dynamic_utility_decay,
        rewire_interval=args.dynamic_rewire_interval,
        unit_replacement_rate=args.dynamic_unit_replacement_rate,
    )

    recent_buffer_size = int(args.final_window)
    states = (
        mlp64.init(feature_dim, keys[0]),
        mlp128.init(feature_dim, keys[1]),
        mlp6464.init(feature_dim, keys[2]),
        upgd.init(feature_dim, keys[3]),
        dynamic.init(feature_dim, keys[4]),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(MLP_METHODS), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(EXPERT_NAMES), dtype=jnp.float32),
        jnp.zeros(len(ROUTER_NAMES), dtype=jnp.float32),
        jnp.zeros(n_heads, dtype=jnp.float32),
        jnp.zeros(n_heads, dtype=jnp.float32),
        jnp.zeros((recent_buffer_size, n_heads), dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )
    eta = jnp.array(args.hedge_eta, dtype=jnp.float32)
    discount = jnp.array(args.hedge_discount, dtype=jnp.float32)
    router_decay = jnp.array(args.router_decay, dtype=jnp.float32)

    def step_fn(
        carry: tuple[
            Any,
            Any,
            Any,
            Any,
            Any,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[
            Any,
            Any,
            Any,
            Any,
            Any,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        jax.Array,
    ]:
        (
            mlp64_s,
            mlp128_s,
            mlp6464_s,
            upgd_s,
            dynamic_s,
            log_w,
            mlp_log_w,
            acc_log_w,
            expert_ema,
            router_ema,
            lifetime_seen,
            recent_class_counts,
            recent_class_buffer,
            recent_class_buffer_idx,
        ) = carry
        obs, tgt = inputs

        preds = jnp.stack(
            [
                mlp64.predict(mlp64_s, obs),
                mlp128.predict(mlp128_s, obs),
                mlp6464.predict(mlp6464_s, obs),
                upgd.predict(upgd_s, obs),
                dynamic.predict(dynamic_s, obs),
            ],
            axis=0,
        )
        weights = jax.nn.softmax(log_w)
        mlp_weights = jax.nn.softmax(mlp_log_w)
        acc_weights = jax.nn.softmax(acc_log_w)
        all_convex_pred = jnp.sum(weights[:, None] * preds, axis=0)
        mlp_convex_pred = jnp.sum(mlp_weights[:, None] * preds[: len(MLP_METHODS)], axis=0)
        all_selector_idx = jnp.argmin(expert_ema)
        mlp_selector_idx = jnp.argmin(expert_ema[: len(MLP_METHODS)])
        router_preds = jnp.stack(
            [
                all_convex_pred,
                preds[all_selector_idx],
                mlp_convex_pred,
                preds[mlp_selector_idx],
            ],
            axis=0,
        )
        if args.router_policy == "convex":
            router_idx = jnp.array(0, dtype=jnp.int32)
        elif args.router_policy == "all_selector":
            router_idx = jnp.array(1, dtype=jnp.int32)
        elif args.router_policy == "mlp_convex":
            router_idx = jnp.array(2, dtype=jnp.int32)
        elif args.router_policy == "mlp_selector":
            router_idx = jnp.array(3, dtype=jnp.int32)
        elif args.router_policy == "guarded_convex":
            router_idx = jnp.where(
                router_ema[0] <= router_ema[2] + args.guard_tolerance,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(2, dtype=jnp.int32),
            )
        elif args.router_policy == "guarded_best_mlp":
            best_mlp_route = jnp.where(
                router_ema[2] <= router_ema[3],
                jnp.array(2, dtype=jnp.int32),
                jnp.array(3, dtype=jnp.int32),
            )
            best_mlp_ema = jnp.minimum(router_ema[2], router_ema[3])
            router_idx = jnp.where(
                router_ema[0] <= best_mlp_ema + args.guard_tolerance,
                jnp.array(0, dtype=jnp.int32),
                best_mlp_route,
            )
        else:
            router_idx = jnp.argmin(router_ema)
        lifetime_class_count = jnp.sum((lifetime_seen > 0.0).astype(jnp.float32))
        recent_class_count = jnp.sum((recent_class_counts > 0.0).astype(jnp.float32))
        lifetime_class_fraction = lifetime_class_count / jnp.asarray(
            max(n_heads, 1), dtype=jnp.float32
        )
        recent_fraction_of_lifetime = recent_class_count / jnp.maximum(
            lifetime_class_count, 1.0
        )
        online_retention_hazard = (
            (
                lifetime_class_fraction
                >= args.online_retention_min_lifetime_class_fraction
            )
            & (
                recent_fraction_of_lifetime
                <= args.online_retention_max_recent_class_fraction
            )
        )
        base_router_idx = router_idx
        if args.online_retention_mse_guard and n_heads == N_DIGIT_CLASSES:
            router_idx = jnp.where(
                online_retention_hazard,
                jnp.array(ONLINE_RETENTION_GUARD_ROUTE_ID, dtype=jnp.int32),
                router_idx,
            )
            mixture_pred = jnp.where(
                online_retention_hazard,
                preds[EXPERT_NAMES.index("mlp_h64_64")],
                router_preds[base_router_idx],
            )
        else:
            mixture_pred = router_preds[router_idx]
        expert_losses = jnp.mean((preds - tgt[None, :]) ** 2, axis=1)
        router_losses = jnp.mean((router_preds - tgt[None, :]) ** 2, axis=1)
        target_class = jnp.argmax(tgt)
        mixture_loss = jnp.mean((mixture_pred - tgt) ** 2)
        expert_pred_classes = jnp.argmax(preds, axis=1)
        expert_accuracy_losses = (
            expert_pred_classes != target_class
        ).astype(jnp.float32)
        current_class = jax.nn.one_hot(
            target_class,
            n_heads,
            dtype=jnp.float32,
        )
        old_recent_class = recent_class_buffer[recent_class_buffer_idx]
        new_recent_class_counts = recent_class_counts - old_recent_class + current_class
        new_recent_class_buffer = recent_class_buffer.at[
            recent_class_buffer_idx
        ].set(current_class)
        new_recent_class_buffer_idx = (
            recent_class_buffer_idx + jnp.array(1, dtype=jnp.int32)
        ) % jnp.array(recent_buffer_size, dtype=jnp.int32)
        new_lifetime_seen = jnp.maximum(lifetime_seen, current_class)

        new_log_w = discount * log_w - eta * expert_losses
        new_log_w = new_log_w - jnp.max(new_log_w)
        new_mlp_log_w = discount * mlp_log_w - eta * expert_losses[: len(MLP_METHODS)]
        new_mlp_log_w = new_mlp_log_w - jnp.max(new_mlp_log_w)
        new_acc_log_w = discount * acc_log_w - eta * expert_accuracy_losses
        new_acc_log_w = new_acc_log_w - jnp.max(new_acc_log_w)
        new_expert_ema = (1.0 - router_decay) * expert_ema + router_decay * expert_losses
        new_router_ema = (1.0 - router_decay) * router_ema + router_decay * router_losses

        mlp64_result = mlp64.update(mlp64_s, obs, tgt)
        mlp128_result = mlp128.update(mlp128_s, obs, tgt)
        mlp6464_result = mlp6464.update(mlp6464_s, obs, tgt)
        upgd_result = upgd.update(upgd_s, obs, tgt)
        dynamic_result = dynamic.update(dynamic_s, obs, tgt)

        pred_classes = jnp.concatenate(
            [
                jnp.asarray([jnp.argmax(mixture_pred)], dtype=jnp.float32),
                jnp.argmax(preds, axis=1).astype(jnp.float32),
            ]
        )
        metric = jnp.concatenate(
            [
                jnp.asarray([mixture_loss], dtype=jnp.float32),
                expert_losses.astype(jnp.float32),
                weights.astype(jnp.float32),
                mlp_weights.astype(jnp.float32),
                acc_weights.astype(jnp.float32),
                jnp.asarray(
                    [
                        router_idx.astype(jnp.float32),
                        all_selector_idx.astype(jnp.float32),
                        mlp_selector_idx.astype(jnp.float32),
                    ],
                    dtype=jnp.float32,
                ),
                pred_classes,
            ]
        )
        return (
            mlp64_result.state,
            mlp128_result.state,
            mlp6464_result.state,
            upgd_result.state,
            dynamic_result.state,
            new_log_w,
            new_mlp_log_w,
            new_acc_log_w,
            new_expert_ema,
            new_router_ema,
            new_lifetime_seen,
            new_recent_class_counts,
            new_recent_class_buffer,
            new_recent_class_buffer_idx,
        ), metric

    final_tuple, metrics = jax.lax.scan(step_fn, states, (observations, targets))
    metrics.block_until_ready()
    final_states = {
        "mlp_h64": (mlp64, final_tuple[0]),
        "mlp_h128": (mlp128, final_tuple[1]),
        "mlp_h64_64": (mlp6464, final_tuple[2]),
        "upgd_low_noise": (upgd, final_tuple[3]),
        "dynamic_sparse": (dynamic, final_tuple[4]),
    }
    return final_states, np.asarray(metrics)


def summarize_prequential(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Summarize online metrics for mixture and every expert."""
    window = min(final_window, metrics.shape[0])
    summary: dict[str, dict[str, float]] = {}
    for method_idx, method in enumerate(METHOD_NAMES):
        loss_col = 0 if method == "mixture" else LOSS_START + method_idx - 1
        entry = {
            "online_mean_mse": float(np.mean(metrics[:, loss_col])),
            "final_window_mse": float(np.mean(metrics[-window:, loss_col])),
        }
        if labels is not None:
            pred_col = PRED_START + method_idx
            correct = metrics[:, pred_col].astype(np.int32) == labels
            entry["online_mean_accuracy"] = float(np.mean(correct))
            entry["final_window_accuracy"] = float(np.mean(correct[-window:]))
        if method == "mixture":
            for idx, expert in enumerate(EXPERT_NAMES):
                weight_col = WEIGHT_START + idx
                entry[f"mean_{expert}_weight"] = float(np.mean(metrics[:, weight_col]))
                entry[f"final_{expert}_weight"] = float(metrics[-1, weight_col])
            for idx, method_name in enumerate(MLP_METHODS):
                weight_col = MLP_WEIGHT_START + idx
                entry[f"mean_mlp_guard_{method_name}_weight"] = float(
                    np.mean(metrics[:, weight_col])
                )
                entry[f"final_mlp_guard_{method_name}_weight"] = float(
                    metrics[-1, weight_col]
                )
            for idx, expert in enumerate(EXPERT_NAMES):
                weight_col = ACC_WEIGHT_START + idx
                entry[f"mean_accuracy_{expert}_weight"] = float(
                    np.mean(metrics[:, weight_col])
                )
                entry[f"final_accuracy_{expert}_weight"] = float(
                    metrics[-1, weight_col]
                )
            entry["mean_meta_route_id"] = float(np.mean(metrics[:, ROUTER_START]))
            entry["final_meta_route_id"] = float(metrics[-1, ROUTER_START])
        summary[method] = entry
    return summary


def evaluate_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate one expert on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    preds = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def deployment_weights(
    tracking_weights: np.ndarray,
    labels: np.ndarray | None,
    n_heads: int,
    final_window: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Return held-out deployment weights for the multi-expert portfolio."""
    weights = np.asarray(tracking_weights, dtype=np.float32).copy()
    weights = np.clip(weights, 0.0, 1.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.full(len(EXPERT_NAMES), 1.0 / len(EXPERT_NAMES), dtype=np.float32)
    else:
        weights = weights / total

    signal: dict[str, float | int | bool | str] = dict(
        class_imbalance_retention_signal(
            labels=labels,
            n_heads=n_heads,
            final_window=final_window,
            min_lifetime_class_fraction=args.retention_min_lifetime_class_fraction,
            max_recent_class_fraction=args.retention_max_recent_class_fraction,
        )
    )
    signal["router"] = args.retention_router
    signal["deployment_source"] = "tracking"
    for idx, expert in enumerate(EXPERT_NAMES):
        signal[f"tracking_{expert}_weight"] = float(weights[idx])

    if args.retention_router == "class_imbalance" and signal["retention_hazard"]:
        upgd_weight = max(float(weights[UPGD_INDEX]), args.retention_upgd_deployment_weight)
        upgd_weight = min(max(upgd_weight, 0.0), 1.0)
        remainder = 1.0 - upgd_weight
        non_upgd = weights.copy()
        non_upgd[UPGD_INDEX] = 0.0
        non_upgd_sum = float(np.sum(non_upgd))
        if non_upgd_sum <= 0.0:
            weights = np.zeros_like(weights)
        else:
            weights = non_upgd / non_upgd_sum * remainder
        weights[UPGD_INDEX] = upgd_weight
        signal["deployment_source"] = "class_imbalance_retention"

    for idx, expert in enumerate(EXPERT_NAMES):
        signal[f"deployment_{expert}_weight"] = float(weights[idx])
    return weights.astype(np.float32), signal


def guarded_tracking_weights(metrics: np.ndarray) -> np.ndarray:
    """Return final online routing weights after applying the MLP guard."""
    route_id = int(round(float(metrics[-1, ROUTER_START])))
    all_weights = metrics[-1, WEIGHT_START : WEIGHT_START + len(EXPERT_NAMES)]
    mlp_weights = metrics[-1, MLP_WEIGHT_START : MLP_WEIGHT_START + len(MLP_METHODS)]
    if route_id == 0:
        return all_weights.astype(np.float32)
    if route_id == 1:
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
        weights[int(round(float(metrics[-1, ALL_SELECTOR_START])))] = 1.0
        return weights
    if route_id == 2:
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
        weights[: len(MLP_METHODS)] = mlp_weights.astype(np.float32)
        return weights
    if route_id == ONLINE_RETENTION_GUARD_ROUTE_ID:
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
        weights[EXPERT_NAMES.index("mlp_h64_64")] = 1.0
        return weights
    weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
    weights[int(round(float(metrics[-1, MLP_SELECTOR_START])))] = 1.0
    return weights


def final_deployment_tracking_weights(
    metrics: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return final deployment weights for held-out digits evaluation."""
    if args.digits_deployment_objective == "accuracy":
        return metrics[-1, ACC_WEIGHT_START : ACC_WEIGHT_START + len(EXPERT_NAMES)].astype(
            np.float32
        )
    if args.digits_deployment_objective == "mlp_h128":
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
        weights[EXPERT_NAMES.index("mlp_h128")] = 1.0
        return weights
    return guarded_tracking_weights(metrics)


def evaluate_mixture_classifier(
    final_states: dict[str, tuple[Any, Any]],
    weights: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final convex expert mixture on held-out digits."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    weights_jax = jnp.asarray(weights.astype(np.float32))

    def predict(obs: jax.Array) -> jax.Array:
        preds = jnp.stack(
            [
                final_states[expert][0].predict(final_states[expert][1], obs)
                for expert in EXPERT_NAMES
            ],
            axis=0,
        )
        return jnp.sum(weights_jax[:, None] * preds, axis=0)

    preds = jax.vmap(predict)(observations)
    mse = jnp.mean((preds - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(preds, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def is_higher_better(metric: str) -> bool:
    """Return whether a metric is higher-is-better."""
    return metric.endswith("accuracy")


def better_value(values: dict[str, float], metric: str) -> tuple[str, float]:
    """Return the best method/value for one metric."""
    if is_higher_better(metric):
        method = max(values, key=values.__getitem__)
    else:
        method = min(values, key=values.__getitem__)
    return method, values[method]


def paired_diff(method_value: float, baseline_value: float, metric: str) -> float:
    """Positive paired difference favors the method."""
    if is_higher_better(metric):
        return method_value - baseline_value
    return baseline_value - method_value


def summary_row(values: np.ndarray) -> dict[str, Any]:
    """Return mean/stderr/values for one aggregate row."""
    return {
        "mean": float(np.mean(values)),
        "stderr": stderr(values),
        "values": values.tolist(),
    }


def paired_mixture_vs_group(
    records: list[dict[str, Any]],
    metric: str,
    group: tuple[str, ...],
    group_name: str,
) -> dict[str, Any]:
    """Compare mixture to the per-seed best method in a group."""
    diffs: list[float] = []
    best_methods: list[str] = []
    for record in records:
        methods = record["methods"]
        method_value = float(methods["mixture"][metric])
        group_values = {method: float(methods[method][metric]) for method in group}
        best_method, best_value = better_value(group_values, metric)
        best_methods.append(best_method)
        diffs.append(paired_diff(method_value, best_value, metric))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "metric": metric,
        "baseline_group": group_name,
        "paired_diff_mean_positive_favors_mixture": float(np.mean(diff_arr)),
        "paired_diff_stderr": stderr(diff_arr),
        "wins_for_mixture": int(np.sum(diff_arr > 0.0)),
        "wins_for_baseline": int(np.sum(diff_arr < 0.0)),
        "ties": int(np.sum(diff_arr == 0.0)),
        "n": int(diff_arr.shape[0]),
        "diffs": diff_arr.tolist(),
        "best_baseline_counts": dict(Counter(best_methods)),
    }


def best_expert_regret(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Aggregate mixture regret against the best non-mixture expert."""
    regrets: list[float] = []
    best_methods: list[str] = []
    for record in records:
        methods = record["methods"]
        mix_value = float(methods["mixture"][metric])
        expert_values = {method: float(methods[method][metric]) for method in EXPERT_NAMES}
        best_method, best_value = better_value(expert_values, metric)
        best_methods.append(best_method)
        regrets.append(-paired_diff(mix_value, best_value, metric))
    regret_arr = np.asarray(regrets, dtype=np.float64)
    return {
        "metric": metric,
        "mean_regret_positive_favors_best_expert": float(np.mean(regret_arr)),
        "stderr": stderr(regret_arr),
        "values": regret_arr.tolist(),
        "failures": int(np.sum(regret_arr > 0.0)),
        "ties_or_beats_best": int(np.sum(regret_arr <= 0.0)),
        "best_expert_counts": dict(Counter(best_methods)),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed records by dataset, method, and metric."""
    aggregate: dict[str, Any] = {}
    datasets = sorted({record["dataset_name"] for record in records})
    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset_name"] == dataset]
        metrics = sorted(dataset_records[0]["methods"]["mixture"])
        aggregate[dataset] = {}
        for method in METHOD_NAMES:
            aggregate[dataset][method] = {}
            for metric in metrics:
                if metric not in dataset_records[0]["methods"][method]:
                    continue
                values = np.asarray(
                    [record["methods"][method][metric] for record in dataset_records],
                    dtype=np.float64,
                )
                aggregate[dataset][method][metric] = summary_row(values)

        main_metrics = [
            metric
            for metric in (
                "final_window_mse",
                "online_mean_mse",
                "test_mse",
                "final_window_accuracy",
                "online_mean_accuracy",
                "test_accuracy",
            )
            if metric in dataset_records[0]["methods"]["mixture"]
        ]
        aggregate[dataset]["comparisons"] = {
            metric: {
                "mixture_vs_best_mlp": paired_mixture_vs_group(
                    dataset_records,
                    metric,
                    MLP_METHODS,
                    "best_mlp",
                ),
                "mixture_vs_best_expert": paired_mixture_vs_group(
                    dataset_records,
                    metric,
                    EXPERT_NAMES,
                    "best_expert",
                ),
            }
            for metric in main_metrics
        }
        aggregate[dataset]["best_expert_regret"] = {
            metric: best_expert_regret(dataset_records, metric)
            for metric in main_metrics
        }
    return aggregate


def load_json_artifact(repo_root: Path, relative_path: Path) -> dict[str, Any] | None:
    """Load an evidence artifact if it exists."""
    path = repo_root / relative_path
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def comparison_diff(
    dataset_agg: dict[str, Any],
    metric: str,
    comparison: str = "mixture_vs_best_mlp",
) -> float | None:
    """Return an aggregate paired diff from a Step 2 portfolio artifact."""
    try:
        return float(
            dataset_agg["comparisons"][metric][comparison][
                "paired_diff_mean_positive_favors_mixture"
            ]
        )
    except KeyError:
        return None


def summarize_strict_supervised_artifact(
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize strict supervised Step 2 portfolio evidence."""
    if artifact is None:
        return {
            "status": "missing",
            "claim": "strict supervised matrix",
            "summary": "Missing strict supervised portfolio artifact.",
        }
    aggregate = artifact.get("aggregate", {})
    checked = 0
    failures: list[str] = []
    for dataset, dataset_agg in aggregate.items():
        for metric in ("final_window_mse", "test_accuracy"):
            diff = comparison_diff(dataset_agg, metric)
            if diff is None:
                continue
            checked += 1
            if diff < 0.0:
                failures.append(f"{dataset}:{metric}={diff:+.6f}")
    status = "closed" if checked > 0 and not failures else "partial"
    return {
        "status": status,
        "claim": "strict supervised matrix",
        "summary": (
            f"Checked {checked} portfolio-vs-best-MLP comparisons; "
            f"failures={len(failures)}."
        ),
        "failures": failures,
        "artifact_evidence_level": artifact.get("evidence_level", ""),
    }


def summarize_recursive_artifact(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize controlled recursive-suite evidence."""
    if artifact is None:
        return {
            "status": "missing",
            "claim": "controlled recursive feature-construction suite",
            "summary": "Missing recursive router suite artifact.",
        }
    suite = artifact.get("aggregate", {}).get("suite_summary", {})
    tasks = int(suite.get("tasks", 0))
    wins = int(suite.get("recursive_mlp_router_beats_best_mlp_tasks", 0))
    ties = int(suite.get("recursive_mlp_router_ties_best_mlp_tasks", 0))
    status = "closed" if tasks > 0 and wins == tasks else "partial"
    return {
        "status": status,
        "claim": "controlled recursive feature-construction suite",
        "summary": (
            f"`recursive_mlp_router` beats best fair MLP on {wins}/{tasks} "
            f"tasks; ties={ties}."
        ),
        "tasks": tasks,
        "wins": wins,
        "ties": ties,
    }


def summarize_opmnist_artifact(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize OPMNIST evidence without treating partial scale as closure."""
    if artifact is None:
        return {
            "status": "missing",
            "claim": "published-scale Online Permuted MNIST",
            "summary": "Missing OPMNIST artifact.",
        }
    status_block = artifact.get("status", {})
    dataset = artifact.get("datasets", {}).get("permuted_mnist_like", {})
    completed_blocks = int(dataset.get("completed_full_task_blocks", 0))
    total_tasks = int(dataset.get("n_permutations", 0))
    core_protocol = bool(status_block.get("matches_dohare_opmnist_core_protocol"))
    published_task_count = bool(
        status_block.get("matches_dohare_opmnist_published_task_count")
    )
    primary_nonnegative = bool(
        status_block.get("all_primary_nonnegative_vs_best_mlp")
    )
    status = "closed" if published_task_count and primary_nonnegative else "partial"
    return {
        "status": status,
        "claim": "published-scale Online Permuted MNIST",
        "summary": (
            f"Core protocol={core_protocol}; completed full 60k blocks="
            f"{completed_blocks}/{total_tasks}; primary nonnegative="
            f"{primary_nonnegative}."
        ),
        "completed_blocks": completed_blocks,
        "total_tasks": total_tasks,
        "core_protocol": core_protocol,
        "published_task_count": published_task_count,
    }


def summarize_scr_artifact(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize published-scale SCR router evidence."""
    if artifact is None:
        return {
            "status": "missing",
            "claim": "published-scale Slowly-Changing Regression",
            "summary": "Missing million-step SCR artifact.",
        }
    closed = bool(artifact.get("published_scale_scr_closed"))
    variant = str(artifact.get("best_variant", "unknown"))
    best_status = artifact.get("best_variant_status", {})
    return {
        "status": "closed" if closed else "partial",
        "claim": "published-scale Slowly-Changing Regression",
        "summary": (
            f"Best router `{variant}`; published-scale SCR closed={closed}; "
            f"public protocol={bool(best_status.get('matches_dohare_public_scr_protocol'))}."
        ),
        "best_variant": variant,
        "published_scale_scr_closed": closed,
    }


def summarize_td_gvf_artifact(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize the Step 2-to-Step 3 TD/GVF bridge artifact."""
    if artifact is None:
        return {
            "status": "missing",
            "claim": "TD/GVF feature-discovery bridge",
            "summary": "Missing TD/GVF bridge artifact.",
        }
    beats_linear = bool(artifact.get("best_discovery_beats_linear"))
    beats_mlp = bool(artifact.get("best_discovery_beats_mlp"))
    best_discovery = str(artifact.get("best_discovery_method", "unknown"))
    status = "partial"
    if beats_linear and beats_mlp:
        status = "partial"
    return {
        "status": status,
        "claim": "TD/GVF feature-discovery bridge",
        "summary": (
            f"Best discovery `{best_discovery}` beats linear={beats_linear}, "
            f"beats MLP={beats_mlp}; kept partial because this is Step 3 "
            "bridge evidence, not a Step 2 portfolio route."
        ),
        "best_discovery_method": best_discovery,
        "best_discovery_beats_linear": beats_linear,
        "best_discovery_beats_mlp": beats_mlp,
    }


def build_all_fronts_portfolio_summary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build an artifact-only all-fronts Step 2 portfolio status."""
    artifacts = {
        name: load_json_artifact(repo_root, path)
        for name, path in ALL_FRONTS_ARTIFACTS.items()
    }
    opmnist_artifact = artifacts["opmnist_partial"] or artifacts["opmnist_10block"]
    fronts = {
        "strict_supervised": summarize_strict_supervised_artifact(
            artifacts["strict_supervised"]
        ),
        "recursive_controlled": summarize_recursive_artifact(
            artifacts["recursive_controlled"]
        ),
        "opmnist": summarize_opmnist_artifact(opmnist_artifact),
        "scr": summarize_scr_artifact(artifacts["scr_million"]),
        "td_gvf_bridge": summarize_td_gvf_artifact(artifacts["td_gvf_bridge"]),
    }
    missing = [name for name, row in fronts.items() if row["status"] == "missing"]
    partial = [name for name, row in fronts.items() if row["status"] == "partial"]
    closed = [name for name, row in fronts.items() if row["status"] == "closed"]
    decision = "promote"
    if missing:
        decision = "reject"
    elif partial:
        decision = "partial"
    return {
        "decision": decision,
        "fronts": fronts,
        "closed_fronts": closed,
        "partial_fronts": partial,
        "missing_fronts": missing,
        "artifact_paths": {name: str(path) for name, path in ALL_FRONTS_ARTIFACTS.items()},
    }


def write_all_fronts_portfolio_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the artifact-only all-fronts portfolio assessment."""
    fronts = summary["fronts"]
    lines = [
        "# Step 2 All-Fronts Portfolio Attempt",
        "",
        "This note is generated from existing artifacts only. It does not rerun "
        "large experiments, does not import Step 3 harnesses into Step 2, and "
        "does not treat missing scale evidence as closure.",
        "",
        f"Decision: **{summary['decision'].upper()}**.",
        "",
        "| Front | Status | Claim | Evidence |",
        "|---|---|---|---|",
    ]
    for name, row in fronts.items():
        lines.append(
            f"| `{name}` | `{row['status']}` | {row['claim']} | {row['summary']} |"
        )

    lines.extend(
        [
            "",
            "## Route Audit",
            "",
            "- Strict supervised matrix: already represented inside "
            "`step2_universal_portfolio.py` by live MLP/UPGD/dynamic-sparse "
            "experts, discounted Hedge, MLP-only guard routes, online "
            "class-imbalance MSE guard, and held-out retention deployment guard.",
            "- Controlled recursive suite: evidence exists for "
            "`recursive_mlp_router`, but this is a separate runner family. "
            "Adding it directly to the strict supervised portfolio would require "
            "routing over different state shapes and task suites, so the honest "
            "integration is artifact-level or a higher-level conclusive runner.",
            "- External OPMNIST/SCR: SCR has a narrowed million-step router "
            "closure. OPMNIST is positive on completed true-MNIST blocks but "
            "remains incomplete until the full 800 x 60,000 task-count gate is "
            "met.",
            "- TD/GVF bridge: positive bridge evidence exists, but it belongs at "
            "the Step 2/3 boundary. Importing the Step 3 harness into this Step 2 "
            "portfolio runner would create coupling without making TD/GVF a live "
            "Step 2 deployment route.",
            "",
            "## Interpretation",
            "",
            "A temporally uniform portfolio is acceptable as the current "
            "portfolio-level Step 2 answer only in a partial sense: it closes the "
            "strict supervised matrix, the controlled recursive suite is closed "
            "by a separate causal router, and SCR has published-scale evidence. "
            "It is weaker than a single feature-construction algorithm because "
            "coverage comes from guarded allocation among known mechanisms, not "
            "from one mechanism that discovers the right representation across "
            "all fronts.",
            "",
            "The all-fronts claim should therefore remain **partial**, not "
            "promoted, until full published-scale OPMNIST completes and TD/GVF "
            "feature finding is either integrated through a clean boundary or "
            "reported as a separate Step 3 bridge rather than Step 2 closure.",
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for name, artifact_path in summary["artifact_paths"].items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Run one dataset/seed combination."""
    labels_np: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    if dataset_name in SYNTHETIC_REGIMES:
        observations, targets, dataset_meta = make_synthetic_stream(
            steps=args.steps,
            seed=seed + 20_000,
            regime=dataset_name,
        )
        n_heads = int(targets.shape[1])
    elif dataset_name in DIGITS_REGIMES:
        x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
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
        dataset_meta.update(stream_meta)
        labels_np = np.asarray(labels)
        n_heads = N_DIGIT_CLASSES
    else:
        raise ValueError(f"unknown dataset_name={dataset_name}")

    final_states, metrics = run_portfolio_stream(
        observations=observations,
        targets=targets,
        key=jr.key(seed),
        args=args,
    )
    methods = summarize_prequential(metrics, args.final_window, labels_np)

    retention_signal: dict[str, float | int | bool | str] | None = None
    if dataset_name in DIGITS_REGIMES:
        assert x_test is not None and y_test is not None
        tracking_weights = final_deployment_tracking_weights(metrics, args)
        final_weights, retention_signal = deployment_weights(
            tracking_weights=tracking_weights,
            labels=labels_np,
            n_heads=n_heads,
            final_window=args.final_window,
            args=args,
        )
        for expert in EXPERT_NAMES:
            learner, state = final_states[expert]
            methods[expert].update(evaluate_classifier(learner, state, x_test, y_test))
        methods["mixture"].update(
            evaluate_mixture_classifier(final_states, final_weights, x_test, y_test)
        )
        for idx, expert in enumerate(EXPERT_NAMES):
            methods["mixture"][f"deployment_{expert}_weight"] = float(final_weights[idx])
        methods["mixture"]["retention_hazard"] = float(
            bool(retention_signal["retention_hazard"])
        )

    record = {
        "dataset_name": dataset_name,
        "seed": seed,
        "dataset": dataset_meta,
        "methods": methods,
    }
    if retention_signal is not None:
        record["retention_router"] = retention_signal
    return record, dataset_meta, metrics


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format an aggregate metric if present."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write the Markdown summary."""
    cfg = results["config"]
    lines = [
        "# Step 2 Universal Portfolio",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} steps, "
            f"final window {cfg['final_window']}, Hedge eta={cfg['hedge_eta']}, "
            f"discount={cfg['hedge_discount']}, dynamic rewire interval="
            f"{cfg.get('dynamic_rewire_interval', 'n/a')}, deployment policy="
            f"{cfg['router_policy']}, retention router="
            f"{cfg['retention_router']}, digits deployment objective="
            f"{cfg.get('digits_deployment_objective', 'mse')}."
        ),
        "",
        "The runner tracks four causal deployment policies: all-expert convex "
        "Hedge, all-expert selector, MLP-only convex Hedge, and MLP-only "
        "selector. The promoted default is all-expert convex Hedge with a "
        "digits class-imbalance online MSE guard; the router policies remain "
        "available as negative-control variants.",
        "",
    ]

    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHOD_NAMES:
            row = dataset_agg[method]
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} |"
            )
        lines.append("")
        for metric in ("final_window_mse", "test_accuracy"):
            if metric not in dataset_agg["comparisons"]:
                continue
            cmp_row = dataset_agg["comparisons"][metric]["mixture_vs_best_mlp"]
            lines.append(
                f"`{metric}` mixture-vs-best-MLP diff: "
                f"{cmp_row['paired_diff_mean_positive_favors_mixture']:+.4f} "
                f"+/- {cmp_row['paired_diff_stderr']:.4f}; "
                f"wins/losses/ties "
                f"{cmp_row['wins_for_mixture']}/"
                f"{cmp_row['wins_for_baseline']}/{cmp_row['ties']}."
            )
        lines.append("")

    lines.extend(
        [
            "## Assessment Rule",
            "",
            "Positive mixture-vs-best-MLP differences favor the portfolio. For MSE, "
            "the difference is best MLP minus mixture; for accuracy, it is mixture "
            "minus best MLP. A universal Step 2 claim still requires this result "
            "to hold on the full matrix with enough seeds and no retained-test "
            "regression.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated regimes or aliases: all, synthetic, digits, "
            f"{', '.join(VALID_DATASETS)}."
        ),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-4)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamic-hidden-size", type=int, default=64)
    parser.add_argument("--dynamic-utility-decay", type=float, default=0.99)
    parser.add_argument("--dynamic-rewire-interval", type=int, default=180)
    parser.add_argument("--dynamic-unit-replacement-rate", type=float, default=0.05)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument(
        "--router-policy",
        choices=(
            "convex",
            "all_selector",
            "mlp_convex",
            "mlp_selector",
            "guarded_convex",
            "guarded_best_mlp",
            "meta",
        ),
        default="convex",
    )
    parser.add_argument("--router-decay", type=float, default=0.02)
    parser.add_argument("--guard-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--digits-deployment-objective",
        choices=("mse", "accuracy", "mlp_h128"),
        default="mse",
        help=(
            "Held-out digits deployment weights. 'mse' uses the final MSE router; "
            "'accuracy' uses a causal online 0-1 accuracy Hedge router; "
            "'mlp_h128' deploys the wide fair-MLP expert except where retention "
            "routing overrides it."
        ),
    )
    parser.add_argument(
        "--online-retention-mse-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Route class-imbalanced online digits tracking to the best "
            "current-block MLP route while leaving held-out deployment to the "
            "retention router."
        ),
    )
    parser.add_argument(
        "--online-retention-min-lifetime-class-fraction",
        type=float,
        default=0.7,
        help="Lifetime class coverage needed for the online class-imbalance MSE guard.",
    )
    parser.add_argument(
        "--online-retention-max-recent-class-fraction",
        type=float,
        default=0.5,
        help=(
            "Maximum recent/lifetime class coverage ratio for the online "
            "class-imbalance MSE guard."
        ),
    )
    parser.add_argument(
        "--retention-router",
        choices=RETENTION_ROUTERS,
        default="class_imbalance",
    )
    parser.add_argument("--retention-upgd-deployment-weight", type=float, default=1.0)
    parser.add_argument("--retention-min-lifetime-class-fraction", type=float, default=0.8)
    parser.add_argument("--retention-max-recent-class-fraction", type=float, default=0.4)
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument(
        "--all-fronts-summary-only",
        action="store_true",
        help=(
            "Do not run experiments. Read existing Step 2/3 evidence artifacts "
            "and write the all-fronts portfolio closure assessment."
        ),
    )
    parser.add_argument(
        "--all-fronts-summary-path",
        type=Path,
        default=DEFAULT_ALL_FRONTS_SUMMARY_PATH,
        help="Markdown path for --all-fronts-summary-only.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if not 0.0 <= args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in [0, 1]")
    if not 0.0 < args.router_decay <= 1.0:
        raise ValueError("--router-decay must be in (0, 1]")
    if args.phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < args.mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")
    if args.mask_noise_std < 0.0:
        raise ValueError("--mask-noise-std must be non-negative")
    if args.perturbation_warmup_steps < 0:
        raise ValueError("--perturbation-warmup-steps must be non-negative")
    if args.perturbation_ramp_steps < 0:
        raise ValueError("--perturbation-ramp-steps must be non-negative")
    if args.dynamic_rewire_interval <= 0:
        raise ValueError("--dynamic-rewire-interval must be positive")
    if not 0.0 <= args.dynamic_unit_replacement_rate <= 1.0:
        raise ValueError("--dynamic-unit-replacement-rate must be in [0, 1]")
    if not 0.0 <= args.retention_upgd_deployment_weight <= 1.0:
        raise ValueError("--retention-upgd-deployment-weight must be in [0, 1]")
    if not 0.0 <= args.online_retention_min_lifetime_class_fraction <= 1.0:
        raise ValueError(
            "--online-retention-min-lifetime-class-fraction must be in [0, 1]"
        )
    if not 0.0 <= args.online_retention_max_recent_class_fraction <= 1.0:
        raise ValueError(
            "--online-retention-max-recent-class-fraction must be in [0, 1]"
        )


def main() -> None:
    """Run the full matrix."""
    args = parse_args()
    if args.all_fronts_summary_only:
        summary = build_all_fronts_portfolio_summary()
        write_all_fronts_portfolio_summary(args.all_fronts_summary_path, summary)
        print(
            "wrote all-fronts portfolio assessment to "
            f"{args.all_fronts_summary_path}; decision={summary['decision']}"
        )
        return

    validate_args(args)
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = expand_dataset_names(args.datasets)
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}

    for dataset_name in dataset_names:
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            print(f"dataset={dataset_name} seed={seed}: running universal portfolio")
            record, meta, metrics = run_one_dataset_seed(dataset_name, seed, args)
            records.append(record)
            datasets_meta[dataset_name] = meta
            np.savez_compressed(
                args.output_dir / f"{dataset_name}_seed{seed}_curves.npz",
                metrics=metrics,
            )
            m = record["methods"]
            print(
                f"dataset={dataset_name} seed={seed}: final MSE "
                f"mix={m['mixture']['final_window_mse']:.4f}, "
                f"best_mlp={min(m[x]['final_window_mse'] for x in MLP_METHODS):.4f}, "
                f"dynamic={m['dynamic_sparse']['final_window_mse']:.4f}"
            )

    results = {
        "config": {
            "datasets": dataset_names,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "final_window": args.final_window,
            "expert_names": list(EXPERT_NAMES),
            "mlp_comparator_methods": list(MLP_METHODS),
            "step_size": args.step_size,
            "sparsity": args.sparsity,
            "perturbation_sigma": args.perturbation_sigma,
            "perturbation_warmup_steps": args.perturbation_warmup_steps,
            "perturbation_ramp_steps": args.perturbation_ramp_steps,
            "dynamic_hidden_size": args.dynamic_hidden_size,
            "dynamic_utility_decay": args.dynamic_utility_decay,
            "dynamic_rewire_interval": args.dynamic_rewire_interval,
            "dynamic_unit_replacement_rate": args.dynamic_unit_replacement_rate,
            "hedge_eta": args.hedge_eta,
            "hedge_discount": args.hedge_discount,
            "router_policy": args.router_policy,
            "router_decay": args.router_decay,
            "guard_tolerance": args.guard_tolerance,
            "digits_deployment_objective": args.digits_deployment_objective,
            "online_retention_mse_guard": args.online_retention_mse_guard,
            "online_retention_min_lifetime_class_fraction": (
                args.online_retention_min_lifetime_class_fraction
            ),
            "online_retention_max_recent_class_fraction": (
                args.online_retention_max_recent_class_fraction
            ),
            "retention_router": args.retention_router,
            "retention_upgd_deployment_weight": args.retention_upgd_deployment_weight,
            "retention_min_lifetime_class_fraction": (
                args.retention_min_lifetime_class_fraction
            ),
            "retention_max_recent_class_fraction": (
                args.retention_max_recent_class_fraction
            ),
            "phase_length": args.phase_length,
            "mask_keep_fraction": args.mask_keep_fraction,
            "mask_noise_std": args.mask_noise_std,
        },
        "datasets": datasets_meta,
        "records": records,
        "aggregate": aggregate_records(records),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "prediction_space_mlp_upgd_dynamic_sparse_portfolio",
    }

    json_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_summary(summary_path, results)
    write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runnable pilots for the D01-D10 Step 2 candidate directions.

These are deliberately small first-pass pilots.  They do not promote any method
to the framework core.  The goal is to move every direction from a markdown spec
to a reproducible comparison against the same fair MLP baseline on common Step 2
regression streams.

Some candidates are direct implementations; some are explicit proxies where the
full method would require a core learner update.  The output records that status
so a proxy win is treated as a reason to implement the full method, not as a
canonical result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from step2_dynamic_sparse import run_dynamic_sparse_arrays  # noqa: E402

from alberta_framework import (  # noqa: E402
    InteractionFeatureDiscoveryStream,
    MLPLearner,
    MultiHeadMLPLearner,
    NonlinearFeatureDiscoveryStream,
    ObGDBounding,
    run_multi_head_learning_loop,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_direction_pilots")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_direction_pilots.md")


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for the candidate pilot suite."""

    num_steps: int = 500
    num_seeds: int = 2
    final_window: int = 100
    feature_dim: int = 10
    n_tasks: int = 2
    n_contexts: int = 4
    context_length: int = 125
    step_size: float = 0.03
    hidden_size: int = 64
    sparsity: float = 0.5
    obgd_kappa: float = 2.0
    hedge_eta: float = 1.0
    hedge_discount: float = 0.995
    gate_decay: float = 0.02
    suites: tuple[str, ...] = ("interaction", "nonlinear")


METHOD_STATUS: dict[str, str] = {
    "fair_mlp": "baseline",
    "d01_rls_random_features": "direct readout-conditioning pilot over fixed nonlinear features",
    "d02_aux_next_projection": "direct auxiliary-prediction pilot using extra heads",
    "d03_dynamic_sparse": "direct existing dynamic-sparse pilot",
    "d04_spline_basis": "direct fixed spline-basis pilot; adaptive knots remain future work",
    "d05_high_leak_homeostasis": "proxy for homeostatic plasticity regularization",
    "d06_online_input_whitening": "proxy for natural-gradient whitening",
    "d07_budgeted_kernel_rls": "direct budgeted kernel-RLS pilot",
    "d08_independent_head_trunks": "proxy for head-conditioned modulation",
    "d09_history_features": "direct causal history feature pilot",
    "d10_precision_weighted_linear": "proxy for uncertainty-weighted losses",
    "portfolio_all_hedge": "causal loss-space Hedge over fair MLP plus D01-D10",
    "portfolio_signal_hedge": "causal loss-space Hedge over fair MLP plus D03-D08",
    "portfolio_signal_ema_gate": "causal EMA gate between fair MLP and signal Hedge",
}

BASE_METHODS = tuple(
    method for method in METHOD_STATUS if not method.startswith("portfolio_")
)
SIGNAL_METHODS = (
    "fair_mlp",
    "d03_dynamic_sparse",
    "d04_spline_basis",
    "d05_high_leak_homeostasis",
    "d06_online_input_whitening",
    "d07_budgeted_kernel_rls",
    "d08_independent_head_trunks",
)


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize one stream into paired arrays."""
    state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(step_fn, state, jnp.arange(num_steps))
    return observations, targets


def make_stream(name: str, config: PilotConfig) -> tuple[Any, str]:
    """Create one benchmark stream."""
    if name == "interaction":
        return (
            InteractionFeatureDiscoveryStream(
                feature_dim=config.feature_dim,
                n_tasks=config.n_tasks,
                n_contexts=config.n_contexts,
                context_length=config.context_length,
                active_pairs_per_context=1,
                noise_std=0.01,
                linear_scale=0.01,
            ),
            "pair-product stream where explicit interaction methods should help",
        )
    if name == "nonlinear":
        return (
            NonlinearFeatureDiscoveryStream(
                feature_dim=config.feature_dim,
                n_tasks=config.n_tasks,
                n_latents=32,
                n_contexts=config.n_contexts,
                context_length=config.context_length,
                active_latents_per_context=4,
                noise_std=0.01,
                linear_scale=0.01,
            ),
            "tanh-latent negative control not generated by simple pair products",
        )
    raise ValueError(f"unknown suite {name!r}")


def final_window_mean(curve: np.ndarray, final_window: int) -> float:
    """Mean over the last window."""
    return float(np.mean(curve[-min(final_window, curve.shape[0]) :]))


def stderr(values: Sequence[float]) -> float:
    """Return standard error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def summarize_curve_list(curves: list[np.ndarray], final_window: int) -> dict[str, Any]:
    """Summarize a list of per-step loss curves."""
    finals = np.asarray([final_window_mean(curve, final_window) for curve in curves])
    means = np.asarray([float(np.mean(curve)) for curve in curves])
    return {
        "final_window_loss_mean": float(np.mean(finals)),
        "final_window_loss_stderr": stderr(finals),
        "mean_loss_mean": float(np.mean(means)),
        "mean_loss_stderr": stderr(means),
        "per_seed_final_window_loss": finals.tolist(),
    }


def paired_lower(method: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    """Positive diff means method has lower loss than baseline."""
    diff = np.asarray(baseline, dtype=np.float64) - np.asarray(method, dtype=np.float64)
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_stderr": stderr(diff),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "n": int(diff.shape[0]),
        "diffs": diff.tolist(),
    }


def make_fair_mlp(config: PilotConfig, n_heads: int) -> MultiHeadMLPLearner:
    """Create the fair shared-trunk MLP baseline."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=(config.hidden_size,),
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=config.sparsity,
        use_layer_norm=True,
    )


def run_multihead_curve(
    learner: MultiHeadMLPLearner,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> np.ndarray:
    """Run a MultiHeadMLPLearner and return mean active-head loss."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    curve = jnp.nanmean(result.per_head_metrics[..., 0], axis=-1)
    curve.block_until_ready()
    return np.asarray(curve)


def random_features(observations: np.ndarray, seed: int, width: int) -> np.ndarray:
    """Fixed nonlinear features for readout-conditioning pilots."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0,
        1.0 / math.sqrt(observations.shape[1]),
        size=(width, observations.shape[1]),
    )
    bias = rng.normal(0.0, 0.25, size=width)
    hidden = np.tanh(observations @ weights.T + bias)
    return np.concatenate([hidden, np.ones((observations.shape[0], 1))], axis=1).astype(np.float32)


def rls_curve(features: np.ndarray, targets: np.ndarray, forgetting: float = 0.995) -> np.ndarray:
    """Vector-output recursive least squares over fixed features."""
    dim = features.shape[1]
    n_tasks = targets.shape[1]
    weights = np.zeros((n_tasks, dim), dtype=np.float64)
    p_mat = np.eye(dim, dtype=np.float64) * 10.0
    losses = np.empty(features.shape[0], dtype=np.float64)
    for idx, (phi, target) in enumerate(zip(features, targets, strict=True)):
        pred = weights @ phi
        err = target - pred
        losses[idx] = float(np.mean(err**2))
        denom = forgetting + float(phi @ p_mat @ phi)
        gain = (p_mat @ phi) / max(denom, 1e-8)
        weights += err[:, None] * gain[None, :]
        p_mat = (p_mat - np.outer(gain, phi @ p_mat)) / forgetting
    return losses.astype(np.float32)


def run_d01_rls(
    observations: jax.Array,
    targets: jax.Array,
    seed: int,
    config: PilotConfig,
) -> np.ndarray:
    """D01: RLS readout over fixed nonlinear features."""
    features = random_features(np.asarray(observations), seed, config.hidden_size)
    return rls_curve(features, np.asarray(targets))


def run_d02_aux(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D02: auxiliary next-observation random-projection prediction heads."""
    aux_dim = min(8, int(observations.shape[1]))
    proj_key, learner_key = jr.split(key)
    projection = jr.normal(proj_key, (aux_dim, observations.shape[1]), dtype=jnp.float32)
    projection = projection / jnp.sqrt(jnp.sum(projection**2, axis=1, keepdims=True) + 1e-6)
    next_obs = jnp.concatenate([observations[1:], observations[-1:]], axis=0)
    aux_targets = next_obs @ projection.T
    full_targets = jnp.concatenate([targets, 0.2 * aux_targets], axis=1)
    learner = MultiHeadMLPLearner(
        n_heads=int(full_targets.shape[1]),
        hidden_sizes=(config.hidden_size,),
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=config.sparsity,
        use_layer_norm=True,
    )
    state = learner.init(int(observations.shape[1]), learner_key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, target, primary = inputs
        result = learner.update(carry, obs, target)
        primary_pred = result.predictions[: targets.shape[1]]
        primary_loss = jnp.mean((primary_pred - primary) ** 2)
        return result.state, primary_loss

    _, curve = jax.lax.scan(step_fn, state, (observations, full_targets, targets))
    curve.block_until_ready()
    return np.asarray(curve)


def spline_features(observations: np.ndarray, knots_per_dim: int = 5) -> np.ndarray:
    """Fixed triangular spline features over each input coordinate."""
    centers = np.linspace(-2.5, 2.5, knots_per_dim, dtype=np.float32)
    width = float(centers[1] - centers[0]) if knots_per_dim > 1 else 1.0
    feats = [observations.astype(np.float32)]
    for dim in range(observations.shape[1]):
        values = observations[:, dim : dim + 1]
        hats = np.maximum(1.0 - np.abs(values - centers[None, :]) / width, 0.0)
        feats.append(hats.astype(np.float32))
    return np.concatenate(feats, axis=1)


def run_linear_feature_curve(
    features: np.ndarray,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """Run a linear multi-head learner on precomputed features."""
    learner = MultiHeadMLPLearner(
        n_heads=int(targets.shape[1]),
        hidden_sizes=(),
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=0.0,
        use_layer_norm=False,
    )
    return run_multihead_curve(learner, jnp.asarray(features), targets, key)


def run_d04_spline(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D04: fixed spline basis readout."""
    features = spline_features(np.asarray(observations), knots_per_dim=5)
    return run_linear_feature_curve(features, targets, key, config)


def online_whiten_features(observations: np.ndarray, decay: float = 0.02) -> np.ndarray:
    """Causal online input whitening proxy for D06."""
    dim = observations.shape[1]
    mean = np.zeros(dim, dtype=np.float64)
    cov = np.eye(dim, dtype=np.float64)
    out = np.empty_like(observations, dtype=np.float32)
    for idx, obs in enumerate(observations):
        centered = obs - mean
        vals, vecs = np.linalg.eigh(cov + 1e-3 * np.eye(dim))
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-4))) @ vecs.T
        out[idx] = (inv_sqrt @ centered).astype(np.float32)
        mean = (1.0 - decay) * mean + decay * obs
        diff = obs - mean
        cov = (1.0 - decay) * cov + decay * np.outer(diff, diff)
    return out


def run_d06_whitened_input(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D06 proxy: fair MLP on causally whitened inputs."""
    features = online_whiten_features(np.asarray(observations))
    learner = make_fair_mlp(config, int(targets.shape[1]))
    return run_multihead_curve(learner, jnp.asarray(features), targets, key)


def history_features(observations: np.ndarray) -> np.ndarray:
    """Causal lag/trace features for D09."""
    decays = (0.25, 0.5, 0.75, 0.9)
    traces = np.zeros((len(decays), observations.shape[1]), dtype=np.float32)
    prev = np.zeros_like(observations[0])
    rows: list[np.ndarray] = []
    for obs in observations.astype(np.float32):
        parts = [obs, prev]
        for idx, decay in enumerate(decays):
            traces[idx] = decay * traces[idx] + (1.0 - decay) * obs
            parts.append(traces[idx].copy())
        rows.append(np.concatenate(parts, axis=0))
        prev = obs.copy()
    return np.stack(rows, axis=0).astype(np.float32)


def run_d09_history(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D09: MLP on causal history features."""
    features = history_features(np.asarray(observations))
    learner = make_fair_mlp(config, int(targets.shape[1]))
    return run_multihead_curve(learner, jnp.asarray(features), targets, key)


def run_d03_dynamic_sparse(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D03: reuse the standalone dynamic sparse pilot learner."""
    _, _, metrics = run_dynamic_sparse_arrays(
        n_heads=int(targets.shape[1]),
        feature_dim=int(observations.shape[1]),
        observations=observations,
        targets=targets,
        key=key,
        hidden_size=config.hidden_size,
        step_size=config.step_size,
        sparsity=config.sparsity,
        utility_decay=0.99,
        rewire_interval=max(25, config.num_steps // 5),
        unit_replacement_rate=0.05,
    )
    return metrics[:, 0]


def run_d05_high_leak(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D05 proxy: higher LeakyReLU slope to reduce dormant units."""
    learner = MultiHeadMLPLearner(
        n_heads=int(targets.shape[1]),
        hidden_sizes=(config.hidden_size,),
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=config.sparsity,
        leaky_relu_slope=0.1,
        use_layer_norm=True,
    )
    return run_multihead_curve(learner, observations, targets, key)


def kernel_rls_curve(
    observations: np.ndarray,
    targets: np.ndarray,
    budget: int = 48,
    sigma: float | None = None,
    forgetting: float = 0.995,
) -> np.ndarray:
    """D07: budgeted Gaussian-kernel RLS with novelty-gated dictionary growth."""
    if sigma is None:
        sigma = math.sqrt(observations.shape[1])
    centers = np.zeros((budget, observations.shape[1]), dtype=np.float64)
    active = 0
    weights = np.zeros((targets.shape[1], budget), dtype=np.float64)
    p_mat = np.eye(budget, dtype=np.float64) * 10.0
    losses = np.empty(observations.shape[0], dtype=np.float64)

    for idx, (obs, target) in enumerate(zip(observations, targets, strict=True)):
        if active == 0:
            phi = np.zeros(budget, dtype=np.float64)
        else:
            dist2 = np.sum((centers[:active] - obs[None, :]) ** 2, axis=1)
            phi = np.zeros(budget, dtype=np.float64)
            phi[:active] = np.exp(-dist2 / (2.0 * sigma**2))

        pred = weights @ phi
        err = target - pred
        losses[idx] = float(np.mean(err**2))

        novelty = (
            np.inf
            if active == 0
            else float(np.min(np.sum((centers[:active] - obs[None, :]) ** 2, axis=1)))
        )
        if active < budget and novelty > 0.5:
            centers[active] = obs
            phi[active] = 1.0
            active += 1

        denom = forgetting + float(phi @ p_mat @ phi)
        gain = (p_mat @ phi) / max(denom, 1e-8)
        weights += err[:, None] * gain[None, :]
        p_mat = (p_mat - np.outer(gain, phi @ p_mat)) / forgetting
    return losses.astype(np.float32)


def run_d07_kernel(observations: jax.Array, targets: jax.Array) -> np.ndarray:
    """D07: budgeted kernel RLS."""
    return kernel_rls_curve(np.asarray(observations), np.asarray(targets))


def run_d08_independent(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D08 proxy: independent per-head trunks instead of shared trunk modulation."""
    keys = jr.split(key, int(targets.shape[1]))
    curves: list[np.ndarray] = []
    for head in range(int(targets.shape[1])):
        learner = MLPLearner(
            hidden_sizes=(config.hidden_size,),
            step_size=config.step_size,
            bounder=ObGDBounding(kappa=config.obgd_kappa),
            sparsity=config.sparsity,
            use_layer_norm=True,
        )
        state = learner.init(int(observations.shape[1]), keys[head])

        def step_fn(
            carry: Any,
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[Any, jax.Array]:
            obs, target = inputs
            result = learner.update(carry, obs, target)
            return result.state, result.metrics[0]

        _, curve = jax.lax.scan(step_fn, state, (observations, targets[:, head]))
        curve.block_until_ready()
        curves.append(np.asarray(curve))
    return np.mean(np.stack(curves, axis=0), axis=0)


def precision_weighted_linear_curve(
    observations: np.ndarray,
    targets: np.ndarray,
    step_size: float,
    decay: float = 0.02,
) -> np.ndarray:
    """D10 proxy: linear readout with residual-variance precision weighting."""
    features = np.concatenate(
        [observations.astype(np.float32), np.ones((observations.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    weights = np.zeros((targets.shape[1], features.shape[1]), dtype=np.float64)
    variance = np.ones(targets.shape[1], dtype=np.float64)
    losses = np.empty(observations.shape[0], dtype=np.float64)
    for idx, (phi, target) in enumerate(zip(features, targets, strict=True)):
        pred = weights @ phi
        err = target - pred
        losses[idx] = float(np.mean(err**2))
        precision = np.clip(1.0 / np.maximum(variance, 1e-4), 0.1, 8.0)
        weights += step_size * precision[:, None] * err[:, None] * phi[None, :]
        variance = (1.0 - decay) * variance + decay * err**2
    return losses.astype(np.float32)


def run_d10_precision(
    observations: jax.Array,
    targets: jax.Array,
    config: PilotConfig,
) -> np.ndarray:
    """D10 proxy: precision-weighted linear readout."""
    return precision_weighted_linear_curve(
        np.asarray(observations),
        np.asarray(targets),
        step_size=config.step_size,
    )


def run_method(
    method: str,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    seed: int,
    config: PilotConfig,
) -> np.ndarray:
    """Run one named method."""
    if method == "fair_mlp":
        return run_multihead_curve(
            make_fair_mlp(config, int(targets.shape[1])),
            observations,
            targets,
            key,
        )
    if method == "d01_rls_random_features":
        return run_d01_rls(observations, targets, seed, config)
    if method == "d02_aux_next_projection":
        return run_d02_aux(observations, targets, key, config)
    if method == "d03_dynamic_sparse":
        return run_d03_dynamic_sparse(observations, targets, key, config)
    if method == "d04_spline_basis":
        return run_d04_spline(observations, targets, key, config)
    if method == "d05_high_leak_homeostasis":
        return run_d05_high_leak(observations, targets, key, config)
    if method == "d06_online_input_whitening":
        return run_d06_whitened_input(observations, targets, key, config)
    if method == "d07_budgeted_kernel_rls":
        return run_d07_kernel(observations, targets)
    if method == "d08_independent_head_trunks":
        return run_d08_independent(observations, targets, key, config)
    if method == "d09_history_features":
        return run_d09_history(observations, targets, key, config)
    if method == "d10_precision_weighted_linear":
        return run_d10_precision(observations, targets, config)
    raise ValueError(f"unknown method {method!r}")


def hedge_curve(
    seed_curves: dict[str, np.ndarray],
    methods: Sequence[str],
    eta: float,
    discount: float,
) -> np.ndarray:
    """Causal loss-space Hedge curve over already-updating experts.

    The curve is the weighted loss of the selected expert pool at each step.
    It is a conservative proxy for a convex prediction mixture because squared
    loss is convex in predictions.
    """
    losses = np.stack([seed_curves[method] for method in methods], axis=1)
    log_weights = np.zeros(len(methods), dtype=np.float64)
    curve = np.empty(losses.shape[0], dtype=np.float64)
    for idx, loss_row in enumerate(losses):
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        curve[idx] = float(np.sum(weights * loss_row))
        log_weights = discount * log_weights - eta * loss_row
        log_weights = log_weights - np.max(log_weights)
    return curve.astype(np.float32)


def ema_gated_curve(
    baseline: np.ndarray,
    candidate: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Causal EMA gate between baseline and candidate loss curves."""
    baseline_ema = 0.0
    candidate_ema = 0.0
    curve = np.empty_like(baseline)
    for idx, (base_loss, candidate_loss) in enumerate(zip(baseline, candidate, strict=True)):
        if idx == 0 or candidate_ema <= baseline_ema:
            curve[idx] = candidate_loss
        else:
            curve[idx] = base_loss
        baseline_ema = (1.0 - decay) * baseline_ema + decay * float(base_loss)
        candidate_ema = (1.0 - decay) * candidate_ema + decay * float(candidate_loss)
    return curve.astype(np.float32)


def portfolio_curves(
    seed_curves: dict[str, np.ndarray],
    config: PilotConfig,
) -> dict[str, np.ndarray]:
    """Compute causal portfolios from per-method loss curves for one seed."""
    all_methods = tuple(seed_curves)
    all_hedge = hedge_curve(
        seed_curves,
        all_methods,
        eta=config.hedge_eta,
        discount=config.hedge_discount,
    )
    signal_hedge = hedge_curve(
        seed_curves,
        SIGNAL_METHODS,
        eta=config.hedge_eta,
        discount=config.hedge_discount,
    )
    return {
        "portfolio_all_hedge": all_hedge,
        "portfolio_signal_hedge": signal_hedge,
        "portfolio_signal_ema_gate": ema_gated_curve(
            seed_curves["fair_mlp"],
            signal_hedge,
            decay=config.gate_decay,
        ),
    }


def run_suite(name: str, config: PilotConfig) -> dict[str, Any]:
    """Run every pilot method on one suite."""
    stream, description = make_stream(name, config)
    base_methods = list(BASE_METHODS)
    methods = list(METHOD_STATUS)
    curves: dict[str, list[np.ndarray]] = {method: [] for method in methods}

    for seed in range(config.num_seeds):
        root_key = jr.key(seed + 100_000 * (1 + list(config.suites).index(name)))
        keys = jr.split(root_key, len(base_methods) + 1)
        observations, targets = collect_stream_arrays(stream, config.num_steps, keys[0])
        seed_curves: dict[str, np.ndarray] = {}
        for idx, method in enumerate(base_methods, start=1):
            curve = run_method(method, observations, targets, keys[idx], seed, config)
            seed_curves[method] = curve
            curves[method].append(curve)
        for method, curve in portfolio_curves(seed_curves, config).items():
            curves[method].append(curve)
        print(f"suite={name} seed={seed} complete")

    per_method = {
        method: summarize_curve_list(method_curves, config.final_window)
        for method, method_curves in curves.items()
    }
    baseline_finals = per_method["fair_mlp"]["per_seed_final_window_loss"]
    paired_vs_mlp = {
        method: paired_lower(per_method[method]["per_seed_final_window_loss"], baseline_finals)
        for method in methods
        if method != "fair_mlp"
    }
    best_method = min(
        methods,
        key=lambda method: per_method[method]["final_window_loss_mean"],
    )
    winners = [
        method
        for method, summary in paired_vs_mlp.items()
        if summary["diff_mean"] > 0.0 and summary["wins"] > summary["losses"]
    ]
    return {
        "description": description,
        "per_method": per_method,
        "paired_vs_fair_mlp": paired_vs_mlp,
        "best_method": best_method,
        "winners_vs_fair_mlp": winners,
    }


def run(config: PilotConfig) -> dict[str, Any]:
    """Run all configured suites."""
    t0 = time.time()
    suites = {name: run_suite(name, config) for name in config.suites}
    all_methods = [method for method in METHOD_STATUS if method != "fair_mlp"]
    win_counts = {
        method: sum(method in suite["winners_vs_fair_mlp"] for suite in suites.values())
        for method in all_methods
    }
    return {
        "experiment": "step2_new_direction_pilots",
        "config": asdict(config),
        "method_status": METHOD_STATUS,
        "elapsed_s": float(time.time() - t0),
        "suites": suites,
        "win_counts": win_counts,
    }


def format_metric(row: dict[str, Any], key: str) -> str:
    """Format mean +/- stderr."""
    return f"{row[key + '_mean']:.4f} +/- {row[key + '_stderr']:.4f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a Markdown summary."""
    lines = [
        "# Step 2 New Direction Pilots",
        "",
        "These are first-pass pilots for D01-D10 against a common fair MLP baseline. "
        "Proxy methods are explicitly marked; proxy wins require a full implementation "
        "before any canonical claim.",
        "",
        "## Overall Wins",
        "",
        "| Method | Status | Suite wins vs fair MLP |",
        "|---|---|---:|",
    ]
    for method, count in sorted(
        payload["win_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"| `{method}` | {payload['method_status'][method]} | {count} |")

    for suite_name, suite in payload["suites"].items():
        lines.extend([
            "",
            f"## Suite: {suite_name}",
            "",
            suite["description"],
            "",
            f"Best method by mean final-window loss: `{suite['best_method']}`.",
            "",
            "| Method | Final-window loss | Mean loss | Paired diff vs MLP | Wins/Losses |",
            "|---|---:|---:|---:|---:|",
        ])
        for method, row in sorted(
            suite["per_method"].items(),
            key=lambda item: item[1]["final_window_loss_mean"],
        ):
            if method == "fair_mlp":
                diff = "baseline"
                wins = "-"
            else:
                paired = suite["paired_vs_fair_mlp"][method]
                diff = f"{paired['diff_mean']:+.4f} +/- {paired['diff_stderr']:.4f}"
                wins = f"{paired['wins']}/{paired['losses']}"
            lines.append(
                f"| `{method}` | {format_metric(row, 'final_window_loss')} | "
                f"{format_metric(row, 'mean_loss')} | {diff} | {wins} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any], output_dir: Path, note_path: Path) -> None:
    """Write JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "SUMMARY.md", payload)
    write_summary(note_path, payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--final-window", type=int, default=None)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument("--gate-decay", type=float, default=0.02)
    parser.add_argument("--suites", nargs="+", default=["interaction", "nonlinear"])
    return parser.parse_args()


def main() -> None:
    """Run pilot suite."""
    args = parse_args()
    final_window = args.final_window or max(1, args.num_steps // 5)
    config = PilotConfig(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        final_window=final_window,
        hedge_eta=args.hedge_eta,
        hedge_discount=args.hedge_discount,
        gate_decay=args.gate_decay,
        suites=tuple(args.suites),
    )
    payload = run(config)
    write_outputs(payload, args.output_dir, args.note_path)
    print("win_counts=", payload["win_counts"])
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Direction 8: does feature/cumulant discovery improve downstream GVF RMSE?

This is a bridge evaluation, not another utility-only discovery diagnostic.
Each seed uses the same stream in two phases:

1. Discovery warmup:
   - Step 2 feature learners see raw observations and target transition
     cumulants, then their final constructed features are frozen.
   - CumulantDiscovery sees raw transitions and its final projections are
     frozen as auxiliary GVF cumulants.
2. Downstream nexting evaluation:
   - Horde GVFs are trained from scratch on the evaluation suffix.
   - Metrics are RMSE of target GVF predictions versus empirical forward-view
     returns from ``utils.nexting``.

The baselines are target-only linear and MLP Hordes on the given observations.
Discovery only counts as useful if it reduces downstream target GVF RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CumulantDiscovery,
    DemonType,
    FixedBudgetFeatureLearner,
    FixedBudgetInteractionLearner,
    GVFSpec,
    HordeLearner,
    InteractionFeatureDiscoveryStream,
    MaskMode,
    ObGDBounding,
    OffPolicyTDLinearLearner,
    PartialObservationWrapper,
    create_horde_spec,
    multi_channel_horizon_returns,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step3_tdgvf_hard")
MethodSpec = tuple[
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    tuple[int, ...],
    float,
    float,
    float,
    jnp.ndarray,
    dict[str, Any],
]
CandidateSpec = tuple[np.ndarray, list[str]]


def _target_specs(
    n_targets: int,
    gammas: tuple[float, ...],
    lamda: float,
) -> list[GVFSpec]:
    specs: list[GVFSpec] = []
    for target_idx in range(n_targets):
        for gamma in gammas:
            specs.append(
                GVFSpec(  # type: ignore[call-arg]
                    name=f"target{target_idx}_g{gamma:g}",
                    demon_type=DemonType.PREDICTION,
                    gamma=gamma,
                    lamda=lamda,
                    cumulant_index=target_idx,
                )
            )
    return specs


def _aux_specs(
    n_aux: int,
    gammas: tuple[float, ...],
    offset: int,
    lamda: float,
) -> list[GVFSpec]:
    specs: list[GVFSpec] = []
    for aux_idx in range(n_aux):
        for gamma in gammas:
            specs.append(
                GVFSpec(  # type: ignore[call-arg]
                    name=f"aux{aux_idx}_g{gamma:g}",
                    demon_type=DemonType.PREDICTION,
                    gamma=gamma,
                    lamda=lamda,
                    cumulant_index=offset + aux_idx,
                )
            )
    return specs


def collect_partial_interaction_arrays(
    *,
    seed: int,
    total_steps: int,
    feature_dim: int,
    n_targets: int,
    n_contexts: int,
    context_length: int,
    active_pairs: int,
    noise_std: float,
    include_squares: bool,
    hide_last_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect masked observations and unmasked stream targets."""
    inner = InteractionFeatureDiscoveryStream(
        feature_dim=feature_dim,
        n_tasks=n_targets,
        n_contexts=n_contexts,
        context_length=context_length,
        active_pairs_per_context=active_pairs,
        noise_std=noise_std,
        include_squares=include_squares,
    )
    mask = np.ones(feature_dim, dtype=bool)
    if hide_last_channels > 0:
        mask[-hide_last_channels:] = False
    stream = PartialObservationWrapper(
        inner,
        mode=MaskMode.FIXED,
        fixed_mask=jnp.asarray(mask),
        sentinel=0.0,
    )
    state = stream.init(jr.key(seed))
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(total_steps):
        timestep, state = stream.step(state, jnp.asarray(idx, dtype=jnp.int32))
        observations.append(np.asarray(timestep.observation, dtype=np.float32))
        targets.append(np.asarray(timestep.target, dtype=np.float32))
    return np.stack(observations), np.stack(targets)


def _mask_observations(
    observations: np.ndarray,
    hide_last_channels: int,
) -> np.ndarray:
    """Apply the same fixed mask convention as PartialObservationWrapper."""
    if hide_last_channels <= 0:
        return observations.astype(np.float32)
    masked = observations.copy()
    masked[:, -hide_last_channels:] = 0.0
    return masked.astype(np.float32)


def collect_markov_interaction_arrays(
    *,
    seed: int,
    total_steps: int,
    feature_dim: int,
    n_targets: int,
    n_contexts: int,
    context_length: int,
    active_pairs: int,
    noise_std: float,
    include_squares: bool,
    hide_last_channels: int,
    ar_rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect AR(1) observations with the same interaction target family.

    ``InteractionFeatureDiscoveryStream`` samples observations i.i.d.; then
    ``target_{t+1}`` is not state-predictable from ``obs_t``. This collector
    keeps the same hidden target construction but gives observations Markov
    dynamics, so pair/square features of ``obs_t`` have a fair chance to
    predict downstream GVF cumulants at ``t+1``.
    """
    if not 0.0 <= ar_rho < 1.0:
        raise ValueError("ar_rho must satisfy 0 <= ar_rho < 1")

    stream = InteractionFeatureDiscoveryStream(
        feature_dim=feature_dim,
        n_tasks=n_targets,
        n_contexts=n_contexts,
        context_length=context_length,
        active_pairs_per_context=active_pairs,
        noise_std=noise_std,
        include_squares=include_squares,
    )
    k_init, k_rollout = jr.split(jr.key(seed), 2)
    stream_state = stream.init(k_init)
    pair_left = np.asarray(stream_state.pair_left)
    pair_right = np.asarray(stream_state.pair_right)
    context_weights = np.asarray(stream_state.context_weights, dtype=np.float32)
    linear_weights = np.asarray(stream_state.linear_weights, dtype=np.float32)

    key = k_rollout
    key, k_x0 = jr.split(key)
    x = np.asarray(jr.normal(k_x0, (feature_dim,), dtype=jnp.float32), dtype=np.float32)
    innovation_scale = float(np.sqrt(max(1.0 - ar_rho**2, 0.0)))

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for step in range(total_steps):
        if step > 0:
            key, k_x = jr.split(key)
            innovation = np.asarray(
                jr.normal(k_x, (feature_dim,), dtype=jnp.float32),
                dtype=np.float32,
            )
            x = (ar_rho * x + innovation_scale * innovation).astype(np.float32)

        interactions = x[pair_left] * x[pair_right]
        context_idx = (step // context_length) % n_contexts
        target = context_weights[context_idx] @ interactions + linear_weights @ x
        if noise_std > 0.0:
            key, k_noise = jr.split(key)
            noise = np.asarray(
                noise_std * jr.normal(k_noise, (n_targets,), dtype=jnp.float32),
                dtype=np.float32,
            )
            target = target + noise

        observations.append(x.copy())
        targets.append(np.asarray(target, dtype=np.float32))

    return _mask_observations(np.stack(observations), hide_last_channels), np.stack(
        targets
    ).astype(np.float32)


def collect_coupled_hidden_ar1_arrays(
    *,
    seed: int,
    total_steps: int,
    feature_dim: int,
    n_targets: int,
    n_contexts: int,
    context_length: int,
    active_pairs: int,
    noise_std: float,
    include_squares: bool,
    hide_last_channels: int,
    ar_rho: float,
    hidden_coupling: float,
    hidden_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a simple POMDP-like AR(1) stream with masked latent channels.

    The returned observations hide the last ``hide_last_channels`` dimensions,
    but the targets are computed from the unmasked state. Hidden dimensions are
    driven by their own AR trace and the previous visible mean, so causal
    history/trace features have a grounded signal to exploit. This is still a
    synthetic positive-control probe, not a solved hidden-state benchmark.
    """
    if hide_last_channels <= 0:
        return collect_markov_interaction_arrays(
            seed=seed,
            total_steps=total_steps,
            feature_dim=feature_dim,
            n_targets=n_targets,
            n_contexts=n_contexts,
            context_length=context_length,
            active_pairs=active_pairs,
            noise_std=noise_std,
            include_squares=include_squares,
            hide_last_channels=0,
            ar_rho=ar_rho,
        )
    if hide_last_channels >= feature_dim:
        raise ValueError("hide_last_channels must leave at least one visible channel")
    if not 0.0 <= ar_rho < 1.0:
        raise ValueError("ar_rho must satisfy 0 <= ar_rho < 1")

    stream = InteractionFeatureDiscoveryStream(
        feature_dim=feature_dim,
        n_tasks=n_targets,
        n_contexts=n_contexts,
        context_length=context_length,
        active_pairs_per_context=active_pairs,
        noise_std=0.0,
        include_squares=include_squares,
    )
    k_init, key = jr.split(jr.key(seed), 2)
    stream_state = stream.init(k_init)
    pair_left = np.asarray(stream_state.pair_left)
    pair_right = np.asarray(stream_state.pair_right)
    context_weights = np.asarray(stream_state.context_weights, dtype=np.float32)
    linear_weights = np.asarray(stream_state.linear_weights, dtype=np.float32)

    visible_dim = feature_dim - hide_last_channels
    key, k_x0 = jr.split(key)
    visible = np.asarray(
        jr.normal(k_x0, (visible_dim,), dtype=jnp.float32),
        dtype=np.float32,
    )
    key, k_h0 = jr.split(key)
    hidden = np.asarray(
        jr.normal(k_h0, (hide_last_channels,), dtype=jnp.float32),
        dtype=np.float32,
    )
    innovation_scale = float(np.sqrt(max(1.0 - ar_rho**2, 0.0)))

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for step in range(total_steps):
        if step > 0:
            key, k_visible, k_hidden = jr.split(key, 3)
            visible_innovation = np.asarray(
                jr.normal(k_visible, (visible_dim,), dtype=jnp.float32),
                dtype=np.float32,
            )
            hidden_innovation = np.asarray(
                hidden_noise_std
                * jr.normal(k_hidden, (hide_last_channels,), dtype=jnp.float32),
                dtype=np.float32,
            )
            visible_mean = np.asarray(np.mean(visible), dtype=np.float32)
            visible = (
                ar_rho * visible + innovation_scale * visible_innovation
            ).astype(np.float32)
            hidden = (
                ar_rho * hidden + hidden_coupling * visible_mean + hidden_innovation
            ).astype(np.float32)

        x = np.concatenate([visible, hidden]).astype(np.float32)
        interactions = x[pair_left] * x[pair_right]
        context_idx = (step // context_length) % n_contexts
        target = context_weights[context_idx] @ interactions + linear_weights @ x
        target = target + 0.5 * np.mean(hidden)
        if noise_std > 0.0:
            key, k_noise = jr.split(key)
            noise = np.asarray(
                noise_std * jr.normal(k_noise, (n_targets,), dtype=jnp.float32),
                dtype=np.float32,
            )
            target = target + noise

        observations.append(x.copy())
        targets.append(np.asarray(target, dtype=np.float32))

    return _mask_observations(np.stack(observations), hide_last_channels), np.stack(
        targets
    ).astype(np.float32)


def transition_view(
    observations: np.ndarray,
    targets: np.ndarray,
    discovery_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-time arrays to transition arrays and split warmup/eval."""
    obs_t = observations[:-1]
    obs_tp1 = observations[1:]
    transition_cumulants = targets[1:]
    return (
        obs_t[:discovery_steps],
        obs_tp1[:discovery_steps],
        transition_cumulants[:discovery_steps],
        obs_t[discovery_steps:],
        obs_tp1[discovery_steps:],
        transition_cumulants[discovery_steps:],
    )


def repeat_by_horizon(cumulants: np.ndarray, gammas: tuple[float, ...]) -> np.ndarray:
    """Map (T, C) cumulants to demon order target0-gamma*, target1-gamma*, ..."""
    return np.repeat(cumulants, len(gammas), axis=1).astype(np.float32)


def target_forward_returns(cumulants: np.ndarray, gammas: tuple[float, ...]) -> np.ndarray:
    returns = multi_channel_horizon_returns(
        jnp.asarray(cumulants, dtype=jnp.float32),
        jnp.asarray(gammas, dtype=jnp.float32),
        terminal_value=0.0,
    )
    return np.asarray(returns, dtype=np.float32).reshape(cumulants.shape[0], -1)


def bounded_sigmoid_policy_probs(
    observations: np.ndarray,
    *,
    scale: float,
    invert: bool = False,
) -> np.ndarray:
    """Return binary-action probabilities bounded away from 0 and 1."""
    logits = scale * observations[:, 0]
    if invert:
        logits = -logits
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return np.asarray(0.05 + 0.90 * probs, dtype=np.float32)


def summarize_target_rmse(
    predictions: np.ndarray,
    returns: np.ndarray,
    n_targets: int,
    gammas: tuple[float, ...],
    burn_in: int,
    burn_tail: int,
) -> dict[str, float]:
    end = predictions.shape[0] - burn_tail if burn_tail > 0 else predictions.shape[0]
    if burn_in >= end:
        raise ValueError(
            f"burn_in={burn_in} and burn_tail={burn_tail} leave no evaluation steps"
        )
    errors = predictions[burn_in:end] - returns[burn_in:end]
    rmse_flat = np.sqrt(np.mean(errors**2, axis=0))
    rmse = rmse_flat.reshape(n_targets, len(gammas))
    summary: dict[str, float] = {
        "target_rmse_mean": float(np.mean(rmse)),
        "target_rmse_max": float(np.max(rmse)),
    }
    for gamma_idx, gamma in enumerate(gammas):
        summary[f"rmse_gamma_{gamma:g}"] = float(np.mean(rmse[:, gamma_idx]))
    return summary


def summarize_scalar_rmse(
    predictions: np.ndarray,
    returns: np.ndarray,
    burn_in: int,
    burn_tail: int,
) -> dict[str, float]:
    end = predictions.shape[0] - burn_tail if burn_tail > 0 else predictions.shape[0]
    if burn_in >= end:
        raise ValueError(
            f"burn_in={burn_in} and burn_tail={burn_tail} leave no evaluation steps"
        )
    errors = predictions[burn_in:end] - returns[burn_in:end]
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "target_rmse_mean": rmse,
        "target_rmse_max": rmse,
        "rmse_gamma_off_policy": rmse,
    }


def run_off_policy_td_probe(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    reward_signal: np.ndarray,
    key: jnp.ndarray,
    gamma: float,
    step_size: float,
    trace_decay: float,
    policy_scale: float,
    retrace_clip: float,
    use_importance_sampling: bool,
    sampled_rewards: np.ndarray | None = None,
    sampled_ratios: np.ndarray | None = None,
    target_returns: np.ndarray | None = None,
    ratio_stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run a binary-action behavior-mismatch TD probe.

    State dynamics are exogenous, so the target-policy value is the discounted
    return of the target policy's expected reward on the same state sequence.
    The sampled reward comes from behavior actions, and the TD learner receives
    either the true per-decision ratio or ``1`` for the uncorrected baseline.
    """
    if sampled_rewards is None or sampled_ratios is None or target_returns is None:
        sampled_rewards, sampled_ratios, target_returns, ratio_stats = (
            sample_off_policy_behavior_rollout(
                observations=observations,
                reward_signal=reward_signal,
                key=key,
                gamma=gamma,
                policy_scale=policy_scale,
            )
        )
    else:
        if sampled_rewards.shape[0] != observations.shape[0]:
            raise ValueError("sampled_rewards must match observations length")
        if sampled_ratios.shape[0] != observations.shape[0]:
            raise ValueError("sampled_ratios must match observations length")
        if target_returns.shape[0] != observations.shape[0]:
            raise ValueError("target_returns must match observations length")
        if ratio_stats is None:
            ratio_stats = {
                "rho_mean": float(np.mean(sampled_ratios)),
                "rho_max": float(np.max(sampled_ratios)),
                "rho_std": float(np.std(sampled_ratios, ddof=0)),
                "behavior_prob_mean": float("nan"),
                "target_prob_mean": float("nan"),
            }
    rewards = np.asarray(sampled_rewards, dtype=np.float32)
    ratios = np.asarray(sampled_ratios, dtype=np.float32)
    returns = np.asarray(target_returns, dtype=np.float32)

    learner = OffPolicyTDLinearLearner(
        step_size=step_size,
        trace_decay=trace_decay,
        retrace_clip=retrace_clip,
    )
    state = learner.init(observations.shape[1])
    predictions = np.zeros(observations.shape[0], dtype=np.float32)
    rhos_used = ratios if use_importance_sampling else np.ones_like(ratios)
    for t in range(observations.shape[0]):
        result = learner.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(rewards[t]),
            jnp.asarray(next_observations[t]),
            jnp.asarray(gamma, dtype=jnp.float32),
            jnp.asarray(rhos_used[t]),
        )
        predictions[t] = float(np.asarray(result.prediction)[0])
        state = result.state

    return predictions, returns, ratio_stats


def sample_off_policy_behavior_rollout(
    *,
    observations: np.ndarray,
    reward_signal: np.ndarray,
    key: jnp.ndarray,
    gamma: float,
    policy_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Sample one behavior-policy rollout for fair paired off-policy rows."""
    behavior_probs = bounded_sigmoid_policy_probs(
        observations, scale=policy_scale, invert=True
    )
    target_probs = bounded_sigmoid_policy_probs(observations, scale=policy_scale)
    uniforms = np.asarray(jr.uniform(key, (observations.shape[0],)), dtype=np.float32)
    actions = uniforms < behavior_probs
    behavior_action_probs = np.where(actions, behavior_probs, 1.0 - behavior_probs)
    target_action_probs = np.where(actions, target_probs, 1.0 - target_probs)
    ratios = np.asarray(target_action_probs / behavior_action_probs, dtype=np.float32)
    rewards = np.asarray(reward_signal * np.where(actions, 1.0, -1.0), dtype=np.float32)
    expected_target_rewards = np.asarray(
        reward_signal * (2.0 * target_probs - 1.0), dtype=np.float32
    )
    returns = target_forward_returns(expected_target_rewards[:, None], (gamma,))[:, 0]
    ratio_stats = {
        "rho_mean": float(np.mean(ratios)),
        "rho_max": float(np.max(ratios)),
        "rho_std": float(np.std(ratios, ddof=0)),
        "behavior_prob_mean": float(np.mean(behavior_probs)),
        "target_prob_mean": float(np.mean(target_probs)),
    }
    return rewards, ratios, returns, ratio_stats


def make_horde(
    *,
    n_targets: int,
    gammas: tuple[float, ...],
    n_aux: int,
    feature_dim: int,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    trace_decay: float,
    key: jnp.ndarray,
) -> tuple[HordeLearner, Any]:
    target_specs = _target_specs(n_targets, gammas, trace_decay)
    aux_specs = (
        _aux_specs(n_aux, gammas, offset=n_targets, lamda=trace_decay)
        if n_aux
        else []
    )
    horde = HordeLearner(
        horde_spec=create_horde_spec([*target_specs, *aux_specs]),
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=sparsity,
        use_layer_norm=len(hidden_sizes) > 0,
    )
    return horde, horde.init(feature_dim, key)


def run_horde_predictions(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    target_cumulants: np.ndarray,
    gammas: tuple[float, ...],
    n_targets: int,
    key: jnp.ndarray,
    hidden_sizes: tuple[int, ...],
    step_size: float,
    sparsity: float,
    trace_decay: float = 0.0,
    aux_cumulants: np.ndarray | None = None,
) -> np.ndarray:
    n_aux = 0 if aux_cumulants is None else aux_cumulants.shape[1]
    horde, state = make_horde(
        n_targets=n_targets,
        gammas=gammas,
        n_aux=n_aux,
        feature_dim=observations.shape[1],
        hidden_sizes=hidden_sizes,
        step_size=step_size,
        sparsity=sparsity,
        trace_decay=trace_decay,
        key=key,
    )
    target_inputs = repeat_by_horizon(target_cumulants, gammas)
    if aux_cumulants is None:
        all_cumulants = target_inputs
    else:
        all_cumulants = np.concatenate(
            [target_inputs, repeat_by_horizon(aux_cumulants, gammas)], axis=1
        )

    n_target_heads = n_targets * len(gammas)
    predictions = np.zeros((observations.shape[0], n_target_heads), dtype=np.float32)
    for t in range(observations.shape[0]):
        result = horde.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(all_cumulants[t]),
            jnp.asarray(next_observations[t]),
        )
        predictions[t] = np.asarray(result.predictions[:n_target_heads])
        state = result.state
    return predictions


def run_gvf_feedback_predictions(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    target_cumulants: np.ndarray,
    gammas: tuple[float, ...],
    n_targets: int,
    key: jnp.ndarray,
    source_step_size: float,
    downstream_step_size: float,
    trace_decay: float,
) -> np.ndarray:
    """Train a downstream Horde on raw observations plus causal GVF predictions.

    Feedback features are computed from a separate source Horde's state before
    either learner consumes the current transition. That keeps the feature at
    ``s_t`` and the bootstrap feature at ``s_{t+1}`` causal with respect to the
    same time step.
    """
    k_source, k_downstream = jr.split(key, 2)
    source_horde, source_state = make_horde(
        n_targets=n_targets,
        gammas=gammas,
        n_aux=0,
        feature_dim=observations.shape[1],
        hidden_sizes=(),
        step_size=source_step_size,
        sparsity=0.0,
        trace_decay=trace_decay,
        key=k_source,
    )
    n_target_heads = n_targets * len(gammas)
    downstream_horde, downstream_state = make_horde(
        n_targets=n_targets,
        gammas=gammas,
        n_aux=0,
        feature_dim=observations.shape[1] + n_target_heads,
        hidden_sizes=(),
        step_size=downstream_step_size,
        sparsity=0.0,
        trace_decay=trace_decay,
        key=k_downstream,
    )
    target_inputs = repeat_by_horizon(target_cumulants, gammas)
    predictions = np.zeros((observations.shape[0], n_target_heads), dtype=np.float32)

    for t in range(observations.shape[0]):
        feedback_t = source_horde.predict(source_state, jnp.asarray(observations[t]))
        feedback_tp1 = source_horde.predict(
            source_state, jnp.asarray(next_observations[t])
        )
        aug_t = jnp.concatenate([jnp.asarray(observations[t]), feedback_t])
        aug_tp1 = jnp.concatenate([jnp.asarray(next_observations[t]), feedback_tp1])
        downstream_result = downstream_horde.update(
            downstream_state,
            aug_t,
            jnp.asarray(target_inputs[t]),
            aug_tp1,
        )
        source_result = source_horde.update(
            source_state,
            jnp.asarray(observations[t]),
            jnp.asarray(target_inputs[t]),
            jnp.asarray(next_observations[t]),
        )
        predictions[t] = np.asarray(downstream_result.predictions[:n_target_heads])
        downstream_state = downstream_result.state
        source_state = source_result.state
    return predictions


def warm_step2_tanh_features(
    obs: np.ndarray,
    cumulants: np.ndarray,
    args: argparse.Namespace,
    key: jnp.ndarray,
) -> tuple[FixedBudgetFeatureLearner, Any]:
    learner = FixedBudgetFeatureLearner(
        n_features=args.n_features,
        n_tasks=cumulants.shape[1],
        step_size_output=args.feature_step_size_output,
        step_size_feature=args.feature_step_size_constructor,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_count=args.candidate_count,
        candidate_min_age=args.candidate_min_age,
        promotion_margin=1.05,
        promotion_blend=0.5,
        generator_mix=(0.35, 0.45, 0.20),
        use_obgd=True,
        obgd_kappa=2.0,
    )
    state = learner.init(obs.shape[1], key)
    for t in range(obs.shape[0]):
        result = learner.update(state, jnp.asarray(obs[t]), jnp.asarray(cumulants[t]))
        state = result.state
    return learner, state


def warm_step2_interaction_features(
    obs: np.ndarray,
    cumulants: np.ndarray,
    args: argparse.Namespace,
    key: jnp.ndarray,
) -> tuple[FixedBudgetInteractionLearner, Any]:
    learner = FixedBudgetInteractionLearner(
        n_features=args.n_features,
        n_tasks=cumulants.shape[1],
        step_size_output=args.feature_step_size_output,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_count=args.candidate_count,
        candidate_min_age=args.candidate_min_age,
        promotion_margin=1.05,
        promotion_blend=0.5,
        candidate_strategy="all_pairs",
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        include_squares=args.include_squares,
        use_obgd=True,
        obgd_kappa=2.0,
    )
    state = learner.init(obs.shape[1], key)
    for t in range(obs.shape[0]):
        result = learner.update(state, jnp.asarray(obs[t]), jnp.asarray(cumulants[t]))
        state = result.state
    return learner, state


def augment_with_frozen_features(
    learner: FixedBudgetFeatureLearner | FixedBudgetInteractionLearner,
    state: Any,
    observations: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(learner.augmented_observation(state, jnp.asarray(obs)))
            for obs in observations
        ]
    ).astype(np.float32)


def augment_with_all_interactions(
    observations: np.ndarray,
    *,
    include_squares: bool,
) -> np.ndarray:
    """Append every pair/square product available from the observed channels."""
    interactions, _ = interaction_candidate_values(
        observations,
        include_squares=include_squares,
    )
    if interactions.shape[1] == 0:
        return observations.astype(np.float32)
    return np.concatenate([observations, interactions], axis=1).astype(np.float32)


def interaction_candidate_values(
    observations: np.ndarray,
    *,
    include_squares: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return causal interaction candidates and their stable pair order."""
    pairs: list[tuple[int, int]] = []
    for i in range(observations.shape[1]):
        start = i if include_squares else i + 1
        for j in range(start, observations.shape[1]):
            pairs.append((i, j))
    if not pairs:
        empty = np.zeros((observations.shape[0], 0), dtype=np.float32)
        return empty, pairs
    left = np.asarray([p[0] for p in pairs], dtype=np.int32)
    right = np.asarray([p[1] for p in pairs], dtype=np.int32)
    interactions = observations[:, left] * observations[:, right]
    return interactions.astype(np.float32), pairs


def augment_with_selected_interactions(
    observations: np.ndarray,
    *,
    selected_indices: np.ndarray,
    include_squares: bool,
) -> np.ndarray:
    """Append selected interaction candidates in canonical candidate order."""
    candidates, _ = interaction_candidate_values(
        observations,
        include_squares=include_squares,
    )
    if selected_indices.size == 0:
        return observations.astype(np.float32)
    selected = candidates[:, selected_indices]
    return np.concatenate([observations, selected], axis=1).astype(np.float32)


def _candidate_scales(candidates: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
    """Return fixed warmup scales for selected candidate columns."""
    if selected_indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    selected = candidates[:, selected_indices]
    scales = np.sqrt(np.mean(selected**2, axis=0))
    return np.asarray(np.maximum(scales, 1e-6), dtype=np.float32)


def select_novel_candidate_indices(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    n_select: int,
    max_abs_corr: float,
) -> np.ndarray:
    """Select high-score candidates while rejecting near-duplicate columns.

    Novelty is measured only on the discovery prefix. If the correlation gate
    cannot fill the budget, the remaining slots fall back to score order so the
    downstream feature dimensionality is stable across seeds.
    """
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2-D array")
    if scores.shape[0] != candidates.shape[1]:
        raise ValueError("scores must contain one value per candidate column")
    if n_select <= 0 or candidates.shape[1] == 0:
        return np.zeros((0,), dtype=np.int32)
    if not 0.0 <= max_abs_corr <= 1.0:
        raise ValueError("max_abs_corr must lie in [0, 1]")

    n_top = min(n_select, candidates.shape[1])
    order = np.argsort(-scores, kind="stable")
    centered = candidates.astype(np.float64) - np.mean(
        candidates, axis=0, keepdims=True
    )
    scale = np.sqrt(np.mean(centered**2, axis=0, keepdims=True))
    standardized = centered / np.maximum(scale, 1e-8)

    selected: list[int] = []
    for idx in order:
        if len(selected) >= n_top:
            break
        if not selected:
            selected.append(int(idx))
            continue
        corr = np.abs(
            standardized[:, idx : idx + 1].T @ standardized[:, selected]
            / standardized.shape[0]
        )
        if float(np.max(corr)) <= max_abs_corr:
            selected.append(int(idx))

    if len(selected) < n_top:
        selected_set = set(selected)
        for idx in order:
            idx_int = int(idx)
            if idx_int not in selected_set:
                selected.append(idx_int)
                selected_set.add(idx_int)
            if len(selected) >= n_top:
                break

    return np.asarray(selected, dtype=np.int32)


def _append_selected_scaled_candidates(
    observations: np.ndarray,
    candidates: np.ndarray,
    *,
    selected_indices: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if selected_indices.size == 0:
        return observations.astype(np.float32)
    selected = candidates[:, selected_indices] / scales
    return np.concatenate([observations, selected], axis=1).astype(np.float32)


def predictive_state_candidate_values(
    observations: np.ndarray,
    *,
    lags: tuple[int, ...],
    trace_rhos: tuple[float, ...],
    include_cross_products: bool,
) -> CandidateSpec:
    """Return causal history/trace candidates plus optional current-by-state products.

    The candidates for row ``t`` only depend on observations ``0..t``. Cross
    products let a linear downstream GVF represent interactions between the
    current visible state and a learned predictive-state trace.
    """
    history_aug = augment_with_history_trace_features(
        observations,
        lags=lags,
        trace_rhos=trace_rhos,
    )
    raw_dim = observations.shape[1]
    state_candidates = history_aug[:, raw_dim:]
    names: list[str] = []
    for lag in lags:
        names.extend([f"lag{lag}_x{i}" for i in range(raw_dim)])
    for rho in trace_rhos:
        names.extend([f"trace{rho:g}_x{i}" for i in range(raw_dim)])

    pieces = []
    if state_candidates.shape[1] > 0:
        pieces.append(state_candidates.astype(np.float32))
    if include_cross_products and state_candidates.shape[1] > 0:
        cross = (
            observations[:, :, None] * state_candidates[:, None, :]
        ).reshape(observations.shape[0], -1)
        pieces.append(cross.astype(np.float32))
        base_names = list(names)
        names.extend(
            f"x{i}*{base_name}"
            for i in range(raw_dim)
            for base_name in base_names
        )
    if not pieces:
        return np.zeros((observations.shape[0], 0), dtype=np.float32), []
    return np.concatenate(pieces, axis=1).astype(np.float32), names


def augment_with_selected_predictive_state_features(
    observations: np.ndarray,
    *,
    selected_indices: np.ndarray,
    scales: np.ndarray,
    lags: tuple[int, ...],
    trace_rhos: tuple[float, ...],
    include_cross_products: bool,
) -> np.ndarray:
    candidates, _ = predictive_state_candidate_values(
        observations,
        lags=lags,
        trace_rhos=trace_rhos,
        include_cross_products=include_cross_products,
    )
    return _append_selected_scaled_candidates(
        observations,
        candidates,
        selected_indices=selected_indices,
        scales=scales,
    )


def score_predictive_state_feature_candidates(
    *,
    observations: np.ndarray,
    target_cumulants: np.ndarray,
    n_select: int,
    lags: tuple[int, ...],
    trace_rhos: tuple[float, ...],
    include_cross_products: bool,
    score_decay: float,
    importance_ratios: np.ndarray | None = None,
    importance_clip: float | None = None,
) -> dict[str, Any]:
    """Score causal predictive-state features by future-cumulant utility.

    This is an online correlation proxy: each candidate at ``s_t`` is scored by
    its clipped-importance-weighted correlation with transition cumulants
    observed after ``s_t``. It uses only the discovery prefix and freezes the
    selected columns and scales before downstream evaluation.
    """
    candidates, names = predictive_state_candidate_values(
        observations,
        lags=lags,
        trace_rhos=trace_rhos,
        include_cross_products=include_cross_products,
    )
    if candidates.shape[1] == 0 or n_select <= 0:
        return {
            "indices": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((candidates.shape[1],), dtype=np.float32),
            "names": names,
            "scales": np.zeros((0,), dtype=np.float32),
            "selected_score_mean": 0.0,
            "importance_weight_mean": 1.0,
            "importance_weight_max": 1.0,
        }
    if not 0.0 <= score_decay < 1.0:
        raise ValueError("score_decay must satisfy 0 <= decay < 1")
    if importance_ratios is not None and importance_ratios.shape[0] != observations.shape[0]:
        raise ValueError("importance_ratios must match observations length")

    cross = np.zeros((candidates.shape[1], target_cumulants.shape[1]), dtype=np.float32)
    cand_sq = np.zeros(candidates.shape[1], dtype=np.float32)
    target_sq = np.zeros(target_cumulants.shape[1], dtype=np.float32)
    weights_seen: list[float] = []
    for t in range(observations.shape[0]):
        weight = 1.0
        if importance_ratios is not None:
            weight = float(importance_ratios[t])
            if importance_clip is not None:
                weight = min(weight, importance_clip)
        weights_seen.append(weight)
        candidate_t = candidates[t]
        target_t = target_cumulants[t]
        cross = (
            score_decay * cross
            + weight * np.outer(candidate_t, target_t)
        ).astype(np.float32)
        cand_sq = (score_decay * cand_sq + weight * candidate_t**2).astype(np.float32)
        target_sq = (score_decay * target_sq + weight * target_t**2).astype(np.float32)

    denom = np.sqrt(np.maximum(cand_sq[:, None] * target_sq[None, :], 1e-8))
    scores = np.asarray(np.mean(np.abs(cross) / denom, axis=1), dtype=np.float32)
    n_top = min(n_select, scores.shape[0])
    selected = np.argsort(-scores, kind="stable")[:n_top].astype(np.int32)
    selected_scores = scores[selected] if selected.size else np.array([])
    return {
        "indices": selected,
        "scores": scores,
        "names": names,
        "scales": _candidate_scales(candidates, selected),
        "selected_score_mean": float(np.mean(selected_scores)) if selected.size else 0.0,
        "importance_weight_mean": float(np.mean(weights_seen)),
        "importance_weight_max": float(np.max(weights_seen)),
    }


def behavior_mismatch_importance_ratios(
    observations: np.ndarray,
    *,
    key: jnp.ndarray,
    policy_scale: float,
) -> np.ndarray:
    """Sample discovery-time behavior actions and return target/behavior ratios."""
    behavior_probs = bounded_sigmoid_policy_probs(
        observations, scale=policy_scale, invert=True
    )
    target_probs = bounded_sigmoid_policy_probs(observations, scale=policy_scale)
    uniforms = np.asarray(jr.uniform(key, (observations.shape[0],)), dtype=np.float32)
    actions = uniforms < behavior_probs
    behavior_action_probs = np.where(actions, behavior_probs, 1.0 - behavior_probs)
    target_action_probs = np.where(actions, target_probs, 1.0 - target_probs)
    return np.asarray(target_action_probs / behavior_action_probs, dtype=np.float32)


def _off_policy_sampled_rewards_and_ratios(
    observations: np.ndarray,
    reward_signal: np.ndarray,
    *,
    key: jnp.ndarray,
    policy_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    behavior_probs = bounded_sigmoid_policy_probs(
        observations, scale=policy_scale, invert=True
    )
    target_probs = bounded_sigmoid_policy_probs(observations, scale=policy_scale)
    uniforms = np.asarray(jr.uniform(key, (observations.shape[0],)), dtype=np.float32)
    actions = uniforms < behavior_probs
    behavior_action_probs = np.where(actions, behavior_probs, 1.0 - behavior_probs)
    target_action_probs = np.where(actions, target_probs, 1.0 - target_probs)
    ratios = np.asarray(target_action_probs / behavior_action_probs, dtype=np.float32)
    rewards = np.asarray(reward_signal * np.where(actions, 1.0, -1.0), dtype=np.float32)
    stats = {
        "rho_mean": float(np.mean(ratios)),
        "rho_max": float(np.max(ratios)),
        "rho_std": float(np.std(ratios, ddof=0)),
        "behavior_prob_mean": float(np.mean(behavior_probs)),
        "target_prob_mean": float(np.mean(target_probs)),
    }
    return rewards, ratios, stats


def score_off_policy_mspbe_feature_candidates(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    candidate_values: np.ndarray,
    next_candidate_values: np.ndarray,
    reward_signal: np.ndarray,
    key: jnp.ndarray,
    gamma: float,
    step_size: float,
    trace_decay: float,
    policy_scale: float,
    retrace_clip: float,
    n_select: int,
    score_decay: float,
) -> dict[str, Any]:
    """Score features by a clipped-IS Bellman-residual gradient proxy.

    The proxy is the online correlation between the TD error and each
    candidate's Bellman feature difference ``phi_t - gamma * phi_{t+1}``,
    weighted by the same clipped ratio used by the off-policy TD update. This
    approximates a one-weight MSPBE reduction signal without a secondary
    gradient-TD learner.
    """
    if candidate_values.shape != next_candidate_values.shape:
        raise ValueError("candidate_values and next_candidate_values must match")
    if candidate_values.shape[0] != observations.shape[0]:
        raise ValueError("candidate_values must match observations length")
    if candidate_values.shape[1] == 0 or n_select <= 0:
        return {
            "indices": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((candidate_values.shape[1],), dtype=np.float32),
            "scales": np.zeros((0,), dtype=np.float32),
            "selected_score_mean": 0.0,
            "importance_weight_mean": 1.0,
            "importance_weight_max": 1.0,
        }
    if not 0.0 <= score_decay < 1.0:
        raise ValueError("score_decay must satisfy 0 <= decay < 1")

    rewards, ratios, ratio_stats = _off_policy_sampled_rewards_and_ratios(
        observations,
        reward_signal,
        key=key,
        policy_scale=policy_scale,
    )
    learner = OffPolicyTDLinearLearner(
        step_size=step_size,
        trace_decay=trace_decay,
        retrace_clip=retrace_clip,
    )
    state = learner.init(observations.shape[1])
    cross = np.zeros(candidate_values.shape[1], dtype=np.float32)
    diff_sq = np.zeros(candidate_values.shape[1], dtype=np.float32)
    delta_sq = 0.0
    weights_seen: list[float] = []
    for t in range(observations.shape[0]):
        pred_t = float(np.asarray(learner.predict(state, jnp.asarray(observations[t])))[0])
        pred_tp1 = float(
            np.asarray(learner.predict(state, jnp.asarray(next_observations[t])))[0]
        )
        delta = float(rewards[t] + gamma * pred_tp1 - pred_t)
        weight = min(float(ratios[t]), retrace_clip)
        weights_seen.append(weight)
        bellman_diff = candidate_values[t] - gamma * next_candidate_values[t]
        cross = (score_decay * cross + weight * delta * bellman_diff).astype(
            np.float32
        )
        diff_sq = (score_decay * diff_sq + weight * bellman_diff**2).astype(np.float32)
        delta_sq = score_decay * delta_sq + weight * delta**2
        result = learner.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(rewards[t]),
            jnp.asarray(next_observations[t]),
            jnp.asarray(gamma, dtype=jnp.float32),
            jnp.asarray(ratios[t], dtype=jnp.float32),
        )
        state = result.state

    scores = np.asarray(
        np.abs(cross) / np.sqrt(np.maximum(diff_sq * delta_sq, 1e-8)),
        dtype=np.float32,
    )
    n_top = min(n_select, scores.shape[0])
    selected = np.argsort(-scores, kind="stable")[:n_top].astype(np.int32)
    selected_scores = scores[selected] if selected.size else np.array([])
    return {
        "indices": selected,
        "scores": scores,
        "scales": _candidate_scales(candidate_values, selected),
        "selected_score_mean": float(np.mean(selected_scores)) if selected.size else 0.0,
        "importance_weight_mean": float(np.mean(weights_seen)),
        "importance_weight_max": float(np.max(weights_seen)),
        **ratio_stats,
    }


def score_td_surprise_interaction_candidates(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    target_cumulants: np.ndarray,
    gammas: tuple[float, ...],
    n_targets: int,
    key: jnp.ndarray,
    n_select: int,
    include_squares: bool,
    step_size: float,
    trace_decay: float,
    candidate_trace_rho: float,
    score_decay: float,
    importance_ratios: np.ndarray | None = None,
    importance_clip: float | None = None,
) -> dict[str, Any]:
    """Score interaction candidates with online TD-error surprise.

    A raw linear source Horde is updated every discovery step. Before each
    update, its one-step TD errors weight causal candidate eligibility traces.
    Optional clipped importance ratios make the same scorer usable in the
    behavior-mismatch probe without changing the temporal order.
    """
    candidates, pairs = interaction_candidate_values(
        observations,
        include_squares=include_squares,
    )
    if candidates.shape[1] == 0 or n_select <= 0:
        return {
            "indices": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((candidates.shape[1],), dtype=np.float32),
            "pairs": pairs,
            "selected_score_mean": 0.0,
            "importance_weight_mean": 1.0,
            "importance_weight_max": 1.0,
        }
    if not 0.0 <= candidate_trace_rho < 1.0:
        raise ValueError("candidate_trace_rho must satisfy 0 <= rho < 1")
    if not 0.0 <= score_decay < 1.0:
        raise ValueError("score_decay must satisfy 0 <= decay < 1")
    if importance_ratios is not None and importance_ratios.shape[0] != observations.shape[0]:
        raise ValueError("importance_ratios must match observations length")

    horde, state = make_horde(
        n_targets=n_targets,
        gammas=gammas,
        n_aux=0,
        feature_dim=observations.shape[1],
        hidden_sizes=(),
        step_size=step_size,
        sparsity=0.0,
        trace_decay=trace_decay,
        key=key,
    )
    gamma_heads = np.tile(np.asarray(gammas, dtype=np.float32), n_targets)
    target_inputs = repeat_by_horizon(target_cumulants, gammas)
    traces = np.zeros(candidates.shape[1], dtype=np.float32)
    scores = np.zeros(candidates.shape[1], dtype=np.float32)
    normalizer = np.zeros(candidates.shape[1], dtype=np.float32)
    weights_seen: list[float] = []

    for t in range(observations.shape[0]):
        pred_t = np.asarray(horde.predict(state, jnp.asarray(observations[t])))
        pred_tp1 = np.asarray(horde.predict(state, jnp.asarray(next_observations[t])))
        td_errors = target_inputs[t] + gamma_heads * pred_tp1 - pred_t
        surprise = float(np.mean(np.abs(td_errors)))
        weight = 1.0
        if importance_ratios is not None:
            weight = float(importance_ratios[t])
            if importance_clip is not None:
                weight = min(weight, importance_clip)
        weights_seen.append(weight)
        traces = (candidate_trace_rho * traces + candidates[t]).astype(np.float32)
        scores = (score_decay * scores + weight * surprise * np.abs(traces)).astype(
            np.float32
        )
        normalizer = (score_decay * normalizer + np.abs(candidates[t])).astype(
            np.float32
        )
        result = horde.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(target_inputs[t]),
            jnp.asarray(next_observations[t]),
        )
        state = result.state

    normalized_scores = np.asarray(scores / np.maximum(normalizer, 1e-8), dtype=np.float32)
    n_top = min(n_select, normalized_scores.shape[0])
    selected = np.argsort(-normalized_scores, kind="stable")[:n_top].astype(np.int32)
    selected_scores = normalized_scores[selected] if selected.size else np.array([])
    return {
        "indices": selected,
        "scores": normalized_scores,
        "pairs": pairs,
        "selected_score_mean": float(np.mean(selected_scores)) if selected.size else 0.0,
        "importance_weight_mean": float(np.mean(weights_seen)),
        "importance_weight_max": float(np.max(weights_seen)),
    }


def score_meta_gradient_interaction_candidates(
    *,
    observations: np.ndarray,
    next_observations: np.ndarray,
    target_cumulants: np.ndarray,
    gammas: tuple[float, ...],
    n_targets: int,
    key: jnp.ndarray,
    n_select: int,
    include_squares: bool,
    step_size: float,
    trace_decay: float,
    score_decay: float,
) -> dict[str, Any]:
    """Score interactions by a Veeriah-style meta-gradient proxy.

    This is not the full Veeriah et al. auxiliary-question meta-gradient
    algorithm. It is a causal candidate-track proxy: before each source-Horde
    update, score each candidate by correlation between the current TD error and
    the candidate's Bellman feature difference, ``phi_t - gamma * phi_{t+1}``.
    That is the one-weight loss-gradient direction a newly added feature would
    expose to the downstream TD learner.
    """
    candidates, pairs = interaction_candidate_values(
        observations,
        include_squares=include_squares,
    )
    next_candidates, _ = interaction_candidate_values(
        next_observations,
        include_squares=include_squares,
    )
    if candidates.shape[1] == 0 or n_select <= 0:
        return {
            "indices": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((candidates.shape[1],), dtype=np.float32),
            "pairs": pairs,
            "scales": np.zeros((0,), dtype=np.float32),
            "selected_score_mean": 0.0,
        }
    if candidates.shape != next_candidates.shape:
        raise ValueError("interaction candidate views must match")
    if not 0.0 <= score_decay < 1.0:
        raise ValueError("score_decay must satisfy 0 <= decay < 1")

    horde, state = make_horde(
        n_targets=n_targets,
        gammas=gammas,
        n_aux=0,
        feature_dim=observations.shape[1],
        hidden_sizes=(),
        step_size=step_size,
        sparsity=0.0,
        trace_decay=trace_decay,
        key=key,
    )
    gamma_heads = np.tile(np.asarray(gammas, dtype=np.float32), n_targets)
    target_inputs = repeat_by_horizon(target_cumulants, gammas)
    cross = np.zeros(candidates.shape[1], dtype=np.float32)
    diff_sq = np.zeros(candidates.shape[1], dtype=np.float32)
    delta_sq = 0.0
    for t in range(observations.shape[0]):
        pred_t = np.asarray(horde.predict(state, jnp.asarray(observations[t])))
        pred_tp1 = np.asarray(horde.predict(state, jnp.asarray(next_observations[t])))
        td_errors = target_inputs[t] + gamma_heads * pred_tp1 - pred_t
        delta = float(np.mean(td_errors))
        bellman_diff = candidates[t] - float(np.mean(gamma_heads)) * next_candidates[t]
        cross = (score_decay * cross + delta * bellman_diff).astype(np.float32)
        diff_sq = (score_decay * diff_sq + bellman_diff**2).astype(np.float32)
        delta_sq = score_decay * delta_sq + delta**2
        result = horde.update(
            state,
            jnp.asarray(observations[t]),
            jnp.asarray(target_inputs[t]),
            jnp.asarray(next_observations[t]),
        )
        state = result.state

    scores = np.asarray(
        np.abs(cross) / np.sqrt(np.maximum(diff_sq * delta_sq, 1e-8)),
        dtype=np.float32,
    )
    selected = select_novel_candidate_indices(
        candidates,
        scores,
        n_select=n_select,
        max_abs_corr=0.98,
    )
    selected_scores = scores[selected] if selected.size else np.array([])
    return {
        "indices": selected,
        "scores": scores,
        "pairs": pairs,
        "scales": _candidate_scales(candidates, selected),
        "selected_score_mean": float(np.mean(selected_scores)) if selected.size else 0.0,
    }


def augment_with_history_trace_features(
    observations: np.ndarray,
    *,
    lags: tuple[int, ...],
    trace_rhos: tuple[float, ...],
) -> np.ndarray:
    """Append causal lag and exponentially decayed trace features.

    Row ``t`` only depends on observations ``0..t``. For a transition split,
    use row ``t`` for ``s_t`` and row ``t + 1`` for ``s_{t+1}``.
    """
    if observations.ndim != 2:
        raise ValueError("observations must be a 2-D array")
    for lag in lags:
        if lag < 1:
            raise ValueError("history lags must be positive")
    for rho in trace_rhos:
        if not 0.0 <= rho < 1.0:
            raise ValueError("trace rhos must satisfy 0 <= rho < 1")

    pieces = [observations.astype(np.float32)]
    zeros = np.zeros_like(observations[:1], dtype=np.float32)
    for lag in lags:
        if lag >= observations.shape[0]:
            lagged = np.repeat(zeros, observations.shape[0], axis=0)
        else:
            lagged = np.concatenate(
                [np.repeat(zeros, lag, axis=0), observations[:-lag]], axis=0
            )
        pieces.append(lagged.astype(np.float32))

    for rho in trace_rhos:
        trace = np.zeros(observations.shape[1], dtype=np.float32)
        trace_rows = np.zeros_like(observations, dtype=np.float32)
        for t, obs in enumerate(observations):
            trace = (rho * trace + (1.0 - rho) * obs).astype(np.float32)
            trace_rows[t] = trace
        pieces.append(trace_rows)

    return np.concatenate(pieces, axis=1).astype(np.float32)


def warm_cumulant_discovery(
    obs: np.ndarray,
    next_obs: np.ndarray,
    args: argparse.Namespace,
    key: jnp.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, int]:
    discovery = CumulantDiscovery(
        raw_dim=obs.shape[1],
        n_candidates=args.n_aux_cumulants,
        decay_rate=0.99,
        replacement_rate=args.cumulant_replacement_rate,
        maturity_threshold=args.cumulant_maturity,
        predictor_step_size=args.cumulant_predictor_step_size,
        gamma=0.0,
        enabled=enabled,
    )
    state = discovery.init(key)
    initial = np.asarray(state.projections).copy()
    for t in range(obs.shape[0]):
        state = discovery.step(state, jnp.asarray(obs[t]), jnp.asarray(next_obs[t]))
        state = discovery.maybe_replace(state)
    projections = np.asarray(state.projections, dtype=np.float32)
    changed = int(
        np.sum(~np.all(np.isclose(projections, initial, atol=1e-6), axis=1))
    )
    return projections, changed


def projected_cumulants(projections: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
    cumulants = next_observations @ projections.T
    return np.asarray(cumulants, dtype=np.float32)


def _normalize_projection_rows(projections: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-8)
    normalized = np.asarray(projections / safe_norms, dtype=np.float32)
    return normalized


def meta_proxy_projections(
    obs: np.ndarray,
    next_obs: np.ndarray,
    cumulants: np.ndarray,
    n_aux: int,
) -> np.ndarray:
    """Pick auxiliary cumulants by a simple downstream-loss proxy.

    This is not Veeriah et al.'s full meta-gradient method. It is a small
    causal proxy: fit a linear one-step target predictor on the discovery
    prefix, compute residuals, and choose next-observation projection
    directions most correlated with those residuals. The resulting auxiliary
    cumulants are still transition signals ``w . obs_{t+1}``.
    """
    if n_aux < 1:
        raise ValueError("n_aux must be positive")
    design = np.column_stack([np.ones(obs.shape[0], dtype=np.float32), obs])
    coef, *_ = np.linalg.lstsq(design, cumulants, rcond=None)
    residuals = cumulants - design @ coef
    centered_next = next_obs - np.mean(next_obs, axis=0, keepdims=True)
    scores = residuals.T @ centered_next
    order = np.argsort(-np.linalg.norm(scores, axis=1))
    rows: list[np.ndarray] = []
    for target_idx in order:
        row = scores[target_idx]
        if np.linalg.norm(row) > 1e-8:
            rows.append(row.astype(np.float32))
        if len(rows) == n_aux:
            break
    fallback = np.eye(next_obs.shape[1], dtype=np.float32)
    fallback_idx = 0
    while len(rows) < n_aux:
        rows.append(fallback[fallback_idx % fallback.shape[0]])
        fallback_idx += 1
    return _normalize_projection_rows(np.stack(rows[:n_aux]))


def run_seed(seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    total_steps = args.discovery_steps + args.eval_steps + 1
    if args.observation_dynamics == "ar1":
        observations, targets = collect_markov_interaction_arrays(
            seed=seed,
            total_steps=total_steps,
            feature_dim=args.feature_dim,
            n_targets=args.n_targets,
            n_contexts=args.n_contexts,
            context_length=args.context_length,
            active_pairs=args.active_pairs,
            noise_std=args.noise_std,
            include_squares=args.include_squares,
            hide_last_channels=args.hide_last_channels,
            ar_rho=args.ar_rho,
        )
    elif args.observation_dynamics == "coupled_hidden_ar1":
        observations, targets = collect_coupled_hidden_ar1_arrays(
            seed=seed,
            total_steps=total_steps,
            feature_dim=args.feature_dim,
            n_targets=args.n_targets,
            n_contexts=args.n_contexts,
            context_length=args.context_length,
            active_pairs=args.active_pairs,
            noise_std=args.noise_std,
            include_squares=args.include_squares,
            hide_last_channels=args.hide_last_channels,
            ar_rho=args.ar_rho,
            hidden_coupling=args.hidden_coupling,
            hidden_noise_std=args.hidden_noise_std,
        )
    else:
        observations, targets = collect_partial_interaction_arrays(
            seed=seed,
            total_steps=total_steps,
            feature_dim=args.feature_dim,
            n_targets=args.n_targets,
            n_contexts=args.n_contexts,
            context_length=args.context_length,
            active_pairs=args.active_pairs,
            noise_std=args.noise_std,
            include_squares=args.include_squares,
            hide_last_channels=args.hide_last_channels,
        )
    disc_obs, disc_next, disc_cums, eval_obs, eval_next, eval_cums = transition_view(
        observations, targets, args.discovery_steps
    )
    history_all = augment_with_history_trace_features(
        observations,
        lags=args.history_lags,
        trace_rhos=args.history_trace_rhos,
    )
    history_eval = history_all[args.discovery_steps : -1]
    history_next = history_all[args.discovery_steps + 1 :]

    root = jr.key(seed + 10_000)
    (
        k_linear,
        k_mlp,
        k_history_horde,
        k_tanh,
        k_tanh_horde,
        k_interaction,
        k_interaction_horde,
        k_random_aux,
        k_random_aux_horde,
        k_discovered_aux,
        k_discovered_aux_horde,
        k_all_interactions,
        k_td_surprise,
        k_td_surprise_horde,
        k_meta_gradient,
        k_meta_gradient_horde,
        k_meta_proxy_aux_horde,
        k_gvf_feedback,
        k_offpolicy_td_surprise,
        k_offpolicy_raw_is,
        k_offpolicy_raw_no_is,
        k_offpolicy_history_is,
        k_offpolicy_td_surprise_is,
        k_predictive_state_horde,
        k_offpolicy_mspbe,
        k_offpolicy_mspbe_is,
    ) = jr.split(root, 26)

    returns = target_forward_returns(eval_cums, args.gammas)
    rows: list[dict[str, Any]] = []

    method_specs: list[MethodSpec] = []
    method_specs.append(
        (
            "given_linear_gvf",
            eval_obs,
            eval_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_linear,
            {"policy_regime": "on_policy_prediction"},
        )
    )
    method_specs.append(
        (
            "given_mlp_gvf",
            eval_obs,
            eval_next,
            None,
            (args.mlp_hidden,),
            args.mlp_step_size,
            args.mlp_sparsity,
            args.trace_lambda,
            k_mlp,
            {"policy_regime": "on_policy_prediction"},
        )
    )

    oracle_eval = augment_with_all_interactions(
        eval_obs,
        include_squares=args.include_squares,
    )
    oracle_next = augment_with_all_interactions(
        eval_next,
        include_squares=args.include_squares,
    )
    method_specs.append(
        (
            "fixed_interaction_linear_gvf",
            oracle_eval,
            oracle_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_all_interactions,
            {
                "feature_dim_after": int(oracle_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
                "positive_control": True,
            },
        )
    )
    method_specs.append(
        (
            "fixed_history_trace_linear_gvf",
            history_eval,
            history_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_history_horde,
            {
                "feature_dim_after": int(history_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
                "positive_control": True,
                "history_lags": " ".join(str(lag) for lag in args.history_lags),
                "history_trace_rhos": " ".join(
                    f"{rho:g}" for rho in args.history_trace_rhos
                ),
            },
        )
    )

    td_surprise_scores = score_td_surprise_interaction_candidates(
        observations=disc_obs,
        next_observations=disc_next,
        target_cumulants=disc_cums,
        gammas=args.gammas,
        n_targets=args.n_targets,
        key=k_td_surprise,
        n_select=args.n_features,
        include_squares=args.include_squares,
        step_size=args.linear_step_size,
        trace_decay=args.trace_lambda,
        candidate_trace_rho=args.td_surprise_trace_rho,
        score_decay=args.td_surprise_score_decay,
    )
    td_surprise_eval = augment_with_selected_interactions(
        eval_obs,
        selected_indices=td_surprise_scores["indices"],
        include_squares=args.include_squares,
    )
    td_surprise_next = augment_with_selected_interactions(
        eval_next,
        selected_indices=td_surprise_scores["indices"],
        include_squares=args.include_squares,
    )
    method_specs.append(
        (
            "td_surprise_interaction_features_linear_gvf",
            td_surprise_eval,
            td_surprise_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_td_surprise_horde,
            {
                "feature_dim_after": int(td_surprise_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
                "aux_selection": "td_error_surprise_interaction_score",
                "selected_score_mean": td_surprise_scores["selected_score_mean"],
            },
        )
    )

    meta_gradient_scores = score_meta_gradient_interaction_candidates(
        observations=disc_obs,
        next_observations=disc_next,
        target_cumulants=disc_cums,
        gammas=args.gammas,
        n_targets=args.n_targets,
        key=k_meta_gradient,
        n_select=args.n_features,
        include_squares=args.include_squares,
        step_size=args.linear_step_size,
        trace_decay=args.trace_lambda,
        score_decay=args.meta_gradient_score_decay,
    )
    meta_gradient_eval = augment_with_selected_interactions(
        eval_obs,
        selected_indices=meta_gradient_scores["indices"],
        include_squares=args.include_squares,
    )
    meta_gradient_next = augment_with_selected_interactions(
        eval_next,
        selected_indices=meta_gradient_scores["indices"],
        include_squares=args.include_squares,
    )
    method_specs.append(
        (
            "meta_gradient_proxy_interaction_features_linear_gvf",
            meta_gradient_eval,
            meta_gradient_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_meta_gradient_horde,
            {
                "feature_dim_after": int(meta_gradient_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
                "aux_selection": "veeriah_style_td_loss_gradient_proxy_interaction_score",
                "selected_score_mean": meta_gradient_scores["selected_score_mean"],
                "proxy_for": "veeriah_auxiliary_question_meta_gradient",
            },
        )
    )

    predictive_scores = score_predictive_state_feature_candidates(
        observations=disc_obs,
        target_cumulants=disc_cums,
        n_select=args.n_features,
        lags=args.history_lags,
        trace_rhos=args.history_trace_rhos,
        include_cross_products=args.predictive_state_cross_products,
        score_decay=args.predictive_state_score_decay,
    )
    predictive_all = augment_with_selected_predictive_state_features(
        observations,
        selected_indices=predictive_scores["indices"],
        scales=predictive_scores["scales"],
        lags=args.history_lags,
        trace_rhos=args.history_trace_rhos,
        include_cross_products=args.predictive_state_cross_products,
    )
    predictive_eval = predictive_all[args.discovery_steps : -1]
    predictive_next = predictive_all[args.discovery_steps + 1 :]
    method_specs.append(
        (
            "predictive_state_features_linear_gvf",
            predictive_eval,
            predictive_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_predictive_state_horde,
            {
                "feature_dim_after": int(predictive_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
                "aux_selection": "future_cumulant_predictive_state_score",
                "selected_score_mean": predictive_scores["selected_score_mean"],
                "history_lags": " ".join(str(lag) for lag in args.history_lags),
                "history_trace_rhos": " ".join(
                    f"{rho:g}" for rho in args.history_trace_rhos
                ),
            },
        )
    )

    tanh_learner, tanh_state = warm_step2_tanh_features(disc_obs, disc_cums, args, k_tanh)
    tanh_eval = augment_with_frozen_features(tanh_learner, tanh_state, eval_obs)
    tanh_next = augment_with_frozen_features(tanh_learner, tanh_state, eval_next)
    method_specs.append(
        (
            "step2_tanh_features_linear_gvf",
            tanh_eval,
            tanh_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_tanh_horde,
            {
                "feature_dim_after": int(tanh_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
            },
        )
    )

    interaction_learner, interaction_state = warm_step2_interaction_features(
        disc_obs, disc_cums, args, k_interaction
    )
    interaction_eval = augment_with_frozen_features(
        interaction_learner, interaction_state, eval_obs
    )
    interaction_next = augment_with_frozen_features(
        interaction_learner, interaction_state, eval_next
    )
    method_specs.append(
        (
            "step2_interaction_features_linear_gvf",
            interaction_eval,
            interaction_next,
            None,
            (),
            args.linear_step_size,
            0.0,
            args.trace_lambda,
            k_interaction_horde,
            {
                "feature_dim_after": int(interaction_eval.shape[1]),
                "policy_regime": "on_policy_prediction",
            },
        )
    )

    random_proj, random_changed = warm_cumulant_discovery(
        disc_obs, disc_next, args, k_random_aux, enabled=False
    )
    discovered_proj, discovered_changed = warm_cumulant_discovery(
        disc_obs, disc_next, args, k_discovered_aux, enabled=True
    )
    method_specs.append(
        (
            "random_aux_cumulants_mlp_gvf",
            eval_obs,
            eval_next,
            projected_cumulants(random_proj, eval_next),
            (args.mlp_hidden,),
            args.mlp_step_size,
            args.mlp_sparsity,
            args.trace_lambda,
            k_random_aux_horde,
            {
                "changed_projections": random_changed,
                "policy_regime": "on_policy_prediction",
            },
        )
    )
    method_specs.append(
        (
            "discovered_aux_cumulants_mlp_gvf",
            eval_obs,
            eval_next,
            projected_cumulants(discovered_proj, eval_next),
            (args.mlp_hidden,),
            args.mlp_step_size,
            args.mlp_sparsity,
            args.trace_lambda,
            k_discovered_aux_horde,
            {
                "changed_projections": discovered_changed,
                "policy_regime": "on_policy_prediction",
                "aux_selection": "surprise_driven",
            },
        )
    )

    meta_proxy_proj = meta_proxy_projections(
        disc_obs,
        disc_next,
        disc_cums,
        args.n_aux_cumulants,
    )
    method_specs.append(
        (
            "meta_proxy_aux_cumulants_mlp_gvf",
            eval_obs,
            eval_next,
            projected_cumulants(meta_proxy_proj, eval_next),
            (args.mlp_hidden,),
            args.mlp_step_size,
            args.mlp_sparsity,
            args.trace_lambda,
            k_meta_proxy_aux_horde,
            {
                "changed_projections": 0,
                "policy_regime": "on_policy_prediction",
                "aux_selection": "linear_residual_proxy",
            },
        )
    )

    for trace_lamda in args.trace_sweep_lambdas:
        if abs(trace_lamda - args.trace_lambda) < 1e-12:
            continue
        method_specs.append(
            (
                f"given_linear_gvf_trace_{trace_lamda:g}",
                eval_obs,
                eval_next,
                None,
                (),
                args.linear_step_size,
                0.0,
                trace_lamda,
                jr.fold_in(k_linear, int(10_000 * trace_lamda)),
                {
                    "policy_regime": "on_policy_prediction",
                    "trace_sweep": True,
                },
            )
        )

    for (
        name,
        obs,
        next_obs,
        aux_cums,
        hidden,
        step_size,
        sparsity,
        trace_decay,
        key,
        extra,
    ) in method_specs:
        preds = run_horde_predictions(
            observations=obs,
            next_observations=next_obs,
            target_cumulants=eval_cums,
            gammas=args.gammas,
            n_targets=args.n_targets,
            key=key,
            hidden_sizes=hidden,
            step_size=step_size,
            sparsity=sparsity,
            trace_decay=trace_decay,
            aux_cumulants=aux_cums,
        )
        row = {
            "seed": seed,
            "method": name,
            **summarize_target_rmse(
                preds,
                returns,
                args.n_targets,
                args.gammas,
                args.burn_in,
                args.burn_tail,
            ),
            **extra,
        }
        rows.append(row)
        print(
            f"seed={seed:>2} {name:<38} "
            f"target_rmse={row['target_rmse_mean']:.5f}"
        )

    feedback_preds = run_gvf_feedback_predictions(
        observations=eval_obs,
        next_observations=eval_next,
        target_cumulants=eval_cums,
        gammas=args.gammas,
        n_targets=args.n_targets,
        key=k_gvf_feedback,
        source_step_size=args.linear_step_size,
        downstream_step_size=args.linear_step_size,
        trace_decay=args.trace_lambda,
    )
    feedback_row = {
        "seed": seed,
        "method": "gvf_feedback_features_linear_gvf",
        **summarize_target_rmse(
            feedback_preds,
            returns,
            args.n_targets,
            args.gammas,
            args.burn_in,
            args.burn_tail,
        ),
        "feature_dim_after": int(eval_obs.shape[1] + args.n_targets * len(args.gammas)),
        "policy_regime": "on_policy_prediction",
    }
    rows.append(feedback_row)
    print(
        f"seed={seed:>2} {'gvf_feedback_features_linear_gvf':<38} "
        f"target_rmse={feedback_row['target_rmse_mean']:.5f}"
    )

    discovery_ratios = behavior_mismatch_importance_ratios(
        disc_obs,
        key=k_offpolicy_td_surprise,
        policy_scale=args.off_policy_scale,
    )
    off_td_surprise_scores = score_td_surprise_interaction_candidates(
        observations=disc_obs,
        next_observations=disc_next,
        target_cumulants=disc_cums[:, :1],
        gammas=(args.off_policy_gamma,),
        n_targets=1,
        key=jr.fold_in(k_offpolicy_td_surprise, 1),
        n_select=args.n_features,
        include_squares=args.include_squares,
        step_size=args.off_policy_step_size,
        trace_decay=args.off_policy_trace_lambda,
        candidate_trace_rho=args.td_surprise_trace_rho,
        score_decay=args.td_surprise_score_decay,
        importance_ratios=discovery_ratios,
        importance_clip=args.off_policy_retrace_clip,
    )
    off_td_surprise_eval = augment_with_selected_interactions(
        eval_obs,
        selected_indices=off_td_surprise_scores["indices"],
        include_squares=args.include_squares,
    )
    off_td_surprise_next = augment_with_selected_interactions(
        eval_next,
        selected_indices=off_td_surprise_scores["indices"],
        include_squares=args.include_squares,
    )
    predictive_candidates_all, _ = predictive_state_candidate_values(
        observations,
        lags=args.history_lags,
        trace_rhos=args.history_trace_rhos,
        include_cross_products=args.predictive_state_cross_products,
    )
    predictive_candidates_disc = predictive_candidates_all[: args.discovery_steps]
    predictive_candidates_disc_next = predictive_candidates_all[
        1 : args.discovery_steps + 1
    ]
    predictive_candidates_eval = predictive_candidates_all[
        args.discovery_steps : -1
    ]
    predictive_candidates_eval_next = predictive_candidates_all[
        args.discovery_steps + 1 :
    ]
    off_mspbe_scores = score_off_policy_mspbe_feature_candidates(
        observations=disc_obs,
        next_observations=disc_next,
        candidate_values=predictive_candidates_disc,
        next_candidate_values=predictive_candidates_disc_next,
        reward_signal=disc_cums[:, 0],
        key=k_offpolicy_mspbe,
        gamma=args.off_policy_gamma,
        step_size=args.off_policy_step_size,
        trace_decay=args.off_policy_trace_lambda,
        policy_scale=args.off_policy_scale,
        retrace_clip=args.off_policy_retrace_clip,
        n_select=args.n_features,
        score_decay=args.predictive_state_score_decay,
    )
    off_mspbe_eval = _append_selected_scaled_candidates(
        eval_obs,
        predictive_candidates_eval,
        selected_indices=off_mspbe_scores["indices"],
        scales=off_mspbe_scores["scales"],
    )
    off_mspbe_next = _append_selected_scaled_candidates(
        eval_next,
        predictive_candidates_eval_next,
        selected_indices=off_mspbe_scores["indices"],
        scales=off_mspbe_scores["scales"],
    )
    off_novel_indices = select_novel_candidate_indices(
        predictive_candidates_disc,
        off_mspbe_scores["scores"],
        n_select=min(args.off_policy_novelty_features, args.n_features),
        max_abs_corr=args.off_policy_novelty_threshold,
    )
    off_novel_scales = _candidate_scales(predictive_candidates_disc, off_novel_indices)
    off_novel_eval = _append_selected_scaled_candidates(
        eval_obs,
        predictive_candidates_eval,
        selected_indices=off_novel_indices,
        scales=off_novel_scales,
    )
    off_novel_next = _append_selected_scaled_candidates(
        eval_next,
        predictive_candidates_eval_next,
        selected_indices=off_novel_indices,
        scales=off_novel_scales,
    )
    off_policy_specs = [
        (
            "off_policy_raw_linear_td_is",
            eval_obs,
            eval_next,
            k_offpolicy_raw_is,
            True,
            args.off_policy_step_size,
        ),
        (
            "off_policy_raw_linear_td_no_is",
            eval_obs,
            eval_next,
            k_offpolicy_raw_no_is,
            False,
            args.off_policy_step_size,
        ),
        (
            "off_policy_history_trace_linear_td_is",
            history_eval,
            history_next,
            k_offpolicy_history_is,
            True,
            args.off_policy_step_size,
        ),
        (
            "off_policy_td_surprise_interaction_linear_td_is",
            off_td_surprise_eval,
            off_td_surprise_next,
            k_offpolicy_td_surprise_is,
            True,
            args.off_policy_step_size,
        ),
        (
            "off_policy_mspbe_predictive_state_linear_td_is",
            off_mspbe_eval,
            off_mspbe_next,
            k_offpolicy_mspbe_is,
            True,
            args.off_policy_feature_step_size,
        ),
        (
            "off_policy_mspbe_novel_predictive_state_linear_td_is",
            off_novel_eval,
            off_novel_next,
            jr.fold_in(k_offpolicy_mspbe_is, 1),
            True,
            args.off_policy_novelty_step_size,
        ),
    ]
    eval_off_rewards, eval_off_ratios, eval_off_returns, eval_ratio_stats = (
        sample_off_policy_behavior_rollout(
            observations=eval_obs,
            reward_signal=eval_cums[:, 0],
            key=k_offpolicy_raw_is,
            gamma=args.off_policy_gamma,
            policy_scale=args.off_policy_scale,
        )
    )
    for name, obs, next_obs, key, use_is, off_step_size in off_policy_specs:
        off_preds, off_returns, ratio_stats = run_off_policy_td_probe(
            observations=obs,
            next_observations=next_obs,
            reward_signal=eval_cums[:, 0],
            key=key,
            gamma=args.off_policy_gamma,
            step_size=off_step_size,
            trace_decay=args.off_policy_trace_lambda,
            policy_scale=args.off_policy_scale,
            retrace_clip=args.off_policy_retrace_clip,
            use_importance_sampling=use_is,
            sampled_rewards=eval_off_rewards,
            sampled_ratios=eval_off_ratios,
            target_returns=eval_off_returns,
            ratio_stats=eval_ratio_stats,
        )
        off_row = {
            "seed": seed,
            "method": name,
            "probe": "off_policy_td",
            **summarize_scalar_rmse(
                off_preds,
                off_returns,
                args.burn_in,
                args.burn_tail,
            ),
            "feature_dim_after": int(obs.shape[1]),
            "policy_regime": "behavior_mismatch",
            "uses_importance_sampling": use_is,
            "off_policy_gamma": args.off_policy_gamma,
            "off_policy_retrace_clip": args.off_policy_retrace_clip,
            "off_policy_step_size_used": off_step_size,
            **ratio_stats,
        }
        if name == "off_policy_td_surprise_interaction_linear_td_is":
            off_row.update(
                {
                    "aux_selection": "td_error_surprise_clipped_is_interaction_score",
                    "selected_score_mean": off_td_surprise_scores[
                        "selected_score_mean"
                    ],
                    "score_importance_weight_mean": off_td_surprise_scores[
                        "importance_weight_mean"
                    ],
                    "score_importance_weight_max": off_td_surprise_scores[
                        "importance_weight_max"
                    ],
                }
            )
        if name == "off_policy_mspbe_predictive_state_linear_td_is":
            off_row.update(
                {
                    "aux_selection": "clipped_is_mspbe_predictive_state_score",
                    "selected_score_mean": off_mspbe_scores["selected_score_mean"],
                    "score_importance_weight_mean": off_mspbe_scores[
                        "importance_weight_mean"
                    ],
                    "score_importance_weight_max": off_mspbe_scores[
                        "importance_weight_max"
                    ],
                }
            )
        if name == "off_policy_mspbe_novel_predictive_state_linear_td_is":
            novel_scores = off_mspbe_scores["scores"][off_novel_indices]
            off_row.update(
                {
                    "aux_selection": "clipped_is_mspbe_predictive_state_score_novelty_gate",
                    "selected_score_mean": (
                        float(np.mean(novel_scores)) if off_novel_indices.size else 0.0
                    ),
                    "score_importance_weight_mean": off_mspbe_scores[
                        "importance_weight_mean"
                    ],
                    "score_importance_weight_max": off_mspbe_scores[
                        "importance_weight_max"
                    ],
                    "novelty_feature_count": int(off_novel_indices.size),
                    "novelty_max_abs_corr": args.off_policy_novelty_threshold,
                }
            )
        rows.append(off_row)
        print(
            f"seed={seed:>2} {name:<38} "
            f"target_rmse={off_row['target_rmse_mean']:.5f}"
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    main_rows = [r for r in rows if r.get("probe") != "off_policy_td"]
    off_policy_rows = [r for r in rows if r.get("probe") == "off_policy_td"]
    methods = sorted({str(r["method"]) for r in main_rows})
    aggregate: dict[str, Any] = {}
    for method in methods:
        sub = [r for r in main_rows if r["method"] == method]
        vals = np.asarray([r["target_rmse_mean"] for r in sub], dtype=np.float64)
        aggregate[method] = {
            "target_rmse_mean": float(np.mean(vals)),
            "target_rmse_std": float(np.std(vals, ddof=0)),
            "target_rmse_stderr": float(np.std(vals, ddof=0) / np.sqrt(len(vals))),
            "n_seeds": len(vals),
        }
        gamma_keys = sorted(k for k in sub[0] if k.startswith("rmse_gamma_"))
        for key in gamma_keys:
            gvals = np.asarray([r[key] for r in sub], dtype=np.float64)
            aggregate[method][f"{key}_mean"] = float(np.mean(gvals))

    off_policy_aggregate: dict[str, Any] = {}
    for method in sorted({str(r["method"]) for r in off_policy_rows}):
        sub = [r for r in off_policy_rows if r["method"] == method]
        vals = np.asarray([r["target_rmse_mean"] for r in sub], dtype=np.float64)
        off_policy_aggregate[method] = {
            "target_rmse_mean": float(np.mean(vals)),
            "target_rmse_std": float(np.std(vals, ddof=0)),
            "target_rmse_stderr": float(np.std(vals, ddof=0) / np.sqrt(len(vals))),
            "n_seeds": len(vals),
            "rho_mean": float(np.mean([r["rho_mean"] for r in sub])),
            "rho_max": float(np.max([r["rho_max"] for r in sub])),
        }

    by_seed = {(int(r["seed"]), str(r["method"])): r for r in main_rows}
    paired: dict[str, Any] = {}
    for baseline in ("given_linear_gvf", "given_mlp_gvf"):
        if baseline not in methods:
            continue
        for method in methods:
            if method == baseline:
                continue
            diffs = []
            wins = 0
            for seed in sorted({int(r["seed"]) for r in main_rows}):
                diff = (
                    by_seed[(seed, baseline)]["target_rmse_mean"]
                    - by_seed[(seed, method)]["target_rmse_mean"]
                )
                diffs.append(float(diff))
                wins += int(diff > 0.0)
            arr = np.asarray(diffs, dtype=np.float64)
            paired[f"{baseline}_minus_{method}"] = {
                "rmse_diff_mean": float(np.mean(arr)),
                "rmse_diff_stderr": float(np.std(arr, ddof=0) / np.sqrt(len(arr))),
                "wins": wins,
                "losses": int(np.sum(arr < 0.0)),
                "ties": int(np.sum(arr == 0.0)),
                "n_seeds": len(arr),
            }

    off_policy_methods = sorted({str(r["method"]) for r in off_policy_rows})
    off_policy_by_seed = {
        (int(r["seed"]), str(r["method"])): r for r in off_policy_rows
    }
    off_policy_paired: dict[str, Any] = {}
    off_policy_baseline = "off_policy_raw_linear_td_is"
    if off_policy_baseline in off_policy_methods:
        off_policy_seeds = sorted({int(r["seed"]) for r in off_policy_rows})
        for method in off_policy_methods:
            if method == off_policy_baseline:
                continue
            diffs = []
            for seed in off_policy_seeds:
                diff = (
                    off_policy_by_seed[(seed, off_policy_baseline)][
                        "target_rmse_mean"
                    ]
                    - off_policy_by_seed[(seed, method)]["target_rmse_mean"]
                )
                diffs.append(float(diff))
            arr = np.asarray(diffs, dtype=np.float64)
            off_policy_paired[f"{off_policy_baseline}_minus_{method}"] = {
                "rmse_diff_mean": float(np.mean(arr)),
                "rmse_diff_stderr": float(np.std(arr, ddof=0) / np.sqrt(len(arr))),
                "wins": int(np.sum(arr > 0.0)),
                "losses": int(np.sum(arr < 0.0)),
                "ties": int(np.sum(arr == 0.0)),
                "n_seeds": len(arr),
            }

    best_method = min(methods, key=lambda m: aggregate[m]["target_rmse_mean"])
    discovery_methods = [
        m
        for m in methods
        if m.startswith("step2_")
        or m.startswith("discovered_aux_")
        or m.startswith("meta_proxy_")
        or m.startswith("meta_gradient_proxy_")
        or m.startswith("gvf_feedback_")
        or m.startswith("td_surprise_")
        or m.startswith("predictive_state_")
    ]
    best_discovery = min(
        discovery_methods,
        key=lambda m: aggregate[m]["target_rmse_mean"],
    )
    beats_linear = (
        aggregate[best_discovery]["target_rmse_mean"]
        < aggregate["given_linear_gvf"]["target_rmse_mean"]
    )
    beats_mlp = (
        aggregate[best_discovery]["target_rmse_mean"]
        < aggregate["given_mlp_gvf"]["target_rmse_mean"]
    )
    return {
        "aggregate": aggregate,
        "paired": paired,
        "best_method": best_method,
        "best_discovery_method": best_discovery,
        "best_discovery_beats_linear": bool(beats_linear),
        "best_discovery_beats_mlp": bool(beats_mlp),
        "off_policy_aggregate": off_policy_aggregate,
        "off_policy_paired": off_policy_paired,
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
    total_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": config,
        "rows": rows,
        **summary,
        "total_seconds": total_seconds,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Direction 8 GVF Feature Discovery Evaluation",
        "",
        f"Total seconds: {total_seconds:.2f}",
        "",
        "## Aggregate Target GVF RMSE",
        "",
    ]
    for method, data in sorted(
        summary["aggregate"].items(),
        key=lambda item: item[1]["target_rmse_mean"],
    ):
        lines.append(
            f"- {method}: {data['target_rmse_mean']:.6f} "
            f"+/- {data['target_rmse_stderr']:.6f}"
        )
    if summary.get("off_policy_aggregate"):
        lines.extend(["", "## Off-Policy Behavior-Mismatch Probe", ""])
        for method, data in sorted(
            summary["off_policy_aggregate"].items(),
            key=lambda item: item[1]["target_rmse_mean"],
        ):
            lines.append(
                f"- {method}: {data['target_rmse_mean']:.6f} "
                f"+/- {data['target_rmse_stderr']:.6f} "
                f"(rho_mean={data['rho_mean']:.3f}, rho_max={data['rho_max']:.3f})"
            )
    if summary.get("off_policy_paired"):
        lines.extend(["", "## Off-Policy Paired Diffs", ""])
        for name, data in sorted(summary["off_policy_paired"].items()):
            lines.append(
                f"- {name}: diff={data['rmse_diff_mean']:.6f} "
                f"+/- {data['rmse_diff_stderr']:.6f}, "
                f"wins/losses/ties={data['wins']}/{data['losses']}/{data['ties']}"
            )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Best overall: {summary['best_method']}",
            f"- Best discovery method: {summary['best_discovery_method']}",
            f"- Best discovery beats linear baseline: {summary['best_discovery_beats_linear']}",
            f"- Best discovery beats MLP baseline: {summary['best_discovery_beats_mlp']}",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Fast smoke settings")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--discovery-steps", type=int, default=800)
    parser.add_argument("--eval-steps", type=int, default=1600)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--burn-tail", type=int, default=50)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--hide-last-channels", type=int, default=1)
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--n-contexts", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=200)
    parser.add_argument("--active-pairs", type=int, default=5)
    parser.add_argument("--include-squares", action="store_true")
    parser.add_argument(
        "--observation-dynamics",
        choices=("iid", "ar1", "coupled_hidden_ar1"),
        default="iid",
        help="Use iid, observable AR(1), or masked coupled-hidden AR(1) observations.",
    )
    parser.add_argument(
        "--ar-rho",
        type=float,
        default=0.95,
        help="Lag-one correlation for --observation-dynamics ar1.",
    )
    parser.add_argument(
        "--hidden-coupling",
        type=float,
        default=0.25,
        help="Visible-to-hidden coupling for coupled_hidden_ar1.",
    )
    parser.add_argument(
        "--hidden-noise-std",
        type=float,
        default=0.05,
        help="Hidden innovation std for coupled_hidden_ar1.",
    )
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--linear-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--mlp-step-size", type=float, default=0.02)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument(
        "--trace-lambda",
        type=float,
        default=0.0,
        help="Eligibility-trace lambda for all GVF heads in the primary run.",
    )
    parser.add_argument(
        "--trace-sweep-lambdas",
        type=float,
        nargs="*",
        default=[],
        help="Optional extra lambda values for raw-linear trace sweep rows.",
    )
    parser.add_argument(
        "--history-lags",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8],
        help="Causal observation lags appended for fixed history positive control.",
    )
    parser.add_argument(
        "--history-trace-rhos",
        type=float,
        nargs="*",
        default=[0.5, 0.8, 0.95],
        help="EWMA trace rhos appended for fixed history positive control.",
    )
    parser.add_argument("--n-features", type=int, default=24)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--replacement-interval", type=int, default=50)
    parser.add_argument("--min-feature-age", type=int, default=50)
    parser.add_argument("--candidate-min-age", type=int, default=25)
    parser.add_argument("--feature-step-size-output", type=float, default=0.03)
    parser.add_argument("--feature-step-size-constructor", type=float, default=0.003)
    parser.add_argument("--n-aux-cumulants", type=int, default=8)
    parser.add_argument("--cumulant-replacement-rate", type=float, default=0.02)
    parser.add_argument("--cumulant-maturity", type=int, default=80)
    parser.add_argument("--cumulant-predictor-step-size", type=float, default=0.05)
    parser.add_argument(
        "--td-surprise-trace-rho",
        type=float,
        default=0.8,
        help="Eligibility trace rho for TD-surprise candidate interaction scoring.",
    )
    parser.add_argument(
        "--td-surprise-score-decay",
        type=float,
        default=0.99,
        help="EMA decay for TD-surprise candidate interaction scores.",
    )
    parser.add_argument(
        "--meta-gradient-score-decay",
        type=float,
        default=0.99,
        help="EMA decay for the Veeriah-style meta-gradient proxy scorer.",
    )
    parser.add_argument(
        "--predictive-state-score-decay",
        type=float,
        default=0.99,
        help="EMA decay for predictive-state feature utility scores.",
    )
    parser.add_argument(
        "--no-predictive-state-cross-products",
        action="store_false",
        dest="predictive_state_cross_products",
        help="Disable current-observation by predictive-state candidate products.",
    )
    parser.add_argument("--off-policy-gamma", type=float, default=0.5)
    parser.add_argument("--off-policy-step-size", type=float, default=0.02)
    parser.add_argument("--off-policy-feature-step-size", type=float, default=0.003)
    parser.add_argument("--off-policy-trace-lambda", type=float, default=0.0)
    parser.add_argument("--off-policy-scale", type=float, default=1.5)
    parser.add_argument("--off-policy-retrace-clip", type=float, default=2.0)
    parser.add_argument(
        "--off-policy-novelty-features",
        type=int,
        default=8,
        help="Feature budget for the off-policy MSPBE novelty-gated row.",
    )
    parser.add_argument(
        "--off-policy-novelty-threshold",
        type=float,
        default=0.95,
        help="Maximum discovery-prefix absolute correlation for novelty-gated features.",
    )
    parser.add_argument(
        "--off-policy-novelty-step-size",
        type=float,
        default=0.005,
        help="TD step-size for the off-policy novelty-gated feature row.",
    )
    args = parser.parse_args()
    args.gammas = tuple(float(g) for g in args.gammas)
    args.history_lags = tuple(int(lag) for lag in args.history_lags)
    args.history_trace_rhos = tuple(float(rho) for rho in args.history_trace_rhos)
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.discovery_steps = min(args.discovery_steps, 250)
        args.eval_steps = min(args.eval_steps, 500)
        args.burn_in = min(args.burn_in, 50)
        args.burn_tail = min(args.burn_tail, 25)
        args.n_features = min(args.n_features, 12)
        args.candidate_count = min(args.candidate_count, 12)
        args.n_aux_cumulants = min(args.n_aux_cumulants, 4)
        args.context_length = min(args.context_length, 80)
        args.replacement_interval = min(args.replacement_interval, 25)
        args.min_feature_age = min(args.min_feature_age, 25)
        args.candidate_min_age = min(args.candidate_min_age, 10)
        args.cumulant_maturity = min(args.cumulant_maturity, 40)
    if not 0.0 <= args.ar_rho < 1.0:
        raise ValueError("--ar-rho must satisfy 0 <= ar_rho < 1")
    if not 0.0 <= args.trace_lambda <= 1.0:
        raise ValueError("--trace-lambda must satisfy 0 <= lambda <= 1")
    for lamda in args.trace_sweep_lambdas:
        if not 0.0 <= lamda <= 1.0:
            raise ValueError("--trace-sweep-lambdas entries must be in [0, 1]")
    for lag in args.history_lags:
        if lag < 1:
            raise ValueError("--history-lags entries must be positive")
    for rho in args.history_trace_rhos:
        if not 0.0 <= rho < 1.0:
            raise ValueError("--history-trace-rhos entries must satisfy 0 <= rho < 1")
    if not 0.0 <= args.off_policy_gamma <= 1.0:
        raise ValueError("--off-policy-gamma must satisfy 0 <= gamma <= 1")
    if args.off_policy_feature_step_size <= 0.0:
        raise ValueError("--off-policy-feature-step-size must be positive")
    if args.off_policy_novelty_features < 0:
        raise ValueError("--off-policy-novelty-features must be non-negative")
    if not 0.0 <= args.off_policy_novelty_threshold <= 1.0:
        raise ValueError("--off-policy-novelty-threshold must lie in [0, 1]")
    if args.off_policy_novelty_step_size <= 0.0:
        raise ValueError("--off-policy-novelty-step-size must be positive")
    if not 0.0 <= args.off_policy_trace_lambda <= 1.0:
        raise ValueError("--off-policy-trace-lambda must satisfy 0 <= lambda <= 1")
    if args.off_policy_retrace_clip <= 0.0:
        raise ValueError("--off-policy-retrace-clip must be positive")
    if not 0.0 <= args.td_surprise_trace_rho < 1.0:
        raise ValueError("--td-surprise-trace-rho must satisfy 0 <= rho < 1")
    if not 0.0 <= args.td_surprise_score_decay < 1.0:
        raise ValueError("--td-surprise-score-decay must satisfy 0 <= decay < 1")
    if not 0.0 <= args.meta_gradient_score_decay < 1.0:
        raise ValueError("--meta-gradient-score-decay must satisfy 0 <= decay < 1")
    if not 0.0 <= args.predictive_state_score_decay < 1.0:
        raise ValueError("--predictive-state-score-decay must satisfy 0 <= decay < 1")
    return args


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        rows.extend(run_seed(seed, args))
    total_seconds = time.perf_counter() - t0
    summary = aggregate_rows(rows)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_outputs(args.output_dir, rows, summary, config, total_seconds)

    print("\n=== Direction 8 summary: target GVF RMSE ===")
    for method, data in sorted(
        summary["aggregate"].items(),
        key=lambda item: item[1]["target_rmse_mean"],
    ):
        print(
            f"{method:<38} {data['target_rmse_mean']:.6f} "
            f"+/- {data['target_rmse_stderr']:.6f}"
        )
    if summary.get("off_policy_aggregate"):
        print("\n=== Off-policy behavior-mismatch probe ===")
        for method, data in sorted(
            summary["off_policy_aggregate"].items(),
            key=lambda item: item[1]["target_rmse_mean"],
        ):
            print(
                f"{method:<38} {data['target_rmse_mean']:.6f} "
                f"+/- {data['target_rmse_stderr']:.6f} "
                f"rho_mean={data['rho_mean']:.3f} rho_max={data['rho_max']:.3f}"
            )
    if summary.get("off_policy_paired"):
        print("\n=== Off-policy paired diffs vs raw clipped-IS TD ===")
        for name, data in sorted(summary["off_policy_paired"].items()):
            print(
                f"{name:<74} {data['rmse_diff_mean']:.6f} "
                f"+/- {data['rmse_diff_stderr']:.6f} "
                f"wins/losses/ties={data['wins']}/{data['losses']}/{data['ties']}"
            )
    print(
        "\nBest discovery method: "
        f"{summary['best_discovery_method']} | "
        f"beats linear={summary['best_discovery_beats_linear']} | "
        f"beats MLP={summary['best_discovery_beats_mlp']}"
    )
    print(f"Wrote {args.output_dir / 'results.csv'} and {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

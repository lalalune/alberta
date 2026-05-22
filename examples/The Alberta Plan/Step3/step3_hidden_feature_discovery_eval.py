#!/usr/bin/env python3
"""DoD-7 Hidden: TD/GVF feature discovery on a hidden-state AR(2) stream.

This is the hidden-state companion to ``step3_feature_discovery_eval.py``. The
observable AR(1) eval already shows TD-surprise interaction features beating
raw linear/MLP GVFs, but only when the relevant nonlinear structure is in the
*visible* part of the state (a positive control). This script tests whether
discovery + history features can recover predictions from a partially observed
AR(2) latent state where neither the linear nor the nonlinear part of the
target is reconstructible from the masked observation alone.

Stream
------
``HiddenStateAR2Stream`` (defined in ``streams/synthetic.py``):

* 8-channel state with stationary AR(2) dynamics
  ``x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + sigma * eps_t``.
* The first ``visible_dim=2`` channels are always visible. The remaining
  6 channels are hidden through ``PartialObservationWrapper(MaskMode.PERIODIC)``
  with a 3-step schedule that hides the hidden block on 2 out of every 3
  steps. (One out of every three steps the agent gets a full-state glimpse.)
* The target is ``y_t = weights @ x_t + alpha * x_t[i] * x_t[j] + eta_t``
  with ``[i, j]`` two distinct indices in the hidden block. This couples
  the target to the hidden state both linearly and through a hidden-by-hidden
  product, so visible-only features cannot represent it.

Conditions
----------
Each condition trains a Horde with a single ``gamma=0`` prediction demon
on the same observed transitions. Conditions differ only in the feature
representation passed to the linear GVF:

1. ``raw_masked_linear_gvf``: raw masked observations (negative control).
2. ``raw_masked_history_linear_gvf``: + multi-timescale history EMAs of the
   masked observations (recurrent baseline).
3. ``raw_masked_history_interaction_linear_gvf``: + active interaction-feature
   bank (FixedBudgetInteractionLearner, frozen after warmup).
4. ``raw_masked_history_cumulant_linear_gvf``: + 8 surprise-driven cumulant
   projections of the augmented (raw + history) observation, frozen after
   warmup, treated as auxiliary cumulants.
5. ``raw_masked_history_interaction_cumulant_linear_gvf``: full stack,
   interaction features and discovered auxiliary cumulants together.

Metric
------
Target-GVF RMSE on a held-out evaluation window of the last
``eval_steps`` transitions (default 500).

Output
------
* ``output/step3_dod7_hidden/results.csv`` — per-seed per-condition rows.
* ``output/step3_dod7_hidden/summary.json`` — aggregate stats and config.
* ``output/step3_dod7_hidden/SUMMARY.md`` — markdown table + verdict.
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
    FixedBudgetInteractionLearner,
    GVFSpec,
    HiddenStateAR2Stream,
    HistoryFeatureExtractor,
    HordeLearner,
    MaskMode,
    ObGDBounding,
    PartialObservationWrapper,
    create_horde_spec,
)

DEFAULT_OUTPUT_DIR = Path("output/step3_dod7_hidden")


# =============================================================================
# Stream collection
# =============================================================================


def collect_hidden_ar2_arrays(
    *,
    seed: int,
    total_steps: int,
    feature_dim: int,
    visible_dim: int,
    phi1: float,
    phi2: float,
    innovation_std: float,
    nonlinear_coeff: float,
    target_noise_std: float,
    mask_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect masked observations and unmasked targets from the AR(2) stream.

    The mask schedule has length ``mask_period`` and hides the last
    ``feature_dim - visible_dim`` channels except on the last step of each
    cycle. This gives the agent occasional glimpses of the hidden block but
    forces it to rely on memory and discovery for the masked steps.

    Returns:
        Tuple ``(observations, targets)`` of shape ``(total_steps, feature_dim)``
        and ``(total_steps, 1)`` respectively.
    """
    inner = HiddenStateAR2Stream(
        feature_dim=feature_dim,
        visible_dim=visible_dim,
        phi1=phi1,
        phi2=phi2,
        innovation_std=innovation_std,
        nonlinear_coeff=nonlinear_coeff,
        target_noise_std=target_noise_std,
    )
    visible_only = jnp.array(
        [True] * visible_dim + [False] * (feature_dim - visible_dim)
    )
    full = jnp.array([True] * feature_dim)
    schedule_list = [visible_only] * (mask_period - 1) + [full]
    schedule = tuple(schedule_list)
    wrapped = PartialObservationWrapper(
        inner,
        mode=MaskMode.PERIODIC,
        schedule=schedule,
    )
    state = wrapped.init(jr.key(seed))
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(total_steps):
        timestep, state = wrapped.step(state, jnp.asarray(idx, dtype=jnp.int32))
        observations.append(np.asarray(timestep.observation, dtype=np.float32))
        targets.append(np.asarray(timestep.target, dtype=np.float32))
    return np.stack(observations), np.stack(targets)


def transition_view(
    observations: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-step arrays to transition arrays ``(s_t, s_{t+1}, c_{t+1})``."""
    obs_t = observations[:-1]
    obs_tp1 = observations[1:]
    transition_cumulants = targets[1:]
    return obs_t, obs_tp1, transition_cumulants


# =============================================================================
# Feature pipelines
# =============================================================================


def history_extract(
    observations: np.ndarray,
    *,
    decay_rates: tuple[float, ...],
) -> np.ndarray:
    """Apply ``HistoryFeatureExtractor`` to each row in temporal order.

    Returns an augmented array of shape ``(T, feature_dim + len(decays)*feature_dim)``
    where the first ``feature_dim`` columns are the raw observation and the
    rest are the per-decay EMA traces (one per channel per decay).
    """
    extractor = HistoryFeatureExtractor(
        raw_dim=observations.shape[1],
        decay_rates=decay_rates,
        include_raw=True,
    )
    state = extractor.init()
    rows: list[np.ndarray] = []
    for obs in observations:
        aug, state = extractor.step(state, jnp.asarray(obs))
        rows.append(np.asarray(aug, dtype=np.float32))
    return np.stack(rows)


def warm_interaction_features(
    obs: np.ndarray,
    cumulants: np.ndarray,
    *,
    n_features: int,
    candidate_count: int,
    step_size_output: float,
    replacement_interval: int,
    min_feature_age: int,
    candidate_min_age: int,
    include_squares: bool,
    key: jnp.ndarray,
) -> tuple[FixedBudgetInteractionLearner, Any]:
    """Run a Step-2 interaction-feature learner over the warmup prefix."""
    learner = FixedBudgetInteractionLearner(
        n_features=n_features,
        n_tasks=cumulants.shape[1],
        step_size_output=step_size_output,
        replacement_interval=replacement_interval,
        min_feature_age=min_feature_age,
        candidate_count=candidate_count,
        candidate_min_age=candidate_min_age,
        promotion_margin=1.05,
        promotion_blend=0.5,
        candidate_strategy="all_pairs",
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        include_squares=include_squares,
        use_obgd=True,
        obgd_kappa=2.0,
    )
    state = learner.init(obs.shape[1], key)
    for t in range(obs.shape[0]):
        result = learner.update(state, jnp.asarray(obs[t]), jnp.asarray(cumulants[t]))
        state = result.state
    return learner, state


def augment_with_frozen_interactions(
    learner: FixedBudgetInteractionLearner,
    state: Any,
    observations: np.ndarray,
) -> np.ndarray:
    """Concatenate raw observation with frozen pair-product features."""
    return np.stack(
        [
            np.asarray(learner.augmented_observation(state, jnp.asarray(obs)))
            for obs in observations
        ]
    ).astype(np.float32)


def warm_cumulant_discovery(
    obs: np.ndarray,
    next_obs: np.ndarray,
    *,
    n_aux: int,
    replacement_rate: float,
    maturity_threshold: int,
    predictor_step_size: float,
    key: jnp.ndarray,
) -> np.ndarray:
    """Run cumulant-discovery warmup; return frozen projection rows.

    The projections are unit-norm linear maps from the augmented observation
    to scalar cumulant candidates. After warmup we freeze them and use
    ``next_obs @ projections.T`` as auxiliary cumulants for the downstream
    Horde, in transition convention.
    """
    discovery = CumulantDiscovery(
        raw_dim=obs.shape[1],
        n_candidates=n_aux,
        decay_rate=0.99,
        replacement_rate=replacement_rate,
        maturity_threshold=maturity_threshold,
        predictor_step_size=predictor_step_size,
        gamma=0.0,
        enabled=True,
    )
    state = discovery.init(key)
    for t in range(obs.shape[0]):
        state = discovery.step(
            state, jnp.asarray(obs[t]), jnp.asarray(next_obs[t])
        )
        state = discovery.maybe_replace(state)
    return np.asarray(state.projections, dtype=np.float32)


def projected_cumulants(
    projections: np.ndarray, next_observations: np.ndarray
) -> np.ndarray:
    """Apply frozen projection rows to ``next_obs`` to produce aux cumulants."""
    return np.asarray(next_observations @ projections.T, dtype=np.float32)


# =============================================================================
# GVF training and evaluation
# =============================================================================


def make_horde(
    *,
    n_target: int,
    n_aux: int,
    feature_dim: int,
    step_size: float,
    key: jnp.ndarray,
) -> tuple[HordeLearner, Any]:
    """Build a linear Horde with one γ=0 target demon and optional aux demons."""
    target_specs = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"target{i}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=i,
        )
        for i in range(n_target)
    ]
    aux_specs = [
        GVFSpec(  # type: ignore[call-arg]
            name=f"aux{i}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=n_target + i,
        )
        for i in range(n_aux)
    ]
    horde = HordeLearner(
        horde_spec=create_horde_spec([*target_specs, *aux_specs]),
        hidden_sizes=(),
        step_size=step_size,
        bounder=ObGDBounding(kappa=2.0),
        sparsity=0.0,
        use_layer_norm=False,
    )
    return horde, horde.init(feature_dim, key)


def run_horde_predictions(
    *,
    obs_t: np.ndarray,
    obs_tp1: np.ndarray,
    target_cumulants: np.ndarray,
    aux_cumulants: np.ndarray | None,
    step_size: float,
    key: jnp.ndarray,
) -> np.ndarray:
    """Train a linear Horde online and return target-demon predictions per step.

    The first ``n_target`` columns of the prediction array are the target
    demons; auxiliary demons coexist but are not scored.
    """
    n_target = target_cumulants.shape[1]
    n_aux = 0 if aux_cumulants is None else aux_cumulants.shape[1]
    horde, state = make_horde(
        n_target=n_target,
        n_aux=n_aux,
        feature_dim=obs_t.shape[1],
        step_size=step_size,
        key=key,
    )
    if aux_cumulants is None:
        all_cumulants = target_cumulants
    else:
        all_cumulants = np.concatenate(
            [target_cumulants, aux_cumulants], axis=1
        ).astype(np.float32)
    predictions = np.zeros((obs_t.shape[0], n_target), dtype=np.float32)
    for t in range(obs_t.shape[0]):
        result = horde.update(
            state,
            jnp.asarray(obs_t[t]),
            jnp.asarray(all_cumulants[t]),
            jnp.asarray(obs_tp1[t]),
        )
        predictions[t] = np.asarray(result.predictions[:n_target])
        state = result.state
    return predictions


def held_out_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    eval_steps: int,
) -> float:
    """RMSE over the last ``eval_steps`` rows."""
    end = predictions.shape[0]
    if eval_steps >= end:
        raise ValueError(f"eval_steps={eval_steps} >= total transitions {end}")
    errors = predictions[-eval_steps:] - targets[-eval_steps:]
    return float(np.sqrt(np.mean(errors**2)))


# =============================================================================
# Single-seed driver
# =============================================================================


def run_seed(seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    total_steps = args.warmup_steps + args.eval_steps + 1
    observations, targets = collect_hidden_ar2_arrays(
        seed=seed,
        total_steps=total_steps,
        feature_dim=args.feature_dim,
        visible_dim=args.visible_dim,
        phi1=args.phi1,
        phi2=args.phi2,
        innovation_std=args.innovation_std,
        nonlinear_coeff=args.nonlinear_coeff,
        target_noise_std=args.target_noise_std,
        mask_period=args.mask_period,
    )
    obs_t, obs_tp1, cumulants = transition_view(observations, targets)
    n_transitions = obs_t.shape[0]

    # Pre-compute history features once over the full sequence.
    history_all = history_extract(
        observations,
        decay_rates=args.history_decay_rates,
    )
    history_t = history_all[:-1]
    history_tp1 = history_all[1:]

    # Discovery warmup uses the first ``warmup_steps`` transitions.
    warmup = args.warmup_steps
    eval_steps = args.eval_steps
    if warmup + eval_steps > n_transitions:
        raise ValueError(
            f"warmup ({warmup}) + eval ({eval_steps}) > transitions ({n_transitions})"
        )

    cumulants_warm = cumulants[:warmup]
    history_t_warm = history_t[:warmup]
    history_tp1_warm = history_tp1[:warmup]

    root = jr.key(seed + 100_000)
    (
        k_baseline_linear,
        k_history_linear,
        k_interaction_warm,
        k_interaction_horde,
        k_cumulant_warm,
        k_cumulant_horde,
        k_full_warm_interaction,
        k_full_warm_cumulant,
        k_full_horde,
    ) = jr.split(root, 9)

    rows: list[dict[str, Any]] = []

    # ---------- Condition 1: raw masked observations + linear GVF ----------
    pred = run_horde_predictions(
        obs_t=obs_t,
        obs_tp1=obs_tp1,
        target_cumulants=cumulants,
        aux_cumulants=None,
        step_size=args.linear_step_size,
        key=k_baseline_linear,
    )
    rows.append(
        {
            "seed": seed,
            "condition": "raw_masked_linear_gvf",
            "feature_dim_after": int(obs_t.shape[1]),
            "rmse": held_out_rmse(pred, cumulants, eval_steps=eval_steps),
        }
    )

    # ---------- Condition 2: raw masked + history features + linear GVF ----------
    pred = run_horde_predictions(
        obs_t=history_t,
        obs_tp1=history_tp1,
        target_cumulants=cumulants,
        aux_cumulants=None,
        step_size=args.linear_step_size,
        key=k_history_linear,
    )
    rows.append(
        {
            "seed": seed,
            "condition": "raw_masked_history_linear_gvf",
            "feature_dim_after": int(history_t.shape[1]),
            "rmse": held_out_rmse(pred, cumulants, eval_steps=eval_steps),
        }
    )

    # ---------- Condition 3: + interaction features ----------
    interaction_learner, interaction_state = warm_interaction_features(
        history_t_warm,
        cumulants_warm,
        n_features=args.n_features,
        candidate_count=args.candidate_count,
        step_size_output=args.feature_step_size_output,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        include_squares=args.include_squares,
        key=k_interaction_warm,
    )
    interaction_t = augment_with_frozen_interactions(
        interaction_learner, interaction_state, history_t
    )
    interaction_tp1 = augment_with_frozen_interactions(
        interaction_learner, interaction_state, history_tp1
    )
    pred = run_horde_predictions(
        obs_t=interaction_t,
        obs_tp1=interaction_tp1,
        target_cumulants=cumulants,
        aux_cumulants=None,
        step_size=args.linear_step_size,
        key=k_interaction_horde,
    )
    rows.append(
        {
            "seed": seed,
            "condition": "raw_masked_history_interaction_linear_gvf",
            "feature_dim_after": int(interaction_t.shape[1]),
            "rmse": held_out_rmse(pred, cumulants, eval_steps=eval_steps),
        }
    )

    # ---------- Condition 4: + cumulant discovery ----------
    projections = warm_cumulant_discovery(
        history_t_warm,
        history_tp1_warm,
        n_aux=args.n_aux_cumulants,
        replacement_rate=args.cumulant_replacement_rate,
        maturity_threshold=args.cumulant_maturity,
        predictor_step_size=args.cumulant_predictor_step_size,
        key=k_cumulant_warm,
    )
    aux_cumulants = projected_cumulants(projections, history_tp1)
    pred = run_horde_predictions(
        obs_t=history_t,
        obs_tp1=history_tp1,
        target_cumulants=cumulants,
        aux_cumulants=aux_cumulants,
        step_size=args.linear_step_size,
        key=k_cumulant_horde,
    )
    rows.append(
        {
            "seed": seed,
            "condition": "raw_masked_history_cumulant_linear_gvf",
            "feature_dim_after": int(history_t.shape[1]),
            "n_aux_cumulants": int(aux_cumulants.shape[1]),
            "rmse": held_out_rmse(pred, cumulants, eval_steps=eval_steps),
        }
    )

    # ---------- Condition 5: full stack ----------
    full_interaction_learner, full_interaction_state = warm_interaction_features(
        history_t_warm,
        cumulants_warm,
        n_features=args.n_features,
        candidate_count=args.candidate_count,
        step_size_output=args.feature_step_size_output,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        include_squares=args.include_squares,
        key=k_full_warm_interaction,
    )
    full_interaction_t = augment_with_frozen_interactions(
        full_interaction_learner, full_interaction_state, history_t
    )
    full_interaction_tp1 = augment_with_frozen_interactions(
        full_interaction_learner, full_interaction_state, history_tp1
    )
    full_projections = warm_cumulant_discovery(
        history_t_warm,
        history_tp1_warm,
        n_aux=args.n_aux_cumulants,
        replacement_rate=args.cumulant_replacement_rate,
        maturity_threshold=args.cumulant_maturity,
        predictor_step_size=args.cumulant_predictor_step_size,
        key=k_full_warm_cumulant,
    )
    full_aux_cumulants = projected_cumulants(full_projections, history_tp1)
    pred = run_horde_predictions(
        obs_t=full_interaction_t,
        obs_tp1=full_interaction_tp1,
        target_cumulants=cumulants,
        aux_cumulants=full_aux_cumulants,
        step_size=args.linear_step_size,
        key=k_full_horde,
    )
    rows.append(
        {
            "seed": seed,
            "condition": "raw_masked_history_interaction_cumulant_linear_gvf",
            "feature_dim_after": int(full_interaction_t.shape[1]),
            "n_aux_cumulants": int(full_aux_cumulants.shape[1]),
            "rmse": held_out_rmse(pred, cumulants, eval_steps=eval_steps),
        }
    )

    return rows


# =============================================================================
# Aggregation and IO
# =============================================================================


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(float(row["rmse"]))
    aggregate_table: dict[str, dict[str, float]] = {}
    for cond, vals in by_condition.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = arr.size
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        stderr = std / np.sqrt(n) if n > 0 else 0.0
        aggregate_table[cond] = {
            "rmse_mean": mean,
            "rmse_std": std,
            "rmse_stderr": float(stderr),
            "n_seeds": int(n),
        }

    history = aggregate_table.get("raw_masked_history_linear_gvf")
    full = aggregate_table.get(
        "raw_masked_history_interaction_cumulant_linear_gvf"
    )

    summary: dict[str, Any] = {
        "aggregate": aggregate_table,
        "best_condition": min(
            aggregate_table.items(),
            key=lambda kv: kv[1]["rmse_mean"],
        )[0],
    }
    if history is not None and full is not None:
        diff = history["rmse_mean"] - full["rmse_mean"]
        # Pooled standard error (unpaired) for an order-of-magnitude d
        history_se = history["rmse_stderr"]
        full_se = full["rmse_stderr"]
        pooled_se = float(np.sqrt(history_se**2 + full_se**2))
        summary["full_vs_history"] = {
            "rmse_diff_mean": float(diff),
            "pooled_stderr": pooled_se,
            "z_like": float(diff / pooled_se) if pooled_se > 0 else float("nan"),
            "history_only_mean": history["rmse_mean"],
            "full_stack_mean": full["rmse_mean"],
        }
        # Verdict d > 1.0 if (history - full) / pooled_se > 1.0 AND full < history
        summary["full_beats_history_d_gt_1"] = (
            diff > 0.0 and pooled_se > 0.0 and (diff / pooled_se) > 1.0
        )
    return summary


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
    total_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = [
        "seed",
        "condition",
        "feature_dim_after",
        "n_aux_cumulants",
        "rmse",
    ]
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # JSON
    payload = {
        "config": config,
        "rows": rows,
        "summary": summary,
        "total_seconds": total_seconds,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # SUMMARY.md
    lines = [
        "# DoD-7 Hidden: TD/GVF Feature Discovery on Hidden-State AR(2)",
        "",
        f"Total seconds: {total_seconds:.2f}",
        "",
        "## Stream",
        "",
        "- `HiddenStateAR2Stream` with stationary AR(2) dynamics.",
        f"- `feature_dim={config['feature_dim']}`, `visible_dim={config['visible_dim']}`.",
        f"- `phi1={config['phi1']}`, `phi2={config['phi2']}`,"
        f" `innovation_std={config['innovation_std']}`.",
        f"- Target: `weights @ x_t + {config['nonlinear_coeff']} *"
        f" x_t[hidden_pair[0]] * x_t[hidden_pair[1]] + N(0, {config['target_noise_std']})`.",
        f"- Mask schedule period={config['mask_period']}, hides hidden block on"
        f" {config['mask_period'] - 1}/{config['mask_period']} steps.",
        "",
        "## Aggregate Target GVF RMSE (held-out tail)",
        "",
        "| Condition | RMSE | StdErr | n |",
        "|---|---:|---:|---:|",
    ]
    for cond, data in sorted(
        summary["aggregate"].items(), key=lambda kv: kv[1]["rmse_mean"]
    ):
        lines.append(
            f"| `{cond}` | {data['rmse_mean']:.6f} |"
            f" {data['rmse_stderr']:.6f} | {data['n_seeds']} |"
        )

    if "full_vs_history" in summary:
        cmp = summary["full_vs_history"]
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                f"- Best condition: `{summary['best_condition']}`.",
                f"- History-only RMSE: {cmp['history_only_mean']:.6f}.",
                f"- Full-stack RMSE: {cmp['full_stack_mean']:.6f}.",
                f"- Mean diff (history - full): {cmp['rmse_diff_mean']:.6f}.",
                f"- Pooled stderr: {cmp['pooled_stderr']:.6f}.",
                f"- z-like (diff / pooled stderr): {cmp['z_like']:.3f}.",
                f"- Full stack beats history at d > 1.0:"
                f" {summary['full_beats_history_d_gt_1']}.",
            ]
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--eval-steps", type=int, default=500)
    # Stream config
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--visible-dim", type=int, default=2)
    parser.add_argument("--phi1", type=float, default=0.6)
    parser.add_argument("--phi2", type=float, default=0.3)
    parser.add_argument("--innovation-std", type=float, default=1.0)
    parser.add_argument("--nonlinear-coeff", type=float, default=0.5)
    parser.add_argument("--target-noise-std", type=float, default=0.05)
    parser.add_argument("--mask-period", type=int, default=3)
    # Linear GVF config
    parser.add_argument("--linear-step-size", type=float, default=0.02)
    # History features
    parser.add_argument(
        "--history-decay-rates",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.95, 0.99],
    )
    # Interaction features
    parser.add_argument("--n-features", type=int, default=24)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--feature-step-size-output", type=float, default=0.03)
    parser.add_argument("--replacement-interval", type=int, default=50)
    parser.add_argument("--min-feature-age", type=int, default=50)
    parser.add_argument("--candidate-min-age", type=int, default=25)
    parser.add_argument("--include-squares", action="store_true", default=True)
    parser.add_argument(
        "--no-include-squares", action="store_false", dest="include_squares"
    )
    # Cumulant discovery
    parser.add_argument("--n-aux-cumulants", type=int, default=8)
    parser.add_argument("--cumulant-replacement-rate", type=float, default=0.02)
    parser.add_argument("--cumulant-maturity", type=int, default=80)
    parser.add_argument(
        "--cumulant-predictor-step-size", type=float, default=0.05
    )
    args = parser.parse_args()
    args.history_decay_rates = tuple(float(r) for r in args.history_decay_rates)
    return args


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        rows.extend(run_seed(seed, args))
    total_seconds = time.perf_counter() - t0
    summary = aggregate(rows)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_outputs(args.output_dir, rows, summary, config, total_seconds)

    print("\n=== DoD-7 Hidden: target GVF RMSE on AR(2) hidden-state stream ===")
    for cond, data in sorted(
        summary["aggregate"].items(), key=lambda kv: kv[1]["rmse_mean"]
    ):
        print(
            f"{cond:<55} {data['rmse_mean']:.6f} "
            f"+/- {data['rmse_stderr']:.6f} (n={data['n_seeds']})"
        )
    if "full_vs_history" in summary:
        cmp = summary["full_vs_history"]
        print(
            f"\nFull stack vs history-only: diff={cmp['rmse_diff_mean']:.6f}, "
            f"pooled_se={cmp['pooled_stderr']:.6f}, "
            f"z-like={cmp['z_like']:.3f}, beats at d>1.0="
            f"{summary['full_beats_history_d_gt_1']}"
        )
    print(
        f"\nWrote {args.output_dir / 'results.csv'},"
        f" {args.output_dir / 'summary.json'},"
        f" {args.output_dir / 'SUMMARY.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

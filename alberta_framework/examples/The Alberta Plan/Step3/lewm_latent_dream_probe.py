"""Low-dimensional LeWM-style latent dreaming probe.

This experiment tests the smallest concrete version of the current idea:

1. Learn a latent action-conditioned world model online.
2. Learn a behavior/agent model online in the same latent state.
3. Score candidate dreams by surprise plus utility instead of using every
   model-generated transition.

The environment has rare useful jumps. The acceptance question is whether the
surprise/utility selector enriches those rare useful transitions compared with
the full stream/candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.behavior_model import BehaviorModel, BehaviorModelConfig
from alberta_framework.core.dreaming import DreamSelectionConfig, score_dream_candidates
from alberta_framework.core.latent_world_model import LatentWorldModel, LatentWorldModelConfig
from alberta_framework.core.sigreg import (
    SIGRegConfig,
    sample_sigreg_directions,
    sigreg_diagnostics,
)

OBSERVATION_DIM = 4
N_ACTIONS = 2


@dataclasses.dataclass(frozen=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    next_observations: np.ndarray
    rare: np.ndarray


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    seeds: int
    steps: int
    candidate_anchors: int
    dream_budget: int
    gamma: float
    confidence_weight: float
    sigreg_samples: int
    sigreg_projections: int
    output_dir: str


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _environment_step(
    observation: np.ndarray,
    action: int,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, float, float, bool]:
    """Transition for a small controlled system with rare useful jumps."""
    force = 1.0 if action == 1 else -1.0
    noise = (
        np.zeros((OBSERVATION_DIM,), dtype=np.float32)
        if rng is None
        else rng.normal(0.0, 0.03, size=OBSERVATION_DIM).astype(np.float32)
    )
    next_obs = np.empty_like(observation)
    next_obs[0] = 0.92 * observation[0] + 0.18 * force + noise[0]
    next_obs[1] = 0.80 * observation[1] + 0.15 * observation[0] + noise[1]
    next_obs[2] = 0.72 * observation[2] + 0.12 * observation[1] + 0.04 * force + noise[2]
    next_obs[3] = 0.85 * observation[3] + 0.05 * observation[2] + noise[3]

    rare = bool(observation[0] > 0.45 and action == 1)
    if rare:
        next_obs[2] = next_obs[2] + 1.0
        next_obs[3] = -0.45 * observation[3] + 1.0

    next_obs = np.clip(next_obs, -2.0, 2.0).astype(np.float32)
    reward = 1.0 if rare else float(-0.01 * np.sum(next_obs * next_obs))
    return next_obs, reward, 0.99, rare


def _behavior_action(observation: np.ndarray, rng: np.random.Generator) -> int:
    """Non-uniform behavior so the agent model has something to learn."""
    probability_action_1 = _sigmoid(1.8 * float(observation[0]) - 0.4 * float(observation[2]))
    return int(rng.random() < probability_action_1)


def collect_transitions(seed: int, steps: int, gamma: float) -> TransitionBatch:
    """Collect one behavior-policy stream."""
    rng = np.random.default_rng(seed)
    observation = rng.normal(0.0, 0.15, size=OBSERVATION_DIM).astype(np.float32)
    observations = np.zeros((steps, OBSERVATION_DIM), dtype=np.float32)
    next_observations = np.zeros((steps, OBSERVATION_DIM), dtype=np.float32)
    actions = np.zeros((steps,), dtype=np.int32)
    rewards = np.zeros((steps,), dtype=np.float32)
    discounts = np.full((steps,), gamma, dtype=np.float32)
    rare = np.zeros((steps,), dtype=np.bool_)

    for idx in range(steps):
        action = _behavior_action(observation, rng)
        next_observation, reward, discount, is_rare = _environment_step(
            observation,
            action,
            rng,
        )
        observations[idx] = observation
        actions[idx] = action
        rewards[idx] = reward
        discounts[idx] = gamma * float(discount > 0.0)
        next_observations[idx] = next_observation
        rare[idx] = is_rare
        observation = next_observation

    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        discounts=discounts,
        next_observations=next_observations,
        rare=rare,
    )


def _train_models(
    seed: int,
    batch: TransitionBatch,
    gamma: float,
) -> tuple[
    LatentWorldModel,
    Any,
    BehaviorModel,
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    latent_model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=OBSERVATION_DIM,
            n_actions=N_ACTIONS,
            latent_dim=8,
            gamma=gamma,
            hidden_sizes=(),
            step_size=0.04,
            sparsity=0.0,
            use_layer_norm=False,
            include_action_interactions=True,
            surprise_decay=0.995,
            collapse_decay=0.995,
        )
    )
    latent_state = latent_model.init(jr.key(seed))
    behavior_model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=N_ACTIONS,
            step_size=0.03,
            max_gradient_norm=5.0,
            diagnostic_decay=0.995,
        )
    )
    behavior_state = behavior_model.init(
        feature_dim=latent_model.config.latent_dim,
        key=jr.key(seed + 10_000),
    )

    surprises = np.zeros((batch.actions.shape[0],), dtype=np.float32)
    behavior_losses = np.zeros_like(surprises)
    collapse_scores = np.zeros_like(surprises)

    for idx in range(batch.actions.shape[0]):
        obs = jnp.asarray(batch.observations[idx], dtype=jnp.float32)
        action = jnp.asarray(batch.actions[idx], dtype=jnp.int32)
        latent = latent_model.encode(latent_state, obs)
        behavior_result = behavior_model.update(behavior_state, latent, action)
        behavior_state = behavior_result.state
        world_result = latent_model.update(
            latent_state,
            obs,
            action,
            jnp.asarray(batch.rewards[idx], dtype=jnp.float32),
            jnp.asarray(batch.discounts[idx], dtype=jnp.float32),
            jnp.asarray(batch.next_observations[idx], dtype=jnp.float32),
        )
        latent_state = world_result.state
        surprises[idx] = float(world_result.surprise)
        behavior_losses[idx] = float(behavior_result.loss)
        collapse_scores[idx] = float(world_result.collapse_score)

    return (
        latent_model,
        latent_state,
        behavior_model,
        behavior_state,
        surprises,
        behavior_losses,
        collapse_scores,
    )


def _candidate_pool(
    model: LatentWorldModel,
    model_state: Any,
    behavior_model: BehaviorModel,
    behavior_state: Any,
    batch: TransitionBatch,
    anchor_count: int,
    anchor_scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if anchor_scores is None:
        anchors = batch.observations[-anchor_count:]
    else:
        anchor_indices = np.argsort(-anchor_scores)[:anchor_count]
        anchors = batch.observations[anchor_indices]
    n_candidates = anchors.shape[0] * N_ACTIONS
    predicted_surprise = np.zeros((n_candidates,), dtype=np.float32)
    predicted_reward = np.zeros((n_candidates,), dtype=np.float32)
    true_reward = np.zeros((n_candidates,), dtype=np.float32)
    true_rare = np.zeros((n_candidates,), dtype=np.bool_)
    behavior_probability = np.zeros((n_candidates,), dtype=np.float32)
    actions = np.zeros((n_candidates,), dtype=np.int32)

    row = 0
    for anchor in anchors:
        latent = model.encode(model_state, jnp.asarray(anchor, dtype=jnp.float32))
        behavior_probs = behavior_model.predict_probabilities(behavior_state, latent)
        for action in range(N_ACTIONS):
            prediction = model.predict_from_latent(
                model_state,
                latent,
                jnp.asarray(action, dtype=jnp.int32),
            )
            predicted_surprise[row] = float(
                jnp.mean((prediction.next_latent - prediction.latent) ** 2)
            )
            predicted_reward[row] = float(prediction.reward)
            behavior_probability[row] = float(behavior_probs[action])
            _, reward, _, rare = _environment_step(anchor, action, None)
            true_reward[row] = reward
            true_rare[row] = rare
            actions[row] = action
            row += 1

    return {
        "predicted_surprise": predicted_surprise,
        "predicted_reward": predicted_reward,
        "true_reward": true_reward,
        "true_rare": true_rare,
        "behavior_probability": behavior_probability,
        "actions": actions,
    }


def _rate(values: np.ndarray) -> float:
    return float(np.mean(values.astype(np.float64))) if values.size else 0.0


def _masked_mean(values: np.ndarray, mask: np.ndarray, *, default: float = 0.0) -> float:
    """Mean over a boolean mask with an explicit empty-set default."""
    selected = values[mask]
    if selected.size == 0:
        return default
    return float(np.mean(selected))


def _selection_metrics(
    *,
    seed: int,
    family: str,
    selector: str,
    rare: np.ndarray,
    true_reward: np.ndarray,
    selected_mask: np.ndarray,
    baseline_rare_rate: float,
    predicted_reward: np.ndarray | None = None,
    behavior_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a common metric row for replay and dream selectors."""
    if predicted_reward is None:
        predicted_reward = true_reward
    if behavior_probability is None:
        behavior_probability = np.full_like(true_reward, np.nan, dtype=np.float32)
    rare_rate = _rate(rare[selected_mask])
    return {
        "seed": seed,
        "family": family,
        "selector": selector,
        "selected_count": int(np.sum(selected_mask)),
        "baseline_rare_rate": baseline_rare_rate,
        "selected_rare_rate": rare_rate,
        "rare_enrichment": rare_rate / max(baseline_rare_rate, 1e-12),
        "selected_true_reward": _masked_mean(true_reward, selected_mask, default=np.nan),
        "selected_predicted_reward": _masked_mean(
            predicted_reward,
            selected_mask,
            default=np.nan,
        ),
        "selected_behavior_probability": _masked_mean(
            behavior_probability,
            selected_mask,
            default=np.nan,
        ),
    }


def _random_mask(size: int, budget: int, seed: int) -> np.ndarray:
    """Return a deterministic random selection mask."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size,), dtype=np.bool_)
    if size == 0:
        return mask
    chosen = rng.choice(size, size=min(size, budget), replace=False)
    mask[chosen] = True
    return mask


def _latent_sigreg_summary(
    model: LatentWorldModel,
    state: Any,
    observations: np.ndarray,
    *,
    seed: int,
    sample_count: int,
    n_projections: int,
) -> dict[str, float]:
    """Compute SIGReg diagnostics on a bounded latent sample."""
    count = min(sample_count, observations.shape[0])
    sampled = observations[-count:]
    latents = np.stack(
        [
            np.asarray(model.encode(state, jnp.asarray(obs, dtype=jnp.float32)))
            for obs in sampled
        ],
        axis=0,
    )
    config = SIGRegConfig(n_projections=n_projections)
    directions = sample_sigreg_directions(
        jr.key(seed + 20_000),
        model.config.latent_dim,
        config,
    )
    diagnostics = sigreg_diagnostics(jnp.asarray(latents), directions, config)
    return {
        "latent_sigreg_loss": float(diagnostics.loss),
        "latent_sigreg_std_mean": float(diagnostics.latent_std_mean),
        "latent_sigreg_std_min": float(diagnostics.latent_std_min),
        "latent_sigreg_projected_std_mean": float(diagnostics.projected_std_mean),
    }


def run_seed(seed: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one seed and return summary metrics."""
    batch = collect_transitions(seed, args.steps, args.gamma)
    (
        latent_model,
        latent_state,
        behavior_model,
        behavior_state,
        surprises,
        behavior_losses,
        collapse_scores,
    ) = _train_models(seed, batch, args.gamma)

    replay_surprise_selection = score_dream_candidates(
        surprises=jnp.asarray(surprises),
        utilities=jnp.zeros_like(jnp.asarray(batch.rewards)),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=1.0,
            utility_weight=0.0,
            min_surprise=args.min_surprise,
            min_utility=-1.0e30,
            model_error_weight=0.0,
        ),
    )
    replay_selection = score_dream_candidates(
        surprises=jnp.asarray(surprises),
        utilities=jnp.asarray(batch.rewards),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=args.surprise_weight,
            utility_weight=args.utility_weight,
            min_surprise=args.min_surprise,
            min_utility=args.min_utility,
            model_error_weight=0.0,
        ),
    )
    replay_surprise_mask = np.asarray(
        replay_surprise_selection.selected_mask,
        dtype=np.bool_,
    )
    replay_mask = np.asarray(replay_selection.selected_mask, dtype=np.bool_)
    anchor_scores = surprises + args.utility_weight * np.maximum(batch.rewards, 0.0)
    pool = _candidate_pool(
        latent_model,
        latent_state,
        behavior_model,
        behavior_state,
        batch,
        args.candidate_anchors,
        anchor_scores,
    )
    confidence = pool["behavior_probability"] / (1.0 + float(latent_state.surprise_ema))

    candidate_random_mask = _random_mask(
        pool["true_rare"].shape[0],
        args.dream_budget,
        seed + 30_000,
    )
    candidate_lwm_only = score_dream_candidates(
        surprises=jnp.asarray(pool["predicted_surprise"]),
        utilities=jnp.zeros_like(jnp.asarray(pool["predicted_reward"])),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=1.0,
            utility_weight=0.0,
            min_surprise=args.min_candidate_surprise,
            min_utility=-1.0e30,
            model_error_weight=0.0,
        ),
    )
    candidate_utility_only = score_dream_candidates(
        surprises=jnp.zeros_like(jnp.asarray(pool["predicted_surprise"])),
        utilities=jnp.asarray(pool["predicted_reward"]),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=0.0,
            utility_weight=1.0,
            min_surprise=0.0,
            min_utility=args.min_candidate_utility,
            model_error_weight=0.0,
        ),
    )
    candidate_lwm_utility = score_dream_candidates(
        surprises=jnp.asarray(pool["predicted_surprise"]),
        utilities=jnp.asarray(pool["predicted_reward"]),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=args.surprise_weight,
            utility_weight=args.utility_weight,
            min_surprise=args.min_candidate_surprise,
            min_utility=args.min_candidate_utility,
            model_error_weight=0.0,
        ),
    )
    candidate_selection = score_dream_candidates(
        surprises=jnp.asarray(pool["predicted_surprise"]),
        utilities=jnp.asarray(pool["predicted_reward"]),
        confidences=jnp.asarray(confidence),
        model_errors=jnp.full_like(
            jnp.asarray(pool["predicted_surprise"]),
            latent_state.prediction_error_ema,
        ),
        config=DreamSelectionConfig(
            max_items=args.dream_budget,
            surprise_weight=args.surprise_weight,
            utility_weight=args.utility_weight,
            confidence_weight=args.confidence_weight,
            min_surprise=args.min_candidate_surprise,
            min_utility=args.min_candidate_utility,
            min_confidence=0.0,
            max_model_error=1.0e30,
            model_error_weight=0.0,
        ),
    )
    candidate_lwm_only_mask = np.asarray(candidate_lwm_only.selected_mask, dtype=np.bool_)
    candidate_utility_only_mask = np.asarray(
        candidate_utility_only.selected_mask,
        dtype=np.bool_,
    )
    candidate_lwm_utility_mask = np.asarray(
        candidate_lwm_utility.selected_mask,
        dtype=np.bool_,
    )
    candidate_mask = np.asarray(candidate_selection.selected_mask, dtype=np.bool_)

    stream_rare_rate = _rate(batch.rare)
    replay_rare_rate = _rate(batch.rare[replay_mask])
    candidate_rare_rate = _rate(pool["true_rare"][candidate_mask])
    all_candidate_rare_rate = _rate(pool["true_rare"])

    ablation_rows = [
        _selection_metrics(
            seed=seed,
            family="replay",
            selector="surprise_only",
            rare=batch.rare,
            true_reward=batch.rewards,
            selected_mask=replay_surprise_mask,
            baseline_rare_rate=stream_rare_rate,
        ),
        _selection_metrics(
            seed=seed,
            family="replay",
            selector="surprise_utility",
            rare=batch.rare,
            true_reward=batch.rewards,
            selected_mask=replay_mask,
            baseline_rare_rate=stream_rare_rate,
        ),
        _selection_metrics(
            seed=seed,
            family="candidate",
            selector="random",
            rare=pool["true_rare"],
            true_reward=pool["true_reward"],
            selected_mask=candidate_random_mask,
            baseline_rare_rate=all_candidate_rare_rate,
            predicted_reward=pool["predicted_reward"],
            behavior_probability=pool["behavior_probability"],
        ),
        _selection_metrics(
            seed=seed,
            family="candidate",
            selector="lwm_surprise_only",
            rare=pool["true_rare"],
            true_reward=pool["true_reward"],
            selected_mask=candidate_lwm_only_mask,
            baseline_rare_rate=all_candidate_rare_rate,
            predicted_reward=pool["predicted_reward"],
            behavior_probability=pool["behavior_probability"],
        ),
        _selection_metrics(
            seed=seed,
            family="candidate",
            selector="utility_only",
            rare=pool["true_rare"],
            true_reward=pool["true_reward"],
            selected_mask=candidate_utility_only_mask,
            baseline_rare_rate=all_candidate_rare_rate,
            predicted_reward=pool["predicted_reward"],
            behavior_probability=pool["behavior_probability"],
        ),
        _selection_metrics(
            seed=seed,
            family="candidate",
            selector="lwm_surprise_utility",
            rare=pool["true_rare"],
            true_reward=pool["true_reward"],
            selected_mask=candidate_lwm_utility_mask,
            baseline_rare_rate=all_candidate_rare_rate,
            predicted_reward=pool["predicted_reward"],
            behavior_probability=pool["behavior_probability"],
        ),
        _selection_metrics(
            seed=seed,
            family="candidate",
            selector="agent_world",
            rare=pool["true_rare"],
            true_reward=pool["true_reward"],
            selected_mask=candidate_mask,
            baseline_rare_rate=all_candidate_rare_rate,
            predicted_reward=pool["predicted_reward"],
            behavior_probability=pool["behavior_probability"],
        ),
    ]

    row = {
        "seed": seed,
        "stream_rare_rate": stream_rare_rate,
        "replay_selected_rare_rate": replay_rare_rate,
        "replay_rare_enrichment": replay_rare_rate / max(stream_rare_rate, 1e-12),
        "candidate_all_rare_rate": all_candidate_rare_rate,
        "candidate_selected_rare_rate": candidate_rare_rate,
        "candidate_rare_enrichment": candidate_rare_rate
        / max(all_candidate_rare_rate, 1e-12),
        "stream_mean_reward": float(np.mean(batch.rewards)),
        "replay_selected_mean_reward": _masked_mean(batch.rewards, replay_mask),
        "candidate_all_true_reward": float(np.mean(pool["true_reward"])),
        "candidate_selected_true_reward": _masked_mean(pool["true_reward"], candidate_mask),
        "candidate_selected_pred_reward": _masked_mean(
            pool["predicted_reward"],
            candidate_mask,
        ),
        "candidate_selected_behavior_prob": _masked_mean(
            pool["behavior_probability"],
            candidate_mask,
        ),
        "candidate_all_behavior_prob": float(np.mean(pool["behavior_probability"])),
        "mean_surprise": float(np.mean(surprises)),
        "replay_selected_mean_surprise": _masked_mean(surprises, replay_mask),
        "final_surprise_ema": float(latent_state.surprise_ema),
        "final_prediction_error_ema": float(latent_state.prediction_error_ema),
        "final_collapse_score_ema": float(latent_state.collapse_score_ema),
        "mean_behavior_nll": float(np.mean(behavior_losses)),
        "mean_collapse_score": float(np.mean(collapse_scores)),
    }
    row.update(
        _latent_sigreg_summary(
            latent_model,
            latent_state,
            batch.observations,
            seed=seed,
            sample_count=args.sigreg_samples,
            n_projections=args.sigreg_projections,
        )
    )
    return row, ablation_rows


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed rows."""
    keys = [
        "stream_rare_rate",
        "replay_selected_rare_rate",
        "replay_rare_enrichment",
        "candidate_all_rare_rate",
        "candidate_selected_rare_rate",
        "candidate_rare_enrichment",
        "stream_mean_reward",
        "replay_selected_mean_reward",
        "candidate_all_true_reward",
        "candidate_selected_true_reward",
        "candidate_selected_pred_reward",
        "candidate_selected_behavior_prob",
        "candidate_all_behavior_prob",
        "mean_surprise",
        "replay_selected_mean_surprise",
        "final_surprise_ema",
        "final_prediction_error_ema",
        "final_collapse_score_ema",
        "mean_behavior_nll",
        "mean_collapse_score",
        "latent_sigreg_loss",
        "latent_sigreg_std_mean",
        "latent_sigreg_std_min",
        "latent_sigreg_projected_std_mean",
    ]
    return {key: _mean_std([float(row[key]) for row in rows]) for key in keys}


def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate long-form selector ablation rows."""
    metrics = [
        "selected_count",
        "baseline_rare_rate",
        "selected_rare_rate",
        "rare_enrichment",
        "selected_true_reward",
        "selected_predicted_reward",
        "selected_behavior_probability",
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['family']}:{row['selector']}"
        groups.setdefault(key, []).append(row)
    return {
        key: {
            metric: _mean_std([float(row[metric]) for row in group_rows])
            for metric in metrics
        }
        for key, group_rows in sorted(groups.items())
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    ablation_summary: dict[str, Any],
    config: ProbeConfig,
    total_seconds: float,
) -> None:
    """Write CSV and JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "config": dataclasses.asdict(config),
                "summary": summary,
                "ablation_summary": ablation_summary,
                "total_seconds": total_seconds,
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--candidate-anchors", type=int, default=256)
    parser.add_argument("--dream-budget", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--surprise-weight", type=float, default=1.0)
    parser.add_argument("--utility-weight", type=float, default=2.0)
    parser.add_argument("--confidence-weight", type=float, default=0.2)
    parser.add_argument("--min-surprise", type=float, default=0.0)
    parser.add_argument("--min-utility", type=float, default=0.0)
    parser.add_argument("--min-candidate-surprise", type=float, default=0.0)
    parser.add_argument("--min-candidate-utility", type=float, default=-1.0e30)
    parser.add_argument("--sigreg-samples", type=int, default=512)
    parser.add_argument("--sigreg-projections", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/lewm_latent_dream_probe"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the probe."""
    args = parse_args()
    if args.seeds <= 0:
        raise ValueError("seeds must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.candidate_anchors <= 0:
        raise ValueError("candidate-anchors must be positive")
    if args.dream_budget <= 0:
        raise ValueError("dream-budget must be positive")
    if args.sigreg_samples <= 0:
        raise ValueError("sigreg-samples must be positive")
    if args.sigreg_projections <= 0:
        raise ValueError("sigreg-projections must be positive")

    start = time.perf_counter()
    rows = []
    ablation_rows = []
    for seed in range(args.seeds):
        row, seed_ablation_rows = run_seed(seed, args)
        rows.append(row)
        ablation_rows.extend(seed_ablation_rows)
        print(
            f"seed={seed} stream_rare={row['stream_rare_rate']:.3f} "
            f"replay_rare={row['replay_selected_rare_rate']:.3f} "
            f"candidate_rare={row['candidate_selected_rare_rate']:.3f} "
            f"candidate_reward={row['candidate_selected_true_reward']:.3f}"
        )

    summary = summarize(rows)
    ablation_summary = summarize_ablation(ablation_rows)
    config = ProbeConfig(
        seeds=args.seeds,
        steps=args.steps,
        candidate_anchors=args.candidate_anchors,
        dream_budget=args.dream_budget,
        gamma=args.gamma,
        confidence_weight=args.confidence_weight,
        sigreg_samples=args.sigreg_samples,
        sigreg_projections=args.sigreg_projections,
        output_dir=str(args.output_dir),
    )
    total_seconds = time.perf_counter() - start
    write_outputs(
        args.output_dir,
        rows,
        ablation_rows,
        summary,
        ablation_summary,
        config,
        total_seconds,
    )
    lwm_only = ablation_summary["candidate:lwm_surprise_only"]["rare_enrichment"][
        "mean"
    ]
    agent_world = ablation_summary["candidate:agent_world"]["rare_enrichment"]["mean"]
    print(
        "\nCandidate rare enrichment "
        f"{summary['candidate_rare_enrichment']['mean']:.2f}x; "
        f"LWM-only {lwm_only:.2f}x; agent-world {agent_world:.2f}x; "
        f"wrote {args.output_dir / 'results.csv'} and {args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()

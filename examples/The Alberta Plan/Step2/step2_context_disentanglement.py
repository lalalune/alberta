#!/usr/bin/env python3
"""Step 2 context/output-adaptation disentanglement experiment.

This experiment separates two questions that are mixed in the ordinary
InteractionFeatureDiscoveryStream final-window loss:

1. Did the Step 2 mechanism construct useful pair-product features?
2. Can the output weights adapt quickly enough when hidden context changes?

The protocol is intentionally two-pass.  First, a fixed-budget interaction
learner discovers a bounded feature bank on the hidden-context stream.  Second,
simple online linear readouts are retrained from scratch on frozen
representations:

* raw observations only
* raw observations plus the discovered Step 2 pair features
* raw observations plus all oracle-active pair features

Each representation is tested with hidden single-head output weights,
observable context as one-hot bias input, explicit context-gated feature slopes,
and context-indexed output heads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    FixedBudgetInteractionLearner,
    InteractionFeatureDiscoveryStream,
    run_interaction_feature_arrays,
)


@dataclass
class ExperimentConfig:
    """Configuration for the context disentanglement experiment."""

    num_steps: int = 600
    seeds: int = 3
    feature_dim: int = 8
    n_tasks: int = 2
    n_contexts: int = 4
    context_length: int = 75
    active_pairs: int = 1
    noise_std: float = 0.01
    n_features: int = 8
    candidate_count: int = 64
    replacement_interval: int = 40
    min_feature_age: int = 40
    candidate_min_age: int = 20
    discovery_step_size: float = 0.04
    readout_step_size: float = 0.03
    utility_decay: float = 0.9995
    obgd_kappa: float = 2.0
    include_squares: bool = False
    output_dir: Path | None = None


def summarize_loss(loss_curve: np.ndarray, cycle_length: int) -> dict[str, float]:
    """Summarize an online pre-update loss curve."""
    final_window = max(1, loss_curve.shape[0] // 5)
    first_window = max(1, loss_curve.shape[0] // 5)
    cycle_window = min(loss_curve.shape[0], max(1, cycle_length))
    first = float(np.mean(loss_curve[:first_window]))
    final = float(np.mean(loss_curve[-final_window:]))
    last_cycle = float(np.mean(loss_curve[-cycle_window:]))
    return {
        "mean_loss": float(np.mean(loss_curve)),
        "first_window_loss": first,
        "final_window_loss": final,
        "last_cycle_loss": last_cycle,
        "final_over_first": final / max(first, 1e-12),
    }


def collect_interaction_arrays(
    stream: InteractionFeatureDiscoveryStream,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, np.ndarray, Any]:
    """Collect one stream realization plus externally observable context ids."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    contexts = (
        (np.arange(num_steps) // stream._context_length) % stream._n_contexts
    ).astype(np.int32)
    return observations, targets, contexts, stream_state


def active_oracle_pairs(stream_state: Any) -> list[tuple[int, int]]:
    """Return all hidden pair-product features with nonzero context/task weight."""
    pair_left = np.asarray(stream_state.pair_left)
    pair_right = np.asarray(stream_state.pair_right)
    context_weights = np.asarray(stream_state.context_weights)
    active = np.any(np.abs(context_weights) > 1e-8, axis=(0, 1))
    pairs = {
        (int(left), int(right))
        for left, right, is_active in zip(pair_left, pair_right, active, strict=True)
        if bool(is_active)
    }
    return sorted(pairs)


def state_pairs(state: Any) -> list[tuple[int, int]]:
    """Return unique active pair-product features from an interaction learner state."""
    pairs = {
        (int(left), int(right))
        for left, right in zip(
            np.asarray(state.feature_left),
            np.asarray(state.feature_right),
            strict=True,
        )
    }
    return sorted(pairs)


def pair_values(observations: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build a matrix of pair-product features."""
    if not pairs:
        return np.zeros((observations.shape[0], 0), dtype=np.float32)
    left = np.array([pair[0] for pair in pairs], dtype=np.int32)
    right = np.array([pair[1] for pair in pairs], dtype=np.int32)
    return (observations[:, left] * observations[:, right]).astype(np.float32)


def one_hot(indices: np.ndarray, n_values: int) -> np.ndarray:
    """Return one-hot context indicators."""
    values = np.zeros((indices.shape[0], n_values), dtype=np.float32)
    values[np.arange(indices.shape[0]), indices] = 1.0
    return values


def context_gated_features(
    features: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
) -> np.ndarray:
    """Cross every feature with one-hot context indicators."""
    indicators = one_hot(contexts, n_contexts)
    gated = indicators[:, :, None] * features[:, None, :]
    gated = gated.reshape(features.shape[0], n_contexts * features.shape[1])
    return np.concatenate([indicators, gated], axis=1).astype(np.float32)


def run_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
    step_size: float,
    mode: str,
) -> np.ndarray:
    """Run an online linear readout probe and return pre-update MSE per step."""
    valid_modes = {
        "hidden_single",
        "observable_single",
        "context_gated_slopes",
        "context_indexed",
    }
    if mode not in valid_modes:
        raise ValueError(f"unknown probe mode {mode}")

    x = features.astype(np.float32)
    if mode == "observable_single":
        x = np.concatenate([x, one_hot(contexts, n_contexts)], axis=1)
    elif mode == "context_gated_slopes":
        x = context_gated_features(x, contexts, n_contexts)

    n_tasks = targets.shape[1]
    active_scale = 1.0 / max(float(n_tasks), 1.0)
    losses = np.empty(x.shape[0], dtype=np.float32)

    if mode == "context_indexed":
        weights = np.zeros((n_contexts, n_tasks, x.shape[1]), dtype=np.float32)
        biases = np.zeros((n_contexts, n_tasks), dtype=np.float32)
        for t, (features_t, target_t, context_t) in enumerate(zip(x, targets, contexts)):
            prediction = weights[context_t] @ features_t + biases[context_t]
            error = target_t - prediction
            losses[t] = float(np.mean(error**2))
            weights[context_t] += step_size * active_scale * error[:, None] * features_t
            biases[context_t] += step_size * active_scale * error
        return losses

    weights = np.zeros((n_tasks, x.shape[1]), dtype=np.float32)
    biases = np.zeros(n_tasks, dtype=np.float32)
    for t, (features_t, target_t) in enumerate(zip(x, targets)):
        prediction = weights @ features_t + biases
        error = target_t - prediction
        losses[t] = float(np.mean(error**2))
        weights += step_size * active_scale * error[:, None] * features_t
        biases += step_size * active_scale * error
    return losses


def discover_features(
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    config: ExperimentConfig,
) -> tuple[Any, np.ndarray]:
    """Run the Step 2 interaction learner used only for feature construction."""
    learner = FixedBudgetInteractionLearner(
        n_features=config.n_features,
        n_tasks=config.n_tasks,
        step_size_output=config.discovery_step_size,
        utility_decay=config.utility_decay,
        replacement_interval=config.replacement_interval,
        min_feature_age=config.min_feature_age,
        candidate_count=config.candidate_count,
        candidate_min_age=config.candidate_min_age,
        promotion_margin=1.05,
        promotion_blend=0.5,
        candidate_strategy="all_pairs",
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        include_squares=config.include_squares,
        use_obgd=True,
        obgd_kappa=config.obgd_kappa,
    )
    initial_state = learner.init(config.feature_dim, key)
    result = run_interaction_feature_arrays(learner, initial_state, observations, targets)
    return result.state, np.asarray(result.metrics[:, 0])


def run_seed(seed: int, config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one seed and return readout rows plus construction diagnostics."""
    root_key = jr.key(seed)
    data_key, discovery_key = jr.split(root_key)
    stream = InteractionFeatureDiscoveryStream(
        feature_dim=config.feature_dim,
        n_tasks=config.n_tasks,
        n_contexts=config.n_contexts,
        context_length=config.context_length,
        active_pairs_per_context=config.active_pairs,
        noise_std=config.noise_std,
        include_squares=config.include_squares,
    )
    observations_jax, targets_jax, contexts, stream_state = collect_interaction_arrays(
        stream,
        config.num_steps,
        data_key,
    )
    discovery_state, discovery_loss = discover_features(
        observations_jax,
        targets_jax,
        discovery_key,
        config,
    )

    observations = np.asarray(observations_jax)
    targets = np.asarray(targets_jax)
    oracle_pairs = active_oracle_pairs(stream_state)
    learned_pairs = state_pairs(discovery_state)
    learned_hits = sorted(set(learned_pairs) & set(oracle_pairs))

    representations = {
        "raw": observations,
        "learned_augmented": np.concatenate(
            [observations, pair_values(observations, learned_pairs)],
            axis=1,
        ),
        "oracle_augmented": np.concatenate(
            [observations, pair_values(observations, oracle_pairs)],
            axis=1,
        ),
    }
    modes = (
        "hidden_single",
        "observable_single",
        "context_gated_slopes",
        "context_indexed",
    )
    cycle_length = config.n_contexts * config.context_length
    rows: list[dict[str, Any]] = []

    for representation_name, features in representations.items():
        for mode in modes:
            loss = run_linear_probe(
                features,
                targets,
                contexts,
                config.n_contexts,
                config.readout_step_size,
                mode,
            )
            rows.append(
                {
                    "seed": seed,
                    "representation": representation_name,
                    "adapter": mode,
                    "feature_dim": int(features.shape[1]),
                    **summarize_loss(loss, cycle_length),
                }
            )

    diagnostics = {
        "seed": seed,
        "discovery_final_window_loss": summarize_loss(
            discovery_loss,
            cycle_length,
        )["final_window_loss"],
        "oracle_pair_count": len(oracle_pairs),
        "learned_unique_pair_count": len(learned_pairs),
        "learned_oracle_pair_hits": len(learned_hits),
        "learned_oracle_pair_fraction": len(learned_hits) / max(len(learned_pairs), 1),
        "oracle_pairs": oracle_pairs,
        "learned_pairs": learned_pairs,
        "learned_oracle_pairs": learned_hits,
    }
    return rows, diagnostics


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed rows by representation and readout adapter."""
    aggregate: dict[str, Any] = {}
    keys = sorted({(row["representation"], row["adapter"]) for row in rows})
    metrics = (
        "mean_loss",
        "first_window_loss",
        "final_window_loss",
        "last_cycle_loss",
        "final_over_first",
    )
    for representation, adapter in keys:
        group = [
            row
            for row in rows
            if row["representation"] == representation and row["adapter"] == adapter
        ]
        name = f"{representation}:{adapter}"
        aggregate[name] = {
            metric: float(np.mean([row[metric] for row in group])) for metric in metrics
        }
        aggregate[name].update(
            {
                f"stderr_{metric}": float(
                    np.std([row[metric] for row in group], ddof=0)
                    / np.sqrt(len(group))
                )
                for metric in metrics
            }
        )
        aggregate[name]["n_seeds"] = len(group)
    return aggregate


def run_suite(config: ExperimentConfig) -> dict[str, Any]:
    """Run all seeds and return JSON-serializable results."""
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for seed in range(config.seeds):
        rows, diag = run_seed(seed, config)
        all_rows.extend(rows)
        diagnostics.append(diag)
        print(
            f"seed={seed:<3} "
            f"pair_hits={diag['learned_oracle_pair_hits']}/"
            f"{diag['learned_unique_pair_count']} "
            f"discovery_final={diag['discovery_final_window_loss']:.6f}"
        )

    aggregate = aggregate_rows(all_rows)
    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": all_rows,
        "construction_diagnostics": diagnostics,
        "aggregate": aggregate,
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a compact table sorted by last-cycle loss."""
    print("\nAggregate probe losses:")
    for name, data in sorted(
        results["aggregate"].items(),
        key=lambda item: item[1]["last_cycle_loss"],
    ):
        print(
            f"{name:<42} "
            f"final={data['final_window_loss']:.6f} "
            f"last_cycle={data['last_cycle_loss']:.6f}"
        )


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into an experiment config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a fast smoke config")
    parser.add_argument("--num-steps", type=int, default=ExperimentConfig.num_steps)
    parser.add_argument("--seeds", type=int, default=ExperimentConfig.seeds)
    parser.add_argument("--feature-dim", type=int, default=ExperimentConfig.feature_dim)
    parser.add_argument("--n-tasks", type=int, default=ExperimentConfig.n_tasks)
    parser.add_argument("--n-contexts", type=int, default=ExperimentConfig.n_contexts)
    parser.add_argument("--context-length", type=int, default=ExperimentConfig.context_length)
    parser.add_argument("--active-pairs", type=int, default=ExperimentConfig.active_pairs)
    parser.add_argument("--noise-std", type=float, default=ExperimentConfig.noise_std)
    parser.add_argument("--n-features", type=int, default=ExperimentConfig.n_features)
    parser.add_argument("--candidate-count", type=int, default=ExperimentConfig.candidate_count)
    parser.add_argument(
        "--replacement-interval",
        type=int,
        default=ExperimentConfig.replacement_interval,
    )
    parser.add_argument("--min-feature-age", type=int, default=ExperimentConfig.min_feature_age)
    parser.add_argument(
        "--candidate-min-age",
        type=int,
        default=ExperimentConfig.candidate_min_age,
    )
    parser.add_argument(
        "--discovery-step-size",
        type=float,
        default=ExperimentConfig.discovery_step_size,
    )
    parser.add_argument(
        "--readout-step-size",
        type=float,
        default=ExperimentConfig.readout_step_size,
    )
    parser.add_argument("--utility-decay", type=float, default=ExperimentConfig.utility_decay)
    parser.add_argument("--obgd-kappa", type=float, default=ExperimentConfig.obgd_kappa)
    parser.add_argument("--include-squares", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = ExperimentConfig(
        num_steps=args.num_steps,
        seeds=args.seeds,
        feature_dim=args.feature_dim,
        n_tasks=args.n_tasks,
        n_contexts=args.n_contexts,
        context_length=args.context_length,
        active_pairs=args.active_pairs,
        noise_std=args.noise_std,
        n_features=args.n_features,
        candidate_count=args.candidate_count,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        discovery_step_size=args.discovery_step_size,
        readout_step_size=args.readout_step_size,
        utility_decay=args.utility_decay,
        obgd_kappa=args.obgd_kappa,
        include_squares=args.include_squares,
        output_dir=args.output_dir,
    )
    if args.quick:
        config.num_steps = min(config.num_steps, 240)
        config.seeds = min(config.seeds, 2)
        config.context_length = min(config.context_length, 40)
        config.candidate_count = min(config.candidate_count, 32)
        config.replacement_interval = min(config.replacement_interval, 20)
        config.min_feature_age = min(config.min_feature_age, 20)
        config.candidate_min_age = min(config.candidate_min_age, 10)
    return config


def main() -> None:
    """Run the experiment from the command line."""
    config = parse_args()
    results = run_suite(config)
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {config.output_dir}")
    print_summary(results)


if __name__ == "__main__":
    main()

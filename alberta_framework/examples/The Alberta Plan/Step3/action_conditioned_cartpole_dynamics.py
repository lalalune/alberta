"""Action-conditioned CartPole environment-prediction sweep.

This is the first empirical harness for the environment-prediction lane:
learn ``(observation_t, action_t) -> (observation_{t+1}, reward_{t+1},
discount_{t+1})`` online under a random behavior policy. It compares a linear
multi-head predictor with a nonlinear MLP predictor and writes prequential
evidence that can be promoted into the Step 3/4 research notes.

Example:
    python "examples/The Alberta Plan/Step3/action_conditioned_cartpole_dynamics.py"
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.dreaming import DreamingConfig, GuardedDreamer
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    run_action_conditioned_world_model_learning_loop,
)

DEFAULT_OBSERVATION_SCALE = (2.4, 2.0, 0.21, 2.0)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Serializable experiment configuration."""

    env_id: str
    seeds: int
    train_steps: int
    final_window: int
    gamma: float
    observation_scale: tuple[float, ...]
    output_dir: str


@dataclasses.dataclass(frozen=True)
class TransitionBatch:
    """Real random-policy transitions shared by paired model conditions."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    next_observations: np.ndarray
    final_observation: np.ndarray
    last_action: int
    n_actions: int
    observation_dim: int
    episodes: int


def _parse_observation_scale(value: str) -> tuple[float, ...]:
    scale = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not scale:
        raise argparse.ArgumentTypeError("observation scale must contain at least one value")
    if any(item <= 0.0 for item in scale):
        raise argparse.ArgumentTypeError("observation scale values must be positive")
    return scale


def _make_model(
    method: str,
    *,
    observation_dim: int,
    n_actions: int,
    gamma: float,
    observation_scale: tuple[float, ...],
) -> ActionConditionedWorldModel:
    if method == "linear":
        config = ActionConditionedWorldModelConfig(
            observation_dim=observation_dim,
            n_actions=n_actions,
            gamma=gamma,
            observation_scale=observation_scale,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.995,
            observation_clip_margin=0.25,
        )
    elif method == "mlp":
        config = ActionConditionedWorldModelConfig(
            observation_dim=observation_dim,
            n_actions=n_actions,
            gamma=gamma,
            observation_scale=observation_scale,
            hidden_sizes=(64,),
            step_size=0.01,
            sparsity=0.5,
            use_layer_norm=True,
            error_decay=0.995,
            observation_clip_margin=0.25,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    return ActionConditionedWorldModel(config)


def _rmse(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(array)))


def _last(values: Any, final_window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if final_window <= 0:
        return array
    return array[-final_window:]


def _collect_transitions(
    seed: int,
    args: argparse.Namespace,
) -> TransitionBatch:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise SystemExit("Install the gymnasium extra: pip install -e '.[gymnasium]'") from exc

    env = gym.make(args.env_id)
    reset_result = env.reset(seed=seed)
    observation = np.asarray(reset_result[0], dtype=np.float32)
    n_actions = int(env.action_space.n)
    observation_dim = int(observation.shape[0])
    if len(args.observation_scale) != observation_dim:
        raise ValueError(
            f"observation scale has length {len(args.observation_scale)}, "
            f"but {args.env_id} observations have length {observation_dim}"
        )

    rng = np.random.default_rng(seed)
    observations = np.zeros((args.train_steps, observation_dim), dtype=np.float32)
    actions = np.zeros((args.train_steps,), dtype=np.int32)
    rewards = np.zeros((args.train_steps,), dtype=np.float32)
    discounts = np.zeros((args.train_steps,), dtype=np.float32)
    next_observations = np.zeros((args.train_steps, observation_dim), dtype=np.float32)
    episode_count = 0
    last_action = 0

    for step in range(args.train_steps):
        action = int(rng.integers(n_actions))
        next_observation_raw, reward_raw, terminated, truncated, _ = env.step(action)
        next_observation = np.asarray(next_observation_raw, dtype=np.float32)
        reward = float(reward_raw)
        done = bool(terminated or truncated)
        discount = 0.0 if done else float(args.gamma)

        observations[step] = observation
        actions[step] = action
        rewards[step] = reward
        discounts[step] = discount
        next_observations[step] = next_observation
        last_action = action

        if done:
            episode_count += 1
            observation = np.asarray(env.reset(seed=seed + step + 1)[0], dtype=np.float32)
        else:
            observation = next_observation

    env.close()
    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        discounts=discounts,
        next_observations=next_observations,
        final_observation=observation,
        last_action=last_action,
        n_actions=n_actions,
        observation_dim=observation_dim,
        episodes=episode_count,
    )


def _run_one(
    method: str,
    seed: int,
    args: argparse.Namespace,
    transitions: TransitionBatch,
) -> dict[str, Any]:
    model = _make_model(
        method,
        observation_dim=transitions.observation_dim,
        n_actions=transitions.n_actions,
        gamma=args.gamma,
        observation_scale=args.observation_scale,
    )
    state = model.init(jr.key(seed))
    result = run_action_conditioned_world_model_learning_loop(
        model,
        state,
        jnp.asarray(transitions.observations, dtype=jnp.float32),
        jnp.asarray(transitions.actions, dtype=jnp.int32),
        jnp.asarray(transitions.rewards, dtype=jnp.float32),
        jnp.asarray(transitions.next_observations, dtype=jnp.float32),
        jnp.asarray(transitions.discounts, dtype=jnp.float32),
    )

    observation_mse = np.asarray(result.observation_mse, dtype=np.float64)
    reward_sq_error = np.asarray(result.reward_errors, dtype=np.float64) ** 2
    discount_sq_error = np.asarray(result.discount_errors, dtype=np.float64) ** 2
    zero_delta_mse = np.mean(
        (transitions.next_observations - transitions.observations) ** 2,
        axis=1,
    )

    final_obs = jnp.asarray(transitions.final_observation, dtype=jnp.float32)
    dreamer = GuardedDreamer(
        DreamingConfig(
            warmup_steps=min(args.train_steps, 100),
            max_model_error_ema=args.max_dream_error,
            max_uncertainty=1.0,
        )
    )
    dream = dreamer.propose(
        model,
        result.state,
        final_obs,
        jnp.asarray(transitions.last_action, dtype=jnp.int32),
        uncertainty=jnp.array(0.0, dtype=jnp.float32),
    )

    last_obs_mse = _last(observation_mse, args.final_window)
    last_reward_sq = _last(reward_sq_error, args.final_window)
    last_discount_sq = _last(discount_sq_error, args.final_window)
    last_zero_mse = _last(zero_delta_mse, args.final_window)
    zero_delta_last_rmse = _rmse(last_zero_mse)
    last_observation_rmse = _rmse(last_obs_mse)

    return {
        "method": method,
        "seed": seed,
        "steps": args.train_steps,
        "episodes": transitions.episodes,
        "mean_observation_rmse": _rmse(observation_mse),
        "last_observation_rmse": last_observation_rmse,
        "zero_delta_last_rmse": zero_delta_last_rmse,
        "obs_rmse_ratio_vs_zero_delta": (
            last_observation_rmse / zero_delta_last_rmse
            if zero_delta_last_rmse > 0.0
            else float("nan")
        ),
        "mean_reward_rmse": _rmse(reward_sq_error),
        "last_reward_rmse": _rmse(last_reward_sq),
        "mean_discount_rmse": _rmse(discount_sq_error),
        "last_discount_rmse": _rmse(last_discount_sq),
        "final_model_error_ema": float(result.state.model_error_ema),
        "dream_accepted_after_training": bool(dream.accepted),
        "dream_reject_code": int(dream.reject_code),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed rows."""
    methods = sorted({str(row["method"]) for row in rows})
    summary: dict[str, Any] = {"methods": {}}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "n": len(method_rows),
            "last_observation_rmse": _mean_std(
                [float(row["last_observation_rmse"]) for row in method_rows]
            ),
            "obs_rmse_ratio_vs_zero_delta": _mean_std(
                [float(row["obs_rmse_ratio_vs_zero_delta"]) for row in method_rows]
            ),
            "last_reward_rmse": _mean_std(
                [float(row["last_reward_rmse"]) for row in method_rows]
            ),
            "last_discount_rmse": _mean_std(
                [float(row["last_discount_rmse"]) for row in method_rows]
            ),
            "dream_accept_rate": float(
                np.mean([bool(row["dream_accepted_after_training"]) for row in method_rows])
            ),
        }

    linear_by_seed = {int(row["seed"]): row for row in rows if row["method"] == "linear"}
    mlp_by_seed = {int(row["seed"]): row for row in rows if row["method"] == "mlp"}
    paired_seeds = sorted(set(linear_by_seed) & set(mlp_by_seed))
    if paired_seeds:
        diffs = [
            float(mlp_by_seed[seed]["last_observation_rmse"])
            - float(linear_by_seed[seed]["last_observation_rmse"])
            for seed in paired_seeds
        ]
        summary["mlp_minus_linear_last_observation_rmse"] = {
            **_mean_std(diffs),
            "mlp_better_seeds": int(sum(diff < 0.0 for diff in diffs)),
            "paired_seeds": len(paired_seeds),
        }

    best_method = min(
        methods,
        key=lambda method: summary["methods"][method]["last_observation_rmse"]["mean"],
    )
    summary["best_method_by_last_observation_rmse"] = best_method
    return summary


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: ExperimentConfig,
    total_seconds: float,
) -> None:
    """Write CSV and JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": dataclasses.asdict(config),
        "summary": summary,
        "total_seconds": total_seconds,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--final-window", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--observation-scale",
        type=_parse_observation_scale,
        default=DEFAULT_OBSERVATION_SCALE,
        help="Comma-separated per-channel scale for normalized observation deltas.",
    )
    parser.add_argument(
        "--max-dream-error",
        type=float,
        default=1.0e30,
        help="GuardedDreamer model-error threshold used for final smoke proposal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/action_conditioned_cartpole"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the sweep."""
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if args.train_steps < 1:
        raise ValueError("train-steps must be positive")
    if args.final_window < 1:
        raise ValueError("final-window must be positive")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        transitions = _collect_transitions(seed, args)
        for method in ("linear", "mlp"):
            row = _run_one(method, seed, args, transitions)
            rows.append(row)
            print(
                f"seed={seed} method={method} "
                f"last_obs_rmse={row['last_observation_rmse']:.5f} "
                f"ratio={row['obs_rmse_ratio_vs_zero_delta']:.3f} "
                f"dream={row['dream_accepted_after_training']}"
            )

    summary = summarize(rows)
    config = ExperimentConfig(
        env_id=args.env_id,
        seeds=args.seeds,
        train_steps=args.train_steps,
        final_window=args.final_window,
        gamma=args.gamma,
        observation_scale=args.observation_scale,
        output_dir=str(args.output_dir),
    )
    total_seconds = time.perf_counter() - start
    write_outputs(args.output_dir, rows, summary, config, total_seconds)
    print(
        f"\nBest method: {summary['best_method_by_last_observation_rmse']}; "
        f"wrote {args.output_dir / 'results.csv'} and {args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()

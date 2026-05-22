#!/usr/bin/env python
"""End-to-end Alberta Plan Step 1-4 pipeline demonstration.

Configures :class:`AlbertaPipeline` with the promoted Step 2 UPGD
featurizer and the Step 4 ``HordeActorCriticAgent`` control. Runs on a
synthetic two-task target stream with a real cumulant function and prints
per-window rewards plus per-task supervised accuracy.

Run:
    python "examples/The Alberta Plan/Step1234_End_To_End/integrated_pipeline_demo.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.pipeline import (
    AlbertaPipelineConfig,
    HordeActorCriticPipelineConfig,
    Step2FeatureConfig,
    Step2UPGDConfig,
    make_alberta_pipeline,
)
from alberta_framework.steps import Step3HordeConfig, Step4SARSAConfig


def make_stream(
    key: Array, n_steps: int, observation_dim: int
) -> tuple[Array, Array, Array]:
    """Build a synthetic two-task supervised + reward stream.

    Returns
    -------
    observations
        Shape ``(n_steps + 1, observation_dim)``. Random Gaussian inputs.
    targets
        Shape ``(n_steps + 1, 2)``. Two deterministic functions of the input,
        used as UPGD heads' supervised targets.
    rewards
        Shape ``(n_steps,)``. Reward = first target component shifted by 1
        (so reward correlates with task 0's signal).
    """
    obs = jr.normal(key, (n_steps + 1, observation_dim), dtype=jnp.float32)
    task_0 = jnp.tanh(obs[:, 0] + 0.5 * obs[:, 1])
    task_1 = jnp.sin(obs[:, 2]) * jnp.cos(obs[:, 3])
    targets = jnp.stack([task_0, task_1], axis=1)
    rewards = task_0[1:]
    return obs, targets, rewards


def real_cumulant_fn(observation: Array, reward: Array, _terminated: Array) -> Array:
    """Real cumulant function for Step 3.

    The first demon tracks the reward (so the value head learns the policy
    return). The second demon tracks ``observation[0]`` so that auxiliary
    head 1 reflects the same signal task 0 must learn.
    """
    obs_1d = jnp.atleast_1d(observation)
    return jnp.stack([jnp.asarray(reward, dtype=jnp.float32), obs_1d[0]])


def main() -> None:
    """Run the integrated demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for any future artifacts (currently unused).",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    observation_dim = 4
    n_actions = 3

    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=observation_dim),
        upgd=Step2UPGDConfig(
            observation_dim=observation_dim,
            n_heads=2,
            hidden_sizes=(32,),
            step_size=0.03,
        ),
        horde=Step3HordeConfig(
            gammas=(0.95, 0.0),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=n_actions,
            hidden_sizes=(),
            epsilon_start=0.1,
            epsilon_end=0.01,
            step_size=0.05,
        ),
        horde_ac=HordeActorCriticPipelineConfig(
            n_actions=n_actions,
            actor_step_size=0.02,
            actor_lamda=0.9,
            value_head_index=0,
        ),
        step2="upgd",
        control_mode="horde_ac",
    )
    pipeline = make_alberta_pipeline(config, cumulant_fn=real_cumulant_fn)

    stream_key, init_key = jr.split(jr.key(args.seed))
    observations, targets, rewards = make_stream(
        stream_key, args.n_steps, observation_dim
    )
    state = pipeline.init(init_key, observations[0])

    print(f"feature_dim={pipeline.feature_dim}")
    print(f"n_demons={pipeline.horde.n_demons}, n_actions={n_actions}")
    print(
        f"step2={config.step2!r}, control_mode={config.control_mode!r}, "
        f"n_steps={args.n_steps}, window={args.window}"
    )
    print()

    sq_errors = jnp.zeros((args.n_steps, 2), dtype=jnp.float32)
    rewards_seen = jnp.zeros(args.n_steps, dtype=jnp.float32)

    for t in range(args.n_steps):
        result = pipeline.update(
            state,
            observations[t + 1],
            rewards[t],
            jnp.array(0.0, dtype=jnp.float32),
            upgd_targets=targets[t + 1],
        )
        state = result.state
        # Use the UPGD's own predictions for accuracy tracking.
        upgd = pipeline.upgd
        assert upgd is not None and state.upgd_state is not None
        upgd_predictions = upgd.predict(state.upgd_state, observations[t + 1])
        sq_errors = sq_errors.at[t].set(
            jnp.square(upgd_predictions - targets[t + 1])
        )
        rewards_seen = rewards_seen.at[t].set(rewards[t])

    n_windows = args.n_steps // args.window
    print(f"Per-window mean reward and per-task MSE (window={args.window}):")
    print(
        f"{'window':>8} {'mean_reward':>12} {'task0_mse':>12} {'task1_mse':>12}"
    )
    for w in range(n_windows):
        lo = w * args.window
        hi = (w + 1) * args.window
        mean_reward = float(jnp.mean(rewards_seen[lo:hi]))
        task0_mse = float(jnp.mean(sq_errors[lo:hi, 0]))
        task1_mse = float(jnp.mean(sq_errors[lo:hi, 1]))
        print(
            f"{w:>8d} {mean_reward:>12.4f} {task0_mse:>12.4f} {task1_mse:>12.4f}"
        )

    initial_mse = jnp.mean(sq_errors[: args.window], axis=0)
    final_mse = jnp.mean(sq_errors[-args.window :], axis=0)
    print()
    print(
        f"Task0 MSE: initial={float(initial_mse[0]):.4f} -> "
        f"final={float(final_mse[0]):.4f}"
    )
    print(
        f"Task1 MSE: initial={float(initial_mse[1]):.4f} -> "
        f"final={float(final_mse[1]):.4f}"
    )
    learned = bool(jnp.all(final_mse < initial_mse))
    print(f"Learning detected (final < initial on both tasks): {learned}")


if __name__ == "__main__":
    main()

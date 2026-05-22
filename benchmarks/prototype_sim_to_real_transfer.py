#!/usr/bin/env python3
"""PrototypeAgent sim-to-real transfer surrogate benchmark.

This benchmark trains the full PrototypeAgent on a lightweight simulated
balance plant, then transfers the same live agent state to a target plant with
shifted dynamics, actuator scale, observation scale, and sensor noise.  It is
not a real robot deployment; it is a reproducible sim-to-real-style acceptance
artifact for the local checkout.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

OBS_DIM = 4
N_ACTIONS = 2
SOURCE_STEPS = 600
TARGET_STEPS = 600
N_SEEDS = 3
EVAL_WINDOW = 200


@dataclasses.dataclass(frozen=True)
class BalancePlantConfig:
    """Parameters for the balance transfer surrogate plant."""

    force_scale: float
    damping: float
    gravity_gain: float
    obs_scale: float
    sensor_noise: float
    initial_angle_scale: float = 0.18
    dt: float = 0.08
    failure_angle: float = 0.9


class BalancePlant:
    """Small continuing balance plant with reset-on-failure dynamics."""

    def __init__(self, config: BalancePlantConfig, seed: int) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._angle = 0.0
        self._velocity = 0.0
        self._last_reset = False

    def reset(self) -> np.ndarray:
        self._angle = float(
            self._rng.uniform(
                -self._config.initial_angle_scale,
                self._config.initial_angle_scale,
            )
        )
        self._velocity = float(self._rng.uniform(-0.02, 0.02))
        self._last_reset = True
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        force = -1.0 if action == 0 else 1.0
        cfg = self._config
        self._velocity = (
            cfg.damping * self._velocity
            + cfg.force_scale * force
            - cfg.gravity_gain * np.sin(self._angle)
        )
        self._angle = self._angle + cfg.dt * self._velocity
        failed = abs(self._angle) > cfg.failure_angle
        if failed:
            reward = 0.0
            obs = self.reset()
        else:
            reward = max(0.0, 1.0 - abs(self._angle) / cfg.failure_angle)
            obs = self._observation()
            self._last_reset = False
        return obs, float(reward), failed

    def _observation(self) -> np.ndarray:
        cfg = self._config
        noisy_angle = self._angle * cfg.obs_scale
        noisy_velocity = self._velocity * cfg.obs_scale
        if cfg.sensor_noise > 0.0:
            noisy_angle += float(self._rng.normal(0.0, cfg.sensor_noise))
            noisy_velocity += float(self._rng.normal(0.0, cfg.sensor_noise))
        return np.asarray(
            [
                noisy_angle,
                noisy_velocity,
                np.sin(noisy_angle),
                np.cos(noisy_angle) - 1.0,
            ],
            dtype=np.float32,
        )


SOURCE_PLANT = BalancePlantConfig(
    force_scale=0.115,
    damping=0.92,
    gravity_gain=0.055,
    obs_scale=1.0,
    sensor_noise=0.0,
)
TARGET_PLANT = BalancePlantConfig(
    force_scale=0.082,
    damping=0.86,
    gravity_gain=0.072,
    obs_scale=1.18,
    sensor_noise=0.01,
)


def make_prototype_agent() -> PrototypeAgent:
    """Create the full PrototypeAgent configuration used for transfer."""
    subtask_specs = (
        SubtaskSpec(
            feature_index=0,
            threshold=0.05,
            pseudo_reward_scale=1.0,
            max_option_steps=20,
        ),
        SubtaskSpec(
            feature_index=1,
            threshold=0.04,
            pseudo_reward_scale=0.5,
            max_option_steps=20,
        ),
    )
    stomp_cfg = STOMPConfig(
        subtask_specs=subtask_specs,
        observation_dim=OBS_DIM,
        n_primitive_actions=N_ACTIONS,
        base_step_size=0.06,
        base_avg_reward_step_size=0.02,
        epsilon_base=0.15,
        epsilon_option=0.1,
    )
    oak_cfg = OaKConfig(stomp=stomp_cfg, utility_ema_decay=0.99)
    world_model_cfg = ActionConditionedWorldModelConfig(
        observation_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=(),
        step_size=0.08,
        error_decay=0.98,
    )
    horde_spec = create_horde_spec(
        [
            GVFSpec(
                name="target_reward",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.9,
                lamda=0.0,
            ),
            GVFSpec(
                name="instant_reward",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.0,
                lamda=0.0,
            ),
        ]
    )
    ia_cfg = IAConfig(
        cerebellum=ExoCerebellumConfig(n_demons=OBS_DIM, obs_dim=OBS_DIM, step_size=0.04),
        cortex=oak_cfg,
    )
    cfg = PrototypeAgentConfig(
        oak=oak_cfg,
        world_model=world_model_cfg,
        dreaming=DreamingConfig(warmup_steps=25, max_model_error_ema=10.0),
        buffer_capacity=64,
        n_dreams_per_step=1,
        horde_spec=horde_spec,
        horde_hidden_sizes=(),
        horde_step_size=0.08,
        ia=ia_cfg,
    )
    return PrototypeAgent(cfg)


def _initial_action(state: PrototypeAgentState) -> int:
    return int(
        jnp.minimum(
            state.oak_state.stomp_state.base_last_action,
            jnp.array(N_ACTIONS - 1, dtype=jnp.int32),
        )
    )


def _assert_finite_tree(tree: Any) -> None:
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    for leaf in leaves:
        if not bool(jnp.all(jnp.isfinite(leaf))):
            raise AssertionError("PrototypeAgent state contains NaN or Inf")


def run_seed(seed: int) -> dict[str, Any]:
    """Train on the source plant and continue the same state on the target."""
    agent = make_prototype_agent()
    state = agent.init(jr.key(seed))

    source = BalancePlant(SOURCE_PLANT, seed=10_000 + seed)
    source_obs = jnp.asarray(source.reset(), dtype=jnp.float32)
    state = agent.start(state, source_obs)
    action = _initial_action(state)

    source_rewards = np.zeros(SOURCE_STEPS, dtype=np.float32)
    for step in range(SOURCE_STEPS):
        next_obs_raw, reward, _failed = source.step(action)
        result = agent.update(
            state,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_obs_raw, dtype=jnp.float32),
        )
        state = result.state
        action = int(result.action)
        source_rewards[step] = reward

    target = BalancePlant(TARGET_PLANT, seed=20_000 + seed)
    target_obs = jnp.asarray(target.reset(), dtype=jnp.float32)
    state = agent.start(state, target_obs)
    action = _initial_action(state)

    target_rewards = np.zeros(TARGET_STEPS, dtype=np.float32)
    failures = 0
    for step in range(TARGET_STEPS):
        next_obs_raw, reward, failed = target.step(action)
        result = agent.update(
            state,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_obs_raw, dtype=jnp.float32),
        )
        state = result.state
        action = int(result.action)
        target_rewards[step] = reward
        failures += int(failed)

    _assert_finite_tree(state)
    source_final = float(np.mean(source_rewards[-EVAL_WINDOW:]))
    target_initial = float(np.mean(target_rewards[:EVAL_WINDOW]))
    target_final = float(np.mean(target_rewards[-EVAL_WINDOW:]))
    return {
        "seed": seed,
        "source_final_reward": source_final,
        "target_initial_reward": target_initial,
        "target_final_reward": target_final,
        "target_reward_delta": target_final - target_initial,
        "target_failures": failures,
        "final_step_count": int(state.step_count),
    }


def run() -> dict[str, Any]:
    """Run the seeded transfer benchmark and return the acceptance payload."""
    started = time.time()
    seed_results = [run_seed(seed) for seed in range(N_SEEDS)]
    source_mean = float(np.mean([row["source_final_reward"] for row in seed_results]))
    target_initial_mean = float(
        np.mean([row["target_initial_reward"] for row in seed_results])
    )
    target_final_mean = float(np.mean([row["target_final_reward"] for row in seed_results]))
    target_delta_mean = target_final_mean - target_initial_mean
    finite = all(np.isfinite(row["target_final_reward"]) for row in seed_results)
    transferred_all_steps = all(
        row["final_step_count"] == SOURCE_STEPS + TARGET_STEPS
        for row in seed_results
    )
    target_positive = sum(1 for row in seed_results if row["target_final_reward"] > 0.4)
    accepted = bool(finite and transferred_all_steps and target_positive >= 2)
    return {
        "schema": "alberta.prototype.sim_to_real_transfer.v1",
        "accepted_sim_to_real_transfer": accepted,
        "claim_scope": "sim_to_shifted_target_surrogate_not_real_robot",
        "source_steps": SOURCE_STEPS,
        "target_steps": TARGET_STEPS,
        "n_seeds": N_SEEDS,
        "eval_window": EVAL_WINDOW,
        "source_plant": dataclasses.asdict(SOURCE_PLANT),
        "target_plant": dataclasses.asdict(TARGET_PLANT),
        "evidence": {
            "source_final_mean_reward": source_mean,
            "target_initial_mean_reward": target_initial_mean,
            "target_final_mean_reward": target_final_mean,
            "target_delta_mean_reward": target_delta_mean,
            "target_positive_seed_count": target_positive,
            "all_numeric_state_finite": finite,
            "transferred_state_updated_for_all_steps": transferred_all_steps,
            "per_seed": seed_results,
        },
        "boundary": (
            "This is a local sim-to-real surrogate using shifted simulated "
            "target dynamics; it does not prove deployment on a physical robot."
        ),
        "elapsed_s": time.time() - started,
    }


def main() -> int:
    """Run and persist the benchmark artifact."""
    output_dir = Path("outputs/prototype_sim_to_real_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted_sim_to_real_transfer"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

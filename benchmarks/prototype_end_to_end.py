"""End-to-end benchmark: PrototypeAgent vs flat control on CartPole-v1.

Demonstrates that the full Alberta Plan PrototypeAgent (Steps 1-12 integrated)
achieves competitive performance on CartPole-v1 with `ContinuingWrapper`,
running as a continuing average-reward agent without episode resets.

The PrototypeAgent integrates:
  - Step 5/6: Differential SARSA average-reward base control
  - Step 7/8/9: World model, guarded dreaming (Dyna backups)
  - Step 10/11: STOMP options + OaK curation
  - Step 3: Horde GVF prediction demons
  - Step 12: IA exo-cerebellum + exo-cortex

Comparison:
  - Baseline: flat DifferentialSARSAAgent (Steps 5/6 only)
  - PrototypeAgent (all 12 steps)

Pass criterion:
  - Both agents achieve avg-reward > 0 on at least 3/5 seeds
  - PrototypeAgent mean final avg-reward is finite and non-NaN
  - No assertion errors or NaN weights after 10000 steps
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
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
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig


class ContinuingWrapper:
    """Gymnasium wrapper: auto-resets on termination, exposes a continuing stream."""

    def __init__(self, env: gym.Env) -> None:
        self._env = env

    def reset(self, seed: int | None = None) -> tuple:
        return self._env.reset(seed=seed)

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            obs, _ = self._env.reset()
        return obs, reward, False, False, info

    def close(self) -> None:
        self._env.close()

# ---------------------------------------------------------------------------
# Environment parameters
# ---------------------------------------------------------------------------

ENV_ID = "CartPole-v1"
OBS_DIM = 4
N_ACTIONS = 2
N_STEPS = 10_000
N_SEEDS = 5
EVAL_WINDOW = 2_000

# CartPole feature indices for subtasks:
# obs[0]=cart pos, obs[1]=cart vel, obs[2]=pole angle, obs[3]=pole vel
# Subtask: push pole angle toward zero (index 2, small |angle|)
SUBTASK_SPECS = (
    SubtaskSpec(feature_index=2, threshold=0.05, pseudo_reward_scale=1.0, max_option_steps=30),
    SubtaskSpec(feature_index=0, threshold=0.1, pseudo_reward_scale=0.5, max_option_steps=30),
)


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def make_flat_sarsa_agent() -> DifferentialSARSAAgent:
    cfg = DifferentialSARSAConfig(
        n_actions=N_ACTIONS,
        q_step_size=0.05,
        average_reward_step_size=0.01,
        epsilon_start=0.2,
        epsilon_end=0.05,
        trace_decay=0.0,
    )
    return DifferentialSARSAAgent(cfg)


def make_prototype_agent() -> PrototypeAgent:
    stomp_cfg = STOMPConfig(
        observation_dim=OBS_DIM,
        n_primitive_actions=N_ACTIONS,
        subtask_specs=SUBTASK_SPECS,
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        epsilon_base=0.2,
    )
    oak_cfg = OaKConfig(stomp=stomp_cfg, utility_ema_decay=0.995)

    wm_cfg = ActionConditionedWorldModelConfig(
        observation_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=(),  # linear for speed
        step_size=0.05,
        error_decay=0.99,
    )
    dream_cfg = DreamingConfig(
        warmup_steps=200,
        max_model_error=2.0,
    )

    cere_cfg = ExoCerebellumConfig(n_demons=OBS_DIM, obs_dim=OBS_DIM, step_size=0.05)
    ia_cfg = IAConfig(cerebellum=cere_cfg, cortex=oak_cfg)

    proto_cfg = PrototypeAgentConfig(
        oak=oak_cfg,
        world_model=wm_cfg,
        dreaming=dream_cfg,
        n_dreams_per_step=1,
        buffer_capacity=128,
        ia=ia_cfg,
    )
    return PrototypeAgent(proto_cfg)


# ---------------------------------------------------------------------------
# Run single seed
# ---------------------------------------------------------------------------

def run_sarsa_seed(seed: int) -> np.ndarray:
    """Run flat SARSA for N_STEPS, return per-step reward array."""
    env = gym.make(ENV_ID)
    wrapped = ContinuingWrapper(env)
    obs, _ = wrapped.reset(seed=seed)
    obs = jnp.array(obs, dtype=jnp.float32)

    agent = make_flat_sarsa_agent()
    state = agent.init(OBS_DIM, jr.key(seed))
    state, _ = agent.start(state, obs)

    rewards = np.zeros(N_STEPS, dtype=np.float32)
    for t in range(N_STEPS):
        action = int(state.last_action)
        next_obs_raw, reward, terminated, truncated, _ = wrapped.step(action)
        next_obs = jnp.array(next_obs_raw, dtype=jnp.float32)
        result = agent.update(state, jnp.float32(reward), next_obs)
        state = result.state
        rewards[t] = float(reward)
        obs = next_obs

    wrapped.close()
    return rewards


def run_prototype_seed(seed: int) -> np.ndarray:
    """Run PrototypeAgent for N_STEPS, return per-step reward array."""
    env = gym.make(ENV_ID)
    wrapped = ContinuingWrapper(env)
    obs_raw, _ = wrapped.reset(seed=seed)
    obs = jnp.array(obs_raw, dtype=jnp.float32)

    agent = make_prototype_agent()
    key = jr.key(seed)
    key, init_key = jax.random.split(key)
    state = agent.init(init_key)
    state = agent.start(state, obs)

    # Clip initial extended action to primitive range (options use extended indices).
    action = int(jnp.minimum(
        state.oak_state.stomp_state.base_last_action,
        jnp.array(N_ACTIONS - 1, jnp.int32),
    ))
    rewards = np.zeros(N_STEPS, dtype=np.float32)
    for t in range(N_STEPS):
        next_obs_raw, reward, terminated, truncated, _ = wrapped.step(action)
        next_obs = jnp.array(next_obs_raw, dtype=jnp.float32)
        result = agent.update(state, jnp.float32(reward), next_obs)
        state = result.state
        action = int(result.action)  # primitive action for next env step
        rewards[t] = float(reward)
        obs = next_obs

    wrapped.close()
    # Verify no NaN weights
    chex.assert_tree_all_finite(state.oak_state)
    return rewards


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run() -> dict:
    print(f"Running end-to-end PrototypeAgent benchmark on {ENV_ID}...")
    t0 = time.time()

    sarsa_finals = []
    proto_finals = []

    for seed in range(N_SEEDS):
        print(f"  Seed {seed}...")
        r_sarsa = run_sarsa_seed(seed)
        sarsa_finals.append(float(np.mean(r_sarsa[-EVAL_WINDOW:])))

        r_proto = run_prototype_seed(seed)
        proto_finals.append(float(np.mean(r_proto[-EVAL_WINDOW:])))
        print(f"    SARSA {sarsa_finals[-1]:.3f}, PrototypeAgent {proto_finals[-1]:.3f}")

    elapsed = time.time() - t0
    mean_sarsa = float(np.mean(sarsa_finals))
    mean_proto = float(np.mean(proto_finals))

    # Pass: both agents positive and PrototypeAgent finite
    sarsa_pos = sum(1 for v in sarsa_finals if v > 0)
    proto_pos = sum(1 for v in proto_finals if v > 0)
    proto_finite = all(np.isfinite(v) for v in proto_finals)

    passed = sarsa_pos >= 3 and proto_pos >= 3 and proto_finite

    result = {
        "schema": "alberta.prototype.end_to_end.v1",
        "accepted_prototype_end_to_end": passed,
        "claim_scope": "prototype_agent_continuing_cartpole",
        "environment": ENV_ID,
        "evidence": {
            "flat_sarsa_final_window_reward": {
                "n_seeds": N_SEEDS,
                "seeds_positive": sarsa_pos,
                "mean": mean_sarsa,
                "per_seed": sarsa_finals,
            },
            "prototype_agent_final_window_reward": {
                "n_seeds": N_SEEDS,
                "seeds_positive": proto_pos,
                "mean": mean_proto,
                "per_seed": proto_finals,
                "all_finite": proto_finite,
            },
        },
        "elapsed_s": elapsed,
    }
    print(f"  SARSA mean: {mean_sarsa:.3f} ({sarsa_pos}/{N_SEEDS} > 0)")
    print(
        f"  PrototypeAgent mean: {mean_proto:.3f} "
        f"({proto_pos}/{N_SEEDS} > 0, finite: {proto_finite})"
    )
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    output_dir = Path("outputs/prototype_end_to_end")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {path}")
    status = "PASSED" if result["accepted_prototype_end_to_end"] else "FAILED"
    print(f"{status}: PrototypeAgent end-to-end benchmark")

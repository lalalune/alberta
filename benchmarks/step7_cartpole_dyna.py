"""Step 7 FA planning benchmark on CartPole (continuing).

Compares flat DifferentialSARSA (real-only) against the same algorithm
augmented with a neural ActionConditionedWorldModel and Dyna planning
(step7_update).  Uses a linear world model (`hidden_sizes=()`) on the
4-dimensional CartPole state space to cleanly attribute any improvement
to the Dyna planning loop rather than the model architecture.

Pass criterion: Dyna wins on ≥6/10 seeds (mean final-window reward higher).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.steps import (
    Step6DifferentialSARSAConfig,
    Step7DynaConfig,
    init_step6_state,
    init_step7_state,
    make_step6_differential_sarsa_agent,
    make_step7_components,
    step6_update,
    step7_update,
)
from alberta_framework.steps.step7 import Step8WorldModelConfig

# ─── Constants ────────────────────────────────────────────────────────────────

N_STEPS = 5_000
N_SEEDS = 10
EVAL_WINDOW = 1_000
OBS_DIM = 4
N_ACTIONS = 2

_CONTROL_CFG = Step6DifferentialSARSAConfig(
    n_actions=N_ACTIONS,
    q_step_size=0.03,
    average_reward_step_size=0.005,
    epsilon_start=0.2,
    epsilon_end=0.05,
    trace_decay=0.0,
)

_DYNA_CFG = Step7DynaConfig(
    control=_CONTROL_CFG,
    world_model=Step8WorldModelConfig(
        observation_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=(),          # linear model in obs space
        step_size=0.05,
        sparsity=0.0,
        predict_delta=True,       # predict Δobs for stability on continuous state
        use_layer_norm=False,
    ),
    planning_steps=4,
    planning_warmup_steps=50,
    planning_memory_size=128,
    planning_strategy="random",
)


# ─── Environment wrapper ───────────────────────────────────────────────────────

class ContinuingCartPole:
    def __init__(self) -> None:
        self._env = gym.make("CartPole-v1")

    def reset(self, seed: int) -> np.ndarray:
        obs, _ = self._env.reset(seed=seed)
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float]:
        obs, reward, terminated, truncated, _ = self._env.step(action)
        if terminated or truncated:
            obs, _ = self._env.reset()
        return obs.astype(np.float32), float(reward)

    def close(self) -> None:
        self._env.close()


# ─── Per-seed runners ──────────────────────────────────────────────────────────

def run_real_only(seed: int) -> float:
    env = ContinuingCartPole()
    obs_np = env.reset(seed)
    obs = jnp.array(obs_np)

    agent = make_step6_differential_sarsa_agent(_CONTROL_CFG)
    state = init_step6_state(agent, feature_dim=OBS_DIM, key=jr.key(seed), initial_features=obs)

    rewards = np.zeros(N_STEPS, dtype=np.float32)
    for t in range(N_STEPS):
        action = int(state.last_action)
        next_obs_np, reward = env.step(action)
        next_obs = jnp.array(next_obs_np)
        result = step6_update(agent, state, jnp.float32(reward), next_obs)
        state = result.state
        rewards[t] = reward

    env.close()
    return float(np.mean(rewards[-EVAL_WINDOW:]))


def run_dyna(seed: int) -> float:
    env = ContinuingCartPole()
    obs_np = env.reset(seed)
    obs = jnp.array(obs_np)

    dyna_agent, world_model = make_step7_components(_DYNA_CFG)
    state = init_step7_state(dyna_agent, world_model, key=jr.key(seed), initial_observation=obs)

    rewards = np.zeros(N_STEPS, dtype=np.float32)
    for t in range(N_STEPS):
        action = int(state.control_state.last_action)
        next_obs_np, reward = env.step(action)
        next_obs = jnp.array(next_obs_np)
        result = step7_update(_DYNA_CFG, dyna_agent, world_model, state,
                              jnp.float32(reward), next_obs)
        state = result.state
        rewards[t] = reward

    env.close()
    return float(np.mean(rewards[-EVAL_WINDOW:]))


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    print(f"Step 7 CartPole FA planning benchmark ({N_SEEDS} seeds × {N_STEPS} steps)...")
    t0 = time.time()

    real_finals, dyna_finals = [], []
    for seed in range(N_SEEDS):
        r_real = run_real_only(seed)
        r_dyna = run_dyna(seed)
        real_finals.append(r_real)
        dyna_finals.append(r_dyna)
        print(f"  seed {seed}: real-only={r_real:.3f}  dyna={r_dyna:.3f}  Δ={r_dyna - r_real:+.3f}")

    wins = sum(1 for r, d in zip(real_finals, dyna_finals) if d > r)
    mean_real = float(np.mean(real_finals))
    mean_dyna = float(np.mean(dyna_finals))
    mean_diff = mean_dyna - mean_real
    elapsed = time.time() - t0
    passed = wins >= 6 and mean_dyna > mean_real

    print(f"  Dyna wins: {wins}/{N_SEEDS}, mean diff: {mean_diff:+.3f}, elapsed: {elapsed:.1f}s")
    print(f"  {'PASSED' if passed else 'FAILED'}")

    return {
        "schema": "alberta.step7.cartpole_dyna.v1",
        "passed": passed,
        "claim_scope": "nonlinear_fa_dyna_planning_cartpole",
        "environment": "CartPole-v1 continuing",
        "config": {
            "n_seeds": N_SEEDS,
            "n_steps": N_STEPS,
            "eval_window": EVAL_WINDOW,
            "planning_steps": _DYNA_CFG.planning_steps,
            "world_model_hidden_sizes": list(_DYNA_CFG.world_model.hidden_sizes),
            "predict_delta": _DYNA_CFG.world_model.predict_delta,
        },
        "aggregate": {
            "mean_real_only_final_window_reward": mean_real,
            "mean_dyna_final_window_reward": mean_dyna,
            "mean_diff": mean_diff,
            "dyna_win_count": wins,
            "n_seeds": N_SEEDS,
            "passed": passed,
        },
        "per_seed": {
            "real_only": real_finals,
            "dyna": dyna_finals,
        },
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    output_dir = Path("outputs/step7_cartpole_dyna")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {path}")

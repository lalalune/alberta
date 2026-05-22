"""Step 7 planning benchmark: Dyna vs. real-only on 6-state chain.

Compares DifferentialSARSA (Step 6 baseline) against Step7DynaAgent on the
deterministic 6-state chain from the Step 6 benchmark. Dyna should reach
the optimal average reward faster by replaying model-generated transitions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    init_step7_state,
    make_step7_components,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

N_STATES = 6
N_ACTIONS = 2
FEATURE_DIM = N_STATES


def one_hot(s: int, n: int = N_STATES) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    x[s] = 1.0
    return x


def step_env(state: int, action: int) -> tuple[int, float]:
    """Deterministic 6-state chain."""
    if action == 0:
        next_state = max(0, state - 1)
        reward = 0.005 if state == 0 else 0.0
    else:
        next_state = min(N_STATES - 1, state + 1)
        reward = 1.0 if next_state == N_STATES - 1 else 0.0
    return next_state, reward


def run_real_only(seed: int, steps: int, final_window: int) -> dict:
    config = DifferentialSARSAConfig(
        n_actions=N_ACTIONS,
        q_step_size=0.1,
        average_reward_step_size=0.01,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay_steps=steps // 2,
    )
    agent = DifferentialSARSAAgent(config)
    rng = np.random.default_rng(seed)
    jax_key = jr.key(seed)

    state_env = int(rng.integers(0, N_STATES))
    agent_state = agent.init(FEATURE_DIM, jax_key)
    agent_state, action_int = agent.start(agent_state, jnp.asarray(one_hot(state_env)))

    rewards = []
    for _ in range(steps):
        action = int(action_int)
        next_s, reward = step_env(state_env, action)
        result = agent.update(
            agent_state,
            jnp.array(reward, dtype=jnp.float32),
            jnp.asarray(one_hot(next_s)),
        )
        agent_state = result.state
        action_int = result.action
        rewards.append(float(reward))
        state_env = next_s

    return {
        "seed": seed,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "average_reward_estimate": float(agent_state.average_reward),
        "rewards": rewards,
    }


def run_dyna(seed: int, steps: int, final_window: int, planning_steps: int = 4) -> dict:
    config = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=N_ACTIONS,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.1,
            epsilon_end=0.01,
            epsilon_decay_steps=steps // 2,
        ),
        world_model=Step8WorldModelConfig(
            observation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=(),
            step_size=0.1,
            sparsity=0.0,
            use_layer_norm=False,
        ),
        planning_steps=planning_steps,
        planning_warmup_steps=4,
    )
    agent, model = make_step7_components(config)
    jax_key = jr.key(seed)
    rng = np.random.default_rng(seed)

    state_env = int(rng.integers(0, N_STATES))
    init_obs = jnp.asarray(one_hot(state_env))
    dyna_state = init_step7_state(agent, model, key=jax_key, initial_observation=init_obs)

    rewards = []
    for _ in range(steps):
        action = int(dyna_state.control_state.last_action)
        next_s, reward = step_env(state_env, action)
        result = step7_update(
            config,
            agent,
            model,
            dyna_state,
            jnp.array(reward, dtype=jnp.float32),
            jnp.asarray(one_hot(next_s)),
        )
        dyna_state = result.state
        rewards.append(float(reward))
        state_env = next_s

    return {
        "seed": seed,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "average_reward_estimate": float(dyna_state.control_state.average_reward),
        "rewards": rewards,
    }


def run_chain_planning_benchmark(
    seeds: list[int] | None = None,
    *,
    steps: int = 500,
    final_window: int = 100,
    planning_steps: int = 4,
) -> dict:
    if seeds is None:
        seeds = list(range(10))

    t0 = time.time()
    real_results = [run_real_only(s, steps, final_window) for s in seeds]
    dyna_results = [run_dyna(s, steps, final_window, planning_steps) for s in seeds]
    elapsed = time.time() - t0

    real_finals = [r["final_window_reward"] for r in real_results]
    dyna_finals = [r["final_window_reward"] for r in dyna_results]
    improvements = [d - r for d, r in zip(dyna_finals, real_finals)]

    mean_real = float(np.mean(real_finals))
    mean_dyna = float(np.mean(dyna_finals))
    mean_improvement = float(np.mean(improvements))
    dyna_wins = int(sum(d > r for d, r in zip(dyna_finals, real_finals)))

    # Pass: Dyna wins on at least 7/10 seeds and mean improvement > 0.05
    passed = mean_dyna > mean_real and dyna_wins >= 6 and mean_improvement > 0.0

    return {
        "schema": "alberta.step7.chain_planning.v1",
        "claim_scope": "six_state_chain_dyna_vs_real_only_differential_sarsa",
        "config": {
            "n_seeds": len(seeds),
            "steps": steps,
            "final_window": final_window,
            "planning_steps": planning_steps,
        },
        "elapsed_s": elapsed,
        "aggregate": {
            "mean_real_only_final_window_reward": mean_real,
            "mean_dyna_final_window_reward": mean_dyna,
            "mean_reward_improvement": mean_improvement,
            "stderr_improvement": float(np.std(improvements) / np.sqrt(len(improvements))),
            "dyna_win_count": dyna_wins,
            "n_seeds": len(seeds),
            "passed": passed,
        },
        "per_seed": [
            {
                "seed": seeds[i],
                "real_final_window_reward": real_finals[i],
                "dyna_final_window_reward": dyna_finals[i],
                "improvement": improvements[i],
            }
            for i in range(len(seeds))
        ],
        "passed": passed,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "outputs" / "step7_chain_planning"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running 10-seed Dyna vs. real-only benchmark on 6-state chain (Step 7)...")
    result = run_chain_planning_benchmark(steps=500, final_window=100, planning_steps=4)

    real_only_reward = result["aggregate"]["mean_real_only_final_window_reward"]
    dyna_reward = result["aggregate"]["mean_dyna_final_window_reward"]
    print(f"  Real-only mean final reward: {real_only_reward:.4f}")
    print(f"  Dyna mean final reward:      {dyna_reward:.4f}")
    print(f"  Mean improvement:            {result['aggregate']['mean_reward_improvement']:+.4f}")
    print(f"  Dyna wins: {result['aggregate']['dyna_win_count']}/10")
    print(f"  passed: {result['passed']}")

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {out_path}")

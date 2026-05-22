"""Step 6 benchmark: DifferentialSARSA on stochastic RiverSwim.

RiverSwim (Strehl & Littman 2008) is the canonical stochastic continuing
control benchmark for average-reward RL. The agent must discover that
swimming against the current (right) is optimal despite sparse rewards.

Environment:
- 6 states (0 = leftmost / downstream, 5 = rightmost / upstream)
- Action 0 (left): deterministic move left, reward 0.005 at state 0
- Action 1 (right): stochastic — current pushes agent back
  - State 0: 0.6 → state 1, 0.4 → stay at 0
  - States 1-4: 0.6 → state+1, 0.05 → stay, 0.35 → state-1
  - State 5: 0.6 → stay at 5, 0.4 → state 4
  - Reward 1.0 when reaching/staying at state 5 via right action
- Optimal policy: always swim right (action 1)
- Optimal avg reward: ~0.43 (due to stochastic current)
- Random policy avg reward: ~0.005 (mostly stuck at state 0)

The agent must overcome:
1. Sparse rewards (only at boundaries)
2. Stochastic transitions (current pushes back)
3. Exploration challenge (state 5 is far from uniform start)

Pass criterion:
- Mean right-action rate in final window > 0.8 on ≥7/10 seeds
- Mean final avg-reward estimate > 0.1 on ≥7/10 seeds
  (random baseline: ~0.005; optimal: ~0.43)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)

# ---------------------------------------------------------------------------
# RiverSwim environment parameters
# ---------------------------------------------------------------------------

N_STATES = 6
N_ACTIONS = 2

# Right-action transition probabilities (Strehl & Littman 2008)
# [p_forward, p_stay, p_back] for each state when choosing "right"
_RIGHT_P_FORWARD = np.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
_RIGHT_P_STAY    = np.array([0.40, 0.05, 0.05, 0.05, 0.05, 0.60])  # state 0: 0.4 stay+back=0.4
_RIGHT_P_BACK    = np.array([0.00, 0.35, 0.35, 0.35, 0.35, 0.00])  # (forward+stay+back=1.0)
# State 0: p_forward=0.6 (→1), p_stay=0.4 (→0), p_back=0.0
# States 1-4: p_forward=0.6 (→s+1), p_stay=0.05 (→s), p_back=0.35 (→s-1)
# State 5: p_forward=0.6 (stay, already rightmost), p_stay=0.4... wait, need to recalculate

# Corrected: state 0 forward+stay = 1.0, state 5 forward+stay = 1.0
# For state 0: right → 0.6 to state 1, 0.4 stay at 0
# For states 1-4: right → 0.6 to s+1, 0.05 stay, 0.35 to s-1
# For state 5: right → 0.6 stay at 5, 0.4 to state 4 (pushed back)


def step_env_riverswim(state: int, action: int, rng: np.random.Generator) -> tuple[int, float]:
    """One step of stochastic RiverSwim."""
    if action == 0:  # left: deterministic
        next_state = max(0, state - 1)
        reward = 0.005 if state == 0 else 0.0
    else:  # right: stochastic
        u = rng.random()
        if state == 0:
            next_state = 1 if u < 0.6 else 0
        elif state == N_STATES - 1:
            next_state = N_STATES - 1 if u < 0.6 else N_STATES - 2
        else:
            if u < 0.6:
                next_state = state + 1
            elif u < 0.65:  # 0.6 + 0.05
                next_state = state
            else:
                next_state = state - 1
        reward = 1.0 if next_state == N_STATES - 1 else 0.0
    return next_state, reward


# ---------------------------------------------------------------------------
# JAX-compatible RiverSwim step (for lax.scan)
# ---------------------------------------------------------------------------

def env_step_jax(
    state_idx: jax.Array, action: jax.Array, rng_key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX-compatible stochastic RiverSwim step."""
    rng_key, use_key = jr.split(rng_key)
    u = jr.uniform(use_key)

    # Left: deterministic
    ns_left = jnp.maximum(jnp.array(0, jnp.int32), state_idx - 1)
    r_left = jnp.where(state_idx == 0, jnp.float32(0.005), jnp.float32(0.0))

    # Right: stochastic
    p_forward = jnp.where(
        state_idx == N_STATES - 1, jnp.float32(0.6),
        jnp.float32(0.6)
    )
    p_stay = jnp.where(
        state_idx == 0, jnp.float32(0.4),
        jnp.where(state_idx == N_STATES - 1, jnp.float32(0.4), jnp.float32(0.05))
    )
    # forward = move right (or stay at rightmost), stay = stay, back = move left
    ns_right_forward = jnp.minimum(state_idx + 1, jnp.array(N_STATES - 1, jnp.int32))
    ns_right_back    = jnp.maximum(state_idx - 1, jnp.array(0, jnp.int32))
    ns_right = jnp.where(
        u < p_forward,
        ns_right_forward,
        jnp.where(u < p_forward + p_stay, state_idx, ns_right_back),
    )
    r_right = jnp.where(ns_right == N_STATES - 1, jnp.float32(1.0), jnp.float32(0.0))

    next_state = jnp.where(action == 0, ns_left, ns_right).astype(jnp.int32)
    reward     = jnp.where(action == 0, r_left, r_right)
    return next_state, reward, rng_key


# ---------------------------------------------------------------------------
# Single-seed run via jax.lax.scan
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    *,
    steps: int = 30_000,
    q_step_size: float = 0.05,
    avg_step_size: float = 0.001,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 20_000,
    trace_decay: float = 0.0,
    optimistic_init: float = 1.0,
) -> np.ndarray:
    """Run one seed and return per-step reward array.

    Uses optimistic Q-initialization (right-action Q = optimistic_init for all states)
    to force sufficient exploration of the upstream states against the stochastic current.
    Without optimism, epsilon-greedy fails to discover state 5 against RiverSwim's current.
    """
    config = DifferentialSARSAConfig(
        n_actions=N_ACTIONS,
        q_step_size=q_step_size,
        average_reward_step_size=avg_step_size,
        trace_decay=trace_decay,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
    )
    agent = DifferentialSARSAAgent(config)
    key = jr.key(seed)
    key, init_key, env_key = jr.split(key, 3)
    agent_state = agent.init(N_STATES, init_key)

    # Optimistic initialization: set Q weights for right action to encourage exploration.
    # Biases the agent toward going right until rewards correct the estimates.
    opt_q = agent_state.q_weights.at[1, :].set(
        jnp.full(N_STATES, jnp.float32(optimistic_init / N_STATES))
    )
    agent_state = agent_state.replace(q_weights=opt_q)

    init_obs = jax.nn.one_hot(jnp.array(0, jnp.int32), N_STATES, dtype=jnp.float32)
    agent_state, _ = agent.start(agent_state, init_obs)

    def scan_fn(carry, _):
        state_idx, ag_state, env_rng = carry
        action = ag_state.last_action
        next_state, reward, new_env_rng = env_step_jax(state_idx, action, env_rng)
        next_obs = jax.nn.one_hot(next_state, N_STATES, dtype=jnp.float32)
        result = agent.update(ag_state, reward, next_obs)
        return (next_state, result.state, new_env_rng), (reward, action)

    _, (rewards, actions) = jax.lax.scan(
        scan_fn,
        (jnp.array(0, jnp.int32), agent_state, env_key),
        None,
        length=steps,
    )
    return np.asarray(rewards), np.asarray(actions)


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_seeds: int = 10,
    steps: int = 30_000,
    final_window: int = 5_000,
    min_right_rate: float = 0.8,
    min_avg_reward: float = 0.3,
) -> dict:
    t0 = time.time()
    all_rewards = []
    all_actions = []

    print(f"Running {n_seeds} seeds × {steps} steps on stochastic RiverSwim ...")
    for seed in range(n_seeds):
        t_seed = time.time()
        rewards, actions = run_seed(seed, steps=steps)
        all_rewards.append(rewards)
        all_actions.append(actions)
        final_r = float(np.mean(rewards[-final_window:]))
        right_rate = float(np.mean(actions[-final_window:]))
        print(
            f"  seed {seed}: final_avg_r={final_r:.4f}  "
            f"right_rate={right_rate:.3f}  [{time.time()-t_seed:.1f}s]"
        )

    rewards_arr = np.stack(all_rewards)  # (n_seeds, steps)
    actions_arr = np.stack(all_actions)

    final_rewards  = np.mean(rewards_arr[:, -final_window:], axis=1)
    final_rights   = np.mean(actions_arr[:, -final_window:], axis=1)

    right_wins = int(np.sum(final_rights >= min_right_rate))
    reward_wins = int(np.sum(final_rewards >= min_avg_reward))

    passed = bool(
        right_wins >= 7 and reward_wins >= 7
    )

    result = {
        "schema": "alberta.step6.riverswim_stochastic_benchmark.v1",
        "config": {
            "n_seeds": n_seeds,
            "steps": steps,
            "final_window": final_window,
            "min_right_rate": min_right_rate,
            "min_avg_reward": min_avg_reward,
            "environment": "stochastic_riverswim_6state",
        },
        "mean_final_reward": float(np.mean(final_rewards)),
        "stderr_final_reward": float(np.std(final_rewards) / np.sqrt(n_seeds)),
        "mean_final_right_rate": float(np.mean(final_rights)),
        "stderr_final_right_rate": float(np.std(final_rights) / np.sqrt(n_seeds)),
        "right_wins": right_wins,
        "reward_wins": reward_wins,
        "n_seeds": n_seeds,
        "random_baseline_avg_reward": 0.005,
        "optimal_avg_reward_approx": 0.43,
        "wall_clock_s": time.time() - t0,
        "passed": passed,
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Step 6 stochastic RiverSwim benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--output-dir", type=str, default="outputs/step6_riverswim")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(n_seeds=args.seeds, steps=args.steps)

    print("\n=== Results ===")
    print(
        f"  Mean final avg reward: {result['mean_final_reward']:.4f}"
        f" ± {result['stderr_final_reward']:.4f}"
    )
    print(
        f"  Mean right-action rate: {result['mean_final_right_rate']:.4f}"
        f" ± {result['stderr_final_right_rate']:.4f}"
    )
    print(
        f"  Seeds with right_rate ≥ {result['config']['min_right_rate']}:"
        f" {result['right_wins']}/{result['n_seeds']}"
    )
    print(
        f"  Seeds with avg_reward ≥ {result['config']['min_avg_reward']}:"
        f" {result['reward_wins']}/{result['n_seeds']}"
    )
    print(f"  Random baseline: {result['random_baseline_avg_reward']}")
    print(f"  Approx optimal: {result['optimal_avg_reward_approx']}")
    print(f"  Wall clock: {result['wall_clock_s']:.1f}s")
    print(f"  PASSED: {result['passed']}")

    out_path = out_dir / "riverswim_stochastic_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

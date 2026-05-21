"""Step 10 benchmark: STOMP options vs flat DifferentialSARSA on 6-state chain.

Proves that STOMP options (generate-test-rank-replace subtasks + intra-option
policies + option outcome models) accelerate control on a continuing task
compared to flat differential SARSA.

Environment: deterministic 6-state chain (same as step6_continuing_control.py).
- 6 one-hot states, 2 actions (left/right)
- Reward 1.0 at state 5 (rightmost), reward 0.005 at state 0 (leftmost)
- Optimal avg reward = 1.0 (stay at state 5 by choosing right)

Subtasks (auto-discovered via feature scores proportional to state index):
- Option 0: "reach state 5" (index 5, highest score)
- Option 1: "reach state 4" (index 4, second highest)

Pass criterion:
- STOMP mean cumulative reward > flat SARSA mean cumulative reward (10 seeds)
- STOMP reaches 0.6 avg reward threshold at least 200 steps sooner than SARSA

Uses jax.lax.scan for both agents to avoid Python-JAX sync overhead per step.
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
from alberta_framework.steps.step10 import (
    Step10STOMPConfig,
    init_step10_state,
    make_step10_stomp_agent,
)

# ---------------------------------------------------------------------------
# Shared environment (JAX-compatible for lax.scan)
# ---------------------------------------------------------------------------

N_STATES = 6
N_ACTIONS = 2
OBS_DIM = N_STATES  # one-hot encoding


def env_step_jax(state_idx: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
    """JAX-compatible 6-state chain step (no Python conditionals)."""
    ns_right = jnp.minimum(jnp.asarray(N_STATES - 1, jnp.int32), state_idx + 1)
    ns_left = jnp.maximum(jnp.asarray(0, jnp.int32), state_idx - 1)
    next_state = jnp.where(action == 0, ns_left, ns_right)
    r_right = jnp.where(next_state == N_STATES - 1, jnp.float32(1.0), jnp.float32(0.0))
    r_left = jnp.where(state_idx == 0, jnp.float32(0.005), jnp.float32(0.0))
    reward = jnp.where(action == 0, r_left, r_right)
    return next_state.astype(jnp.int32), reward


# ---------------------------------------------------------------------------
# Flat DifferentialSARSA baseline (Step 6) — scan-based
# ---------------------------------------------------------------------------

def run_sarsa_seed(
    seed: int,
    *,
    steps: int = 10_000,
    q_step_size: float = 0.1,
    avg_step_size: float = 0.01,
    epsilon: float = 0.1,
    trace_decay: float = 0.0,
) -> np.ndarray:
    """Run DifferentialSARSA for `steps` via jax.lax.scan."""
    config = DifferentialSARSAConfig(
        n_actions=N_ACTIONS,
        q_step_size=q_step_size,
        average_reward_step_size=avg_step_size,
        epsilon_start=epsilon,
        trace_decay=trace_decay,
    )
    sarsa_agent = DifferentialSARSAAgent(config)
    agent_state = sarsa_agent.init(OBS_DIM, jr.key(seed))
    init_obs = jnp.zeros(OBS_DIM, dtype=jnp.float32).at[0].set(1.0)
    agent_state, _ = sarsa_agent.start(agent_state, init_obs)

    def scan_fn(carry, _):
        state_idx, ag_state = carry
        action = ag_state.last_action
        next_state, reward = env_step_jax(state_idx, action)
        next_obs = jax.nn.one_hot(next_state, OBS_DIM, dtype=jnp.float32)
        result = sarsa_agent.update(ag_state, reward, next_obs)
        return (next_state, result.state), reward

    (_, _), rewards = jax.lax.scan(
        scan_fn, (jnp.array(0, jnp.int32), agent_state), None, length=steps
    )
    return np.asarray(rewards)


# ---------------------------------------------------------------------------
# STOMP agent (Step 10) — scan-based
# ---------------------------------------------------------------------------

def run_stomp_seed(
    seed: int,
    *,
    steps: int = 10_000,
    q_step_size: float = 0.1,
    avg_step_size: float = 0.01,
    epsilon_base: float = 0.1,
    epsilon_option: float = 0.2,
    option_step_size: float = 0.1,
    option_avg_step_size: float = 0.01,
) -> np.ndarray:
    """Run STOMPAgent with 2 auto-discovered subtasks via jax.lax.scan."""
    from alberta_framework.core.options import subtasks_from_feature_scores

    # Feature scores proportional to state index — rightward states are "more important"
    feature_scores = jnp.array([0.0, 0.1, 0.2, 0.4, 0.7, 1.0], dtype=jnp.float32)
    specs = subtasks_from_feature_scores(
        feature_scores,
        top_k=2,
        min_score=0.3,
        threshold=0.5,
        pseudo_reward_scale=1.0,
        max_option_steps=8,
    )

    cfg = Step10STOMPConfig(
        subtask_specs=tuple(specs),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_ACTIONS,
        base_step_size=q_step_size,
        base_avg_reward_step_size=avg_step_size,
        epsilon_base=epsilon_base,
        option_step_size=option_step_size,
        option_avg_reward_step_size=option_avg_step_size,
        epsilon_option=epsilon_option,
    )
    agent = make_step10_stomp_agent(cfg)
    init_obs = jnp.zeros(OBS_DIM, dtype=jnp.float32).at[0].set(1.0)
    agent_state = init_step10_state(agent, key=jr.key(seed), initial_observation=init_obs)

    # Prime: call update once with dummy reward so primitive_action is fully resolved
    # for the first scan step (option may be selected on first start() call).
    prime_result = agent.update(agent_state, jnp.float32(0.0), init_obs)
    agent_state = prime_result.state

    def scan_fn(carry, _):
        state_idx, ag_state, next_prim = carry
        next_state, reward = env_step_jax(state_idx, next_prim)
        next_obs = jax.nn.one_hot(next_state, OBS_DIM, dtype=jnp.float32)
        result = agent.update(ag_state, reward, next_obs)
        return (next_state, result.state, result.primitive_action), reward

    _, rewards = jax.lax.scan(
        scan_fn,
        (jnp.array(0, jnp.int32), agent_state, prime_result.primitive_action),
        None,
        length=steps,
    )
    return np.asarray(rewards)


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_seeds: int = 10,
    steps: int = 10_000,
    final_window: int = 2_000,
    convergence_threshold: float = 0.6,
) -> dict:
    t0 = time.time()
    sarsa_rewards = []
    stomp_rewards = []

    print(f"Running {n_seeds} seeds × {steps} steps ...")
    for seed in range(n_seeds):
        t_seed = time.time()
        sr = run_sarsa_seed(seed, steps=steps)
        st = run_stomp_seed(seed, steps=steps)
        sarsa_rewards.append(sr)
        stomp_rewards.append(st)
        sarsa_avg = float(np.mean(sr[-final_window:]))
        stomp_avg = float(np.mean(st[-final_window:]))
        print(
            f"  seed {seed}: sarsa final={sarsa_avg:.4f}  stomp final={stomp_avg:.4f}"
            f"  [{time.time() - t_seed:.1f}s]"
        )

    sarsa_arr = np.stack(sarsa_rewards)  # (n_seeds, steps)
    stomp_arr = np.stack(stomp_rewards)

    # Final-window average reward per seed
    sarsa_final = np.mean(sarsa_arr[:, -final_window:], axis=1)
    stomp_final = np.mean(stomp_arr[:, -final_window:], axis=1)

    diff = stomp_final - sarsa_final
    wins = int(np.sum(diff > 0))

    # Running average reward
    window = 200
    sarsa_running = np.array([
        np.convolve(sarsa_arr[i], np.ones(window) / window, mode="valid")
        for i in range(n_seeds)
    ])
    stomp_running = np.array([
        np.convolve(stomp_arr[i], np.ones(window) / window, mode="valid")
        for i in range(n_seeds)
    ])

    def steps_to_threshold(running_avg: np.ndarray, threshold: float) -> float:
        """Mean steps to first exceed threshold (inf if never reached)."""
        first_steps = []
        for s in range(n_seeds):
            indices = np.where(running_avg[s] >= threshold)[0]
            first_steps.append(int(indices[0]) + window if len(indices) > 0 else steps)
        return float(np.mean(first_steps))

    sarsa_steps_to_conv = steps_to_threshold(sarsa_running, convergence_threshold)
    stomp_steps_to_conv = steps_to_threshold(stomp_running, convergence_threshold)
    convergence_speedup = sarsa_steps_to_conv - stomp_steps_to_conv

    passed = bool(
        float(np.mean(diff)) > 0.0 and wins >= 6
        and convergence_speedup > 0
    )

    result = {
        "schema": "alberta.step10.stomp_benchmark.v1",
        "config": {
            "n_seeds": n_seeds,
            "steps": steps,
            "final_window": final_window,
            "convergence_threshold": convergence_threshold,
        },
        "sarsa_mean_final": float(np.mean(sarsa_final)),
        "sarsa_stderr_final": float(np.std(sarsa_final) / np.sqrt(n_seeds)),
        "stomp_mean_final": float(np.mean(stomp_final)),
        "stomp_stderr_final": float(np.std(stomp_final) / np.sqrt(n_seeds)),
        "mean_diff_stomp_minus_sarsa": float(np.mean(diff)),
        "diff_stderr": float(np.std(diff) / np.sqrt(n_seeds)),
        "stomp_wins": wins,
        "n_seeds": n_seeds,
        "sarsa_steps_to_convergence": sarsa_steps_to_conv,
        "stomp_steps_to_convergence": stomp_steps_to_conv,
        "convergence_speedup_steps": convergence_speedup,
        "wall_clock_s": time.time() - t0,
        "passed": passed,
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Step 10 STOMP vs SARSA benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="outputs/step10_stomp")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(n_seeds=args.seeds, steps=args.steps)

    print("\n=== Results ===")
    print(
        f"  SARSA  final avg reward: {result['sarsa_mean_final']:.4f} "
        f"± {result['sarsa_stderr_final']:.4f}"
    )
    print(
        f"  STOMP  final avg reward: {result['stomp_mean_final']:.4f} "
        f"± {result['stomp_stderr_final']:.4f}"
    )
    print(
        f"  Diff (STOMP - SARSA):    "
        f"{result['mean_diff_stomp_minus_sarsa']:.4f} ± {result['diff_stderr']:.4f}"
    )
    print(f"  STOMP wins: {result['stomp_wins']}/{result['n_seeds']}")
    print(
        f"  SARSA steps to conv@{result['config']['convergence_threshold']}: "
        f"{result['sarsa_steps_to_convergence']:.0f}"
    )
    print(
        f"  STOMP steps to conv@{result['config']['convergence_threshold']}: "
        f"{result['stomp_steps_to_convergence']:.0f}"
    )
    print(f"  Convergence speedup:     {result['convergence_speedup_steps']:.0f} steps")
    print(f"  Wall clock: {result['wall_clock_s']:.1f}s")
    print(f"  PASSED: {result['passed']}")

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

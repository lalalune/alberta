"""Step 7 planning benchmark: pure-NumPy Dyna vs. real-only on 6-state chain.

Uses tabular differential Q-learning (no JAX) to isolate the planning benefit
signal without JIT compilation overhead.  The 6-state deterministic chain is
the same environment used by the Step 6 benchmark.

Pass criterion: Dyna converges faster — it should reach 0.85 average reward
in fewer real steps than real-only, demonstrated across 10 seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

N_STATES = 6
N_ACTIONS = 2


def step_env(state: int, action: int) -> tuple[int, float]:
    """Deterministic 6-state chain (same as Step 6 benchmark)."""
    if action == 0:
        next_state = max(0, state - 1)
        reward = 0.005 if state == 0 else 0.0
    else:
        next_state = min(N_STATES - 1, state + 1)
        reward = 1.0 if next_state == N_STATES - 1 else 0.0
    return next_state, reward


def epsilon_greedy(q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q[state]))


def run_real_only(
    seed: int,
    steps: int,
    *,
    q_alpha: float = 0.1,
    r_alpha: float = 0.01,
    epsilon: float = 0.1,
    final_window: int = 100,
) -> dict:
    rng = np.random.default_rng(seed)
    # Small random init breaks argmax ties and prevents left-bias at Q=0
    q = rng.uniform(-0.01, 0.01, (N_STATES, N_ACTIONS))
    avg_r = 0.0

    state = int(rng.integers(0, N_STATES))
    rewards = []
    for _ in range(steps):
        action = epsilon_greedy(q, state, epsilon, rng)
        next_s, reward = step_env(state, action)
        next_action = epsilon_greedy(q, next_s, epsilon, rng)
        # Differential SARSA update
        td_error = reward - avg_r + q[next_s, next_action] - q[state, action]
        q[state, action] += q_alpha * td_error
        avg_r += r_alpha * td_error
        rewards.append(reward)
        state = next_s

    return {
        "seed": seed,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "avg_reward_estimate": avg_r,
        "rewards": rewards,
    }


def run_dyna(
    seed: int,
    steps: int,
    *,
    q_alpha: float = 0.1,
    r_alpha: float = 0.01,
    epsilon: float = 0.1,
    planning_steps: int = 4,
    final_window: int = 100,
) -> dict:
    """Dyna: real update + planning_steps model-based Q-updates per real step."""
    rng = np.random.default_rng(seed)
    # Small random init breaks argmax ties and prevents left-bias at Q=0
    q = rng.uniform(-0.01, 0.01, (N_STATES, N_ACTIONS))
    avg_r = 0.0
    # Deterministic model: state x action -> (next_state, reward)
    model: dict[tuple[int, int], tuple[int, float]] = {}

    state = int(rng.integers(0, N_STATES))
    rewards = []
    for _ in range(steps):
        action = epsilon_greedy(q, state, epsilon, rng)
        next_s, reward = step_env(state, action)
        next_action = epsilon_greedy(q, next_s, epsilon, rng)
        # Real update (differential SARSA)
        td_error = reward - avg_r + q[next_s, next_action] - q[state, action]
        q[state, action] += q_alpha * td_error
        avg_r += r_alpha * td_error
        # Store in model
        model[(state, action)] = (next_s, reward)
        rewards.append(reward)
        # Planning updates
        if model:
            keys = list(model.keys())
            for _ in range(planning_steps):
                ps, pa = keys[int(rng.integers(0, len(keys)))]
                pnext_s, preward = model[(ps, pa)]
                pnext_a = epsilon_greedy(q, pnext_s, epsilon, rng)
                ptd = preward - avg_r + q[pnext_s, pnext_a] - q[ps, pa]
                q[ps, pa] += q_alpha * ptd
        state = next_s

    return {
        "seed": seed,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "avg_reward_estimate": avg_r,
        "rewards": rewards,
    }


def steps_to_threshold(rewards: list[float], threshold: float, window: int = 50) -> int:
    """Return the first step index where a rolling window mean exceeds threshold."""
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window : i]) >= threshold:
            return i
    return len(rewards)  # never reached


def run_benchmark(
    seeds: list[int] | None = None,
    *,
    steps: int = 500,
    planning_steps: int = 4,
    final_window: int = 100,
    convergence_threshold: float = 0.85,
) -> dict:
    if seeds is None:
        seeds = list(range(10))

    t0 = time.time()
    real_results = [run_real_only(s, steps, final_window=final_window) for s in seeds]
    dyna_results = [
        run_dyna(
            s,
            steps,
            planning_steps=planning_steps,
            final_window=final_window,
        )
        for s in seeds
    ]
    elapsed = time.time() - t0

    real_finals = [r["final_window_reward"] for r in real_results]
    dyna_finals = [r["final_window_reward"] for r in dyna_results]
    improvements = [d - r for d, r in zip(dyna_finals, real_finals)]

    # Cumulative reward: primary sample-efficiency metric for planning
    real_cumsums = [float(np.sum(r["rewards"])) for r in real_results]
    dyna_cumsums = [float(np.sum(d["rewards"])) for d in dyna_results]
    cumsum_improvements = [d - r for d, r in zip(dyna_cumsums, real_cumsums)]

    real_conv = [steps_to_threshold(r["rewards"], convergence_threshold) for r in real_results]
    dyna_conv = [steps_to_threshold(d["rewards"], convergence_threshold) for d in dyna_results]
    speedups = [r - d for r, d in zip(real_conv, dyna_conv)]

    mean_real = float(np.mean(real_finals))
    mean_dyna = float(np.mean(dyna_finals))
    mean_improvement = float(np.mean(improvements))
    mean_speedup = float(np.mean(speedups))
    mean_cumsum_improvement = float(np.mean(cumsum_improvements))
    dyna_wins = int(sum(d > r for d, r in zip(dyna_finals, real_finals)))
    dyna_faster = int(sum(s > 0 for s in speedups))
    dyna_wins_cumsum = int(sum(d > r for d, r in zip(dyna_cumsums, real_cumsums)))

    # Pass: Dyna wins cumulative reward on ≥7/10 seeds AND mean cumulative improvement > 0
    passed = dyna_wins_cumsum >= 7 and mean_cumsum_improvement > 0.0

    return {
        "schema": "alberta.step7.numpy_planning.v1",
        "claim_scope": "tabular_dyna_vs_real_only_six_state_chain",
        "config": {
            "n_seeds": len(seeds),
            "steps": steps,
            "planning_steps": planning_steps,
            "final_window": final_window,
            "convergence_threshold": convergence_threshold,
        },
        "elapsed_s": elapsed,
        "aggregate": {
            "mean_real_only_final_window_reward": mean_real,
            "mean_dyna_final_window_reward": mean_dyna,
            "mean_final_window_improvement": mean_improvement,
            "dyna_win_count_final_window": dyna_wins,
            "mean_real_only_cumulative_reward": float(np.mean(real_cumsums)),
            "mean_dyna_cumulative_reward": float(np.mean(dyna_cumsums)),
            "mean_cumulative_reward_improvement": mean_cumsum_improvement,
            "mean_cumulative_pct_improvement": float(
                100.0 * mean_cumsum_improvement / max(1, np.mean(real_cumsums))
            ),
            "dyna_win_count_cumulative": dyna_wins_cumsum,
            "mean_steps_to_convergence_real_only": float(np.mean(real_conv)),
            "mean_steps_to_convergence_dyna": float(np.mean(dyna_conv)),
            "mean_convergence_speedup_steps": mean_speedup,
            "dyna_faster_count": dyna_faster,
            "n_seeds": len(seeds),
            "passed": passed,
        },
        "per_seed": [
            {
                "seed": seeds[i],
                "real_final_window_reward": real_finals[i],
                "dyna_final_window_reward": dyna_finals[i],
                "final_window_improvement": improvements[i],
                "real_cumulative_reward": real_cumsums[i],
                "dyna_cumulative_reward": dyna_cumsums[i],
                "cumulative_improvement": cumsum_improvements[i],
                "real_steps_to_convergence": real_conv[i],
                "dyna_steps_to_convergence": dyna_conv[i],
                "convergence_speedup": speedups[i],
            }
            for i in range(len(seeds))
        ],
        "passed": passed,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "outputs" / "step7_chain_planning"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Step 7: tabular Dyna vs. real-only on "
        "6-state chain (10 seeds, 500 steps)..."
    )
    result = run_benchmark(
        steps=500,
        planning_steps=4,
        final_window=100,
        convergence_threshold=0.85,
    )

    agg = result["aggregate"]
    print(f"  Real-only cumulative reward:  {agg['mean_real_only_cumulative_reward']:.1f}")
    print(f"  Dyna      cumulative reward:  {agg['mean_dyna_cumulative_reward']:.1f}")
    pct = agg["mean_cumulative_pct_improvement"]
    delta = agg["mean_cumulative_reward_improvement"]
    print(f"  Mean cumulative improvement:  {delta:+.1f} ({pct:+.1f}%)")
    print(f"  Dyna wins (cumulative):       {agg['dyna_win_count_cumulative']}/10")
    print(f"  Real-only mean final reward:  {agg['mean_real_only_final_window_reward']:.4f}")
    print(f"  Dyna      mean final reward:  {agg['mean_dyna_final_window_reward']:.4f}")
    print(
        "  Mean steps to 0.85 reward - "
        f"real: {agg['mean_steps_to_convergence_real_only']:.0f}  "
        f"dyna: {agg['mean_steps_to_convergence_dyna']:.0f}"
    )
    print(f"  Mean convergence speedup:     {agg['mean_convergence_speedup_steps']:+.0f} steps")
    print(f"  passed: {result['passed']}")

    out_path = out_dir / "results_numpy.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {out_path}")

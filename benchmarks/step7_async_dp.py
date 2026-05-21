"""Step 7 benchmark: Async DP (prioritized sweeping) vs random Dyna vs real-only.

Asynchronous dynamic programming (Barto, Bradtke & Singh 1995; Sutton 1996)
selectively updates state-action pairs in order of priority — choosing WHICH
states to back up rather than updating all states or choosing randomly.

Key insight: after discovering a high-reward transition, predecessor-aware
prioritized sweeping cascades value backward through predecessors in O(N) steps.
Random Dyna spreads k backups uniformly, wasting compute on uninformative pairs.

Algorithms compared:
1. real-only: DifferentialSARSA, no planning
2. random-Dyna: DifferentialSARSA + random backup from model (Dyna-Q style)
3. async-DP: DifferentialSARSA + predecessor-aware prioritized sweeping (Sutton 1996)

Environment: Deterministic 20-state chain.
- State 0 = leftmost, state 19 = rightmost (reward boundary)
- Action right (1): next_state = min(s+1, 19), reward=1.0 when reaching state 19
- Action left (0): next_state = max(s-1, 0), reward=0.005 at state 0
- Deterministic transitions → world model is exact after first visit to each (s,a)

Why async DP wins here: first visit to state 19 creates TD error ≈1.0 for (18,right).
Prioritized sweeping immediately queues all predecessors: (17,right), (16,right), …
With k=20 planning steps, value cascades from state 18 back to state 0 in ONE
planning phase. Random Dyna needs ~40 random draws (growing buffer) to cover the
same 20 state-action pairs — a clear asymptotic advantage.

Pass criterion:
- async-DP mean final reward >= random-Dyna mean final reward
- async-DP wins > random-Dyna on ≥6/10 seeds
- async-DP mean steps to 0.3 threshold <= random-Dyna mean steps
"""
from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

N_STATES = 20
N_ACTIONS = 2


def step_env_chain(state: int, action: int) -> tuple[int, float]:
    """Deterministic 20-state chain."""
    if action == 0:  # left
        next_state = max(0, state - 1)
        reward = 0.005 if state == 0 else 0.0
    else:  # right
        next_state = min(N_STATES - 1, state + 1)
        reward = 1.0 if next_state == N_STATES - 1 else 0.0
    return next_state, reward


def epsilon_greedy(
    q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q[state]))


# ---------------------------------------------------------------------------
# 1. Real-only baseline
# ---------------------------------------------------------------------------

def run_real_only(
    seed: int,
    steps: int = 4_000,
    *,
    q_alpha: float = 0.1,
    r_alpha: float = 0.01,
    epsilon: float = 0.4,
    final_window: int = 500,
) -> dict:
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    q[:, 1] = 1.0 / N_STATES  # optimistic right-action init
    avg_r = 0.0
    state = 0
    rewards = []
    for _ in range(steps):
        action = epsilon_greedy(q, state, epsilon, rng)
        ns, r = step_env_chain(state, action)
        na = epsilon_greedy(q, ns, epsilon, rng)
        td = r - avg_r + q[ns, na] - q[state, action]
        q[state, action] += q_alpha * td
        avg_r += r_alpha * td
        rewards.append(r)
        state = ns
    return {
        "seed": seed,
        "rewards": rewards,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "avg_reward_estimate": float(avg_r),
    }


# ---------------------------------------------------------------------------
# 2. Random Dyna
# ---------------------------------------------------------------------------

def run_random_dyna(
    seed: int,
    steps: int = 4_000,
    *,
    q_alpha: float = 0.1,
    r_alpha: float = 0.01,
    epsilon: float = 0.4,
    planning_steps: int = 20,
    final_window: int = 500,
) -> dict:
    """Random Dyna: after each real step, back up `planning_steps` random model entries."""
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    q[:, 1] = 1.0 / N_STATES
    avg_r = 0.0
    model_list: list[tuple[int, int, int, float]] = []

    state = 0
    rewards = []
    for _ in range(steps):
        action = epsilon_greedy(q, state, epsilon, rng)
        ns, r = step_env_chain(state, action)
        na = epsilon_greedy(q, ns, epsilon, rng)
        td = r - avg_r + q[ns, na] - q[state, action]
        q[state, action] += q_alpha * td
        avg_r += r_alpha * td
        model_list.append((state, action, ns, r))
        rewards.append(r)

        if model_list:
            for _ in range(planning_steps):
                idx = int(rng.integers(0, len(model_list)))
                ps, pa, pns, pr = model_list[idx]
                pna = epsilon_greedy(q, pns, epsilon, rng)
                ptd = pr - avg_r + q[pns, pna] - q[ps, pa]
                q[ps, pa] += q_alpha * ptd
        state = ns

    return {
        "seed": seed,
        "rewards": rewards,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "avg_reward_estimate": float(avg_r),
    }


# ---------------------------------------------------------------------------
# 3. Async DP (predecessor-aware prioritized sweeping — Sutton & Barto Fig 8.4)
# ---------------------------------------------------------------------------

def run_async_dp(
    seed: int,
    steps: int = 4_000,
    *,
    q_alpha: float = 0.1,
    r_alpha: float = 0.01,
    epsilon: float = 0.4,
    planning_steps: int = 20,
    theta: float = 0.001,
    final_window: int = 500,
) -> dict:
    """Predecessor-aware prioritized sweeping (Sutton & Barto 2nd ed., Figure 8.4).

    After each real step, insert (s,a) into priority queue by |TD error|.
    Pop highest-priority pair, back it up, then re-prioritize all predecessor
    state-action pairs whose values may have changed. This creates a backward
    cascade: one reward discovery propagates values through all predecessors
    in a single planning phase — O(N) efficiency vs O(N^2) for random Dyna.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    q[:, 1] = 1.0 / N_STATES
    avg_r = 0.0
    model: dict[tuple[int, int], tuple[int, float]] = {}
    # predecessors[ns] = set of (s,a) pairs that transition to ns under the model
    predecessors: dict[int, set[tuple[int, int]]] = defaultdict(set)
    priority_queue: list[tuple[float, int, int]] = []
    priorities: dict[tuple[int, int], float] = {}

    state = 0
    rewards = []

    for _ in range(steps):
        action = epsilon_greedy(q, state, epsilon, rng)
        ns, r = step_env_chain(state, action)
        na = epsilon_greedy(q, ns, epsilon, rng)
        td = r - avg_r + q[ns, na] - q[state, action]
        q[state, action] += q_alpha * td
        avg_r += r_alpha * td

        # Update model and predecessor index
        old_entry = model.get((state, action))
        if old_entry is not None and old_entry[0] != ns:
            predecessors[old_entry[0]].discard((state, action))
        model[(state, action)] = (ns, r)
        predecessors[ns].add((state, action))
        rewards.append(r)

        abs_td = abs(td)
        if abs_td > theta:
            priorities[(state, action)] = abs_td
            heapq.heappush(priority_queue, (-abs_td, state, action))

        # Prioritized sweeping loop
        backed_up = 0
        while priority_queue and backed_up < planning_steps:
            neg_p, ps, pa = heapq.heappop(priority_queue)
            current_p = priorities.get((ps, pa), 0.0)
            if abs(-neg_p - current_p) > 1e-8:
                continue  # stale entry
            if (ps, pa) not in model:
                continue

            pns, pr = model[(ps, pa)]
            pna = epsilon_greedy(q, pns, epsilon, rng)
            ptd = pr - avg_r + q[pns, pna] - q[ps, pa]
            q[ps, pa] += q_alpha * ptd
            backed_up += 1
            priorities.pop((ps, pa), None)

            # Re-prioritize predecessors of ps (the key cascade step)
            for ss, sa in predecessors.get(ps, set()):
                if (ss, sa) not in model:
                    continue
                sns, sr = model[(ss, sa)]
                sna = epsilon_greedy(q, sns, epsilon, rng)
                std_val = sr - avg_r + q[sns, sna] - q[ss, sa]
                abs_std = abs(std_val)
                if abs_std > theta:
                    priorities[(ss, sa)] = abs_std
                    heapq.heappush(priority_queue, (-abs_std, ss, sa))

        state = ns

    return {
        "seed": seed,
        "rewards": rewards,
        "final_window_reward": float(np.mean(rewards[-final_window:])),
        "avg_reward_estimate": float(avg_r),
    }


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def steps_to_threshold(rewards: list[float], threshold: float, window: int = 200) -> int:
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window:i]) >= threshold:
            return i
    return len(rewards)


def run_experiment(
    n_seeds: int = 10,
    steps: int = 4_000,
    planning_steps: int = 20,
    final_window: int = 500,
    convergence_threshold: float = 0.3,
) -> dict:
    t0 = time.time()
    real_results = []
    dyna_results = []
    dp_results = []

    print(f"Running {n_seeds} seeds × {steps} steps on deterministic 20-state chain ...")
    for seed in range(n_seeds):
        t_s = time.time()
        ro = run_real_only(seed, steps, final_window=final_window)
        rd = run_random_dyna(seed, steps, planning_steps=planning_steps, final_window=final_window)
        rp = run_async_dp(seed, steps, planning_steps=planning_steps, final_window=final_window)
        real_results.append(ro)
        dyna_results.append(rd)
        dp_results.append(rp)
        print(
            f"  seed {seed}: real={ro['final_window_reward']:.4f}  "
            f"dyna={rd['final_window_reward']:.4f}  "
            f"async_dp={rp['final_window_reward']:.4f}  [{time.time()-t_s:.2f}s]"
        )

    real_finals = np.array([r["final_window_reward"] for r in real_results])
    dyna_finals = np.array([r["final_window_reward"] for r in dyna_results])
    dp_finals   = np.array([r["final_window_reward"] for r in dp_results])

    dp_vs_dyna_diff = dp_finals - dyna_finals
    dp_wins = int(np.sum(dp_vs_dyna_diff > 0))

    real_conv = [steps_to_threshold(r["rewards"], convergence_threshold) for r in real_results]
    dyna_conv = [steps_to_threshold(r["rewards"], convergence_threshold) for r in dyna_results]
    dp_conv   = [steps_to_threshold(r["rewards"], convergence_threshold) for r in dp_results]

    passed = bool(
        float(np.mean(dp_finals)) >= float(np.mean(dyna_finals))
        and dp_wins >= 6
        and float(np.mean(dp_conv)) <= float(np.mean(dyna_conv))
    )

    result = {
        "schema": "alberta.step7.async_dp_benchmark.v2",
        "config": {
            "n_seeds": n_seeds,
            "steps": steps,
            "planning_steps": planning_steps,
            "final_window": final_window,
            "convergence_threshold": convergence_threshold,
            "environment": "deterministic_chain_20state",
        },
        "real_only": {
            "mean_final": float(np.mean(real_finals)),
            "stderr_final": float(np.std(real_finals) / np.sqrt(n_seeds)),
            "mean_steps_to_conv": float(np.mean(real_conv)),
        },
        "random_dyna": {
            "mean_final": float(np.mean(dyna_finals)),
            "stderr_final": float(np.std(dyna_finals) / np.sqrt(n_seeds)),
            "mean_steps_to_conv": float(np.mean(dyna_conv)),
        },
        "async_dp_prioritized": {
            "mean_final": float(np.mean(dp_finals)),
            "stderr_final": float(np.std(dp_finals) / np.sqrt(n_seeds)),
            "mean_steps_to_conv": float(np.mean(dp_conv)),
        },
        "async_dp_vs_dyna_mean_diff": float(np.mean(dp_vs_dyna_diff)),
        "async_dp_wins": dp_wins,
        "n_seeds": n_seeds,
        "wall_clock_s": time.time() - t0,
        "passed": passed,
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Step 7 async DP benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--planning-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/step7_dyna")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        n_seeds=args.seeds,
        steps=args.steps,
        planning_steps=args.planning_steps,
    )

    print("\n=== Results ===")
    print(
        f"  Real-only:    final={result['real_only']['mean_final']:.4f}±"
        f"{result['real_only']['stderr_final']:.4f}  "
        f"steps_to_conv={result['real_only']['mean_steps_to_conv']:.0f}"
    )
    print(
        f"  Random Dyna:  final={result['random_dyna']['mean_final']:.4f}±"
        f"{result['random_dyna']['stderr_final']:.4f}  "
        f"steps_to_conv={result['random_dyna']['mean_steps_to_conv']:.0f}"
    )
    print(
        f"  Async DP:     final={result['async_dp_prioritized']['mean_final']:.4f}±"
        f"{result['async_dp_prioritized']['stderr_final']:.4f}  "
        f"steps_to_conv={result['async_dp_prioritized']['mean_steps_to_conv']:.0f}"
    )
    print(
        f"  Async DP vs Dyna: {result['async_dp_vs_dyna_mean_diff']:+.4f}  "
        f"wins={result['async_dp_wins']}/{result['n_seeds']}"
    )
    print(f"  Wall clock: {result['wall_clock_s']:.1f}s")
    print(f"  PASSED: {result['passed']}")

    out_path = out_dir / "async_dp_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

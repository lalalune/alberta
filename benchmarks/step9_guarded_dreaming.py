"""Step 9 benchmark: Guarded dreaming vs naive Dyna on a switching bandit.

Tests the core claim of Step 9: error-gated dreaming prevents model-bias
corruption when the environment changes. After a reward-action flip at step
1000, the world model still predicts the old reward structure. The guard
detects this via a rising prediction-error EMA and suspends dreaming until
the model converges to the new distribution, enabling faster policy adaptation.

Algorithms compared:
1. real-only: DifferentialSARSA, no planning
2. naive-Dyna: planning enabled with error gate disabled (max_error=1e30)
3. guarded-Dyna: planning enabled with error gate active (max_error=0.05)

Environment: Non-stationary 1-state 2-action bandit.
- Phase 1 (steps 0–999):  action 1 -> reward 1.0, action 0 -> reward 0
- Phase 2 (steps 1000–1999): action 0 -> reward 1.0, action 1 -> reward 0

Why this environment works cleanly:
- A linear world model exactly fits both reward functions after convergence,
  so model_error_ema converges to ~0 during Phase 1.
- Immediately after the Phase 2 flip, EVERY action produces wrong reward
  predictions (model_error spikes to 1.0 per step) -> EMA = 0.1 after just
  1 step -> guard fires.
- Naive Dyna continues imagining (wrong) action_1 -> reward 1.0, reinforcing
  the stale policy and slowing Phase 2 adaptation.
- Guarded Dyna suspends dreaming immediately -> pure real updates -> policy
  switches to action_0 faster -> higher Phase 2 reward.

Pass criterion:
- Guarded Dyna mean Phase 2 reward >= Naive Dyna mean Phase 2 reward
- Guarded Dyna wins >=6/10 seeds in Phase 2 mean reward

Performance note: JAX component objects (agent, model, buffer) are created
ONCE per config and reused across all seeds so that JIT compilations are
paid only once per config (not once per seed). step9_update is JIT-compiled
with static_argnums=(0,1,2,3), so the dream scan compiles once per unique
(config, agent, model, buffer) tuple.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.average_reward import DifferentialSARSAAgent
from alberta_framework.core.world_model import ActionConditionedWorldModel
from alberta_framework.steps.step6 import (
    Step6DifferentialSARSAConfig,
)
from alberta_framework.steps.step9 import (
    RecentObservationBuffer,
    Step9DreamingConfig,
    init_step9_state,
    make_step9_components,
    step9_update,
)

N_STATES = 1   # single-state bandit: observation is always [1.0]
N_ACTIONS = 2
PHASE_SHIFT = 1_000

# Constant observation for the 1-state bandit
_OBS = jax.nn.one_hot(jnp.array(0, jnp.int32), N_STATES, dtype=jnp.float32)


def step_env(action: int, phase: int) -> float:
    """1-state bandit with phase-dependent reward assignment."""
    if phase == 0:
        return 1.0 if action == 1 else 0.0
    else:
        return 1.0 if action == 0 else 0.0


def _make_config(
    planning_budget: int,
    max_model_error: float,
    q_step_size: float = 0.1,
    avg_reward_step_size: float = 0.05,
    epsilon: float = 0.3,
) -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=N_ACTIONS,
            q_step_size=q_step_size,
            average_reward_step_size=avg_reward_step_size,
            epsilon_start=epsilon,
            epsilon_end=epsilon,
            epsilon_decay_steps=1,
        ),
        observation_dim=N_STATES,
        n_actions=N_ACTIONS,
        model_hidden_sizes=(),  # linear: fast compilation, exact fit for bandit
        model_step_size=0.1,
        model_sparsity=0.0,
        model_use_layer_norm=False,
        model_gamma=0.0,
        dreaming_warmup_steps=50,
        dreaming_max_model_error=max_model_error,
        model_error_decay=0.9,  # fast EMA: spike detectable within ~5 steps
        planning_budget=planning_budget,
        buffer_capacity=8,
    )


def run_seed_with_components(
    seed: int,
    config: Step9DreamingConfig,
    agent: DifferentialSARSAAgent,
    model: ActionConditionedWorldModel,
    buffer: RecentObservationBuffer,
    steps: int = 2_000,
) -> dict:
    """Run one seed reusing pre-created component objects (avoids JIT re-compile)."""
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(seed),
        initial_observation=_OBS,
    )

    rewards = []
    model_errors = []

    for step in range(steps):
        phase = 0 if step < PHASE_SHIFT else 1
        action = int(state.control_state.last_action)
        r = step_env(action, phase)
        result = step9_update(config, agent, model, buffer, state, jnp.float32(r), _OBS)
        state = result.state
        rewards.append(float(r))
        model_errors.append(float(state.world_model_state.model_error_ema))

    phase1 = float(np.mean(rewards[:PHASE_SHIFT]))
    phase2 = float(np.mean(rewards[PHASE_SHIFT:]))
    early_adapt_error = float(np.mean(model_errors[PHASE_SHIFT:PHASE_SHIFT + 100]))

    return {
        "seed": seed,
        "phase1_mean": phase1,
        "phase2_mean": phase2,
        "early_adapt_model_error": early_adapt_error,
        "rewards": rewards,
        "model_errors": model_errors,
    }


def run_experiment(
    n_seeds: int = 10,
    steps: int = 2_000,
    planning_budget: int = 4,
) -> dict:
    t0 = time.time()
    real_results = []
    naive_results = []
    guarded_results = []

    print(
        f"Running {n_seeds} seeds x {steps} steps on 1-state switching bandit "
        f"(action-reward flip at step {PHASE_SHIFT}) ..."
    )

    # Create component objects ONCE per planning budget (not per config).
    # naive and guarded share the same planning_budget=4 so they can reuse
    # the same agent/model/buffer -> JAX compiles the dream scan only ONCE.
    real_config = _make_config(planning_budget=0, max_model_error=1e30)
    naive_config = _make_config(planning_budget, max_model_error=1e30)
    guard_config = _make_config(planning_budget, max_model_error=0.05)

    print("  Creating components and warming up JIT (2 budgets: 0 + 4)...")
    t_warmup = time.time()
    real_agent, real_model, real_buffer = make_step9_components(real_config)
    dyna_agent, dyna_model, dyna_buffer = make_step9_components(naive_config)

    for cfg, ag, mo, bu in [
        (real_config, real_agent, real_model, real_buffer),
        (naive_config, dyna_agent, dyna_model, dyna_buffer),
        (guard_config, dyna_agent, dyna_model, dyna_buffer),
    ]:
        ws = init_step9_state(ag, mo, bu, key=jr.key(9999), initial_observation=_OBS)
        step9_update(cfg, ag, mo, bu, ws, jnp.float32(0.0), _OBS)

    print(f"  JIT warmup done [{time.time()-t_warmup:.1f}s].")

    for seed in range(n_seeds):
        t_s = time.time()
        ro = run_seed_with_components(
            seed, real_config, real_agent, real_model, real_buffer, steps
        )
        rn = run_seed_with_components(
            seed, naive_config, dyna_agent, dyna_model, dyna_buffer, steps
        )
        rg = run_seed_with_components(
            seed, guard_config, dyna_agent, dyna_model, dyna_buffer, steps
        )
        real_results.append(ro)
        naive_results.append(rn)
        guarded_results.append(rg)
        print(
            f"  seed {seed}: real_p2={ro['phase2_mean']:.4f}  "
            f"naive_p2={rn['phase2_mean']:.4f}  "
            f"guarded_p2={rg['phase2_mean']:.4f}  "
            f"guarded_err={rg['early_adapt_model_error']:.4f}  [{time.time()-t_s:.1f}s]"
        )

    real_p2   = np.array([r["phase2_mean"] for r in real_results])
    naive_p2  = np.array([r["phase2_mean"] for r in naive_results])
    guard_p2  = np.array([r["phase2_mean"] for r in guarded_results])
    real_p1   = np.array([r["phase1_mean"] for r in real_results])
    naive_p1  = np.array([r["phase1_mean"] for r in naive_results])
    guard_p1  = np.array([r["phase1_mean"] for r in guarded_results])

    guard_vs_naive = guard_p2 - naive_p2
    guard_wins = int(np.sum(guard_vs_naive > 0))

    guard_err = np.array([r["early_adapt_model_error"] for r in guarded_results])
    naive_err = np.array([r["early_adapt_model_error"] for r in naive_results])

    passed = bool(
        float(np.mean(guard_p2)) >= float(np.mean(naive_p2))
        and guard_wins >= 6
    )

    result = {
        "schema": "alberta.step9.guarded_dreaming_benchmark.v1",
        "config": {
            "n_seeds": n_seeds,
            "steps": steps,
            "phase_shift": PHASE_SHIFT,
            "planning_budget": planning_budget,
            "guarded_max_model_error": 0.05,
            "model_error_decay": 0.9,
            "environment": "nonstationary_1state_switching_bandit",
        },
        "real_only": {
            "phase1_mean": float(np.mean(real_p1)),
            "phase2_mean": float(np.mean(real_p2)),
            "stderr_phase2": float(np.std(real_p2) / np.sqrt(n_seeds)),
        },
        "naive_dyna": {
            "phase1_mean": float(np.mean(naive_p1)),
            "phase2_mean": float(np.mean(naive_p2)),
            "stderr_phase2": float(np.std(naive_p2) / np.sqrt(n_seeds)),
            "mean_early_adapt_error": float(np.mean(naive_err)),
        },
        "guarded_dyna": {
            "phase1_mean": float(np.mean(guard_p1)),
            "phase2_mean": float(np.mean(guard_p2)),
            "stderr_phase2": float(np.std(guard_p2) / np.sqrt(n_seeds)),
            "mean_early_adapt_error": float(np.mean(guard_err)),
        },
        "guard_vs_naive_mean_diff": float(np.mean(guard_vs_naive)),
        "guard_wins": guard_wins,
        "n_seeds": n_seeds,
        "wall_clock_s": time.time() - t0,
        "passed": passed,
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Step 9 guarded dreaming benchmark")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--planning-budget", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/step9_dreaming")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        n_seeds=args.seeds,
        steps=args.steps,
        planning_budget=args.planning_budget,
    )

    print("\n=== Results ===")
    print(
        f"  Real-only:    p2={result['real_only']['phase2_mean']:.4f}+-"
        f"{result['real_only']['stderr_phase2']:.4f}"
    )
    print(
        f"  Naive Dyna:   p2={result['naive_dyna']['phase2_mean']:.4f}+-"
        f"{result['naive_dyna']['stderr_phase2']:.4f}  "
        f"err_1000-1100={result['naive_dyna']['mean_early_adapt_error']:.4f}"
    )
    print(
        f"  Guarded Dyna: p2={result['guarded_dyna']['phase2_mean']:.4f}+-"
        f"{result['guarded_dyna']['stderr_phase2']:.4f}  "
        f"err_1000-1100={result['guarded_dyna']['mean_early_adapt_error']:.4f}"
    )
    print(
        f"  Guard vs Naive: {result['guard_vs_naive_mean_diff']:+.4f}  "
        f"wins={result['guard_wins']}/{result['n_seeds']}"
    )
    print(f"  Wall clock: {result['wall_clock_s']:.1f}s")
    print(f"  PASSED: {result['passed']}")

    out_path = out_dir / "guarded_dreaming_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

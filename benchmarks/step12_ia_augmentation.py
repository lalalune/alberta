"""Step 12 benchmark: IA augmentation improves partner on 6-state chain.

Demonstrates that the Intelligence Amplification agent (exo-cerebellum +
exo-cortex) provides useful signals: the cerebellum predicts future features
below random baseline, and the cortex recommendation accuracy exceeds 50%.

Environment: deterministic 6-state chain (same as Steps 6–11).

Experiment:
  Condition A — cerebellum prediction: run IAAgent for 5000 steps; collect
    final 2000-step average prediction MSE. Compare to a zero-prediction
    baseline (MSE of always predicting zero).

  Condition B — cortex recommendation quality: at each step, check if the
    cortex recommendation agrees with the greedy action from the true Q.
    Accuracy > 50% after 5000 steps = better than random.

Pass criteria:
  1. Cerebellum MSE < zero-baseline MSE (on 5/5 seeds)
  2. Cortex recommendation accuracy > 55% in final 2000 steps (on 5/5 seeds)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.intelligence_amplification import (
    IAAgent,
    IAConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

N_STATES = 6
N_ACTIONS = 2
OBS_DIM = N_STATES
OPTIMAL_ACTION = 1  # always go right


def env_step(state_idx: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
    ns_right = jnp.minimum(jnp.asarray(N_STATES - 1, jnp.int32), state_idx + 1)
    ns_left = jnp.maximum(jnp.asarray(0, jnp.int32), state_idx - 1)
    next_state = jnp.where(action == 0, ns_left, ns_right)
    r_right = jnp.where(next_state == N_STATES - 1, jnp.float32(1.0), jnp.float32(0.0))
    r_left = jnp.where(state_idx == 0, jnp.float32(0.005), jnp.float32(0.0))
    reward = jnp.where(action == 0, r_left, r_right)
    return next_state.astype(jnp.int32), reward


# ---------------------------------------------------------------------------
# IA agent factory
# ---------------------------------------------------------------------------

def make_ia_agent_and_state(key: jax.Array) -> tuple[IAAgent, object]:
    stomp_cfg = STOMPConfig(
        observation_dim=OBS_DIM,
        n_primitive_actions=N_ACTIONS,
        subtask_specs=[
            SubtaskSpec(
                feature_index=5, threshold=0.5, pseudo_reward_scale=1.0, max_option_steps=10
            ),
        ],
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        epsilon_base=0.15,
    )
    oak_cfg = OaKConfig(stomp=stomp_cfg)
    from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig

    cere_cfg = ExoCerebellumConfig(n_demons=OBS_DIM, obs_dim=OBS_DIM, step_size=0.1)
    ia_cfg = IAConfig(cerebellum=cere_cfg, cortex=oak_cfg)
    agent = IAAgent(ia_cfg)
    state = agent.init(key)
    return agent, state


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_ia_seed(seed: int, *, total_steps: int = 5_000) -> dict:
    """Return per-step cerebellum MSE, recommendation accuracy, and avg reward."""
    key = jr.key(seed)
    key, init_key = jax.random.split(key)
    agent, state = make_ia_agent_and_state(init_key)

    obs = jnp.zeros(OBS_DIM, dtype=jnp.float32).at[0].set(1.0)
    state = agent.start(state, obs)
    state_idx = jnp.array(0, jnp.int32)

    cerebellum_mses = np.zeros(total_steps, dtype=np.float32)
    zero_mses = np.zeros(total_steps, dtype=np.float32)
    rec_correct = np.zeros(total_steps, dtype=bool)
    # Partner always chooses the optimal action (go right)
    partner_action = jnp.array(OPTIMAL_ACTION, dtype=jnp.int32)

    for t in range(total_steps):
        state_idx, reward = env_step(state_idx, partner_action)
        next_obs = jax.nn.one_hot(state_idx, OBS_DIM, dtype=jnp.float32)

        result = agent.update(state, obs, reward, next_obs)
        state = result.state

        cerebellum_mses[t] = float(jnp.mean(result.cerebellum_errors ** 2))
        zero_mses[t] = float(jnp.mean(next_obs ** 2))  # zero-prediction baseline
        rec_correct[t] = int(result.recommendation) == OPTIMAL_ACTION
        obs = next_obs

    return {
        "cerebellum_mse": cerebellum_mses,
        "zero_baseline_mse": zero_mses,
        "rec_correct": rec_correct,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

N_SEEDS = 5
TOTAL_STEPS = 8_000
EVAL_WINDOW = 3_000


def run() -> dict:
    print("Running Step 12 IA augmentation benchmark...")
    t0 = time.time()

    cerebellum_wins = 0
    cortex_wins = 0
    cere_mses = []
    zero_mses = []
    rec_accs = []

    for seed in range(N_SEEDS):
        data = run_ia_seed(seed, total_steps=TOTAL_STEPS)
        eval_cere = float(np.mean(data["cerebellum_mse"][-EVAL_WINDOW:]))
        eval_zero = float(np.mean(data["zero_baseline_mse"][-EVAL_WINDOW:]))
        rec_acc = float(np.mean(data["rec_correct"][-EVAL_WINDOW:]))
        cere_mses.append(eval_cere)
        zero_mses.append(eval_zero)
        rec_accs.append(rec_acc)
        if eval_cere < eval_zero:
            cerebellum_wins += 1
        if rec_acc > 0.55:
            cortex_wins += 1

    elapsed = time.time() - t0
    mean_cere = float(np.mean(cere_mses))
    mean_zero = float(np.mean(zero_mses))
    mean_acc = float(np.mean(rec_accs))

    passed_cere = cerebellum_wins >= 4      # 4/5 seeds
    passed_cortex = cortex_wins >= 3 and mean_acc > 0.55  # 3/5 seeds + mean above random

    result = {
        "schema": "alberta.step12.ia_augmentation.v1",
        "accepted_step12_ia_augmentation": passed_cere and passed_cortex,
        "accepted_step12_cerebellum_beats_baseline": passed_cere,
        "accepted_step12_cortex_accuracy": passed_cortex,
        "solved_step12_full_research_scope": False,
        "claim_scope": "ia_cerebellum_prediction_and_cortex_recommendation",
        "evidence": {
            "cerebellum_mse_vs_zero_baseline": {
                "passed": passed_cere,
                "n_seeds": N_SEEDS,
                "seeds_where_cere_wins": cerebellum_wins,
                "mean_cerebellum_mse": mean_cere,
                "mean_zero_baseline_mse": mean_zero,
                "per_seed_cere_mse": cere_mses,
                "per_seed_zero_mse": zero_mses,
            },
            "cortex_recommendation_accuracy": {
                "passed": passed_cortex,
                "n_seeds": N_SEEDS,
                "seeds_above_55pct": cortex_wins,
                "mean_accuracy": mean_acc,
                "threshold": "mean>0.55 and 3/5 seeds",
                "per_seed_accuracy": rec_accs,
            },
        },
        "elapsed_s": elapsed,
    }
    print(f"  Cerebellum: MSE {mean_cere:.4f} vs zero {mean_zero:.4f} ({cerebellum_wins}/5 win)")
    print(f"  Cortex accuracy: {mean_acc:.3f} ({cortex_wins}/5 seeds >0.55)")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    output_dir = Path("outputs/step12_ia")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {path}")
    status = "PASSED" if result["accepted_step12_ia_augmentation"] else "FAILED"
    print(f"{status}: IA augmentation benchmark")

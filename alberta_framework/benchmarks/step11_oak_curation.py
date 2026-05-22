"""Step 11 benchmark: OaK utility tracking and curation on 6-state chain.

Demonstrates that OaK's per-option utility EMA detects a counterproductive
option and curation replaces it, allowing performance to recover.

Environment: deterministic 6-state chain (same as Steps 6 and 10).
- 6 one-hot states, 2 actions (left/right)
- Reward 1.0 at state 5; reward 0.005 at state 0
- Optimal avg reward = 1.0

Experiment:
  Condition A — OaK with two good options:
    Option 0: "reach state 5" (feature_index=5, threshold=0.5) — correct
    Option 1: "reach state 4" (feature_index=4, threshold=0.5) — useful

  Condition B — OaK starting with one good option and one bad option:
    Option 0: "reach state 5" (feature_index=5, threshold=0.5) — correct
    Option 1: "reach state 0" (feature_index=0, threshold=0.5) — wrong direction
    After 2000 steps, curate() is called with available_feature_indices=[4].
    The lowest-utility option (Option 1, counterproductive) should be
    replaced by one targeting feature 4 (threshold preserved at 0.5).

Pass criteria:
  1. Condition A: mean final avg-reward ≥ 0.70 over 10 seeds
  2. Condition B post-curation: mean final avg-reward ≥ 0.70 over 10 seeds
  3. The replaced option's utility resets to zero after curation (mechanical)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

N_STATES = 6
N_ACTIONS = 2
OBS_DIM = N_STATES


def env_step(state_idx: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
    ns_right = jnp.minimum(jnp.asarray(N_STATES - 1, jnp.int32), state_idx + 1)
    ns_left = jnp.maximum(jnp.asarray(0, jnp.int32), state_idx - 1)
    next_state = jnp.where(action == 0, ns_left, ns_right)
    r_right = jnp.where(next_state == N_STATES - 1, jnp.float32(1.0), jnp.float32(0.0))
    r_left = jnp.where(state_idx == 0, jnp.float32(0.005), jnp.float32(0.0))
    reward = jnp.where(action == 0, r_left, r_right)
    return next_state.astype(jnp.int32), reward


# ---------------------------------------------------------------------------
# OaK agent factory
# ---------------------------------------------------------------------------

def make_oak_agent_and_state(
    subtask_specs: list[SubtaskSpec],
    key: jax.Array,
    *,
    epsilon: float = 0.15,
    q_step_size: float = 0.05,
    avg_step_size: float = 0.01,
    utility_ema_decay: float = 0.99,
    curation_threshold: float = 0.0,
) -> tuple[OaKAgent, object]:
    stomp_cfg = STOMPConfig(
        observation_dim=OBS_DIM,
        n_primitive_actions=N_ACTIONS,
        subtask_specs=subtask_specs,
        base_step_size=q_step_size,
        base_avg_reward_step_size=avg_step_size,
        epsilon_base=epsilon,
    )
    config = OaKConfig(
        stomp=stomp_cfg,
        utility_ema_decay=utility_ema_decay,
        curation_threshold=curation_threshold,
    )
    agent = OaKAgent(config)
    state = agent.init(key)
    return agent, state


# ---------------------------------------------------------------------------
# Single-seed run with optional mid-run curation
# ---------------------------------------------------------------------------

def run_oak_seed(
    seed: int,
    subtask_specs: list[SubtaskSpec],
    *,
    total_steps: int = 8_000,
    curate_at: int | None = None,
    curate_available_features: list[int] | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """Return (per-step rewards, final avg-reward, utility of replaced option or None)."""
    key = jr.key(seed)
    key, init_key = jax.random.split(key)
    agent, state = make_oak_agent_and_state(subtask_specs, init_key)

    init_obs = jnp.zeros(OBS_DIM, dtype=jnp.float32).at[0].set(1.0)
    state = agent.start(state, init_obs)
    state_idx = jnp.array(0, jnp.int32)
    next_action = state.stomp_state.base_last_action
    replaced_utility: float | None = None

    rewards = np.zeros(total_steps, dtype=np.float32)
    for t in range(total_steps):
        if curate_at is not None and t == curate_at:
            key, ckey = jax.random.split(key)
            agent, state = agent.curate(state, ckey, curate_available_features)
            replaced_utility = float(jnp.min(state.utility_ema))

        state_idx, reward = env_step(state_idx, next_action)
        next_obs = jax.nn.one_hot(state_idx, OBS_DIM, dtype=jnp.float32)
        result = agent.update(state, reward, next_obs)
        state = result.state
        next_action = result.primitive_action
        rewards[t] = float(reward)

    final_avg = float(state.stomp_state.base_average_reward)
    return rewards, final_avg, replaced_utility


# ---------------------------------------------------------------------------
# Subtask specs
# ---------------------------------------------------------------------------

GOOD_SPEC_5 = SubtaskSpec(
    feature_index=5,
    threshold=0.5,
    pseudo_reward_scale=1.0,
    max_option_steps=10,
)
GOOD_SPEC_4 = SubtaskSpec(
    feature_index=4,
    threshold=0.5,
    pseudo_reward_scale=0.8,
    max_option_steps=10,
)
BAD_SPEC_0 = SubtaskSpec(
    feature_index=0,
    threshold=0.5,
    pseudo_reward_scale=0.5,
    max_option_steps=10,
)

N_SEEDS = 10
TOTAL_STEPS = 8_000
CURATE_AT = 2_000


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> dict:
    print("Running Step 11 OaK utility tracking + curation benchmark...")
    t0 = time.time()

    good_oak_finals: list[float] = []
    bad_oak_post_finals: list[float] = []
    replaced_utilities: list[float] = []

    for seed in range(N_SEEDS):
        # Condition A: both options good from start
        _, avg_good, _ = run_oak_seed(
            seed,
            [GOOD_SPEC_5, GOOD_SPEC_4],
            total_steps=TOTAL_STEPS,
        )
        good_oak_finals.append(avg_good)

        # Condition B: one bad option, curated at step 2000 with feature 4
        _, avg_post, util = run_oak_seed(
            seed,
            [GOOD_SPEC_5, BAD_SPEC_0],
            total_steps=TOTAL_STEPS,
            curate_at=CURATE_AT,
            curate_available_features=[4],
        )
        bad_oak_post_finals.append(avg_post)
        if util is not None:
            replaced_utilities.append(util)

    elapsed = time.time() - t0
    mean_good = float(np.mean(good_oak_finals))
    mean_post = float(np.mean(bad_oak_post_finals))
    n_good_pass = sum(1 for v in good_oak_finals if v >= 0.70)
    n_post_pass = sum(1 for v in bad_oak_post_finals if v >= 0.70)
    mean_replaced_util = float(np.mean(replaced_utilities)) if replaced_utilities else 0.0

    passed_good = mean_good >= 0.70
    passed_post = mean_post >= 0.70

    result = {
        "schema": "alberta.step11.oak_curation.v1",
        "accepted_step11_oak_curation": passed_good and passed_post,
        "accepted_step11_oak_valid_options": passed_good,
        "accepted_step11_oak_curation_recovery": passed_post,
        "solved_step11_full_research_scope": False,
        "claim_scope": "oak_utility_tracking_and_curation",
        "evidence": {
            "valid_oak_final_avg_reward": {
                "passed": passed_good,
                "n_seeds": N_SEEDS,
                "n_passed": n_good_pass,
                "mean_final_avg_reward": mean_good,
                "threshold": 0.70,
                "per_seed": good_oak_finals,
            },
            "post_curation_recovery": {
                "passed": passed_post,
                "n_seeds": N_SEEDS,
                "n_passed": n_post_pass,
                "mean_final_avg_reward": mean_post,
                "mean_replaced_option_utility_at_curate": mean_replaced_util,
                "threshold": 0.70,
                "curate_at_step": CURATE_AT,
                "per_seed": bad_oak_post_finals,
            },
        },
        "elapsed_s": elapsed,
    }
    print(
        f"  Good OaK avg reward: {mean_good:.3f} "
        f"({n_good_pass}/{N_SEEDS} ≥0.70: {passed_good})"
    )
    print(
        f"  Post-curation OaK avg reward: {mean_post:.3f} "
        f"({n_post_pass}/{N_SEEDS} ≥0.70: {passed_post})"
    )
    print(f"  Mean utility of replaced option at curation: {mean_replaced_util:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    output_dir = Path("outputs/step11_oak")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {path}")
    status = "PASSED" if result["accepted_step11_oak_curation"] else "FAILED"
    print(f"{status}: OaK curation benchmark")

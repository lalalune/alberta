"""Tests for the Step 8 world-model facade."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step8 import (
    Step8WorldModelConfig,
    init_step8_state,
    make_step8_world_model,
    run_step8_scan,
    run_step8_smoke,
    step8_update,
)


def test_step8_config_roundtrip_and_smoke() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=3,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        predict_delta=True,
    )
    assert Step8WorldModelConfig.from_dict(cfg.to_dict()) == cfg

    smoke = run_step8_smoke(cfg, steps=8, seed=0)
    assert smoke.finite
    assert smoke.reward_predictions_shape == (8,)
    assert smoke.next_observation_predictions_shape == (8, 3)


def test_step8_one_step_and_scan_facade() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = make_step8_world_model(cfg)
    state = init_step8_state(model, key=jr.key(1))

    one = step8_update(
        model,
        state,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0, 0.5], dtype=jnp.float32),
    )
    assert int(one.state.step_count) == 1

    observations = jnp.zeros((4, 2), dtype=jnp.float32)
    actions = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    rewards = actions.astype(jnp.float32)
    next_observations = jnp.stack([rewards, 1.0 - rewards], axis=1)
    result = run_step8_scan(
        model,
        one.state,
        observations,
        actions,
        rewards,
        next_observations,
    )
    chex.assert_shape(result.reward_errors, (4,))
    chex.assert_shape(result.next_observation_errors, (4, 2))
    chex.assert_tree_all_finite(result.reward_predictions)

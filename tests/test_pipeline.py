# mypy: disable-error-code="no-untyped-def"
"""Production Step 1-4 pipeline tests."""

import json

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

import alberta_framework as af
from alberta_framework.pipeline import (
    AlbertaPipeline,
    AlbertaPipelineConfig,
    HordeActorCriticPipelineConfig,
    Step2FeatureConfig,
    Step2UPGDConfig,
    make_alberta_pipeline,
    run_pipeline_smoke,
)
from alberta_framework.steps import (
    Step3HordeConfig,
    Step4SARSAConfig,
    run_step3_smoke,
    run_step4_smoke,
)


def _small_pipeline_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.0, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
    )


def _small_upgd_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        upgd=Step2UPGDConfig(
            observation_dim=3,
            n_heads=1,
            hidden_sizes=(8,),
            step_size=0.03,
        ),
        horde=Step3HordeConfig(
            gammas=(0.0, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
        step2="upgd",
    )


def _small_horde_ac_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.95, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
        horde_ac=HordeActorCriticPipelineConfig(
            n_actions=2,
            actor_step_size=0.02,
            actor_lamda=0.0,
            value_head_index=0,
        ),
        control_mode="horde_ac",
    )


def test_pipeline_config_roundtrip_is_json_serializable() -> None:
    config = _small_pipeline_config()
    payload = config.to_dict()
    encoded = json.dumps(payload)
    roundtrip = AlbertaPipelineConfig.from_dict(json.loads(encoded))

    assert roundtrip == config
    assert roundtrip.feature_dim() == 3
    assert af.AlbertaPipeline is AlbertaPipeline


def test_pipeline_config_roundtrip_with_upgd_and_horde_ac() -> None:
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        upgd=Step2UPGDConfig(observation_dim=3, n_heads=1, hidden_sizes=(8,)),
        horde=Step3HordeConfig(
            gammas=(0.95, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
        ),
        control=Step4SARSAConfig(n_actions=2, hidden_sizes=()),
        horde_ac=HordeActorCriticPipelineConfig(n_actions=2, value_head_index=0),
        step2="upgd",
        control_mode="horde_ac",
    )
    payload = config.to_dict()
    roundtrip = AlbertaPipelineConfig.from_dict(json.loads(json.dumps(payload)))
    assert roundtrip == config
    assert roundtrip.feature_dim() == 8


def test_step3_and_step4_facade_smokes_are_finite() -> None:
    step3_config = Step3HordeConfig(
        gammas=(0.0, 0.5),
        lamdas=(0.0, 0.0),
        hidden_sizes=(),
        step_size=0.05,
    )
    step3_result = run_step3_smoke(
        step3_config,
        steps=8,
        final_window=2,
        raw_feature_dim=3,
        constructed_feature_dim=2,
    )
    assert step3_result.finite
    assert step3_result.per_demon_metrics_shape == (8, 2, 3)
    assert step3_result.handoff.feature_dim == 5

    step4_config = Step4SARSAConfig(
        n_actions=2,
        hidden_sizes=(),
        epsilon_start=0.0,
        epsilon_end=0.0,
        step_size=0.05,
    )
    step4_result = run_step4_smoke(step4_config, steps=8, feature_dim=3)
    assert step4_result.finite
    assert step4_result.q_values_shape == (8, 2)
    assert step4_result.actions_shape == (8,)


def test_pipeline_init_predict_and_one_step_update_are_finite() -> None:
    config = _small_pipeline_config()
    pipeline = make_alberta_pipeline(config)
    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(0), initial_observation)

    horde_predictions, q_values = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(q_values, (2,))
    chex.assert_tree_all_finite((horde_predictions, q_values))

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 1
    chex.assert_shape(result.features, (3,))
    chex.assert_shape(result.horde_predictions, (2,))
    chex.assert_shape(result.q_values, (2,))
    chex.assert_tree_all_finite(
        (
            result.features,
            result.horde_predictions,
            result.horde_td_errors,
            result.q_values,
            result.control_td_error,
        )
    )
    assert 0 <= int(result.action) < config.control.n_actions


def test_pipeline_scan_smoke_is_finite() -> None:
    config = _small_pipeline_config()
    result = run_pipeline_smoke(config, steps=8, seed=3)

    assert result.finite
    assert result.feature_shape == (8, 3)
    assert result.horde_predictions_shape == (8, 2)
    assert result.q_values_shape == (8, 2)
    assert result.actions_shape == (8,)
    assert result.to_dict()["config"] == config.to_dict()


def test_pipeline_with_upgd_step2_smoke() -> None:
    """UPGD-backed Step 2 produces finite features that drive Step 3 and Step 4."""
    config = _small_upgd_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.upgd is not None

    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(7), initial_observation)
    chex.assert_shape(state.last_features, (8,))
    assert state.upgd_state is not None

    horde_predictions, q_values = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(q_values, (2,))
    chex.assert_tree_all_finite((horde_predictions, q_values))

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
        upgd_targets=jnp.asarray([0.5], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_shape(result.features, (8,))
    chex.assert_tree_all_finite(
        (result.features, result.horde_predictions, result.q_values)
    )
    smoke = run_pipeline_smoke(config, steps=4, seed=11)
    assert smoke.finite
    assert smoke.feature_shape == (4, 8)


def test_pipeline_with_horde_ac_control_smoke() -> None:
    """Horde actor-critic control returns sensible policies and updates."""
    config = _small_horde_ac_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.config.control_mode == "horde_ac"

    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(0), initial_observation)

    horde_predictions, policy = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(policy, (2,))
    chex.assert_tree_all_finite((horde_predictions, policy))
    assert float(jnp.abs(jnp.sum(policy) - 1.0)) < 1e-4

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_shape(result.q_values, (2,))
    chex.assert_tree_all_finite(
        (result.features, result.horde_predictions, result.q_values)
    )
    assert 0 <= int(result.action) < config.horde_ac.n_actions

    smoke = run_pipeline_smoke(config, steps=4, seed=2)
    assert smoke.finite
    assert smoke.q_values_shape == (4, 2)


def test_pipeline_behavioral_learns() -> None:
    """A 2000-step run on a fixed-target stream should reduce final-window MSE.

    The temporal-context Step 3 path tracks a single deterministic cumulant
    derived from the first observation channel. Final-window MSE is required
    to be strictly lower than initial-window MSE: a real learning signal.
    """
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.0,),
            lamdas=(0.0,),
            hidden_sizes=(),
            step_size=0.1,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
    )
    pipeline = make_alberta_pipeline(config)

    n_steps = 2000
    key = jr.key(0)
    obs_key, _ = jr.split(key)
    observations = jr.normal(obs_key, (n_steps + 1, 3), dtype=jnp.float32)
    # Cumulant: the first channel of the next observation. The Horde must
    # learn to track this from the previous observation.
    cumulants = observations[1:, :1]
    rewards = jnp.zeros(n_steps, dtype=jnp.float32)
    terminated = jnp.zeros(n_steps, dtype=jnp.float32)

    state = pipeline.init(jr.key(99), observations[0])
    result = pipeline.run_arrays(
        state, observations[1:], rewards, terminated, cumulants
    )
    # Per-step squared error between the (single) demon's TD target and prediction.
    sq_err = jnp.square(
        result.horde_predictions[:, 0] - cumulants[:, 0]
    )
    initial_mse = float(jnp.mean(sq_err[:200]))
    final_mse = float(jnp.mean(sq_err[-200:]))

    assert jnp.isfinite(initial_mse)
    assert jnp.isfinite(final_mse)
    assert final_mse < initial_mse, (
        f"final-window MSE ({final_mse:.4f}) should be lower than "
        f"initial-window MSE ({initial_mse:.4f})"
    )


def test_pipeline_cumulant_fn_overrides_default() -> None:
    """Caller-provided cumulant_fn is used instead of the legacy placeholder."""
    sentinel = jnp.array([0.123, 0.456], dtype=jnp.float32)

    def cumulant_fn(_obs, _reward, _terminated):
        return sentinel

    config = _small_pipeline_config()
    pipeline = make_alberta_pipeline(config, cumulant_fn=cumulant_fn)
    state = pipeline.init(
        jr.key(0), jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    )
    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    # With gamma=0 in our test config, td_target ≈ cumulant. Verify the demon
    # 0 target equals the sentinel.
    chex.assert_trees_all_close(
        result.horde_td_targets[0], sentinel[0], atol=1e-5
    )


# silence the import lint warnings used in the test runner
_ = jax

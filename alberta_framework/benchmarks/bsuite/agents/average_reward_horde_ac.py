"""bsuite adapter for average-reward Horde actor-critic control."""

from __future__ import annotations

import importlib
from typing import Any

import dm_env
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    Autostep,
    AverageRewardHordeActorCriticAgent,
    AverageRewardHordeActorCriticConfig,
)
from alberta_framework.core.history_features import HistoryFeatureExtractor
from alberta_framework.benchmarks.bsuite._bsuite_path import add_bsuite_to_path, bsuite_missing_message
from alberta_framework.benchmarks.bsuite.agents.horde_actor_critic import _FeatureLift

add_bsuite_to_path()
try:
    base: Any = importlib.import_module("bsuite.baselines.base")
except ModuleNotFoundError as exc:
    if exc.name == "bsuite":
        raise ModuleNotFoundError(bsuite_missing_message()) from exc
    raise


class BSuiteAverageRewardHordeActorCriticAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to average-reward Horde AC."""

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: AverageRewardHordeActorCriticAgent,
        seed: int = 0,
        history_extractor: HistoryFeatureExtractor | None = None,
        feature_lift: _FeatureLift | None = None,
    ) -> None:
        self._agent = agent
        self._num_actions: int = action_spec.num_values
        self._raw_dim = int(np.prod(obs_spec.shape))
        self._history_extractor = history_extractor
        self._history_state: Any = None
        if history_extractor is not None:
            base_feature_dim = history_extractor.feature_dim()
            self._history_state = history_extractor.init()
        else:
            base_feature_dim = self._raw_dim
        self._feature_lift = feature_lift or _FeatureLift(base_feature_dim)
        feature_dim = self._feature_lift.feature_dim()
        self._state = agent.init(observation_dim=feature_dim, key=jr.key(seed))
        self._jit_start = jax.jit(agent.start)
        self._jit_update = jax.jit(agent.update)
        self._pending_action: int | None = None
        self._step_count = 0

    @property
    def state(self) -> Any:
        """Current average-reward actor-critic state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of completed updates."""
        return self._step_count

    @property
    def feature_lift_mode(self) -> str:
        """Configured feature lift mode."""
        return self._feature_lift.mode

    def _features(self, observation: Any) -> jnp.ndarray:
        raw_obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        obs = raw_obs
        if self._history_extractor is not None:
            obs, new_history = self._history_extractor.step(
                self._history_state,
                raw_obs,
            )
            self._history_state = new_history
        return self._feature_lift.transform(jnp.asarray(obs))

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Return the cached on-policy action or sample a fresh one."""
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return int(action)
        obs = self._features(timestep.observation)
        self._state, action = self._jit_start(self._state, obs)
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update from one bsuite transition."""
        del timestep, action
        next_obs = self._features(new_timestep.observation)
        reward = jnp.array(new_timestep.reward or 0.0, dtype=jnp.float32)
        result = self._jit_update(self._state, reward, next_obs)
        self._state = result.state
        self._step_count += 1

        if hasattr(new_timestep, "last") and new_timestep.last():
            self._pending_action = None
            if self._history_extractor is not None:
                self._history_state = self._history_extractor.init()
        else:
            self._pending_action = int(result.action)

    def get_value(self, observation: np.ndarray) -> float:
        """Return scalar differential critic value."""
        raw_obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        obs = raw_obs
        if self._history_extractor is not None:
            obs, _ = self._history_extractor.step(self._history_state, raw_obs)
        obs = self._feature_lift.transform(jnp.asarray(obs))
        prediction = self._agent.critic.predict(self._state.critic_state, obs)
        return float(prediction[0])

    def get_policy(self, observation: np.ndarray) -> np.ndarray:
        """Return the current softmax policy."""
        raw_obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        obs = raw_obs
        if self._history_extractor is not None:
            obs, _ = self._history_extractor.step(self._history_state, raw_obs)
        obs = self._feature_lift.transform(jnp.asarray(obs))
        return np.asarray(self._agent.policy(self._state, obs))


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (32,),
    critic_step_size: float = 0.02,
    average_reward_step_size: float = 0.01,
    temperature: float = 0.5,
    epsilon: float = 0.0,
    actor_update_clip: float = 0.1,
    logit_clip: float = 20.0,
    actor_initial_step_size: float = 0.05,
    actor_step_size: float | None = None,
    actor_meta_step_size: float = 0.01,
    actor_tau: float = 10000.0,
    use_history_features: bool = False,
    history_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99),
    history_include_raw: bool = True,
    feature_lift: str = "raw",
    max_feature_dim: int = 4096,
    seed: int = 0,
    **_: Any,
) -> BSuiteAverageRewardHordeActorCriticAgent:
    """Create a bsuite-compatible average-reward Horde actor-critic agent."""
    if actor_step_size is not None:
        actor_initial_step_size = actor_step_size
    history_extractor: HistoryFeatureExtractor | None = None
    if use_history_features:
        raw_dim = int(np.prod(obs_spec.shape))
        history_extractor = HistoryFeatureExtractor(
            raw_dim=raw_dim,
            decay_rates=history_decay_rates,
            include_raw=history_include_raw,
        )
        lift_input_dim = history_extractor.feature_dim()
    else:
        lift_input_dim = int(np.prod(obs_spec.shape))
    feature_lift_transform = _FeatureLift(
        raw_dim=lift_input_dim,
        mode=feature_lift,
        max_feature_dim=max_feature_dim,
    )
    actor_optimizer = Autostep(
        initial_step_size=actor_initial_step_size,
        meta_step_size=actor_meta_step_size,
        tau=actor_tau,
    )
    agent = AverageRewardHordeActorCriticAgent(
        AverageRewardHordeActorCriticConfig(
            n_actions=action_spec.num_values,
            hidden_sizes=hidden_sizes,
            critic_step_size=critic_step_size,
            average_reward_step_size=average_reward_step_size,
            temperature=temperature,
            epsilon=epsilon,
            actor_update_clip=actor_update_clip,
            logit_clip=logit_clip,
        ),
        actor_optimizer=actor_optimizer,
    )
    return BSuiteAverageRewardHordeActorCriticAgent(
        obs_spec,
        action_spec,
        agent,
        seed=seed,
        history_extractor=history_extractor,
        feature_lift=feature_lift_transform,
    )

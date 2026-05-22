"""Horde actor-critic bsuite adapter.

This adapter wraps :class:`alberta_framework.HordeActorCriticAgent` for use as a
bsuite control agent. It is a deliberately distinct entry point from the
existing ``actor_critic`` adapter (which exposes the ``horde_core`` /
``mlp`` / ``linear_core`` switch and defaults to zero auxiliary demons).
This adapter:

- always uses the Step 3 :class:`HordeLearner` critic via
  :class:`HordeActorCriticAgent`,
- defaults to ``hidden_sizes=(32,)`` and 3 auxiliary prediction demons with
  ``gamma in {0.0, 0.5, 0.9}`` predicting the same reward cumulant as the
  value head, providing multi-timescale auxiliary tasks for the shared trunk,
- supports optional :class:`HistoryFeatureExtractor` preprocessing for
  partially observable settings, run on the raw observation before the
  Horde adapter sees it.

The action-selection / update contract follows the existing actor-critic
adapter: ``select_action`` invokes ``HordeActorCriticAgent.start`` to seed the
last_observation/last_action; ``update`` consumes the next transition and the
agent caches the next on-policy action so the following ``select_action``
returns the exact action used for the policy gradient.
"""

from __future__ import annotations

import importlib
from typing import Any

import dm_env
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    LMS,
    Autostep,
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    HordeLearner,
    ObGDBounding,
    create_horde_spec,
)
from alberta_framework.core.history_features import HistoryFeatureExtractor
from alberta_framework.benchmarks.bsuite._bsuite_path import add_bsuite_to_path, bsuite_missing_message

add_bsuite_to_path()
try:
    base: Any = importlib.import_module("bsuite.baselines.base")
except ModuleNotFoundError as exc:
    if exc.name == "bsuite":
        raise ModuleNotFoundError(bsuite_missing_message()) from exc
    raise


DEFAULT_AUX_GAMMAS: tuple[float, ...] = (0.0, 0.5, 0.9)
_FEATURE_LIFT_MODES = {"raw", "quadratic", "pairwise"}


class _FeatureLift:
    """Stateless causal observation lift for the linear Horde-AC actor.

    The core ``HordeActorCriticAgent`` intentionally keeps a linear softmax
    actor. This adapter-level lift tests whether the remaining bsuite gap is
    actor feature capacity, while preserving temporal uniformity: the same
    deterministic map is applied on every step and uses only the current
    observation/history vector.
    """

    def __init__(
        self,
        raw_dim: int,
        mode: str = "raw",
        max_feature_dim: int = 4096,
    ) -> None:
        if raw_dim <= 0:
            raise ValueError("raw_dim must be positive")
        if mode not in _FEATURE_LIFT_MODES:
            raise ValueError(
                f"feature_lift must be one of {sorted(_FEATURE_LIFT_MODES)}, got {mode!r}"
            )
        if max_feature_dim <= 0:
            raise ValueError("max_feature_dim must be positive")
        self.raw_dim = raw_dim
        self.mode = mode
        self.max_feature_dim = max_feature_dim
        dim = self.feature_dim()
        if dim > max_feature_dim:
            raise ValueError(
                f"feature_lift={mode!r} expands {raw_dim} inputs to {dim} features, "
                f"exceeding max_feature_dim={max_feature_dim}"
            )

    def feature_dim(self) -> int:
        """Return output dimensionality."""
        if self.mode == "raw":
            return self.raw_dim
        if self.mode == "quadratic":
            return 2 * self.raw_dim
        # raw plus upper-triangular pairwise products, including squares.
        return self.raw_dim + self.raw_dim * (self.raw_dim + 1) // 2

    def transform(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Return the lifted feature vector."""
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((self.raw_dim,))
        if self.mode == "raw":
            return obs
        if self.mode == "quadratic":
            return jnp.concatenate((obs, obs * obs))
        outer = obs[:, None] * obs[None, :]
        tri_i, tri_j = jnp.triu_indices(self.raw_dim)
        return jnp.concatenate((obs, outer[tri_i, tri_j]))


def _make_horde_actor_critic_agent(
    *,
    n_actions: int,
    hidden_sizes: tuple[int, ...],
    optimizer: Any,
    bounder: Any,
    normalizer: Any,
    discount: float,
    temperature: float,
    actor_step_size: float,
    actor_lamda: float,
    critic_lamda: float,
    aux_gammas: tuple[float, ...],
    sparsity: float,
    actor_bounder: Any,
    actor_td_error_clip: float | None,
) -> HordeActorCriticAgent:
    """Build a HordeActorCriticAgent with multi-timescale auxiliary demons.

    The first head is the scalar value critic. Auxiliary heads share the same
    reward cumulant as the value head -- they are predictions of the same
    reward stream at different discount factors, providing multi-horizon
    structure to the shared trunk without changing the actor's TD signal.
    """
    demons = [
        GVFSpec(
            name="value",
            demon_type=DemonType.PREDICTION,
            gamma=discount,
            lamda=critic_lamda,
            cumulant_index=0,
        )
    ]
    demons.extend(
        GVFSpec(
            name=f"aux_gamma_{gamma}",
            demon_type=DemonType.PREDICTION,
            gamma=float(gamma),
            lamda=0.0,
            cumulant_index=0,
        )
        for gamma in aux_gammas
    )
    critic = HordeLearner(
        horde_spec=create_horde_spec(demons),
        hidden_sizes=hidden_sizes,
        optimizer=optimizer,
        bounder=bounder,
        normalizer=normalizer,
        sparsity=sparsity,
    )
    return HordeActorCriticAgent(
        config=HordeActorCriticConfig(
            n_actions=n_actions,
            actor_step_size=actor_step_size,
            actor_lamda=actor_lamda,
            temperature=temperature,
            value_head_index=0,
            actor_td_error_clip=actor_td_error_clip,
        ),
        critic=critic,
        actor_bounder=actor_bounder,
    )


class BSuiteHordeActorCriticAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to ``HordeActorCriticAgent``.

    Optionally augments the raw observation with multi-timescale EMA history
    features via :class:`HistoryFeatureExtractor`. The agent receives the
    augmented observation; bsuite's environment never sees the augmentation.
    """

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: HordeActorCriticAgent,
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
        self._state = agent.init(feature_dim=feature_dim, key=jr.key(seed))
        self._jit_start = jax.jit(agent.start)
        self._jit_update = jax.jit(agent.update)
        self._horde_value_discount = float(
            agent.critic.horde_spec.gammas[agent.config.value_head_index]
        )
        self._n_aux = agent.critic.n_demons - 1
        self._pending_action: int | None = None
        self._step_count = 0

    @property
    def state(self) -> Any:
        """Current Horde actor-critic state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of completed updates."""
        return self._step_count

    @property
    def n_aux(self) -> int:
        """Number of auxiliary prediction demons attached to the critic."""
        return self._n_aux

    @property
    def feature_lift_mode(self) -> str:
        """Configured feature lift mode."""
        return self._feature_lift.mode

    def _augment(self, raw_obs: jnp.ndarray) -> jnp.ndarray:
        """Apply the history-feature extractor when configured."""
        obs = raw_obs
        if self._history_extractor is None:
            return self._feature_lift.transform(obs)
        obs, new_history = self._history_extractor.step(self._history_state, raw_obs)
        self._history_state = new_history
        return self._feature_lift.transform(jnp.asarray(obs))

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Return the cached on-policy action or sample a fresh one."""
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return int(action)

        raw_obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        obs = self._augment(raw_obs)
        self._state, action, _policy = self._jit_start(self._state, obs)
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update from one bsuite transition."""
        del timestep, action
        raw_next = jnp.asarray(
            new_timestep.observation, dtype=jnp.float32
        ).flatten()
        next_obs = self._augment(raw_next)
        reward = jnp.array(new_timestep.reward or 0.0, dtype=jnp.float32)
        env_discount = (
            1.0 if new_timestep.discount is None else float(new_timestep.discount)
        )
        transition_discount = jnp.array(
            0.0 if env_discount == 0.0 else self._horde_value_discount,
            dtype=jnp.float32,
        )
        # Auxiliary demons predict the same reward cumulant as the value head;
        # the adapter passes ``None`` so HordeActorCriticAgent.update fills in
        # zeros; we override below to broadcast the reward across the aux
        # heads.
        if self._n_aux > 0:
            aux_cumulants = jnp.full((self._n_aux,), reward, dtype=jnp.float32)
        else:
            aux_cumulants = None
        result = self._jit_update(
            self._state,
            reward,
            next_obs,
            aux_cumulants,
            transition_discount,
        )
        self._state = result.state
        self._step_count += 1

        if hasattr(new_timestep, "last") and new_timestep.last():
            self._pending_action = None
            if self._history_extractor is not None:
                # Reset the history trace at episode boundaries.
                self._history_state = self._history_extractor.init()
        else:
            self._pending_action = int(result.action)

    def get_value(self, observation: np.ndarray) -> float:
        """Return scalar critic value at the current head."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        if self._history_extractor is not None:
            obs, _ = self._history_extractor.step(self._history_state, obs)
        obs = self._feature_lift.transform(jnp.asarray(obs))
        return float(self._agent.value(self._state, obs))

    def get_policy(self, observation: np.ndarray) -> np.ndarray:
        """Return the current softmax policy for one observation."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        if self._history_extractor is not None:
            obs, _ = self._history_extractor.step(self._history_state, obs)
        obs = self._feature_lift.transform(jnp.asarray(obs))
        return np.asarray(self._agent.policy(self._state, obs))


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (32,),
    optimizer_name: str = "autostep",
    step_size: float = 0.03,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
    tau: float = 10000.0,
    kappa: float = 2.0,
    actor_kappa: float | None = None,
    normalizer_decay: float = 0.99,
    discount: float = 0.99,
    temperature: float = 1.0,
    actor_step_size: float = 0.03,
    actor_lamda: float = 0.9,
    critic_lamda: float = 0.0,
    aux_gammas: tuple[float, ...] = DEFAULT_AUX_GAMMAS,
    sparsity: float = 0.0,
    actor_td_error_clip: float | None = None,
    use_history_features: bool = False,
    history_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99),
    history_include_raw: bool = True,
    feature_lift: str = "raw",
    max_feature_dim: int = 4096,
    seed: int = 0,
    **_: Any,
) -> BSuiteHordeActorCriticAgent:
    """Create a bsuite-compatible Horde actor-critic agent.

    Parameters mirror the existing ``actor_critic`` adapter where possible.
    Notable additions:

    - ``aux_gammas``: discount factors for auxiliary prediction demons. Each
      auxiliary demon predicts the same reward cumulant as the value head at a
      different timescale.
    - ``use_history_features``: when True, observations are first passed
      through :class:`HistoryFeatureExtractor` before being given to the Horde
      adapter. The history state resets at episode boundaries.
    - ``feature_lift``: deterministic causal lift applied after optional
      history features. ``pairwise`` tests the known structural blocker where
      the core softmax actor is linear while Q/SARSA use nonlinear MLP heads.
    """
    optimizer: Any
    if optimizer_name == "autostep":
        optimizer = Autostep(
            initial_step_size=initial_step_size,
            meta_step_size=meta_step_size,
            tau=tau,
        )
    elif optimizer_name == "lms":
        optimizer = LMS(step_size=step_size)
    else:
        raise ValueError("optimizer_name must be 'autostep' or 'lms'")

    bounder = ObGDBounding(kappa=kappa)
    actor_bounder = bounder if actor_kappa is None else ObGDBounding(kappa=actor_kappa)
    normalizer = EMANormalizer(decay=normalizer_decay)

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

    agent = _make_horde_actor_critic_agent(
        n_actions=action_spec.num_values,
        hidden_sizes=hidden_sizes,
        optimizer=optimizer,
        bounder=bounder,
        normalizer=normalizer,
        discount=discount,
        temperature=temperature,
        actor_step_size=actor_step_size,
        actor_lamda=actor_lamda,
        critic_lamda=critic_lamda,
        aux_gammas=aux_gammas,
        sparsity=sparsity,
        actor_bounder=actor_bounder,
        actor_td_error_clip=actor_td_error_clip,
    )
    return BSuiteHordeActorCriticAgent(
        obs_spec,
        action_spec,
        agent,
        seed=seed,
        history_extractor=history_extractor,
        feature_lift=feature_lift_transform,
    )

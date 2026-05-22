"""Actor-critic bsuite agent adapter for Step 4 control benchmarks.

This module exposes three implementation paths. The ``horde_core`` path uses
the Step 3 ``HordeLearner`` critic through ``HordeActorCriticAgent``. The ``mlp``
path uses existing
Alberta learners: a multi-head preference learner for the actor and an MLP
value learner for the critic. The ``linear_core`` path uses the core
``ActorCriticAgent``, which is intentionally linear and does not consume
``hidden_sizes`` or Autostep meta-parameters.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import chex
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
    MLPLearner,
    MultiHeadMLPLearner,
    ObGDBounding,
    create_horde_spec,
)
from benchmarks.bsuite._bsuite_path import add_bsuite_to_path, bsuite_missing_message

add_bsuite_to_path()
try:
    base: Any = importlib.import_module("bsuite.baselines.base")
except ModuleNotFoundError as exc:
    if exc.name == "bsuite":
        raise ModuleNotFoundError(bsuite_missing_message()) from exc
    raise


@chex.dataclass(frozen=True)
class _FallbackActorCriticState:
    """State for the local actor-critic fallback."""

    actor_state: Any
    critic_state: Any
    rng_key: chex.Array
    step_count: chex.Array


@chex.dataclass(frozen=True)
class _FallbackActorCriticUpdate:
    """Result returned by the fallback update."""

    state: _FallbackActorCriticState
    td_error: chex.Array
    value: chex.Array
    next_value: chex.Array


class _FallbackActorCriticAgent:
    """Minimal actor-critic implementation using framework learners.

    The actor learns action preferences. On each transition, the selected
    action's preference is nudged toward ``current_preference + actor_step_size
    * td_error``; the critic learns the one-step TD target. This keeps the
    benchmark adapter operational while preserving the replacement boundary for
    a future core ``ActorCriticAgent``.
    """

    def __init__(
        self,
        n_actions: int,
        hidden_sizes: tuple[int, ...],
        optimizer: Any,
        bounder: Any,
        normalizer: Any,
        discount: float,
        temperature: float,
        actor_step_size: float,
        sparsity: float,
    ) -> None:
        self.n_actions = n_actions
        self.discount = discount
        self.temperature = temperature
        self.actor_step_size = actor_step_size
        self.actor = MultiHeadMLPLearner(
            n_heads=n_actions,
            hidden_sizes=hidden_sizes,
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            sparsity=sparsity,
        )
        self.critic = MLPLearner(
            hidden_sizes=hidden_sizes,
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            sparsity=sparsity,
        )

    def init(self, feature_dim: int, key: chex.Array) -> _FallbackActorCriticState:
        """Initialize actor and critic states."""
        actor_key, critic_key, rng_key = jr.split(key, 3)
        return _FallbackActorCriticState(
            actor_state=self.actor.init(feature_dim, actor_key),
            critic_state=self.critic.init(feature_dim, critic_key),
            rng_key=rng_key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def preferences(
        self,
        state: _FallbackActorCriticState,
        observation: chex.Array,
    ) -> chex.Array:
        """Return current action preferences."""
        return self.actor.predict(state.actor_state, observation)

    def value(self, state: _FallbackActorCriticState, observation: chex.Array) -> chex.Array:
        """Return scalar state value estimate."""
        return jnp.squeeze(self.critic.predict(state.critic_state, observation))

    def select_action(
        self,
        state: _FallbackActorCriticState,
        observation: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """Sample an action from the actor's softmax policy."""
        prefs = self.preferences(state, observation)
        logits = prefs / jnp.maximum(jnp.asarray(self.temperature), 1e-6)
        key, subkey = jr.split(state.rng_key)
        action = jr.categorical(subkey, logits).astype(jnp.int32)
        return action, key

    def update(
        self,
        state: _FallbackActorCriticState,
        observation: chex.Array,
        action: chex.Array,
        reward: chex.Array,
        next_observation: chex.Array,
        discount: chex.Array,
    ) -> _FallbackActorCriticUpdate:
        """Apply one actor-critic update."""
        value = self.value(state, observation)
        next_value = self.value(state, next_observation)
        td_target = reward + discount * self.discount * next_value
        td_error = td_target - value

        critic_result = self.critic.update(
            state.critic_state,
            observation,
            jnp.atleast_1d(td_target),
        )

        prefs = self.preferences(state, observation)
        actor_targets = jnp.full(self.n_actions, jnp.nan)
        actor_target = prefs[action] + self.actor_step_size * td_error
        actor_targets = actor_targets.at[action].set(actor_target)
        actor_result = self.actor.update(state.actor_state, observation, actor_targets)

        new_state = state.replace(
            actor_state=actor_result.state,
            critic_state=critic_result.state,
            step_count=state.step_count + 1,
        )
        return _FallbackActorCriticUpdate(
            state=new_state,
            td_error=td_error,
            value=value,
            next_value=next_value,
        )


def _load_core_actor_critic() -> type[Any] | None:
    """Return core ActorCriticAgent if it exists in the installed package."""
    try:
        module = importlib.import_module("alberta_framework")
        return cast(type[Any], getattr(module, "ActorCriticAgent"))
    except (AttributeError, ModuleNotFoundError):
        return None


def _load_core_actor_critic_config() -> type[Any] | None:
    """Return core ActorCriticConfig if it exists in the installed package."""
    try:
        module = importlib.import_module("alberta_framework")
        return cast(type[Any], getattr(module, "ActorCriticConfig"))
    except (AttributeError, ModuleNotFoundError):
        return None


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
    n_auxiliary_demons: int,
    sparsity: float,
) -> HordeActorCriticAgent:
    """Create the real Step 3-critic actor-critic implementation."""
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
            name=f"aux_{idx}",
            demon_type=DemonType.PREDICTION,
            gamma=0.0,
            lamda=0.0,
            cumulant_index=idx + 1,
        )
        for idx in range(n_auxiliary_demons)
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
        ),
        critic=critic,
        actor_bounder=bounder,
    )


class BSuiteActorCriticAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to an actor-critic control agent."""

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: Any,
        seed: int = 0,
    ) -> None:
        self._agent = agent
        self._num_actions: int = action_spec.num_values
        feature_dim = int(np.prod(obs_spec.shape))
        self._state = agent.init(feature_dim=feature_dim, key=jr.key(seed))
        self._jit_select = jax.jit(agent.select_action)
        self._jit_update = jax.jit(agent.update)
        self._jit_start = jax.jit(agent.start) if hasattr(agent, "start") else None
        self._implementation = (
            "horde_core"
            if isinstance(agent, HordeActorCriticAgent)
            else "linear_core"
            if self._jit_start is not None
            else "mlp"
        )
        self._core_mode = self._implementation != "mlp"
        self._horde_value_discount = (
            float(agent.critic.horde_spec.gammas[agent.config.value_head_index])
            if isinstance(agent, HordeActorCriticAgent)
            else 0.0
        )
        self._pending_action: int | None = None
        self._step_count = 0

    @property
    def state(self) -> Any:
        """Current actor-critic state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of completed updates."""
        return self._step_count

    @property
    def implementation(self) -> str:
        """Name of the wrapped actor-critic implementation path."""
        return self._implementation

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Select an action from the current policy."""
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return int(action)

        if self._core_mode:
            assert self._jit_start is not None
            self._state, action, _policy = self._jit_start(self._state, obs)
            return int(action)

        action, new_key = self._jit_select(self._state, obs)
        self._state = self._state.replace(rng_key=new_key)
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update from one bsuite transition."""
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        next_obs = jnp.asarray(new_timestep.observation, dtype=jnp.float32).flatten()
        reward = jnp.array(new_timestep.reward or 0.0, dtype=jnp.float32)
        discount = 1.0 if new_timestep.discount is None else float(new_timestep.discount)
        if self._implementation == "horde_core":
            del obs, action
            transition_discount = jnp.array(
                0.0 if discount == 0.0 else self._horde_value_discount,
                dtype=jnp.float32,
            )
            result = self._jit_update(
                self._state,
                reward,
                next_obs,
                None,
                transition_discount,
            )
            if hasattr(new_timestep, "last") and new_timestep.last():
                self._pending_action = None
            else:
                self._pending_action = int(result.action)
        elif self._implementation == "linear_core":
            del obs, action
            terminated = jnp.array(discount == 0.0, dtype=jnp.float32)
            result = self._jit_update(self._state, reward, next_obs, terminated)
            if hasattr(new_timestep, "last") and new_timestep.last():
                self._pending_action = None
            else:
                self._pending_action = int(result.action)
        else:
            result = self._jit_update(
                self._state,
                obs,
                jnp.array(action, dtype=jnp.int32),
                reward,
                next_obs,
                jnp.array(discount, dtype=jnp.float32),
            )
        self._state = result.state
        self._step_count += 1

    def get_preferences(self, observation: np.ndarray) -> np.ndarray:
        """Return current action preferences for debugging and tests."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        if hasattr(self._agent, "preferences"):
            return np.asarray(self._agent.preferences(self._state, obs))
        if hasattr(self._agent, "policy"):
            return np.asarray(self._agent.policy(self._state, obs))
        raise AttributeError("wrapped actor-critic agent has no preferences or policy")


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (64, 64),
    optimizer_name: str = "autostep",
    step_size: float = 0.03,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
    tau: float = 10000.0,
    kappa: float = 2.0,
    normalizer_decay: float = 0.99,
    discount: float = 0.99,
    temperature: float = 1.0,
    actor_step_size: float = 0.1,
    actor_lamda: float = 0.9,
    critic_lamda: float = 0.0,
    n_auxiliary_demons: int = 0,
    sparsity: float = 0.0,
    implementation: str = "horde_core",
    seed: int = 0,
    **_: Any,
) -> BSuiteActorCriticAgent:
    """Create a bsuite-compatible actor-critic agent."""
    if implementation not in {"horde_core", "mlp", "linear_core"}:
        raise ValueError("implementation must be 'horde_core', 'mlp', or 'linear_core'")
    if n_auxiliary_demons < 0:
        raise ValueError("n_auxiliary_demons must be non-negative")

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
    normalizer = EMANormalizer(decay=normalizer_decay)

    agent: Any
    if implementation == "horde_core":
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
            n_auxiliary_demons=n_auxiliary_demons,
            sparsity=sparsity,
        )
    elif implementation == "mlp":
        agent = _FallbackActorCriticAgent(
            n_actions=action_spec.num_values,
            hidden_sizes=hidden_sizes,
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            discount=discount,
            temperature=temperature,
            actor_step_size=actor_step_size,
            sparsity=sparsity,
        )
    else:
        if hidden_sizes:
            raise ValueError(
                "implementation='linear_core' is linear; set hidden_sizes=() "
                "or use implementation='mlp'."
            )
        if optimizer_name != "lms":
            raise ValueError(
                "implementation='linear_core' only supports optimizer_name='lms'; "
                "use implementation='mlp' for Autostep."
            )
        core_agent_cls = _load_core_actor_critic()
        core_config_cls = _load_core_actor_critic_config()
        if core_agent_cls is None or core_config_cls is None:
            raise RuntimeError("core ActorCriticAgent is not available")
        agent = core_agent_cls(
            config=core_config_cls(
                n_actions=action_spec.num_values,
                gamma=discount,
                actor_step_size=actor_step_size,
                critic_step_size=step_size,
                temperature=temperature,
            ),
            bounder=ObGDBounding(kappa=kappa),
        )

    return BSuiteActorCriticAgent(obs_spec, action_spec, agent, seed=seed)

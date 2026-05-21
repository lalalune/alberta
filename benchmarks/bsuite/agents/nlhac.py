"""bsuite adapter for NonlinearHordeActorCriticAgent — canonical Step 4.

This adapter wraps :class:`alberta_framework.NonlinearHordeActorCriticAgent`
for bsuite evaluation.  The MLP actor replaces the linear softmax actor of the
legacy ``horde_ac`` adapter, removing the structural bottleneck on high-dimensional
tasks (e.g. catch/0 with a 50-dim board representation).
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
    HordeLearner,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
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


DEFAULT_AUX_GAMMAS: tuple[float, ...] = (0.0, 0.5, 0.9)


def _make_nlhac_agent(
    *,
    n_actions: int,
    hidden_sizes: tuple[int, ...],
    actor_hidden_sizes: tuple[int, ...],
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
) -> NonlinearHordeActorCriticAgent:
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
    return NonlinearHordeActorCriticAgent(
        config=NonlinearHordeActorCriticConfig(
            n_actions=n_actions,
            actor_step_size=actor_step_size,
            actor_lamda=actor_lamda,
            temperature=temperature,
            value_head_index=0,
            hidden_sizes=actor_hidden_sizes,
            actor_sparsity=sparsity,
        ),
        critic=critic,
    )


class BSuiteNLHACAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to ``NonlinearHordeActorCriticAgent``."""

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: NonlinearHordeActorCriticAgent,
        seed: int = 0,
    ) -> None:
        self._agent = agent
        self._num_actions: int = action_spec.num_values
        feature_dim = int(np.prod(obs_spec.shape))
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
        """Current nonlinear Horde AC state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of completed updates."""
        return self._step_count

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Return the cached on-policy action or sample a fresh one."""
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            return int(action)
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
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
        next_obs = jnp.asarray(
            new_timestep.observation, dtype=jnp.float32
        ).flatten()
        reward = jnp.array(new_timestep.reward or 0.0, dtype=jnp.float32)
        env_discount = (
            1.0 if new_timestep.discount is None else float(new_timestep.discount)
        )
        transition_discount = jnp.array(
            0.0 if env_discount == 0.0 else self._horde_value_discount,
            dtype=jnp.float32,
        )
        aux_cumulants = (
            jnp.full((self._n_aux,), reward, dtype=jnp.float32)
            if self._n_aux > 0
            else None
        )
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
        else:
            self._pending_action = int(result.action)

    def get_value(self, observation: np.ndarray) -> float:
        """Return scalar critic value."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        return float(self._agent.value(self._state, obs))

    def get_policy(self, observation: np.ndarray) -> np.ndarray:
        """Return the current softmax policy."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        return np.asarray(self._agent.policy(self._state, obs))


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (64, 64),
    actor_hidden_sizes: tuple[int, ...] = (64,),
    optimizer_name: str = "autostep",
    step_size: float = 0.03,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
    tau: float = 10000.0,
    kappa: float = 2.0,
    normalizer_decay: float = 0.99,
    discount: float = 0.99,
    temperature: float = 0.5,
    actor_step_size: float = 0.01,
    actor_lamda: float = 0.9,
    critic_lamda: float = 0.0,
    aux_gammas: tuple[float, ...] = DEFAULT_AUX_GAMMAS,
    sparsity: float = 0.0,
    seed: int = 0,
    **_: Any,
) -> BSuiteNLHACAgent:
    """Create a bsuite-compatible nonlinear Horde actor-critic agent."""
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
    agent = _make_nlhac_agent(
        n_actions=action_spec.num_values,
        hidden_sizes=hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
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
    )
    return BSuiteNLHACAgent(obs_spec, action_spec, agent, seed=seed)

"""SARSA bsuite agent: on-policy control through the framework SARSAAgent."""

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
    EMANormalizer,
    ObGDBounding,
    SARSAAgent,
    SARSAConfig,
)
from alberta_framework.benchmarks.bsuite._bsuite_path import add_bsuite_to_path, bsuite_missing_message

add_bsuite_to_path()
try:
    base: Any = importlib.import_module("bsuite.baselines.base")
except ModuleNotFoundError as exc:
    if exc.name == "bsuite":
        raise ModuleNotFoundError(bsuite_missing_message()) from exc
    raise


class BSuiteSARSAAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to :class:`SARSAAgent`.

    bsuite asks the agent to select an action before the transition and then
    calls ``update`` after the transition. SARSA needs the next on-policy
    action before it can update. This adapter selects that next action inside
    ``update`` and caches it so the next ``select_action`` call returns the
    exact action used in the SARSA target.
    """

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: SARSAAgent,
        seed: int = 0,
    ) -> None:
        self._agent = agent
        self._num_actions: int = action_spec.num_values
        feature_dim = int(np.prod(obs_spec.shape))
        self._state = agent.init(feature_dim=feature_dim, key=jr.key(seed))
        self._jit_select = jax.jit(agent.select_action)
        self._jit_update = jax.jit(agent.update)
        self._pending_action: int | None = None
        self._step_count = 0

    @property
    def state(self) -> Any:
        """Current SARSA state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of completed updates."""
        return self._step_count

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """Return the pending on-policy action or choose a fresh one."""
        obs = jnp.asarray(timestep.observation, dtype=jnp.float32).flatten()
        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            self._state = self._state.replace(
                last_action=jnp.array(action, dtype=jnp.int32),
                last_observation=obs,
            )
            return int(action)

        action, new_key = self._jit_select(self._state, obs)
        self._state = self._state.replace(
            last_action=action,
            last_observation=obs,
            rng_key=new_key,
        )
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        """Update from one transition with a true on-policy SARSA target."""
        del timestep, action
        next_obs = jnp.asarray(new_timestep.observation, dtype=jnp.float32).flatten()
        reward = jnp.array(new_timestep.reward or 0.0, dtype=jnp.float32)
        discount = 1.0 if new_timestep.discount is None else float(new_timestep.discount)
        terminated = jnp.array(discount == 0.0, dtype=jnp.float32)

        next_action, new_key = self._jit_select(self._state, next_obs)
        self._state = self._state.replace(rng_key=new_key)
        result = self._jit_update(
            self._state,
            reward,
            next_obs,
            terminated,
            next_action,
        )
        self._state = result.state
        self._step_count += 1

        if hasattr(new_timestep, "last") and new_timestep.last():
            self._pending_action = None
        else:
            self._pending_action = int(next_action)

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return current Q-values for debugging and tests."""
        obs = jnp.asarray(observation, dtype=jnp.float32).flatten()
        preds = self._agent.horde.predict(self._state.learner_state, obs)
        return np.asarray(preds[: self._num_actions])


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
    epsilon: float = 0.05,
    epsilon_decay_steps: int = 0,
    lamda: float = 0.0,
    seed: int = 0,
    **_: Any,
) -> BSuiteSARSAAgent:
    """Create a bsuite-compatible SARSA agent."""
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

    sarsa = SARSAAgent(
        sarsa_config=SARSAConfig(
            n_actions=action_spec.num_values,
            gamma=discount,
            epsilon_start=epsilon,
            epsilon_end=epsilon,
            epsilon_decay_steps=epsilon_decay_steps,
        ),
        hidden_sizes=hidden_sizes,
        optimizer=optimizer,
        bounder=ObGDBounding(kappa=kappa),
        normalizer=EMANormalizer(decay=normalizer_decay),
        sparsity=0.0,
        lamda=lamda,
    )
    return BSuiteSARSAAgent(obs_spec, action_spec, sarsa, seed=seed)

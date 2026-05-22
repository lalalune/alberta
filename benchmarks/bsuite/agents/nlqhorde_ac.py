"""bsuite adapter for nonlinear action-value Horde actor-critic."""

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
    AdaptiveObGDBounding,
    Autostep,
    DemonType,
    EMANormalizer,
    GVFSpec,
    HordeLearner,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
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


def _make_agent(
    *,
    n_actions: int,
    hidden_sizes: tuple[int, ...],
    actor_hidden_sizes: tuple[int, ...],
    optimizer: Any,
    actor_optimizer: Autostep,
    bounder: Any,
    normalizer: Any,
    discount: float,
    temperature: float,
    actor_lamda: float,
    critic_lamda: float,
    critic_target: str,
    actor_update: str,
    aux_gammas: tuple[float, ...],
    sparsity: float,
    actor_bounder: Any,
    actor_td_error_clip: float | None,
    actor_gradient_clip_norm: float | None,
) -> NonlinearQHordeActorCriticAgent:
    """Build nonlinear Q-Horde actor-critic with one control demon per action."""
    demons = [
        GVFSpec(
            name=f"q_{action}",
            demon_type=DemonType.CONTROL,
            gamma=0.0,
            lamda=critic_lamda,
            cumulant_index=-1,
        )
        for action in range(n_actions)
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
    return NonlinearQHordeActorCriticAgent(
        config=NonlinearQHordeActorCriticConfig(
            n_actions=n_actions,
            gamma=discount,
            actor_lamda=actor_lamda,
            temperature=temperature,
            hidden_sizes=actor_hidden_sizes,
            actor_sparsity=sparsity,
            actor_td_error_clip=actor_td_error_clip,
            actor_gradient_clip_norm=actor_gradient_clip_norm,
            critic_target=critic_target,
            actor_update=actor_update,
        ),
        critic=critic,
        actor_optimizer=actor_optimizer,
        actor_bounder=actor_bounder,
    )


class BSuiteNLQHordeActorCriticAgent(base.Agent):  # type: ignore[misc]
    """Bridge bsuite's mutable Agent API to ``NonlinearQHordeActorCriticAgent``."""

    def __init__(
        self,
        obs_spec: dm_env.specs.Array,
        action_spec: dm_env.specs.DiscreteArray,
        agent: NonlinearQHordeActorCriticAgent,
        seed: int = 0,
    ) -> None:
        self._agent = agent
        self._n_actions = action_spec.num_values
        feature_dim = int(np.prod(obs_spec.shape))
        self._state = agent.init(feature_dim=feature_dim, key=jr.key(seed))
        self._jit_start = jax.jit(agent.start)
        self._jit_update = jax.jit(agent.update)
        self._n_aux = agent.critic.n_demons - self._n_actions
        self._pending_action: int | None = None
        self._step_count = 0

    @property
    def state(self) -> Any:
        """Current nonlinear Q-Horde actor-critic state."""
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
        terminated = jnp.array(env_discount == 0.0, dtype=jnp.float32)
        aux = (
            jnp.full((self._n_aux,), reward, dtype=jnp.float32)
            if self._n_aux > 0
            else None
        )
        result = self._jit_update(self._state, reward, next_obs, terminated, aux)
        self._state = result.state
        self._step_count += 1
        if hasattr(new_timestep, "last") and new_timestep.last():
            self._pending_action = None
        else:
            self._pending_action = int(result.action)


def default_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    hidden_sizes: tuple[int, ...] = (32,),
    actor_hidden_sizes: tuple[int, ...] = (32,),
    optimizer_name: str = "autostep",
    step_size: float = 0.03,
    actor_step_size: float | None = None,
    initial_step_size: float = 0.01,
    meta_step_size: float = 0.01,
    tau: float = 10000.0,
    kappa: float = 2.0,
    actor_kappa: float | None = None,
    actor_initial_step_size: float = 0.01,
    actor_meta_step_size: float = 0.01,
    actor_tau: float = 10000.0,
    bounder_name: str = "obgd",
    actor_bounder_name: str | None = None,
    normalizer_decay: float = 0.99,
    discount: float = 0.99,
    temperature: float = 0.5,
    actor_lamda: float = 0.9,
    critic_lamda: float = 0.0,
    critic_target: str = "expected_sarsa",
    actor_update: str = "td_error",
    aux_gammas: tuple[float, ...] = DEFAULT_AUX_GAMMAS,
    sparsity: float = 0.0,
    actor_td_error_clip: float | None = None,
    actor_gradient_clip_norm: float | None = None,
    seed: int = 0,
    **_: Any,
) -> BSuiteNLQHordeActorCriticAgent:
    """Create a bsuite-compatible nonlinear action-value Horde AC agent."""
    if actor_step_size is not None:
        actor_initial_step_size = actor_step_size

    if optimizer_name == "autostep":
        optimizer: Any = Autostep(
            initial_step_size=initial_step_size,
            meta_step_size=meta_step_size,
            tau=tau,
        )
    elif optimizer_name == "lms":
        optimizer = LMS(step_size=step_size)
    else:
        raise ValueError("optimizer_name must be 'autostep' or 'lms'")

    if bounder_name == "obgd":
        bounder = ObGDBounding(kappa=kappa)
    elif bounder_name == "adaptive_obgd":
        bounder = AdaptiveObGDBounding(kappa=kappa)
    else:
        raise ValueError("bounder_name must be 'obgd' or 'adaptive_obgd'")

    actor_bounder_choice = actor_bounder_name or bounder_name
    actor_kappa_value = kappa if actor_kappa is None else actor_kappa
    if actor_bounder_choice == "obgd":
        actor_bounder = ObGDBounding(kappa=actor_kappa_value)
    elif actor_bounder_choice == "adaptive_obgd":
        actor_bounder = AdaptiveObGDBounding(kappa=actor_kappa_value)
    else:
        raise ValueError("actor_bounder_name must be 'obgd' or 'adaptive_obgd'")
    actor_optimizer = Autostep(
        initial_step_size=actor_initial_step_size,
        meta_step_size=actor_meta_step_size,
        tau=actor_tau,
    )
    agent = _make_agent(
        n_actions=action_spec.num_values,
        hidden_sizes=hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        optimizer=optimizer,
        actor_optimizer=actor_optimizer,
        bounder=bounder,
        normalizer=EMANormalizer(decay=normalizer_decay),
        discount=discount,
        temperature=temperature,
        actor_lamda=actor_lamda,
        critic_lamda=critic_lamda,
        critic_target=critic_target,
        actor_update=actor_update,
        aux_gammas=aux_gammas,
        sparsity=sparsity,
        actor_bounder=actor_bounder,
        actor_td_error_clip=actor_td_error_clip,
        actor_gradient_clip_norm=actor_gradient_clip_norm,
    )
    return BSuiteNLQHordeActorCriticAgent(obs_spec, action_spec, agent, seed=seed)

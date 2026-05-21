"""Shared framework-backed Q-learning agent factory for bsuite adapters."""

from __future__ import annotations

from typing import Any

import dm_env

from alberta_framework import EMANormalizer, MultiHeadMLPLearner, ObGDBounding
from benchmarks.bsuite.agents.base import AlbertaAgent


def make_framework_q_agent(
    obs_spec: dm_env.specs.Array,
    action_spec: dm_env.specs.DiscreteArray,
    *,
    hidden_sizes: tuple[int, ...],
    discount: float,
    epsilon: float,
    seed: int,
    kappa: float,
    normalizer_decay: float,
    log_representation: bool,
    log_interval: int,
    optimizer: Any | None = None,
    step_size: float | None = None,
) -> AlbertaAgent:
    """Create an :class:`AlbertaAgent` with common Q-learning wiring."""
    learner_kwargs: dict[str, Any] = {
        "n_heads": action_spec.num_values,
        "hidden_sizes": hidden_sizes,
        "bounder": ObGDBounding(kappa=kappa),
        "normalizer": EMANormalizer(decay=normalizer_decay),
    }
    if optimizer is not None:
        learner_kwargs["optimizer"] = optimizer
    if step_size is not None:
        learner_kwargs["step_size"] = step_size

    learner = MultiHeadMLPLearner(**learner_kwargs)
    return AlbertaAgent(
        obs_spec=obs_spec,
        action_spec=action_spec,
        learner=learner,
        discount=discount,
        epsilon=epsilon,
        seed=seed,
        log_representation=log_representation,
        log_interval=log_interval,
    )

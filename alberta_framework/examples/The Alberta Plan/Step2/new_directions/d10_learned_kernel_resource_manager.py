#!/usr/bin/env python3
"""D10: learned resource management for additive kernel banks.

This runner targets the remaining D07 conflict directly.  It is not a
prediction router: the prediction is always the sum of all active RKHS banks.
Every bank updates its existing coefficients on every online step against the
same global residual.  The learned decision is only where to spend scarce
dictionary-growth resource: which bank, if any, receives the next center slot or
replacement.

The default banks mirror the useful D07 mechanisms:

* raw degree-3 polynomial RKHS for exact low-dimensional polynomial/frequency
  structure,
* normalized algebraic-Green RKHS with slower center allocation for stateful
  external memory,
* arc-cosine / NNGP RKHS for compositional smooth nonlinear structure.

The manager is a cost-sensitive softmax bandit over allocation opportunities.
It updates from online evidence: residual loss, ALD novelty, active-budget
pressure, one-step same-sample loss reduction, and configured compute costs.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step2_expert_mixture import (  # noqa: E402
    DIGITS_REGIMES,
    N_DIGIT_CLASSES,
)

from d07_budgeted_kernel_recursive import (  # noqa: E402
    MLP_METHODS,
    BudgetedDiffusionKRLS,
    BudgetedKRLSState,
    KernelConfig,
    aggregate_records,
    compare_to_group,
    evaluate_mlp_classifier,
    expand_dataset_names as expand_d07_dataset_names,
    make_dataset,
    make_mlp,
    masked_mse_np,
    paired_diff,
    run_mlp_stream,
    stderr,
    summarize_prequential,
    transform_observation,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d10_learned_kernel_resource_manager")
DEFAULT_NOTE_PATH = Path(
    "docs/research/step2_new_directions/d10_learned_kernel_resource_manager.md"
)
BANK_NAMES = ("raw_poly", "algebraic_green", "arccosine")
MANAGER_METHODS = ("learned_softmax", "novelty_greedy", "round_robin")
DEFAULT_DATASETS = (
    "controlled_frequency",
    "synthetic_compositional",
    "digits_label_drift",
    "digits_mask_noise",
    "digits_permuted_pixels",
    "controlled_nonlinear",
)
ManagerMethod = Literal["learned_softmax", "novelty_greedy", "round_robin"]


@dataclass(frozen=True)
class BankSpec:
    """One additive RKHS bank and its resource metadata."""

    name: str
    config: KernelConfig
    update_scale: float
    resource_cost: float
    utility_scale: float


@dataclass(frozen=True)
class ManagerConfig:
    """Hyperparameters for the learned allocation manager."""

    method: ManagerMethod
    learning_rate: float
    discount: float
    exploration: float
    utility_decay: float
    cost_weight: float
    advantage_clip: float
    ucb_bonus: float
    residual_power: float
    novelty_power: float
    actual_gain_weight: float
    total_center_budget: int
    allow_rebalance: bool
    rho_span: float
    min_rho: float


@dataclass
class AllocationManagerState:
    """Mutable NumPy state for one allocation manager."""

    log_weights: np.ndarray
    utility_ema: np.ndarray
    action_counts: np.ndarray
    allocations: np.ndarray
    resource_transfers: int
    denied_allocations: int
    no_eligible_steps: int
    step_count: int
    round_robin_cursor: int


@dataclass
class MultiBankState:
    """Mutable state for the additive multi-bank learner."""

    bank_states: list[BudgetedKRLSState]
    manager_state: AllocationManagerState
    rho_values: np.ndarray
    steps: int


class SoftmaxAllocationManager:
    """Cost-sensitive learned manager over center-allocation opportunities."""

    def __init__(self, n_actions: int, config: ManagerConfig) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not 0.0 <= config.discount <= 1.0:
            raise ValueError("manager discount must be in [0, 1]")
        if not 0.0 <= config.exploration < 1.0:
            raise ValueError("manager exploration must be in [0, 1)")
        if not 0.0 <= config.utility_decay < 1.0:
            raise ValueError("manager utility decay must be in [0, 1)")
        if config.learning_rate < 0.0:
            raise ValueError("manager learning rate must be non-negative")
        if config.cost_weight < 0.0:
            raise ValueError("manager cost weight must be non-negative")
        if config.advantage_clip <= 0.0:
            raise ValueError("manager advantage clip must be positive")
        if config.ucb_bonus < 0.0:
            raise ValueError("manager UCB bonus must be non-negative")
        self.n_actions = int(n_actions)
        self.config = config

    def init(self) -> AllocationManagerState:
        """Return a uniform manager state."""
        return AllocationManagerState(
            log_weights=np.zeros(self.n_actions, dtype=np.float64),
            utility_ema=np.zeros(self.n_actions, dtype=np.float64),
            action_counts=np.zeros(self.n_actions, dtype=np.float64),
            allocations=np.zeros(self.n_actions, dtype=np.int64),
            resource_transfers=0,
            denied_allocations=0,
            no_eligible_steps=0,
            step_count=0,
            round_robin_cursor=0,
        )

    def weights(self, state: AllocationManagerState) -> np.ndarray:
        """Return causal pre-update allocation weights."""
        shifted = state.log_weights - np.max(state.log_weights)
        exp = np.exp(shifted)
        weights = exp / max(float(np.sum(exp)), 1e-12)
        if self.config.exploration > 0.0:
            uniform = np.full_like(weights, 1.0 / self.n_actions)
            weights = (1.0 - self.config.exploration) * weights + (
                self.config.exploration * uniform
            )
        return np.asarray(weights, dtype=np.float64)

    def choose(
        self,
        state: AllocationManagerState,
        utilities: np.ndarray,
        eligible: np.ndarray,
    ) -> int:
        """Choose one eligible action or ``-1`` when no bank can receive a center."""
        if not np.any(eligible):
            state.no_eligible_steps += 1
            return -1
        if self.config.method == "round_robin":
            for offset in range(self.n_actions):
                idx = (state.round_robin_cursor + offset) % self.n_actions
                if bool(eligible[idx]):
                    state.round_robin_cursor = (idx + 1) % self.n_actions
                    return int(idx)
            return -1
        if self.config.method == "novelty_greedy":
            scores = np.where(eligible, utilities, -np.inf)
            return int(np.argmax(scores))

        weights = self.weights(state)
        counts = np.maximum(state.action_counts, 1.0)
        bonus = self.config.ucb_bonus * np.sqrt(
            math.log(max(state.step_count + 2, 2)) / counts
        )
        scores = weights * np.maximum(utilities, 0.0) + bonus
        scores = np.where(eligible, scores, -np.inf)
        return int(np.argmax(scores))

    def update(
        self,
        state: AllocationManagerState,
        utilities: np.ndarray,
        costs: np.ndarray,
        finite: np.ndarray,
    ) -> None:
        """Update preferences from full-information utility estimates."""
        state.step_count += 1
        if self.config.method != "learned_softmax":
            state.action_counts += finite.astype(np.float64)
            decay = self.config.utility_decay
            state.utility_ema = np.where(
                finite,
                decay * state.utility_ema + (1.0 - decay) * utilities,
                state.utility_ema,
            )
            return

        weights = self.weights(state)
        adjusted_losses = -utilities + self.config.cost_weight * costs
        finite_weights = np.where(finite, weights, 0.0)
        finite_weight_sum = max(float(np.sum(finite_weights)), 1e-12)
        masked_weights = finite_weights / finite_weight_sum
        baseline = float(np.sum(masked_weights * adjusted_losses))
        advantages = np.where(finite, baseline - adjusted_losses, 0.0)
        advantages = np.clip(
            advantages,
            -self.config.advantage_clip,
            self.config.advantage_clip,
        )
        state.log_weights = self.config.discount * state.log_weights + (
            self.config.learning_rate * advantages
        )
        state.log_weights -= float(np.mean(state.log_weights))
        state.action_counts += finite.astype(np.float64)
        decay = self.config.utility_decay
        state.utility_ema = np.where(
            finite,
            decay * state.utility_ema + (1.0 - decay) * utilities,
            state.utility_ema,
        )


class ManagedMultiBankKRLS:
    """Additive multi-bank RKHS learner with learned center allocation."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        bank_specs: tuple[BankSpec, ...],
        manager_config: ManagerConfig,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.bank_specs = bank_specs
        self.banks = tuple(
            BudgetedDiffusionKRLS(
                n_heads=n_heads,
                feature_dim=feature_dim,
                config=spec.config,
            )
            for spec in bank_specs
        )
        self.manager = SoftmaxAllocationManager(
            n_actions=len(bank_specs),
            config=manager_config,
        )
        self.manager_config = manager_config
        max_bank_budget = sum(spec.config.budget for spec in bank_specs)
        self.total_center_budget = min(
            int(manager_config.total_center_budget),
            int(max_bank_budget),
        )
        if self.total_center_budget <= 0:
            raise ValueError("total center budget must be positive")

    def init(self) -> MultiBankState:
        """Create an empty state for every bank and the manager."""
        return MultiBankState(
            bank_states=[bank.init() for bank in self.banks],
            manager_state=self.manager.init(),
            rho_values=np.asarray(
                [spec.config.rho for spec in self.bank_specs],
                dtype=np.float64,
            ),
            steps=0,
        )

    def predict(self, state: MultiBankState, observation: np.ndarray) -> np.ndarray:
        """Return the additive prediction across all banks."""
        preds = [
            bank.predict(bank_state, observation)
            for bank, bank_state in zip(self.banks, state.bank_states, strict=True)
        ]
        if not preds:
            return np.zeros(self.n_heads, dtype=np.float64)
        return cast(np.ndarray, np.sum(np.stack(preds, axis=0), axis=0))

    def _total_active_centers(self, bank_states: list[BudgetedKRLSState]) -> int:
        """Return the active centers across all banks."""
        return int(sum(bank_state.active_count for bank_state in bank_states))

    def _can_allocate(
        self,
        bank: BudgetedDiffusionKRLS,
        bank_state: BudgetedKRLSState,
        all_bank_states: list[BudgetedKRLSState],
        novelty: float,
    ) -> bool:
        cfg = bank.config
        if novelty <= cfg.novelty_threshold:
            return False
        if (bank_state.steps - bank_state.last_center_step) < cfg.center_add_interval:
            return False
        if bank_state.active_count >= cfg.budget:
            return cfg.replace_when_full and bank_state.active_count > 0
        if self._total_active_centers(all_bank_states) < self.total_center_budget:
            return True
        return self.manager_config.allow_rebalance and any(
            state.active_count > 0 for state in all_bank_states
        )

    def _remove_center(
        self,
        bank: BudgetedDiffusionKRLS,
        bank_state: BudgetedKRLSState,
        index: int,
    ) -> bool:
        """Remove one center by swapping with the active tail and rebuilding state."""
        if bank_state.active_count <= 0:
            return False
        idx = int(np.clip(index, 0, bank_state.active_count - 1))
        last = bank_state.active_count - 1
        if idx != last:
            bank_state.centers[idx] = bank_state.centers[last]
            bank_state.alpha[idx] = bank_state.alpha[last]
            bank_state.activation_ema[idx] = bank_state.activation_ema[last]
            bank_state.coefficient_ema[idx] = bank_state.coefficient_ema[last]
            bank_state.ages[idx] = bank_state.ages[last]
        bank_state.centers[last] = 0.0
        bank_state.alpha[last] = 0.0
        bank_state.activation_ema[last] = 0.0
        bank_state.coefficient_ema[last] = 0.0
        bank_state.ages[last] = 0
        bank_state.active_count -= 1
        bank_state.p_matrix.fill(0.0)
        if bank_state.active_count > 0:
            active = bank_state.active_count
            bank_state.p_matrix[:active, :active] = (
                np.eye(active, dtype=np.float64) * bank.config.rls_delta
            )
        bank._rebuild_k_inv(bank_state)  # noqa: SLF001
        return True

    def _choose_transfer_donor(
        self,
        state: MultiBankState,
        selected: int,
        utilities: np.ndarray,
        costs: np.ndarray,
    ) -> int:
        """Choose the least useful active bank to donate one center."""
        scores = (
            state.manager_state.utility_ema
            + utilities
            - self.manager_config.cost_weight * costs
        )
        donor_candidates = [
            idx
            for idx, bank_state in enumerate(state.bank_states)
            if bank_state.active_count > 0 and idx != selected
        ]
        if not donor_candidates:
            donor_candidates = [
                idx
                for idx, bank_state in enumerate(state.bank_states)
                if bank_state.active_count > 0
            ]
        if not donor_candidates:
            return -1
        return int(min(donor_candidates, key=lambda idx: scores[idx]))

    def _rebalance_for_selected_bank(
        self,
        state: MultiBankState,
        selected: int,
        utilities: np.ndarray,
        costs: np.ndarray,
    ) -> None:
        """Free one global center slot when the learned manager reallocates budget."""
        if selected < 0 or not self.manager_config.allow_rebalance:
            return
        selected_state = state.bank_states[selected]
        selected_bank = self.banks[selected]
        if selected_state.active_count >= selected_bank.config.budget:
            return
        if self._total_active_centers(state.bank_states) < self.total_center_budget:
            return
        donor = self._choose_transfer_donor(state, selected, utilities, costs)
        if donor < 0:
            return
        donor_bank = self.banks[donor]
        donor_state = state.bank_states[donor]
        donor_idx = donor_bank._replacement_index(donor_state)  # noqa: SLF001
        if self._remove_center(donor_bank, donor_state, donor_idx):
            state.manager_state.resource_transfers += 1

    def _rho_values_from_weights(self, weights: np.ndarray) -> np.ndarray:
        """Map learned resource weights to per-bank RLS forgetting factors."""
        base = np.asarray([spec.config.rho for spec in self.bank_specs], dtype=np.float64)
        if self.manager_config.rho_span <= 0.0:
            return base
        uniform = 1.0 / len(self.bank_specs)
        rho = base + self.manager_config.rho_span * (weights - uniform)
        return np.clip(rho, self.manager_config.min_rho, 0.9999)

    def _bank_update(
        self,
        bank: BudgetedDiffusionKRLS,
        bank_state: BudgetedKRLSState,
        spec: BankSpec,
        observation: np.ndarray,
        target_for_bank: np.ndarray,
        allow_center: bool,
        novelty: float,
        effective_rho: float,
    ) -> dict[str, float]:
        """Update one bank on the global additive residual."""
        z = transform_observation(observation, bank.config.input_clip)
        if allow_center:
            before_additions = bank_state.additions
            before_replacements = bank_state.replacements
            bank._add_or_replace_center(bank_state, z, novelty)  # noqa: SLF001
            if (
                before_additions == bank_state.additions
                and before_replacements == bank_state.replacements
                and novelty > bank.config.novelty_threshold
            ):
                bank_state.skipped_novel += 1
        elif novelty > bank.config.novelty_threshold:
            bank_state.skipped_novel += 1

        m = bank_state.active_count
        leverage = 0.0
        if m > 0:
            phi = bank._kernel(z, bank_state.centers[:m]).reshape(m)  # noqa: SLF001
            active = ~np.isnan(target_for_bank)
            safe_target = np.where(active, target_for_bank, 0.0)
            update_prediction = phi @ bank_state.alpha[:m]
            errors = np.where(active, safe_target - update_prediction, 0.0)
            if bank.config.coefficient_update == "rls":
                rho = float(np.clip(effective_rho, self.manager_config.min_rho, 0.9999))
                p_active = bank_state.p_matrix[:m, :m]
                p_phi = p_active @ phi
                denom = rho + float(phi @ p_phi)
                if denom <= 1e-12 or not np.isfinite(denom):
                    bank_state.finite_failures += 1
                else:
                    gain = p_phi / denom
                    bank_state.alpha[:m] += spec.update_scale * np.outer(gain, errors)
                    next_p = (p_active - np.outer(gain, phi @ p_active)) / rho
                    bank_state.p_matrix[:m, :m] = 0.5 * (next_p + next_p.T)
                    leverage = float(phi @ p_phi)
            else:
                normalizer = 1.0 + float(phi @ phi)
                bank_state.alpha[:m] += (
                    spec.update_scale
                    * bank.config.lms_step_size
                    * np.outer(phi, errors)
                    / normalizer
                )
                leverage = float(phi @ phi)

            decay = bank.config.utility_decay
            bank_state.activation_ema[:m] = decay * bank_state.activation_ema[:m] + (
                1.0 - decay
            ) * np.abs(phi)
            bank_state.coefficient_ema[:m] = decay * bank_state.coefficient_ema[:m] + (
                1.0 - decay
            ) * np.mean(np.abs(bank_state.alpha[:m]), axis=1)
            bank_state.ages[:m] += 1

        if not np.all(np.isfinite(bank_state.alpha[:m])):
            bank_state.finite_failures += 1
            bank_state.alpha[:m] = np.nan_to_num(bank_state.alpha[:m], copy=False)
        bank_state.novelty_sum += float(novelty)
        bank_state.leverage_sum += leverage
        bank_state.steps += 1
        return {
            "active_centers": float(bank_state.active_count),
            "novelty": float(novelty),
            "leverage": float(leverage),
            "additions": float(bank_state.additions),
            "replacements": float(bank_state.replacements),
            "skipped_novel": float(bank_state.skipped_novel),
            "finite_failures": float(bank_state.finite_failures),
        }

    def _allocation_utilities(
        self,
        bank_states: list[BudgetedKRLSState],
        residual_loss: float,
        novelties: np.ndarray,
        eligible: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate allocation utility and costs before spending the resource."""
        utilities = np.zeros(len(self.bank_specs), dtype=np.float64)
        costs = np.zeros(len(self.bank_specs), dtype=np.float64)
        residual_term = max(residual_loss, 0.0) ** self.manager_config.residual_power
        for idx, (spec, bank_state) in enumerate(
            zip(self.bank_specs, bank_states, strict=True)
        ):
            budget = max(float(spec.config.budget), 1.0)
            pressure = float(bank_state.active_count) / budget
            threshold = max(spec.config.novelty_threshold, 1e-12)
            novelty_term = (novelties[idx] / (threshold + novelties[idx] + 1e-12)) ** (
                self.manager_config.novelty_power
            )
            utility = spec.utility_scale * residual_term * novelty_term
            utilities[idx] = utility if eligible[idx] else 0.1 * utility
            costs[idx] = spec.resource_cost * (1.0 + pressure)
        return utilities, costs

    def step(
        self,
        state: MultiBankState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, allocate one center opportunity, update all banks, update manager."""
        bank_predictions = [
            bank.predict(bank_state, observation)
            for bank, bank_state in zip(self.banks, state.bank_states, strict=True)
        ]
        total_prediction = np.sum(np.stack(bank_predictions, axis=0), axis=0)
        pre_loss = masked_mse_np(total_prediction, target)
        z_values = [
            transform_observation(observation, spec.config.input_clip)
            for spec in self.bank_specs
        ]
        novelties = np.asarray(
            [
                bank._novelty(bank_state, z)  # noqa: SLF001
                for bank, bank_state, z in zip(
                    self.banks,
                    state.bank_states,
                    z_values,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        eligible = np.asarray(
            [
                self._can_allocate(
                    bank,
                    bank_state,
                    state.bank_states,
                    float(novelty),
                )
                for bank, bank_state, novelty in zip(
                    self.banks,
                    state.bank_states,
                    novelties,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        utilities, costs = self._allocation_utilities(
            state.bank_states,
            pre_loss,
            novelties,
            eligible,
        )
        weights = self.manager.weights(state.manager_state)
        state.rho_values = self._rho_values_from_weights(weights)
        selected = self.manager.choose(state.manager_state, utilities, eligible)
        self._rebalance_for_selected_bank(state, selected, utilities, costs)
        selected_before_allocations = None
        if selected >= 0:
            selected_before_allocations = (
                state.bank_states[selected].additions
                + state.bank_states[selected].replacements
            )
        target_for_banks = [
            np.where(~np.isnan(target), target - (total_prediction - pred), np.nan)
            for pred in bank_predictions
        ]
        diagnostics_by_bank: list[dict[str, float]] = []
        for idx, (bank, bank_state, spec, target_for_bank) in enumerate(
            zip(
                self.banks,
                state.bank_states,
                self.bank_specs,
                target_for_banks,
                strict=True,
            )
        ):
            diagnostics_by_bank.append(
                self._bank_update(
                    bank=bank,
                    bank_state=bank_state,
                    spec=spec,
                    observation=observation,
                    target_for_bank=target_for_bank,
                    allow_center=idx == selected,
                    novelty=float(novelties[idx]),
                    effective_rho=float(state.rho_values[idx]),
                )
            )
        if selected >= 0 and selected_before_allocations is not None:
            selected_after_allocations = (
                state.bank_states[selected].additions
                + state.bank_states[selected].replacements
            )
            if selected_after_allocations > selected_before_allocations:
                state.manager_state.allocations[selected] += 1
            else:
                state.manager_state.denied_allocations += 1

        post_prediction = self.predict(state, observation)
        post_loss = masked_mse_np(post_prediction, target)
        actual_gain = max(0.0, pre_loss - post_loss)
        learned_utilities = utilities.copy()
        if selected >= 0:
            learned_utilities[selected] += self.manager_config.actual_gain_weight * actual_gain
        self.manager.update(
            state.manager_state,
            utilities=learned_utilities,
            costs=costs,
            finite=np.ones_like(eligible, dtype=bool),
        )
        state.steps += 1
        totals = {
            "active_centers": float(
                sum(bank_state.active_count for bank_state in state.bank_states)
            ),
            "additions": float(sum(bank_state.additions for bank_state in state.bank_states)),
            "replacements": float(
                sum(bank_state.replacements for bank_state in state.bank_states)
            ),
            "skipped_novel": float(
                sum(bank_state.skipped_novel for bank_state in state.bank_states)
            ),
            "finite_failures": float(
                sum(bank_state.finite_failures for bank_state in state.bank_states)
            ),
            "pre_loss": float(pre_loss),
            "post_loss": float(post_loss),
            "actual_gain": float(actual_gain),
            "selected_bank": float(selected),
            "resource_transfers": float(state.manager_state.resource_transfers),
        }
        for idx, name in enumerate(BANK_NAMES):
            totals[f"{name}_centers"] = diagnostics_by_bank[idx]["active_centers"]
            totals[f"{name}_novelty"] = diagnostics_by_bank[idx]["novelty"]
            totals[f"{name}_weight"] = float(weights[idx])
            totals[f"{name}_utility"] = float(learned_utilities[idx])
            totals[f"{name}_allocations"] = float(state.manager_state.allocations[idx])
            totals[f"{name}_rho"] = float(state.rho_values[idx])
        return total_prediction, totals


def make_bank_specs(args: argparse.Namespace) -> tuple[BankSpec, ...]:
    """Create the three canonical D07-derived bank specifications."""
    common = {
        "sigma": 1.0,
        "rho": args.rho,
        "ridge": args.ridge,
        "rls_delta": args.rls_delta,
        "utility_decay": args.kernel_utility_decay,
        "min_center_age": args.min_center_age,
        "input_clip": args.input_clip,
        "bandwidth_multipliers": tuple(args.bandwidth_multipliers),
        "kernel_weight_variance": args.kernel_weight_variance,
        "kernel_bias_variance": args.kernel_bias_variance,
        "coefficient_update": "rls",
        "lms_step_size": args.kernel_lms_step_size,
        "replace_when_full": args.replace_when_full,
    }
    raw_poly = BankSpec(
        name="raw_poly",
        config=KernelConfig(
            budget=args.raw_poly_budget,
            novelty_threshold=args.raw_poly_novelty,
            kernel="polynomial",
            polynomial_degree=3,
            algebraic_weight=1.0,
            normalize_polynomial=False,
            arccosine_depth=args.arccosine_depth,
            center_add_interval=args.raw_poly_add_interval,
            **common,
        ),
        update_scale=args.raw_poly_update_scale,
        resource_cost=args.raw_poly_cost,
        utility_scale=args.raw_poly_utility_scale,
    )
    algebraic_green = BankSpec(
        name="algebraic_green",
        config=KernelConfig(
            budget=args.algebraic_budget,
            novelty_threshold=args.algebraic_novelty,
            kernel="algebraic_green",
            polynomial_degree=3,
            algebraic_weight=args.algebraic_weight,
            normalize_polynomial=True,
            arccosine_depth=args.arccosine_depth,
            center_add_interval=args.algebraic_add_interval,
            **common,
        ),
        update_scale=args.algebraic_update_scale,
        resource_cost=args.algebraic_cost,
        utility_scale=args.algebraic_utility_scale,
    )
    arccosine = BankSpec(
        name="arccosine",
        config=KernelConfig(
            budget=args.arccosine_budget,
            novelty_threshold=args.arccosine_novelty,
            kernel="arccosine",
            polynomial_degree=3,
            algebraic_weight=0.0,
            normalize_polynomial=True,
            arccosine_depth=args.arccosine_depth,
            center_add_interval=args.arccosine_add_interval,
            **common,
        ),
        update_scale=args.arccosine_update_scale,
        resource_cost=args.arccosine_cost,
        utility_scale=args.arccosine_utility_scale,
    )
    return raw_poly, algebraic_green, arccosine


def make_manager_config(method: str, args: argparse.Namespace) -> ManagerConfig:
    """Create one manager configuration."""
    if method not in MANAGER_METHODS:
        raise ValueError(f"unknown manager method {method!r}")
    return ManagerConfig(
        method=cast(ManagerMethod, method),
        learning_rate=args.manager_learning_rate,
        discount=args.manager_discount,
        exploration=args.manager_exploration,
        utility_decay=args.manager_utility_decay,
        cost_weight=args.manager_cost_weight,
        advantage_clip=args.manager_advantage_clip,
        ucb_bonus=args.manager_ucb_bonus,
        residual_power=args.manager_residual_power,
        novelty_power=args.manager_novelty_power,
        actual_gain_weight=args.manager_actual_gain_weight,
        total_center_budget=args.total_center_budget,
        allow_rebalance=args.allow_rebalance,
        rho_span=args.manager_rho_span,
        min_rho=args.manager_min_rho,
    )


def managed_method_name(method: str) -> str:
    """Return the stable result key for one manager method."""
    return f"multibank_resource_{method}"


def run_multibank_stream(
    observations: Any,
    targets: Any,
    bank_specs: tuple[BankSpec, ...],
    manager_config: ManagerConfig,
) -> tuple[ManagedMultiBankKRLS, MultiBankState, np.ndarray]:
    """Run one managed additive multi-bank learner."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = ManagedMultiBankKRLS(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        bank_specs=bank_specs,
        manager_config=manager_config,
    )
    state = learner.init()
    metrics = np.zeros((obs_np.shape[0], 27), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["post_loss"]
        metrics[idx, 3] = diagnostics["selected_bank"]
        metrics[idx, 4] = diagnostics["active_centers"]
        metrics[idx, 5] = diagnostics["additions"]
        metrics[idx, 6] = diagnostics["replacements"]
        metrics[idx, 7] = diagnostics["skipped_novel"]
        metrics[idx, 8] = diagnostics["finite_failures"]
        metrics[idx, 9] = diagnostics["actual_gain"]
        for bank_idx, name in enumerate(BANK_NAMES):
            base = 10 + bank_idx * 4
            metrics[idx, base] = diagnostics[f"{name}_centers"]
            metrics[idx, base + 1] = diagnostics[f"{name}_weight"]
            metrics[idx, base + 2] = diagnostics[f"{name}_utility"]
            metrics[idx, base + 3] = diagnostics[f"{name}_allocations"]
        metrics[idx, 22] = diagnostics["pre_loss"]
        metrics[idx, 23] = diagnostics["raw_poly_rho"]
        metrics[idx, 24] = diagnostics["algebraic_green_rho"]
        metrics[idx, 25] = diagnostics["arccosine_rho"]
        metrics[idx, 26] = diagnostics["resource_transfers"]
    return learner, state, metrics


def evaluate_multibank_classifier(
    learner: ManagedMultiBankKRLS,
    state: MultiBankState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final additive multi-bank classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def summarize_multibank(
    metrics: np.ndarray,
    final_window: int,
    labels: np.ndarray | None,
    state: MultiBankState,
) -> dict[str, float]:
    """Summarize one multi-bank run with manager and compute diagnostics."""
    summary = summarize_prequential(metrics, final_window, labels)
    window = min(final_window, metrics.shape[0])
    manager_state = state.manager_state
    summary.update(
        {
            "post_update_final_window_mse": float(np.mean(metrics[-window:, 2])),
            "active_centers": float(sum(s.active_count for s in state.bank_states)),
            "additions": float(sum(s.additions for s in state.bank_states)),
            "replacements": float(sum(s.replacements for s in state.bank_states)),
            "skipped_novel": float(sum(s.skipped_novel for s in state.bank_states)),
            "finite_failures": float(sum(s.finite_failures for s in state.bank_states)),
            "mean_actual_gain": float(np.mean(metrics[:, 9])),
            "final_raw_poly_weight": float(metrics[-1, 11]),
            "final_algebraic_green_weight": float(metrics[-1, 15]),
            "final_arccosine_weight": float(metrics[-1, 19]),
            "raw_poly_centers": float(state.bank_states[0].active_count),
            "algebraic_green_centers": float(state.bank_states[1].active_count),
            "arccosine_centers": float(state.bank_states[2].active_count),
            "raw_poly_allocations": float(manager_state.allocations[0]),
            "algebraic_green_allocations": float(manager_state.allocations[1]),
            "arccosine_allocations": float(manager_state.allocations[2]),
            "resource_transfers": float(manager_state.resource_transfers),
            "denied_allocations": float(manager_state.denied_allocations),
            "no_eligible_steps": float(manager_state.no_eligible_steps),
            "final_raw_poly_rho": float(metrics[-1, 23]),
            "final_algebraic_green_rho": float(metrics[-1, 24]),
            "final_arccosine_rho": float(metrics[-1, 25]),
        }
    )
    return summary


def parse_manager_methods(spec: str) -> tuple[str, ...]:
    """Parse and validate manager-method names."""
    methods = tuple(item.strip() for item in spec.split(",") if item.strip())
    unknown = sorted(set(methods).difference(MANAGER_METHODS))
    if unknown:
        raise ValueError(f"unknown manager methods {unknown}; valid: {MANAGER_METHODS}")
    return methods


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    bank_specs: tuple[BankSpec, ...],
    manager_methods: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all baselines and managed learners for one paired dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    for method in MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp_learner = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        mlp_state, metrics = run_mlp_stream(
            mlp_learner,
            observations,
            targets,
            jr.key(seed + 30_000 + MLP_METHODS.index(method)),
        )
        methods[method] = summarize_prequential(metrics, args.final_window, labels)
        methods[method]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_mlp_classifier(mlp_learner, mlp_state, x_test, y_test)
            )

    for manager_method in manager_methods:
        method_name = managed_method_name(manager_method)
        print(f"{dataset_name} seed={seed}: running {method_name}")
        t0 = time.time()
        multibank_learner, multibank_state, metrics = run_multibank_stream(
            observations,
            targets,
            bank_specs=bank_specs,
            manager_config=make_manager_config(manager_method, args),
        )
        methods[method_name] = summarize_multibank(
            metrics,
            args.final_window,
            labels,
            multibank_state,
        )
        methods[method_name]["runtime_s"] = float(time.time() - t0)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method_name].update(
                evaluate_multibank_classifier(
                    multibank_learner,
                    multibank_state,
                    x_test,
                    y_test,
                )
            )

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "methods": methods,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def aggregate_manager_vs_mlp(
    records: list[dict[str, Any]],
    manager_methods: tuple[str, ...],
) -> dict[str, Any]:
    """Add compact manager-vs-best-MLP comparisons for report text."""
    aggregate: dict[str, Any] = {}
    manager_method_names = tuple(managed_method_name(method) for method in manager_methods)
    for dataset in sorted({record["dataset_name"] for record in records}):
        dataset_records = [record for record in records if record["dataset_name"] == dataset]
        metric_names = [
            "final_window_mse",
            "online_mean_mse",
            "test_mse",
            "final_window_accuracy",
            "online_mean_accuracy",
            "test_accuracy",
        ]
        rows: dict[str, Any] = {}
        for metric in metric_names:
            if metric not in dataset_records[0]["methods"][manager_method_names[0]]:
                continue
            rows[metric] = {
                method: compare_to_group(dataset_records, method, metric, MLP_METHODS)
                for method in manager_method_names
            }
            best_diffs: list[float] = []
            best_manager_names: list[str] = []
            for record in dataset_records:
                methods = record["methods"]
                if metric.endswith("accuracy"):
                    best_manager = max(
                        manager_method_names,
                        key=lambda name: methods[name][metric],
                    )
                    best_mlp = max(MLP_METHODS, key=lambda name: methods[name][metric])
                else:
                    best_manager = min(
                        manager_method_names,
                        key=lambda name: methods[name][metric],
                    )
                    best_mlp = min(MLP_METHODS, key=lambda name: methods[name][metric])
                best_manager_names.append(best_manager)
                best_diffs.append(
                    paired_diff(
                        float(methods[best_manager][metric]),
                        float(methods[best_mlp][metric]),
                        metric,
                    )
                )
            diff_arr = np.asarray(best_diffs, dtype=np.float64)
            rows[metric]["best_manager_vs_best_mlp"] = {
                "paired_diff_mean_positive_favors_manager": float(np.mean(diff_arr)),
                "paired_diff_stderr": stderr(diff_arr),
                "wins_for_manager": int(np.sum(diff_arr > 0.0)),
                "wins_for_mlp": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n": int(diff_arr.shape[0]),
                "diffs": diff_arr.tolist(),
                "best_manager_counts": dict(
                    sorted(
                        (name, best_manager_names.count(name))
                        for name in set(best_manager_names)
                    )
                ),
            }
        aggregate[dataset] = rows
    return aggregate


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.4f} +/- {row[metric]['stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a detailed Markdown assessment for the learned manager."""
    cfg = results["config"]
    lines = [
        "# D10 Learned Kernel Resource Manager",
        "",
        "## Algorithm",
        "",
        (
            "The learner is one additive predictor, not a router: "
            "`prediction = raw_poly + algebraic_green + arccosine`.  Every bank "
            "updates its active coefficients at every time step against the same "
            "global residual.  The learned manager controls only scarce center "
            "allocation and replacement opportunities."
        ),
        "",
        (
            "The manager receives online evidence from ALD novelty, current "
            "residual loss, active-budget pressure, configured resource cost, and "
            "same-sample loss reduction after the update.  The `learned_softmax` "
            "variant performs a cost-sensitive exponentiated-gradient update; "
            "`novelty_greedy` and `round_robin` are allocation ablations."
        ),
        "",
        (
            "The center budget is global.  When the shared budget is full, a "
            "learned allocation may transfer one center from the currently least "
            "useful active bank into the selected bank.  The same learned weights "
            "also allocate a small per-bank RLS forgetting adjustment: high-weight "
            "banks retain slightly longer memory, and low-weight banks forget "
            "slightly faster.  Prediction is still additive; no expert output is "
            "selected or gated."
        ),
        "",
        "## Protocol",
        "",
        (
            f"Datasets: {', '.join(cfg['datasets'])}.  Seeds: {cfg['n_seeds']} "
            f"paired seeds from {cfg['seed']}.  Steps: {cfg['steps']}; final "
            f"window: {cfg['final_window']}.  MLP baselines: "
            f"{', '.join(results['mlp_methods'])}."
        ),
        "",
        (
            "Bank budgets: raw polynomial "
            f"{cfg['raw_poly_budget']}, algebraic-Green {cfg['algebraic_budget']}, "
            f"arc-cosine {cfg['arccosine_budget']}; shared budget "
            f"{cfg['total_center_budget']}; rebalance={cfg['allow_rebalance']}.  "
            f"Center intervals: raw {cfg['raw_poly_add_interval']}, algebraic "
            f"{cfg['algebraic_add_interval']}, arc {cfg['arccosine_add_interval']}.  "
            f"Manager rho span {cfg['manager_rho_span']} with minimum "
            f"{cfg['manager_min_rho']}."
        ),
        "",
    ]
    manager_method_names = [managed_method_name(method) for method in cfg["manager_methods"]]
    learned_failures: list[str] = []
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Test MSE | Final Acc | Test Acc | "
                "Centers | Runtime s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in dataset_agg.items():
            if method == "comparisons":
                continue
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'test_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} | "
                f"{metric_cell(row, 'active_centers')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        manager_comparisons = results["manager_vs_mlp"].get(dataset, {})
        for metric in ("final_window_mse", "test_accuracy", "test_mse"):
            if metric not in manager_comparisons:
                continue
            best = manager_comparisons[metric]["best_manager_vs_best_mlp"]
            lines.append(
                f"`{metric}` best-manager-vs-best-MLP diff: "
                f"{best['paired_diff_mean_positive_favors_manager']:+.4f} +/- "
                f"{best['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{best['wins_for_manager']}/{best['wins_for_mlp']}/{best['ties']}; "
                f"best-manager counts {best['best_manager_counts']}."
            )
            learned = manager_comparisons[metric].get(
                managed_method_name("learned_softmax")
            )
            if learned is not None:
                lines.append(
                    f"`{metric}` learned-softmax-vs-best-MLP diff: "
                    f"{learned['paired_diff_mean_positive_favors_method']:+.4f} +/- "
                    f"{learned['paired_diff_stderr']:.4f}; wins/losses/ties "
                    f"{learned['wins_for_method']}/{learned['wins_for_baseline']}/"
                    f"{learned['ties']}."
                )
                if learned["wins_for_baseline"] > 0:
                    learned_failures.append(f"{dataset}:{metric}")
        lines.append("")
        for method in manager_method_names:
            row = dataset_agg.get(method)
            if row is None:
                continue
            runtime_ratio = ""
            if "runtime_s" in row:
                manager_runtime = row["runtime_s"]["mean"]
                mlp_runtimes = [
                    dataset_agg[mlp]["runtime_s"]["mean"]
                    for mlp in MLP_METHODS
                    if mlp in dataset_agg and "runtime_s" in dataset_agg[mlp]
                ]
                if mlp_runtimes:
                    runtime_ratio = (
                        f"; runtime ratio vs fastest MLP "
                        f"{manager_runtime / min(mlp_runtimes):.2f}x"
                    )
            lines.append(
                f"`{method}` final allocation: raw "
                f"{metric_cell(row, 'raw_poly_allocations')}, algebraic "
                f"{metric_cell(row, 'algebraic_green_allocations')}, arc "
                f"{metric_cell(row, 'arccosine_allocations')}; final weights raw "
                f"{metric_cell(row, 'final_raw_poly_weight')}, algebraic "
                f"{metric_cell(row, 'final_algebraic_green_weight')}, arc "
                f"{metric_cell(row, 'final_arccosine_weight')}; final rho raw "
                f"{metric_cell(row, 'final_raw_poly_rho')}, algebraic "
                f"{metric_cell(row, 'final_algebraic_green_rho')}, arc "
                f"{metric_cell(row, 'final_arccosine_rho')}; transfers "
                f"{metric_cell(row, 'resource_transfers')}{runtime_ratio}."
            )
        lines.append("")
    if learned_failures:
        verdict = (
            "The learned manager does not close the full blocker suite under "
            "this run.  Remaining learned-softmax losses: "
            f"{', '.join(learned_failures)}."
        )
    else:
        verdict = (
            "The learned manager beat or tied the best fair MLP on every reported "
            "primary metric in this run.  This is promotable only after rerunning "
            "the same fixed configuration at larger seed count."
        )
    lines.extend(
        [
            "## Assessment",
            "",
            verdict,
            "",
            (
                "This closes the specific implementation gap of a learned "
                "resource manager only if the learned allocation beats or matches "
                "the allocation ablations while also beating the best fair MLP on "
                "the blocker metrics.  A per-dataset best manager result is "
                "evidence of headroom; the canonical Step 2 claim still requires "
                "one fixed promoted configuration run without per-dataset "
                "selection."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rare-period", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--phase-length", type=int, default=400)
    parser.add_argument("--mask-keep-fraction", type=float, default=0.5)
    parser.add_argument("--mask-noise-std", type=float, default=0.05)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rls-delta", type=float, default=100.0)
    parser.add_argument("--kernel-utility-decay", type=float, default=0.99)
    parser.add_argument("--min-center-age", type=int, default=50)
    parser.add_argument("--input-clip", type=float, default=5.0)
    parser.add_argument(
        "--bandwidth-multipliers",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
    )
    parser.add_argument("--kernel-weight-variance", type=float, default=2.0)
    parser.add_argument("--kernel-bias-variance", type=float, default=0.1)
    parser.add_argument("--kernel-lms-step-size", type=float, default=0.5)
    parser.add_argument("--replace-when-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--raw-poly-budget", type=int, default=64)
    parser.add_argument("--algebraic-budget", type=int, default=128)
    parser.add_argument("--arccosine-budget", type=int, default=128)
    parser.add_argument("--raw-poly-novelty", type=float, default=1e-5)
    parser.add_argument("--algebraic-novelty", type=float, default=1e-3)
    parser.add_argument("--arccosine-novelty", type=float, default=1e-3)
    parser.add_argument("--raw-poly-add-interval", type=int, default=1)
    parser.add_argument("--algebraic-add-interval", type=int, default=8)
    parser.add_argument("--arccosine-add-interval", type=int, default=2)
    parser.add_argument("--raw-poly-update-scale", type=float, default=0.55)
    parser.add_argument("--algebraic-update-scale", type=float, default=0.45)
    parser.add_argument("--arccosine-update-scale", type=float, default=0.45)
    parser.add_argument("--raw-poly-cost", type=float, default=1.0)
    parser.add_argument("--algebraic-cost", type=float, default=1.4)
    parser.add_argument("--arccosine-cost", type=float, default=1.8)
    parser.add_argument("--raw-poly-utility-scale", type=float, default=1.0)
    parser.add_argument("--algebraic-utility-scale", type=float, default=1.05)
    parser.add_argument("--arccosine-utility-scale", type=float, default=1.15)
    parser.add_argument("--algebraic-weight", type=float, default=0.75)
    parser.add_argument("--arccosine-depth", type=int, default=1)
    parser.add_argument(
        "--total-center-budget",
        type=int,
        default=320,
        help=(
            "Global center budget shared by all banks.  This must be smaller "
            "than the sum of per-bank budgets for allocation to matter."
        ),
    )
    parser.add_argument(
        "--allow-rebalance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow learned cross-bank center transfers after the global budget is full.",
    )
    parser.add_argument("--manager-methods", default="learned_softmax")
    parser.add_argument("--manager-learning-rate", type=float, default=4.0)
    parser.add_argument("--manager-discount", type=float, default=0.995)
    parser.add_argument("--manager-exploration", type=float, default=0.05)
    parser.add_argument("--manager-utility-decay", type=float, default=0.98)
    parser.add_argument("--manager-cost-weight", type=float, default=0.01)
    parser.add_argument("--manager-advantage-clip", type=float, default=5.0)
    parser.add_argument("--manager-ucb-bonus", type=float, default=0.02)
    parser.add_argument("--manager-residual-power", type=float, default=1.0)
    parser.add_argument("--manager-novelty-power", type=float, default=1.0)
    parser.add_argument("--manager-actual-gain-weight", type=float, default=2.0)
    parser.add_argument("--manager-rho-span", type=float, default=0.0)
    parser.add_argument("--manager-min-rho", type=float, default=0.97)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments with explicit failures."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    for name in ("raw_poly_budget", "algebraic_budget", "arccosine_budget"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.total_center_budget <= 0:
        raise ValueError("--total-center-budget must be positive")
    per_bank_budget = (
        args.raw_poly_budget + args.algebraic_budget + args.arccosine_budget
    )
    if args.total_center_budget > per_bank_budget:
        raise ValueError("--total-center-budget cannot exceed summed per-bank budgets")
    for name in ("raw_poly_add_interval", "algebraic_add_interval", "arccosine_add_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.manager_exploration < 1.0:
        raise ValueError("--manager-exploration must be in [0, 1)")
    if not 0.0 <= args.manager_discount <= 1.0:
        raise ValueError("--manager-discount must be in [0, 1]")
    if not 0.0 <= args.manager_utility_decay < 1.0:
        raise ValueError("--manager-utility-decay must be in [0, 1)")
    if args.manager_cost_weight < 0.0:
        raise ValueError("--manager-cost-weight must be non-negative")
    if args.manager_advantage_clip <= 0.0:
        raise ValueError("--manager-advantage-clip must be positive")
    if args.manager_rho_span < 0.0:
        raise ValueError("--manager-rho-span must be non-negative")
    if not 0.0 < args.manager_min_rho < 1.0:
        raise ValueError("--manager-min-rho must be in (0, 1)")
    if not 0.0 <= args.algebraic_weight <= 1.0:
        raise ValueError("--algebraic-weight must be in [0, 1]")
    if args.arccosine_depth <= 0:
        raise ValueError("--arccosine-depth must be positive")


def main() -> None:
    """Run the learned resource-manager experiment."""
    args = parse_args()
    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.datasets = "controlled_nonlinear"
        args.raw_poly_budget = 12
        args.algebraic_budget = 12
        args.arccosine_budget = 12
        args.total_center_budget = 24
        args.manager_methods = "learned_softmax,round_robin"
    validate_args(args)
    datasets = expand_d07_dataset_names(args.datasets)
    manager_methods = parse_manager_methods(args.manager_methods)
    bank_specs = make_bank_specs(args)
    candidate_methods = tuple(managed_method_name(method) for method in manager_methods)
    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name=dataset_name,
                seed=seed,
                bank_specs=bank_specs,
                manager_methods=manager_methods,
                args=args,
            )
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta
    results = {
        "config": {
            "datasets": datasets,
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "final_window": args.final_window,
            "feature_dim": args.feature_dim,
            "noise_std": args.noise_std,
            "rare_period": args.rare_period,
            "train_fraction": args.train_fraction,
            "phase_length": args.phase_length,
            "mask_keep_fraction": args.mask_keep_fraction,
            "mask_noise_std": args.mask_noise_std,
            "mlp_step_size": args.mlp_step_size,
            "mlp_sparsity": args.mlp_sparsity,
            "rho": args.rho,
            "ridge": args.ridge,
            "rls_delta": args.rls_delta,
            "raw_poly_budget": args.raw_poly_budget,
            "algebraic_budget": args.algebraic_budget,
            "arccosine_budget": args.arccosine_budget,
            "total_center_budget": args.total_center_budget,
            "allow_rebalance": args.allow_rebalance,
            "raw_poly_add_interval": args.raw_poly_add_interval,
            "algebraic_add_interval": args.algebraic_add_interval,
            "arccosine_add_interval": args.arccosine_add_interval,
            "raw_poly_update_scale": args.raw_poly_update_scale,
            "algebraic_update_scale": args.algebraic_update_scale,
            "arccosine_update_scale": args.arccosine_update_scale,
            "algebraic_weight": args.algebraic_weight,
            "arccosine_depth": args.arccosine_depth,
            "manager_methods": list(manager_methods),
            "manager_learning_rate": args.manager_learning_rate,
            "manager_discount": args.manager_discount,
            "manager_exploration": args.manager_exploration,
            "manager_cost_weight": args.manager_cost_weight,
            "manager_ucb_bonus": args.manager_ucb_bonus,
            "manager_actual_gain_weight": args.manager_actual_gain_weight,
            "manager_rho_span": args.manager_rho_span,
            "manager_min_rho": args.manager_min_rho,
        },
        "datasets": datasets_meta,
        "bank_specs": [
            {
                "name": spec.name,
                "config": spec.config.__dict__,
                "update_scale": spec.update_scale,
                "resource_cost": spec.resource_cost,
                "utility_scale": spec.utility_scale,
            }
            for spec in bank_specs
        ],
        "manager_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_records(records, candidate_methods),
        "manager_vs_mlp": aggregate_manager_vs_mlp(records, manager_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "learned_center_allocation_for_additive_multibank_rkhs",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    if args.note_path:
        write_summary(args.note_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.note_path:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

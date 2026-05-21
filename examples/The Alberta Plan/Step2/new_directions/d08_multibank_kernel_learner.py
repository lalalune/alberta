#!/usr/bin/env python3
"""D08: additive multi-bank RKHS learner for Step 2 blockers.

The D07 result left a concrete conflict:

* raw degree-3 polynomial RKHS solves frequency-like controlled tasks;
* normalized algebraic-Green RKHS with throttled allocation solves stateful
  digit retention/adaptation tasks;
* tanh-compositional streams need a neural/compositional basis.

This runner tests the direct next hypothesis: one single predictor with several
complementary feature banks and one shared online loss.  It is not a router,
stacker, or MLP residual.  Each bank generates features at every time step, all
active features are concatenated, and one output coefficient matrix is updated
from the additive prediction error.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
STEP2_DIR = Path(__file__).resolve().parents[1]
NEW_DIRECTIONS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, STEP2_DIR, NEW_DIRECTIONS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from d07_budgeted_kernel_recursive import (  # noqa: E402
    CONTROLLED_DATASETS,
    MLP_METHODS,
    KernelConfig,
    BudgetedDiffusionKRLS,
    aggregate_records as aggregate_candidate_records,
    evaluate_mlp_classifier,
    expand_dataset_names as expand_d07_dataset_names,
    kernel_method_name,
    make_dataset,
    make_mlp,
    masked_mse_np,
    metric_cell,
    run_mlp_stream,
    summarize_prequential,
)
from step2_expert_mixture import DIGITS_REGIMES, SYNTHETIC_REGIMES  # noqa: E402

N_DIGIT_CLASSES = 10
DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/d08_multibank_kernel_learner")
DEFAULT_NOTE_PATH = Path("docs/research/step2_new_directions/d08_multibank_kernel_learner.md")
BLOCKER_DATASETS = (
    "controlled_frequency",
    "synthetic_compositional",
    "synthetic_frequency",
    "digits_label_drift",
    "digits_mask_noise",
    "digits_permuted_pixels",
)
VALID_DATASETS = (*CONTROLLED_DATASETS, *SYNTHETIC_REGIMES, *DIGITS_REGIMES)

CoefficientUpdate = Literal[
    "rls",
    "nlms",
    "bank_nlms",
    "bank_hybrid",
    "bank_hybrid_seq",
]


@dataclass(frozen=True)
class DictionaryBankConfig:
    """Configuration for one ALD dictionary kernel bank."""

    name: str
    kernel_config: KernelConfig
    feature_scale: float = 1.0
    step_size: float = 0.5


@dataclass(frozen=True)
class RandomTanhBankConfig:
    """Configuration for a fixed random tanh RKHS feature bank."""

    name: str
    width: int
    weight_scale: float
    feature_scale: float
    step_size: float
    input_clip: float
    seed_offset: int
    include_bias: bool = True


@dataclass(frozen=True)
class FourierBankConfig:
    """Configuration for a fixed sin/cos RKHS feature bank."""

    name: str
    max_input_dim: int
    frequencies: tuple[float, ...]
    feature_scale: float
    step_size: float
    input_clip: float
    include_bias: bool = True
    include_raw: bool = True


BankConfig = DictionaryBankConfig | RandomTanhBankConfig | FourierBankConfig


@dataclass(frozen=True)
class MultiBankConfig:
    """Configuration for one single additive multi-bank learner."""

    name: str
    banks: tuple[BankConfig, ...]
    coefficient_update: CoefficientUpdate
    rho: float
    rls_delta: float
    nlms_step_size: float
    output_clip: float


@dataclass
class DictionaryBankState:
    """Runtime state for one mutable dictionary bank."""

    config: DictionaryBankConfig
    learner: BudgetedDiffusionKRLS
    offset: int
    centers: np.ndarray
    k_inv: np.ndarray
    activation_ema: np.ndarray
    coefficient_ema: np.ndarray
    ages: np.ndarray
    active_count: int
    additions: int
    replacements: int
    skipped_novel: int
    throttled_novel: int
    finite_failures: int
    novelty_sum: float
    last_center_step: int


@dataclass
class RandomTanhBankState:
    """Runtime state for one fixed random tanh feature bank."""

    config: RandomTanhBankConfig
    offset: int
    feature_weights: np.ndarray
    feature_biases: np.ndarray
    activation_ema: np.ndarray


@dataclass
class FourierBankState:
    """Runtime state for one fixed Fourier feature bank."""

    config: FourierBankConfig
    offset: int
    active_width: int
    activation_ema: np.ndarray


RuntimeBankState = DictionaryBankState | RandomTanhBankState | FourierBankState


@dataclass
class MultiBankState:
    """Mutable state for a single additive multi-bank predictor."""

    banks: list[RuntimeBankState]
    alpha: np.ndarray
    p_matrix: np.ndarray
    active_mask: np.ndarray
    leverage_sum: float
    finite_failures: int
    steps: int


def stderr(values: np.ndarray) -> float:
    """Return standard error for a one-dimensional array."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def scalar_token(value: float | int | str) -> str:
    """Return a compact token safe for method names."""
    return str(value).replace("-", "m").replace(".", "p")


def bank_width(config: BankConfig) -> int:
    """Return the maximum feature width contributed by one bank."""
    if isinstance(config, DictionaryBankConfig):
        return int(config.kernel_config.budget)
    if isinstance(config, RandomTanhBankConfig):
        return int(config.width + int(config.include_bias))
    d = int(config.max_input_dim)
    return int(
        int(config.include_bias)
        + int(config.include_raw) * d
        + 2 * d * len(config.frequencies)
    )


def dictionary_config(
    *,
    budget: int,
    kernel: str,
    sigma: float,
    novelty_threshold: float,
    center_add_interval: int,
    normalize_polynomial: bool,
    algebraic_weight: float,
    polynomial_degree: int,
    arccosine_depth: int,
    replace_when_full: bool = False,
) -> KernelConfig:
    """Build the D07 kernel config used by one D08 dictionary bank."""
    return KernelConfig(
        budget=int(budget),
        sigma=float(sigma),
        rho=0.99,
        novelty_threshold=float(novelty_threshold),
        ridge=1e-3,
        rls_delta=100.0,
        utility_decay=0.99,
        min_center_age=50,
        input_clip=5.0,
        kernel=kernel,
        bandwidth_multipliers=(0.5, 1.0, 2.0),
        polynomial_degree=int(polynomial_degree),
        algebraic_weight=float(algebraic_weight),
        normalize_polynomial=bool(normalize_polynomial),
        arccosine_depth=int(arccosine_depth),
        kernel_weight_variance=2.0,
        kernel_bias_variance=0.1,
        coefficient_update="rls",
        lms_step_size=0.5,
        replace_when_full=bool(replace_when_full),
        center_add_interval=int(center_add_interval),
    )


def make_preset_configs(args: argparse.Namespace) -> list[MultiBankConfig]:
    """Expand CLI preset names into concrete multi-bank configurations."""
    requested = tuple(item.strip() for item in args.configs.split(",") if item.strip())
    if not requested:
        raise ValueError("--configs must name at least one preset")

    valid = {
        "canonical_bank_nlms",
        "canonical_hybrid",
        "canonical_hybrid_seq",
        "canonical_full_seq",
        "canonical_green_first_full_seq",
        "canonical_tanh_first_seq",
        "canonical_nlms",
        "canonical_rls",
        "compact_bank_nlms",
        "compact_hybrid",
        "compact_hybrid_seq",
        "compact_full_seq",
        "compact_green_first_full_seq",
        "compact_tanh_first_seq",
        "compact_rls",
        "arc_rls",
        "tanh_only_nlms",
    }
    unknown = sorted(set(requested).difference(valid))
    if unknown:
        raise ValueError(f"unknown --configs entries {unknown}; valid: {sorted(valid)}")

    def raw_poly_bank(budget: int) -> DictionaryBankConfig:
        return DictionaryBankConfig(
            name="raw_poly_d3",
            kernel_config=dictionary_config(
                budget=budget,
                kernel="polynomial",
                sigma=1.0,
                novelty_threshold=1e-3,
                center_add_interval=1,
                normalize_polynomial=False,
                algebraic_weight=0.5,
                polynomial_degree=3,
                arccosine_depth=1,
            ),
            feature_scale=float(args.raw_poly_feature_scale),
            step_size=float(args.raw_poly_step_size),
        )

    def algebraic_green_bank(budget: int, interval: int) -> DictionaryBankConfig:
        return DictionaryBankConfig(
            name="algebraic_green_memory",
            kernel_config=dictionary_config(
                budget=budget,
                kernel="algebraic_green",
                sigma=float(args.green_sigma),
                novelty_threshold=1e-3,
                center_add_interval=interval,
                normalize_polynomial=True,
                algebraic_weight=0.75,
                polynomial_degree=3,
                arccosine_depth=1,
            ),
            feature_scale=float(args.green_feature_scale),
            step_size=float(args.green_step_size),
        )

    def arccosine_bank(budget: int, interval: int) -> DictionaryBankConfig:
        return DictionaryBankConfig(
            name="arccosine_nngp",
            kernel_config=dictionary_config(
                budget=budget,
                kernel="arccosine",
                sigma=1.0,
                novelty_threshold=0.05,
                center_add_interval=interval,
                normalize_polynomial=True,
                algebraic_weight=0.5,
                polynomial_degree=3,
                arccosine_depth=int(args.arccosine_depth),
            ),
            feature_scale=float(args.arccosine_feature_scale),
            step_size=float(args.arccosine_step_size),
        )

    def tanh_bank(width: int) -> RandomTanhBankConfig:
        return RandomTanhBankConfig(
            name="random_tanh_compositional",
            width=width,
            weight_scale=float(args.tanh_weight_scale),
            feature_scale=float(args.tanh_feature_scale),
            step_size=float(args.tanh_step_size),
            input_clip=float(args.tanh_input_clip),
            seed_offset=70_000,
            include_bias=True,
        )

    def fourier_bank() -> FourierBankConfig:
        return FourierBankConfig(
            name="fourier_frequency",
            max_input_dim=int(args.fourier_max_input_dim),
            frequencies=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            feature_scale=float(args.fourier_feature_scale),
            step_size=float(args.fourier_step_size),
            input_clip=float(args.fourier_input_clip),
            include_bias=True,
            include_raw=True,
        )

    configs: list[MultiBankConfig] = []
    for name in requested:
        if name == "canonical_bank_nlms":
            banks: tuple[BankConfig, ...] = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_nlms",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_nlms":
            banks = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="nlms",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_hybrid":
            banks = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_hybrid_seq":
            banks = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_tanh_first_seq":
            banks = (
                tanh_bank(args.tanh_width),
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_full_seq":
            banks = (
                tanh_bank(args.tanh_width),
                fourier_bank(),
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_green_first_full_seq":
            banks = (
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
                fourier_bank(),
                raw_poly_bank(args.raw_poly_budget),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "canonical_rls":
            banks = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                tanh_bank(args.tanh_width),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="rls",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_bank_nlms":
            banks = (
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
                tanh_bank(max(64, args.tanh_width // 2)),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_nlms",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_hybrid":
            banks = (
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
                tanh_bank(max(64, args.tanh_width // 2)),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_hybrid_seq":
            banks = (
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
                tanh_bank(max(64, args.tanh_width // 2)),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_tanh_first_seq":
            banks = (
                tanh_bank(max(64, args.tanh_width // 2)),
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_full_seq":
            banks = (
                tanh_bank(max(64, args.tanh_width // 2)),
                fourier_bank(),
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_green_first_full_seq":
            banks = (
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
                tanh_bank(max(64, args.tanh_width // 2)),
                fourier_bank(),
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="bank_hybrid_seq",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "compact_rls":
            banks = (
                raw_poly_bank(max(16, args.raw_poly_budget // 2)),
                algebraic_green_bank(max(48, args.green_budget // 2), args.green_center_interval),
                tanh_bank(max(64, args.tanh_width // 2)),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="rls",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "arc_rls":
            banks = (
                raw_poly_bank(args.raw_poly_budget),
                algebraic_green_bank(args.green_budget, args.green_center_interval),
                arccosine_bank(args.arc_budget, args.arc_center_interval),
            )
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="rls",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
        elif name == "tanh_only_nlms":
            banks = (tanh_bank(args.tanh_width),)
            configs.append(
                MultiBankConfig(
                    name=name,
                    banks=banks,
                    coefficient_update="nlms",
                    rho=float(args.rho),
                    rls_delta=float(args.rls_delta),
                    nlms_step_size=float(args.nlms_step_size),
                    output_clip=float(args.output_clip),
                )
            )
    return configs


def multibank_method_name(config: MultiBankConfig) -> str:
    """Return a stable compact name for one D08 learner preset."""
    widths = "_".join(f"{bank.name}{bank_width(bank)}" for bank in config.banks)
    update = config.coefficient_update
    if update == "rls":
        suffix = f"rho{scalar_token(config.rho)}_delta{scalar_token(config.rls_delta)}"
    elif update == "bank_nlms":
        suffix = "blocknorm"
    elif update == "bank_hybrid":
        suffix = f"localrls_rho{scalar_token(config.rho)}"
    elif update == "bank_hybrid_seq":
        suffix = f"seqrls_rho{scalar_token(config.rho)}"
    else:
        suffix = f"eta{scalar_token(config.nlms_step_size)}"
    return f"multibank_{config.name}_{update}_{suffix}_{widths}"


class MultiBankRKHSLearner:
    """Single additive predictor over several complementary feature banks."""

    def __init__(
        self,
        n_heads: int,
        feature_dim: int,
        config: MultiBankConfig,
    ) -> None:
        self.n_heads = int(n_heads)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.total_width = sum(bank_width(bank) for bank in config.banks)

    def _fourier_active_width(self, config: FourierBankConfig) -> int:
        """Return the realized Fourier width for this input dimension."""
        d = min(self.feature_dim, int(config.max_input_dim))
        return int(
            int(config.include_bias)
            + int(config.include_raw) * d
            + 2 * d * len(config.frequencies)
        )

    def init(self, seed: int) -> MultiBankState:
        """Initialize bank dictionaries, fixed features, and output weights."""
        banks: list[RuntimeBankState] = []
        offset = 0
        for bank_config in self.config.banks:
            if isinstance(bank_config, DictionaryBankConfig):
                budget = bank_config.kernel_config.budget
                learner = BudgetedDiffusionKRLS(
                    n_heads=self.n_heads,
                    feature_dim=self.feature_dim,
                    config=bank_config.kernel_config,
                )
                banks.append(
                    DictionaryBankState(
                        config=bank_config,
                        learner=learner,
                        offset=offset,
                        centers=np.zeros((budget, self.feature_dim), dtype=np.float64),
                        k_inv=np.zeros((budget, budget), dtype=np.float64),
                        activation_ema=np.zeros(budget, dtype=np.float64),
                        coefficient_ema=np.zeros(budget, dtype=np.float64),
                        ages=np.zeros(budget, dtype=np.int64),
                        active_count=0,
                        additions=0,
                        replacements=0,
                        skipped_novel=0,
                        throttled_novel=0,
                        finite_failures=0,
                        novelty_sum=0.0,
                        last_center_step=-bank_config.kernel_config.center_add_interval,
                    )
                )
            elif isinstance(bank_config, RandomTanhBankConfig):
                rng = np.random.default_rng(seed + bank_config.seed_offset)
                scale = bank_config.weight_scale / math.sqrt(max(self.feature_dim, 1))
                weights = scale * rng.standard_normal(
                    (bank_config.width, self.feature_dim)
                )
                biases = rng.uniform(
                    -bank_config.weight_scale,
                    bank_config.weight_scale,
                    size=bank_config.width,
                )
                width = bank_width(bank_config)
                banks.append(
                    RandomTanhBankState(
                        config=bank_config,
                        offset=offset,
                        feature_weights=weights.astype(np.float64),
                        feature_biases=biases.astype(np.float64),
                        activation_ema=np.zeros(width, dtype=np.float64),
                    )
                )
            else:
                active_width = self._fourier_active_width(bank_config)
                banks.append(
                    FourierBankState(
                        config=bank_config,
                        offset=offset,
                        active_width=active_width,
                        activation_ema=np.zeros(bank_width(bank_config), dtype=np.float64),
                    )
                )
            offset += bank_width(bank_config)

        active_mask = np.zeros(self.total_width, dtype=bool)
        for bank in banks:
            if isinstance(bank, RandomTanhBankState):
                active_mask[bank.offset : bank.offset + bank_width(bank.config)] = True
            elif isinstance(bank, FourierBankState):
                active_mask[bank.offset : bank.offset + bank.active_width] = True

        return MultiBankState(
            banks=banks,
            alpha=np.zeros((self.total_width, self.n_heads), dtype=np.float64),
            p_matrix=np.eye(self.total_width, dtype=np.float64) * self.config.rls_delta,
            active_mask=active_mask,
            leverage_sum=0.0,
            finite_failures=0,
            steps=0,
        )

    def _rebuild_dictionary_inverse(self, bank: DictionaryBankState) -> None:
        """Recompute one bank's ALD inverse matrix."""
        m = bank.active_count
        bank.k_inv.fill(0.0)
        if m <= 0:
            return
        ridge = bank.config.kernel_config.ridge
        k_dd = bank.learner._kernel(bank.centers[:m], bank.centers[:m])
        k_dd = k_dd + ridge * np.eye(m, dtype=np.float64)
        try:
            bank.k_inv[:m, :m] = np.linalg.inv(k_dd)
        except np.linalg.LinAlgError:
            bank.k_inv[:m, :m] = np.linalg.pinv(k_dd)
            bank.finite_failures += 1

    def _dictionary_novelty(self, bank: DictionaryBankState, z: np.ndarray) -> float:
        """Return the ALD residual variance for a candidate center."""
        cfg = bank.config.kernel_config
        self_kernel = float(bank.learner._kernel(z, z).reshape(())) + cfg.ridge
        if bank.active_count == 0:
            return self_kernel
        phi = bank.learner._kernel(z, bank.centers[: bank.active_count]).reshape(
            bank.active_count
        )
        k_inv = bank.k_inv[: bank.active_count, : bank.active_count]
        residual = self_kernel - float(phi @ k_inv @ phi)
        return max(residual, 0.0)

    def _replacement_index(self, bank: DictionaryBankState) -> int:
        """Choose the weakest mature center in one dictionary."""
        m = bank.active_count
        score = bank.activation_ema[:m] + 0.05 * bank.coefficient_ema[:m]
        mature = bank.ages[:m] >= bank.config.kernel_config.min_center_age
        if np.any(mature):
            return int(np.argmin(np.where(mature, score, np.inf)))
        return int(np.argmin(score))

    def _reset_global_feature_slot(
        self,
        state: MultiBankState,
        global_index: int,
    ) -> None:
        """Reset one global coefficient/covariance slot after allocation."""
        state.alpha[global_index] = 0.0
        state.p_matrix[global_index, :] = 0.0
        state.p_matrix[:, global_index] = 0.0
        state.p_matrix[global_index, global_index] = self.config.rls_delta
        state.active_mask[global_index] = True

    def _maybe_add_dictionary_center(
        self,
        state: MultiBankState,
        bank: DictionaryBankState,
        z: np.ndarray,
        novelty: float,
    ) -> None:
        """Run one bank's ALD resource rule before coefficient update."""
        cfg = bank.config.kernel_config
        if novelty <= cfg.novelty_threshold:
            return
        if state.steps - bank.last_center_step < cfg.center_add_interval:
            bank.throttled_novel += 1
            return
        if bank.active_count < cfg.budget:
            idx = bank.active_count
            bank.active_count += 1
            bank.additions += 1
        elif cfg.replace_when_full:
            idx = self._replacement_index(bank)
            bank.replacements += 1
        else:
            bank.skipped_novel += 1
            return

        global_index = bank.offset + idx
        bank.centers[idx] = z
        bank.activation_ema[idx] = 0.0
        bank.coefficient_ema[idx] = 0.0
        bank.ages[idx] = 0
        bank.last_center_step = state.steps
        self._reset_global_feature_slot(state, global_index)
        self._rebuild_dictionary_inverse(bank)

    def _dictionary_features(
        self,
        bank: DictionaryBankState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return scaled active features for one dictionary bank."""
        features = np.zeros(bank_width(bank.config), dtype=np.float64)
        if bank.active_count == 0:
            return features
        x = np.asarray(observation, dtype=np.float64)
        input_clip = bank.config.kernel_config.input_clip
        if input_clip > 0.0:
            x = np.clip(x, -input_clip, input_clip)
        phi = bank.learner._kernel(x, bank.centers[: bank.active_count]).reshape(
            bank.active_count
        )
        features[: bank.active_count] = bank.config.feature_scale * phi
        return features

    def _random_tanh_features(
        self,
        bank: RandomTanhBankState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return scaled fixed random tanh features for one bank."""
        cfg = bank.config
        x = np.asarray(observation, dtype=np.float64)
        if cfg.input_clip > 0.0:
            x = np.clip(x, -cfg.input_clip, cfg.input_clip)
        hidden = np.tanh(bank.feature_weights @ x + bank.feature_biases)
        if cfg.include_bias:
            raw = np.concatenate([np.ones(1, dtype=np.float64), hidden])
        else:
            raw = hidden
        return cfg.feature_scale * raw

    def _fourier_features(
        self,
        bank: FourierBankState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return scaled fixed Fourier features for one bank."""
        cfg = bank.config
        width = bank_width(cfg)
        features = np.zeros(width, dtype=np.float64)
        x = np.asarray(observation, dtype=np.float64)[: cfg.max_input_dim]
        if cfg.input_clip > 0.0:
            x = np.clip(x, -cfg.input_clip, cfg.input_clip)
        terms: list[float] = []
        if cfg.include_bias:
            terms.append(1.0)
        if cfg.include_raw:
            terms.extend(float(value) for value in x)
        for value in x:
            for frequency in cfg.frequencies:
                z = float(frequency) * float(value)
                terms.append(math.sin(z))
                terms.append(math.cos(z))
        active = np.asarray(terms, dtype=np.float64)
        features[: active.shape[0]] = cfg.feature_scale * active
        return features

    def _feature_vector(
        self,
        state: MultiBankState,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return the current concatenated active feature vector."""
        features = np.zeros(self.total_width, dtype=np.float64)
        for bank in state.banks:
            if isinstance(bank, DictionaryBankState):
                bank_features = self._dictionary_features(bank, observation)
            elif isinstance(bank, RandomTanhBankState):
                bank_features = self._random_tanh_features(bank, observation)
            else:
                bank_features = self._fourier_features(bank, observation)
            width = bank_features.shape[0]
            features[bank.offset : bank.offset + width] = bank_features
        return features

    def _allocate_dictionary_features(
        self,
        state: MultiBankState,
        observation: np.ndarray,
    ) -> None:
        """Let each dictionary bank evaluate novelty and allocate if warranted."""
        for bank in state.banks:
            if not isinstance(bank, DictionaryBankState):
                continue
            x = np.asarray(observation, dtype=np.float64)
            input_clip = bank.config.kernel_config.input_clip
            if input_clip > 0.0:
                x = np.clip(x, -input_clip, input_clip)
            novelty = self._dictionary_novelty(bank, x)
            bank.novelty_sum += novelty
            self._maybe_add_dictionary_center(state, bank, x, novelty)

    def predict(self, state: MultiBankState, observation: np.ndarray) -> np.ndarray:
        """Predict all heads without mutating state."""
        features = self._feature_vector(state, observation)
        prediction = features @ state.alpha
        if self.config.output_clip > 0.0:
            prediction = np.clip(
                prediction,
                -self.config.output_clip,
                self.config.output_clip,
            )
        return np.asarray(prediction, dtype=np.float64)

    def _update_coefficients(
        self,
        state: MultiBankState,
        features: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Update the single output matrix from one additive prediction error."""
        active_target = ~np.isnan(target)
        if not np.any(active_target):
            return 0.0
        active_features = np.flatnonzero(state.active_mask)
        if active_features.size == 0:
            return 0.0

        phi = features[active_features]
        update_prediction = phi @ state.alpha[active_features]
        if self.config.output_clip > 0.0:
            update_prediction = np.clip(
                update_prediction,
                -self.config.output_clip,
                self.config.output_clip,
            )
        safe_target = np.where(active_target, target, 0.0)
        errors = np.where(active_target, safe_target - update_prediction, 0.0)

        if self.config.coefficient_update == "rls":
            p_active = state.p_matrix[np.ix_(active_features, active_features)]
            p_phi = p_active @ phi
            denom = self.config.rho + float(phi @ p_phi)
            if denom <= 1e-12 or not np.isfinite(denom):
                state.finite_failures += 1
                return 0.0
            gain = p_phi / denom
            state.alpha[active_features] += np.outer(gain, errors)
            next_p = (p_active - np.outer(gain, phi @ p_active)) / self.config.rho
            state.p_matrix[np.ix_(active_features, active_features)] = 0.5 * (
                next_p + next_p.T
            )
            return float(phi @ p_phi)

        if self.config.coefficient_update == "bank_nlms":
            leverage = 0.0
            for bank in state.banks:
                width = bank_width(bank.config)
                start = bank.offset
                stop = bank.offset + width
                local_mask = state.active_mask[start:stop]
                if not np.any(local_mask):
                    continue
                local_indices = np.flatnonzero(local_mask) + start
                local_phi = features[local_indices]
                normalizer = 1.0 + float(local_phi @ local_phi)
                state.alpha[local_indices] += (
                    bank.config.step_size
                    * np.outer(local_phi, errors)
                    / normalizer
                )
                leverage += float(local_phi @ local_phi)
            return leverage

        if self.config.coefficient_update == "bank_hybrid":
            leverage = 0.0
            for bank in state.banks:
                width = bank_width(bank.config)
                start = bank.offset
                stop = bank.offset + width
                local_mask = state.active_mask[start:stop]
                if not np.any(local_mask):
                    continue
                local_indices = np.flatnonzero(local_mask) + start
                local_phi = features[local_indices]
                if isinstance(bank, DictionaryBankState):
                    p_local = state.p_matrix[np.ix_(local_indices, local_indices)]
                    p_phi = p_local @ local_phi
                    denom = self.config.rho + float(local_phi @ p_phi)
                    if denom <= 1e-12 or not np.isfinite(denom):
                        state.finite_failures += 1
                        continue
                    gain = p_phi / denom
                    state.alpha[local_indices] += (
                        bank.config.step_size * np.outer(gain, errors)
                    )
                    next_p = (
                        p_local - np.outer(gain, local_phi @ p_local)
                    ) / self.config.rho
                    state.p_matrix[np.ix_(local_indices, local_indices)] = 0.5 * (
                        next_p + next_p.T
                    )
                    leverage += float(local_phi @ p_phi)
                else:
                    normalizer = 1.0 + float(local_phi @ local_phi)
                    state.alpha[local_indices] += (
                        bank.config.step_size
                        * np.outer(local_phi, errors)
                        / normalizer
                    )
                    leverage += float(local_phi @ local_phi)
            return leverage

        if self.config.coefficient_update == "bank_hybrid_seq":
            leverage = 0.0
            residual = errors.copy()
            for bank in state.banks:
                width = bank_width(bank.config)
                start = bank.offset
                stop = bank.offset + width
                local_mask = state.active_mask[start:stop]
                if not np.any(local_mask):
                    continue
                local_indices = np.flatnonzero(local_mask) + start
                local_phi = features[local_indices]
                if isinstance(bank, DictionaryBankState):
                    p_local = state.p_matrix[np.ix_(local_indices, local_indices)]
                    p_phi = p_local @ local_phi
                    denom = self.config.rho + float(local_phi @ p_phi)
                    if denom <= 1e-12 or not np.isfinite(denom):
                        state.finite_failures += 1
                        continue
                    gain = p_phi / denom
                    delta_alpha = bank.config.step_size * np.outer(gain, residual)
                    state.alpha[local_indices] += delta_alpha
                    next_p = (
                        p_local - np.outer(gain, local_phi @ p_local)
                    ) / self.config.rho
                    state.p_matrix[np.ix_(local_indices, local_indices)] = 0.5 * (
                        next_p + next_p.T
                    )
                    leverage += float(local_phi @ p_phi)
                else:
                    normalizer = 1.0 + float(local_phi @ local_phi)
                    delta_alpha = (
                        bank.config.step_size
                        * np.outer(local_phi, residual)
                        / normalizer
                    )
                    state.alpha[local_indices] += delta_alpha
                    leverage += float(local_phi @ local_phi)
                residual = residual - local_phi @ delta_alpha
                residual = np.where(active_target, residual, 0.0)
            return leverage

        normalizer = 1.0 + float(phi @ phi)
        step = self.config.nlms_step_size
        state.alpha[active_features] += step * np.outer(phi, errors) / normalizer
        return float(phi @ phi)

    def _update_bank_utilities(
        self,
        state: MultiBankState,
        features: np.ndarray,
    ) -> None:
        """Refresh bank-local utility traces used for diagnostics/replacement."""
        for bank in state.banks:
            width = bank_width(bank.config)
            sl = slice(bank.offset, bank.offset + width)
            local_features = np.abs(features[sl])
            if isinstance(bank, DictionaryBankState):
                m = bank.active_count
                if m <= 0:
                    continue
                decay = bank.config.kernel_config.utility_decay
                bank.activation_ema[:m] = (
                    decay * bank.activation_ema[:m]
                    + (1.0 - decay) * local_features[:m]
                )
                bank.coefficient_ema[:m] = (
                    decay * bank.coefficient_ema[:m]
                    + (1.0 - decay) * np.mean(np.abs(state.alpha[sl][:m]), axis=1)
                )
                bank.ages[:m] += 1
            else:
                decay = 0.99
                bank.activation_ema = (
                    decay * bank.activation_ema + (1.0 - decay) * local_features
                )

    def step(
        self,
        state: MultiBankState,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict, allocate bank resources, then update the shared predictor."""
        prediction = self.predict(state, observation)
        self._allocate_dictionary_features(state, observation)
        features = self._feature_vector(state, observation)
        leverage = self._update_coefficients(state, features, target)
        self._update_bank_utilities(state, features)

        if not np.all(np.isfinite(state.alpha)):
            state.finite_failures += 1
            state.alpha = np.nan_to_num(state.alpha, copy=False)
        state.leverage_sum += leverage
        state.steps += 1
        active_centers = sum(
            bank.active_count
            for bank in state.banks
            if isinstance(bank, DictionaryBankState)
        )
        additions = sum(
            bank.additions for bank in state.banks if isinstance(bank, DictionaryBankState)
        )
        replacements = sum(
            bank.replacements
            for bank in state.banks
            if isinstance(bank, DictionaryBankState)
        )
        throttled = sum(
            bank.throttled_novel
            for bank in state.banks
            if isinstance(bank, DictionaryBankState)
        )
        skipped = sum(
            bank.skipped_novel
            for bank in state.banks
            if isinstance(bank, DictionaryBankState)
        )
        finite_failures = state.finite_failures + sum(
            bank.finite_failures
            for bank in state.banks
            if isinstance(bank, DictionaryBankState)
        )
        diagnostics = {
            "active_features": float(np.sum(state.active_mask)),
            "active_centers": float(active_centers),
            "leverage": float(leverage),
            "additions": float(additions),
            "replacements": float(replacements),
            "skipped_novel": float(skipped),
            "throttled_novel": float(throttled),
            "finite_failures": float(finite_failures),
        }
        return prediction, diagnostics


def run_multibank_stream(
    observations: jax.Array,
    targets: jax.Array,
    config: MultiBankConfig,
    seed: int,
) -> tuple[MultiBankRKHSLearner, MultiBankState, np.ndarray]:
    """Run one multi-bank configuration on a materialized stream."""
    obs_np = np.asarray(observations, dtype=np.float64)
    tgt_np = np.asarray(targets, dtype=np.float64)
    learner = MultiBankRKHSLearner(
        n_heads=int(tgt_np.shape[1]),
        feature_dim=int(obs_np.shape[1]),
        config=config,
    )
    state = learner.init(seed=seed)
    metrics = np.zeros((obs_np.shape[0], 9), dtype=np.float64)
    for idx, (obs, target) in enumerate(zip(obs_np, tgt_np, strict=True)):
        prediction, diagnostics = learner.step(state, obs, target)
        metrics[idx, 0] = masked_mse_np(prediction, target)
        metrics[idx, 1] = float(np.argmax(prediction))
        metrics[idx, 2] = diagnostics["active_features"]
        metrics[idx, 3] = diagnostics["active_centers"]
        metrics[idx, 4] = diagnostics["leverage"]
        metrics[idx, 5] = diagnostics["additions"]
        metrics[idx, 6] = diagnostics["replacements"]
        metrics[idx, 7] = diagnostics["throttled_novel"]
        metrics[idx, 8] = diagnostics["finite_failures"]
    return learner, state, metrics


def evaluate_multibank_classifier(
    learner: MultiBankRKHSLearner,
    state: MultiBankState,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the final multi-bank classifier on held-out digits."""
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float64)[y_test]
    preds = np.stack([learner.predict(state, obs) for obs in x_test.astype(np.float64)])
    return {
        "test_mse": float(np.mean((preds - targets) ** 2)),
        "test_accuracy": float(np.mean(np.argmax(preds, axis=1) == y_test)),
    }


def bank_diagnostics(state: MultiBankState) -> dict[str, Any]:
    """Return final per-bank resource diagnostics."""
    diagnostics: dict[str, Any] = {}
    for bank in state.banks:
        width = bank_width(bank.config)
        sl = slice(bank.offset, bank.offset + width)
        if isinstance(bank, DictionaryBankState):
            diagnostics[bank.config.name] = {
                "kind": "dictionary",
                "kernel_method": kernel_method_name(bank.config.kernel_config),
                "feature_scale": bank.config.feature_scale,
                "width": width,
                "active_centers": int(bank.active_count),
                "additions": int(bank.additions),
                "replacements": int(bank.replacements),
                "skipped_novel": int(bank.skipped_novel),
                "throttled_novel": int(bank.throttled_novel),
                "mean_novelty": float(bank.novelty_sum / max(state.steps, 1)),
                "mean_abs_alpha": float(np.mean(np.abs(state.alpha[sl]))),
            }
        elif isinstance(bank, RandomTanhBankState):
            diagnostics[bank.config.name] = {
                "kind": "random_tanh",
                "feature_scale": bank.config.feature_scale,
                "width": width,
                "active_features": width,
                "weight_scale": bank.config.weight_scale,
                "mean_abs_alpha": float(np.mean(np.abs(state.alpha[sl]))),
                "mean_abs_activation": float(np.mean(bank.activation_ema)),
            }
        else:
            diagnostics[bank.config.name] = {
                "kind": "fourier",
                "feature_scale": bank.config.feature_scale,
                "width": width,
                "active_features": bank.active_width,
                "frequencies": list(bank.config.frequencies),
                "mean_abs_alpha": float(np.mean(np.abs(state.alpha[sl]))),
                "mean_abs_activation": float(np.mean(bank.activation_ema)),
            }
    return diagnostics


def summarize_multibank_metrics(
    metrics: np.ndarray,
    state: MultiBankState,
    final_window: int,
    labels: np.ndarray | None,
) -> dict[str, float]:
    """Summarize one multi-bank run's prequential curve and resource state."""
    summary = summarize_prequential(
        metrics,
        final_window=final_window,
        labels=labels,
        loss_col=0,
        pred_col=1,
    )
    summary.update(
        {
            "active_features": float(np.sum(state.active_mask)),
            "active_centers": float(np.mean(metrics[-min(final_window, metrics.shape[0]) :, 3])),
            "additions": float(metrics[-1, 5]),
            "replacements": float(metrics[-1, 6]),
            "throttled_novel": float(metrics[-1, 7]),
            "mean_leverage": float(state.leverage_sum / max(state.steps, 1)),
            "finite_failures": float(metrics[-1, 8]),
        }
    )
    return summary


def expand_dataset_names(spec: str) -> list[str]:
    """Expand D08 aliases and delegate all ordinary names to D07."""
    if spec.strip() == "blockers":
        return list(BLOCKER_DATASETS)
    return expand_d07_dataset_names(spec)


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    multibank_configs: list[MultiBankConfig],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run MLP baselines and all D08 learners for one paired dataset/seed."""
    observations, targets, labels, x_test, y_test, dataset_meta = make_dataset(
        dataset_name,
        seed,
        args,
    )
    methods: dict[str, dict[str, float]] = {}
    method_diagnostics: dict[str, Any] = {}

    for method in MLP_METHODS:
        print(f"{dataset_name} seed={seed}: running {method}")
        mlp_learner = make_mlp(
            method=method,
            n_heads=int(targets.shape[1]),
            step_size=args.mlp_step_size,
            sparsity=args.mlp_sparsity,
        )
        t0 = time.time()
        state, metrics = run_mlp_stream(
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
                evaluate_mlp_classifier(mlp_learner, state, x_test, y_test)
            )

    for config in multibank_configs:
        method = multibank_method_name(config)
        print(f"{dataset_name} seed={seed}: running {method}")
        t0 = time.time()
        mb_learner, mb_state, metrics = run_multibank_stream(
            observations=observations,
            targets=targets,
            config=config,
            seed=seed + 90_000,
        )
        methods[method] = summarize_multibank_metrics(
            metrics=metrics,
            state=mb_state,
            final_window=args.final_window,
            labels=labels,
        )
        methods[method]["runtime_s"] = float(time.time() - t0)
        method_diagnostics[method] = bank_diagnostics(mb_state)
        if dataset_name in DIGITS_REGIMES:
            assert x_test is not None and y_test is not None
            methods[method].update(
                evaluate_multibank_classifier(mb_learner, mb_state, x_test, y_test)
            )

    return (
        {
            "dataset_name": dataset_name,
            "seed": seed,
            "methods": methods,
            "method_diagnostics": method_diagnostics,
            "dataset": dataset_meta,
        },
        dataset_meta,
    )


def config_to_json(config: MultiBankConfig) -> dict[str, Any]:
    """Return a JSON-serializable representation of one multi-bank config."""
    banks: list[dict[str, Any]] = []
    for bank in config.banks:
        if isinstance(bank, DictionaryBankConfig):
            row = {
                "kind": "dictionary",
                "name": bank.name,
                "feature_scale": bank.feature_scale,
                "step_size": bank.step_size,
                "kernel_config": asdict(bank.kernel_config),
            }
        elif isinstance(bank, RandomTanhBankConfig):
            row = {"kind": "random_tanh", **asdict(bank)}
        else:
            row = {"kind": "fourier", **asdict(bank)}
        banks.append(row)
    return {
        "name": config.name,
        "coefficient_update": config.coefficient_update,
        "rho": config.rho,
        "rls_delta": config.rls_delta,
        "nlms_step_size": config.nlms_step_size,
        "output_clip": config.output_clip,
        "banks": banks,
        "total_width": int(sum(bank_width(bank) for bank in config.banks)),
    }


def primary_metrics_for_dataset(dataset_agg: dict[str, Any]) -> tuple[str, ...]:
    """Return metrics worth printing for one dataset aggregate."""
    metrics = ["final_window_mse", "online_mean_mse"]
    first_method = next(method for method in dataset_agg if method != "comparisons")
    if "final_window_accuracy" in dataset_agg[first_method]:
        metrics.append("final_window_accuracy")
    if "test_accuracy" in dataset_agg[first_method]:
        metrics.append("test_accuracy")
    if "test_mse" in dataset_agg[first_method]:
        metrics.append("test_mse")
    return tuple(metrics)


def best_candidate_summary(
    dataset_agg: dict[str, Any],
    metric: str,
) -> dict[str, Any] | None:
    """Return D07 aggregate's best-candidate-vs-best-MLP comparison row."""
    comparisons = dataset_agg.get("comparisons", {})
    metric_rows = comparisons.get(metric)
    if not isinstance(metric_rows, dict):
        return None
    row = metric_rows.get("best_kernel_vs_best_mlp")
    return row if isinstance(row, dict) else None


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write the D08 Markdown report."""
    cfg = results["config"]
    lines = [
        "# D08 Multi-Bank Kernel Learner",
        "",
        (
            f"Protocol: {cfg['n_seeds']} paired seeds, {cfg['steps']} online "
            f"steps, final window {cfg['final_window']}. Datasets: "
            f"{', '.join(cfg['datasets'])}."
        ),
        "",
        "Mechanism: one additive predictor over complementary feature banks. "
        "Dictionary banks allocate centers by their own ALD novelty rules; the "
        "active bank features are concatenated and one shared output matrix is "
        "updated from the single prequential prediction error. There is no MLP "
        "inside the D08 learner and no prediction router.",
        "",
        "Positive paired differences favor the D08 candidate when comparing MSE "
        "against the best fair MLP baseline; for accuracy, positive differences "
        "mean D08 has higher accuracy.",
        "",
        "## Configurations",
        "",
    ]
    for method, config in zip(
        results["candidate_methods"],
        results["multibank_configs"],
        strict=True,
    ):
        lines.extend(
            [
                f"### `{method}`",
                "",
                (
                    f"Update: `{config['coefficient_update']}`; total width "
                    f"{config['total_width']}; rho {config['rho']}; "
                    f"RLS delta {config['rls_delta']}; NLMS step "
                    f"{config['nlms_step_size']}."
                ),
                "",
                "| Bank | Kind | Width | Allocation | Feature Scale | Step |",
                "|---|---|---:|---|---:|---:|",
            ]
        )
        for bank in config["banks"]:
            if bank["kind"] == "dictionary":
                kernel_cfg = bank["kernel_config"]
                allocation = (
                    f"`{kernel_cfg['kernel']}`, novelty {kernel_cfg['novelty_threshold']}, "
                    f"interval {kernel_cfg['center_add_interval']}, replace "
                    f"{kernel_cfg['replace_when_full']}"
                )
                width = kernel_cfg["budget"]
            elif bank["kind"] == "random_tanh":
                allocation = (
                    f"fixed tanh random features, weight scale "
                    f"{bank['weight_scale']}, bias {bank['include_bias']}"
                )
                width = bank["width"] + int(bank["include_bias"])
            else:
                allocation = (
                    f"fixed Fourier features, max dim {bank['max_input_dim']}, "
                    f"frequencies {bank['frequencies']}"
                )
                width = (
                    int(bank["include_bias"])
                    + int(bank["include_raw"]) * int(bank["max_input_dim"])
                    + 2 * int(bank["max_input_dim"]) * len(bank["frequencies"])
                )
            lines.append(
                f"| `{bank['name']}` | {bank['kind']} | {width} | "
                f"{allocation} | {bank['feature_scale']:.3f} | "
                f"{bank['step_size']:.3f} |"
            )
        lines.append("")

    positive_tasks = 0
    total_tasks = 0
    lines.extend(["## Results", ""])
    for dataset, dataset_agg in results["aggregate"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Final MSE | Mean MSE | Final Acc | Test Acc | "
                "Test MSE | Active Features | Runtime s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in dataset_agg.items():
            if method == "comparisons":
                continue
            lines.append(
                f"| `{method}` | {metric_cell(row, 'final_window_mse')} | "
                f"{metric_cell(row, 'online_mean_mse')} | "
                f"{metric_cell(row, 'final_window_accuracy')} | "
                f"{metric_cell(row, 'test_accuracy')} | "
                f"{metric_cell(row, 'test_mse')} | "
                f"{metric_cell(row, 'active_features')} | "
                f"{metric_cell(row, 'runtime_s')} |"
            )
        lines.append("")
        primary = "test_accuracy" if dataset in DIGITS_REGIMES else "final_window_mse"
        for metric in primary_metrics_for_dataset(dataset_agg):
            row = best_candidate_summary(dataset_agg, metric)
            if row is None:
                continue
            diff = row["paired_diff_mean_positive_favors_kernel"]
            if metric == primary:
                total_tasks += 1
                if diff > 0.0:
                    positive_tasks += 1
            lines.append(
                f"`{metric}` best-D08-vs-best-MLP diff: {diff:+.4f} +/- "
                f"{row['paired_diff_stderr']:.4f}; wins/losses/ties "
                f"{row['wins_for_kernel']}/{row['wins_for_mlp']}/{row['ties']}; "
                f"best-D08 counts {row['best_kernel_counts']}."
            )
        lines.append("")

    lines.extend(
        [
            "## Assessment",
            "",
            (
                f"Using the primary metric per dataset, D08 has a positive mean "
                f"paired difference on {positive_tasks}/{total_tasks} configured "
                "datasets."
            ),
            "",
            "Promotion bar: a Step 2 closure claim requires one fixed D08 method, "
            "not best-of-config selection, to beat the best fair MLP on the full "
            "benchmark set and retain that advantage under more seeds. The table "
            "above reports both individual methods and the best-D08 row so the "
            "search headroom and the canonical-config result can be separated.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="blockers",
        help=(
            "Comma-separated regimes or aliases. D08 adds `blockers` for "
            "controlled_frequency, synthetic_compositional, synthetic_frequency, "
            "digits_label_drift, digits_mask_noise, and digits_permuted_pixels. "
            "D07 aliases such as all, controlled, synthetic, and digits also work."
        ),
    )
    parser.add_argument("--configs", default="canonical_bank_nlms")
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
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--rls-delta", type=float, default=10.0)
    parser.add_argument("--nlms-step-size", type=float, default=0.4)
    parser.add_argument("--output-clip", type=float, default=0.0)
    parser.add_argument("--raw-poly-budget", type=int, default=64)
    parser.add_argument("--green-budget", type=int, default=128)
    parser.add_argument("--arc-budget", type=int, default=128)
    parser.add_argument("--tanh-width", type=int, default=256)
    parser.add_argument("--green-center-interval", type=int, default=8)
    parser.add_argument("--arc-center-interval", type=int, default=4)
    parser.add_argument("--green-sigma", type=float, default=1.0)
    parser.add_argument("--arccosine-depth", type=int, default=1)
    parser.add_argument("--raw-poly-feature-scale", type=float, default=1.0)
    parser.add_argument("--green-feature-scale", type=float, default=1.0)
    parser.add_argument("--arccosine-feature-scale", type=float, default=1.0)
    parser.add_argument("--tanh-feature-scale", type=float, default=1.0)
    parser.add_argument("--fourier-feature-scale", type=float, default=1.0)
    parser.add_argument("--raw-poly-step-size", type=float, default=0.5)
    parser.add_argument("--green-step-size", type=float, default=0.4)
    parser.add_argument("--arccosine-step-size", type=float, default=0.4)
    parser.add_argument("--tanh-step-size", type=float, default=0.4)
    parser.add_argument("--fourier-step-size", type=float, default=0.3)
    parser.add_argument("--tanh-weight-scale", type=float, default=1.0)
    parser.add_argument("--tanh-input-clip", type=float, default=5.0)
    parser.add_argument("--fourier-max-input-dim", type=int, default=8)
    parser.add_argument("--fourier-input-clip", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--smoke", action="store_true", help="Tiny harness check.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.raw_poly_budget <= 0 or args.green_budget <= 0 or args.arc_budget <= 0:
        raise ValueError("all dictionary budgets must be positive")
    if args.tanh_width <= 0:
        raise ValueError("--tanh-width must be positive")
    if args.rho <= 0.0 or args.rho > 1.0:
        raise ValueError("--rho must be in (0, 1]")
    if args.rls_delta <= 0.0:
        raise ValueError("--rls-delta must be positive")
    if args.nlms_step_size < 0.0:
        raise ValueError("--nlms-step-size must be non-negative")
    if (
        args.raw_poly_step_size < 0.0
        or args.green_step_size < 0.0
        or args.arccosine_step_size < 0.0
        or args.tanh_step_size < 0.0
        or args.fourier_step_size < 0.0
    ):
        raise ValueError("bank step sizes must be non-negative")
    if args.fourier_max_input_dim <= 0:
        raise ValueError("--fourier-max-input-dim must be positive")
    if args.green_center_interval <= 0 or args.arc_center_interval <= 0:
        raise ValueError("center intervals must be positive")
    if args.tanh_weight_scale <= 0.0:
        raise ValueError("--tanh-weight-scale must be positive")


def main() -> None:
    """Run the D08 experiment and write JSON/Markdown artifacts."""
    args = parse_args()
    if args.smoke:
        args.datasets = "controlled_frequency"
        args.configs = "compact_rls"
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.raw_poly_budget = 16
        args.green_budget = 24
        args.tanh_width = 32
    validate_args(args)
    datasets = expand_dataset_names(args.datasets)
    multibank_configs = make_preset_configs(args)
    candidate_methods = tuple(multibank_method_name(config) for config in multibank_configs)

    t0 = time.time()
    records: list[dict[str, Any]] = []
    datasets_meta: dict[str, Any] = {}
    for dataset_name in datasets:
        for offset in range(args.n_seeds):
            seed = args.seed + offset
            record, dataset_meta = run_one_dataset_seed(
                dataset_name=dataset_name,
                seed=seed,
                multibank_configs=multibank_configs,
                args=args,
            )
            records.append(record)
            datasets_meta[dataset_name] = dataset_meta

    results = {
        "config": {
            "datasets": datasets,
            "configs": args.configs,
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
        },
        "datasets": datasets_meta,
        "multibank_configs": [config_to_json(config) for config in multibank_configs],
        "candidate_methods": list(candidate_methods),
        "mlp_methods": list(MLP_METHODS),
        "records": records,
        "aggregate": aggregate_candidate_records(records, candidate_methods),
        "wall_clock_s": time.time() - t0,
        "evidence_level": "single_predictor_additive_multibank_rkhs_probe",
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

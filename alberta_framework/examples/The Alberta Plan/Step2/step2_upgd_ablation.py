#!/usr/bin/env python3
"""Worker S2A: focused UPGD tuning and ablation for Step 2.

This script keeps the UPGD question separate from the canonical Step 2
benchmarks.  It runs small, reproducible sweeps against two fair MLP baselines:
``MLP(64)`` and ``MLP(64, 64)``.  The goal is not to declare a final canonical
winner from a tiny sweep, but to identify UPGD settings that deserve a larger
follow-up and settings that should be discarded.

Supported suites:

* ``digits``: online prequential classification on
  ``sklearn.datasets.load_digits`` with IID, permuted-pixel, class-blocked,
  label-drift, and mask-noise streams.
* ``synthetic``: the Step 2 out-of-hypothesis-class stream family
  (polynomial, frequency mismatch, and compositional).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import (  # noqa: E402
    CompositionalStream,
    FrequencyMismatchStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    OutOfClassPolynomialStream,
    UPGDLearner,
)

N_DIGIT_CLASSES = 10
STEP_SIZE = 0.03
SPARSITY = 0.5
OBGD_KAPPA = 2.0
DEFAULT_OUTPUT_DIR = Path("output/direction2_upgd_ablation")


@dataclass(frozen=True)
class MethodConfig:
    """Common method metadata stored with every run."""

    name: str
    method_type: str
    hidden_sizes: tuple[int, ...]
    step_size: float = STEP_SIZE
    sparsity: float = SPARSITY
    use_layer_norm: bool = True
    perturbation_sigma: float | None = None
    utility_decay: float | None = None
    perturbation_beta: float | None = None
    perturbation_interval: int | None = None
    loss_normalization: str = "mean"
    head_gradient_scale: str = "none"
    head_step_size_multiplier: float = 1.0
    bounder_kappa: float = OBGD_KAPPA
    positive_target_loss_scale: float = 1.0
    negative_target_loss_scale: float = 1.0
    head_bias_step_size_multiplier: float = 1.0
    head_loss_pressure_gate_ratio: float = 0.0
    head_loss_pressure_multiplier: float = 0.0
    head_loss_pressure_warmup_steps: int = 0
    head_repetition_multiplier: float = 0.0
    head_repetition_decay: float = 0.9
    head_repetition_delta_threshold: float = 0.05
    head_repetition_pressure_threshold: float = 0.0
    head_repetition_warmup_steps: int = 0
    adaptive_kappa_mode: str = "none"
    adaptive_kappa_base: float = 0.5
    adaptive_kappa_min: float = 0.25
    adaptive_kappa_max: float = 1.0
    adaptive_kappa_exponent: float = 0.5
    adaptive_kappa_warmup_steps: int = 0
    adaptive_kappa_meta_step_size: float = 0.0
    adaptive_kappa_meta_min_multiplier: float = 0.5
    adaptive_kappa_meta_max_multiplier: float = 2.0
    adaptive_kappa_meta_warmup_steps: int = 0
    meta_plasticity_mode: str = "none"
    meta_plasticity_step_size: float = 0.0
    meta_plasticity_min_multiplier: float = 0.25
    meta_plasticity_max_multiplier: float = 4.0
    meta_plasticity_warmup_steps: int = 0
    meta_plasticity_trunk_enabled: bool = True
    meta_plasticity_head_weight_enabled: bool = True
    meta_plasticity_head_bias_enabled: bool = True
    meta_plasticity_repetition_enabled: bool = True
    readout_mode: str = "linear_mse"
    readout_loss_mode: str | None = None
    readout_prediction_mode: str | None = None
    readout_robust_q: float = 0.7
    readout_input_mode: str = "hidden"
    readout_head_normalization: str = "none"
    readout_margin: float = 0.0
    readout_margin_step_size: float = 0.0
    readout_simplex_bias_decay: float = 0.0
    readout_simplex_bias_centering_rate: float = 0.0
    readout_adapter_step_size: float = 0.0
    readout_adapter_identity_reg: float = 0.0
    readout_adapter_entropy_reg: float | None = None
    readout_fast_head_step_size_multiplier: float = 1.0
    readout_fast_head_bias_step_size_multiplier: float = 1.0
    readout_fast_trunk_gradient_multiplier: float = 0.0
    readout_fast_head_bounder_mode: str = "shared"
    readout_slow_simplex_gradient_multiplier: float = 1.0


DENSITY_ADAPTK_KWARGS: dict[str, Any] = {
    "step_size": STEP_SIZE * 0.6,
    "bounder_kappa": 0.5,
    "perturbation_sigma": 1e-4,
    "utility_decay": 0.995,
    "perturbation_beta": 2.0,
    "perturbation_interval": 1,
    "loss_normalization": "target_density",
    "adaptive_kappa_mode": "loss_ratio",
    "adaptive_kappa_base": 0.5,
    "adaptive_kappa_min": 0.35,
    "adaptive_kappa_max": 0.65,
    "adaptive_kappa_exponent": 0.5,
    "adaptive_kappa_warmup_steps": 120,
}

STRUCTURE_ADAPTK_KWARGS: dict[str, Any] = {
    **DENSITY_ADAPTK_KWARGS,
    "loss_normalization": "target_structure",
}

HEAD_ONLY_META_KWARGS: dict[str, Any] = {
    "meta_plasticity_mode": "gradient_alignment",
    "meta_plasticity_step_size": 0.001,
    "meta_plasticity_min_multiplier": 0.5,
    "meta_plasticity_max_multiplier": 2.0,
    "meta_plasticity_warmup_steps": 30,
    "meta_plasticity_trunk_enabled": False,
    "meta_plasticity_head_weight_enabled": True,
    "meta_plasticity_head_bias_enabled": True,
    "meta_plasticity_repetition_enabled": True,
}


BASELINE_CONFIGS: tuple[MethodConfig, ...] = (
    MethodConfig("mlp64", "mlp", (64,)),
    MethodConfig("mlp64_64", "mlp", (64, 64)),
)

UPGD_CATALOG: dict[str, MethodConfig] = {
    "upgd64_no_noise": MethodConfig(
        "upgd64_no_noise",
        "upgd",
        (64,),
        perturbation_sigma=0.0,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_sigma1e_4": MethodConfig(
        "upgd64_sigma1e_4",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_sigma3e_4": MethodConfig(
        "upgd64_sigma3e_4",
        "upgd",
        (64,),
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_default": MethodConfig(
        "upgd64_default",
        "upgd",
        (64,),
        perturbation_sigma=1e-3,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_sigma3e_3": MethodConfig(
        "upgd64_sigma3e_3",
        "upgd",
        (64,),
        perturbation_sigma=3e-3,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_fast_utility": MethodConfig(
        "upgd64_fast_utility",
        "upgd",
        (64,),
        perturbation_sigma=1e-3,
        utility_decay=0.99,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_slow_utility": MethodConfig(
        "upgd64_slow_utility",
        "upgd",
        (64,),
        perturbation_sigma=1e-3,
        utility_decay=0.999,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_interval10": MethodConfig(
        "upgd64_interval10",
        "upgd",
        (64,),
        perturbation_sigma=1e-3,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=10,
    ),
    "upgd64_beta1": MethodConfig(
        "upgd64_beta1",
        "upgd",
        (64,),
        perturbation_sigma=1e-3,
        utility_decay=0.995,
        perturbation_beta=1.0,
        perturbation_interval=1,
    ),
    "upgd64_beta0_sigma3e_4": MethodConfig(
        "upgd64_beta0_sigma3e_4",
        "upgd",
        (64,),
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=0.0,
        perturbation_interval=1,
    ),
    "upgd64_dense_sigma3e_4": MethodConfig(
        "upgd64_dense_sigma3e_4",
        "upgd",
        (64,),
        sparsity=0.0,
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_no_ln_sigma3e_4": MethodConfig(
        "upgd64_no_ln_sigma3e_4",
        "upgd",
        (64,),
        use_layer_norm=False,
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_lr1e_2_sigma3e_4": MethodConfig(
        "upgd64_lr1e_2_sigma3e_4",
        "upgd",
        (64,),
        step_size=0.01,
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_64_default": MethodConfig(
        "upgd64_64_default",
        "upgd",
        (64, 64),
        perturbation_sigma=1e-3,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_64_sigma3e_4": MethodConfig(
        "upgd64_64_sigma3e_4",
        "upgd",
        (64, 64),
        perturbation_sigma=3e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd128_default": MethodConfig(
        "upgd128_default",
        "upgd",
        (128,),
        perturbation_sigma=1e-3,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
    ),
    "upgd64_sum_sigma0": MethodConfig(
        "upgd64_sum_sigma0",
        "upgd",
        (64,),
        perturbation_sigma=0.0,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="sum",
    ),
    "upgd64_sum_sigma1e_4": MethodConfig(
        "upgd64_sum_sigma1e_4",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="sum",
    ),
    "upgd64_density_sigma1e_4": MethodConfig(
        "upgd64_density_sigma1e_4",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="target_density",
    ),
    "upgd64_structure_sigma1e_4": MethodConfig(
        "upgd64_structure_sigma1e_4",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="target_structure",
    ),
    "upgd64_headx2_sigma0": MethodConfig(
        "upgd64_headx2_sigma0",
        "upgd",
        (64,),
        perturbation_sigma=0.0,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        head_step_size_multiplier=2.0,
    ),
    "upgd64_headx3_sigma0": MethodConfig(
        "upgd64_headx3_sigma0",
        "upgd",
        (64,),
        perturbation_sigma=0.0,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        head_step_size_multiplier=3.0,
    ),
    "upgd64_headscale_sigma0": MethodConfig(
        "upgd64_headscale_sigma0",
        "upgd",
        (64,),
        perturbation_sigma=0.0,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        head_gradient_scale="active_count",
    ),
    "upgd64_mean_sigma1e_4_kappa05": MethodConfig(
        "upgd64_mean_sigma1e_4_kappa05",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="mean",
        bounder_kappa=0.5,
    ),
    "upgd64_density_sigma1e_4_kappa05": MethodConfig(
        "upgd64_density_sigma1e_4_kappa05",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="target_density",
        bounder_kappa=0.5,
    ),
    "upgd64_sum_sigma1e_4_kappa05": MethodConfig(
        "upgd64_sum_sigma1e_4_kappa05",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="sum",
        bounder_kappa=0.5,
    ),
    "upgd64_structure_sigma1e_4_kappa05": MethodConfig(
        "upgd64_structure_sigma1e_4_kappa05",
        "upgd",
        (64,),
        perturbation_sigma=1e-4,
        utility_decay=0.995,
        perturbation_beta=2.0,
        perturbation_interval=1,
        loss_normalization="target_structure",
        bounder_kappa=0.5,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx0_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx0_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        head_repetition_multiplier=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.5,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx035_meta001_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx035_meta001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.35,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias025_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias025_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.5,
        head_bias_step_size_multiplier=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias0_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias0_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.5,
        head_bias_step_size_multiplier=0.0,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_slowutil999_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_slowutil999_notrunk_tight",
        "upgd",
        (64,),
        **(DENSITY_ADAPTK_KWARGS | {"utility_decay": 0.999}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_biasmeta001_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_biasmeta001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(
            HEAD_ONLY_META_KWARGS
            | {
                "meta_plasticity_head_weight_enabled": False,
                "meta_plasticity_repetition_enabled": False,
            }
        ),
        head_repetition_multiplier=0.25,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_tiny_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_tiny_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.003,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_m001_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.001,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_m0025_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m0025_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.0025,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin01_m0025_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin01_m0025_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.1,
        readout_margin_step_size=0.0025,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_"
        "meta001_margin_m0025_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_margin_m0025_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.5,
        readout_margin=0.2,
        readout_margin_step_size=0.0025,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_center_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_center_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.003,
        readout_simplex_bias_centering_rate=1.0,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_decay_center_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_decay_center_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.003,
        readout_simplex_bias_decay=0.001,
        readout_simplex_bias_centering_rate=1.0,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_center_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_center_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_simplex_bias_centering_rate=1.0,
    ),
    (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_"
        "margin_center_notrunk_tight"
    ): MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_margin_center_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(HEAD_ONLY_META_KWARGS | {"meta_plasticity_step_size": 0.003}),
        readout_margin=0.2,
        readout_margin_step_size=0.003,
        readout_simplex_bias_centering_rate=1.0,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_bias_step_size_multiplier=0.0,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx025_"
        "meta001_margin_m001_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
        readout_margin=0.2,
        readout_margin_step_size=0.001,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_bias0_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_bias_step_size_multiplier=0.0,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_center_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_center_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_simplex_bias_centering_rate=1.0,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_bias0_neg05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_neg05_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_bias_step_size_multiplier=0.0,
        negative_target_loss_scale=0.5,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_bias0_neg025_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_neg025_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_bias_step_size_multiplier=0.0,
        negative_target_loss_scale=0.25,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_center_neg05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_center_neg05_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_simplex_bias_centering_rate=1.0,
        negative_target_loss_scale=0.5,
    ),
    (
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_center_neg025_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_center_neg025_notrunk_tight",
        "upgd",
        (64, 64),
        **DENSITY_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_simplex_bias_centering_rate=1.0,
        negative_target_loss_scale=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(HEAD_ONLY_META_KWARGS | {"meta_plasticity_step_size": 0.003}),
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_repx025_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_repx025_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(HEAD_ONLY_META_KWARGS | {"meta_plasticity_step_size": 0.003}),
        head_repetition_multiplier=0.25,
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_trunk_head_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_trunk_head_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(
            HEAD_ONLY_META_KWARGS
            | {
                "meta_plasticity_step_size": 0.003,
                "meta_plasticity_trunk_enabled": True,
            }
        ),
    ),
    "upgd_density_sigma1e_4_adaptk035_065_lr06_rep_learned_notrunk_tight": MethodConfig(
        "upgd_density_sigma1e_4_adaptk035_065_lr06_rep_learned_notrunk_tight",
        "upgd",
        (64,),
        **DENSITY_ADAPTK_KWARGS,
        **(
            HEAD_ONLY_META_KWARGS
            | {
                "meta_plasticity_step_size": 0.003,
                "meta_plasticity_repetition_enabled": True,
            }
        ),
    ),
    "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight": MethodConfig(
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd",
        (64,),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.25,
    ),
    "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight": MethodConfig(
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64,),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    "upgd_structure_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight": MethodConfig(
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd",
        (64,),
        **STRUCTURE_ADAPTK_KWARGS,
        **(HEAD_ONLY_META_KWARGS | {"meta_plasticity_step_size": 0.003}),
    ),
    "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight": MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_headx2_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_headx2_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_step_size_multiplier=2.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_headscale_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_headscale_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_gradient_scale="active_count",
    ),
    "upgd64_64_structure_sigma1e_4_adaptk05_10_lr06_repx075_meta001_notrunk_tight": MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk05_10_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64, 64),
        **(
            STRUCTURE_ADAPTK_KWARGS
            | {
                "adaptive_kappa_min": 0.5,
                "adaptive_kappa_max": 1.0,
                "bounder_kappa": 0.75,
            }
        ),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    "upgd64_64_structure_sigma1e_4_adaptk075_15_lr06_repx075_meta001_notrunk_tight": MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk075_15_lr06_repx075_meta001_notrunk_tight",
        "upgd",
        (64, 64),
        **(
            STRUCTURE_ADAPTK_KWARGS
            | {
                "adaptive_kappa_min": 0.75,
                "adaptive_kappa_max": 1.5,
                "bounder_kappa": 1.0,
            }
        ),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma0_adaptk035_065_lr06_repx075_"
        "meta001_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma0_adaptk035_065_lr06_repx075_meta001_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"perturbation_sigma": 0.0}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr05_repx075_"
        "meta001_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr05_repx075_meta001_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.5}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_"
        "meta001_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.7}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx05_"
        "meta001_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx05_meta001_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.5,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_softmax_headx15_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_softmax_headx15_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        head_step_size_multiplier=1.5,
        readout_mode="softmax_ce",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptive_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptive_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_simplex",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_"
        "meta001_adaptive_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_adaptive_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.7}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_simplex",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_linearloss_softmaxpred_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_linearloss_softmaxpred_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="linear_mse",
        readout_loss_mode="linear_mse",
        readout_prediction_mode="softmax",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_celoss_identitypred_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_celoss_identitypred_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
        readout_loss_mode="softmax_ce",
        readout_prediction_mode="identity",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_celoss_clippred_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_celoss_clippred_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
        readout_loss_mode="softmax_ce",
        readout_prediction_mode="unit_clip",
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_gceq07_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_gceq07_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
        readout_loss_mode="gce",
        readout_prediction_mode="softmax",
        readout_robust_q=0.7,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_gceq03_softmax_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_gceq03_softmax_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="softmax_ce",
        readout_loss_mode="gce",
        readout_prediction_mode="softmax",
        readout_robust_q=0.3,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptivegceq07_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivegceq07_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_simplex",
        readout_loss_mode="adaptive_gce",
        readout_prediction_mode="adaptive_simplex",
        readout_robust_q=0.7,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptivegceq03_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivegceq03_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_simplex",
        readout_loss_mode="adaptive_gce",
        readout_prediction_mode="adaptive_simplex",
        readout_robust_q=0.3,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adapterslow_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adapterslow_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.003,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adaptermoderate_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.01,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adapterfast_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adapterfast_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.03,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_"
        "meta001_factorized_adaptermoderate_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_factorized_adaptermoderate_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.7}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.01,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adaptermoderate_idreg1e_4_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_4_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.01,
        readout_adapter_identity_reg=1e-4,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adaptermoderate_idreg1e_3_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_3_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.01,
        readout_adapter_identity_reg=1e-3,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_factorized_adaptermoderate_idreg1e_2_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_2_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="factorized_simplex",
        readout_adapter_step_size=0.01,
        readout_adapter_identity_reg=1e-2,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptivefactorized_adapterslow_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adapterslow_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_factorized_simplex",
        readout_adapter_step_size=0.003,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptivefactorized_adaptermoderate_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adaptermoderate_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_factorized_simplex",
        readout_adapter_step_size=0.01,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_adaptivefactorized_adaptermoderate_idreg1e_3_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adaptermoderate_idreg1e_3_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="adaptive_factorized_simplex",
        readout_adapter_step_size=0.01,
        readout_adapter_identity_reg=1e-3,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx05_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=0.5,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx1_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=1.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_"
        "meta001_twotime_fastx1_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_twotime_fastx1_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.7}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=1.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx1_trunk025_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_trunk025_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=1.0,
        readout_fast_trunk_gradient_multiplier=0.25,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx1_trunk05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_trunk05_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=1.0,
        readout_fast_trunk_gradient_multiplier=0.5,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk05_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=0.5,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk1_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk1_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=1.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk2_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=2.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx3_trunk1_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=3.0,
        readout_fast_trunk_gradient_multiplier=1.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk2_slow0_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=2.0,
        readout_slow_simplex_gradient_multiplier=0.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk2_slow025_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow025_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=2.0,
        readout_slow_simplex_gradient_multiplier=0.25,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx3_trunk1_slow0_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow0_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=3.0,
        readout_fast_trunk_gradient_multiplier=1.0,
        readout_slow_simplex_gradient_multiplier=0.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx3_trunk1_slow025_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow025_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=3.0,
        readout_fast_trunk_gradient_multiplier=1.0,
        readout_slow_simplex_gradient_multiplier=0.25,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk2_slow0_sepbound_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_sepbound_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=2.0,
        readout_fast_head_bounder_mode="separate",
        readout_slow_simplex_gradient_multiplier=0.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx2_trunk2_slow025_sepbound_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow025_sepbound_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=2.0,
        readout_fast_trunk_gradient_multiplier=2.0,
        readout_fast_head_bounder_mode="separate",
        readout_slow_simplex_gradient_multiplier=0.25,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx3_trunk1_slow0_sepbound_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow0_sepbound_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=3.0,
        readout_fast_trunk_gradient_multiplier=1.0,
        readout_fast_head_bounder_mode="separate",
        readout_slow_simplex_gradient_multiplier=0.0,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_"
        "meta001_twotime_fastx3_trunk1_slow025_sepbound_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow025_sepbound_notrunk_tight",
        "upgd",
        (64, 64),
        **STRUCTURE_ADAPTK_KWARGS,
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=3.0,
        readout_fast_trunk_gradient_multiplier=1.0,
        readout_fast_head_bounder_mode="separate",
        readout_slow_simplex_gradient_multiplier=0.25,
    ),
    (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_"
        "meta001_twotime_fastx1_trunk05_notrunk_tight"
    ): MethodConfig(
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_twotime_fastx1_trunk05_notrunk_tight",
        "upgd",
        (64, 64),
        **(STRUCTURE_ADAPTK_KWARGS | {"step_size": STEP_SIZE * 0.7}),
        **HEAD_ONLY_META_KWARGS,
        head_repetition_multiplier=0.75,
        readout_mode="two_timescale_simplex",
        readout_fast_head_step_size_multiplier=1.0,
        readout_fast_trunk_gradient_multiplier=0.5,
    ),
}

PRESET_CONFIGS: dict[str, tuple[str, ...]] = {
    "smoke": (
        "upgd64_no_noise",
        "upgd64_sigma3e_4",
        "upgd64_default",
        "upgd64_slow_utility",
        "upgd64_density_sigma1e_4",
        "upgd64_structure_sigma1e_4",
        "upgd64_beta0_sigma3e_4",
        "upgd64_dense_sigma3e_4",
        "upgd64_no_ln_sigma3e_4",
        "upgd64_64_default",
    ),
    "focused": (
        "upgd64_no_noise",
        "upgd64_sigma1e_4",
        "upgd64_sigma3e_4",
        "upgd64_default",
        "upgd64_sigma3e_3",
        "upgd64_fast_utility",
        "upgd64_slow_utility",
        "upgd64_sum_sigma1e_4",
        "upgd64_density_sigma1e_4",
        "upgd64_structure_sigma1e_4",
        "upgd64_interval10",
        "upgd64_beta1",
        "upgd64_beta0_sigma3e_4",
        "upgd64_dense_sigma3e_4",
        "upgd64_no_ln_sigma3e_4",
        "upgd64_lr1e_2_sigma3e_4",
        "upgd64_64_default",
        "upgd64_64_sigma3e_4",
        "upgd128_default",
    ),
    "promotion_matrix": (
        "upgd64_mean_sigma1e_4_kappa05",
        "upgd64_density_sigma1e_4_kappa05",
        "upgd64_sum_sigma1e_4_kappa05",
        "upgd64_structure_sigma1e_4_kappa05",
    ),
    "class_blocked_retention": (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx0_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_trunk_head_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_rep_learned_notrunk_tight",
    ),
    "class_blocked_retention_followup": (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_slowutil999_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_biasmeta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_tiny_notrunk_tight",
    ),
    "class_blocked_combined_head": (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_tiny_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_center_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_decay_center_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_center_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_margin_center_notrunk_tight",
    ),
    "class_blocked_margin_grid": (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_m0025_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin_tiny_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_margin01_m0025_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_margin_m0025_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
    ),
    "structure_adaptive_retention": (
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
    ),
    "class_blocked_bias_repetition_rescue": (
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx035_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias025_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx05_meta001_bias0_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_bias0_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight",
        "upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_repx025_notrunk_tight",
    ),
    "class_blocked_deep_rescue": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_headx2_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_headscale_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk05_10_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk075_15_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_softmax_notrunk_tight",
    ),
    "class_blocked_softmax_closeout": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma0_adaptk035_065_lr06_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr05_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx05_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_softmax_headx15_notrunk_tight",
    ),
    "readout_consistency_adaptive": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptive_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_adaptive_notrunk_tight",
    ),
    "readout_consistency_decoupled": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_linearloss_softmaxpred_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_celoss_identitypred_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_celoss_clippred_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_gceq07_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_gceq03_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptive_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivegceq07_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivegceq03_notrunk_tight",
    ),
    "readout_consistency_factorized": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptive_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adapterslow_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adapterfast_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_factorized_adaptermoderate_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_4_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_3_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_factorized_adaptermoderate_idreg1e_2_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adapterslow_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adaptermoderate_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adaptermoderate_idreg1e_3_notrunk_tight",
    ),
    "readout_consistency_twotime": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptive_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_adaptivefactorized_adaptermoderate_idreg1e_3_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx05_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_twotime_fastx1_notrunk_tight",
    ),
    "readout_consistency_fasttrunk": (
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_trunk025_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx1_trunk05_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk05_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk1_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow025_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow0_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow025_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow0_sepbound_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx2_trunk2_slow025_sepbound_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow0_sepbound_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr06_repx075_meta001_twotime_fastx3_trunk1_slow025_sepbound_notrunk_tight",
        "upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_twotime_fastx1_trunk05_notrunk_tight",
    ),
}
PRESET_CONFIGS["default"] = PRESET_CONFIGS["focused"]


def method_key(seed: int, method_name: str) -> jax.Array:
    """Return a stable PRNG key for a seed/method pair."""
    checksum = zlib.crc32(method_name.encode("utf-8")) & 0x7FFFFFFF
    return jr.fold_in(jr.key(seed), checksum)


def make_mlp(config: MethodConfig, n_heads: int) -> MultiHeadMLPLearner:
    """Create a fair MLP baseline."""
    return MultiHeadMLPLearner(
        n_heads=n_heads,
        hidden_sizes=config.hidden_sizes,
        step_size=config.step_size,
        bounder=ObGDBounding(kappa=config.bounder_kappa),
        sparsity=config.sparsity,
        use_layer_norm=config.use_layer_norm,
    )


def make_upgd(config: MethodConfig, n_heads: int) -> UPGDLearner:
    """Create one UPGD candidate from a sweep config."""
    if config.perturbation_sigma is None:
        raise ValueError(f"{config.name} missing perturbation_sigma")
    if config.utility_decay is None:
        raise ValueError(f"{config.name} missing utility_decay")
    if config.perturbation_beta is None:
        raise ValueError(f"{config.name} missing perturbation_beta")
    if config.perturbation_interval is None:
        raise ValueError(f"{config.name} missing perturbation_interval")
    upgd_kwargs: dict[str, Any] = {
        "n_heads": n_heads,
        "hidden_sizes": config.hidden_sizes,
        "step_size": config.step_size,
        "bounder": ObGDBounding(kappa=config.bounder_kappa),
        "sparsity": config.sparsity,
        "use_layer_norm": config.use_layer_norm,
        "perturbation_sigma": config.perturbation_sigma,
        "utility_decay": config.utility_decay,
        "perturbation_beta": config.perturbation_beta,
        "perturbation_interval": config.perturbation_interval,
        "loss_normalization": config.loss_normalization,
        "positive_target_loss_scale": config.positive_target_loss_scale,
        "negative_target_loss_scale": config.negative_target_loss_scale,
        "head_gradient_scale": config.head_gradient_scale,
        "head_step_size_multiplier": config.head_step_size_multiplier,
        "head_bias_step_size_multiplier": config.head_bias_step_size_multiplier,
        "head_loss_pressure_gate_ratio": config.head_loss_pressure_gate_ratio,
        "head_loss_pressure_multiplier": config.head_loss_pressure_multiplier,
        "head_loss_pressure_warmup_steps": config.head_loss_pressure_warmup_steps,
        "head_repetition_multiplier": config.head_repetition_multiplier,
        "head_repetition_decay": config.head_repetition_decay,
        "head_repetition_delta_threshold": config.head_repetition_delta_threshold,
        "head_repetition_pressure_threshold": (
            config.head_repetition_pressure_threshold
        ),
        "head_repetition_warmup_steps": config.head_repetition_warmup_steps,
        "adaptive_kappa_mode": config.adaptive_kappa_mode,
        "adaptive_kappa_base": config.adaptive_kappa_base,
        "adaptive_kappa_min": config.adaptive_kappa_min,
        "adaptive_kappa_max": config.adaptive_kappa_max,
        "adaptive_kappa_exponent": config.adaptive_kappa_exponent,
        "adaptive_kappa_warmup_steps": config.adaptive_kappa_warmup_steps,
        "adaptive_kappa_meta_step_size": config.adaptive_kappa_meta_step_size,
        "adaptive_kappa_meta_min_multiplier": (
            config.adaptive_kappa_meta_min_multiplier
        ),
        "adaptive_kappa_meta_max_multiplier": (
            config.adaptive_kappa_meta_max_multiplier
        ),
        "adaptive_kappa_meta_warmup_steps": (
            config.adaptive_kappa_meta_warmup_steps
        ),
        "meta_plasticity_mode": config.meta_plasticity_mode,
        "meta_plasticity_step_size": config.meta_plasticity_step_size,
        "meta_plasticity_min_multiplier": config.meta_plasticity_min_multiplier,
        "meta_plasticity_max_multiplier": config.meta_plasticity_max_multiplier,
        "meta_plasticity_warmup_steps": config.meta_plasticity_warmup_steps,
        "meta_plasticity_trunk_enabled": config.meta_plasticity_trunk_enabled,
        "meta_plasticity_head_weight_enabled": (
            config.meta_plasticity_head_weight_enabled
        ),
        "meta_plasticity_head_bias_enabled": (
            config.meta_plasticity_head_bias_enabled
        ),
        "meta_plasticity_repetition_enabled": (
            config.meta_plasticity_repetition_enabled
        ),
        "readout_mode": config.readout_mode,
        "readout_loss_mode": config.readout_loss_mode,
        "readout_prediction_mode": config.readout_prediction_mode,
        "readout_robust_q": config.readout_robust_q,
        "readout_input_mode": config.readout_input_mode,
        "readout_head_normalization": config.readout_head_normalization,
        "readout_margin": config.readout_margin,
        "readout_margin_step_size": config.readout_margin_step_size,
        "readout_simplex_bias_decay": config.readout_simplex_bias_decay,
        "readout_simplex_bias_centering_rate": (
            config.readout_simplex_bias_centering_rate
        ),
    }
    optional_readout_kwargs = {
        "readout_label_adapter_step_size": config.readout_adapter_step_size,
        "readout_label_adapter_identity_regularization": (
            config.readout_adapter_identity_reg
        ),
        "readout_label_adapter_entropy_regularization": (
            config.readout_adapter_entropy_reg
        ),
        "readout_fast_head_step_size_multiplier": (
            config.readout_fast_head_step_size_multiplier
        ),
        "readout_fast_head_bias_step_size_multiplier": (
            config.readout_fast_head_bias_step_size_multiplier
        ),
        "readout_fast_trunk_gradient_multiplier": (
            config.readout_fast_trunk_gradient_multiplier
        ),
        "readout_fast_head_bounder_mode": config.readout_fast_head_bounder_mode,
        "readout_slow_simplex_gradient_multiplier": (
            config.readout_slow_simplex_gradient_multiplier
        ),
    }
    upgd_parameters = inspect.signature(UPGDLearner).parameters
    for key, value in optional_readout_kwargs.items():
        if key in upgd_parameters and value is not None:
            upgd_kwargs[key] = value
    return UPGDLearner(**upgd_kwargs)


def active_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error over active non-NaN targets."""
    active = ~jnp.isnan(targets)
    safe_targets = jnp.where(active, targets, 0.0)
    squared_error = jnp.where(active, (predictions - safe_targets) ** 2, 0.0)
    denom = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return jnp.sum(squared_error) / denom


def run_online_regression(
    learner: Any,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run an online supervised stream and return per-step MSE."""
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt = inputs
        result = learner.update(carry, obs, tgt)
        mse = active_mse(result.predictions, tgt)
        return result.state, jnp.asarray([mse], dtype=jnp.float32)

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def run_online_digits(
    learner: Any,
    key: jax.Array,
    observations: jax.Array,
    targets: jax.Array,
    labels: jax.Array,
) -> tuple[Any, np.ndarray]:
    """Run the digits stream and return per-step MSE/accuracy."""
    state = learner.init(observations.shape[1], key)

    def step_fn(
        carry: Any,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[Any, jax.Array]:
        obs, tgt, label = inputs
        result = learner.update(carry, obs, tgt)
        mse = active_mse(result.predictions, tgt)
        correct = jnp.argmax(result.predictions) == label
        return result.state, jnp.stack([mse, correct.astype(jnp.float32)])

    final_state, metrics = jax.lax.scan(step_fn, state, (observations, targets, labels))
    metrics.block_until_ready()
    return final_state, np.asarray(metrics)


def summarize_regression_metrics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize a per-step MSE curve."""
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
    }


def summarize_digits_metrics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize per-step MSE and accuracy."""
    window = min(final_window, metrics.shape[0])
    return {
        "online_mean_mse": float(np.mean(metrics[:, 0])),
        "online_mean_accuracy": float(np.mean(metrics[:, 1])),
        "final_window_mse": float(np.mean(metrics[-window:, 0])),
        "final_window_accuracy": float(np.mean(metrics[-window:, 1])),
    }


def evaluate_digits_classifier(
    learner: Any,
    state: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a final learner state on the held-out digits split."""
    observations = jnp.asarray(x_test.astype(np.float32))
    targets = jnp.asarray(np.eye(N_DIGIT_CLASSES, dtype=np.float32)[y_test])
    labels = jnp.asarray(y_test.astype(np.int32))
    predictions = jax.vmap(lambda obs: learner.predict(state, obs))(observations)
    mse = jnp.mean((predictions - targets) ** 2)
    accuracy = jnp.mean((jnp.argmax(predictions, axis=1) == labels).astype(jnp.float32))
    mse.block_until_ready()
    return {"test_mse": float(mse), "test_accuracy": float(accuracy)}


def summarize_numeric_array(values: Any) -> dict[str, float]:
    """Return JSON-safe distribution statistics for a numeric array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def activation_rank_diagnostics(
    learner: Any,
    state: Any,
    observations: jax.Array,
    max_samples: int = 512,
) -> dict[str, float]:
    """Compute stable/effective rank of final hidden activations."""
    if not isinstance(learner, UPGDLearner):
        return {}
    if len(state.trunk_params.weights) == 0:
        return {"activation_stable_rank": 0.0, "activation_effective_rank": 0.0}
    sample = observations[-min(max_samples, int(observations.shape[0])) :]
    cfg = learner.to_config()
    slope = cfg["leaky_relu_slope"]
    use_layer_norm = cfg["use_layer_norm"]
    hidden = jax.vmap(
        lambda obs: UPGDLearner._trunk_forward(
            state.trunk_params.weights,
            state.trunk_params.biases,
            obs,
            slope,
            use_layer_norm,
        )
    )(sample)
    hidden_np = np.asarray(hidden, dtype=np.float64)
    hidden_np = hidden_np - np.mean(hidden_np, axis=0, keepdims=True)
    if hidden_np.size == 0:
        return {"activation_stable_rank": 0.0, "activation_effective_rank": 0.0}
    singular_values = np.linalg.svd(hidden_np, compute_uv=False)
    if singular_values.size == 0 or float(singular_values[0]) <= 1e-12:
        return {"activation_stable_rank": 0.0, "activation_effective_rank": 0.0}
    squared = singular_values**2
    stable_rank = float(np.sum(squared) / (np.max(squared) + 1e-12))
    probs = singular_values / (np.sum(singular_values) + 1e-12)
    effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    return {
        "activation_stable_rank": stable_rank,
        "activation_effective_rank": effective_rank,
    }


def collect_upgd_diagnostics(
    learner: Any,
    state: Any,
    observations: jax.Array,
) -> dict[str, Any]:
    """Collect UPGD state diagnostics requested by the Step 2 evidence gate."""
    if not isinstance(learner, UPGDLearner):
        return {}
    cfg = learner.to_config()
    beta = float(cfg["perturbation_beta"])
    utility_layers = [np.asarray(layer) for layer in state.utilities]
    energy_fractions: list[float] = []
    for layer in utility_layers:
        flat = layer.reshape(-1)
        if flat.size == 0:
            energy_fractions.append(0.0)
            continue
        threshold = np.percentile(flat, 25)
        u_norm = flat / (np.max(flat) + 1e-12)
        expected_scale = np.maximum(1.0 - u_norm, 0.0) ** beta
        expected_energy = expected_scale**2
        denom = float(np.sum(expected_energy))
        if denom <= 1e-12:
            energy_fractions.append(0.0)
        else:
            energy_fractions.append(
                float(np.sum(expected_energy[flat <= threshold]) / denom)
            )
    diagnostics: dict[str, Any] = {
        "utility_by_layer": [
            summarize_numeric_array(layer) for layer in utility_layers
        ],
        "unit_utility_by_layer": [
            summarize_numeric_array(layer) for layer in state.unit_utilities
        ],
        "low_utility_perturbation_energy_fraction_by_layer": energy_fractions,
        "learned_controls": {
            "adaptive_kappa_multiplier": float(
                np.exp(np.asarray(state.adaptive_kappa_log_scale))
            ),
            "meta_trunk_multiplier": float(
                np.exp(np.asarray(state.meta_trunk_log_scale))
            ),
            "meta_head_weight_multiplier": float(
                np.exp(np.asarray(state.meta_head_weight_log_scale))
            ),
            "meta_head_bias_multiplier": float(
                np.exp(np.asarray(state.meta_head_bias_log_scale))
            ),
            "meta_repetition_multiplier": float(
                np.exp(np.asarray(state.meta_repetition_log_scale))
            ),
            "target_repeat_ema": float(np.asarray(state.target_repeat_ema)),
            "target_simplex_ema": float(np.asarray(state.target_simplex_ema)),
            "loss_fast_ema": float(np.asarray(state.loss_fast_ema)),
            "loss_slow_ema": float(np.asarray(state.loss_slow_ema)),
        },
        "unit_replacement_counts": [
            float(x) for x in np.asarray(state.unit_replacement_counts).reshape(-1)
        ],
        "unit_age_by_layer": [
            summarize_numeric_array(layer) for layer in state.unit_ages
        ],
        "unit_replacement_accumulators": [
            float(x)
            for x in np.asarray(state.unit_replacement_accumulators).reshape(-1)
        ],
    }
    diagnostics.update(activation_rank_diagnostics(learner, state, observations))
    return diagnostics


def load_digits_arrays(
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and standardize sklearn digits using train-split statistics only."""
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        msg = (
            "scikit-learn is required for the digits suite. "
            "Install with `pip install -e '.[external]'`."
        )
        raise RuntimeError(msg) from exc

    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float32) / 16.0
    y = np.asarray(digits.target, dtype=np.int32)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for cls in range(N_DIGIT_CLASSES):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(train_fraction * len(cls_idx)))
        train_indices.extend(cls_idx[:n_train].tolist())
        test_indices.extend(cls_idx[n_train:].tolist())

    train_indices_arr = np.asarray(train_indices, dtype=np.int32)
    test_indices_arr = np.asarray(test_indices, dtype=np.int32)
    rng.shuffle(train_indices_arr)
    rng.shuffle(test_indices_arr)

    x_train = x[train_indices_arr]
    y_train = y[train_indices_arr]
    x_test = x[test_indices_arr]
    y_test = y[test_indices_arr]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    meta = {
        "dataset": "sklearn.datasets.load_digits",
        "n_total": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": N_DIGIT_CLASSES,
        "train_fraction": train_fraction,
        "split_seed": seed,
    }
    return x_train, y_train, x_test, y_test, meta


def make_digits_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    class_blocked: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create an online stream from repeated shuffled or class-blocked epochs."""
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    total = 0
    class_indices = [np.flatnonzero(y_train == cls) for cls in range(N_DIGIT_CLASSES)]

    while total < steps:
        if class_blocked:
            order = rng.permutation(N_DIGIT_CLASSES)
            epoch_parts: list[np.ndarray] = []
            for cls in order:
                cls_idx = class_indices[int(cls)].copy()
                rng.shuffle(cls_idx)
                epoch_parts.append(cls_idx)
            indices = np.concatenate(epoch_parts)
        else:
            indices = rng.permutation(len(y_train))
        chunks_x.append(x_train[indices])
        chunks_y.append(y_train[indices])
        total += len(indices)

    observations = np.concatenate(chunks_x, axis=0)[:steps].astype(np.float32)
    labels = np.concatenate(chunks_y, axis=0)[:steps].astype(np.int32)
    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float32)[labels]
    return jnp.asarray(observations), jnp.asarray(targets), jnp.asarray(labels)


def make_digits_regime_sequence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int,
    seed: int,
    regime: str,
    phase_length: int,
    mask_keep_fraction: float,
    mask_noise_std: float,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Create one named digits stream for the UPGD promotion matrix."""
    if phase_length <= 0:
        raise ValueError("--phase-length must be positive")
    if not 0.0 < mask_keep_fraction <= 1.0:
        raise ValueError("--mask-keep-fraction must be in (0, 1]")

    class_blocked = regime == "class_blocked"
    observations_jax, _, labels_jax = make_digits_sequence(
        x_train=x_train,
        y_train=y_train,
        steps=steps,
        seed=seed,
        class_blocked=class_blocked,
    )
    observations = np.asarray(observations_jax).astype(np.float32)
    labels = np.asarray(labels_jax).astype(np.int32)
    rng = np.random.default_rng(seed + 50_000)
    phase_ids = np.arange(steps, dtype=np.int32) // phase_length
    n_phases = int(phase_ids[-1]) + 1
    meta: dict[str, Any] = {
        "regime": regime,
        "phase_length": int(phase_length),
        "n_phases": int(n_phases),
    }

    if regime == "iid":
        meta["description"] = "stationary shuffled IID digits stream"
    elif regime == "class_blocked":
        meta["description"] = "digits grouped into class blocks"
    elif regime == "label_drift":
        permutations = [np.arange(N_DIGIT_CLASSES, dtype=np.int32)]
        permutations.extend(
            rng.permutation(N_DIGIT_CLASSES).astype(np.int32)
            for _ in range(1, n_phases)
        )
        for phase, permutation in enumerate(permutations):
            labels[phase_ids == phase] = permutation[labels[phase_ids == phase]]
        meta["description"] = "class-head labels are permuted by phase"
    elif regime == "permuted":
        permutations = [np.arange(observations.shape[1], dtype=np.int32)]
        permutations.extend(
            rng.permutation(observations.shape[1]).astype(np.int32)
            for _ in range(1, n_phases)
        )
        for phase, permutation in enumerate(permutations):
            observations[phase_ids == phase] = observations[phase_ids == phase][:, permutation]
        meta["description"] = "pixel order is permuted by phase"
    elif regime == "mask_noise":
        feature_dim = observations.shape[1]
        n_keep = max(1, min(feature_dim, int(round(feature_dim * mask_keep_fraction))))
        masks = []
        for _ in range(n_phases):
            keep = rng.choice(feature_dim, size=n_keep, replace=False)
            mask = np.zeros(feature_dim, dtype=np.float32)
            mask[keep] = 1.0
            masks.append(mask)
        for phase, mask in enumerate(masks):
            phase_mask = phase_ids == phase
            phase_obs = observations[phase_mask] * mask
            if mask_noise_std > 0.0:
                phase_obs = phase_obs + rng.normal(
                    0.0,
                    mask_noise_std,
                    size=phase_obs.shape,
                ).astype(np.float32)
                phase_obs = phase_obs * mask
            observations[phase_mask] = phase_obs
        meta.update({
            "description": "feature masks with masked additive noise by phase",
            "mask_keep_fraction": float(mask_keep_fraction),
            "mask_noise_std": float(mask_noise_std),
        })
    else:
        raise ValueError(f"unknown digits regime: {regime}")

    targets = np.eye(N_DIGIT_CLASSES, dtype=np.float32)[labels]
    return (
        jnp.asarray(observations),
        jnp.asarray(targets),
        jnp.asarray(labels),
        meta,
    )


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize a Step 2 stream into observation/target arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    observations.block_until_ready()
    targets.block_until_ready()
    return observations, targets


def synthetic_stream_factories() -> dict[str, tuple[Any, int, int]]:
    """Return named synthetic out-of-class stream factories."""
    return {
        "polynomial": (
            lambda: OutOfClassPolynomialStream(
                feature_dim=8,
                n_tasks=3,
                n_contexts=4,
                context_length=400,
                active_triples_per_context=2,
                noise_std=0.05,
            ),
            8,
            3,
        ),
        "frequency": (
            lambda: FrequencyMismatchStream(
                feature_dim=4,
                n_tasks=2,
                n_components_per_task=3,
                n_contexts=4,
                context_length=400,
                noise_std=0.05,
            ),
            4,
            2,
        ),
        "compositional": (
            lambda: CompositionalStream(
                feature_dim=6,
                n_tasks=3,
                inner_hidden=4,
                outer_components=5,
                n_contexts=4,
                context_length=400,
                noise_std=0.05,
            ),
            6,
            3,
        ),
    }


def stderr(values: list[float]) -> float:
    """Standard error of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0]))


def mean(values: list[float]) -> float:
    """Mean for JSON-safe float lists."""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def metric_value(record: dict[str, Any], metric: str) -> float:
    """Read one metric from a flat run record."""
    return float(record["metrics"][metric])


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate records by scenario and method."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], record["method"])] .append(record)

    aggregate: dict[str, dict[str, Any]] = defaultdict(dict)
    for (scenario, method), group in sorted(grouped.items()):
        metric_names = sorted(group[0]["metrics"].keys())
        method_summary: dict[str, Any] = {
            "method_config": group[0]["method_config"],
            "n_seeds": len(group),
        }
        for metric in metric_names:
            vals = [metric_value(record, metric) for record in group]
            method_summary[f"{metric}_mean"] = mean(vals)
            method_summary[f"{metric}_stderr"] = stderr(vals)
            method_summary[f"{metric}_min"] = float(np.min(vals))
            method_summary[f"{metric}_max"] = float(np.max(vals))
        aggregate[scenario][method] = method_summary
    return dict(aggregate)


def build_paired_vs_best_mlp(
    records: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool,
) -> dict[str, dict[str, Any]]:
    """Build paired comparisons against the best fair MLP for each scenario."""
    scenario_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_records[record["scenario"]].append(record)

    paired: dict[str, dict[str, Any]] = {}
    for scenario, group in sorted(scenario_records.items()):
        baseline_names = [cfg.name for cfg in BASELINE_CONFIGS]
        if higher_is_better:
            best_mlp = max(
                baseline_names,
                key=lambda name: aggregate[scenario][name][f"{primary_metric}_mean"],
            )
        else:
            best_mlp = min(
                baseline_names,
                key=lambda name: aggregate[scenario][name][f"{primary_metric}_mean"],
            )

        by_method_seed: dict[tuple[str, int], dict[str, Any]] = {}
        for record in group:
            by_method_seed[(record["method"], int(record["seed"]))] = record
        seeds = sorted(int(record["seed"]) for record in group if record["method"] == best_mlp)

        scenario_paired: dict[str, Any] = {"_best_mlp": best_mlp}
        for method in sorted({record["method"] for record in group}):
            if method == best_mlp:
                continue
            diffs: list[float] = []
            for seed in seeds:
                baseline_record = by_method_seed[(best_mlp, seed)]
                method_record = by_method_seed[(method, seed)]
                best_value = metric_value(baseline_record, primary_metric)
                method_value_i = metric_value(method_record, primary_metric)
                diff = (
                    method_value_i - best_value
                    if higher_is_better
                    else best_value - method_value_i
                )
                diffs.append(diff)
            diff_arr = np.asarray(diffs, dtype=np.float64)
            sd = float(np.std(diff_arr, ddof=1)) if diff_arr.shape[0] > 1 else 0.0
            scenario_paired[method] = {
                "metric": primary_metric,
                "positive_means_method_beats_best_mlp": True,
                "mean_diff": float(np.mean(diff_arr)),
                "stderr": stderr(diffs),
                "wins": int(np.sum(diff_arr > 0.0)),
                "losses": int(np.sum(diff_arr < 0.0)),
                "ties": int(np.sum(diff_arr == 0.0)),
                "n_seeds": int(diff_arr.shape[0]),
                "cohens_d": float(np.mean(diff_arr) / sd) if sd > 0.0 else 0.0,
            }
        paired[scenario] = scenario_paired
    return paired


def config_to_json(config: MethodConfig) -> dict[str, Any]:
    """Serialize a dataclass config with tuple fields as lists."""
    data = asdict(config)
    data["hidden_sizes"] = list(config.hidden_sizes)
    return data


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize argparse results with paths converted to strings."""
    data = vars(args).copy()
    data["output_dir"] = str(data["output_dir"])
    return data


def run_digits_suite(args: argparse.Namespace, upgd_configs: list[MethodConfig]) -> dict[str, Any]:
    """Run the digits ablation suite."""
    records: list[dict[str, Any]] = []
    if args.digits_regimes:
        regimes = parse_csv(args.digits_regimes)
    else:
        regimes = {
            "shuffled": ["iid"],
            "blocked": ["class_blocked"],
            "both": ["iid", "class_blocked"],
        }[args.class_blocked_mode]
    valid_regimes = {"iid", "permuted", "class_blocked", "label_drift", "mask_noise"}
    unknown_regimes = sorted(set(regimes) - valid_regimes)
    if unknown_regimes:
        raise ValueError(f"unknown digits regimes: {unknown_regimes}")
    t0 = time.time()
    dataset_meta: dict[str, Any] | None = None
    regime_meta: dict[str, Any] = {}

    for regime in regimes:
        scenario = f"digits_{regime}"
        print(f"\n=== {scenario}: seeds={args.n_seeds}, steps={args.steps} ===")
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            x_train, y_train, x_test, y_test, dataset_meta = load_digits_arrays(
                seed=seed,
                train_fraction=args.train_fraction,
            )
            observations, targets, labels, meta = make_digits_regime_sequence(
                x_train=x_train,
                y_train=y_train,
                steps=args.steps,
                seed=seed + 10_000,
                regime=regime,
                phase_length=args.phase_length,
                mask_keep_fraction=args.mask_keep_fraction,
                mask_noise_std=args.mask_noise_std,
            )
            regime_meta.setdefault(scenario, meta)

            for config in [*BASELINE_CONFIGS, *upgd_configs]:
                learner = (
                    make_mlp(config, N_DIGIT_CLASSES)
                    if config.method_type == "mlp"
                    else make_upgd(config, N_DIGIT_CLASSES)
                )
                print(f"  seed={seed} method={config.name}")
                state, metrics = run_online_digits(
                    learner,
                    method_key(seed, config.name),
                    observations,
                    targets,
                    labels,
                )
                summary = summarize_digits_metrics(metrics, args.final_window)
                summary.update(evaluate_digits_classifier(learner, state, x_test, y_test))
                records.append({
                    "suite": "digits",
                    "scenario": scenario,
                    "seed": seed,
                    "method": config.name,
                    "method_config": config_to_json(config),
                    "metrics": summary,
                    "diagnostics": collect_upgd_diagnostics(
                        learner,
                        state,
                        observations,
                    ),
                })

    aggregate = aggregate_records(records)
    paired_final_mse = build_paired_vs_best_mlp(
        records,
        aggregate,
        primary_metric="final_window_mse",
        higher_is_better=False,
    )
    paired_test_accuracy = build_paired_vs_best_mlp(
        records,
        aggregate,
        primary_metric="test_accuracy",
        higher_is_better=True,
    )
    return {
        "suite": "digits",
        "config": args_to_json(args),
        "upgd_configs": [config_to_json(cfg) for cfg in upgd_configs],
        "dataset": dataset_meta,
        "regimes": regime_meta,
        "records": records,
        "aggregate": aggregate,
        "paired_vs_best_mlp": {
            "final_window_mse": paired_final_mse,
            "test_accuracy": paired_test_accuracy,
        },
        "wall_clock_s": time.time() - t0,
    }


def run_synthetic_suite(
    args: argparse.Namespace,
    upgd_configs: list[MethodConfig],
) -> dict[str, Any]:
    """Run the synthetic out-of-class ablation suite."""
    records: list[dict[str, Any]] = []
    requested_streams = parse_csv(args.streams)
    factories = synthetic_stream_factories()
    unknown = sorted(set(requested_streams) - set(factories))
    if unknown:
        raise ValueError(f"unknown synthetic streams: {unknown}")

    t0 = time.time()
    for stream_name in requested_streams:
        factory, feature_dim, n_tasks = factories[stream_name]
        scenario = f"synthetic_{stream_name}"
        print(f"\n=== {scenario}: seeds={args.n_seeds}, steps={args.steps} ===")
        for run_idx in range(args.n_seeds):
            seed = args.seed + run_idx
            stream = factory()
            observations, targets = collect_stream_arrays(
                stream,
                args.steps,
                method_key(seed, f"{stream_name}_stream"),
            )

            for config in [*BASELINE_CONFIGS, *upgd_configs]:
                learner = (
                    make_mlp(config, n_tasks)
                    if config.method_type == "mlp"
                    else make_upgd(config, n_tasks)
                )
                print(f"  seed={seed} stream={stream_name} method={config.name}")
                state, metrics = run_online_regression(
                    learner,
                    method_key(seed, f"{stream_name}_{config.name}"),
                    observations,
                    targets,
                )
                records.append({
                    "suite": "synthetic",
                    "scenario": scenario,
                    "stream": stream_name,
                    "seed": seed,
                    "method": config.name,
                    "method_config": config_to_json(config),
                    "metrics": summarize_regression_metrics(metrics, args.final_window),
                    "diagnostics": collect_upgd_diagnostics(
                        learner,
                        state,
                        observations,
                    ),
                })

    aggregate = aggregate_records(records)
    paired_final_mse = build_paired_vs_best_mlp(
        records,
        aggregate,
        primary_metric="final_window_mse",
        higher_is_better=False,
    )
    return {
        "suite": "synthetic",
        "config": args_to_json(args),
        "upgd_configs": [config_to_json(cfg) for cfg in upgd_configs],
        "records": records,
        "aggregate": aggregate,
        "paired_vs_best_mlp": {"final_window_mse": paired_final_mse},
        "wall_clock_s": time.time() - t0,
    }


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated CLI value."""
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_upgd_configs(args: argparse.Namespace) -> list[MethodConfig]:
    """Select UPGD configs from a preset, optionally filtered by name."""
    names = list(PRESET_CONFIGS[args.preset])
    if args.upgd_configs:
        requested = parse_csv(args.upgd_configs)
        unknown = sorted(set(requested) - set(UPGD_CATALOG))
        if unknown:
            raise ValueError(f"unknown UPGD configs: {unknown}")
        names = requested
    return [UPGD_CATALOG[name] for name in names]


def fmt_mean_stderr(row: dict[str, Any], metric: str) -> str:
    """Format an aggregate mean +/- stderr pair."""
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_stderr']:.4f}"


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a markdown summary next to the JSON output."""
    suite = results["suite"]
    aggregate = results["aggregate"]
    paired = results["paired_vs_best_mlp"]["final_window_mse"]

    lines: list[str] = [
        f"# Worker S2A UPGD {suite.title()} Ablation",
        "",
        f"Wall clock: {results['wall_clock_s']:.1f}s.",
        (
            f"Seeds: {results['config']['n_seeds']}; "
            f"steps: {results['config']['steps']}; "
            f"final window: {results['config']['final_window']}."
        ),
        "",
        "Every scenario includes fair `MLP(64)` and `MLP(64,64)` baselines. "
        "Positive paired differences mean the UPGD candidate beats the best "
        "fair MLP on final-window MSE for that scenario.",
        "",
    ]

    for scenario, by_method in sorted(aggregate.items()):
        best_mlp = paired[scenario]["_best_mlp"]
        lines.extend([
            f"## {scenario}",
            "",
            f"Best fair MLP by final-window MSE: `{best_mlp}`.",
            "",
            (
                "| Method | Hidden | kappa | loss norm | sigma | decay | beta | interval | "
                "sparsity | LN | Final-window MSE | Online mean MSE | Paired diff | Wins |"
            ),
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        ordered_methods = sorted(
            by_method,
            key=lambda method: by_method[method]["final_window_mse_mean"],
        )
        for method in ordered_methods:
            row = by_method[method]
            cfg = row["method_config"]
            paired_row = paired[scenario].get(method)
            diff = ""
            wins = ""
            if paired_row is not None:
                diff = f"{paired_row['mean_diff']:+.4f}"
                wins = f"{paired_row['wins']}/{paired_row['n_seeds']}"
            interval = (
                cfg["perturbation_interval"]
                if cfg["perturbation_interval"] is not None
                else ""
            )
            lines.append(
                f"| `{method}` | {cfg['hidden_sizes']} | "
                f"{cfg['bounder_kappa']} | {cfg['loss_normalization']} | "
                f"{cfg['perturbation_sigma'] if cfg['perturbation_sigma'] is not None else ''} | "
                f"{cfg['utility_decay'] if cfg['utility_decay'] is not None else ''} | "
                f"{cfg['perturbation_beta'] if cfg['perturbation_beta'] is not None else ''} | "
                f"{interval} | "
                f"{cfg['sparsity']} | {cfg['use_layer_norm']} | "
                f"{fmt_mean_stderr(row, 'final_window_mse')} | "
                f"{fmt_mean_stderr(row, 'online_mean_mse')} | {diff} | {wins} |"
            )

        if suite == "digits":
            acc_paired = results["paired_vs_best_mlp"]["test_accuracy"][scenario]
            lines.extend([
                "",
                "Held-out test accuracy:",
                "",
                "| Method | Test accuracy | Paired diff vs best MLP | Wins |",
                "|---|---:|---:|---:|",
            ])
            ordered_acc = sorted(
                by_method,
                key=lambda method: by_method[method]["test_accuracy_mean"],
                reverse=True,
            )
            for method in ordered_acc:
                row = by_method[method]
                acc_pair = acc_paired.get(method)
                diff = ""
                wins = ""
                if acc_pair is not None:
                    diff = f"{acc_pair['mean_diff']:+.4f}"
                    wins = f"{acc_pair['wins']}/{acc_pair['n_seeds']}"
                lines.append(
                    f"| `{method}` | {fmt_mean_stderr(row, 'test_accuracy')} | "
                    f"{diff} | {wins} |"
                )
        lines.append("")

    lines.extend([
        "## Candidate Rule",
        "",
        "A candidate deserves canonical follow-up only if it beats the best fair "
        "MLP on mean final-window MSE and wins a majority of paired seeds in at "
        "least one non-smoke scenario, without a compensating held-out accuracy "
        "loss on digits.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("digits", "synthetic"), required=True)
    parser.add_argument("--preset", choices=tuple(PRESET_CONFIGS), default="smoke")
    parser.add_argument("--upgd-configs", default="")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=400)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--class-blocked-mode",
        choices=("shuffled", "blocked", "both"),
        default="both",
        help=(
            "Backward-compatible digits selector used only when "
            "--digits-regimes is empty."
        ),
    )
    parser.add_argument(
        "--digits-regimes",
        default="",
        help=(
            "Comma-separated digits regimes. Known regimes: iid, permuted, "
            "class_blocked, label_drift, mask_noise. Overrides "
            "--class-blocked-mode."
        ),
    )
    parser.add_argument(
        "--phase-length",
        type=int,
        default=500,
        help="Phase length for permuted, label_drift, and mask_noise digits regimes.",
    )
    parser.add_argument(
        "--mask-keep-fraction",
        type=float,
        default=0.5,
        help="Fraction of input features retained in each mask_noise phase.",
    )
    parser.add_argument(
        "--mask-noise-std",
        type=float,
        default=0.05,
        help="Additive noise standard deviation for retained mask_noise features.",
    )
    parser.add_argument(
        "--streams",
        default="polynomial,frequency,compositional",
        help="Comma-separated synthetic stream names.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the selected ablation suite and write JSON/Markdown outputs."""
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    upgd_configs = selected_upgd_configs(args)

    results = (
        run_digits_suite(args, upgd_configs)
        if args.suite == "digits"
        else run_synthetic_suite(args, upgd_configs)
    )

    json_path = output_dir / f"{args.suite}_ablation_results.json"
    md_path = output_dir / f"{args.suite}_ablation_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(md_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

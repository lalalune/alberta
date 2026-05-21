#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined,import-untyped,no-any-return,no-untyped-call"
"""Build and run a pure-deployable DiffEML evidence suite.

This suite is deliberately narrower than the image demo. It focuses on evidence
that survives hardening into an EML-derived Boolean circuit with a pure Boolean
readout. Training may use differentiable relaxations, but reported evidence must
include the soft-vs-hard gap, packed hard agreement, circuit byte count, and
baseline columns.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax.random as jr
import numpy as np

from alberta_framework.core import diffeml_image
from alberta_framework.core.diffeml import (
    BOOLEAN_GATE_NAMES,
    DiffEMLGateSelector,
    boolean_truth_table,
)

DEMO_SCRIPT = Path(__file__).with_name("step2_diffeml_image_demo.py")
DEFAULT_OUTPUT = Path("outputs/diffeml_image_demo/pure_eml_scale_suite_smoke.json")
PURE_READOUT_MODES = ("group_sum", "class_vote", "signed_class_vote")

Scale = Literal["smoke", "full"]
RunMode = Literal["matrix", "boolean", "continuous", "images", "all"]


@dataclass(frozen=True)
class BooleanGateSpec:
    """One exact two-input Boolean gate-learning run."""

    run_id: str
    target_mask: int
    seed: int
    num_steps: int
    step_size: float = 0.1
    initial_temperature: float = 1.0
    min_temperature: float = 0.03
    entropy_weight: float = 0.002


@dataclass(frozen=True)
class ContinuousThresholdSpec:
    """One synthetic thresholded continuous-function run."""

    run_id: str
    task: str
    seed: int
    train_samples: int
    test_samples: int
    input_dim: int
    pixel_thresholds: int
    input_bits: int
    layers: int
    width: int
    epochs: int
    batch_size: int
    head_mode: str = "class_vote"
    wiring_mode: str = "random"
    group_sum_tau: float = 4.0
    readout_entropy_weight: float = 0.01
    readout_balance_weight: float = 1.0


@dataclass(frozen=True)
class ImageSmokeSpec:
    """One planned real-image pure-readout smoke run."""

    run_id: str
    dataset: str
    seed: int
    max_train: int
    max_test: int
    feature_mode: str
    input_bits: int
    pixel_thresholds: int
    layers: int
    width: int
    epochs: int
    batch_size: int
    head_mode: str = "class_vote"
    wiring_mode: str = "random"
    compare_mlp: bool = True
    group_sum_tau: float = 10.0
    readout_entropy_weight: float = 0.01
    readout_balance_weight: float = 5.0


def json_default(value: Any) -> Any:
    """Convert NumPy, path, and dataclass-adjacent values for JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list."""
    try:
        seeds = tuple(int(part) for part in text.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must contain integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def build_boolean_specs(*, scale: Scale, seeds: tuple[int, ...]) -> tuple[BooleanGateSpec, ...]:
    """Return exact Boolean functions to train as hard EML gates."""
    masks = tuple(range(16)) if scale == "full" else (6, 8, 9, 14)
    steps = 80 if scale == "full" else 60
    return tuple(
        BooleanGateSpec(
            run_id=f"boolean_{BOOLEAN_GATE_NAMES[mask].lower()}_seed{seed}",
            target_mask=mask,
            seed=seed,
            num_steps=steps,
        )
        for seed in seeds
        for mask in masks
    )


def build_continuous_specs(
    *,
    scale: Scale,
    seeds: tuple[int, ...],
) -> tuple[ContinuousThresholdSpec, ...]:
    """Return small continuous-domain tasks that become Boolean after thresholds."""
    tasks: tuple[str, ...]
    topology_templates: tuple[tuple[str, str], ...]
    if scale == "full":
        tasks = ("xor_quadrants", "diagonal_halfspace", "checkerboard4")
        train_samples = 1024
        test_samples = 512
        width = 256
        layers = 4
        epochs = 15
        batch_size = 128
        input_bits = 8
        pixel_thresholds = 4
        topology_templates = (
            ("random", "class_vote"),
            ("affine_expander", "class_vote"),
            ("butterfly_class_bank", "group_sum"),
        )
    else:
        tasks = ("xor_quadrants", "diagonal_halfspace")
        train_samples = 128
        test_samples = 64
        width = 64
        layers = 2
        epochs = 2
        batch_size = 32
        input_bits = 6
        pixel_thresholds = 3
        topology_templates = (
            ("random", "class_vote"),
            ("affine_expander", "class_vote"),
        )
    return tuple(
        ContinuousThresholdSpec(
            run_id=(
                f"continuous_{task}_{wiring_mode}_{head_mode}"
                f"_w{width}_l{layers}_seed{seed}"
            ),
            task=task,
            seed=seed,
            train_samples=train_samples,
            test_samples=test_samples,
            input_dim=2,
            pixel_thresholds=pixel_thresholds,
            input_bits=input_bits,
            layers=layers,
            width=width,
            epochs=epochs,
            batch_size=batch_size,
            head_mode=head_mode,
            wiring_mode=wiring_mode,
        )
        for seed in seeds
        for task in tasks
        for wiring_mode, head_mode in topology_templates
    )


def build_image_specs(*, scale: Scale, seeds: tuple[int, ...]) -> tuple[ImageSmokeSpec, ...]:
    """Return real-image smoke configs that keep deployment pure Boolean."""
    templates: tuple[tuple[str, int, int, str, int, int, int, int, int, str, str], ...]
    if scale == "full":
        templates = (
            ("digits", 1200, 300, "threshold_pixels", 64, 1, 4, 512, 15, "random", "class_vote"),
            (
                "digits",
                1200,
                300,
                "threshold_pixels",
                64,
                1,
                4,
                512,
                15,
                "affine_expander",
                "class_vote",
            ),
            (
                "digits",
                1200,
                300,
                "threshold_pixels",
                64,
                1,
                4,
                512,
                15,
                "butterfly_class_bank",
                "group_sum",
            ),
            (
                "mnist",
                60000,
                10000,
                "threshold_pixels",
                784,
                1,
                6,
                2048,
                10,
                "affine_expander",
                "class_vote",
            ),
            (
                "cifar",
                20000,
                5000,
                "detector_thresholds",
                24576,
                4,
                6,
                2048,
                10,
                "affine_expander",
                "class_vote",
            ),
        )
        batch_size = 128
    else:
        templates = (
            ("digits", 256, 128, "threshold_pixels", 64, 1, 2, 96, 2, "random", "class_vote"),
            (
                "digits",
                256,
                128,
                "threshold_pixels",
                64,
                1,
                2,
                96,
                2,
                "affine_expander",
                "class_vote",
            ),
            (
                "digits",
                256,
                128,
                "threshold_pixels",
                64,
                1,
                2,
                100,
                2,
                "butterfly_class_bank",
                "group_sum",
            ),
            (
                "cifar",
                1200,
                300,
                "detector_thresholds",
                512,
                2,
                3,
                512,
                3,
                "affine_expander",
                "class_vote",
            ),
        )
        batch_size = 64
    return tuple(
        ImageSmokeSpec(
            run_id=(
                f"image_{dataset}_{feature_mode}_{wiring_mode}_{head_mode}"
                f"_w{width}"
                f"_l{layers}_seed{seed}"
            ),
            dataset=dataset,
            seed=seed,
            max_train=max_train,
            max_test=max_test,
            feature_mode=feature_mode,
            input_bits=input_bits,
            pixel_thresholds=pixel_thresholds,
            layers=layers,
            width=width,
            epochs=epochs,
            batch_size=batch_size,
            head_mode=head_mode,
            wiring_mode=wiring_mode,
        )
        for seed in seeds
        for (
            dataset,
            max_train,
            max_test,
            feature_mode,
            input_bits,
            pixel_thresholds,
            layers,
            width,
            epochs,
            wiring_mode,
            head_mode,
        ) in templates
    )


def _base_demo_config(
    *,
    datasets: tuple[str, ...],
    seed: int,
    max_train: int,
    max_test: int,
    feature_mode: str,
    input_bits: int,
    pixel_thresholds: int,
    layers: int,
    width: int,
    epochs: int,
    batch_size: int,
    head_mode: str,
    wiring_mode: str,
    group_sum_tau: float,
    readout_entropy_weight: float,
    readout_balance_weight: float,
    compare_mlp: bool,
) -> diffeml_image.DemoConfig:
    """Build a DemoConfig constrained to hard deployable EML evidence."""
    return diffeml_image.DemoConfig(
        datasets=datasets,
        seed=seed,
        train_fraction=0.8,
        max_train=max_train,
        max_test=max_test,
        feature_mode=feature_mode,
        input_bits=input_bits,
        pixel_thresholds=pixel_thresholds,
        layers=layers,
        width=width,
        wiring_mode=wiring_mode,
        local_patch_size=3,
        tree_stage_depths=(1, 1),
        epochs=epochs,
        batch_size=batch_size,
        step_size=0.01,
        initial_temperature=1.0,
        min_temperature=0.05,
        entropy_weight=0.005,
        head_l2=0.0,
        gate_init_scale=0.2,
        head_init_scale=0.2,
        max_grad_norm=10.0,
        eml_template_depth=2,
        eml_eps=0.05,
        gate_mode="eml_template",
        eml_threshold_temperature=0.75,
        threshold_init_scale=0.1,
        direction_init_scale=0.2,
        hard_loss_weight=1.0,
        input_drop_rate=0.0,
        feature_drop_rate=0.0,
        residual_gate="none",
        residual_gate_bias=0.0,
        head_mode=head_mode,
        group_sum_tau=group_sum_tau,
        readout_entropy_weight=readout_entropy_weight,
        readout_balance_weight=readout_balance_weight,
        packed_eval=True,
        compare_mlp=compare_mlp,
        mlp_hidden_sizes=(128,),
        mlp_epochs=epochs,
        mlp_step_size=0.001,
        mlp_weight_decay=1e-4,
        mlp_max_grad_norm=10.0,
        mlp_init_scale=1.0,
    )


def continuous_config(spec: ContinuousThresholdSpec) -> diffeml_image.DemoConfig:
    """Return the existing image-circuit config for one synthetic threshold task."""
    return _base_demo_config(
        datasets=(spec.task,),
        seed=spec.seed,
        max_train=spec.train_samples,
        max_test=spec.test_samples,
        feature_mode="threshold_pixels",
        input_bits=spec.input_bits,
        pixel_thresholds=spec.pixel_thresholds,
        layers=spec.layers,
        width=spec.width,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        head_mode=spec.head_mode,
        wiring_mode=spec.wiring_mode,
        group_sum_tau=spec.group_sum_tau,
        readout_entropy_weight=spec.readout_entropy_weight,
        readout_balance_weight=spec.readout_balance_weight,
        compare_mlp=False,
    )


def image_config(spec: ImageSmokeSpec) -> diffeml_image.DemoConfig:
    """Return the existing image-circuit config for one real-image smoke run."""
    return _base_demo_config(
        datasets=(spec.dataset,),
        seed=spec.seed,
        max_train=spec.max_train,
        max_test=spec.max_test,
        feature_mode=spec.feature_mode,
        input_bits=spec.input_bits,
        pixel_thresholds=spec.pixel_thresholds,
        layers=spec.layers,
        width=spec.width,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        head_mode=spec.head_mode,
        wiring_mode=spec.wiring_mode,
        group_sum_tau=spec.group_sum_tau,
        readout_entropy_weight=spec.readout_entropy_weight,
        readout_balance_weight=spec.readout_balance_weight,
        compare_mlp=spec.compare_mlp,
    )


def anti_larp_config_checks(config: diffeml_image.DemoConfig) -> dict[str, Any]:
    """Return static checks that prevent the common DiffEML evidence mistakes."""
    flags: list[str] = []
    if config.head_mode not in PURE_READOUT_MODES:
        flags.append("non_pure_readout")
    if config.gate_mode != "eml_template":
        flags.append("not_executable_eml_template_gate_mode")
    if not config.packed_eval:
        flags.append("missing_packed_hard_eval")
    if config.head_l2 != 0.0 and config.head_mode in PURE_READOUT_MODES:
        flags.append("irrelevant_float_head_regularization")
    return {
        "pure_readout_only": config.head_mode in PURE_READOUT_MODES,
        "executable_eml_templates": config.gate_mode == "eml_template",
        "packed_hard_eval_required": config.packed_eval,
        "linear_head_forbidden": config.head_mode != "linear",
        "flags": flags,
    }


def _flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _format_cli_value(value: object) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def image_demo_command(
    config: diffeml_image.DemoConfig,
    *,
    output: Path,
    data_dir: Path,
) -> str:
    """Return a reproducible direct image-demo command for a pure EML config."""
    args = ["python", str(DEMO_SCRIPT), "--datasets"]
    args.extend(str(dataset) for dataset in config.datasets)
    args.extend(("--data-dir", str(data_dir)))
    omitted = {"datasets", "packed_eval", "compare_mlp"}
    for field_name, value in asdict(config).items():
        if field_name in omitted:
            continue
        args.extend((_flag_name(field_name), _format_cli_value(value)))
    if config.packed_eval:
        args.append("--packed-eval")
    if config.compare_mlp:
        args.append("--compare-mlp")
    args.extend(("--output", str(output)))
    return shlex.join(args)


def _literal_baseline_masks() -> tuple[int, ...]:
    return (0, 3, 5, 10, 12, 15)


def majority_baseline_accuracy(labels: np.ndarray) -> float:
    """Return the accuracy of always predicting the majority class."""
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(labels)
    return float(np.max(counts) / labels.shape[0])


def best_literal_boolean_baseline(target: np.ndarray) -> dict[str, Any]:
    """Return the best constant/input/negated-input baseline for a truth table."""
    target = np.asarray(target, dtype=np.float32)
    best_mask = -1
    best_accuracy = -1.0
    for mask in _literal_baseline_masks():
        values = np.asarray(boolean_truth_table(mask), dtype=np.float32)
        accuracy = float(np.mean(values == target))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mask = mask
    return {
        "best_literal_mask": best_mask,
        "best_literal_name": BOOLEAN_GATE_NAMES[best_mask],
        "best_literal_accuracy": best_accuracy,
    }


def continuous_targets(x: np.ndarray, task: str) -> np.ndarray:
    """Return binary labels for a low-dimensional thresholded function."""
    if task == "xor_quadrants":
        labels = np.logical_xor(x[:, 0] > 0.5, x[:, 1] > 0.5)
    elif task == "diagonal_halfspace":
        labels = x[:, 0] + x[:, 1] > 1.0
    elif task == "checkerboard4":
        cells = np.floor(np.clip(x[:, :2], 0.0, 0.999999) * 4.0).astype(np.int32)
        labels = ((cells[:, 0] + cells[:, 1]) % 2) == 1
    else:
        raise ValueError(
            "task must be 'xor_quadrants', 'diagonal_halfspace', or 'checkerboard4'"
        )
    return labels.astype(np.int32)


def make_continuous_split(spec: ContinuousThresholdSpec) -> diffeml_image.DatasetSplit:
    """Build a deterministic synthetic split for a thresholded continuous task."""
    rng = np.random.default_rng(spec.seed)
    x_train = rng.uniform(0.0, 1.0, size=(spec.train_samples, spec.input_dim)).astype(
        np.float32
    )
    x_test = rng.uniform(0.0, 1.0, size=(spec.test_samples, spec.input_dim)).astype(
        np.float32
    )
    y_train = continuous_targets(x_train, spec.task)
    y_test = continuous_targets(x_test, spec.task)
    return diffeml_image.DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        meta={
            "dataset": f"synthetic/{spec.task}",
            "source": "generated",
            "train_examples": spec.train_samples,
            "test_examples": spec.test_samples,
            "num_features": spec.input_dim,
            "num_classes": 2,
        },
    )


def extract_anti_larp_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract deployment-first evidence metrics from one run result."""
    if "error" in result:
        return {"error": result["error"], "flags": ["run_failed"]}
    metrics = result["metrics"]
    model = result["model"]
    storage = model.get("compiled_storage", {})
    test_soft = float(metrics["test_soft_accuracy"])
    test_hard = float(metrics["test_hard_accuracy"])
    packed = metrics.get("packed_hard_test_accuracy")
    soft_vs_hard_gap = abs(test_soft - test_hard)
    packed_vs_hard_gap = None if packed is None else abs(float(packed) - test_hard)
    packed_prediction_disagreement = metrics.get(
        "test_hard_packed_prediction_disagreement"
    )
    flags: list[str] = []
    if model.get("head_mode") not in PURE_READOUT_MODES:
        flags.append("non_pure_readout")
    if packed is None:
        flags.append("missing_packed_hard_accuracy")
    elif packed_prediction_disagreement is not None:
        if float(packed_prediction_disagreement) > 0.0:
            flags.append("packed_hard_mismatch")
    elif packed_vs_hard_gap is not None and packed_vs_hard_gap > 1e-6:
        flags.append("packed_hard_mismatch")
    if soft_vs_hard_gap > 0.05:
        flags.append("large_soft_hard_gap")
    if int(storage.get("head_fp32_bytes", 1)) != 0:
        flags.append("float_head_deployed")
    return {
        "test_soft_accuracy": test_soft,
        "test_hard_accuracy": test_hard,
        "packed_hard_test_accuracy": packed,
        "soft_vs_hard_gap": soft_vs_hard_gap,
        "packed_vs_hard_gap": packed_vs_hard_gap,
        "pure_readout_only": model.get("head_mode") in PURE_READOUT_MODES,
        "no_float_head_deployed": int(storage.get("head_fp32_bytes", 1)) == 0,
        "compiled_packed_bytes": storage.get("compiled_packed_bytes"),
        "soft_train_bytes": storage.get("soft_train_bytes"),
        "soft_to_packed_compression": storage.get("soft_to_compiled_packed_compression"),
        "flags": flags,
    }


def run_boolean_case(spec: BooleanGateSpec) -> dict[str, Any]:
    """Train one selector and report the selected hard EML gate."""
    learner = DiffEMLGateSelector(
        step_size=spec.step_size,
        initial_temperature=spec.initial_temperature,
        min_temperature=spec.min_temperature,
        anneal_steps=spec.num_steps,
        entropy_weight=spec.entropy_weight,
    )
    state = learner.init(jr.key(spec.seed))
    target = boolean_truth_table(spec.target_mask)
    trained = learner.train_truth_table(state, target, num_steps=spec.num_steps)
    soft_truth_table = np.asarray(learner.predict_truth_table(trained.state))
    hard_truth_table = np.asarray(learner.predict_hard_truth_table(trained.state))
    target_np = np.asarray(target)
    soft_bits = (soft_truth_table >= 0.5).astype(np.float32)
    hard_accuracy = float(np.mean(hard_truth_table == target_np))
    soft_accuracy = float(np.mean(soft_bits == target_np))
    soft_vs_hard_gap = abs(soft_accuracy - hard_accuracy)
    literal_baseline = best_literal_boolean_baseline(target_np)
    majority_baseline = majority_baseline_accuracy(target_np.astype(np.int32))
    selected_mask = learner.selected_gate_mask(trained.state)
    expression = learner.selected_gate_expression(trained.state)
    return {
        "run_id": spec.run_id,
        "kind": "boolean_gate",
        "spec": asdict(spec),
        "target": {
            "mask": spec.target_mask,
            "name": BOOLEAN_GATE_NAMES[spec.target_mask],
            "truth_table": target_np.tolist(),
        },
        "selected": {
            "mask": selected_mask,
            "name": learner.selected_gate_name(trained.state),
            "expression": expression,
        },
        "metrics": {
            "soft_accuracy": soft_accuracy,
            "hard_accuracy": hard_accuracy,
            "soft_vs_hard_gap": soft_vs_hard_gap,
            "selected_probability": float(trained.metrics[-1, 5]),
            "selector_entropy": float(trained.metrics[-1, 4]),
            "circuit_byte_count": 1,
            "expanded_eml_threshold_ops": 3,
            "pure_readout_only": True,
            "no_float_head_deployed": True,
        },
        "baselines": {
            "majority_accuracy": majority_baseline,
            **literal_baseline,
        },
        "anti_larp": {
            "hard_matches_target": selected_mask == spec.target_mask,
            "uses_executable_eml_expression": "eml(" in expression
            or expression in {"0", "1", "A", "B"},
            "deployable_as_boolean_gate": True,
            "flags": [] if selected_mask == spec.target_mask else ["wrong_hard_gate"],
        },
    }


def run_continuous_case(spec: ContinuousThresholdSpec) -> dict[str, Any]:
    """Run one synthetic thresholded continuous task through the image circuit."""
    config = continuous_config(spec)
    split = make_continuous_split(spec)
    result = diffeml_image.run_one_dataset(spec.run_id, split, config)
    result["baselines"]["majority"] = {
        "test_accuracy": majority_baseline_accuracy(split.y_test),
        "train_accuracy": majority_baseline_accuracy(split.y_train),
    }
    return {
        "run_id": spec.run_id,
        "kind": "continuous_threshold",
        "spec": asdict(spec),
        "config": asdict(config),
        "result": result,
        "anti_larp": extract_anti_larp_metrics(result),
    }


def run_image_case(spec: ImageSmokeSpec, *, data_dir: Path) -> dict[str, Any]:
    """Run one real-image smoke config through the existing image demo helpers."""
    config = image_config(spec)
    payload = diffeml_image.run_demo(config, data_dir)
    results = payload["results"]
    anti_larp = [
        extract_anti_larp_metrics(result) if "error" not in result else {"error": result["error"]}
        for result in results
    ]
    return {
        "run_id": spec.run_id,
        "kind": "image_smoke",
        "spec": asdict(spec),
        "config": payload["config"],
        "results": results,
        "anti_larp": anti_larp,
    }


def build_matrix(
    *,
    scale: Scale,
    seeds: tuple[int, ...],
    data_dir: Path = Path("outputs/diffeml_image_data"),
    run_output_dir: Path = Path("outputs/diffeml_image_demo/pure_eml_scale_suite"),
) -> dict[str, Any]:
    """Build a concrete, reproducible pure-EML experiment matrix."""
    boolean_specs = build_boolean_specs(scale=scale, seeds=seeds)
    continuous_specs = build_continuous_specs(scale=scale, seeds=seeds)
    image_specs = build_image_specs(scale=scale, seeds=seeds)
    continuous_rows = []
    for continuous_spec in continuous_specs:
        config = continuous_config(continuous_spec)
        continuous_rows.append(
            {
                **asdict(continuous_spec),
                "config_checks": anti_larp_config_checks(config),
                "baseline_columns": ("majority_accuracy",),
            }
        )
    image_rows = []
    for image_spec in image_specs:
        config = image_config(image_spec)
        output = run_output_dir / f"{image_spec.run_id}.json"
        image_rows.append(
            {
                **asdict(image_spec),
                "config_checks": anti_larp_config_checks(config),
                "baseline_columns": (
                    "majority_accuracy",
                    "same_feature_mlp_test_accuracy",
                ),
                "command": image_demo_command(config, output=output, data_dir=data_dir),
            }
        )
    return {
        "schema_version": "diffeml.pure_eml_scale_suite.v1",
        "created_at_unix_s": time.time(),
        "scale": scale,
        "seeds": seeds,
        "anti_larp_contract": {
            "deployment_must_use_hard_eml_template_selectors": True,
            "deployment_must_use_packed_hard_eval": True,
            "forbidden_deployed_readout": "linear",
            "allowed_pure_readouts": PURE_READOUT_MODES,
            "required_metrics": (
                "soft_vs_hard_gap",
                "packed_vs_hard_gap",
                "compiled_packed_bytes",
                "pure_readout_only",
                "baseline_columns",
            ),
        },
        "boolean_gate_specs": [asdict(spec) for spec in boolean_specs],
        "continuous_threshold_specs": continuous_rows,
        "image_smoke_specs": image_rows,
        "claim_rejection_rules": (
            "Reject any accuracy claim missing packed_hard_test_accuracy.",
            "Reject any promoted run with head_mode='linear'.",
            "Reject any run where packed hard accuracy differs from JAX hard accuracy.",
            "Treat soft accuracy as diagnostic only; hard/packed accuracy is the claim metric.",
            "Report majority and same-feature baselines before comparing to DiffLogic.",
        ),
    }


def run_suite(
    *,
    scale: Scale,
    seeds: tuple[int, ...],
    mode: RunMode,
    data_dir: Path,
) -> dict[str, Any]:
    """Run the selected part of the pure EML suite."""
    payload: dict[str, Any] = {
        "matrix": build_matrix(scale=scale, seeds=seeds, data_dir=data_dir),
        "runs": [],
    }
    if mode in {"boolean", "all"}:
        payload["runs"].extend(
            run_boolean_case(spec) for spec in build_boolean_specs(scale=scale, seeds=seeds)
        )
    if mode in {"continuous", "all"}:
        payload["runs"].extend(
            run_continuous_case(spec)
            for spec in build_continuous_specs(scale=scale, seeds=seeds)
        )
    if mode in {"images", "all"}:
        payload["runs"].extend(
            run_image_case(spec, data_dir=data_dir)
            for spec in build_image_specs(scale=scale, seeds=seeds)
        )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", type=parse_seeds, default=(0,))
    parser.add_argument(
        "--run",
        choices=("matrix", "boolean", "continuous", "images", "all"),
        default="matrix",
        help="matrix only emits configs; other modes run selected experiments.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/diffeml_image_data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Emit or run the pure EML scale suite."""
    args = parse_args()
    if args.run == "matrix":
        payload = build_matrix(scale=args.scale, seeds=args.seeds, data_dir=args.data_dir)
    else:
        payload = run_suite(
            scale=args.scale,
            seeds=args.seeds,
            mode=args.run,
            data_dir=args.data_dir,
        )
    text = json.dumps(payload, indent=2, default=json_default)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

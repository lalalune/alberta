#!/usr/bin/env python3
"""Emit and dry-run the DiffEML hard-synthesis experiment matrix.

This suite is a planning and provenance harness for the stricter DiffEML
direction: packed bitset gate synthesis, ECOC readouts, ANF sparse Boolean
polynomials, and decision-tree/BDD compilation. Backends for those mechanisms
may arrive independently, so this file imports them lazily and treats missing
backends as skipped run records rather than matrix-construction failures.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

SCRIPT_PATH = Path(__file__)
DEFAULT_OUTPUT = Path("outputs/diffeml_hard_synthesis/matrix_smoke.json")
DEFAULT_RUN_OUTPUT_DIR = Path("outputs/diffeml_hard_synthesis/runs")

Scale = Literal["smoke", "full"]
RunMode = Literal["matrix", "boolean", "continuous", "all"]
TaskKind = Literal["boolean", "continuous", "image_bits", "multiclass"]
SynthesisFamily = Literal[
    "packed_bitset_gate_synthesis",
    "ecoc_readout",
    "anf_sparse_boolean_polynomial",
    "tree_bdd_compilation",
]
BackendRunner = Callable[[dict[str, Any]], Mapping[str, Any]]

PRIMARY_METRICS = (
    "packed_hard_accuracy",
    "deployed_hard_accuracy",
    "soft_hard_gap",
    "compiled_gate_bytes",
    "eml_witness_coverage",
)

CLAIM_REJECTION_RULES = (
    {
        "rule_id": "no_float_head",
        "reject_if": "any deployed readout or classifier head uses floating weights",
    },
    {
        "rule_id": "hard_packed_metrics_primary",
        "reject_if": "the primary metric is soft accuracy instead of packed/deployed accuracy",
    },
    {
        "rule_id": "eml_witness_per_gate_mask",
        "reject_if": "any deployed gate mask lacks an executable EML witness",
    },
    {
        "rule_id": "same_feature_baselines_for_image_tasks",
        "reject_if": "image-bit tasks omit same-feature non-EML baselines",
    },
)

FAMILY_SHORT: dict[SynthesisFamily, str] = {
    "packed_bitset_gate_synthesis": "packed",
    "ecoc_readout": "ecoc",
    "anf_sparse_boolean_polynomial": "anf",
    "tree_bdd_compilation": "tree_bdd",
}

FAMILY_DESCRIPTIONS: dict[SynthesisFamily, str] = {
    "packed_bitset_gate_synthesis": (
        "Learn hard two-input gate masks and evaluate many examples with packed bitsets."
    ),
    "ecoc_readout": (
        "Compile multiclass prediction into binary code heads and Hamming-distance decode."
    ),
    "anf_sparse_boolean_polynomial": (
        "Search sparse algebraic normal form terms over GF(2), then compile XOR/AND gates."
    ),
    "tree_bdd_compilation": (
        "Fit thresholded decision structure and compile to an ordered BDD or tree circuit."
    ),
}

FAMILY_BACKENDS: dict[SynthesisFamily, tuple[tuple[str, str], ...]] = {
    "packed_bitset_gate_synthesis": (
        (
            "alberta_framework.core.diffeml_hard_synthesis",
            "run_packed_bitset_gate_synthesis",
        ),
        ("alberta_framework.core.diffeml_bitset", "run_experiment"),
    ),
    "ecoc_readout": (
        ("alberta_framework.core.diffeml_hard_synthesis", "run_ecoc_readout"),
        ("alberta_framework.core.diffeml_ecoc", "run_experiment"),
    ),
    "anf_sparse_boolean_polynomial": (
        ("alberta_framework.core.diffeml_hard_synthesis", "run_anf_sparse_polynomial"),
        ("alberta_framework.core.diffeml_anf", "run_experiment"),
    ),
    "tree_bdd_compilation": (
        ("alberta_framework.core.diffeml_hard_synthesis", "run_tree_bdd_compilation"),
        ("alberta_framework.core.diffeml_bdd", "run_experiment"),
    ),
}


@dataclass(frozen=True)
class TaskTemplate:
    """One task family used by the hard-synthesis matrix."""

    task_id: str
    task_kind: TaskKind
    description: str
    input_bits: int
    num_classes: int
    smoke_train_samples: int
    smoke_test_samples: int
    full_train_samples: int
    full_test_samples: int
    baseline_columns: tuple[str, ...]
    target_rule: str


@dataclass(frozen=True)
class HardSynthesisSpec:
    """One planned hard-synthesis experiment."""

    run_id: str
    scale: Scale
    family: SynthesisFamily
    task_id: str
    task_kind: TaskKind
    seed: int
    input_bits: int
    num_classes: int
    train_samples: int
    test_samples: int
    gate_budget: int
    packed_word_bits: int
    objective: str
    readout: str
    max_terms: int
    max_depth: int
    ecoc_bits: int
    required_modules: tuple[str, ...]
    baseline_columns: tuple[str, ...]
    required_metrics: tuple[str, ...]


@dataclass(frozen=True)
class BackendResolution:
    """Resolved optional backend runner."""

    module_name: str
    function_name: str
    runner: BackendRunner
    attempted: tuple[str, ...]


TASKS: dict[str, TaskTemplate] = {
    "xor": TaskTemplate(
        task_id="xor",
        task_kind="boolean",
        description="Two-bit XOR truth table.",
        input_bits=2,
        num_classes=2,
        smoke_train_samples=4,
        smoke_test_samples=4,
        full_train_samples=4,
        full_test_samples=4,
        baseline_columns=("majority_accuracy", "best_literal_accuracy"),
        target_rule="x0 xor x1",
    ),
    "diagonal_halfspace": TaskTemplate(
        task_id="diagonal_halfspace",
        task_kind="continuous",
        description="Thresholded two-dimensional halfspace: x0 + x1 > 1.",
        input_bits=8,
        num_classes=2,
        smoke_train_samples=128,
        smoke_test_samples=64,
        full_train_samples=4096,
        full_test_samples=2048,
        baseline_columns=("majority_accuracy", "linear_threshold_accuracy"),
        target_rule="1[x0 + x1 > 1]",
    ),
    "checkerboard": TaskTemplate(
        task_id="checkerboard",
        task_kind="continuous",
        description="Four-by-four checkerboard after thresholding continuous coordinates.",
        input_bits=16,
        num_classes=2,
        smoke_train_samples=256,
        smoke_test_samples=128,
        full_train_samples=8192,
        full_test_samples=4096,
        baseline_columns=("majority_accuracy", "linear_threshold_accuracy"),
        target_rule="parity(floor(4*x0), floor(4*x1))",
    ),
    "small_digits_even_odd_bits": TaskTemplate(
        task_id="small_digits_even_odd_bits",
        task_kind="image_bits",
        description="Sklearn digits threshold bits with even-versus-odd labels.",
        input_bits=64,
        num_classes=2,
        smoke_train_samples=256,
        smoke_test_samples=128,
        full_train_samples=1200,
        full_test_samples=500,
        baseline_columns=(
            "majority_accuracy",
            "same_feature_logistic_accuracy",
            "same_feature_mlp_accuracy",
        ),
        target_rule="digit mod 2",
    ),
    "small_digits_mod3_bits": TaskTemplate(
        task_id="small_digits_mod3_bits",
        task_kind="image_bits",
        description="Sklearn digits threshold bits with digit modulo three labels.",
        input_bits=64,
        num_classes=3,
        smoke_train_samples=256,
        smoke_test_samples=128,
        full_train_samples=1200,
        full_test_samples=500,
        baseline_columns=(
            "majority_accuracy",
            "same_feature_logistic_accuracy",
            "same_feature_mlp_accuracy",
        ),
        target_rule="digit mod 3",
    ),
    "multiclass_ecoc_toy": TaskTemplate(
        task_id="multiclass_ecoc_toy",
        task_kind="multiclass",
        description="Four-class Boolean toy decoded by error-correcting output codes.",
        input_bits=6,
        num_classes=4,
        smoke_train_samples=128,
        smoke_test_samples=128,
        full_train_samples=1024,
        full_test_samples=1024,
        baseline_columns=("majority_accuracy", "nearest_centroid_bits_accuracy"),
        target_rule="four disjoint Boolean prototypes decoded by Hamming distance",
    ),
}

SMOKE_PLAN: tuple[tuple[SynthesisFamily, str], ...] = (
    ("packed_bitset_gate_synthesis", "xor"),
    ("anf_sparse_boolean_polynomial", "xor"),
    ("packed_bitset_gate_synthesis", "diagonal_halfspace"),
    ("tree_bdd_compilation", "diagonal_halfspace"),
    ("packed_bitset_gate_synthesis", "checkerboard"),
    ("anf_sparse_boolean_polynomial", "checkerboard"),
    ("tree_bdd_compilation", "checkerboard"),
    ("packed_bitset_gate_synthesis", "small_digits_even_odd_bits"),
    ("tree_bdd_compilation", "small_digits_even_odd_bits"),
    ("ecoc_readout", "small_digits_mod3_bits"),
    ("ecoc_readout", "multiclass_ecoc_toy"),
)

FULL_PLAN: tuple[tuple[SynthesisFamily, str], ...] = (
    ("packed_bitset_gate_synthesis", "xor"),
    ("packed_bitset_gate_synthesis", "diagonal_halfspace"),
    ("packed_bitset_gate_synthesis", "checkerboard"),
    ("packed_bitset_gate_synthesis", "small_digits_even_odd_bits"),
    ("anf_sparse_boolean_polynomial", "xor"),
    ("anf_sparse_boolean_polynomial", "diagonal_halfspace"),
    ("anf_sparse_boolean_polynomial", "checkerboard"),
    ("anf_sparse_boolean_polynomial", "small_digits_even_odd_bits"),
    ("tree_bdd_compilation", "xor"),
    ("tree_bdd_compilation", "diagonal_halfspace"),
    ("tree_bdd_compilation", "checkerboard"),
    ("tree_bdd_compilation", "small_digits_even_odd_bits"),
    ("ecoc_readout", "small_digits_mod3_bits"),
    ("ecoc_readout", "multiclass_ecoc_toy"),
)


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse comma-separated seed text."""
    try:
        seeds = tuple(int(part) for part in text.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must contain integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _train_test_samples(task: TaskTemplate, scale: Scale) -> tuple[int, int]:
    if scale == "full":
        return task.full_train_samples, task.full_test_samples
    return task.smoke_train_samples, task.smoke_test_samples


def _family_gate_budget(family: SynthesisFamily, task: TaskTemplate, scale: Scale) -> int:
    base = {
        "packed_bitset_gate_synthesis": 64,
        "ecoc_readout": 96,
        "anf_sparse_boolean_polynomial": 48,
        "tree_bdd_compilation": 80,
    }[family]
    multiplier = 8 if scale == "full" else 1
    class_multiplier = max(task.num_classes - 1, 1)
    return base * multiplier * class_multiplier


def _family_max_terms(family: SynthesisFamily, scale: Scale) -> int:
    if family != "anf_sparse_boolean_polynomial":
        return 0
    return 128 if scale == "full" else 16


def _family_max_depth(family: SynthesisFamily, scale: Scale) -> int:
    if family != "tree_bdd_compilation":
        return 0
    return 10 if scale == "full" else 4


def _family_ecoc_bits(family: SynthesisFamily, task: TaskTemplate, scale: Scale) -> int:
    if family != "ecoc_readout":
        return 0
    minimum_bits = max(3, task.num_classes + 1)
    return max(minimum_bits, 16 if scale == "full" else minimum_bits)


def _readout_for(family: SynthesisFamily, task: TaskTemplate) -> str:
    if family == "ecoc_readout":
        return "hamming_ecoc"
    if task.num_classes > 2:
        return "one_vs_rest_hard_vote"
    return "single_bit_hard_vote"


def _objective_for(family: SynthesisFamily) -> str:
    return {
        "packed_bitset_gate_synthesis": "maximize packed truth-table agreement",
        "ecoc_readout": "maximize minimum Hamming margin of hard code heads",
        "anf_sparse_boolean_polynomial": "minimize sparse GF(2) polynomial error",
        "tree_bdd_compilation": "minimize hard tree or BDD classification error",
    }[family]


def _make_spec(
    *,
    family: SynthesisFamily,
    task: TaskTemplate,
    scale: Scale,
    seed: int,
) -> HardSynthesisSpec:
    train_samples, test_samples = _train_test_samples(task, scale)
    short = FAMILY_SHORT[family]
    run_id = f"{short}_{task.task_id}_seed{seed}"
    required_modules = tuple(module for module, _ in FAMILY_BACKENDS[family])
    return HardSynthesisSpec(
        run_id=run_id,
        scale=scale,
        family=family,
        task_id=task.task_id,
        task_kind=task.task_kind,
        seed=seed,
        input_bits=task.input_bits,
        num_classes=task.num_classes,
        train_samples=train_samples,
        test_samples=test_samples,
        gate_budget=_family_gate_budget(family, task, scale),
        packed_word_bits=64,
        objective=_objective_for(family),
        readout=_readout_for(family, task),
        max_terms=_family_max_terms(family, scale),
        max_depth=_family_max_depth(family, scale),
        ecoc_bits=_family_ecoc_bits(family, task, scale),
        required_modules=required_modules,
        baseline_columns=task.baseline_columns,
        required_metrics=PRIMARY_METRICS,
    )


def build_specs(*, scale: Scale, seeds: tuple[int, ...]) -> tuple[HardSynthesisSpec, ...]:
    """Construct hard-synthesis specs for the selected scale and seeds."""
    plan = FULL_PLAN if scale == "full" else SMOKE_PLAN
    specs: list[HardSynthesisSpec] = []
    for seed in seeds:
        for family, task_id in plan:
            specs.append(_make_spec(family=family, task=TASKS[task_id], scale=scale, seed=seed))
    return tuple(specs)


def claim_contract_checks(spec: HardSynthesisSpec) -> dict[str, Any]:
    """Return static claim checks for a planned experiment."""
    flags: list[str] = []
    if spec.readout == "linear":
        flags.append("float_head")
    if "packed_hard_accuracy" not in spec.required_metrics:
        flags.append("missing_packed_primary_metric")
    if spec.task_kind == "image_bits" and not any(
        column.startswith("same_feature_") for column in spec.baseline_columns
    ):
        flags.append("missing_same_feature_image_baseline")
    return {
        "float_head_forbidden": spec.readout != "linear",
        "hard_packed_metrics_primary": "packed_hard_accuracy" in spec.required_metrics,
        "eml_witness_required_for_every_gate_mask": True,
        "same_feature_baseline_required": spec.task_kind == "image_bits",
        "same_feature_baseline_present": spec.task_kind != "image_bits" or not flags,
        "primary_metric": "packed_hard_accuracy",
        "forbidden_deployed_readout": "linear",
        "baseline_columns": spec.baseline_columns,
        "flags": flags,
    }


def experiment_config(spec: HardSynthesisSpec) -> dict[str, Any]:
    """Return the backend-facing config for one spec."""
    task = TASKS[spec.task_id]
    return {
        "schema_version": "diffeml.hard_synthesis.config.v1",
        "run_id": spec.run_id,
        "seed": spec.seed,
        "scale": spec.scale,
        "task": {
            "task_id": spec.task_id,
            "task_kind": spec.task_kind,
            "description": task.description,
            "target_rule": task.target_rule,
            "input_bits": spec.input_bits,
            "num_classes": spec.num_classes,
            "train_samples": spec.train_samples,
            "test_samples": spec.test_samples,
        },
        "synthesis": {
            "family": spec.family,
            "objective": spec.objective,
            "gate_budget": spec.gate_budget,
            "packed_word_bits": spec.packed_word_bits,
            "readout": spec.readout,
            "max_terms": spec.max_terms,
            "max_depth": spec.max_depth,
            "ecoc_bits": spec.ecoc_bits,
            "required_modules": spec.required_modules,
        },
        "claim_contract": claim_contract_checks(spec),
        "required_metrics": spec.required_metrics,
    }


def _run_mode_for_spec(spec: HardSynthesisSpec) -> RunMode:
    return "continuous" if spec.task_kind == "continuous" else "boolean"


def suite_command(spec: HardSynthesisSpec, *, output: Path) -> str:
    """Return the direct command to run one experiment."""
    args = [
        "python",
        str(SCRIPT_PATH),
        "--run",
        _run_mode_for_spec(spec),
        "--scale",
        spec.scale,
        "--seeds",
        str(spec.seed),
        "--experiment",
        spec.run_id,
        "--output",
        str(output),
    ]
    return shlex.join(args)


def build_matrix(
    *,
    scale: Scale,
    seeds: tuple[int, ...],
    run_output_dir: Path = DEFAULT_RUN_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build a concrete hard-synthesis experiment matrix."""
    specs = build_specs(scale=scale, seeds=seeds)
    rows = []
    for spec in specs:
        output = run_output_dir / f"{spec.run_id}.json"
        rows.append(
            {
                **asdict(spec),
                "command": suite_command(spec, output=output),
                "config": experiment_config(spec),
                "claim_checks": claim_contract_checks(spec),
            }
        )
    task_ids = tuple(dict.fromkeys(spec.task_id for spec in specs))
    directions = [
        {
            "family": family,
            "short": FAMILY_SHORT[family],
            "description": FAMILY_DESCRIPTIONS[family],
            "backend_candidates": FAMILY_BACKENDS[family],
        }
        for family in FAMILY_SHORT
    ]
    return {
        "schema_version": "diffeml.hard_synthesis_suite.v1",
        "created_at_unix_s": time.time(),
        "scale": scale,
        "seeds": seeds,
        "directions": directions,
        "tasks": [asdict(TASKS[task_id]) for task_id in task_ids],
        "experiments": rows,
        "claim_rejection_rules": CLAIM_REJECTION_RULES,
        "primary_claim_metric": "packed_hard_accuracy",
        "matrix_notes": (
            "Matrix construction is backend-free. Non-dry runs lazily import the "
            "listed candidate modules and skip missing families."
        ),
    }


def _spec_selected_by_mode(spec: HardSynthesisSpec, mode: RunMode) -> bool:
    if mode == "all":
        return True
    if mode == "boolean":
        return spec.task_kind != "continuous"
    if mode == "continuous":
        return spec.task_kind == "continuous"
    return False


def select_specs(
    specs: tuple[HardSynthesisSpec, ...],
    *,
    mode: RunMode,
    experiment_id: str | None = None,
) -> tuple[HardSynthesisSpec, ...]:
    """Filter specs by CLI run mode and optional experiment id."""
    selected = tuple(spec for spec in specs if _spec_selected_by_mode(spec, mode))
    if experiment_id is not None:
        selected = tuple(spec for spec in selected if spec.run_id == experiment_id)
        if not selected:
            raise ValueError(f"unknown experiment for mode {mode!r}: {experiment_id}")
    return selected


def resolve_backend(spec: HardSynthesisSpec) -> BackendResolution | None:
    """Resolve the first available optional backend for a spec."""
    attempted: list[str] = []
    for module_name, function_name in FAMILY_BACKENDS[spec.family]:
        attempted.append(f"{module_name}:{function_name}")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        runner = getattr(module, function_name, None)
        if callable(runner):
            return BackendResolution(
                module_name=module_name,
                function_name=function_name,
                runner=cast(BackendRunner, runner),
                attempted=tuple(attempted),
            )
    return None


def run_experiment(
    spec: HardSynthesisSpec,
    *,
    dry_run: bool,
    run_output_dir: Path = DEFAULT_RUN_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run or dry-run one hard-synthesis spec."""
    output = run_output_dir / f"{spec.run_id}.json"
    config = experiment_config(spec)
    record: dict[str, Any] = {
        "run_id": spec.run_id,
        "family": spec.family,
        "task_id": spec.task_id,
        "task_kind": spec.task_kind,
        "dry_run": dry_run,
        "command": suite_command(spec, output=output),
        "config": config,
        "required_modules": spec.required_modules,
    }
    if dry_run:
        record["status"] = "dry_run"
        record["result"] = None
        return record

    backend = resolve_backend(spec)
    if backend is None:
        record["status"] = "skipped_missing_backend"
        record["attempted_backends"] = tuple(
            f"{module}:{function}" for module, function in FAMILY_BACKENDS[spec.family]
        )
        record["result"] = None
        return record

    try:
        result = dict(backend.runner(config))
    except Exception as exc:  # pragma: no cover - defensive around future modules.
        record["status"] = "backend_error"
        record["backend"] = {
            "module": backend.module_name,
            "function": backend.function_name,
        }
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["status"] = str(result.get("status", "completed"))
    record["backend"] = {
        "module": backend.module_name,
        "function": backend.function_name,
        "attempted": backend.attempted,
    }
    record["result"] = result
    return record


def run_suite(
    *,
    scale: Scale,
    seeds: tuple[int, ...],
    mode: RunMode,
    dry_run: bool,
    run_output_dir: Path = DEFAULT_RUN_OUTPUT_DIR,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run or dry-run the selected part of the suite."""
    matrix = build_matrix(scale=scale, seeds=seeds, run_output_dir=run_output_dir)
    specs = build_specs(scale=scale, seeds=seeds)
    selected = select_specs(specs, mode=mode, experiment_id=experiment_id)
    return {
        "schema_version": "diffeml.hard_synthesis_suite.run.v1",
        "created_at_unix_s": time.time(),
        "scale": scale,
        "mode": mode,
        "dry_run": dry_run,
        "matrix": matrix,
        "runs": [
            run_experiment(spec, dry_run=dry_run, run_output_dir=run_output_dir)
            for spec in selected
        ],
    }


def json_default(value: Any) -> Any:
    """Convert dataclasses and paths for JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=("matrix", "boolean", "continuous", "all"),
        default="matrix",
    )
    parser.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR)
    parser.add_argument("--seeds", type=parse_seeds, default=(0,))
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional run_id to select a single experiment within the chosen run mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit selected run records without importing synthesis backends.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the matrix or execute selected hard-synthesis specs."""
    args = parse_args(argv)
    if args.run == "matrix":
        payload = build_matrix(
            scale=args.scale,
            seeds=args.seeds,
            run_output_dir=args.run_output_dir,
        )
    else:
        payload = run_suite(
            scale=args.scale,
            seeds=args.seeds,
            mode=args.run,
            dry_run=args.dry_run,
            run_output_dir=args.run_output_dir,
            experiment_id=args.experiment,
        )
    text = json.dumps(payload, indent=2, default=json_default)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# mypy: disable-error-code="no-any-return"
"""Emit a DiffEML versus DiffLogic/LogicTreeNet benchmark matrix.

This script does not train DiffEML and does not import or install ``difflogic``.
It owns benchmark construction, command provenance, published baseline rows,
and conservative acceptance checks for paper planning.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

DEMO_SCRIPT = Path(__file__).with_name("step2_diffeml_image_demo.py")
DEFAULT_OUTPUT = Path("outputs/diffeml_image_demo/logic_benchmark_matrix.json")
DEFAULT_RUN_OUTPUT_DIR = Path("outputs/diffeml_image_demo/logic_benchmark")

Provenance = Literal["paper_reported", "local_reproduced", "pending"]
AcceptanceStatus = Literal[
    "accepted",
    "missing_candidate_metric",
    "missing_baseline_metric",
    "insufficient_provenance",
    "insufficient_seeds",
    "packed_mismatch",
    "below_baseline",
]


@dataclass(frozen=True)
class DiffEMLRunSpec:
    """One planned matched DiffEML image-demo run."""

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
    wiring_mode: str
    epochs: int
    batch_size: int = 128
    gate_mode: str = "eml_template"
    eml_template_depth: int = 2
    eml_eps: float = 0.05
    head_mode: str = "linear"
    hard_loss_weight: float = 0.5
    input_drop_rate: float = 0.0
    feature_drop_rate: float = 0.0
    packed_eval: bool = True
    compare_mlp: bool = True
    mlp_hidden_sizes: tuple[int, ...] = (512,)
    mlp_epochs: int = 0
    train_fraction: float = 0.8
    tree_stage_depths: tuple[int, ...] = (2, 2, 2, 2)
    local_patch_size: int = 3
    step_size: float = 0.01
    min_temperature: float = 0.1
    entropy_weight: float = 0.002
    head_l2: float = 1e-4
    residual_gate: str = "none"
    residual_gate_bias: float = 0.0


@dataclass(frozen=True)
class BaselineRow:
    """Published or planned external baseline row."""

    row_id: str
    method: str
    family: str
    dataset: str
    provenance: Provenance
    source: str
    accuracy: float | None
    gates: int | None
    parameters: int | None = None
    metric: str = "test_accuracy"
    split: str = "paper default"
    hard_or_discrete: bool = True
    local_command: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class AcceptanceCheck:
    """A conservative claim check for one candidate against one baseline."""

    candidate_id: str
    baseline_id: str
    status: AcceptanceStatus
    margin: float | None
    required_seeds: int
    observed_seeds: int
    notes: str


def template_threshold_count(depth: int) -> int:
    """Return internal EML-threshold nodes in a full binary template tree."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return (2**depth) - 1


def estimate_diffeml_gate_budget(spec: DiffEMLRunSpec) -> dict[str, int | str]:
    """Estimate comparable hard-circuit budgets for a planned DiffEML run."""
    selector_nodes = spec.layers * spec.width
    threshold_nodes = selector_nodes * template_threshold_count(spec.eml_template_depth)
    linear_head_parameters = 0 if spec.head_mode == "group_sum" else spec.width * 10 + 10
    return {
        "budget_kind": "estimate_without_training",
        "selector_nodes": selector_nodes,
        "selected_two_input_boolean_gates": selector_nodes,
        "eml_threshold_ops_if_expanded": threshold_nodes,
        "active_selector_logits": selector_nodes * 16
        if spec.gate_mode in {"eml_template", "truth_table"}
        else selector_nodes * 2,
        "non_boolean_linear_head_parameters": linear_head_parameters,
    }


def _flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _format_cli_value(value: object) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def diffeml_command(
    spec: DiffEMLRunSpec,
    *,
    output: Path,
    data_dir: Path,
) -> str:
    """Return the direct image-demo command for one planned DiffEML run."""
    args = ["python", str(DEMO_SCRIPT), "--datasets", spec.dataset, "--data-dir", str(data_dir)]
    omitted = {"run_id", "dataset", "packed_eval", "compare_mlp"}
    for field_name, value in asdict(spec).items():
        if field_name in omitted:
            continue
        args.extend((_flag_name(field_name), _format_cli_value(value)))
    if spec.packed_eval:
        args.append("--packed-eval")
    if spec.compare_mlp:
        args.append("--compare-mlp")
    args.extend(("--output", str(output)))
    return shlex.join(args)


def _diffeml_spec(
    *,
    dataset: str,
    seed: int,
    max_train: int,
    max_test: int,
    feature_mode: str,
    input_bits: int,
    pixel_thresholds: int,
    layers: int,
    width: int,
    wiring_mode: str,
    epochs: int,
    hard_loss_weight: float = 0.5,
    input_drop_rate: float = 0.0,
    feature_drop_rate: float = 0.0,
    min_temperature: float = 0.1,
    entropy_weight: float = 0.002,
) -> DiffEMLRunSpec:
    run_id = (
        f"{dataset}_{feature_mode}_{wiring_mode}_w{width}_l{layers}"
        f"_train{max_train}_seed{seed}"
    )
    return DiffEMLRunSpec(
        run_id=run_id,
        dataset=dataset,
        seed=seed,
        max_train=max_train,
        max_test=max_test,
        feature_mode=feature_mode,
        input_bits=input_bits,
        pixel_thresholds=pixel_thresholds,
        layers=layers,
        width=width,
        wiring_mode=wiring_mode,
        epochs=epochs,
        hard_loss_weight=hard_loss_weight,
        input_drop_rate=input_drop_rate,
        feature_drop_rate=feature_drop_rate,
        min_temperature=min_temperature,
        entropy_weight=entropy_weight,
    )


def build_diffeml_specs(*, scale: str, seeds: tuple[int, ...]) -> tuple[DiffEMLRunSpec, ...]:
    """Construct matched DiffEML runs for smoke or paper-scale planning."""
    if scale not in {"smoke", "paper"}:
        raise ValueError("scale must be 'smoke' or 'paper'")
    if scale == "smoke":
        return tuple(
            _diffeml_spec(
                dataset="digits",
                seed=seed,
                max_train=128,
                max_test=64,
                feature_mode="threshold_pixels",
                input_bits=64,
                pixel_thresholds=1,
                layers=2,
                width=96,
                wiring_mode="random",
                epochs=1,
            )
            for seed in seeds
        )

    specs: list[DiffEMLRunSpec] = []
    for seed in seeds:
        specs.extend(
            [
                _diffeml_spec(
                    dataset="mnist",
                    seed=seed,
                    max_train=60000,
                    max_test=10000,
                    feature_mode="threshold_pixels",
                    input_bits=784,
                    pixel_thresholds=1,
                    layers=6,
                    width=2048,
                    wiring_mode="random",
                    epochs=10,
                ),
                _diffeml_spec(
                    dataset="cifar",
                    seed=seed,
                    max_train=50000,
                    max_test=10000,
                    feature_mode="threshold_pixels",
                    input_bits=9216,
                    pixel_thresholds=3,
                    layers=6,
                    width=2048,
                    wiring_mode="random",
                    epochs=10,
                ),
                _diffeml_spec(
                    dataset="cifar",
                    seed=seed,
                    max_train=20000,
                    max_test=5000,
                    feature_mode="detector_thresholds",
                    input_bits=24576,
                    pixel_thresholds=4,
                    layers=6,
                    width=2048,
                    wiring_mode="random",
                    epochs=10,
                    hard_loss_weight=0.8,
                    input_drop_rate=0.02,
                    feature_drop_rate=0.3,
                    min_temperature=0.05,
                    entropy_weight=0.01,
                ),
            ]
        )
    return tuple(specs)


def external_baseline_rows() -> tuple[BaselineRow, ...]:
    """Return paper-reported and pending external reproduction rows."""
    deep_diff_logic = "https://papers.nips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Paper-Conference.pdf"
    logic_tree_net = "https://papers.neurips.cc/paper_files/paper/2024/file/db988b089d8d97d0f159c15ed0be6a71-Paper-Conference.pdf"
    github = "https://github.com/Felix-Petersen/difflogic"
    rows = [
        BaselineRow(
            "difflogic_mnist_small_paper",
            "Diff Logic Net (small)",
            "DiffLogic",
            "mnist",
            "paper_reported",
            deep_diff_logic,
            0.9769,
            48_000,
            parameters=48_000,
            split="binarized MNIST, paper Table 4",
        ),
        BaselineRow(
            "difflogic_mnist_largest_paper",
            "Diff Logic Net",
            "DiffLogic",
            "mnist",
            "paper_reported",
            deep_diff_logic,
            0.9847,
            384_000,
            parameters=384_000,
            split="binarized MNIST, paper Table 4",
        ),
        BaselineRow(
            "difflogic_cifar_small_paper",
            "Diff Logic Net (small)",
            "DiffLogic",
            "cifar",
            "paper_reported",
            deep_diff_logic,
            0.5127,
            48_000,
            parameters=48_000,
            split="CIFAR-10 color-channel resolution 4, paper Table 5",
        ),
        BaselineRow(
            "difflogic_cifar_medium_paper",
            "Diff Logic Net (medium)",
            "DiffLogic",
            "cifar",
            "paper_reported",
            deep_diff_logic,
            0.5739,
            512_000,
            parameters=512_000,
            split="CIFAR-10 color-channel resolution 4, paper Table 5",
        ),
        BaselineRow(
            "difflogic_cifar_large_x4_paper",
            "Diff Logic Net (large x4)",
            "DiffLogic",
            "cifar",
            "paper_reported",
            deep_diff_logic,
            0.6214,
            5_120_000,
            parameters=5_120_000,
            split="CIFAR-10 color-channel resolution 32, paper Table 5",
        ),
        BaselineRow(
            "logictreenet_cifar_s_paper",
            "LogicTreeNet-S",
            "LogicTreeNet",
            "cifar",
            "paper_reported",
            logic_tree_net,
            0.6038,
            400_000,
            split="CIFAR-10 discretized LGN, paper Table 1",
        ),
        BaselineRow(
            "logictreenet_cifar_m_paper",
            "LogicTreeNet-M",
            "LogicTreeNet",
            "cifar",
            "paper_reported",
            logic_tree_net,
            0.7101,
            3_080_000,
            split="CIFAR-10 discretized LGN, paper Table 1",
        ),
        BaselineRow(
            "logictreenet_cifar_b_paper",
            "LogicTreeNet-B",
            "LogicTreeNet",
            "cifar",
            "paper_reported",
            logic_tree_net,
            0.8017,
            16_000_000,
            split="CIFAR-10 discretized LGN, paper Table 1",
        ),
        BaselineRow(
            "logictreenet_cifar_g_paper",
            "LogicTreeNet-G",
            "LogicTreeNet",
            "cifar",
            "paper_reported",
            logic_tree_net,
            0.8629,
            61_000_000,
            split="CIFAR-10 discretized LGN, paper Table 1",
        ),
        BaselineRow(
            "logictreenet_mnist_m_paper",
            "LogicTreeNet-M",
            "LogicTreeNet",
            "mnist",
            "paper_reported",
            logic_tree_net,
            0.9923,
            566_000,
            split="MNIST discretized LGN, paper Table 3",
        ),
        BaselineRow(
            "difflogic_local_reproduction_pending",
            "Official difflogic reproduction",
            "DiffLogic",
            "cifar",
            "pending",
            github,
            None,
            None,
            local_command="python -m examples.train --dataset cifar-10  # pending CUDA setup",
            notes=(
                "Requires official difflogic installation, CUDA toolkit, "
                "and exact paper config capture."
            ),
        ),
    ]
    return tuple(rows)


def load_diffeml_artifact_row(path: Path) -> BaselineRow | None:
    """Load the best known local DiffEML artifact as evidence when present."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    result = payload.get("results", [{}])[0]
    metrics = result.get("metrics", {})
    model = result.get("model", {})
    data = result.get("data", {})
    packed_accuracy = metrics.get("packed_hard_test_accuracy")
    hard_accuracy = (
        packed_accuracy
        if packed_accuracy is not None
        else metrics.get("test_hard_accuracy")
    )
    if not isinstance(hard_accuracy, int | float):
        return None
    dataset = str(result.get("dataset", data.get("dataset", "unknown")))
    feature_mode = str(config.get("feature_mode", data.get("feature_mode", "unknown_features")))
    pixel_thresholds = config.get("pixel_thresholds", data.get("pixel_thresholds", "unknown"))
    width = int(config.get("width", model.get("width", 0)) or 0)
    layers = int(model.get("layers", config.get("layers", 0)) or 0)
    seed = config.get("seed", "unknown")
    input_drop = config.get("input_drop_rate", 0.0)
    feature_drop = config.get("feature_drop_rate", 0.0)
    regularization_suffix = (
        f"_drop{input_drop}_{feature_drop}"
        if input_drop or feature_drop
        else ""
    )
    metric = (
        "packed_hard_test_accuracy"
        if packed_accuracy is not None
        else "test_hard_accuracy"
    )
    return BaselineRow(
        row_id=(
            f"diffeml_{dataset}_{feature_mode}_t{pixel_thresholds}"
            f"_w{width}_l{layers}{regularization_suffix}_seed{seed}_local"
        ),
        method=(
            f"DiffEML {feature_mode} t{pixel_thresholds} "
            f"w{width} l{layers} input_drop={input_drop} "
            f"feature_drop={feature_drop} seed{seed}"
        ),
        family="DiffEML",
        dataset=dataset,
        provenance="local_reproduced",
        source=str(path),
        accuracy=float(hard_accuracy),
        gates=int(model.get("nodes", 0)) or None,
        parameters=int(model.get("active_node_parameters", 0)) or None,
        metric=metric,
        split=(
            f"{data.get('train_examples', 'unknown')} train / "
            f"{data.get('test_examples', 'unknown')} test"
        ),
        notes="Internal artifact only; not a claim against external baselines.",
    )


def acceptance_check(
    candidate: BaselineRow,
    baseline: BaselineRow,
    *,
    observed_seeds: int,
    required_seeds: int = 5,
    packed_matches: bool = True,
) -> AcceptanceCheck:
    """Decide whether a superiority claim is supported by local evidence."""
    if candidate.accuracy is None:
        status: AcceptanceStatus = "missing_candidate_metric"
        margin = None
    elif baseline.accuracy is None:
        status = "missing_baseline_metric"
        margin = None
    elif candidate.provenance != "local_reproduced":
        status = "insufficient_provenance"
        margin = None
    elif candidate.accuracy < baseline.accuracy:
        status = "below_baseline"
        margin = candidate.accuracy - baseline.accuracy
    elif observed_seeds < required_seeds:
        status = "insufficient_seeds"
        margin = candidate.accuracy - baseline.accuracy
    elif not packed_matches:
        status = "packed_mismatch"
        margin = candidate.accuracy - baseline.accuracy
    else:
        margin = candidate.accuracy - baseline.accuracy
        status = "accepted" if margin >= 0.0 else "below_baseline"
    notes = (
        "Claim allowed only for local reproduced DiffEML hard/packed metrics "
        "with the required seed matrix."
    )
    return AcceptanceCheck(
        candidate_id=candidate.row_id,
        baseline_id=baseline.row_id,
        status=status,
        margin=margin,
        required_seeds=required_seeds,
        observed_seeds=observed_seeds,
        notes=notes,
    )


def build_matrix(
    *,
    scale: str,
    seeds: tuple[int, ...],
    data_dir: Path,
    run_output_dir: Path,
    current_artifact: Path | None = Path(
        "outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json"
    ),
) -> dict[str, Any]:
    """Build the complete benchmark-planning payload."""
    specs = build_diffeml_specs(scale=scale, seeds=seeds)
    planned_runs = []
    for spec in specs:
        output = run_output_dir / f"{spec.run_id}.json"
        planned_runs.append(
            {
                "run_id": spec.run_id,
                "config": asdict(spec),
                "command": diffeml_command(spec, output=output, data_dir=data_dir),
                "gate_budget": estimate_diffeml_gate_budget(spec),
                "required_metrics": [
                    "test_soft_accuracy",
                    "test_hard_accuracy",
                    "packed_hard_test_accuracy",
                    "mlp_same_features.test_accuracy",
                ],
            }
        )

    external_rows = list(external_baseline_rows())
    local_rows: list[BaselineRow] = []
    if current_artifact is not None:
        row = load_diffeml_artifact_row(current_artifact)
        if row is not None:
            local_rows.append(row)

    acceptance_checks: list[AcceptanceCheck] = []
    if local_rows:
        for local_row in local_rows:
            packed_matches = local_row.metric == "packed_hard_test_accuracy"
            for external_row in external_rows:
                if external_row.dataset != local_row.dataset:
                    continue
                if external_row.accuracy is None:
                    continue
                acceptance_checks.append(
                    acceptance_check(
                        local_row,
                        external_row,
                        observed_seeds=1,
                        required_seeds=5,
                        packed_matches=packed_matches,
                    )
                )

    return {
        "schema_version": "diffeml.logic_benchmark_matrix.v1",
        "created_at_unix": time.time(),
        "scale": scale,
        "seeds": list(seeds),
        "planned_diffeml_runs": planned_runs,
        "external_baselines": [asdict(row) for row in external_rows],
        "local_evidence_rows": [asdict(row) for row in local_rows],
        "acceptance_checks": [asdict(check) for check in acceptance_checks],
        "honesty_note": (
            "Paper-reported external rows are comparison targets. DiffEML superiority "
            "requires local reproduced hard and packed metrics over the required seeds."
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/diffeml_image_data"))
    parser.add_argument("--scale", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--current-artifact",
        type=Path,
        default=Path(
            "outputs/diffeml_image_demo/"
            "cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json"
        ),
        help="Optional existing DiffEML artifact to include as local evidence.",
    )
    parser.add_argument(
        "--no-current-artifact",
        action="store_true",
        help="Do not inspect existing DiffEML artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """Write the benchmark matrix JSON."""
    args = parse_args()
    seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2, 3, 4)
    current_artifact = None if args.no_current_artifact else args.current_artifact
    payload = build_matrix(
        scale=args.scale,
        seeds=seeds,
        data_dir=args.data_dir,
        run_output_dir=args.run_output_dir,
        current_artifact=current_artifact,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

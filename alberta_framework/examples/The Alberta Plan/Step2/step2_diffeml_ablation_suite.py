#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined,no-any-return,no-untyped-call"
"""Run fixed DiffEML image ablations and summarize their deltas.

The suite intentionally delegates training to ``step2_diffeml_image_demo.py``.
It owns only experiment construction, command reproduction, and lightweight
summary logic so the ablation surface stays reproducible without forking the
image runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DEMO_SCRIPT = Path(__file__).with_name("step2_diffeml_image_demo.py")
DEFAULT_OUTPUT = Path("outputs/diffeml_image_demo/ablation_suite_smoke.json")


@dataclass(frozen=True)
class AblationSpec:
    """One named DiffEML image ablation run."""

    name: str
    question: str
    config: Any


@dataclass(frozen=True)
class SuiteScale:
    """Shared size settings for smoke or full ablation configs."""

    datasets: tuple[str, ...]
    max_train: int
    max_test: int
    input_bits: int
    layers: int
    width: int
    epochs: int
    batch_size: int
    tree_stage_depths: tuple[int, ...]
    mlp_hidden_sizes: tuple[int, ...]
    mlp_epochs: int


def load_demo_module() -> Any:
    """Import the image demo despite spaces in the example path."""
    try:
        from alberta_framework.core import diffeml_image

        return diffeml_image
    except ImportError:
        pass

    module_name = "step2_diffeml_image_demo"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "DemoConfig"):
        return existing
    spec = importlib.util.spec_from_file_location(module_name, DEMO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {DEMO_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "DemoConfig"):
        return module

    from alberta_framework.core import diffeml_image

    return diffeml_image


def scale_defaults(*, full: bool) -> SuiteScale:
    """Return quick smoke settings or larger image-demo settings."""
    if full:
        return SuiteScale(
            datasets=("digits", "mnist", "cifar"),
            max_train=1200,
            max_test=300,
            input_bits=512,
            layers=3,
            width=768,
            epochs=15,
            batch_size=128,
            tree_stage_depths=(2, 2, 2, 2),
            mlp_hidden_sizes=(512,),
            mlp_epochs=15,
        )
    return SuiteScale(
        datasets=("digits",),
        max_train=96,
        max_test=48,
        input_bits=64,
        layers=2,
        width=80,
        epochs=1,
        batch_size=32,
        tree_stage_depths=(1, 1),
        mlp_hidden_sizes=(64,),
        mlp_epochs=1,
    )


def base_config(scale: SuiteScale, *, seed: int) -> Any:
    """Build the canonical same-feature DiffEML image config."""
    demo = load_demo_module()
    return demo.DemoConfig(
        datasets=scale.datasets,
        seed=seed,
        train_fraction=0.8,
        max_train=scale.max_train,
        max_test=scale.max_test,
        feature_mode="threshold_pixels",
        input_bits=scale.input_bits,
        pixel_thresholds=1,
        layers=scale.layers,
        width=scale.width,
        wiring_mode="local_tree_hierarchy",
        local_patch_size=3,
        tree_stage_depths=scale.tree_stage_depths,
        epochs=scale.epochs,
        batch_size=scale.batch_size,
        step_size=0.01,
        initial_temperature=1.0,
        min_temperature=0.1,
        entropy_weight=0.002,
        head_l2=1e-4,
        gate_init_scale=0.2,
        head_init_scale=0.2,
        max_grad_norm=10.0,
        eml_template_depth=2,
        eml_eps=0.05,
        gate_mode="eml_template",
        eml_threshold_temperature=0.75,
        threshold_init_scale=0.1,
        direction_init_scale=0.2,
        hard_loss_weight=0.5,
        input_drop_rate=0.0,
        feature_drop_rate=0.0,
        residual_gate="or",
        residual_gate_bias=0.5,
        head_mode="linear",
        group_sum_tau=30.0,
        readout_entropy_weight=0.0,
        readout_balance_weight=0.0,
        packed_eval=False,
        compare_mlp=False,
        mlp_hidden_sizes=scale.mlp_hidden_sizes,
        mlp_epochs=scale.mlp_epochs,
        mlp_step_size=0.001,
        mlp_weight_decay=1e-4,
        mlp_max_grad_norm=10.0,
        mlp_init_scale=1.0,
    )


def build_ablation_specs(*, full: bool = False, seed: int = 0) -> tuple[AblationSpec, ...]:
    """Construct the fixed suite of ablation configs."""
    scale = scale_defaults(full=full)
    base = base_config(scale, seed=seed)
    return (
        AblationSpec(
            name="eml_template",
            question="Executable EML templates on threshold pixels and local tree wiring.",
            config=base,
        ),
        AblationSpec(
            name="truth_table",
            question="Truth-table interpolation on the same features and wiring.",
            config=replace(base, gate_mode="truth_table"),
        ),
        AblationSpec(
            name="linear_head",
            question="Trainable linear head on the canonical circuit features.",
            config=base,
        ),
        AblationSpec(
            name="group_sum_head",
            question="Fixed grouped summation head on the same circuit features.",
            config=replace(base, head_mode="group_sum", head_l2=0.0),
        ),
        AblationSpec(
            name="class_vote_head",
            question=(
                "Learned class assignment per final Boolean feature, hardened to "
                "packed popcount votes without a float head."
            ),
            config=replace(base, head_mode="class_vote", head_l2=0.0, group_sum_tau=10.0),
        ),
        AblationSpec(
            name="signed_class_vote_head",
            question=(
                "Learned class plus polarity assignment per final Boolean feature, "
                "compiled to signed popcount metadata."
            ),
            config=replace(
                base,
                head_mode="signed_class_vote",
                head_l2=0.0,
                group_sum_tau=10.0,
            ),
        ),
        AblationSpec(
            name="class_bank_group_sum",
            question=(
                "Class-bank final wiring with fixed grouped popcount readout, "
                "testing pure Boolean class evidence construction."
            ),
            config=replace(
                base,
                wiring_mode="class_bank_random",
                head_mode="group_sum",
                head_l2=0.0,
                group_sum_tau=1.0,
            ),
        ),
        AblationSpec(
            name="class_bank_class_vote",
            question=(
                "Class-bank final wiring with learned discrete class-vote metadata."
            ),
            config=replace(
                base,
                wiring_mode="class_bank_random",
                head_mode="class_vote",
                head_l2=0.0,
                group_sum_tau=1.0,
            ),
        ),
        AblationSpec(
            name="threshold_pixels",
            question="Raw thresholded pixels as binary logic inputs.",
            config=base,
        ),
        AblationSpec(
            name="detector_thresholds",
            question="Fixed detector threshold features as binary logic inputs.",
            config=replace(base, feature_mode="detector_thresholds"),
        ),
        AblationSpec(
            name="packed_eval",
            question="Packed Boolean evaluation should match hard selector evaluation.",
            config=replace(base, packed_eval=True),
        ),
        AblationSpec(
            name="mlp_same_features",
            question="MLP trained on the exact same binary features as DiffEML.",
            config=replace(base, compare_mlp=True),
        ),
    )


def _flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _format_cli_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def demo_command(
    config: Any,
    *,
    output: Path | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return a reproducible direct image-demo command for a config."""
    args = ["python", str(DEMO_SCRIPT), "--datasets"]
    args.extend(str(dataset) for dataset in config.datasets)
    if data_dir is not None:
        args.extend(("--data-dir", str(data_dir)))
    for field_name, value in asdict(config).items():
        if field_name in {"datasets", "packed_eval", "compare_mlp"}:
            continue
        args.extend((_flag_name(field_name), _format_cli_value(value)))
    if config.packed_eval:
        args.append("--packed-eval")
    if config.compare_mlp:
        args.append("--compare-mlp")
    if output is not None:
        args.extend(("--output", str(output)))
    return shlex.join(args)


def extract_dataset_summaries(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract compact per-dataset metrics from a demo payload."""
    summaries: list[dict[str, Any]] = []
    for result in result_payload.get("results", []):
        summary: dict[str, Any] = {"dataset": result.get("dataset")}
        if "error" in result:
            summary["error"] = result["error"]
            summaries.append(summary)
            continue
        summary["metrics"] = result.get("metrics", {})
        summary["training"] = {
            key: result.get("training", {}).get(key)
            for key in ("epochs", "updates", "elapsed_s", "last_loss", "last_grad_norm")
        }
        summary["data"] = {
            key: result.get("data", {}).get(key)
            for key in (
                "feature_mode",
                "selected_bits",
                "expanded_bits",
                "detector_maps",
                "train_examples",
                "test_examples",
            )
            if key in result.get("data", {})
        }
        summary["model"] = {
            key: result.get("model", {}).get(key)
            for key in (
                "gate_mode",
                "head_mode",
                "wiring_mode",
                "layers",
                "width",
                "nodes",
                "active_node_parameters",
                "head_parameters",
            )
            if key in result.get("model", {})
        }
        baselines = result.get("baselines", {})
        if "mlp_same_features" in baselines:
            summary["baselines"] = {"mlp_same_features": baselines["mlp_same_features"]}
        summaries.append(summary)
    return summaries


def summarize_run(
    spec: AblationSpec,
    payload: dict[str, Any],
    *,
    output: Path | None,
    data_dir: Path | None,
) -> dict[str, Any]:
    """Create a compact run record with config, command, and metrics."""
    return {
        "name": spec.name,
        "question": spec.question,
        "command": demo_command(spec.config, output=output, data_dir=data_dir),
        "config": asdict(spec.config),
        "results": extract_dataset_summaries(payload),
    }


def _metric_by_dataset(run: dict[str, Any], metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in run.get("results", []):
        dataset = result.get("dataset")
        value = result.get("metrics", {}).get(metric)
        if isinstance(dataset, str) and isinstance(value, int | float):
            values[dataset] = float(value)
    return values


def _baseline_by_dataset(run: dict[str, Any], baseline: str, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in run.get("results", []):
        dataset = result.get("dataset")
        value = result.get("baselines", {}).get(baseline, {}).get(metric)
        if isinstance(dataset, str) and isinstance(value, int | float):
            values[dataset] = float(value)
    return values


def metric_delta(
    runs_by_name: dict[str, dict[str, Any]],
    *,
    left: str,
    right: str,
    metric: str = "test_hard_accuracy",
) -> dict[str, Any]:
    """Compute left-minus-right deltas for a metric by dataset."""
    left_values = _metric_by_dataset(runs_by_name[left], metric)
    right_values = _metric_by_dataset(runs_by_name[right], metric)
    datasets = sorted(left_values.keys() & right_values.keys())
    deltas = {
        dataset: round(left_values[dataset] - right_values[dataset], 12)
        for dataset in datasets
    }
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "deltas": deltas,
        "mean_delta": sum(deltas.values()) / len(deltas) if deltas else None,
    }


def packed_eval_delta(run: dict[str, Any]) -> dict[str, Any]:
    """Summarize packed hard accuracy equality against regular hard accuracy."""
    hard_values = _metric_by_dataset(run, "test_hard_accuracy")
    packed_values = _metric_by_dataset(run, "packed_hard_test_accuracy")
    datasets = sorted(hard_values.keys() & packed_values.keys())
    deltas = {
        dataset: round(packed_values[dataset] - hard_values[dataset], 12)
        for dataset in datasets
    }
    return {
        "run": run["name"],
        "metric": "packed_hard_test_accuracy - test_hard_accuracy",
        "deltas": deltas,
        "max_abs_delta": max((abs(value) for value in deltas.values()), default=None),
        "equal": all(abs(value) <= 1e-12 for value in deltas.values()) if deltas else None,
    }


def mlp_comparison(run: dict[str, Any]) -> dict[str, Any]:
    """Summarize DiffEML hard accuracy against the same-feature MLP baseline."""
    diffeml_values = _metric_by_dataset(run, "test_hard_accuracy")
    mlp_values = _baseline_by_dataset(run, "mlp_same_features", "test_accuracy")
    datasets = sorted(diffeml_values.keys() & mlp_values.keys())
    deltas = {
        dataset: round(diffeml_values[dataset] - mlp_values[dataset], 12)
        for dataset in datasets
    }
    return {
        "run": run["name"],
        "metric": "diffeml_test_hard_accuracy - mlp_same_feature_test_accuracy",
        "deltas": deltas,
        "mean_delta": sum(deltas.values()) / len(deltas) if deltas else None,
    }


def summarize_suite(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Create fixed cross-run comparison summaries."""
    runs_by_name = {run["name"]: run for run in runs}
    comparisons = [
        {
            "name": "eml_template_vs_truth_table",
            **metric_delta(runs_by_name, left="eml_template", right="truth_table"),
        },
        {
            "name": "linear_head_vs_group_sum",
            **metric_delta(runs_by_name, left="linear_head", right="group_sum_head"),
        },
        {
            "name": "class_vote_vs_group_sum",
            **metric_delta(runs_by_name, left="class_vote_head", right="group_sum_head"),
        },
        {
            "name": "linear_head_vs_class_vote",
            **metric_delta(runs_by_name, left="linear_head", right="class_vote_head"),
        },
        {
            "name": "signed_class_vote_vs_class_vote",
            **metric_delta(
                runs_by_name,
                left="signed_class_vote_head",
                right="class_vote_head",
            ),
        },
        {
            "name": "class_bank_group_sum_vs_group_sum",
            **metric_delta(runs_by_name, left="class_bank_group_sum", right="group_sum_head"),
        },
        {
            "name": "class_bank_class_vote_vs_class_vote",
            **metric_delta(
                runs_by_name,
                left="class_bank_class_vote",
                right="class_vote_head",
            ),
        },
        {
            "name": "threshold_pixels_vs_detector_thresholds",
            **metric_delta(runs_by_name, left="threshold_pixels", right="detector_thresholds"),
        },
        {"name": "packed_eval_equality", **packed_eval_delta(runs_by_name["packed_eval"])},
        {
            "name": "mlp_same_feature_comparison",
            **mlp_comparison(runs_by_name["mlp_same_features"]),
        },
    ]
    return {"comparisons": comparisons}


def run_suite(
    specs: tuple[AblationSpec, ...],
    *,
    data_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Execute or dry-run the fixed ablation configs."""
    demo = load_demo_module()
    json_default = getattr(demo, "json_default", getattr(demo, "_json_default", None))
    runs: list[dict[str, Any]] = []
    for spec in specs:
        run_output = output_dir / f"{spec.name}.json"
        print(f"running ablation: {spec.name}", flush=True)
        if dry_run:
            payload = {"config": asdict(spec.config), "results": []}
        else:
            payload = demo.run_demo(spec.config, data_dir)
            run_output.parent.mkdir(parents=True, exist_ok=True)
            run_output.write_text(
                json.dumps(payload, indent=2, default=json_default) + "\n",
                encoding="utf-8",
            )
        runs.append(summarize_run(spec, payload, output=run_output, data_dir=data_dir))
    summary = summarize_suite(runs) if not dry_run else {"comparisons": []}
    return {"runs": runs, **summary}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diffeml_image_demo/ablations"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/diffeml_image_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("digits", "mnist", "cifar"),
        default=None,
        help="Override the smoke/full dataset set.",
    )
    parser.add_argument("--full", action="store_true", help="Use full-scale image-demo defaults.")
    parser.add_argument("--dry-run", action="store_true", help="Emit configs without training.")
    return parser.parse_args()


def main() -> int:
    """Run the ablation suite."""
    args = parse_args()
    t0 = time.time()
    specs = build_ablation_specs(full=args.full, seed=args.seed)
    if args.datasets is not None:
        specs = tuple(
            replace(spec, config=replace(spec.config, datasets=tuple(args.datasets)))
            for spec in specs
        )
    payload = {
        "suite": {
            "name": "diffeml_image_ablation_suite",
            "scale": "full" if args.full else "smoke",
            "dry_run": args.dry_run,
            "seed": args.seed,
            "elapsed_s": None,
        },
        **run_suite(
            specs,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        ),
    }
    payload["suite"]["elapsed_s"] = time.time() - t0
    text = json.dumps(payload, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Worker C report for D18 complexity-pruning evidence.

The script intentionally does not import or modify D18.  It reads completed
``results.json`` files, computes paired margins against the same-run fair MLP
baselines, and writes a concise pruning report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = REPO_ROOT / "outputs" / "step2_new_directions"
DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "research"
    / "step2_new_directions"
    / "worker_c_d18_complexity_pruning_report.md"
)

DIGIT_DATASETS = {
    "digits_class_blocked",
    "digits_iid",
    "digits_label_drift",
    "digits_mask_noise",
    "digits_permuted_pixels",
}
MLP_METHODS = ("mlp_h64", "mlp_h128", "mlp_h64_64")


@dataclass(frozen=True)
class RunSpec:
    label: str
    result_dir: str
    method: str = "d18_step2_canonical"
    notes: str = ""


ALL14_RUNS = (
    RunSpec(
        "canonical full budget",
        "d18_step2_canonical_thresholdproto_all14_10seed",
        notes="full RKHS banks; pre-tanh128 pruning reference",
    ),
    RunSpec(
        "full budget tanh128",
        "d18_step2_simplify_full_budget_tanh128_all14_10seed",
        notes="same RKHS banks, tanh basis width 128",
    ),
    RunSpec(
        "half budget tanh128",
        "d18_step2_simplify_half_budget_tanh128_all14_10seed",
        notes="raw/algebraic/arccosine banks halved",
    ),
    RunSpec(
        "quarter budget tanh256",
        "d18_step2_simplify_quarter_budget_tanh256_all14_3seed",
        notes="raw/algebraic/arccosine banks quartered",
    ),
    RunSpec(
        "full budget tanh64",
        "d18_step2_simplify_full_budget_tanh64_all14_3seed",
        notes="new Worker C low-cost ablation if present",
    ),
)

AVAILABLE_VARIANTS = (
    RunSpec(
        "synthetic compositional canonical",
        "d18_step2_comp_config_sweep_10seed",
        "d18_step2_canonical",
    ),
    RunSpec(
        "synthetic compositional no unified",
        "d18_step2_comp_config_sweep_10seed",
        "d18_step2_no_unified",
    ),
    RunSpec(
        "synthetic compositional no poly",
        "d18_step2_comp_config_sweep_10seed",
        "d18_step2_no_poly",
    ),
)

OPMNIST_RUNS = (
    (
        "OPMNIST 31 full task blocks",
        REPO_ROOT / "outputs" / "step2_canonical" / "opmnist_true_mnist_31block_mse_results.json",
        "mixture",
    ),
    (
        "OPMNIST 40 full task blocks",
        REPO_ROOT / "outputs" / "step2_canonical" / "opmnist_true_mnist_40block_mse_results.json",
        "mixture",
    ),
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text()))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def result_path(result_dir: str) -> Path:
    return RESULT_ROOT / result_dir / "results.json"


def comparison_for(
    aggregate: dict[str, Any],
    metric: str,
    method: str,
) -> dict[str, Any] | None:
    comparisons = aggregate.get("comparisons", {}).get(metric, {})
    if method in comparisons:
        return cast(dict[str, Any], comparisons[method])
    return cast(dict[str, Any] | None, comparisons.get("best_kernel_vs_best_mlp"))


def diff_from_comparison(comparison: dict[str, Any]) -> float:
    if "paired_diff_mean_positive_favors_method" in comparison:
        return float(comparison["paired_diff_mean_positive_favors_method"])
    return float(comparison["paired_diff_mean_positive_favors_kernel"])


def wins_from_comparison(comparison: dict[str, Any]) -> tuple[int, int, int]:
    if "wins_for_method" in comparison:
        return (
            int(comparison["wins_for_method"]),
            int(comparison["wins_for_baseline"]),
            int(comparison.get("ties", 0)),
        )
    return (
        int(comparison["wins_for_kernel"]),
        int(comparison["wins_for_mlp"]),
        int(comparison.get("ties", 0)),
    )


def fastest_mlp_runtime(aggregate: dict[str, Any]) -> float | None:
    runtimes = []
    for method in MLP_METHODS:
        value = aggregate.get(method, {}).get("runtime_s", {}).get("mean")
        if value is not None:
            runtimes.append(float(value))
    return min(runtimes) if runtimes else None


def best_mlp_metric(aggregate: dict[str, Any], metric: str, larger_is_better: bool) -> float | None:
    values = []
    for method in MLP_METHODS:
        value = aggregate.get(method, {}).get(metric, {}).get("mean")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return max(values) if larger_is_better else min(values)


def all14_summary(spec: RunSpec) -> dict[str, Any] | None:
    data = load_json(result_path(spec.result_dir))
    if data is None or spec.method not in data.get("candidate_methods", []):
        return None
    aggregate = data["aggregate"]
    rows = []
    ratios = []
    total_wins = total_losses = total_ties = 0
    for dataset, dataset_aggregate in aggregate.items():
        comparison = comparison_for(dataset_aggregate, "final_window_mse", spec.method)
        if comparison is None:
            continue
        diff = diff_from_comparison(comparison)
        wins, losses, ties = wins_from_comparison(comparison)
        total_wins += wins
        total_losses += losses
        total_ties += ties
        method_runtime = dataset_aggregate[spec.method].get("runtime_s", {}).get("mean")
        baseline_runtime = fastest_mlp_runtime(dataset_aggregate)
        if method_runtime is not None and baseline_runtime:
            ratios.append(float(method_runtime) / baseline_runtime)
        rows.append(
            {
                "dataset": dataset,
                "diff": diff,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    if not rows:
        return None
    worst = min(rows, key=lambda row: row["diff"])
    negative = [row for row in rows if row["diff"] < 0]
    return {
        "label": spec.label,
        "result_dir": spec.result_dir,
        "method": spec.method,
        "notes": spec.notes,
        "n_seeds": data.get("config", {}).get("n_seeds"),
        "datasets": len(rows),
        "wall_clock_s": data.get("wall_clock_s"),
        "mean_final_mse_margin": mean([row["diff"] for row in rows]),
        "worst_dataset": worst["dataset"],
        "worst_margin": worst["diff"],
        "worst_wins": f"{worst['wins']}/{worst['losses']}/{worst['ties']}",
        "paired_wlt": f"{total_wins}/{total_losses}/{total_ties}",
        "negative_count": len(negative),
        "runtime_ratio_mean": mean(ratios),
        "runtime_ratio_max": max(ratios) if ratios else float("nan"),
    }


def digit_metric_summary(spec: RunSpec) -> dict[str, Any] | None:
    data = load_json(result_path(spec.result_dir))
    if data is None or spec.method not in data.get("candidate_methods", []):
        return None
    rows = []
    for dataset, dataset_aggregate in data["aggregate"].items():
        if dataset not in DIGIT_DATASETS:
            continue
        for metric in ("test_mse", "test_accuracy"):
            comparison = comparison_for(dataset_aggregate, metric, spec.method)
            if comparison is None:
                continue
            diff = diff_from_comparison(comparison)
            wins, losses, ties = wins_from_comparison(comparison)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "diff": diff,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                }
            )
    if not rows:
        return None
    worst = min(rows, key=lambda row: row["diff"])
    negative = [row for row in rows if row["diff"] < 0]
    return {
        "label": spec.label,
        "metric_cells": len(rows),
        "negative_count": len(negative),
        "worst_cell": f"{worst['dataset']}:{worst['metric']}",
        "worst_margin": worst["diff"],
        "worst_wins": f"{worst['wins']}/{worst['losses']}/{worst['ties']}",
        "mean_digit_margin": mean([row["diff"] for row in rows]),
    }


def available_variant_summary(spec: RunSpec) -> dict[str, Any] | None:
    data = load_json(result_path(spec.result_dir))
    if data is None:
        return None
    rows = []
    ratios = []
    for dataset, dataset_aggregate in data["aggregate"].items():
        if spec.method not in dataset_aggregate:
            continue
        comparison = comparison_for(dataset_aggregate, "final_window_mse", spec.method)
        if comparison is None:
            continue
        diff = diff_from_comparison(comparison)
        wins, losses, ties = wins_from_comparison(comparison)
        method_runtime = dataset_aggregate[spec.method].get("runtime_s", {}).get("mean")
        baseline_runtime = fastest_mlp_runtime(dataset_aggregate)
        if method_runtime is not None and baseline_runtime:
            ratios.append(float(method_runtime) / baseline_runtime)
        rows.append((dataset, diff, wins, losses, ties))
    if not rows:
        return None
    worst = min(rows, key=lambda row: row[1])
    return {
        "label": spec.label,
        "method": spec.method,
        "datasets": ", ".join(row[0] for row in rows),
        "mean_final_mse_margin": mean([row[1] for row in rows]),
        "worst_margin": worst[1],
        "worst_wins": f"{worst[2]}/{worst[3]}/{worst[4]}",
        "runtime_ratio_mean": mean(ratios),
    }


def opmnist_summary(label: str, path: Path, method: str) -> dict[str, Any] | None:
    data = load_json(path)
    if data is None:
        return None
    aggregate = data.get("aggregate", {}).get("permuted_mnist_like", {})
    if method not in aggregate:
        return None
    method_metrics = aggregate[method]
    config = data.get("config", {})

    def margin(metric: str, larger_is_better: bool) -> float | None:
        baseline = best_mlp_metric(aggregate, metric, larger_is_better)
        candidate = method_metrics.get(metric, {}).get("mean")
        if baseline is None or candidate is None:
            return None
        if larger_is_better:
            return float(candidate) - baseline
        return baseline - float(candidate)

    dataset_meta = data.get("datasets", {}).get("permuted_mnist_like", {})
    return {
        "label": label,
        "method": method,
        "steps": config.get("steps"),
        "task_blocks": dataset_meta.get("opmnist_completed_full_task_blocks")
        or dataset_meta.get("opmnist_completed_full_60000_task_blocks"),
        "core_protocol": dataset_meta.get("matches_dohare_opmnist_core_protocol"),
        "published_task_count": dataset_meta.get("matches_dohare_opmnist_published_task_count"),
        "final_mse_margin": margin("final_window_mse", larger_is_better=False),
        "final_accuracy_margin": margin("final_window_accuracy", larger_is_better=True),
        "test_mse_margin": margin("test_mse", larger_is_better=False),
        "test_accuracy_margin": margin("test_accuracy", larger_is_better=True),
        "wall_clock_s": data.get("wall_clock_s"),
        "opmnist_elapsed_s": dataset_meta.get("opmnist_elapsed_s"),
        "opmnist_steps_per_second": dataset_meta.get("opmnist_overall_steps_per_second"),
    }


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def recommendation(
    all14_rows: list[dict[str, Any]],
    digit_rows: list[dict[str, Any]],
) -> str:
    digit_by_label = {row["label"]: row for row in digit_rows}
    candidates = []
    for row in all14_rows:
        digit = digit_by_label.get(row["label"])
        if digit is None:
            continue
        if (
            row["negative_count"] == 0
            and row["paired_wlt"].split("/")[1] == "0"
            and digit["negative_count"] == 0
        ):
            candidates.append(row)
    priority = {
        "full budget tanh64": 0,
        "full budget tanh128": 1,
        "canonical full budget": 2,
    }
    candidates.sort(key=lambda row: priority.get(row["label"], 99))
    if candidates and candidates[0]["label"] == "full budget tanh64":
        return (
            "Promote full-budget tanh64 to a 10-seed all14 confirmation run. "
            "It is the smallest passing candidate in the available table, but "
            "its evidence is only 3 seeds until that run is repeated."
        )
    if candidates and candidates[0]["label"] == "full budget tanh128":
        return (
            "Keep full-budget tanh128 as the current smallest confirmed candidate. "
            "Do not halve RKHS banks yet; the half-budget run already has a paired "
            "primary loss and weaker digit heldout behavior."
        )
    return (
        "Keep the existing full-budget canonical as the safe candidate; the current "
        "simplification artifacts do not establish a smaller no-gap version."
    )


def build_report() -> str:
    all14_rows = [row for spec in ALL14_RUNS if (row := all14_summary(spec)) is not None]
    digit_rows = [row for spec in ALL14_RUNS if (row := digit_metric_summary(spec)) is not None]
    variant_rows = [
        row for spec in AVAILABLE_VARIANTS if (row := available_variant_summary(spec)) is not None
    ]
    opmnist_rows = [
        row
        for label, path, method in OPMNIST_RUNS
        if (row := opmnist_summary(label, path, method)) is not None
    ]

    lines = [
        "# Worker C D18 Complexity Pruning Report",
        "",
        "Scope: analysis only. This report reads existing result artifacts plus the optional "
        "Worker C tanh64 ablation when present. It does not import or modify D18.",
        "",
        "## All14 Primary Final-Window MSE",
        "",
        markdown_table(
            [
                "run",
                "seeds",
                "datasets",
                "mean margin",
                "worst dataset",
                "worst margin",
                "worst W/L/T",
                "all W/L/T",
                "neg cells",
                "runtime vs fastest MLP",
                "wall s",
            ],
            [
                [
                    row["label"],
                    str(row["n_seeds"]),
                    str(row["datasets"]),
                    fmt_float(row["mean_final_mse_margin"]),
                    row["worst_dataset"],
                    fmt_float(row["worst_margin"]),
                    row["worst_wins"],
                    row["paired_wlt"],
                    str(row["negative_count"]),
                    (
                        f"{fmt_float(row['runtime_ratio_mean'], 2)}x mean / "
                        f"{fmt_float(row['runtime_ratio_max'], 2)}x max"
                    ),
                    fmt_float(row["wall_clock_s"], 1),
                ]
                for row in all14_rows
            ],
        ),
        "",
        "Margins are paired best-MLP MSE minus candidate MSE, so positive values favor D18.",
        "",
        "## Digit Heldout Cells",
        "",
        markdown_table(
            [
                "run",
                "metric cells",
                "neg cells",
                "worst cell",
                "worst margin",
                "worst W/L/T",
                "mean heldout margin",
            ],
            [
                [
                    row["label"],
                    str(row["metric_cells"]),
                    str(row["negative_count"]),
                    row["worst_cell"],
                    fmt_float(row["worst_margin"]),
                    row["worst_wins"],
                    fmt_float(row["mean_digit_margin"]),
                ]
                for row in digit_rows
            ],
        ),
        "",
        "Digit heldout cells include test MSE and test accuracy for the five digit regimes. "
        "For MSE, positive means lower MSE than best MLP; for accuracy, positive means higher "
        "accuracy than best MLP.",
        "",
        "## Available No-Poly / No-Unified Evidence",
        "",
        markdown_table(
            [
                "run",
                "method",
                "datasets",
                "mean final MSE margin",
                "worst margin",
                "worst W/L/T",
                "runtime vs fastest MLP",
            ],
            [
                [
                    row["label"],
                    row["method"],
                    row["datasets"],
                    fmt_float(row["mean_final_mse_margin"]),
                    fmt_float(row["worst_margin"]),
                    row["worst_wins"],
                    f"{fmt_float(row['runtime_ratio_mean'], 2)}x",
                ]
                for row in variant_rows
            ],
        ),
        "",
        "The no-poly and no-unified artifacts are only available on synthetic_compositional; "
        "they are not all14 evidence.",
        "",
        "## Existing OPMNIST Evidence",
        "",
        markdown_table(
            [
                "run",
                "method",
                "steps",
                "blocks",
                "core protocol",
                "published task count",
                "final MSE margin",
                "final acc margin",
                "test MSE margin",
                "test acc margin",
                "elapsed s",
                "steps/s",
            ],
            [
                [
                    row["label"],
                    row["method"],
                    str(row["steps"]),
                    str(row["task_blocks"]),
                    str(row["core_protocol"]),
                    str(row["published_task_count"]),
                    fmt_float(row["final_mse_margin"]),
                    fmt_float(row["final_accuracy_margin"]),
                    fmt_float(row["test_mse_margin"]),
                    fmt_float(row["test_accuracy_margin"]),
                    fmt_float(row["opmnist_elapsed_s"] or row["wall_clock_s"], 1),
                    fmt_float(row["opmnist_steps_per_second"], 1),
                ]
                for row in opmnist_rows
            ],
        ),
        "",
        "This OPMNIST evidence is for the existing strict Step 2 portfolio, not D18. It "
        "therefore proves that the project has a partial true-MNIST OPMNIST runner and a "
        "positive non-D18 result, but it does not yet prove the simplified D18 candidate on "
        "OPMNIST.",
        "",
        "## Recommendation",
        "",
        recommendation(all14_rows, digit_rows),
        "",
        "Engineering conclusion: prune the random tanh basis first, then confirm at 10 seeds. "
        "Do not prune the RKHS bank budgets until a learned allocator or replacement mechanism "
        "recovers the half/quarter-budget losses. The next production track should be a fused "
        "JAX implementation of the smallest confirmed candidate, with D18 wired into the "
        "OPMNIST runner before making external-benchmark claims.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    DOC_PATH.write_text(report)
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

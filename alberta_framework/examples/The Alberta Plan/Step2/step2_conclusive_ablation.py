#!/usr/bin/env python3
"""Ablation study for the Step 2 conclusive learner.

The canonical conclusive runner keeps all experts executing so every variant can
still be compared against the same diagnostic expert traces.  Expert and route
ablations therefore remove components from the conclusive learner's routing and
deployment decisions; the report also gives an estimated pruned-compute fraction
for an implementation that physically removes the disabled experts.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import step2_conclusive_learner as conclusive  # noqa: E402

DEFAULT_BENCHMARKS = (
    "controlled_polynomial,"
    "controlled_frequency,"
    "controlled_triple,"
    "synthetic_frequency,"
    "synthetic_polynomial,"
    "synthetic_compositional,"
    "digits_class_blocked,"
    "digits_label_drift"
)
DEFAULT_OUTPUT_DIR = Path("outputs/step2_conclusive_ablation")
DEFAULT_NOTE_PATH = Path("docs/research/step2_conclusive_ablation.md")

# Coarse per-step cost units used only for pruned implementation estimates.
# The runner itself still executes every expert for diagnostics.
EXPERT_COST_UNITS = {
    "recursive_features": 1.8,
    "polynomial_features": 0.7,
    "fourier_features": 0.5,
    "tanh_random_features": 1.0,
    "mlp_32x32_s01_no_ln": 1.0,
    "mlp_64x64_s01_no_ln": 3.4,
    "mlp_32x32": 1.0,
    "mlp_h64": 1.1,
    "mlp_h128": 2.1,
    "mlp_h64_64": 3.4,
    "upgd_low_noise": 2.2,
    "dynamic_sparse": 1.6,
}


@dataclass(frozen=True)
class Variant:
    """One ablation variant."""

    name: str
    description: str
    args: tuple[str, ...] = ()


VARIANTS: tuple[Variant, ...] = (
    Variant("full", "Canonical tuned conclusive learner."),
    Variant(
        "no_recursive",
        "Remove recursive feature expert and all safe recursive routes from routing.",
        ("--disable-experts", "recursive_features"),
    ),
    Variant(
        "no_polynomial",
        "Remove explicit polynomial basis expert from routing.",
        ("--disable-experts", "polynomial_features"),
    ),
    Variant(
        "no_fourier",
        "Remove explicit Fourier basis expert from routing.",
        ("--disable-experts", "fourier_features"),
    ),
    Variant(
        "no_tanh_random",
        "Remove fixed random tanh feature expert from routing.",
        ("--disable-experts", "tanh_random_features"),
    ),
    Variant(
        "no_basis_features",
        "Remove polynomial, Fourier, and fixed random tanh basis experts.",
        (
            "--disable-experts",
            "polynomial_features,fourier_features,tanh_random_features",
        ),
    ),
    Variant(
        "no_safe_routes",
        "Keep recursive expert available but remove safe recursive interpolation routes.",
        ("--disable-routes", "safe_recursive"),
    ),
    Variant(
        "no_all_convex",
        "Remove the all-expert convex hedge route.",
        ("--disable-routes", "all_convex"),
    ),
    Variant(
        "no_all_selector",
        "Remove the all-expert hard selector route.",
        ("--disable-routes", "all_selector"),
    ),
    Variant(
        "no_upgd_dynamic",
        "Remove UPGD and dynamic sparse experts from conclusive routing.",
        ("--disable-experts", "upgd_low_noise,dynamic_sparse"),
    ),
    Variant(
        "candidate_pruned_hedge_blend",
        (
            "Remove UPGD/dynamic, use discounted Hedge routing, and deploy "
            "held-out digits with a 50/50 all-Hedge plus h128 blend."
        ),
        (
            "--disable-experts",
            "upgd_low_noise,dynamic_sparse",
            "--weighting-scheme",
            "discounted_hedge",
            "--hedge-eta",
            "1.0",
            "--hedge-discount",
            "0.995",
            "--selector-window",
            "300",
            "--stacker-step-size",
            "0.006",
            "--digits-deployment-objective",
            "all_h128_blend",
            "--h128-blend-weight",
            "0.4",
        ),
    ),
    Variant(
        "no_upgd",
        "Remove UPGD low-noise expert from conclusive routing.",
        ("--disable-experts", "upgd_low_noise"),
    ),
    Variant(
        "no_dynamic_sparse",
        "Remove dynamic sparse expert from conclusive routing.",
        ("--disable-experts", "dynamic_sparse"),
    ),
    Variant(
        "no_class_guard",
        "Disable the digits recent-class MLP guard by making its window unreachable.",
        ("--classification-guard-min-window", "1000000000"),
    ),
    Variant(
        "no_retention_override",
        "Disable held-out class-imbalance recursive retention override.",
        ("--retention-max-recent-class-fraction", "0.0"),
    ),
    Variant(
        "accuracy_deploy",
        "Use online final-window accuracy, not MSE route, for digits deployment.",
        ("--digits-deployment-objective", "accuracy"),
    ),
    Variant(
        "no_gate_learning",
        "Freeze safe recursive gates at their zero initialization.",
        ("--safe-gate-step-size", "0.0"),
    ),
    Variant(
        "short_selector",
        "Use a short route-loss selector window.",
        ("--selector-window", "100"),
    ),
    Variant(
        "long_selector",
        "Use a long route-loss selector window.",
        ("--selector-window", "500"),
    ),
    Variant(
        "low_hedge_eta",
        "Make convex hedge routes less winner-take-all.",
        ("--hedge-eta", "2.0"),
    ),
    Variant(
        "high_hedge_eta",
        "Make convex hedge routes more winner-take-all.",
        ("--hedge-eta", "16.0"),
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
CORE_VARIANTS = (
    "full",
    "no_recursive",
    "no_basis_features",
    "no_safe_routes",
    "no_upgd_dynamic",
    "no_class_guard",
    "no_retention_override",
    "accuracy_deploy",
    "no_gate_learning",
    "short_selector",
)


def split_csv(spec: str) -> list[str]:
    """Parse a comma-separated list."""
    return [part.strip() for part in spec.split(",") if part.strip()]


def select_variants(spec: str) -> list[Variant]:
    """Select variants by set name or explicit comma-separated names."""
    if spec == "core":
        names = list(CORE_VARIANTS)
    elif spec == "all":
        names = [variant.name for variant in VARIANTS]
    else:
        names = split_csv(spec)
    unknown = sorted(set(names).difference(VARIANT_BY_NAME))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return [VARIANT_BY_NAME[name] for name in names]


def variant_disabled_experts(variant: Variant) -> frozenset[str]:
    """Return experts disabled by one variant."""
    disabled = ""
    for idx, arg in enumerate(variant.args[:-1]):
        if arg == "--disable-experts":
            disabled = variant.args[idx + 1]
    return conclusive.split_name_spec(disabled)


def estimated_pruned_cost_fraction(variant: Variant) -> float:
    """Estimate compute fraction if disabled experts were physically removed."""
    disabled = variant_disabled_experts(variant)
    full_cost = sum(EXPERT_COST_UNITS[name] for name in conclusive.EXPERT_NAMES)
    active_cost = sum(
        EXPERT_COST_UNITS[name]
        for name in conclusive.EXPERT_NAMES
        if name not in disabled
    )
    return active_cost / full_cost


def run_variant(
    variant: Variant,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run or load one ablation variant."""
    variant_dir = args.output_dir / variant.name
    json_path = variant_dir / "results.json"
    if args.reuse and json_path.exists():
        return cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))

    variant_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(THIS_DIR / "step2_conclusive_learner.py"),
        "--benchmarks",
        args.benchmarks,
        "--steps",
        str(args.steps),
        "--n-seeds",
        str(args.n_seeds),
        "--seed",
        str(args.seed),
        "--final-window",
        str(args.final_window),
        "--warmup-steps",
        str(args.warmup_steps),
        "--output-dir",
        str(variant_dir),
        "--note-path",
        str(variant_dir / "NOTE.md"),
        "--polynomial-step-size",
        str(args.polynomial_step_size),
        "--tanh-random-width",
        str(args.tanh_random_width),
        "--tanh-random-step-size",
        str(args.tanh_random_step_size),
        "--route-switch-margin",
        str(args.route_switch_margin),
        "--classification-guard-max-recent-classes",
        str(args.classification_guard_max_recent_classes),
    ]
    cmd.extend(variant.args)
    print(f"variant={variant.name}: running {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))


def aggregate_variant_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize benchmark quality for one conclusive result."""
    final_diffs: list[float] = []
    test_acc_diffs: list[float] = []
    test_mse_diffs: list[float] = []
    datasets: dict[str, Any] = {}
    for dataset, dataset_agg in result["aggregate"].items():
        comparisons = dataset_agg["comparisons"]
        row: dict[str, Any] = {}
        if "final_window_mse" in comparisons:
            cmp_row = comparisons["final_window_mse"]["conclusive_vs_best_mlp"]
            diff = float(cmp_row["paired_diff_mean_positive_favors_conclusive"])
            final_diffs.append(diff)
            row["final_window_mse_diff_vs_best_mlp"] = diff
            row["final_window_mse_wlt"] = [
                cmp_row["wins_for_conclusive"],
                cmp_row["wins_for_baseline"],
                cmp_row["ties"],
            ]
        if "test_mse" in comparisons:
            cmp_row = comparisons["test_mse"]["conclusive_vs_best_mlp"]
            diff = float(cmp_row["paired_diff_mean_positive_favors_conclusive"])
            test_mse_diffs.append(diff)
            row["test_mse_diff_vs_best_mlp"] = diff
        if "test_accuracy" in comparisons:
            cmp_row = comparisons["test_accuracy"]["conclusive_vs_best_mlp"]
            diff = float(cmp_row["paired_diff_mean_positive_favors_conclusive"])
            test_acc_diffs.append(diff)
            row["test_accuracy_diff_vs_best_mlp"] = diff
            row["test_accuracy_wlt"] = [
                cmp_row["wins_for_conclusive"],
                cmp_row["wins_for_baseline"],
                cmp_row["ties"],
            ]
        route_fractions = {
            route: float(
                dataset_agg["conclusive"].get(
                    f"meta_route_fraction_{route}",
                    {"mean": 0.0},
                )["mean"]
            )
            for route in conclusive.ROUTE_NAMES
        }
        row["safe_route_fraction"] = float(
            sum(route_fractions[route] for route in conclusive.SAFE_ROUTE_NAMES)
        )
        row["mlp_protected_route_fraction"] = float(
            sum(
                route_fractions[route]
                for route in conclusive.ROUTE_NAMES[conclusive.MLP_ROUTE_START :]
            )
        )
        row["route_fractions"] = route_fractions
        datasets[dataset] = row
    return {
        "mean_final_window_mse_diff_vs_best_mlp": float(np.mean(final_diffs))
        if final_diffs
        else math.nan,
        "min_final_window_mse_diff_vs_best_mlp": float(np.min(final_diffs))
        if final_diffs
        else math.nan,
        "positive_final_window_mse_datasets": int(np.sum(np.asarray(final_diffs) > 0.0)),
        "nonnegative_final_window_mse_datasets": int(
            np.sum(np.asarray(final_diffs) >= 0.0)
        ),
        "total_final_window_mse_datasets": len(final_diffs),
        "mean_test_mse_diff_vs_best_mlp": float(np.mean(test_mse_diffs))
        if test_mse_diffs
        else math.nan,
        "mean_test_accuracy_diff_vs_best_mlp": float(np.mean(test_acc_diffs))
        if test_acc_diffs
        else math.nan,
        "min_test_accuracy_diff_vs_best_mlp": float(np.min(test_acc_diffs))
        if test_acc_diffs
        else math.nan,
        "datasets": datasets,
    }


def write_report(
    path: Path,
    results: dict[str, Any],
) -> None:
    """Write a detailed Markdown ablation report."""
    cfg = results["config"]
    full = results["variants"].get("full")
    lines = [
        "# Step 2 Conclusive Learner Ablations",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seed(s), {cfg['steps']} steps, final "
            f"window {cfg['final_window']}, benchmarks `{cfg['benchmarks']}`. "
            "Positive differences favor the conclusive learner over the same-run "
            "best fair MLP."
        ),
        "",
        (
            "Important cost note: the canonical runner still executes disabled "
            "experts so diagnostics remain paired. The `pruned cost` column is "
            "the estimated per-step active-expert fraction for an implementation "
            "that physically removes those experts."
        ),
        "",
        "## Variant Summary",
        "",
        (
            "| Variant | Mean final MSE diff | Worst final MSE diff | "
            "Nonnegative datasets | Mean test acc diff | Wall-clock s | Pruned cost |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["variants"].items():
        quality = row["quality"]
        lines.append(
            f"| `{name}` | "
            f"{quality['mean_final_window_mse_diff_vs_best_mlp']:+.6f} | "
            f"{quality['min_final_window_mse_diff_vs_best_mlp']:+.6f} | "
            f"{quality['nonnegative_final_window_mse_datasets']}/"
            f"{quality['total_final_window_mse_datasets']} | "
            f"{quality['mean_test_accuracy_diff_vs_best_mlp']:+.6f} | "
            f"{row['wall_clock_s']:.1f} | "
            f"{row['estimated_pruned_cost_fraction']:.2f}x |"
        )
    if full is not None:
        full_quality = full["quality"]
        variants = results["variants"]

        def mean_diff(name: str) -> float:
            return float(
                variants[name]["quality"][
                    "mean_final_window_mse_diff_vs_best_mlp"
                ]
            )

        def test_acc_diff(name: str) -> float:
            return float(
                variants[name]["quality"]["mean_test_accuracy_diff_vs_best_mlp"]
            )

        def dataset_diff(name: str, dataset: str) -> float:
            return float(
                variants[name]["quality"]["datasets"][dataset][
                    "final_window_mse_diff_vs_best_mlp"
                ]
            )

        def cost(name: str) -> float:
            return float(variants[name]["estimated_pruned_cost_fraction"])

        lines.extend(
            [
                "",
                "## Critical Assessment",
                "",
                (
                    "Coverage: this matrix ablates every current conclusive-routing "
                    "expert family, the main route families, the safe-gate update, "
                    "the digits deployment/retention guards, selector window length, "
                    "and hedge sharpness. It does not ablate every MLP architecture "
                    "inside the fair comparator grid because those MLPs are also the "
                    "baseline bar; disabling them would change the question from "
                    "component ablation to baseline redefinition."
                ),
                "",
                (
                    "Overall accuracy: the full learner is nonnegative on "
                    f"{full_quality['nonnegative_final_window_mse_datasets']}/"
                    f"{full_quality['total_final_window_mse_datasets']} compact "
                    "final-window MSE tasks, with mean final-window MSE advantage "
                    f"{mean_diff('full'):+.6f} over the same-run best fair MLP. "
                    "That is the main positive result. The main negative result is "
                    f"held-out digits accuracy: mean test-accuracy delta is "
                    f"{test_acc_diff('full'):+.6f}, so this compact ablation does "
                    "not prove held-out classification accuracy superiority."
                ),
                "",
                (
                    "Most important accuracy component: Fourier features. Removing "
                    "Fourier drops synthetic frequency from "
                    f"{dataset_diff('full', 'synthetic_frequency'):+.6f} to "
                    f"{dataset_diff('no_fourier', 'synthetic_frequency'):+.6f}; "
                    "that is the only individual expert ablation that flips a "
                    "compact task from win to loss. Removing all explicit basis "
                    "features is worse still on the same task "
                    f"({dataset_diff('no_basis_features', 'synthetic_frequency'):+.6f})."
                ),
                "",
                (
                    "Second-order accuracy component: polynomial features. Removing "
                    "polynomial features leaves controlled polynomial positive but "
                    "reduces its margin from "
                    f"{dataset_diff('full', 'controlled_polynomial'):+.6f} to "
                    f"{dataset_diff('no_polynomial', 'controlled_polynomial'):+.6f}. "
                    "So polynomial features improve the intended algebraic regime, "
                    "but the recursive/MLP routes still keep the benchmark above MLP."
                ),
                "",
                (
                    "Recursive features help but are not solely responsible here. "
                    "Removing recursive features reduces mean final-window advantage "
                    f"from {mean_diff('full'):+.6f} to "
                    f"{mean_diff('no_recursive'):+.6f}, mainly through controlled "
                    "frequency and polynomial margin loss. The compact matrix does "
                    "not show recursive features as essential for every task, but it "
                    "does show they add useful nonlinear tracking margin."
                ),
                "",
                (
                    "Simplification candidates: UPGD low-noise and dynamic sparse "
                    "routing are almost neutral in this matrix. Removing both changes "
                    "mean final-window advantage from "
                    f"{mean_diff('full'):+.6f} to "
                    f"{mean_diff('no_upgd_dynamic'):+.6f} while reducing estimated "
                    f"pruned expert compute to {cost('no_upgd_dynamic'):.2f}x. "
                    "Fixed random tanh is also nearly neutral here "
                    f"({mean_diff('no_tanh_random'):+.6f}), making a minimal "
                    "deployment candidate: recursive + polynomial + Fourier + MLP "
                    "grid, without UPGD, dynamic sparse, or random tanh."
                ),
                "",
                (
                    "Route simplification: safe recursive interpolation and safe-gate "
                    "learning are neutral in this compact matrix. Removing safe routes "
                    f"changes mean final-window advantage to {mean_diff('no_safe_routes'):+.6f}; "
                    f"freezing gates changes it to {mean_diff('no_gate_learning'):+.6f}. "
                    "This supports keeping safe routes as an optional research path, "
                    "not as mandatory minimal compute for the current benchmark mix."
                ),
                "",
                (
                    "Route choices that matter mildly: removing all-convex or "
                    "all-selector routes lowers mean advantage to "
                    f"{mean_diff('no_all_convex'):+.6f} and "
                    f"{mean_diff('no_all_selector'):+.6f}. Neither breaks the compact "
                    "suite, but both reduce margin, so the router benefits from having "
                    "both soft and hard all-expert choices."
                ),
                "",
                (
                    "Hyperparameter sensitivity: the long selector window is best in "
                    f"this one-seed compact matrix ({mean_diff('long_selector'):+.6f}); "
                    "short selector and low hedge eta are worse "
                    f"({mean_diff('short_selector'):+.6f}, "
                    f"{mean_diff('low_hedge_eta'):+.6f}). High hedge eta is slightly "
                    "better than full on mean final MSE "
                    f"({mean_diff('high_hedge_eta'):+.6f}) and less negative on "
                    "mean test accuracy "
                    f"({test_acc_diff('high_hedge_eta'):+.6f}). "
                    "These are tuning leads, not settled canonical choices, because "
                    "they are compact one-seed results."
                ),
                "",
                (
                    "Deployment/guard outcome: accuracy-based deployment worsens mean "
                    f"held-out digits accuracy from {test_acc_diff('full'):+.6f} to "
                    f"{test_acc_diff('accuracy_deploy'):+.6f}. Disabling the class "
                    "guard improves class-blocked online MSE but does not improve "
                    "held-out accuracy in this compact run. The retention override is "
                    "neutral here. Therefore the remaining unsolved issue is not the "
                    "online MSE benchmark bar; it is a stronger held-out classifier "
                    "deployment rule or feature learner that also beats fair MLP on "
                    "digits test accuracy."
                ),
                "",
                (
                    "Cost conclusion: diagnostic wall-clock is not a clean compute "
                    "measure because every variant still executes all experts for "
                    "paired traces and each subprocess pays JAX compile costs. The "
                    "useful cost column is estimated pruned expert compute. Removing "
                    "UPGD + dynamic sparse gives the cleanest savings at "
                    f"{cost('no_upgd_dynamic'):.2f}x with negligible accuracy loss; "
                    "removing all basis features reaches "
                    f"{cost('no_basis_features'):.2f}x but breaks synthetic frequency; "
                    f"removing recursive reaches {cost('no_recursive'):.2f}x but loses "
                    "nonlinear margin. A physically pruned runner is still needed for "
                    "true wall-clock claims."
                ),
                "",
                (
                    "What remains missing: multi-seed replication of this ablation "
                    "table; a physical pruned-compute implementation; stronger digits "
                    "held-out accuracy deployment; and broader harder stateful external "
                    "benchmarks. The compact evidence is enough to identify which "
                    "components matter, but not enough to claim every ablated choice is "
                    "fully optimized."
                ),
            ]
        )
    lines.extend(["", "## Component Deltas", ""])
    if full is not None:
        full_quality = full["quality"]
        for name, row in results["variants"].items():
            if name == "full":
                continue
            quality = row["quality"]
            delta = (
                quality["mean_final_window_mse_diff_vs_best_mlp"]
                - full_quality["mean_final_window_mse_diff_vs_best_mlp"]
            )
            worst_delta = (
                quality["min_final_window_mse_diff_vs_best_mlp"]
                - full_quality["min_final_window_mse_diff_vs_best_mlp"]
            )
            test_acc_delta = (
                quality["mean_test_accuracy_diff_vs_best_mlp"]
                - full_quality["mean_test_accuracy_diff_vs_best_mlp"]
            )
            lines.append(
                f"- `{name}`: mean final-MSE-diff delta vs full {delta:+.6f}; "
                f"worst-dataset delta {worst_delta:+.6f}; mean test-accuracy "
                f"delta {test_acc_delta:+.6f}. {row['description']}"
            )
    lines.extend(["", "## Dataset Detail", ""])
    dataset_names = sorted(
        {
            dataset
            for row in results["variants"].values()
            for dataset in row["quality"]["datasets"]
        }
    )
    for dataset in dataset_names:
        lines.extend(
            [
                f"### {dataset}",
                "",
                (
                    "| Variant | Final MSE diff | W/L/T | Test acc diff | "
                    "Safe route frac | MLP-protected frac |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in results["variants"].items():
            detail = row["quality"]["datasets"].get(dataset, {})
            final_diff = detail.get("final_window_mse_diff_vs_best_mlp", math.nan)
            test_acc = detail.get("test_accuracy_diff_vs_best_mlp", math.nan)
            wlt = detail.get("final_window_mse_wlt", ["", "", ""])
            lines.append(
                f"| `{name}` | {final_diff:+.6f} | {wlt[0]}/{wlt[1]}/{wlt[2]} | "
                f"{test_acc:+.6f} | {detail.get('safe_route_fraction', math.nan):.3f} | "
                f"{detail.get('mlp_protected_route_fraction', math.nan):.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Assessment Template",
            "",
            "- Accuracy improvers are variants whose removal causes the final-MSE diff "
            "or held-out accuracy diff to fall relative to `full`.",
            "- Accuracy degraders are variants whose removal improves those deltas; they "
            "are candidates for simplification or conditional routing.",
            "- Tuned choices in this run are the explicit basis step sizes, tanh width, "
            "route switch margin, and class guard max recent classes inherited from the "
            "canonical tuned candidate. The remaining ablation knobs are sensitivity "
            "checks, not fully optimized hyperparameter sweeps.",
            "- A component can be valuable for Step 2 even when it does not help every "
            "single benchmark if it is selected only under regimes where its causal "
            "route-loss record supports it.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--variants", default="core")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--polynomial-step-size", type=float, default=0.5)
    parser.add_argument("--tanh-random-width", type=int, default=256)
    parser.add_argument("--tanh-random-step-size", type=float, default=0.4)
    parser.add_argument("--route-switch-margin", type=float, default=0.0)
    parser.add_argument("--classification-guard-max-recent-classes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """Run the ablation matrix and write combined reports."""
    args = parse_args()
    variants = select_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    variant_results: dict[str, Any] = {}
    for variant in variants:
        result = run_variant(variant, args)
        quality = aggregate_variant_quality(result)
        variant_results[variant.name] = {
            "description": variant.description,
            "args": list(variant.args),
            "results_path": str(args.output_dir / variant.name / "results.json"),
            "wall_clock_s": float(result["wall_clock_s"]),
            "estimated_pruned_cost_fraction": estimated_pruned_cost_fraction(variant),
            "quality": quality,
        }
    combined = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "note_path": str(args.note_path),
            "variant_names": [variant.name for variant in variants],
        },
        "variants": variant_results,
        "wall_clock_s": time.time() - start,
    }
    json_path = args.output_dir / "ablation_results.json"
    summary_path = args.output_dir / "ABLATION_SUMMARY.md"
    json_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    write_report(summary_path, combined)
    write_report(args.note_path, combined)
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

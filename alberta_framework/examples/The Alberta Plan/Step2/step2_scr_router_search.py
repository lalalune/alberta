#!/usr/bin/env python3
"""Focused Slowly-Changing Regression router search.

This runner targets the remaining Step 2 SCR gap without changing the stream
or expert learners.  It reuses the published-stressor SCR generator and the
strict universal-portfolio expert grid, then compares causal router policies:

* the current all-expert convex Hedge reference;
* stable selection among only the fair MLP widths;
* a guarded route that falls back to the best fair-MLP router unless the full
  portfolio is clearly better;
* a slower meta router for lower-variance route switching.

The result is intentionally scoped.  A positive local ``--long-scr`` result
closes the feasible SCR comparator against the fair MLP grid, but it is not a
published-scale Dohare reproduction unless the run uses the paper SCR preset
for at least one million online examples.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_stressors: Any = importlib.import_module("step2_published_stressors")
_portfolio: Any = importlib.import_module("step2_universal_portfolio")

EXPERT_NAMES: tuple[str, ...] = tuple(_portfolio.EXPERT_NAMES)
METHOD_NAMES: tuple[str, ...] = tuple(_portfolio.METHOD_NAMES)
MLP_METHODS: tuple[str, ...] = tuple(_portfolio.MLP_METHODS)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_scr_router_search")
DEFAULT_RESULT_PREFIX = "scr_router_search"
SCR_BENCHMARK = "slowly_changing_regression"


@dataclass(frozen=True)
class RouterVariant:
    """A causal SCR router configuration."""

    name: str
    description: str
    overrides: dict[str, Any]


SCR_ROUTER_VARIANTS: tuple[RouterVariant, ...] = (
    RouterVariant(
        name="convex_reference",
        description=(
            "Current all-expert convex Hedge reference from the published "
            "stressor runner."
        ),
        overrides={
            "router_policy": "convex",
            "hedge_eta": 1.0,
            "hedge_discount": 0.995,
            "router_decay": 0.02,
            "guard_tolerance": 0.0,
        },
    ),
    RouterVariant(
        name="stable_mlp_selector",
        description=(
            "Causal EMA selector restricted to the fair MLP widths; this is a "
            "stable MLP fallback route for SCR."
        ),
        overrides={
            "router_policy": "mlp_selector",
            "hedge_eta": 1.0,
            "hedge_discount": 0.995,
            "router_decay": 0.02,
            "guard_tolerance": 0.0,
        },
    ),
    RouterVariant(
        name="guarded_best_mlp",
        description=(
            "Use the all-expert convex route only when its causal EMA is within "
            "a small tolerance of the better fair-MLP route."
        ),
        overrides={
            "router_policy": "guarded_best_mlp",
            "hedge_eta": 1.0,
            "hedge_discount": 0.995,
            "router_decay": 0.02,
            "guard_tolerance": 1e-4,
        },
    ),
    RouterVariant(
        name="slow_meta",
        description=(
            "Lower-variance meta route with slower router EMA and slower Hedge "
            "forgetting."
        ),
        overrides={
            "router_policy": "meta",
            "hedge_eta": 1.0,
            "hedge_discount": 0.999,
            "router_decay": 0.005,
            "guard_tolerance": 0.0,
        },
    ),
    RouterVariant(
        name="guarded_best_mlp_slow_meta",
        description=(
            "Guarded best-MLP route with slower Hedge forgetting and slower "
            "causal EMA tracking."
        ),
        overrides={
            "router_policy": "guarded_best_mlp",
            "hedge_eta": 1.0,
            "hedge_discount": 0.999,
            "router_decay": 0.005,
            "guard_tolerance": 1e-4,
        },
    ),
    RouterVariant(
        name="guarded_best_mlp_slow_rewire",
        description=(
            "Guarded best-MLP route with less frequent dynamic sparse rewiring; "
            "the fair MLP grid is unchanged."
        ),
        overrides={
            "router_policy": "guarded_best_mlp",
            "hedge_eta": 1.0,
            "hedge_discount": 0.995,
            "router_decay": 0.02,
            "guard_tolerance": 1e-4,
            "dynamic_rewire_interval": 2_000,
        },
    ),
    RouterVariant(
        name="guarded_best_mlp_step_002",
        description=(
            "Guarded best-MLP route with a smaller learner step size applied "
            "uniformly to all experts, including the fair MLP comparator."
        ),
        overrides={
            "router_policy": "guarded_best_mlp",
            "hedge_eta": 1.0,
            "hedge_discount": 0.995,
            "router_decay": 0.02,
            "guard_tolerance": 1e-4,
            "step_size": 0.02,
        },
    ),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in SCR_ROUTER_VARIANTS}


def stderr(values: np.ndarray) -> float:
    """Return standard error."""
    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.shape[0]))


def parse_csv(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI values."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def expand_variant_names(value: str) -> tuple[str, ...]:
    """Expand a router-variant selector."""
    if value == "all":
        return tuple(variant.name for variant in SCR_ROUTER_VARIANTS)
    names = parse_csv(value)
    unknown = sorted(set(names) - set(VARIANTS_BY_NAME))
    if unknown:
        raise ValueError(f"unknown router variant(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("--router-variants cannot be empty")
    return names


def clone_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Return a detached argparse namespace."""
    return argparse.Namespace(**copy.deepcopy(vars(args)))


def args_for_variant(
    args: argparse.Namespace,
    variant: RouterVariant,
) -> argparse.Namespace:
    """Return run args with one variant's overrides applied."""
    run_args = clone_namespace(args)
    for key, value in variant.overrides.items():
        setattr(run_args, key, value)
    return run_args


def best_mlp_values(records: list[dict[str, Any]], metric: str) -> np.ndarray:
    """Return the per-record best fair MLP value for one metric."""
    values: list[float] = []
    for record in records:
        method_values = [
            float(record["methods"][method][metric])
            for method in MLP_METHODS
            if metric in record["methods"][method]
        ]
        if not method_values:
            raise KeyError(f"metric {metric!r} not found on fair MLP methods")
        values.append(min(method_values))
    return np.asarray(values, dtype=np.float64)


def summarize_best_mlp(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Summarize the per-seed best fair MLP metric."""
    values = best_mlp_values(records, metric)
    return {
        "mean": float(np.mean(values)),
        "stderr": stderr(values),
        "values": values.tolist(),
    }


def variant_summary(
    variant: RouterVariant,
    records: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate one router variant."""
    aggregate_by_dataset = _stressors.aggregate_records(records)
    aggregate = aggregate_by_dataset[SCR_BENCHMARK]
    status_input = {
        "aggregate": aggregate_by_dataset,
        "datasets": {SCR_BENCHMARK: dataset_meta},
    }
    status = _stressors.benchmark_status(status_input)
    diff = aggregate["comparisons"]["final_window_mse"]["mixture_vs_best_mlp"]
    feasible_closed = bool(
        diff["paired_diff_mean_positive_favors_mixture"] >= 0.0
        and diff["wins_for_baseline"] == 0
    )
    published_scale_closed = bool(
        feasible_closed
        and status.get("published_scale_scr_claim_supported", False)
        and status.get("matches_dohare_public_scr_protocol", False)
    )
    return {
        "description": variant.description,
        "overrides": dict(variant.overrides),
        "records": records,
        "aggregate": aggregate,
        "best_fair_mlp": {
            "final_window_mse": summarize_best_mlp(records, "final_window_mse"),
            "online_mean_mse": summarize_best_mlp(records, "online_mean_mse"),
        },
        "status": status,
        "scr_protocol": {
            "uses_dohare_public_scr_config": status[
                "uses_dohare_public_scr_config"
            ],
            "scr_steps": status["scr_steps"],
            "scr_min_published_steps": status["scr_min_published_steps"],
            "scr_meets_published_step_count": status[
                "scr_meets_published_step_count"
            ],
            "scr_task_id_provided_to_learner": status[
                "scr_task_id_provided_to_learner"
            ],
            "scr_uses_online_stream_only": status["scr_uses_online_stream_only"],
            "matches_dohare_public_scr_protocol": status[
                "matches_dohare_public_scr_protocol"
            ],
        },
        "scr_feasible_gap_closed_vs_best_fair_mlp": feasible_closed,
        "published_scale_scr_closed": published_scale_closed,
    }


def run_variant(
    variant: RouterVariant,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all seeds for one SCR router variant."""
    run_args = args_for_variant(args, variant)
    records: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}
    for run_idx in range(run_args.n_seeds):
        seed = run_args.seed + run_idx
        print(f"variant={variant.name} seed={seed}: running SCR portfolio")
        record, meta = _stressors.run_slowly_changing_regression_seed(seed, run_args)
        record["router_variant"] = variant.name
        records.append(record)
        dataset_meta = meta
        methods = record["methods"]
        best_mlp = min(methods[name]["final_window_mse"] for name in MLP_METHODS)
        print(
            f"variant={variant.name} seed={seed}: final MSE "
            f"router={methods['mixture']['final_window_mse']:.6f}, "
            f"best_mlp={best_mlp:.6f}"
        )
    return records, dataset_meta


def select_best_variant(variant_results: dict[str, Any]) -> str:
    """Return the variant with the best mean paired final-window MSE diff."""

    def score(name: str) -> float:
        row = variant_results[name]["aggregate"]["comparisons"]["final_window_mse"][
            "mixture_vs_best_mlp"
        ]
        return float(row["paired_diff_mean_positive_favors_mixture"])

    return max(variant_results, key=score)


def metric_cell(row: dict[str, Any], metric: str) -> str:
    """Format one aggregate metric cell."""
    if metric not in row:
        return ""
    return f"{row[metric]['mean']:.6f} +/- {row[metric]['stderr']:.6f}"


def comparison_cell(variant_result: dict[str, Any]) -> str:
    """Format the primary comparison."""
    row = variant_result["aggregate"]["comparisons"]["final_window_mse"][
        "mixture_vs_best_mlp"
    ]
    return (
        f"{row['paired_diff_mean_positive_favors_mixture']:+.6f} +/- "
        f"{row['paired_diff_stderr']:.6f}; "
        f"{row['wins_for_mixture']}/{row['wins_for_baseline']}/{row['ties']}"
    )


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write a Markdown run summary."""
    cfg = results["config"]
    best_name = results["best_variant"]
    best = results["variants"][best_name]
    lines = [
        "# Step 2 SCR Router Search",
        "",
        (
            f"Protocol: {cfg['n_seeds']} seeds, {cfg['steps']} online steps, "
            f"final window {cfg['final_window']}, SCR preset `{cfg['scr_preset']}`."
        ),
        "",
        (
            "All variants reuse the same SCR stream family, the same fair MLP "
            "comparator grid, and the same non-MLP experts. Positive paired "
            "differences mean best fair MLP final-window MSE minus router "
            "final-window MSE."
        ),
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## Router Results",
        "",
        "| Variant | Router Final MSE | Best Fair MLP Final MSE | Diff vs Best MLP | Closed? |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, variant_result in results["variants"].items():
        aggregate = variant_result["aggregate"]
        best_mlp = variant_result["best_fair_mlp"]["final_window_mse"]
        closed = variant_result["scr_feasible_gap_closed_vs_best_fair_mlp"]
        lines.append(
            f"| `{name}` | {metric_cell(aggregate['mixture'], 'final_window_mse')} | "
            f"{best_mlp['mean']:.6f} +/- {best_mlp['stderr']:.6f} | "
            f"{comparison_cell(variant_result)} | `{closed}` |"
        )
    lines.extend(
        [
            "",
            "## Best Variant",
            "",
            f"Best router: `{best_name}`.",
            "",
            best["description"],
            "",
            (
                "Feasible SCR comparator closed vs best fair MLP: "
                f"`{best['scr_feasible_gap_closed_vs_best_fair_mlp']}`."
            ),
        (
            "Published-scale SCR reproduction closed: "
            f"`{best['published_scale_scr_closed']}`."
        ),
        (
            "SCR protocol gate: "
            f"`matches_dohare_public_scr_protocol="
            f"{best['scr_protocol']['matches_dohare_public_scr_protocol']}` "
            f"(config={best['scr_protocol']['uses_dohare_public_scr_config']}, "
            f"steps={best['scr_protocol']['scr_steps']}/"
            f"{best['scr_protocol']['scr_min_published_steps']}, "
            f"no task id="
            f"{not best['scr_protocol']['scr_task_id_provided_to_learner']}, "
            f"online stream="
            f"{best['scr_protocol']['scr_uses_online_stream_only']})."
        ),
        "",
    ]
    )
    if not best["published_scale_scr_closed"]:
        lines.extend(
            [
                "The published-scale flag remains false unless the run uses the "
                "Dohare public SCR configuration for at least 1,000,000 online "
                "examples. A shorter `--long-scr` run should be reported as a "
                "feasible local SCR closure only.",
                "",
            ]
        )
    lines.extend(["## Variant Details", ""])
    for name, variant_result in results["variants"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                variant_result["description"],
                "",
                "Overrides:",
                "",
                "```json",
                json.dumps(variant_result["overrides"], indent=2),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def config_dict(
    args: argparse.Namespace,
    variant_names: Iterable[str],
) -> dict[str, Any]:
    """Return JSON-serializable command configuration."""
    return {
        "variant_names": list(variant_names),
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "final_window": args.final_window,
        "scr_preset": args.scr_preset,
        "long_scr": args.long_scr,
        "million_scr": args.million_scr,
        "regression_bits": args.regression_bits,
        "regression_slow_bits": args.regression_slow_bits,
        "regression_flip_interval": args.regression_flip_interval,
        "regression_target_hidden": args.regression_target_hidden,
        "regression_beta": args.regression_beta,
        "regression_noise_std": args.regression_noise_std,
        "expert_names": list(EXPERT_NAMES),
        "mlp_comparator_methods": list(MLP_METHODS),
        "step_size": args.step_size,
        "sparsity": args.sparsity,
        "perturbation_sigma": args.perturbation_sigma,
        "perturbation_warmup_steps": args.perturbation_warmup_steps,
        "perturbation_ramp_steps": args.perturbation_ramp_steps,
        "dynamic_hidden_size": args.dynamic_hidden_size,
        "dynamic_utility_decay": args.dynamic_utility_decay,
        "dynamic_rewire_interval": args.dynamic_rewire_interval,
        "dynamic_unit_replacement_rate": args.dynamic_unit_replacement_rate,
        "base_hedge_eta": args.hedge_eta,
        "base_hedge_discount": args.hedge_discount,
        "base_router_policy": args.router_policy,
        "base_router_decay": args.router_decay,
        "base_guard_tolerance": args.guard_tolerance,
        "online_retention_mse_guard": args.online_retention_mse_guard,
        "online_retention_min_lifetime_class_fraction": (
            args.online_retention_min_lifetime_class_fraction
        ),
        "online_retention_max_recent_class_fraction": (
            args.online_retention_max_recent_class_fraction
        ),
        "output_dir": str(args.output_dir),
        "result_prefix": args.result_prefix,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument(
        "--router-variants",
        default="all",
        help=(
            "Comma-separated variants or 'all'. Available: "
            + ", ".join(VARIANTS_BY_NAME)
        ),
    )
    parser.add_argument(
        "--scr-preset",
        choices=("compact", "dohare_small", "dohare_paper"),
        default="compact",
    )
    parser.add_argument(
        "--long-scr",
        action="store_true",
        help=(
            "Feasible longer local SCR run: 3 seeds, 20000 steps, final "
            "window 5000, Dohare-small SCR parameters."
        ),
    )
    parser.add_argument(
        "--million-scr",
        action="store_true",
        help=(
            "Published-scale SCR attempt: 1 seed, 1,000,000 steps, final "
            "window 100,000, Dohare public SCR parameters."
        ),
    )
    parser.add_argument("--regression-bits", type=int, default=20)
    parser.add_argument("--regression-slow-bits", type=int, default=5)
    parser.add_argument("--regression-flip-interval", type=int, default=50)
    parser.add_argument("--regression-target-hidden", type=int, default=100)
    parser.add_argument("--regression-beta", type=float, default=0.7)
    parser.add_argument("--regression-noise-std", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--perturbation-sigma", type=float, default=1e-4)
    parser.add_argument("--perturbation-warmup-steps", type=int, default=0)
    parser.add_argument("--perturbation-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamic-hidden-size", type=int, default=64)
    parser.add_argument("--dynamic-utility-decay", type=float, default=0.99)
    parser.add_argument("--dynamic-rewire-interval", type=int, default=240)
    parser.add_argument("--dynamic-unit-replacement-rate", type=float, default=0.05)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--hedge-discount", type=float, default=0.995)
    parser.add_argument(
        "--router-policy",
        choices=(
            "convex",
            "all_selector",
            "mlp_convex",
            "mlp_selector",
            "guarded_convex",
            "guarded_best_mlp",
            "meta",
        ),
        default="convex",
    )
    parser.add_argument("--router-decay", type=float, default=0.02)
    parser.add_argument("--guard-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--online-retention-mse-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--online-retention-min-lifetime-class-fraction",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--online-retention-max-recent-class-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--note-path", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast one-seed harness check with tiny SCR streams.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    return build_parser().parse_args()


def apply_run_preset(args: argparse.Namespace) -> None:
    """Apply SCR run presets in-place."""
    if sum(bool(value) for value in (args.smoke, args.long_scr, args.million_scr)) > 1:
        raise ValueError("--smoke, --long-scr, and --million-scr are mutually exclusive")

    if args.scr_preset == "dohare_small":
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 1_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7
    elif args.scr_preset == "dohare_paper":
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 10_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7

    if args.smoke:
        args.steps = 120
        args.n_seeds = 1
        args.final_window = 40
        args.regression_bits = 8
        args.regression_slow_bits = 3
        args.regression_flip_interval = 20
        args.regression_target_hidden = 16
        args.dynamic_rewire_interval = 60
    elif args.long_scr:
        args.steps = 20_000
        args.n_seeds = 3
        args.final_window = 5_000
        args.scr_preset = "dohare_small"
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 1_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7
        args.dynamic_rewire_interval = 500
    elif args.million_scr:
        args.steps = 1_000_000
        args.n_seeds = 1
        args.final_window = 100_000
        args.scr_preset = "dohare_paper"
        args.regression_bits = 20
        args.regression_slow_bits = 15
        args.regression_flip_interval = 10_000
        args.regression_target_hidden = 100
        args.regression_beta = 0.7
        args.dynamic_rewire_interval = 2_000


def validate_args(args: argparse.Namespace) -> None:
    """Validate command arguments."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive")
    if args.final_window <= 0:
        raise ValueError("--final-window must be positive")
    if args.regression_bits <= 0:
        raise ValueError("--regression-bits must be positive")
    if not 0 < args.regression_slow_bits <= args.regression_bits:
        raise ValueError("--regression-slow-bits must be in [1, regression_bits]")
    if args.regression_flip_interval <= 0:
        raise ValueError("--regression-flip-interval must be positive")
    if args.regression_target_hidden <= 0:
        raise ValueError("--regression-target-hidden must be positive")
    if args.regression_noise_std < 0.0:
        raise ValueError("--regression-noise-std must be non-negative")
    if args.perturbation_warmup_steps < 0:
        raise ValueError("--perturbation-warmup-steps must be non-negative")
    if args.perturbation_ramp_steps < 0:
        raise ValueError("--perturbation-ramp-steps must be non-negative")
    if args.dynamic_rewire_interval <= 0:
        raise ValueError("--dynamic-rewire-interval must be positive")
    if not 0.0 <= args.dynamic_unit_replacement_rate <= 1.0:
        raise ValueError("--dynamic-unit-replacement-rate must be in [0, 1]")
    if not 0.0 <= args.hedge_discount <= 1.0:
        raise ValueError("--hedge-discount must be in [0, 1]")
    if not 0.0 < args.router_decay <= 1.0:
        raise ValueError("--router-decay must be in (0, 1]")


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected SCR router search."""
    apply_run_preset(args)
    validate_args(args)
    variant_names = expand_variant_names(args.router_variants)

    t0 = time.time()
    variant_results: dict[str, Any] = {}
    datasets_meta: dict[str, Any] = {}
    for name in variant_names:
        variant = VARIANTS_BY_NAME[name]
        records, meta = run_variant(variant, args)
        variant_results[name] = variant_summary(variant, records, meta)
        datasets_meta[name] = meta

    best_name = select_best_variant(variant_results)
    results = {
        "config": config_dict(args, variant_names),
        "datasets": datasets_meta,
        "variants": variant_results,
        "best_variant": best_name,
        "best_variant_status": variant_results[best_name]["status"],
        "scr_feasible_gap_closed_vs_best_fair_mlp": variant_results[best_name][
            "scr_feasible_gap_closed_vs_best_fair_mlp"
        ],
        "published_scale_scr_closed": variant_results[best_name][
            "published_scale_scr_closed"
        ],
        "wall_clock_s": time.time() - t0,
        "evidence_level": "focused_scr_router_search",
    }
    return results


def main() -> None:
    """Run the focused SCR router search."""
    args = parse_args()
    results = run_search(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.result_prefix}_results.json"
    summary_path = args.output_dir / f"{args.result_prefix}_SUMMARY.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_summary(summary_path, results)
    if args.note_path is not None:
        write_summary(args.note_path, results)
    print(f"best_variant={results['best_variant']}")
    print(
        "scr_feasible_gap_closed_vs_best_fair_mlp="
        f"{results['scr_feasible_gap_closed_vs_best_fair_mlp']}"
    )
    print(f"published_scale_scr_closed={results['published_scale_scr_closed']}")
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    if args.note_path is not None:
        print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

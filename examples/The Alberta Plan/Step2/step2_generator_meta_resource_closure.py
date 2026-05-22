#!/usr/bin/env python3
"""Generator-internal meta-resource closure probes for Alberta Plan Step 2.

This runner tests whether learned resource management helps inside the
compositional feature generator itself.  It compares a fixed residual-imprint
generator against compact learned variants:

* contextual Hedge over generator policies,
* EXP3-style sampled-policy credit,
* operation/parent priors with promotion credit and resource costs.

The probes are deliberately small and online: nonlinear, pair interaction,
recursive triple interaction, and rare-feature masked-head retention.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

PROBES = ("nonlinear", "interaction", "recursive", "rare_feature")
VARIANTS = (
    "fixed_residual",
    "hedge",
    "exp3",
    "hedge_priors_credit_budget",
)


def make_probe_data(
    probe: str,
    seed: int,
    num_steps: int,
    feature_dim: int,
    rare_period: int,
    noise_std: float,
) -> tuple[Array, Array, np.ndarray]:
    """Create one online supervised probe."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_steps, feature_dim)).astype(np.float32)
    noise = noise_std * rng.standard_normal(num_steps).astype(np.float32)
    active_heads = np.zeros(num_steps, dtype=np.int32)

    if probe == "nonlinear":
        y = (
            np.sin(x[:, 0])
            + 0.5 * x[:, 1] * x[:, 2]
            + 0.25 * np.tanh(2.0 * x[:, 3] + x[:, 4])
            + noise
        )[:, None]
    elif probe == "interaction":
        y = (
            x[:, 0] * x[:, 1]
            - 0.7 * x[:, 2] * x[:, 3]
            + 0.5 * x[:, 4] * x[:, 5]
            + noise
        )[:, None]
    elif probe == "recursive":
        y = (x[:, 0] * x[:, 1] * x[:, 2] + 0.3 * x[:, 3] + noise)[:, None]
    elif probe == "rare_feature":
        y = np.full((num_steps, 3), np.nan, dtype=np.float32)
        for step in range(num_steps):
            if step % rare_period == 0:
                y[step, 2] = 4.0 * x[step, 0] * x[step, 1] + noise[step]
                active_heads[step] = 2
            elif step % 2 == 0:
                y[step, 0] = x[step, 2] * x[step, 3] + noise[step]
                active_heads[step] = 0
            else:
                y[step, 1] = x[step, 4] * x[step, 5] + noise[step]
                active_heads[step] = 1
    else:
        raise ValueError(f"unknown probe: {probe}")

    return jnp.asarray(x), jnp.asarray(y), active_heads


def make_learner(
    variant: str,
    n_tasks: int,
    args: argparse.Namespace,
) -> CompositionalFeatureLearner:
    """Create one generator-resource variant."""
    common: dict[str, Any] = {
        "n_features": args.n_features,
        "n_tasks": n_tasks,
        "candidate_count": args.candidate_count,
        "step_size_output": args.step_size_output,
        "step_size_theta": args.step_size_feature,
        "utility_decay": args.utility_decay,
        "replacement_interval": args.replacement_interval,
        "min_feature_age": args.min_feature_age,
        "candidate_min_age": args.candidate_min_age,
        "promotion_margin": args.promotion_margin,
        "promotion_blend": 0.5,
        "promotion_output_mode": "blend",
        "max_depth": 3,
        "use_obgd": True,
        "obgd_kappa": args.obgd_kappa,
        "generation_strategy": "residual_imprint",
        "parent_temperature": 0.75,
        "residual_guidance": 0.75,
        "candidate_imprint_scale": 0.2,
        "future_utility_mix": 1.0,
        "future_utility_trace_decay": 0.9,
    }
    if variant == "fixed_residual":
        return CompositionalFeatureLearner(**common)
    if variant == "hedge":
        return CompositionalFeatureLearner(
            **common,
            learn_generator_resources=True,
            generator_resource_learning_rate=args.generator_resource_learning_rate,
            generator_resource_exploration=args.generator_resource_exploration,
            generator_resource_update_rule="hedge",
        )
    if variant == "exp3":
        return CompositionalFeatureLearner(
            **common,
            learn_generator_resources=True,
            generator_resource_learning_rate=args.exp3_learning_rate,
            generator_resource_exploration=max(args.generator_resource_exploration, 0.10),
            generator_resource_advantage_clip=2.0,
            generator_resource_update_rule="exp3",
        )
    if variant == "hedge_priors_credit_budget":
        return CompositionalFeatureLearner(
            **common,
            learn_generator_resources=True,
            generator_resource_learning_rate=args.generator_resource_learning_rate,
            generator_resource_exploration=args.generator_resource_exploration,
            generator_resource_cost_weight=0.05,
            generator_resource_promotion_credit=1.0,
            generator_resource_initial_preferences=(0.0, 0.4, 0.6, 0.4),
            generator_resource_update_rule="hedge",
        )
    raise ValueError(f"unknown variant: {variant}")


def run_arrays_with_errors(
    learner: CompositionalFeatureLearner,
    state: CompositionalFeatureState,
    observations: Array,
    targets: Array,
) -> tuple[CompositionalFeatureState, Array, Array]:
    """Run a learner and return final state, metrics, and per-head errors."""

    def step_fn(
        carry: CompositionalFeatureState,
        inputs: tuple[Array, Array],
    ) -> tuple[CompositionalFeatureState, tuple[Array, Array]]:
        observation, target = inputs
        result = learner.update(carry, observation, target)
        return result.state, (result.metrics, result.errors)

    final_state, (metrics, errors) = jax.lax.scan(
        step_fn, state, (observations, targets)
    )
    return final_state, metrics, errors


def final_policy_weights(
    learner: CompositionalFeatureLearner,
    state: CompositionalFeatureState,
) -> list[float]:
    """Return final generator-policy weights if the variant has a manager."""
    manager = learner._generator_resource_manager  # noqa: SLF001
    return [float(value) for value in np.asarray(manager.weights(state.generator_resource_state))]


def run_one(
    probe: str,
    variant: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one probe/variant/seed cell."""
    observations, targets, active_heads = make_probe_data(
        probe=probe,
        seed=1000 * seed + 17,
        num_steps=args.num_steps,
        feature_dim=args.feature_dim,
        rare_period=args.rare_period,
        noise_std=args.noise_std,
    )
    learner = make_learner(variant, targets.shape[1], args)
    state = learner.init(args.feature_dim, jr.key(50_000 + seed))
    t0 = time.time()
    final_state, metrics, errors = run_arrays_with_errors(
        learner, state, observations, targets
    )
    metrics.block_until_ready()
    elapsed = time.time() - t0

    metrics_np = np.asarray(metrics, dtype=np.float64)
    errors_np = np.asarray(errors, dtype=np.float64)
    final_window = min(args.final_window, args.num_steps)
    final_metrics = metrics_np[-final_window:]
    final_errors = errors_np[-final_window:]
    active_final_mse = float(np.nanmean(final_errors**2))
    rare_final_mse = float("nan")
    if probe == "rare_feature":
        rare_mask = active_heads[-final_window:] == 2
        rare_final_mse = float(np.nanmean(final_errors[rare_mask, 2] ** 2))

    policy_weights = (
        final_policy_weights(learner, final_state)
        if variant != "fixed_residual"
        else []
    )
    depths = np.asarray(final_state.depth)
    return {
        "probe": probe,
        "variant": variant,
        "seed": seed,
        "final_window_mse": float(np.mean(final_metrics[:, 0])),
        "active_final_mse": active_final_mse,
        "rare_final_mse": rare_final_mse,
        "mean_mse": float(np.mean(metrics_np[:, 0])),
        "replacement_events": float(np.sum(metrics_np[:, 5])),
        "depth2_plus_count": int(np.sum(depths >= 2)),
        "max_depth": int(np.max(depths)),
        "policy_weights": policy_weights,
        "finite_metrics": bool(np.all(np.isfinite(metrics_np))),
        "wall_clock_s": elapsed,
    }


def stderr(values: list[float]) -> float:
    """Return standard error."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by probe and variant, plus paired fixed-minus-variant deltas."""
    out: dict[str, Any] = {}
    for probe in PROBES:
        out[probe] = {}
        probe_rows = [row for row in records if row["probe"] == probe]
        for variant in VARIANTS:
            rows = [row for row in probe_rows if row["variant"] == variant]
            if not rows:
                continue
            finals = [float(row["final_window_mse"]) for row in rows]
            active = [float(row["active_final_mse"]) for row in rows]
            rare = [
                float(row["rare_final_mse"])
                for row in rows
                if math.isfinite(float(row["rare_final_mse"]))
            ]
            out[probe][variant] = {
                "mean_final_window_mse": float(np.mean(finals)),
                "stderr_final_window_mse": stderr(finals),
                "mean_active_final_mse": float(np.mean(active)),
                "stderr_active_final_mse": stderr(active),
                "mean_rare_final_mse": float(np.mean(rare)) if rare else float("nan"),
                "stderr_rare_final_mse": stderr(rare) if rare else float("nan"),
                "mean_replacement_events": float(
                    np.mean([row["replacement_events"] for row in rows])
                ),
                "mean_depth2_plus_count": float(
                    np.mean([row["depth2_plus_count"] for row in rows])
                ),
                "all_metrics_finite": bool(all(row["finite_metrics"] for row in rows)),
            }
        fixed_by_seed = {
            int(row["seed"]): row
            for row in probe_rows
            if row["variant"] == "fixed_residual"
        }
        paired: dict[str, Any] = {}
        for variant in VARIANTS:
            if variant == "fixed_residual":
                continue
            variant_rows = [row for row in probe_rows if row["variant"] == variant]
            deltas = [
                float(fixed_by_seed[int(row["seed"])]["final_window_mse"])
                - float(row["final_window_mse"])
                for row in variant_rows
                if int(row["seed"]) in fixed_by_seed
            ]
            paired[variant] = {
                "fixed_minus_variant_final_mse": (
                    float(np.mean(deltas)) if deltas else float("nan")
                ),
                "variant_wins": int(sum(delta > 0.0 for delta in deltas)),
                "n": len(deltas),
            }
        out[probe]["paired_vs_fixed"] = paired
    return out


def choose_recommendation(summary: dict[str, Any]) -> str:
    """Return a conservative integration recommendation."""
    wins_by_variant = {variant: 0 for variant in VARIANTS if variant != "fixed_residual"}
    total_by_variant = {variant: 0 for variant in wins_by_variant}
    positive_probes: dict[str, list[str]] = {}
    for probe, probe_summary in summary.items():
        for variant, row in probe_summary["paired_vs_fixed"].items():
            total_by_variant[variant] += int(row["n"])
            wins_by_variant[variant] += int(row["variant_wins"])
            if row["fixed_minus_variant_final_mse"] > 0.0:
                positive_probes.setdefault(variant, []).append(probe)

    best = max(
        wins_by_variant,
        key=lambda variant: (
            len(positive_probes.get(variant, [])),
            wins_by_variant[variant],
        ),
    )
    enough_positive_probes = len(positive_probes.get(best, [])) >= 3
    more_seed_wins_than_losses = wins_by_variant[best] > total_by_variant[best] / 2
    if enough_positive_probes and more_seed_wins_than_losses:
        return (
            f"Canonical candidate: `{best}`. It improves the fixed generator on "
            "most probe families, but should still be staged behind an opt-in flag."
        )
    return (
        "Partial result: keep `learn_generator_resources` opt-in. The mechanism "
        "is generator-internal and causal, but this sweep is not strong enough "
        "to replace the fixed residual-imprint generator as the Step 2 default."
    )


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write Markdown summary with command and tables."""
    config = results["config"]
    lines = [
        "# Step 2 Generator-Internal Meta-Resource Closure",
        "",
        "## Command",
        "",
        "```bash",
        config["command"],
        "```",
        "",
        "## Variants",
        "",
        "- `fixed_residual`: fixed residual-imprint compositional generator.",
        "- `hedge`: contextual Hedge over generator policies.",
        "- `exp3`: sampled-policy EXP3-style update with explicit exploration.",
        "- `hedge_priors_credit_budget`: operation/parent priors, delayed "
        "promotion credit, and residual-imprint/replacement/promotion costs.",
        "",
        "## Aggregate Final-Window MSE",
        "",
    ]
    for probe in PROBES:
        lines.extend(
            [
                f"### {probe}",
                "",
                "| Variant | Final MSE | Active MSE | Rare MSE | "
                "Replacements | Depth>=2 | Finite |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        probe_summary = results["summary"][probe]
        for variant in VARIANTS:
            row = probe_summary[variant]
            rare = row["mean_rare_final_mse"]
            rare_text = "n/a" if not math.isfinite(rare) else f"{rare:.6f}"
            lines.append(
                f"| `{variant}` | {row['mean_final_window_mse']:.6f} +/- "
                f"{row['stderr_final_window_mse']:.6f} | "
                f"{row['mean_active_final_mse']:.6f} | {rare_text} | "
                f"{row['mean_replacement_events']:.1f} | "
                f"{row['mean_depth2_plus_count']:.1f} | "
                f"{row['all_metrics_finite']} |"
            )
        lines.extend(
            [
                "",
                "| Variant | Fixed - variant final MSE | Wins |",
                "|---|---:|---:|",
            ]
        )
        for variant, row in probe_summary["paired_vs_fixed"].items():
            lines.append(
                f"| `{variant}` | {row['fixed_minus_variant_final_mse']:+.6f} | "
                f"{row['variant_wins']}/{row['n']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            results["recommendation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_generator_meta_closure"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--final-window", type=int, default=300)
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--replacement-interval", type=int, default=30)
    parser.add_argument("--min-feature-age", type=int, default=40)
    parser.add_argument("--candidate-min-age", type=int, default=20)
    parser.add_argument("--promotion-margin", type=float, default=1.03)
    parser.add_argument("--step-size-output", type=float, default=0.04)
    parser.add_argument("--step-size-feature", type=float, default=0.004)
    parser.add_argument("--utility-decay", type=float, default=0.99)
    parser.add_argument("--obgd-kappa", type=float, default=2.0)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--rare-period", type=int, default=80)
    parser.add_argument("--generator-resource-learning-rate", type=float, default=0.75)
    parser.add_argument("--exp3-learning-rate", type=float, default=0.05)
    parser.add_argument("--generator-resource-exploration", type=float, default=0.03)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all probes."""
    args = parse_args()
    if args.smoke:
        args.seeds = min(args.seeds, 2)
        args.num_steps = min(args.num_steps, 300)
        args.final_window = min(args.final_window, 100)
        args.n_features = min(args.n_features, 14)
        args.candidate_count = min(args.candidate_count, 8)
        args.replacement_interval = min(args.replacement_interval, 20)
        args.min_feature_age = min(args.min_feature_age, 20)
        args.candidate_min_age = min(args.candidate_min_age, 10)
    args.final_window = min(args.final_window, args.num_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for probe in PROBES:
        for seed in range(args.seeds):
            for variant in VARIANTS:
                record = run_one(probe, variant, seed, args)
                records.append(record)
                print(
                    f"{probe:12s} seed={seed:02d} {variant:28s} "
                    f"final={record['final_window_mse']:.5f} "
                    f"finite={record['finite_metrics']}"
                )

    summary = aggregate(records)
    results = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "command": (
                'python "examples/The Alberta Plan/Step2/'
                'step2_generator_meta_resource_closure.py" '
                f"--seeds {args.seeds} --num-steps {args.num_steps} "
                f"--final-window {args.final_window} "
                f"--output-dir {args.output_dir}"
            ),
        },
        "records": records,
        "summary": summary,
        "recommendation": choose_recommendation(summary),
    }
    result_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_path, results)
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(results["recommendation"])


if __name__ == "__main__":
    main()

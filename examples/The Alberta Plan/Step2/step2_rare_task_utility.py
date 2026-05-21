"""Rare-task utility ablation for Alberta Plan Step 2.

This experiment isolates one lifecycle failure mode: a bounded feature bank can
delete a feature that matters for a recurrent but infrequent target head.  The
stream masks inactive heads with NaN targets, activates two common heads often,
and activates one rare head periodically.  All methods start with the rare
oracle pair in the active bank, so the question is retention and ranking rather
than discovery by chance.

The ablation compares the default mean utility against max, active-head
task-balanced, inverse-frequency, retention, task-balanced-retention, and
one-step counterfactual future-utility variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework import FixedBudgetInteractionLearner
from alberta_framework.core.interaction_features import InteractionFeatureState

FEATURE_DIM = 6
N_TASKS = 5
RARE_HEAD = 4
RARE_PAIR = (0, 1)
COMMON_PAIRS_BY_HEAD = ((2, 3), (2, 3), (4, 5), (4, 5))
INITIAL_LEFT = jnp.array([0, 2, 4, 0, 1, 3], dtype=jnp.int32)
INITIAL_RIGHT = jnp.array([1, 3, 5, 2, 4, 5], dtype=jnp.int32)


METHODS: dict[str, dict[str, Any]] = {
    "mean": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "none",
        "utility_retention_decay": None,
    },
    "max": {
        "utility_aggregation": "max",
        "utility_task_balancing": "none",
        "utility_retention_decay": None,
    },
    "task_balanced": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active",
        "utility_retention_decay": None,
    },
    "inverse_frequency": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active_inverse_frequency",
        "utility_retention_decay": None,
    },
    "retention": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "none",
        "utility_retention_decay": 0.9995,
    },
    "task_balanced_retention": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active",
        "utility_retention_decay": 0.9995,
    },
    "future_counterfactual": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "none",
        "utility_retention_decay": None,
        "future_utility_mix": 1.0,
    },
    "task_balanced_future": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active",
        "utility_retention_decay": None,
        "future_utility_mix": 1.0,
    },
    "task_balanced_future_retention": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active",
        "utility_retention_decay": 0.9995,
        "future_utility_mix": 0.5,
    },
    "rare_protected": {
        "utility_aggregation": "mean",
        "utility_task_balancing": "active_inverse_frequency",
        "utility_retention_decay": 0.9995,
        "future_utility_mix": 0.5,
    },
}


def make_stream(
    seed: int,
    num_steps: int,
    rare_period: int,
    rare_scale: float,
    common_scale: float,
) -> tuple[Array, Array, np.ndarray]:
    """Create a recurring-context pair-product stream with masked heads."""
    rng = np.random.default_rng(seed)
    observations = np.zeros((num_steps, FEATURE_DIM), dtype=np.float32)
    targets = np.full((num_steps, N_TASKS), np.nan, dtype=np.float32)
    active_heads = np.zeros(num_steps, dtype=np.int32)

    for step in range(num_steps):
        is_rare = step % rare_period == 0
        if is_rare:
            head = RARE_HEAD
            left, right = RARE_PAIR
            scale = rare_scale
        else:
            head = step % len(COMMON_PAIRS_BY_HEAD)
            left, right = COMMON_PAIRS_BY_HEAD[head]
            scale = common_scale

        observations[step, left] = rng.normal()
        observations[step, right] = rng.normal()
        inactive_noise = 0.02 * rng.normal(size=FEATURE_DIM).astype(np.float32)
        observations[step] += inactive_noise
        targets[step, head] = scale * observations[step, left] * observations[step, right]
        active_heads[step] = head

    return jnp.asarray(observations), jnp.asarray(targets), active_heads


def init_oracle_state(
    learner: FixedBudgetInteractionLearner,
    seed: int,
) -> InteractionFeatureState:
    """Initialize a fixed bank containing the two common pairs and rare pair."""
    state = learner.init(feature_dim=FEATURE_DIM, key=jr.key(seed))
    return state.replace(  # type: ignore[attr-defined]
        feature_left=INITIAL_LEFT,
        feature_right=INITIAL_RIGHT,
        utilities=jnp.zeros(learner.n_features, dtype=jnp.float32),
        output_weights=jnp.zeros((N_TASKS, learner.n_features), dtype=jnp.float32),
        ages=jnp.zeros(learner.n_features, dtype=jnp.int32),
    )


def run_sequence(
    learner: FixedBudgetInteractionLearner,
    state: InteractionFeatureState,
    observations: Array,
    targets: Array,
) -> tuple[InteractionFeatureState, Array, Array]:
    """Run one learner with scan while returning per-head errors."""

    def step_fn(
        carry: InteractionFeatureState,
        inputs: tuple[Array, Array],
    ) -> tuple[InteractionFeatureState, tuple[Array, Array]]:
        observation, target = inputs
        result = learner.update(carry, observation, target)
        return result.state, (result.errors, result.metrics)

    final_state, (errors, metrics) = jax.lax.scan(step_fn, state, (observations, targets))
    return final_state, errors, metrics


def contains_pair(state: InteractionFeatureState, pair: tuple[int, int]) -> bool:
    """Return whether a final feature bank contains a pair."""
    left = np.asarray(state.feature_left)
    right = np.asarray(state.feature_right)
    return bool(np.any((left == pair[0]) & (right == pair[1])))


def nanmean_or_nan(values: np.ndarray) -> float:
    """Mean that returns NaN instead of warning on an all-NaN slice."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def run_method(
    method_name: str,
    method_config: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one method/seed pair and summarize rare/common performance."""
    observations, targets, active_heads = make_stream(
        seed=10_000 + seed,
        num_steps=args.num_steps,
        rare_period=args.rare_period,
        rare_scale=args.rare_scale,
        common_scale=args.common_scale,
    )
    learner = FixedBudgetInteractionLearner(
        n_features=INITIAL_LEFT.shape[0],
        n_tasks=N_TASKS,
        step_size_output=args.step_size,
        utility_decay=args.utility_decay,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_count=0,
        generator_mix=(1.0, 0.0, 0.0),
        use_obgd=True,
        obgd_kappa=args.obgd_kappa,
        **method_config,
    )
    state = init_oracle_state(learner, seed=20_000 + seed)
    final_state, errors, metrics = run_sequence(learner, state, observations, targets)

    sq_errors = np.asarray(errors) ** 2
    final_window = min(args.final_window, args.num_steps)
    final_sq = sq_errors[-final_window:]
    final_heads = active_heads[-final_window:]
    common_mask = final_heads != RARE_HEAD

    rare_final_mse = nanmean_or_nan(final_sq[:, RARE_HEAD])
    common_final_mse = nanmean_or_nan(final_sq[common_mask, :RARE_HEAD])
    active_final_mse = nanmean_or_nan(final_sq)
    total_active_mse = nanmean_or_nan(sq_errors)
    final_pair_present = contains_pair(final_state, RARE_PAIR)
    utilities = np.asarray(final_state.utilities)
    rare_slots = (np.asarray(final_state.feature_left) == RARE_PAIR[0]) & (
        np.asarray(final_state.feature_right) == RARE_PAIR[1]
    )
    rare_utility = float(np.max(utilities[rare_slots])) if np.any(rare_slots) else 0.0

    return {
        "method": method_name,
        "seed": seed,
        "rare_final_mse": rare_final_mse,
        "common_final_mse": common_final_mse,
        "active_final_mse": active_final_mse,
        "total_active_mse": total_active_mse,
        "rare_pair_present": final_pair_present,
        "rare_utility": rare_utility,
        "replacement_count": float(np.sum(np.asarray(metrics)[:, 5])),
        "final_utilities": utilities.tolist(),
        "final_pairs": [
            [int(left), int(right)]
            for left, right in zip(
                np.asarray(final_state.feature_left),
                np.asarray(final_state.feature_right),
                strict=True,
            )
        ],
    }


def stderr(values: np.ndarray) -> float:
    """Population-standard-error helper for seed summaries."""
    if values.size == 0:
        return float("nan")
    return float(np.std(values, ddof=0) / np.sqrt(values.size))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed records and paired differences against mean utility."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record)

    summary: dict[str, Any] = {}
    for method, method_records in by_method.items():
        rare = np.array([row["rare_final_mse"] for row in method_records], dtype=np.float64)
        common = np.array(
            [row["common_final_mse"] for row in method_records], dtype=np.float64
        )
        active = np.array(
            [row["active_final_mse"] for row in method_records], dtype=np.float64
        )
        total = np.array(
            [row["total_active_mse"] for row in method_records], dtype=np.float64
        )
        summary[method] = {
            "rare_final_mse_mean": float(np.mean(rare)),
            "rare_final_mse_stderr": stderr(rare),
            "common_final_mse_mean": float(np.mean(common)),
            "common_final_mse_stderr": stderr(common),
            "active_final_mse_mean": float(np.mean(active)),
            "active_final_mse_stderr": stderr(active),
            "total_active_mse_mean": float(np.mean(total)),
            "total_active_mse_stderr": stderr(total),
            "rare_pair_survival_rate": float(
                np.mean([row["rare_pair_present"] for row in method_records])
            ),
            "replacement_count_mean": float(
                np.mean([row["replacement_count"] for row in method_records])
            ),
        }

    mean_records = {
        int(row["seed"]): row for row in by_method.get("mean", [])
    }
    paired: dict[str, Any] = {}
    for method, method_records in by_method.items():
        if method == "mean":
            continue
        paired_rows = [
            (mean_records[int(row["seed"])], row)
            for row in method_records
            if int(row["seed"]) in mean_records
        ]
        rare_diffs = np.array(
            [base["rare_final_mse"] - row["rare_final_mse"] for base, row in paired_rows],
            dtype=np.float64,
        )
        common_diffs = np.array(
            [
                row["common_final_mse"] - base["common_final_mse"]
                for base, row in paired_rows
            ],
            dtype=np.float64,
        )
        active_diffs = np.array(
            [
                base["active_final_mse"] - row["active_final_mse"]
                for base, row in paired_rows
            ],
            dtype=np.float64,
        )
        paired[method] = {
            "mean_minus_method_rare_final_mse": float(np.mean(rare_diffs)),
            "rare_wins_vs_mean": int(np.sum(rare_diffs > 0.0)),
            "method_minus_mean_common_final_mse": float(np.mean(common_diffs)),
            "common_nonharm_wins_vs_mean": int(np.sum(common_diffs <= 0.0)),
            "mean_minus_method_active_final_mse": float(np.mean(active_diffs)),
            "active_wins_vs_mean": int(np.sum(active_diffs > 0.0)),
            "n_seeds": len(paired_rows),
        }
    return {"per_method": summary, "paired_vs_mean": paired}


def choose_recommendation(summary: dict[str, Any]) -> str:
    """Return a conservative canonical recommendation from the ablation."""
    paired = summary["paired_vs_mean"]
    candidates = [
        "rare_protected",
        "task_balanced_future_retention",
        "task_balanced_future",
        "future_counterfactual",
        "task_balanced_retention",
        "task_balanced",
        "retention",
        "max",
    ]
    viable: list[tuple[str, float]] = []
    for name in candidates:
        if name not in paired:
            continue
        row = paired[name]
        n_seeds = max(int(row["n_seeds"]), 1)
        rare_win_rate = row["rare_wins_vs_mean"] / n_seeds
        common_harm = row["method_minus_mean_common_final_mse"]
        rare_gain = row["mean_minus_method_rare_final_mse"]
        if rare_win_rate >= 0.6 and rare_gain > 0.0 and common_harm <= 0.02:
            viable.append((name, rare_gain))
    if not viable:
        return "Keep mean utility as the default; use rare-task knobs only as opt-in."
    best = max(viable, key=lambda item: item[1])[0]
    return f"Use `{best}` as the rare-task-protected opt-in; do not make it global yet."


def write_summary(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Write a compact Markdown summary for review."""
    lines = [
        "# Step 2 Rare-Task Utility Ablation",
        "",
        "Controlled recurring-context test for rare-head feature retention.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {args.seeds}",
        f"- Steps per seed: {args.num_steps}",
        f"- Rare period: every {args.rare_period} steps",
        f"- Final window: {args.final_window}",
        f"- Replacement interval: {args.replacement_interval}",
        f"- Minimum feature age: {args.min_feature_age}",
        "",
        "## Aggregate Results",
        "",
        "| Method | Rare final MSE | Common final MSE | Active final MSE | "
        "Rare pair survival |",
        "|---|---:|---:|---:|---:|",
    ]
    per_method = result["summary"]["per_method"]
    for method, row in sorted(
        per_method.items(), key=lambda item: item[1]["rare_final_mse_mean"]
    ):
        lines.append(
            f"| `{method}` | {row['rare_final_mse_mean']:.6f} +/- "
            f"{row['rare_final_mse_stderr']:.6f} | "
            f"{row['common_final_mse_mean']:.6f} +/- "
            f"{row['common_final_mse_stderr']:.6f} | "
            f"{row['active_final_mse_mean']:.6f} +/- "
            f"{row['active_final_mse_stderr']:.6f} | "
            f"{row['rare_pair_survival_rate']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Comparison Against Mean Utility",
            "",
            "Positive rare/active values mean the method beat default mean utility. "
            "Negative common-harm values mean the method improved common heads too.",
            "",
            "| Method | Mean - method rare MSE | Rare wins | "
            "Method - mean common MSE | Active wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, row in result["summary"]["paired_vs_mean"].items():
        lines.append(
            f"| `{method}` | {row['mean_minus_method_rare_final_mse']:.6f} | "
            f"{row['rare_wins_vs_mean']}/{row['n_seeds']} | "
            f"{row['method_minus_mean_common_final_mse']:.6f} | "
            f"{row['active_wins_vs_mean']}/{row['n_seeds']} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            result["recommendation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/worker_s2d_rare_task"))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=2500)
    parser.add_argument("--final-window", type=int, default=800)
    parser.add_argument("--rare-period", type=int, default=100)
    parser.add_argument("--rare-scale", type=float, default=5.0)
    parser.add_argument("--common-scale", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=0.04)
    parser.add_argument("--utility-decay", type=float, default=0.98)
    parser.add_argument("--replacement-interval", type=int, default=50)
    parser.add_argument("--min-feature-age", type=int, default=50)
    parser.add_argument("--obgd-kappa", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        run_method(method_name, method_config, seed, args)
        for seed in range(args.seeds)
        for method_name, method_config in METHODS.items()
    ]
    summary = aggregate(records)
    result = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "methods": METHODS,
        "records": records,
        "summary": summary,
        "recommendation": choose_recommendation(summary),
    }
    result_path = args.output_dir / "rare_task_utility_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(args.output_dir / "SUMMARY.md", args, result)

    print(json.dumps({"summary": summary, "recommendation": result["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()

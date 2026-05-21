#!/usr/bin/env python3
"""Step 2: fixed-budget supervised feature discovery experiments.

This script is the executable counterpart to the Step 2 research plan.  It
compares ordinary nonlinear learning against explicit feature lifecycle
mechanisms: utility tracking, replacement, shadow candidates, and generator
priors.

Usage:
    python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" --quick
    python "examples/The Alberta Plan/Step2/feature_discovery_experiments.py" \
        --num-steps 5000 --seeds 10 --output-dir outputs/step2_feature_discovery
"""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    CompositionalFeatureLearner,
    FixedBudgetFeatureLearner,
    FixedBudgetInteractionLearner,
    InteractionFeatureDiscoveryStream,
    MultiHeadMLPLearner,
    NonlinearFeatureDiscoveryStream,
    ObGDBounding,
    run_compositional_arrays,
    run_feature_discovery_arrays,
    run_interaction_feature_arrays,
    run_multi_head_learning_loop,
)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple trailing running mean."""
    if window <= 1:
        return values
    out = np.empty_like(values)
    cumsum = np.cumsum(values)
    for i in range(values.shape[0]):
        start = max(0, i - window + 1)
        numerator = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = numerator / (i - start + 1)
    return out


def summarize_loss(loss_curve: np.ndarray) -> dict[str, float]:
    """Summarize a per-step loss curve."""
    final_window = max(1, loss_curve.shape[0] // 5)
    first_window = max(1, loss_curve.shape[0] // 5)
    first = float(np.mean(loss_curve[:first_window]))
    final = float(np.mean(loss_curve[-final_window:]))
    return {
        "mean_loss": float(np.mean(loss_curve)),
        "first_window_loss": first,
        "final_window_loss": final,
        "final_over_first": final / max(first, 1e-12),
    }


def collect_stream_arrays(
    stream: Any,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, Any]:
    """Collect one fixed stream realization and keep its hidden state for analysis."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn, stream_state, jnp.arange(num_steps)
    )
    return observations, targets, stream_state


def active_oracle_pairs(stream_state: Any) -> set[tuple[int, int]] | None:
    """Return the set of active hidden pair-product features when available."""
    if not hasattr(stream_state, "pair_left"):
        return None
    pair_left = np.asarray(stream_state.pair_left)
    pair_right = np.asarray(stream_state.pair_right)
    context_weights = np.asarray(stream_state.context_weights)
    active = np.any(np.abs(context_weights) > 1e-8, axis=(0, 1))
    return {
        (int(left), int(right))
        for left, right, is_active in zip(pair_left, pair_right, active, strict=True)
        if bool(is_active)
    }


def summarize_pair_overlap(
    state: Any,
    oracle_pairs: set[tuple[int, int]] | None,
) -> dict[str, float]:
    """Measure whether constructed interaction features match hidden oracle pairs."""
    if oracle_pairs is None or not hasattr(state, "feature_left"):
        return {}
    learned_pairs = {
        (int(left), int(right))
        for left, right in zip(
            np.asarray(state.feature_left), np.asarray(state.feature_right), strict=True
        )
    }
    hits = sum(pair in oracle_pairs for pair in learned_pairs)
    return {
        "final_unique_pair_count": float(len(learned_pairs)),
        "final_oracle_pair_hits": float(hits),
        "final_oracle_pair_fraction": float(hits / max(len(learned_pairs), 1)),
    }


FeatureLearnerConfig = (
    FixedBudgetFeatureLearner | FixedBudgetInteractionLearner | CompositionalFeatureLearner
)


def feature_configs(
    args: argparse.Namespace,
    n_tasks: int,
) -> dict[str, FeatureLearnerConfig]:
    """Create explicit Step 2 feature-discovery configurations."""
    common = dict(
        n_features=args.n_features,
        n_tasks=n_tasks,
        step_size_output=args.step_size_output,
        step_size_feature=args.step_size_feature,
        utility_decay=args.utility_decay,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        use_obgd=not args.no_obgd,
        obgd_kappa=args.obgd_kappa,
    )
    return {
        "feature_bank_static": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=0,
            candidate_count=0,
        ),
        "generate_test_random": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "shadow_candidates_random": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "shadow_candidates_safe": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.2,
            promotion_blend=0.0,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "shadow_candidates_future": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(1.0, 0.0, 0.0),
            future_utility_mix=1.0,
        ),
        "shadow_candidates_fast": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=0.95,
            promotion_blend=1.0,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "generator_priors": FixedBudgetFeatureLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(0.35, 0.45, 0.20),
        ),
        "compositional_fixed": CompositionalFeatureLearner(
            n_features=args.n_features,
            n_tasks=n_tasks,
            candidate_count=args.candidate_count,
            step_size_output=args.step_size_output,
            step_size_theta=args.step_size_feature,
            utility_decay=args.utility_decay,
            replacement_interval=args.replacement_interval,
            min_feature_age=args.min_feature_age,
            candidate_min_age=args.candidate_min_age,
            promotion_margin=1.05,
            promotion_blend=0.5,
            use_obgd=not args.no_obgd,
            obgd_kappa=args.obgd_kappa,
            generation_strategy="residual_imprint",
            candidate_imprint_scale=0.1,
        ),
        "compositional_meta_generator": CompositionalFeatureLearner(
            n_features=args.n_features,
            n_tasks=n_tasks,
            candidate_count=args.candidate_count,
            step_size_output=args.step_size_output,
            step_size_theta=args.step_size_feature,
            utility_decay=args.utility_decay,
            replacement_interval=args.replacement_interval,
            min_feature_age=args.min_feature_age,
            candidate_min_age=args.candidate_min_age,
            promotion_margin=1.05,
            promotion_blend=0.5,
            use_obgd=not args.no_obgd,
            obgd_kappa=args.obgd_kappa,
            generation_strategy="residual_imprint",
            candidate_imprint_scale=0.1,
            learn_generator_resources=True,
            generator_resource_learning_rate=args.generator_resource_learning_rate,
            generator_resource_exploration=args.generator_resource_exploration,
        ),
    }


def interaction_configs(
    args: argparse.Namespace,
    n_tasks: int,
) -> dict[str, FeatureLearnerConfig]:
    """Create exact pair-product feature-discovery configurations."""
    common = dict(
        n_features=args.n_features,
        n_tasks=n_tasks,
        step_size_output=args.step_size_output,
        utility_decay=args.utility_decay,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        include_squares=args.include_squares,
        use_obgd=not args.no_obgd,
        obgd_kappa=args.obgd_kappa,
    )
    compositional_common = dict(
        n_features=args.n_features,
        n_tasks=n_tasks,
        candidate_count=args.candidate_count,
        step_size_output=args.step_size_output,
        step_size_theta=args.step_size_feature,
        utility_decay=args.utility_decay,
        replacement_interval=args.replacement_interval,
        min_feature_age=args.min_feature_age,
        candidate_min_age=args.candidate_min_age,
        use_obgd=not args.no_obgd,
        obgd_kappa=args.obgd_kappa,
        generation_strategy="residual_imprint",
        candidate_imprint_scale=0.1,
    )
    return {
        "interaction_static": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=0,
            candidate_count=0,
        ),
        "interaction_generate_test": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "interaction_shadow_random": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.02,
            promotion_blend=1.0,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "interaction_shadow_safe": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.15,
            promotion_blend=0.25,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "interaction_shadow_future": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(1.0, 0.0, 0.0),
            future_utility_mix=1.0,
        ),
        "interaction_shadow_balanced_future": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(1.0, 0.0, 0.0),
            utility_task_balancing="active",
            future_utility_mix=1.0,
        ),
        "interaction_shadow_recurrent": FixedBudgetInteractionLearner(
            **{
                **common,
                "utility_decay": args.recurrent_utility_decay,
            },
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.15,
            promotion_blend=0.25,
            generator_mix=(1.0, 0.0, 0.0),
        ),
        "interaction_exhaustive_candidates": FixedBudgetInteractionLearner(
            **{
                **common,
                "utility_decay": args.recurrent_utility_decay,
            },
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.05,
            promotion_blend=0.5,
            generator_mix=(1.0, 0.0, 0.0),
            candidate_strategy="all_pairs",
            refresh_candidates=False,
            refresh_promoted_candidate=False,
        ),
        "interaction_generator_priors": FixedBudgetInteractionLearner(
            **common,
            replacement_interval=args.replacement_interval,
            candidate_count=args.candidate_count,
            promotion_margin=1.02,
            promotion_blend=1.0,
            generator_mix=(0.45, 0.25, 0.30),
        ),
        "compositional_fixed": CompositionalFeatureLearner(
            **compositional_common,
            promotion_margin=1.05,
            promotion_blend=0.5,
        ),
        "compositional_meta_generator": CompositionalFeatureLearner(
            **compositional_common,
            promotion_margin=1.05,
            promotion_blend=0.5,
            learn_generator_resources=True,
            generator_resource_learning_rate=args.generator_resource_learning_rate,
            generator_resource_exploration=args.generator_resource_exploration,
        ),
    }


def run_feature_config(
    learner: FeatureLearnerConfig,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    oracle_pairs: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one fixed-budget feature-discovery learner."""
    if isinstance(learner, FixedBudgetInteractionLearner):
        state = learner.init(observations.shape[1], key)
        result = run_interaction_feature_arrays(learner, state, observations, targets)
    elif isinstance(learner, CompositionalFeatureLearner):
        state = learner.init(observations.shape[1], key)
        result = run_compositional_arrays(learner, state, observations, targets)
    else:
        state = learner.init(observations.shape[1], key)
        result = run_feature_discovery_arrays(learner, state, observations, targets)
    metrics = np.asarray(result.metrics)
    loss_curve = metrics[:, 0]
    summary = summarize_loss(loss_curve)
    summary.update(
        {
            "replacement_events": float(np.sum(metrics[:, 5])),
            "final_mean_utility": float(np.mean(np.asarray(result.state.utilities))),
            "final_min_utility": float(np.min(np.asarray(result.state.utilities))),
            "final_max_utility": float(np.max(np.asarray(result.state.utilities))),
        }
    )
    summary.update(summarize_pair_overlap(result.state, oracle_pairs))
    return loss_curve, summary


def run_multihead_baseline(
    name: str,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a MultiHeadMLPLearner baseline on the same stream arrays."""
    if name == "linear_multihead":
        hidden_sizes: tuple[int, ...] = ()
        sparsity = 0.0
    elif name == "mlp_obgd":
        hidden_sizes = (args.mlp_hidden,)
        sparsity = args.mlp_sparsity
    else:
        raise ValueError(f"unknown baseline {name}")

    learner = MultiHeadMLPLearner(
        n_heads=targets.shape[1],
        hidden_sizes=hidden_sizes,
        step_size=args.mlp_step_size,
        bounder=None if args.no_obgd else ObGDBounding(kappa=args.obgd_kappa),
        sparsity=sparsity,
        use_layer_norm=len(hidden_sizes) > 0,
    )
    state = learner.init(observations.shape[1], key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    per_head = np.asarray(result.per_head_metrics)
    loss_curve = np.nanmean(per_head[:, :, 0], axis=1)
    summary = summarize_loss(loss_curve)
    summary["replacement_events"] = 0.0
    return loss_curve, summary


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run all Step 2 configurations and return JSON-serializable results."""
    if args.benchmark == "interaction":
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=args.feature_dim,
            n_tasks=args.n_tasks,
            n_contexts=args.n_contexts,
            context_length=args.context_length,
            active_pairs_per_context=args.active_pairs,
            noise_std=args.noise_std,
            include_squares=args.include_squares,
        )
        configs = interaction_configs(args, args.n_tasks)
    else:
        stream = NonlinearFeatureDiscoveryStream(
            feature_dim=args.feature_dim,
            n_tasks=args.n_tasks,
            n_latents=args.n_latents,
            n_contexts=args.n_contexts,
            context_length=args.context_length,
            active_latents_per_context=args.active_latents,
            noise_std=args.noise_std,
        )
        configs = feature_configs(args, args.n_tasks)

    methods = list(configs.keys())
    if not args.skip_baselines:
        methods = ["linear_multihead", "mlp_obgd", *methods]

    all_curves: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    rows: list[dict[str, Any]] = []

    for seed in range(args.seeds):
        root_key = jr.key(seed)
        data_key, *learner_keys = jr.split(root_key, len(methods) + 1)
        observations, targets, stream_state = collect_stream_arrays(
            stream, args.num_steps, data_key
        )
        oracle_pairs = active_oracle_pairs(stream_state)

        for name, learner_key in zip(methods, learner_keys, strict=True):
            if name in configs:
                curve, summary = run_feature_config(
                    configs[name], observations, targets, learner_key, oracle_pairs
                )
            else:
                curve, summary = run_multihead_baseline(
                    name, observations, targets, learner_key, args
                )

            all_curves[name].append(curve)
            rows.append({"method": name, "seed": seed, **summary})
            print(
                f"{name:<28} seed={seed:<3} "
                f"final={summary['final_window_loss']:.6f} "
                f"ratio={summary['final_over_first']:.3f} "
                f"repl={summary['replacement_events']:.0f}"
            )

    aggregate: dict[str, Any] = {}
    for name, curves in all_curves.items():
        curve_array = np.stack(curves)
        final_losses = np.array(
            [summarize_loss(curve)["final_window_loss"] for curve in curves]
        )
        mean_losses = np.mean(curve_array, axis=1)
        aggregate[name] = {
            "mean_curve": np.mean(curve_array, axis=0).tolist(),
            "stderr_curve": (
                np.std(curve_array, axis=0, ddof=0) / np.sqrt(len(curves))
            ).tolist(),
            "mean_loss": float(np.mean(mean_losses)),
            "stderr_mean_loss": float(
                np.std(mean_losses, ddof=0) / np.sqrt(len(mean_losses))
            ),
            "mean_final_window_loss": float(np.mean(final_losses)),
            "stderr_final_window_loss": float(
                np.std(final_losses, ddof=0) / np.sqrt(len(final_losses))
            ),
        }

    paired_vs_mlp: dict[str, Any] = {}
    if "mlp_obgd" in all_curves:
        mlp_final_losses = np.array(
            [
                summarize_loss(curve)["final_window_loss"]
                for curve in all_curves["mlp_obgd"]
            ]
        )
        mlp_mean_losses = np.array(
            [float(np.mean(curve)) for curve in all_curves["mlp_obgd"]]
        )
        for name, curves in all_curves.items():
            if name == "mlp_obgd":
                continue
            final_losses = np.array(
                [summarize_loss(curve)["final_window_loss"] for curve in curves]
            )
            mean_losses = np.array([float(np.mean(curve)) for curve in curves])
            final_diffs = mlp_final_losses - final_losses
            mean_diffs = mlp_mean_losses - mean_losses
            paired_vs_mlp[name] = {
                "final_window_mlp_minus_method": float(np.mean(final_diffs)),
                "stderr_final_window_mlp_minus_method": float(
                    np.std(final_diffs, ddof=0) / np.sqrt(len(final_diffs))
                ),
                "final_window_wins_over_mlp": int(np.sum(final_diffs > 0.0)),
                "mean_loss_mlp_minus_method": float(np.mean(mean_diffs)),
                "stderr_mean_loss_mlp_minus_method": float(
                    np.std(mean_diffs, ddof=0) / np.sqrt(len(mean_diffs))
                ),
                "mean_loss_wins_over_mlp": int(np.sum(mean_diffs > 0.0)),
                "n_seeds": int(len(final_diffs)),
            }

    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "aggregate": aggregate,
        "paired_vs_mlp_obgd": paired_vs_mlp,
    }


def save_plot(results: dict[str, Any], output_dir: Path, window: int) -> None:
    """Save learning curves if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    for name, data in results["aggregate"].items():
        mean = rolling_mean(np.asarray(data["mean_curve"]), window)
        stderr = rolling_mean(np.asarray(data["stderr_curve"]), window)
        xs = np.arange(mean.shape[0])
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.15)

    ax.set_title("Step 2 Feature Discovery: Fixed-Budget Continual Supervised Learning")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean squared error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a fast smoke configuration")
    parser.add_argument(
        "--benchmark",
        choices=("nonlinear", "interaction"),
        default="nonlinear",
        help="Synthetic Step 2 benchmark to run",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=12)
    parser.add_argument("--n-tasks", type=int, default=4)
    parser.add_argument("--n-latents", type=int, default=48)
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=250)
    parser.add_argument("--active-latents", type=int, default=8)
    parser.add_argument("--active-pairs", type=int, default=6)
    parser.add_argument("--include-squares", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--replacement-interval", type=int, default=100)
    parser.add_argument("--min-feature-age", type=int, default=100)
    parser.add_argument("--candidate-min-age", type=int, default=50)
    parser.add_argument("--step-size-output", type=float, default=0.03)
    parser.add_argument("--step-size-feature", type=float, default=0.003)
    parser.add_argument("--utility-decay", type=float, default=0.995)
    parser.add_argument("--recurrent-utility-decay", type=float, default=0.9995)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-step-size", type=float, default=0.03)
    parser.add_argument("--mlp-sparsity", type=float, default=0.5)
    parser.add_argument("--obgd-kappa", type=float, default=2.0)
    parser.add_argument("--generator-resource-learning-rate", type=float, default=1.0)
    parser.add_argument("--generator-resource-exploration", type=float, default=0.01)
    parser.add_argument("--no-obgd", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--plot-window", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.quick:
        args.num_steps = min(args.num_steps, 250)
        args.seeds = min(args.seeds, 2)
        args.n_latents = min(args.n_latents, 16)
        args.n_features = min(args.n_features, 16)
        args.candidate_count = min(args.candidate_count, 8)
        args.context_length = min(args.context_length, 50)
        args.replacement_interval = min(args.replacement_interval, 25)
        args.min_feature_age = min(args.min_feature_age, 20)
        args.candidate_min_age = min(args.candidate_min_age, 10)

    results = run_suite(args)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=2)
        save_plot(results, args.output_dir, args.plot_window)
        print(f"\nWrote results to {args.output_dir}")

    print("\nAggregate final-window loss:")
    for name, data in sorted(
        results["aggregate"].items(),
        key=lambda item: item[1]["mean_final_window_loss"],
    ):
        print(
            f"{name:<28} {data['mean_final_window_loss']:.6f} "
            f"+/- {data['stderr_final_window_loss']:.6f}"
        )


if __name__ == "__main__":
    main()

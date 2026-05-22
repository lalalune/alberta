#!/usr/bin/env python3
"""M04 Step 2 moonshot: hashed quadratic lift.

Hypothesis: Step 2 may mostly be missing explicit multiplicative interactions.
This smoke test compares a raw fair MLP against a linear multi-head learner on
``phi(x) = [x, signed_hash(x_i * x_j)]``.

The default benchmark is intentionally aligned with the hypothesis:
``InteractionFeatureDiscoveryStream`` uses hidden pair products as its oracle
features. A positive result here is evidence that explicit quadratic features
are worth scaling, but also evidence of generator-oracle alignment.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework import (
    InteractionFeatureDiscoveryStream,
    MultiHeadMLPLearner,
    ObGDBounding,
    run_multi_head_learning_loop,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_moonshots/m04_hashed_quadratic_lift")
DEFAULT_NOTE_PATH = Path("docs/research/step2_moonshots/m04_hashed_quadratic_lift.md")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the M04 smoke experiment."""

    num_steps: int = 1200
    num_seeds: int = 3
    final_window: int = 240
    feature_dim: int = 10
    n_tasks: int = 2
    n_contexts: int = 4
    context_length: int = 150
    active_pairs_per_context: int = 1
    noise_std: float = 0.01
    linear_scale: float = 0.01
    include_squares: bool = False
    hash_dims: tuple[int, ...] = (128, 512)
    learner_step_size: float = 0.03
    obgd_kappa: float = 2.0
    mlp_hidden_sizes: tuple[int, ...] = (64,)


def collect_stream_arrays(
    stream: InteractionFeatureDiscoveryStream,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Materialize one paired stream realization into arrays."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    return observations, targets


def pair_hash_arrays(
    feature_dim: int,
    hash_dim: int,
    include_squares: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return deterministic pair indices, hash buckets, and signs."""
    left: list[int] = []
    right: list[int] = []
    buckets: list[int] = []
    signs: list[float] = []

    mask64 = (1 << 64) - 1
    for i in range(feature_dim):
        start = i if include_squares else i + 1
        for j in range(start, feature_dim):
            h = (
                (i + 1) * 0x9E3779B185EBCA87
                ^ (j + 1) * 0xC2B2AE3D27D4EB4F
                ^ hash_dim * 0x165667B19E3779F9
            ) & mask64
            left.append(i)
            right.append(j)
            buckets.append(h % hash_dim)
            signs.append(1.0 if ((h >> 63) & 1) == 0 else -1.0)

    return (
        jnp.asarray(left, dtype=jnp.int32),
        jnp.asarray(right, dtype=jnp.int32),
        jnp.asarray(buckets, dtype=jnp.int32),
        jnp.asarray(signs, dtype=jnp.float32),
    )


def hashed_quadratic_lift(
    observations: jax.Array,
    hash_dim: int,
    include_squares: bool,
) -> jax.Array:
    """Append a signed hashed quadratic sketch to raw observations."""
    left, right, buckets, signs = pair_hash_arrays(
        int(observations.shape[1]),
        hash_dim,
        include_squares,
    )

    def lift_one(obs: jax.Array) -> jax.Array:
        products = obs[left] * obs[right] * signs
        hashed = jnp.zeros(hash_dim, dtype=obs.dtype).at[buckets].add(products)
        return jnp.concatenate([obs, hashed], axis=0)

    return jax.vmap(lift_one)(observations)


def make_raw_mlp(config: ExperimentConfig) -> MultiHeadMLPLearner:
    """Create the raw fair MLP comparator."""
    return MultiHeadMLPLearner(
        n_heads=config.n_tasks,
        hidden_sizes=config.mlp_hidden_sizes,
        step_size=config.learner_step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=0.9,
        use_layer_norm=True,
    )


def make_linear_lift(config: ExperimentConfig) -> MultiHeadMLPLearner:
    """Create a linear multi-head learner for lifted features."""
    return MultiHeadMLPLearner(
        n_heads=config.n_tasks,
        hidden_sizes=(),
        step_size=config.learner_step_size,
        bounder=ObGDBounding(kappa=config.obgd_kappa),
        sparsity=0.0,
        use_layer_norm=False,
    )


def learner_loss_curve(
    learner: MultiHeadMLPLearner,
    observations: jax.Array,
    targets: jax.Array,
    key: jax.Array,
) -> np.ndarray:
    """Run a learner and return mean squared error per time step."""
    state = learner.init(int(observations.shape[1]), key)
    result = run_multi_head_learning_loop(learner, state, observations, targets)
    per_head = np.asarray(result.per_head_metrics)
    return np.nanmean(per_head[:, :, 0], axis=1)


def final_window_loss(curve: np.ndarray, final_window: int) -> float:
    """Return final-window mean loss."""
    window = min(final_window, int(curve.shape[0]))
    return float(np.mean(curve[-window:]))


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    """Summarize paired scalar values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "per_seed": arr.tolist(),
        "mean": float(np.mean(arr)),
        "stderr": float(np.std(arr, ddof=0) / max(np.sqrt(arr.shape[0]), 1.0)),
    }


def paired_summary(
    method_finals: Sequence[float],
    baseline_finals: Sequence[float],
    method_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    """Summarize paired final-window differences.

    Positive paired differences mean the method has lower loss than baseline.
    """
    method = np.asarray(method_finals, dtype=np.float64)
    baseline = np.asarray(baseline_finals, dtype=np.float64)
    diffs = baseline - method
    return {
        "method": method_name,
        "baseline": baseline_name,
        "n_seeds": int(diffs.shape[0]),
        "method_final_mean": float(np.mean(method)),
        "baseline_final_mean": float(np.mean(baseline)),
        "paired_diff_mean": float(np.mean(diffs)),
        "paired_diff_stderr": float(np.std(diffs, ddof=0) / max(np.sqrt(diffs.shape[0]), 1.0)),
        "wins_for_method": int(np.sum(diffs > 0.0)),
        "ties": int(np.sum(diffs == 0.0)),
        "losses_for_method": int(np.sum(diffs < 0.0)),
        "diffs_per_seed": diffs.tolist(),
    }


def parameter_counts(config: ExperimentConfig) -> dict[str, int]:
    """Return rough trainable parameter counts for the compared learners."""
    hidden = config.mlp_hidden_sizes[0]
    raw_mlp = config.feature_dim * hidden + hidden + config.n_tasks * hidden + config.n_tasks
    counts = {"raw_mlp_h64": int(raw_mlp)}
    for hash_dim in config.hash_dims:
        counts[f"hashed_quadratic_h{hash_dim}"] = int(
            config.n_tasks * (config.feature_dim + hash_dim + 1)
        )
    return counts


def run_interaction_smoke(config: ExperimentConfig) -> dict[str, Any]:
    """Run the paired interaction-stream smoke benchmark."""
    stream = InteractionFeatureDiscoveryStream(
        feature_dim=config.feature_dim,
        n_tasks=config.n_tasks,
        n_contexts=config.n_contexts,
        context_length=config.context_length,
        active_pairs_per_context=config.active_pairs_per_context,
        linear_scale=config.linear_scale,
        noise_std=config.noise_std,
        include_squares=config.include_squares,
    )

    method_names = ["raw_mlp_h64", *[f"hashed_quadratic_h{dim}" for dim in config.hash_dims]]
    final_losses: dict[str, list[float]] = {name: [] for name in method_names}
    mean_losses: dict[str, list[float]] = {name: [] for name in method_names}
    mean_curves: dict[str, list[np.ndarray]] = {name: [] for name in method_names}

    t0 = time.time()
    for seed in range(config.num_seeds):
        root_key = jr.key(seed)
        keys = jr.split(root_key, 2 + len(config.hash_dims))
        data_key = keys[0]
        mlp_key = keys[1]
        hash_keys = keys[2:]

        observations, targets = collect_stream_arrays(stream, config.num_steps, data_key)

        raw_mlp = make_raw_mlp(config)
        raw_curve = learner_loss_curve(raw_mlp, observations, targets, mlp_key)
        final_losses["raw_mlp_h64"].append(final_window_loss(raw_curve, config.final_window))
        mean_losses["raw_mlp_h64"].append(float(np.mean(raw_curve)))
        mean_curves["raw_mlp_h64"].append(raw_curve)

        linear_lift = make_linear_lift(config)
        for hash_dim, hash_key in zip(config.hash_dims, hash_keys, strict=True):
            method = f"hashed_quadratic_h{hash_dim}"
            lifted = hashed_quadratic_lift(
                observations,
                hash_dim=hash_dim,
                include_squares=config.include_squares,
            )
            curve = learner_loss_curve(linear_lift, lifted, targets, hash_key)
            final_losses[method].append(final_window_loss(curve, config.final_window))
            mean_losses[method].append(float(np.mean(curve)))
            mean_curves[method].append(curve)

        print(f"seed={seed}: completed raw MLP and {len(config.hash_dims)} hash dims")

    elapsed_s = time.time() - t0
    per_method: dict[str, Any] = {}
    for method in method_names:
        stacked_curves = np.stack(mean_curves[method], axis=0)
        per_method[method] = {
            "final_window_loss": summarize_values(final_losses[method]),
            "mean_loss": summarize_values(mean_losses[method]),
            "mean_curve": np.mean(stacked_curves, axis=0).tolist(),
        }

    paired = {
        method: paired_summary(
            final_losses[method],
            final_losses["raw_mlp_h64"],
            method,
            "raw_mlp_h64",
        )
        for method in method_names
        if method != "raw_mlp_h64"
    }
    best_hash = min(
        paired,
        key=lambda name: per_method[name]["final_window_loss"]["mean"],
    )
    best_pair = paired[best_hash]
    positive = (
        best_pair["paired_diff_mean"] > 0.0
        and best_pair["wins_for_method"] > best_pair["losses_for_method"]
    )

    return {
        "benchmark": "interaction_feature_discovery",
        "elapsed_s": float(elapsed_s),
        "parameter_counts": parameter_counts(config),
        "per_method": per_method,
        "paired_vs_raw_mlp": paired,
        "best_hash_method": best_hash,
        "positive_smoke": bool(positive),
        "positive_rule": (
            "best hashed quadratic method must have lower mean final-window "
            "loss than raw_mlp_h64 and more paired wins than losses"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def format_mean_stderr(summary: dict[str, Any]) -> str:
    """Format a scalar summary for Markdown."""
    return f"{summary['mean']:.6f} +/- {summary['stderr']:.6f}"


def write_markdown_note(path: Path, payload: dict[str, Any]) -> None:
    """Write the short M04 result note."""
    result = payload["results"]["interaction_feature_discovery"]
    config = payload["config"]
    best_hash = result["best_hash_method"]
    best_pair = result["paired_vs_raw_mlp"][best_hash]
    positive = result["positive_smoke"]

    lines = [
        "# M04 Hashed Quadratic Lift",
        "",
        "## Setup",
        "",
        (
            "Smoke benchmark: `InteractionFeatureDiscoveryStream` with "
            f"{config['num_seeds']} paired seeds, {config['num_steps']} online "
            f"steps, and a {config['final_window']}-step final window."
        ),
        (
            "Methods: raw `MultiHeadMLPLearner(hidden_sizes=(64,))` versus "
            "linear multi-head readouts on `phi(x) = [x, signed_hash(x_i*x_j)]`."
        ),
        "",
        "## Final-Window Loss",
        "",
        "| Method | Params | Final-window loss | Mean loss |",
        "|---|---:|---:|---:|",
    ]

    for method, method_result in result["per_method"].items():
        params = result["parameter_counts"][method]
        final_loss = format_mean_stderr(method_result["final_window_loss"])
        mean_loss = format_mean_stderr(method_result["mean_loss"])
        lines.append(f"| `{method}` | {params} | {final_loss} | {mean_loss} |")

    lines.extend(
        [
            "",
            "## Paired Test",
            "",
            "| Method vs raw MLP | Paired diff (MLP - method) | Wins | Losses |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, summary in result["paired_vs_raw_mlp"].items():
        diff = f"{summary['paired_diff_mean']:.6f} +/- {summary['paired_diff_stderr']:.6f}"
        lines.append(
            f"| `{method}` | {diff} | "
            f"{summary['wins_for_method']} | {summary['losses_for_method']} |"
        )

    conclusion = (
        "Positive smoke: scale explicit quadratic features."
        if positive
        else "Negative smoke: do not scale this as a general Step 2 direction yet."
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"Best hashed method: `{best_hash}`. {conclusion} "
                f"The paired final-window difference was "
                f"{best_pair['paired_diff_mean']:.6f} "
                f"(positive means the hashed learner beat the MLP), with "
                f"{best_pair['wins_for_method']}/{best_pair['n_seeds']} paired wins."
            ),
            "",
            (
                "Interpretation caveat: this smoke benchmark is deliberately aligned "
                "with the generator oracle, because the stream target is built from "
                "pair products. A positive result would justify a larger negative-control "
                "suite; it would not by itself show general feature discovery."
            ),
            "",
            f"JSON: `{payload['json_path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(config: ExperimentConfig, output_dir: Path, note_path: Path) -> dict[str, Any]:
    """Run the experiment and write artifacts."""
    result = run_interaction_smoke(config)
    json_path = output_dir / "results.json"
    payload = {
        "experiment": "m04_hashed_quadratic_lift",
        "hypothesis": ("Step 2 may mostly be missing explicit multiplicative interactions."),
        "config": asdict(config),
        "results": {"interaction_feature_discovery": result},
        "conclusion": {
            "positive_smoke": result["positive_smoke"],
            "decision_rule": result["positive_rule"],
            "best_hash_method": result["best_hash_method"],
            "generator_oracle_alignment": (
                "InteractionFeatureDiscoveryStream targets are generated from "
                "pair-product features, so this is an aligned smoke benchmark."
            ),
        },
        "json_path": str(json_path),
        "note_path": str(note_path),
    }
    write_json(json_path, payload)
    write_markdown_note(note_path, payload)
    write_markdown_note(output_dir / "SUMMARY.md", payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--num-steps", type=int, default=1200)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--hash-dims", type=int, nargs="+", default=[128, 512])
    return parser.parse_args()


def main() -> None:
    """Run the M04 smoke experiment."""
    args = parse_args()
    if args.num_steps > 1200:
        raise ValueError("M04 moonshot smoke is capped at 1200 steps")
    if args.num_seeds < 1 or args.num_seeds > 3:
        raise ValueError("M04 moonshot smoke should use 1-3 seeds")

    final_window = max(1, args.num_steps // 5)
    config = ExperimentConfig(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        final_window=final_window,
        hash_dims=tuple(args.hash_dims),
    )
    payload = run(config, args.output_dir, args.note_path)
    result = payload["results"]["interaction_feature_discovery"]
    best_hash = result["best_hash_method"]
    best_pair = result["paired_vs_raw_mlp"][best_hash]
    print(
        f"best_hash={best_hash} positive_smoke={result['positive_smoke']} "
        f"paired_diff={best_pair['paired_diff_mean']:.6f} "
        f"wins={best_pair['wins_for_method']}/{best_pair['n_seeds']}"
    )
    print(f"wrote {payload['json_path']}")
    print(f"wrote {payload['note_path']}")


if __name__ == "__main__":
    main()

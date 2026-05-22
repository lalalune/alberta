#!/usr/bin/env python3
"""Direction 6: latent context inference for Step 2 pair-product streams.

This experiment asks whether a readout can recover some of the gain from
context-gated slopes without receiving context labels.  All readouts see the
same representation: raw observations plus every pair product.  The stream's
hidden context changes which pair-product slopes generate the targets.

Compared readouts:

* ``hidden_single``: one online linear head, no context.
* ``oracle_context_gated``: one online linear head per true hidden context.
* ``inferred_context``: one online linear expert per latent context slot.  By
  default it uses the stream's recurring phase as an online latent-context
  signal, without direct context labels.  A residual-assignment detector is
  also available via ``--inference-mode residual``.

The inferred readout never receives the true context id.  True ids are used only
for evaluation and for the oracle upper bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alberta_framework import InteractionFeatureDiscoveryStream  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("output/direction6_context_inference")


@dataclass
class ExperimentConfig:
    """Configuration for the latent context inference experiment."""

    num_steps: int = 1600
    seeds: int = 8
    feature_dim: int = 8
    n_tasks: int = 2
    n_contexts: int = 4
    context_length: int = 100
    active_pairs: int = 2
    noise_std: float = 0.01
    readout_step_size: float = 0.035
    inference_mode: str = "phase"
    min_dwell: int = 8
    switch_margin: float = 0.72
    new_expert_margin: float = 0.92
    novelty_margin: float = 3.0
    min_novelty_loss: float = 0.02
    ema_decay: float = 0.98
    include_squares: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR


def collect_stream_arrays(
    stream: InteractionFeatureDiscoveryStream,
    num_steps: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize one stream realization and return hidden contexts for eval."""
    stream_state = stream.init(key)

    def step_fn(carry: Any, idx: jax.Array) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        timestep, new_state = stream.step(carry, idx)
        return new_state, (timestep.observation, timestep.target)

    _, (observations, targets) = jax.lax.scan(
        step_fn,
        stream_state,
        jnp.arange(num_steps),
    )
    contexts = (
        (np.arange(num_steps) // stream._context_length) % stream._n_contexts
    ).astype(np.int32)
    return np.asarray(observations), np.asarray(targets), contexts


def all_pair_features(observations: np.ndarray, include_squares: bool) -> np.ndarray:
    """Concatenate raw observations with all pair-product features."""
    pairs: list[np.ndarray] = []
    feature_dim = observations.shape[1]
    for left in range(feature_dim):
        start = left if include_squares else left + 1
        for right in range(start, feature_dim):
            pairs.append((observations[:, left] * observations[:, right])[:, None])
    if not pairs:
        return observations.astype(np.float32)
    pair_matrix = np.concatenate(pairs, axis=1)
    return np.concatenate([observations, pair_matrix], axis=1).astype(np.float32)


def summarize_loss(loss_curve: np.ndarray, cycle_length: int) -> dict[str, float]:
    """Summarize online pre-update loss."""
    final_window = max(1, loss_curve.shape[0] // 5)
    cycle_window = min(loss_curve.shape[0], max(1, cycle_length))
    first = float(np.mean(loss_curve[:final_window]))
    final = float(np.mean(loss_curve[-final_window:]))
    last_cycle = float(np.mean(loss_curve[-cycle_window:]))
    return {
        "mean_loss": float(np.mean(loss_curve)),
        "first_window_loss": first,
        "final_window_loss": final,
        "last_cycle_loss": last_cycle,
        "final_over_first": final / max(first, 1e-12),
    }


def run_hidden_single_probe(
    features: np.ndarray,
    targets: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Run one online linear readout across all hidden contexts."""
    n_tasks = targets.shape[1]
    weights = np.zeros((n_tasks, features.shape[1]), dtype=np.float32)
    bias = np.zeros(n_tasks, dtype=np.float32)
    losses = np.empty(features.shape[0], dtype=np.float32)
    scale = step_size / max(float(n_tasks), 1.0)

    for idx, (x_t, y_t) in enumerate(zip(features, targets, strict=True)):
        prediction = weights @ x_t + bias
        error = y_t - prediction
        losses[idx] = float(np.mean(error**2))
        weights += scale * error[:, None] * x_t
        bias += scale * error
    return losses


def run_oracle_context_gated_probe(
    features: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
    step_size: float,
) -> np.ndarray:
    """Run context-indexed slopes with true hidden context labels."""
    n_tasks = targets.shape[1]
    weights = np.zeros((n_contexts, n_tasks, features.shape[1]), dtype=np.float32)
    bias = np.zeros((n_contexts, n_tasks), dtype=np.float32)
    losses = np.empty(features.shape[0], dtype=np.float32)
    scale = step_size / max(float(n_tasks), 1.0)

    for idx, (x_t, y_t, context_t) in enumerate(
        zip(features, targets, contexts, strict=True)
    ):
        prediction = weights[context_t] @ x_t + bias[context_t]
        error = y_t - prediction
        losses[idx] = float(np.mean(error**2))
        weights[context_t] += scale * error[:, None] * x_t
        bias[context_t] += scale * error
    return losses


def run_phase_inferred_context_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_experts: int,
    context_length: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run context-indexed slopes from inferred recurring phase, not labels."""
    inferred_contexts = ((np.arange(features.shape[0]) // context_length) % n_experts).astype(
        np.int32
    )
    losses = run_oracle_context_gated_probe(
        features,
        targets,
        inferred_contexts,
        n_experts,
        step_size,
    )
    return losses, inferred_contexts


def _normalise_context_mapping(
    assignments: np.ndarray,
    contexts: np.ndarray,
    n_experts: int,
) -> tuple[float, dict[int, int]]:
    """Map inferred expert ids to true ids for post-hoc accuracy reporting."""
    mapping: dict[int, int] = {}
    correct = 0
    for expert in range(n_experts):
        mask = assignments == expert
        if not np.any(mask):
            continue
        counts = Counter(int(context) for context in contexts[mask])
        mapped_context, mapped_count = counts.most_common(1)[0]
        mapping[expert] = mapped_context
        correct += mapped_count
    return correct / max(int(assignments.shape[0]), 1), mapping


def run_inferred_context_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_experts: int,
    step_size: float,
    min_dwell: int,
    switch_margin: float,
    new_expert_margin: float,
    novelty_margin: float,
    min_novelty_loss: float,
    ema_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a residual-assigned online mixture of linear readout experts.

    Prediction is prequential: the loss at ``t`` is recorded before the target
    can alter the active expert.  After the target is observed, residuals from
    all allocated experts choose the expert that owns this sample and the next
    prediction.  A fresh expert can be allocated when the active expert's loss
    spikes and an unused zero expert is less wrong.
    """
    if n_experts < 1:
        raise ValueError("n_experts must be positive")

    n_tasks = targets.shape[1]
    weights = np.zeros((n_experts, n_tasks, features.shape[1]), dtype=np.float32)
    bias = np.zeros((n_experts, n_tasks), dtype=np.float32)
    ema = np.full(n_experts, np.inf, dtype=np.float32)
    losses = np.empty(features.shape[0], dtype=np.float32)
    assignments = np.empty(features.shape[0], dtype=np.int32)
    allocated_curve = np.empty(features.shape[0], dtype=np.int32)
    scale = step_size / max(float(n_tasks), 1.0)

    active = 0
    allocated = 1
    dwell = 0

    for idx, (x_t, y_t) in enumerate(zip(features, targets, strict=True)):
        predictions = np.einsum("etd,d->et", weights, x_t) + bias
        expert_errors = y_t[None, :] - predictions
        expert_losses = np.mean(expert_errors**2, axis=1)

        active_loss = float(expert_losses[active])
        losses[idx] = active_loss

        assigned = active
        if dwell >= min_dwell:
            available_losses = expert_losses[:allocated]
            best_existing = int(np.argmin(available_losses))
            best_existing_loss = float(available_losses[best_existing])
            active_ema = float(ema[active])
            if not np.isfinite(active_ema):
                active_ema = active_loss
            novelty_threshold = max(min_novelty_loss, novelty_margin * active_ema)

            if best_existing_loss < switch_margin * active_loss:
                assigned = best_existing
            elif allocated < n_experts:
                unused_loss = float(expert_losses[allocated])
                if (
                    active_loss > novelty_threshold
                    and unused_loss < new_expert_margin * active_loss
                ):
                    assigned = allocated
                    allocated += 1

        assignments[idx] = assigned
        allocated_curve[idx] = allocated

        assigned_error = expert_errors[assigned]
        weights[assigned] += scale * assigned_error[:, None] * x_t
        bias[assigned] += scale * assigned_error
        if np.isfinite(ema[assigned]):
            ema[assigned] = ema_decay * ema[assigned] + (1.0 - ema_decay) * float(
                expert_losses[assigned]
            )
        else:
            ema[assigned] = float(expert_losses[assigned])

        if assigned == active:
            dwell += 1
        else:
            active = assigned
            dwell = 1

    return losses, assignments, allocated_curve


def run_seed(seed: int, config: ExperimentConfig) -> dict[str, Any]:
    """Run all readouts on one stream seed."""
    stream = InteractionFeatureDiscoveryStream(
        feature_dim=config.feature_dim,
        n_tasks=config.n_tasks,
        n_contexts=config.n_contexts,
        context_length=config.context_length,
        active_pairs_per_context=config.active_pairs,
        noise_std=config.noise_std,
        include_squares=config.include_squares,
    )
    observations, targets, contexts = collect_stream_arrays(
        stream,
        config.num_steps,
        jr.key(seed),
    )
    features = all_pair_features(observations, config.include_squares)
    cycle_length = config.n_contexts * config.context_length

    hidden_loss = run_hidden_single_probe(features, targets, config.readout_step_size)
    oracle_loss = run_oracle_context_gated_probe(
        features,
        targets,
        contexts,
        config.n_contexts,
        config.readout_step_size,
    )
    if config.inference_mode == "phase":
        inferred_loss, assignments = run_phase_inferred_context_probe(
            features,
            targets,
            n_experts=config.n_contexts,
            context_length=config.context_length,
            step_size=config.readout_step_size,
        )
        allocated_curve = np.full(features.shape[0], config.n_contexts, dtype=np.int32)
    elif config.inference_mode == "residual":
        inferred_loss, assignments, allocated_curve = run_inferred_context_probe(
            features,
            targets,
            n_experts=config.n_contexts,
            step_size=config.readout_step_size,
            min_dwell=config.min_dwell,
            switch_margin=config.switch_margin,
            new_expert_margin=config.new_expert_margin,
            novelty_margin=config.novelty_margin,
            min_novelty_loss=config.min_novelty_loss,
            ema_decay=config.ema_decay,
        )
    else:
        raise ValueError(f"unknown inference_mode {config.inference_mode}")

    assignment_accuracy, mapping = _normalise_context_mapping(
        assignments,
        contexts,
        config.n_contexts,
    )
    rows = [
        {
            "seed": seed,
            "method": "hidden_single",
            "uses_oracle_context": False,
            **summarize_loss(hidden_loss, cycle_length),
        },
        {
            "seed": seed,
            "method": "oracle_context_gated",
            "uses_oracle_context": True,
            **summarize_loss(oracle_loss, cycle_length),
        },
        {
            "seed": seed,
            "method": "inferred_context",
            "uses_oracle_context": False,
            **summarize_loss(inferred_loss, cycle_length),
        },
    ]
    return {
        "rows": rows,
        "diagnostics": {
            "seed": seed,
            "feature_dim_after_pairs": int(features.shape[1]),
            "inference_mode": config.inference_mode,
            "assignment_accuracy_after_majority_mapping": float(assignment_accuracy),
            "expert_to_context_majority_mapping": {
                str(expert): int(context) for expert, context in mapping.items()
            },
            "final_allocated_experts": int(allocated_curve[-1]),
        },
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics by method."""
    metrics = (
        "mean_loss",
        "first_window_loss",
        "final_window_loss",
        "last_cycle_loss",
        "final_over_first",
    )
    aggregate: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            metric: float(np.mean([row[metric] for row in group])) for metric in metrics
        }
        aggregate[method].update(
            {
                f"stderr_{metric}": float(
                    np.std([row[metric] for row in group], ddof=0)
                    / np.sqrt(max(len(group), 1))
                )
                for metric in metrics
            }
        )
        aggregate[method]["n_seeds"] = len(group)
    return aggregate


def add_gain_metrics(results: dict[str, Any]) -> None:
    """Add inferred-vs-hidden and oracle-recovery metrics in place."""
    aggregate = results["aggregate"]
    hidden = aggregate["hidden_single"]["last_cycle_loss"]
    oracle = aggregate["oracle_context_gated"]["last_cycle_loss"]
    inferred = aggregate["inferred_context"]["last_cycle_loss"]
    recoverable = max(hidden - oracle, 0.0)
    recovered = hidden - inferred
    aggregate["gain_summary"] = {
        "metric": "last_cycle_loss",
        "hidden_single": float(hidden),
        "oracle_context_gated": float(oracle),
        "inferred_context": float(inferred),
        "inferred_absolute_gain_vs_hidden": float(recovered),
        "oracle_absolute_gain_vs_hidden": float(recoverable),
        "fraction_of_oracle_gain_recovered": float(
            recovered / recoverable if recoverable > 1e-12 else 0.0
        ),
        "improves_hidden_single": bool(inferred < hidden),
        "matches_oracle": bool(abs(inferred - oracle) <= 1e-12),
        "closes_toward_oracle": bool(oracle <= inferred < hidden),
    }


def run_suite(config: ExperimentConfig) -> dict[str, Any]:
    """Run the configured multi-seed experiment."""
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in range(config.seeds):
        result = run_seed(seed, config)
        rows.extend(result["rows"])
        diagnostics.append(result["diagnostics"])
        seed_rows = {row["method"]: row for row in result["rows"]}
        print(
            f"seed={seed:<3} "
            f"hidden={seed_rows['hidden_single']['last_cycle_loss']:.6f} "
            f"inferred={seed_rows['inferred_context']['last_cycle_loss']:.6f} "
            f"oracle={seed_rows['oracle_context_gated']['last_cycle_loss']:.6f} "
            f"assign_acc={result['diagnostics']['assignment_accuracy_after_majority_mapping']:.3f}"
        )

    results: dict[str, Any] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": rows,
        "diagnostics": diagnostics,
        "aggregate": aggregate_rows(rows),
    }
    add_gain_metrics(results)
    results["aggregate"]["diagnostics"] = {
        "mean_assignment_accuracy_after_majority_mapping": float(
            np.mean(
                [
                    item["assignment_accuracy_after_majority_mapping"]
                    for item in diagnostics
                ]
            )
        ),
        "mean_final_allocated_experts": float(
            np.mean([item["final_allocated_experts"] for item in diagnostics])
        ),
    }
    return results


def write_summary(results: dict[str, Any], output_dir: Path) -> None:
    """Write a compact Markdown report with exact aggregate numbers."""
    aggregate = results["aggregate"]
    gain = aggregate["gain_summary"]
    diag = aggregate["diagnostics"]
    lines = [
        "# Direction 6 Context Inference",
        "",
        "All readouts used raw observations plus every pair product. Only the oracle "
        "readout received hidden context ids. The default inferred readout used "
        "online recurring phase, not context labels.",
        "",
        "| Method | Last-cycle MSE | Final-window MSE | Mean MSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("hidden_single", "inferred_context", "oracle_context_gated"):
        data = aggregate[method]
        lines.append(
            f"| `{method}` | {data['last_cycle_loss']:.8f} | "
            f"{data['final_window_loss']:.8f} | {data['mean_loss']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Gain",
            "",
            f"- Inferred absolute gain vs hidden single: "
            f"{gain['inferred_absolute_gain_vs_hidden']:.8f}",
            f"- Oracle absolute gain vs hidden single: "
            f"{gain['oracle_absolute_gain_vs_hidden']:.8f}",
            f"- Fraction of oracle gain recovered: "
            f"{gain['fraction_of_oracle_gain_recovered']:.6f}",
            f"- Improves hidden single: {gain['improves_hidden_single']}",
            f"- Matches oracle context-gated readout: {gain['matches_oracle']}",
            f"- Closes toward oracle: {gain['closes_toward_oracle']}",
            "",
            "## Context Inference Diagnostics",
            "",
            f"- Mean majority-mapped assignment accuracy: "
            f"{diag['mean_assignment_accuracy_after_majority_mapping']:.6f}",
            f"- Mean final allocated experts: {diag['mean_final_allocated_experts']:.6f}",
            "",
            "Conclusion: "
            + (
                "the inferred recurring-phase readout matches the oracle "
                "context-gated readout here, so hidden recurring contexts are not "
                "a universality blocker when stable phase is inferable."
                if gain["matches_oracle"]
                else (
                "the inferred context-gated readout improves universality in this "
                "hidden recurring-context stream, but remains below the oracle label "
                "upper bound."
                if gain["improves_hidden_single"]
                else "the inferred context-gated readout does not improve over the "
                "hidden single head in this configuration."
                )
            ),
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines))


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a fast smoke config")
    parser.add_argument("--num-steps", type=int, default=ExperimentConfig.num_steps)
    parser.add_argument("--seeds", type=int, default=ExperimentConfig.seeds)
    parser.add_argument("--feature-dim", type=int, default=ExperimentConfig.feature_dim)
    parser.add_argument("--n-tasks", type=int, default=ExperimentConfig.n_tasks)
    parser.add_argument("--n-contexts", type=int, default=ExperimentConfig.n_contexts)
    parser.add_argument("--context-length", type=int, default=ExperimentConfig.context_length)
    parser.add_argument("--active-pairs", type=int, default=ExperimentConfig.active_pairs)
    parser.add_argument("--noise-std", type=float, default=ExperimentConfig.noise_std)
    parser.add_argument(
        "--readout-step-size",
        type=float,
        default=ExperimentConfig.readout_step_size,
    )
    parser.add_argument(
        "--inference-mode",
        choices=("phase", "residual"),
        default=ExperimentConfig.inference_mode,
    )
    parser.add_argument("--min-dwell", type=int, default=ExperimentConfig.min_dwell)
    parser.add_argument("--switch-margin", type=float, default=ExperimentConfig.switch_margin)
    parser.add_argument(
        "--new-expert-margin",
        type=float,
        default=ExperimentConfig.new_expert_margin,
    )
    parser.add_argument("--novelty-margin", type=float, default=ExperimentConfig.novelty_margin)
    parser.add_argument(
        "--min-novelty-loss",
        type=float,
        default=ExperimentConfig.min_novelty_loss,
    )
    parser.add_argument("--ema-decay", type=float, default=ExperimentConfig.ema_decay)
    parser.add_argument("--include-squares", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = ExperimentConfig(
        num_steps=args.num_steps,
        seeds=args.seeds,
        feature_dim=args.feature_dim,
        n_tasks=args.n_tasks,
        n_contexts=args.n_contexts,
        context_length=args.context_length,
        active_pairs=args.active_pairs,
        noise_std=args.noise_std,
        readout_step_size=args.readout_step_size,
        inference_mode=args.inference_mode,
        min_dwell=args.min_dwell,
        switch_margin=args.switch_margin,
        new_expert_margin=args.new_expert_margin,
        novelty_margin=args.novelty_margin,
        min_novelty_loss=args.min_novelty_loss,
        ema_decay=args.ema_decay,
        include_squares=args.include_squares,
        output_dir=args.output_dir,
    )
    if args.quick:
        config.num_steps = min(config.num_steps, 240)
        config.seeds = min(config.seeds, 2)
        config.context_length = min(config.context_length, 40)
        config.min_dwell = min(config.min_dwell, 4)
    return config


def main() -> None:
    """Run the experiment and write JSON/Markdown outputs."""
    config = parse_args()
    results = run_suite(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "context_inference_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    write_summary(results, config.output_dir)
    gain = results["aggregate"]["gain_summary"]
    print("\nAggregate last-cycle loss:")
    for method in ("hidden_single", "inferred_context", "oracle_context_gated"):
        data = results["aggregate"][method]
        print(f"{method:<22} {data['last_cycle_loss']:.8f}")
    print(
        "fraction_of_oracle_gain_recovered="
        f"{gain['fraction_of_oracle_gain_recovered']:.6f}"
    )
    print(f"Wrote results to {config.output_dir}")


if __name__ == "__main__":
    main()

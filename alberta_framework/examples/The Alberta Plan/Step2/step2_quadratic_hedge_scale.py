#!/usr/bin/env python3
"""Hedge mixture for the scaled Step 2 quadratic direction.

The direct quadratic lift wins only on the pair-product stream and loses on
negative controls.  This experiment asks whether a tiny expert aggregator can
make the mechanism scale: combine fair MLP experts and quadratic experts with
exponential weights updated from prequential loss.

This is still a standalone research script, not a core learner.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from step2_quadratic_centered_scale import (
    MethodSpec,
    ScaleConfig,
    collect_stream_arrays,
    final_window,
    make_learner,
    maybe_lift,
    mean_row,
    paired_lower,
    regression_stream_factory,
)

DEFAULT_OUTPUT_DIR = Path("outputs/step2_quadratic_hedge_scale")
DEFAULT_NOTE_PATH = Path("docs/research/step2_quadratic_hedge_scale.md")


@dataclass(frozen=True)
class HedgeConfig:
    """Configuration for the Hedge regression experiment."""

    num_steps: int = 900
    num_seeds: int = 3
    final_window: int = 180
    eta: float = 1.0
    scale: ScaleConfig = ScaleConfig(num_steps=900, num_seeds=3, final_window=180)
    suites: tuple[str, ...] = ("interaction", "nonlinear")


def expert_specs() -> dict[str, MethodSpec]:
    """Return the expert set."""
    return {
        "mlp_h64": MethodSpec(hidden_sizes=(64,), hash_dim=None),
        "mlp_h128": MethodSpec(hidden_sizes=(128,), hash_dim=None),
        "quad_linear_h128": MethodSpec(hidden_sizes=(), hash_dim=128),
        "quad_linear_h512": MethodSpec(hidden_sizes=(), hash_dim=512),
    }


def run_hedge_on_arrays(
    observations: jax.Array,
    targets: jax.Array,
    config: HedgeConfig,
    key: jax.Array,
) -> dict[str, Any]:
    """Run Hedge experts on one fixed stream."""
    specs = expert_specs()
    keys = jr.split(key, len(specs))
    learners = {
        name: make_learner(int(targets.shape[1]), spec, config.scale)
        for name, spec in specs.items()
    }
    lifted = {name: maybe_lift(observations, spec.hash_dim) for name, spec in specs.items()}
    states = {
        name: learners[name].init(int(lifted[name].shape[1]), keys[idx])
        for idx, name in enumerate(specs)
    }
    log_weights = jnp.zeros(len(specs), dtype=jnp.float32)
    names = tuple(specs.keys())
    expert_losses: dict[str, list[float]] = {name: [] for name in names}
    hedge_losses: list[float] = []
    hedge_weights: list[list[float]] = []

    for t in range(int(observations.shape[0])):
        target = targets[t]
        preds = []
        losses = []
        for name in names:
            pred = learners[name].predict(states[name], lifted[name][t])
            preds.append(pred)
            loss = jnp.mean((pred - target) ** 2)
            losses.append(loss)
            expert_losses[name].append(float(loss))

        weights = jax.nn.softmax(log_weights)
        pred_stack = jnp.stack(preds)
        hedge_pred = jnp.sum(weights[:, None] * pred_stack, axis=0)
        hedge_loss = jnp.mean((hedge_pred - target) ** 2)
        hedge_losses.append(float(hedge_loss))
        hedge_weights.append(np.asarray(weights).tolist())

        loss_vec = jnp.stack(losses)
        log_weights = log_weights - config.eta * loss_vec
        log_weights = log_weights - jnp.max(log_weights)

        for name in names:
            result = learners[name].update(states[name], lifted[name][t], target)
            states[name] = result.state

    return {
        "expert_losses": {name: np.asarray(losses) for name, losses in expert_losses.items()},
        "hedge_losses": np.asarray(hedge_losses),
        "final_weights": hedge_weights[-1],
        "weight_names": list(names),
    }


def run_suite(name: str, config: HedgeConfig) -> dict[str, Any]:
    """Run one regression suite."""
    stream, description = regression_stream_factory(name, config.scale)
    finals: dict[str, list[float]] = {method: [] for method in expert_specs()}
    finals["hedge"] = []
    means: dict[str, list[float]] = {method: [] for method in expert_specs()}
    means["hedge"] = []
    final_weights: list[list[float]] = []

    for seed in range(config.num_seeds):
        root_key = jr.key(seed + 70_000 + 10_000 * list(config.suites).index(name))
        data_key, run_key = jr.split(root_key)
        observations, targets = collect_stream_arrays(stream, config.num_steps, data_key)
        result = run_hedge_on_arrays(observations, targets, config, run_key)
        for method, curve in result["expert_losses"].items():
            finals[method].append(final_window(curve, config.final_window))
            means[method].append(float(np.mean(curve)))
        finals["hedge"].append(final_window(result["hedge_losses"], config.final_window))
        means["hedge"].append(float(np.mean(result["hedge_losses"])))
        final_weights.append(result["final_weights"])
        print(f"hedge={name} seed={seed} complete")

    per_method = {
        method: {
            "final_window_loss": mean_row(finals[method]),
            "mean_loss": mean_row(means[method]),
        }
        for method in finals
    }
    mlp_methods = [name for name in expert_specs() if name.startswith("mlp_")]
    best_mlp = min(mlp_methods, key=lambda method: per_method[method]["final_window_loss"]["mean"])
    best_expert = min(
        expert_specs(),
        key=lambda method: per_method[method]["final_window_loss"]["mean"],
    )
    hedge_pair = paired_lower(finals["hedge"], finals[best_mlp])
    return {
        "description": description,
        "per_method": per_method,
        "best_mlp": best_mlp,
        "best_expert": best_expert,
        "hedge_vs_best_mlp": hedge_pair,
        "hedge_beats_best_mlp": bool(
            hedge_pair["diff_mean"] > 0.0 and hedge_pair["wins"] > hedge_pair["losses"]
        ),
        "weight_names": list(expert_specs()),
        "final_weights_mean": np.mean(np.asarray(final_weights), axis=0).tolist(),
    }


def run(config: HedgeConfig) -> dict[str, Any]:
    """Run all suites."""
    t0 = time.time()
    suites = {name: run_suite(name, config) for name in config.suites}
    flags = [suite["hedge_beats_best_mlp"] for suite in suites.values()]
    return {
        "experiment": "step2_quadratic_hedge_scale",
        "config": {
            "num_steps": config.num_steps,
            "num_seeds": config.num_seeds,
            "final_window": config.final_window,
            "eta": config.eta,
            "scale": asdict(config.scale),
            "suites": list(config.suites),
        },
        "elapsed_s": float(time.time() - t0),
        "suites": suites,
        "universal_hedge_win": bool(all(flags)),
        "suite_win_count": int(sum(flags)),
        "suite_count": int(len(flags)),
    }


def cell(row: dict[str, Any], metric: str) -> str:
    """Format mean +/- stderr."""
    data = row[metric]
    return f"{data['mean']:.4f} +/- {data['stderr']:.4f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    lines = [
        "# Step 2 Quadratic Hedge Scaling",
        "",
        "Hedge mixes fair MLP experts with quadratic experts using exponential "
        "weights over prequential loss. It should use quadratic experts on "
        "pair-product streams and fall back to MLP experts on controls.",
        "",
        f"Universal Hedge win: `{payload['universal_hedge_win']}` "
        f"({payload['suite_win_count']}/{payload['suite_count']} suites).",
        "",
    ]
    for name, suite in payload["suites"].items():
        lines.extend([
            f"## {name}",
            "",
            suite["description"],
            "",
            f"Best MLP: `{suite['best_mlp']}`. Best expert: `{suite['best_expert']}`. "
            f"Hedge beats best MLP: `{suite['hedge_beats_best_mlp']}`.",
            "",
            "| Method | Final-window loss | Mean loss |",
            "|---|---:|---:|",
        ])
        for method, data in sorted(
            suite["per_method"].items(),
            key=lambda item: item[1]["final_window_loss"]["mean"],
        ):
            lines.append(
                f"| `{method}` | {cell(data, 'final_window_loss')} | "
                f"{cell(data, 'mean_loss')} |"
            )
        weights = ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(
                suite["weight_names"],
                suite["final_weights_mean"],
                strict=True,
            )
        )
        lines.extend(["", f"Mean final Hedge weights: {weights}.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any], output_dir: Path, note_path: Path) -> None:
    """Write JSON and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_summary(output_dir / "SUMMARY.md", payload)
    write_summary(note_path, payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE_PATH)
    parser.add_argument("--num-steps", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--suites", nargs="+", default=["interaction", "nonlinear"])
    return parser.parse_args()


def main() -> None:
    """Run Hedge scaling experiment."""
    args = parse_args()
    final_window_size = max(1, args.num_steps // 5)
    scale = ScaleConfig(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        final_window=final_window_size,
    )
    config = HedgeConfig(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        final_window=final_window_size,
        eta=args.eta,
        scale=scale,
        suites=tuple(args.suites),
    )
    payload = run(config)
    write_outputs(payload, args.output_dir, args.note_path)
    print(
        f"universal_hedge_win={payload['universal_hedge_win']} "
        f"suite_wins={payload['suite_win_count']}/{payload['suite_count']}"
    )
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.note_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,no-untyped-call"
"""Probe DiffLogic-style selection over hard EML-derived Boolean gates.

The direct 1-bit EML microtree can learn all two-input Boolean functions, but
its relaxed tree does not always harden from one random seed. This probe tests
the complementary path: enumerate hard EML threshold templates first, then learn
a differentiable selector over that discrete gate library.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array

BOOLEAN_INPUTS = jnp.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=jnp.float32,
)

GATE_NAMES = {
    0: "FALSE",
    1: "NOR",
    2: "not_a_and_b",
    3: "NOT_A",
    4: "a_and_not_b",
    5: "NOT_B",
    6: "XOR",
    7: "NAND",
    8: "AND",
    9: "XNOR",
    10: "B",
    11: "a_implies_b",
    12: "A",
    13: "b_implies_a",
    14: "OR",
    15: "TRUE",
}


class SelectorParams(NamedTuple):
    """Parameters for a differentiable selector over a hard gate library."""

    gate_logits: Array


@dataclass(frozen=True)
class GateSelectorConfig:
    """Configuration for the EML gate selector probe."""

    num_updates: int = 50
    seeds: int = 32
    base_seed: int = 20000
    step_size: float = 0.1
    initial_temperature: float = 1.0
    min_temperature: float = 0.03
    entropy_weight: float = 0.002
    max_grad_norm: float = 10.0
    init_scale: float = 0.2
    eml_template_depth: int = 2
    eml_eps: float = 0.05


TrainSelector = Callable[[Array, Array], tuple[SelectorParams, Array]]


def truth_table(mask: int) -> Array:
    """Return the Boolean target for rows ``00, 01, 10, 11``."""
    if mask < 0 or mask > 15:
        raise ValueError("mask must be in [0, 15]")
    return jnp.array([(mask >> i) & 1 for i in range(4)], dtype=jnp.float32)


def mask_from_values(values: Array) -> int:
    """Convert a four-row Boolean vector into the integer mask convention."""
    bits = [int(value) for value in values.astype(jnp.int32).tolist()]
    return sum(bit << idx for idx, bit in enumerate(bits))


def known_boolean_library() -> tuple[Array, list[str], list[int]]:
    """Return the exact 16 two-input Boolean gates."""
    masks = list(range(16))
    outputs = jnp.stack([truth_table(mask) for mask in masks], axis=0)
    names = [GATE_NAMES[mask] for mask in masks]
    return outputs, names, masks


def _threshold_masks(values: Array) -> list[tuple[int, str]]:
    """Enumerate thresholded bit functions from one scalar signal."""
    unique_values = sorted({round(float(value), 8) for value in values.tolist()})
    candidates = {0: "always_0", 15: "always_1"}
    for threshold in unique_values:
        ge_mask = mask_from_values(values >= threshold)
        le_mask = mask_from_values(values <= threshold)
        candidates.setdefault(ge_mask, f">={threshold:.6g}")
        candidates.setdefault(le_mask, f"<={threshold:.6g}")
    return sorted(candidates.items())


def eml_template_library(*, depth: int, eps: float) -> tuple[Array, list[str], list[int]]:
    """Enumerate hard EML-threshold templates up to a small depth.

    For Boolean inputs, ``bit(scale * eml(left, right) + bias)`` is equivalent
    to thresholding the scalar EML value. Repeated enumeration therefore gives a
    compact hard gate library whose members are still EML-derived circuits.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    expressions: dict[int, str] = {0: "0", 10: "B", 12: "A", 15: "1"}
    frontier = dict(expressions)

    for _ in range(depth):
        new_frontier: dict[int, str] = {}
        for left_mask, left_expr in frontier.items():
            left = truth_table(left_mask)
            for right_mask, right_expr in frontier.items():
                right = truth_table(right_mask)
                eml_values = jnp.exp(left) - jnp.log(eps + right)
                for mask, threshold_expr in _threshold_masks(eml_values):
                    expr = f"bit(eml({left_expr}, {right_expr}) {threshold_expr})"
                    expressions.setdefault(mask, expr)
                    new_frontier.setdefault(mask, expr)
        frontier = new_frontier
        if len(expressions) == 16:
            break

    masks = sorted(expressions)
    outputs = jnp.stack([truth_table(mask) for mask in masks], axis=0)
    names = [
        f"{GATE_NAMES.get(mask, f'mask_{mask}')}: {expressions[mask]}" for mask in masks
    ]
    return outputs, names, masks


def validate_config(config: GateSelectorConfig) -> None:
    """Validate hyperparameters before JAX tracing."""
    if config.num_updates < 1:
        raise ValueError("num_updates must be >= 1")
    if config.seeds < 1:
        raise ValueError("seeds must be >= 1")
    if config.step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if config.initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if config.min_temperature <= 0.0:
        raise ValueError("min_temperature must be positive")
    if config.min_temperature > config.initial_temperature:
        raise ValueError("min_temperature must be <= initial_temperature")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if config.init_scale <= 0.0:
        raise ValueError("init_scale must be positive")


def build_train_selector(
    library_outputs: Array,
    config: GateSelectorConfig,
) -> TrainSelector:
    """Build a JIT-compiled selector trainer for one fixed gate library."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.step_size),
    )

    def predict(params: SelectorParams, temperature: Array) -> Array:
        probs = jax.nn.softmax(params.gate_logits / temperature, axis=0)
        return probs @ library_outputs

    def loss_fn(params: SelectorParams, target: Array, temperature: Array) -> Array:
        predictions = predict(params, temperature)
        bce = -jnp.mean(
            target * jnp.log(predictions + 1e-6)
            + (1.0 - target) * jnp.log(1.0 - predictions + 1e-6)
        )
        probs = jax.nn.softmax(params.gate_logits / temperature, axis=0)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        return bce + config.entropy_weight * entropy

    @jax.jit
    def train_seed(key: Array, target: Array) -> tuple[SelectorParams, Array]:
        params = SelectorParams(
            gate_logits=config.init_scale
            * jr.normal(key, (library_outputs.shape[0],), dtype=jnp.float32)
        )
        opt_state = optimizer.init(params)

        def scan_step(
            carry: tuple[SelectorParams, optax.OptState],
            idx: Array,
        ) -> tuple[tuple[SelectorParams, optax.OptState], Array]:
            params, opt_state = carry
            fraction = idx / jnp.array(config.num_updates, dtype=jnp.float32)
            temperature = jnp.maximum(
                config.min_temperature,
                config.initial_temperature
                * (config.min_temperature / config.initial_temperature) ** fraction,
            )
            loss, grads = jax.value_and_grad(loss_fn)(params, target, temperature)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, _), losses = jax.lax.scan(
            scan_step,
            (params, opt_state),
            jnp.arange(config.num_updates),
        )
        return params, losses[-1]

    return train_seed


def evaluate_selector(
    params: SelectorParams,
    target: Array,
    library_outputs: Array,
    names: list[str],
    masks: list[int],
    config: GateSelectorConfig,
) -> dict[str, Any]:
    """Evaluate soft and hard selector predictions."""
    probs = jax.nn.softmax(params.gate_logits / config.min_temperature, axis=0)
    soft_predictions = probs @ library_outputs
    selected_idx = int(jnp.argmax(params.gate_logits))
    hard_predictions = library_outputs[selected_idx]
    soft_bits = (soft_predictions >= 0.5).astype(jnp.float32)
    return {
        "selected_mask": int(masks[selected_idx]),
        "selected_name": names[selected_idx],
        "soft_predictions": [round(float(value), 6) for value in soft_predictions.tolist()],
        "hard_predictions": [int(value) for value in hard_predictions.tolist()],
        "soft_accuracy": float(jnp.mean(soft_bits == target)),
        "hard_accuracy": float(jnp.mean(hard_predictions == target)),
        "selected_probability": float(probs[selected_idx]),
        "entropy": float(-jnp.sum(probs * jnp.log(probs + 1e-8))),
    }


def run_library(
    library_name: str,
    library_outputs: Array,
    names: list[str],
    masks: list[int],
    config: GateSelectorConfig,
    target_masks: list[int],
) -> dict[str, Any]:
    """Train selector restarts for one gate library."""
    train_seed = build_train_selector(library_outputs, config)
    records = []
    hard_successes = 0
    soft_successes = 0

    for mask in target_masks:
        target = truth_table(mask)
        best_key: tuple[float, float, float] | None = None
        best_record: dict[str, Any] | None = None
        gate_hard_successes = 0
        gate_soft_successes = 0
        for seed in range(config.seeds):
            key = jr.key(config.base_seed + 1000 * mask + seed)
            params, final_loss = train_seed(key, target)
            evaluation = evaluate_selector(
                params,
                target,
                library_outputs,
                names,
                masks,
                config,
            )
            gate_hard_successes += int(evaluation["hard_accuracy"] == 1.0)
            gate_soft_successes += int(evaluation["soft_accuracy"] == 1.0)
            selection_key = (
                evaluation["hard_accuracy"],
                evaluation["soft_accuracy"],
                evaluation["selected_probability"],
            )
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_record = {
                    "mask": mask,
                    "gate": GATE_NAMES.get(mask, f"mask_{mask}"),
                    "target": [int(value) for value in target.tolist()],
                    "best_seed": seed,
                    "final_loss": float(final_loss),
                    **evaluation,
                }

        if best_record is None:
            raise RuntimeError("at least one seed is required")
        best_record["hard_successes"] = gate_hard_successes
        best_record["soft_successes"] = gate_soft_successes
        best_record["seeds"] = config.seeds
        hard_successes += gate_hard_successes
        soft_successes += gate_soft_successes
        records.append(best_record)

    total_attempts = config.seeds * len(target_masks)
    return {
        "library": library_name,
        "library_size": int(library_outputs.shape[0]),
        "library_masks": [int(mask) for mask in masks],
        "config": asdict(config),
        "summary": {
            "num_gates": len(target_masks),
            "best_of_restarts_hard_solved": sum(
                record["hard_accuracy"] == 1.0 for record in records
            ),
            "best_of_restarts_soft_solved": sum(
                record["soft_accuracy"] == 1.0 for record in records
            ),
            "single_seed_hard_successes": hard_successes,
            "single_seed_soft_successes": soft_successes,
            "total_attempts": total_attempts,
            "single_seed_hard_success_rate": hard_successes / total_attempts,
            "single_seed_soft_success_rate": soft_successes / total_attempts,
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=("known", "eml_templates"),
        default=["eml_templates"],
    )
    parser.add_argument("--num-updates", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20000)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.03)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--init-scale", type=float, default=0.2)
    parser.add_argument("--eml-template-depth", type=int, default=2)
    parser.add_argument("--eml-eps", type=float, default=0.05)
    parser.add_argument("--masks", nargs="+", type=int, default=list(range(16)))
    return parser.parse_args()


def main() -> int:
    """Run DiffLogic-style gate selection over hard gate libraries."""
    args = parse_args()
    config = GateSelectorConfig(
        num_updates=args.num_updates,
        seeds=args.seeds,
        base_seed=args.base_seed,
        step_size=args.step_size,
        initial_temperature=args.initial_temperature,
        min_temperature=args.min_temperature,
        entropy_weight=args.entropy_weight,
        max_grad_norm=args.max_grad_norm,
        init_scale=args.init_scale,
        eml_template_depth=args.eml_template_depth,
        eml_eps=args.eml_eps,
    )
    validate_config(config)

    results = []
    if "known" in args.libraries:
        outputs, names, masks = known_boolean_library()
        results.append(
            run_library(
                "known_boolean_gates",
                outputs,
                names,
                masks,
                config,
                args.masks,
            )
        )
    if "eml_templates" in args.libraries:
        outputs, names, masks = eml_template_library(
            depth=config.eml_template_depth,
            eps=config.eml_eps,
        )
        results.append(
            run_library(
                f"eml_threshold_templates_depth_{config.eml_template_depth}",
                outputs,
                names,
                masks,
                config,
                args.masks,
            )
        )

    payload = {"results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

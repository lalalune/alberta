#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,no-untyped-call"
"""Probe 1-bit EML activation trees on all two-input Boolean gates.

This is the stricter Boolean companion to ``step2_eml_tree_probe.py``. During
training, every internal EML node outputs a bit probability or a
straight-through hard bit. During hard evaluation, leaf choices are argmaxed and
every internal node is thresholded, so the resulting object is a real 1-bit EML
circuit rather than just a real-valued symbolic regressor.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

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


class BitEMLParams(NamedTuple):
    """Trainable parameters for a fixed-depth 1-bit EML tree."""

    leaf_logits: Array
    constant_logits: Array
    node_scale: Array
    node_bias: Array


@dataclass(frozen=True)
class BitEMLProbeConfig:
    """Configuration for the 1-bit EML Boolean gate probe."""

    variant: str = "route_bit_ste"
    depth: int = 3
    n_constants: int = 0
    num_updates: int = 6000
    seeds: int = 32
    base_seed: int = 20000
    step_size: float = 0.03
    initial_temperature: float = 1.0
    min_temperature: float = 0.8
    eps: float = 0.05
    route_entropy_weight: float = 0.002
    bit_entropy_weight: float = 0.02
    l2_weight: float = 1e-5
    max_grad_norm: float = 10.0
    init_scale: float = 0.2
    ste_warmup_fraction: float = 0.0


TrainManyFn = Callable[[Array, Array], tuple[BitEMLParams, Array]]
BatchPredictFn = Callable[[BitEMLParams, Array], Array]
ExpressionFn = Callable[[BitEMLParams], str]
ConstantsFn = Callable[[BitEMLParams], list[float]]


def truth_table(mask: int) -> Array:
    """Return the Boolean target for rows ``00, 01, 10, 11``."""
    if mask < 0 or mask > 15:
        raise ValueError("mask must be in [0, 15]")
    return jnp.array([(mask >> i) & 1 for i in range(4)], dtype=jnp.float32)


def _validate_config(config: BitEMLProbeConfig) -> None:
    """Validate hyperparameters before JAX tracing starts."""
    if config.variant not in {"soft", "bit_ste", "route_ste", "route_bit_ste"}:
        raise ValueError(
            "variant must be one of: soft, bit_ste, route_ste, route_bit_ste"
        )
    if config.depth < 1:
        raise ValueError("depth must be >= 1")
    if config.n_constants < 0:
        raise ValueError("n_constants must be >= 0")
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
    if config.eps <= 0.0:
        raise ValueError("eps must be positive")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if config.init_scale <= 0.0:
        raise ValueError("init_scale must be positive")
    if not 0.0 <= config.ste_warmup_fraction <= 1.0:
        raise ValueError("ste_warmup_fraction must be in [0, 1]")


def build_probe_functions(
    config: BitEMLProbeConfig,
) -> tuple[TrainManyFn, BatchPredictFn, BatchPredictFn, ExpressionFn, ConstantsFn]:
    """Build JIT-compiled train/eval functions for one fixed configuration."""
    _validate_config(config)
    n_leaves = 2**config.depth
    n_nodes = n_leaves - 1
    candidate_dim = 4 + config.n_constants
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.step_size),
    )

    def init_params(key: Array) -> BitEMLParams:
        leaf_key, constant_key, scale_key, bias_key = jr.split(key, 4)
        return BitEMLParams(
            leaf_logits=config.init_scale
            * jr.normal(leaf_key, (n_leaves, candidate_dim), dtype=jnp.float32),
            constant_logits=config.init_scale
            * jr.normal(constant_key, (config.n_constants,), dtype=jnp.float32),
            node_scale=0.5
            + config.init_scale * jr.normal(scale_key, (n_nodes,), dtype=jnp.float32),
            node_bias=config.init_scale * jr.normal(bias_key, (n_nodes,), dtype=jnp.float32),
        )

    def constant_values(params: BitEMLParams) -> Array:
        return jax.nn.sigmoid(params.constant_logits)

    def candidates(params: BitEMLParams, observation: Array) -> Array:
        return jnp.concatenate(
            (
                observation,
                jnp.array([0.0, 1.0], dtype=jnp.float32),
                constant_values(params),
            )
        )

    def straight_through(hard_value: Array, soft_value: Array) -> Array:
        return jax.lax.stop_gradient(hard_value - soft_value) + soft_value

    def evaluate_one(
        params: BitEMLParams,
        observation: Array,
        route_temperature: Array,
        bit_temperature: Array,
        *,
        hard_route: bool,
        hard_bits: bool,
        route_ste: bool,
        bit_ste: bool,
    ) -> Array:
        candidate_values = candidates(params, observation)
        if hard_route:
            nodes = candidate_values[jnp.argmax(params.leaf_logits, axis=-1)]
        else:
            route_probs = jax.nn.softmax(params.leaf_logits / route_temperature, axis=-1)
            if route_ste:
                hard_routes = jax.nn.one_hot(
                    jnp.argmax(route_probs, axis=-1),
                    candidate_dim,
                    dtype=jnp.float32,
                )
                route_probs = straight_through(hard_routes, route_probs)
            nodes = route_probs @ candidate_values

        node_offset = 0
        for _ in range(config.depth):
            next_width = nodes.shape[0] // 2
            left = nodes[0::2]
            right = nodes[1::2]
            eml_value = jnp.exp(jnp.clip(left, 0.0, 1.0)) - jnp.log(
                config.eps + jnp.clip(right, 0.0, 1.0)
            )
            scale = params.node_scale[node_offset : node_offset + next_width]
            bias = params.node_bias[node_offset : node_offset + next_width]
            nodes = jax.nn.sigmoid((scale * eml_value + bias) / bit_temperature)
            if bit_ste:
                nodes = straight_through((nodes >= 0.5).astype(jnp.float32), nodes)
            if hard_bits:
                nodes = (nodes >= 0.5).astype(jnp.float32)
            node_offset += next_width

        return nodes[0]

    batch_soft = jax.vmap(
        lambda params, observation, route_temperature, bit_temperature: evaluate_one(
            params,
            observation,
            route_temperature,
            bit_temperature,
            hard_route=False,
            hard_bits=False,
            route_ste=False,
            bit_ste=False,
        ),
        in_axes=(None, 0, None, None),
    )
    batch_route_ste = jax.vmap(
        lambda params, observation, route_temperature, bit_temperature: evaluate_one(
            params,
            observation,
            route_temperature,
            bit_temperature,
            hard_route=False,
            hard_bits=False,
            route_ste=True,
            bit_ste=False,
        ),
        in_axes=(None, 0, None, None),
    )
    batch_bit_ste = jax.vmap(
        lambda params, observation, route_temperature, bit_temperature: evaluate_one(
            params,
            observation,
            route_temperature,
            bit_temperature,
            hard_route=False,
            hard_bits=False,
            route_ste=False,
            bit_ste=True,
        ),
        in_axes=(None, 0, None, None),
    )
    batch_route_bit_ste = jax.vmap(
        lambda params, observation, route_temperature, bit_temperature: evaluate_one(
            params,
            observation,
            route_temperature,
            bit_temperature,
            hard_route=False,
            hard_bits=False,
            route_ste=True,
            bit_ste=True,
        ),
        in_axes=(None, 0, None, None),
    )
    batch_hard = jax.vmap(
        lambda params, observation: evaluate_one(
            params,
            observation,
            jnp.array(config.min_temperature, dtype=jnp.float32),
            jnp.array(config.min_temperature, dtype=jnp.float32),
            hard_route=True,
            hard_bits=True,
            route_ste=False,
            bit_ste=False,
        ),
        in_axes=(None, 0),
    )

    def train_predictions(
        params: BitEMLParams,
        route_temperature: Array,
        bit_temperature: Array,
        fraction: Array,
    ) -> Array:
        if config.variant == "soft":
            return batch_soft(params, BOOLEAN_INPUTS, route_temperature, bit_temperature)

        def soft_branch(_: None) -> Array:
            return batch_soft(params, BOOLEAN_INPUTS, route_temperature, bit_temperature)

        def ste_branch(_: None) -> Array:
            if config.variant == "route_ste":
                return batch_route_ste(
                    params,
                    BOOLEAN_INPUTS,
                    route_temperature,
                    bit_temperature,
                )
            if config.variant == "bit_ste":
                return batch_bit_ste(
                    params,
                    BOOLEAN_INPUTS,
                    route_temperature,
                    bit_temperature,
                )
            return batch_route_bit_ste(
                params,
                BOOLEAN_INPUTS,
                route_temperature,
                bit_temperature,
            )

        if config.ste_warmup_fraction == 0.0:
            return ste_branch(None)
        return jax.lax.cond(
            fraction >= config.ste_warmup_fraction,
            ste_branch,
            soft_branch,
            operand=None,
        )

    def loss_fn(
        params: BitEMLParams,
        target: Array,
        route_temperature: Array,
        bit_temperature: Array,
        fraction: Array,
    ) -> Array:
        predictions = train_predictions(
            params,
            route_temperature,
            bit_temperature,
            fraction,
        )
        soft_predictions = batch_soft(
            params,
            BOOLEAN_INPUTS,
            route_temperature,
            bit_temperature,
        )
        bce = -jnp.mean(
            target * jnp.log(predictions + 1e-6)
            + (1.0 - target) * jnp.log(1.0 - predictions + 1e-6)
        )
        route_probs = jax.nn.softmax(params.leaf_logits / route_temperature, axis=-1)
        route_entropy = jnp.mean(
            -jnp.sum(route_probs * jnp.log(route_probs + 1e-8), axis=-1)
        )
        bit_entropy = jnp.mean(soft_predictions * (1.0 - soft_predictions))
        l2 = jnp.mean(params.node_scale**2) + jnp.mean(params.node_bias**2)
        return (
            bce
            + config.route_entropy_weight * route_entropy
            + config.bit_entropy_weight * bit_entropy
            + config.l2_weight * l2
        )

    @jax.jit
    def train_seed(key: Array, target: Array) -> tuple[BitEMLParams, Array]:
        params = init_params(key)
        opt_state = optimizer.init(params)

        def scan_step(
            carry: tuple[BitEMLParams, Any],
            idx: Array,
        ) -> tuple[tuple[BitEMLParams, Any], Array]:
            params, opt_state = carry
            fraction = idx / jnp.array(config.num_updates, dtype=jnp.float32)
            temperature = jnp.maximum(
                config.min_temperature,
                config.initial_temperature
                * (config.min_temperature / config.initial_temperature) ** fraction,
            )
            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                target,
                temperature,
                temperature,
                fraction,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, _), losses = jax.lax.scan(
            scan_step,
            (params, opt_state),
            jnp.arange(config.num_updates),
        )
        return params, losses[-1]

    train_many = jax.jit(jax.vmap(train_seed, in_axes=(0, None)))

    def hard_expression(params: BitEMLParams) -> str:
        names = ("a", "b", "0", "1", *tuple(f"c{i}" for i in range(config.n_constants)))
        nodes = [names[int(idx)] for idx in jnp.argmax(params.leaf_logits, axis=-1)]
        node_offset = 0
        for _ in range(config.depth):
            next_nodes = []
            for i in range(0, len(nodes), 2):
                scale = float(params.node_scale[node_offset])
                bias = float(params.node_bias[node_offset])
                next_nodes.append(
                    f"bit({scale:.3g} * eml({nodes[i]}, {nodes[i + 1]}) {bias:+.3g})"
                )
                node_offset += 1
            nodes = next_nodes
        return nodes[0]

    def hard_constants(params: BitEMLParams) -> list[float]:
        return [float(value) for value in constant_values(params).tolist()]

    return (
        cast(TrainManyFn, train_many),
        cast(BatchPredictFn, batch_soft),
        cast(BatchPredictFn, batch_hard),
        hard_expression,
        hard_constants,
    )


def _accuracy(predictions: Array, target: Array) -> float:
    return float(jnp.mean((predictions >= 0.5) == (target >= 0.5)))


def _mse(predictions: Array, target: Array) -> float:
    return float(jnp.mean((predictions - target) ** 2))


def _rounded_list(values: Array, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def run_gate(
    config: BitEMLProbeConfig,
    mask: int,
    train_many: TrainManyFn,
    batch_soft: BatchPredictFn,
    batch_hard: BatchPredictFn,
    hard_expression: ExpressionFn,
    hard_constants: ConstantsFn,
) -> dict[str, Any]:
    """Train multiple random restarts for one Boolean gate."""
    target = truth_table(mask)
    best_key: tuple[float, float, float, float] | None = None
    best_record: dict[str, Any] | None = None
    soft_successes = 0
    hard_successes = 0
    seed_values = config.base_seed + 1000 * mask + jnp.arange(config.seeds)
    keys = jax.vmap(jr.key)(seed_values)
    params_many, losses = train_many(keys, target)
    soft_predictions_many = jax.vmap(
        lambda params: batch_soft(
            params,
            BOOLEAN_INPUTS,
            jnp.array(config.min_temperature, dtype=jnp.float32),
            jnp.array(config.min_temperature, dtype=jnp.float32),
        )
    )(params_many)
    hard_predictions_many = jax.vmap(lambda params: batch_hard(params, BOOLEAN_INPUTS))(
        params_many
    )

    for seed in range(config.seeds):
        params = jax.tree.map(lambda value: value[seed], params_many)
        loss = losses[seed]
        soft_predictions = soft_predictions_many[seed]
        hard_predictions = hard_predictions_many[seed]
        soft_accuracy = _accuracy(soft_predictions, target)
        hard_accuracy = float(jnp.mean(hard_predictions == target))
        soft_mse = _mse(soft_predictions, target)
        hard_mse = _mse(hard_predictions, target)
        soft_successes += int(soft_accuracy == 1.0)
        hard_successes += int(hard_accuracy == 1.0)

        selection_key = (hard_accuracy, soft_accuracy, -hard_mse, -soft_mse)
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_record = {
                "mask": mask,
                "gate": GATE_NAMES.get(mask, f"mask_{mask}"),
                "target": [int(value) for value in target.tolist()],
                "best_seed": seed,
                "soft_predictions": _rounded_list(soft_predictions),
                "hard_predictions": [int(value) for value in hard_predictions.tolist()],
                "soft_accuracy": soft_accuracy,
                "hard_accuracy": hard_accuracy,
                "soft_mse": soft_mse,
                "hard_mse": hard_mse,
                "final_loss": float(loss),
                "hard_expression": hard_expression(params),
                "constants": _rounded_list(jnp.array(hard_constants(params))),
            }

    if best_record is None:
        raise RuntimeError("at least one seed is required")
    best_record["soft_successes"] = soft_successes
    best_record["hard_successes"] = hard_successes
    best_record["seeds"] = config.seeds
    return best_record


def run_sweep(config: BitEMLProbeConfig, masks: list[int]) -> dict[str, Any]:
    """Run the 1-bit EML probe over selected Boolean gate masks."""
    train_many, batch_soft, batch_hard, hard_expression, hard_constants = (
        build_probe_functions(config)
    )
    records = [
        run_gate(
            config,
            mask,
            train_many,
            batch_soft,
            batch_hard,
            hard_expression,
            hard_constants,
        )
        for mask in masks
    ]
    total_attempts = config.seeds * len(records)
    hard_solved = sum(record["hard_accuracy"] == 1.0 for record in records)
    soft_solved = sum(record["soft_accuracy"] == 1.0 for record in records)
    hard_successes = sum(int(record["hard_successes"]) for record in records)
    soft_successes = sum(int(record["soft_successes"]) for record in records)
    return {
        "config": asdict(config),
        "summary": {
            "num_gates": len(records),
            "best_of_restarts_hard_solved": hard_solved,
            "best_of_restarts_soft_solved": soft_solved,
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
        "--variant",
        choices=("soft", "bit_ste", "route_ste", "route_bit_ste"),
        default="route_bit_ste",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-constants", type=int, default=0)
    parser.add_argument("--num-updates", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20000)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--min-temperature", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--route-entropy-weight", type=float, default=0.002)
    parser.add_argument("--bit-entropy-weight", type=float, default=0.02)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--init-scale", type=float, default=0.2)
    parser.add_argument("--ste-warmup-fraction", type=float, default=0.0)
    parser.add_argument("--masks", nargs="+", type=int, default=list(range(16)))
    return parser.parse_args()


def main() -> int:
    """Run the 1-bit EML Boolean gate probe."""
    args = parse_args()
    config = BitEMLProbeConfig(
        variant=args.variant,
        depth=args.depth,
        n_constants=args.n_constants,
        num_updates=args.num_updates,
        seeds=args.seeds,
        base_seed=args.base_seed,
        step_size=args.step_size,
        initial_temperature=args.initial_temperature,
        min_temperature=args.min_temperature,
        eps=args.eps,
        route_entropy_weight=args.route_entropy_weight,
        bit_entropy_weight=args.bit_entropy_weight,
        l2_weight=args.l2_weight,
        max_grad_norm=args.max_grad_norm,
        init_scale=args.init_scale,
        ste_warmup_fraction=args.ste_warmup_fraction,
    )
    result = run_sweep(config, args.masks)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train continuous DiffEML on a real-valued synthetic classification task."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.continuous_diffeml import (
    compile_sparse_continuous_eml_circuit,
    compiled_sparse_continuous_eml_circuit_forward,
    compiled_sparse_continuous_eml_circuit_forward_approx,
    continuous_diffeml_forward,
    continuous_diffeml_train_step,
    init_continuous_diffeml_state,
    init_sparse_continuous_eml_circuit_state,
    sparse_continuous_eml_circuit_forward,
    sparse_continuous_eml_circuit_train_step,
)


@dataclass(frozen=True)
class DemoConfig:
    steps: int
    seeds: int
    train_size: int
    test_size: int
    input_dim: int
    hidden_size: int
    learning_rate: float
    sparse_final_temperature: float
    sparse_entropy_weight: float
    sparse_hard_loss_weight: float
    output_dir: str
    seed: int


class BaselineParams(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array


class AdamState(NamedTuple):
    m: BaselineParams
    v: BaselineParams
    step: jax.Array


def make_dataset(
    key: jax.Array,
    *,
    n_samples: int,
    input_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Create a nonlinear real-valued classification problem."""
    x = jr.normal(key, (n_samples, input_dim), dtype=jnp.float32)
    score = (
        jnp.sin(1.5 * x[:, 0] * x[:, 1])
        + 0.35 * x[:, 2] ** 2
        - 0.6 * x[:, 3]
        + 0.25 * x[:, 4]
    )
    labels = (score > jnp.median(score)).astype(jnp.int32)
    x = (x - jnp.mean(x, axis=0, keepdims=True)) / (jnp.std(x, axis=0, keepdims=True) + 1e-6)
    return x, labels


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))


def accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))


def logit_error(candidate: jax.Array, reference: jax.Array) -> dict[str, float]:
    """Compare candidate logits to reference logits."""
    delta = candidate - reference
    return {
        "mean_abs_logit_error": float(jnp.mean(jnp.abs(delta))),
        "max_abs_logit_error": float(jnp.max(jnp.abs(delta))),
        "rms_logit_error": float(jnp.sqrt(jnp.mean(delta * delta))),
        "top1_agreement": float(
            jnp.mean(jnp.argmax(candidate, axis=-1) == jnp.argmax(reference, axis=-1))
        ),
    }


def init_baseline(
    key: jax.Array,
    input_dim: int,
    hidden_size: int,
) -> tuple[BaselineParams, AdamState]:
    k1, k2 = jr.split(key)
    params = BaselineParams(
        w1=jr.normal(k1, (input_dim, hidden_size), dtype=jnp.float32)
        / jnp.sqrt(jnp.asarray(input_dim, dtype=jnp.float32)),
        b1=jnp.zeros(hidden_size, dtype=jnp.float32),
        w2=jr.normal(k2, (hidden_size, 2), dtype=jnp.float32)
        / jnp.sqrt(jnp.asarray(hidden_size, dtype=jnp.float32)),
        b2=jnp.zeros(2, dtype=jnp.float32),
    )
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return params, AdamState(m=zeros, v=zeros, step=jnp.array(0, dtype=jnp.int32))


def baseline_forward(params: BaselineParams, x: jax.Array, *, linear: bool) -> jax.Array:
    if linear:
        return x @ params.w1[:, :2] + params.b1[:2]
    hidden = jax.nn.gelu(x @ params.w1 + params.b1)
    return hidden @ params.w2 + params.b2


def baseline_step(
    params: BaselineParams,
    opt: AdamState,
    x: jax.Array,
    labels: jax.Array,
    *,
    learning_rate: float,
    linear: bool,
) -> tuple[BaselineParams, AdamState, jax.Array]:
    def loss_fn(candidate: BaselineParams) -> jax.Array:
        return cross_entropy(baseline_forward(candidate, x, linear=linear), labels)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    step = opt.step + jnp.array(1, dtype=jnp.int32)
    m = jax.tree_util.tree_map(lambda old, grad: 0.9 * old + 0.1 * grad, opt.m, grads)
    v = jax.tree_util.tree_map(lambda old, grad: 0.999 * old + 0.001 * grad**2, opt.v, grads)
    step_f = step.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.9**step_f), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.999**step_f), v)
    new_params = jax.tree_util.tree_map(
        lambda param, mean, variance: param
        - learning_rate * mean / (jnp.sqrt(variance) + 1e-8),
        params,
        m_hat,
        v_hat,
    )
    return new_params, AdamState(m=m, v=v, step=step), loss


def run_continuous_diffeml(
    train_x: jax.Array,
    train_y: jax.Array,
    test_x: jax.Array,
    test_y: jax.Array,
    key: jax.Array,
    config: DemoConfig,
) -> dict[str, float]:
    state = init_continuous_diffeml_state(
        key,
        input_dim=config.input_dim,
        output_dim=2,
        hidden_sizes=(config.hidden_size, config.hidden_size),
    )

    @jax.jit
    def train(state: Any) -> tuple[Any, jax.Array]:
        def step(carry: Any, _: jax.Array) -> tuple[Any, jax.Array]:
            result = continuous_diffeml_train_step(
                carry,
                train_x,
                train_y,
                learning_rate=config.learning_rate,
                loss="softmax_cross_entropy",
                max_grad_norm=10.0,
                l2_penalty=1e-5,
            )
            return result.state, result.loss

        return jax.lax.scan(step, state, jnp.arange(config.steps))

    state, losses = train(state)
    train_logits = continuous_diffeml_forward(state.params, train_x)
    test_logits = continuous_diffeml_forward(state.params, test_x)
    return {
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "train_accuracy": float(accuracy(train_logits, train_y)),
        "test_accuracy": float(accuracy(test_logits, test_y)),
    }


def run_sparse_continuous_eml_circuit(
    train_x: jax.Array,
    train_y: jax.Array,
    test_x: jax.Array,
    test_y: jax.Array,
    key: jax.Array,
    config: DemoConfig,
) -> dict[str, float]:
    """Train the sparse source-selector EML circuit by backprop."""
    state = init_sparse_continuous_eml_circuit_state(
        key,
        input_dim=config.input_dim,
        output_dim=2,
        depth=2,
        width=config.hidden_size,
    )

    @jax.jit
    def train(state: Any) -> tuple[Any, jax.Array]:
        def step(carry: Any, idx: jax.Array) -> tuple[Any, jax.Array]:
            fraction = idx.astype(jnp.float32) / jnp.maximum(
                1.0,
                jnp.asarray(config.steps - 1, dtype=jnp.float32),
            )
            temperature = 1.0 * (config.sparse_final_temperature / 1.0) ** fraction
            result = sparse_continuous_eml_circuit_train_step(
                carry,
                train_x,
                train_y,
                learning_rate=config.learning_rate,
                loss="softmax_cross_entropy",
                temperature=temperature,
                entropy_weight=config.sparse_entropy_weight,
                hard_loss_weight=config.sparse_hard_loss_weight,
                max_grad_norm=10.0,
                l2_penalty=1e-5,
            )
            return result.state, result.loss

        return jax.lax.scan(step, state, jnp.arange(config.steps))

    state, losses = train(state)
    soft_train_logits = sparse_continuous_eml_circuit_forward(
        state.params,
        train_x,
        temperature=config.sparse_final_temperature,
    )
    soft_test_logits = sparse_continuous_eml_circuit_forward(
        state.params,
        test_x,
        temperature=config.sparse_final_temperature,
    )
    hard_train_logits = sparse_continuous_eml_circuit_forward(state.params, train_x, hard=True)
    hard_test_logits = sparse_continuous_eml_circuit_forward(state.params, test_x, hard=True)
    compiled = compile_sparse_continuous_eml_circuit(
        state.params,
        input_dim=config.input_dim,
    )
    compiled_train_logits = compiled_sparse_continuous_eml_circuit_forward(
        compiled,
        train_x,
    )
    compiled_test_logits = compiled_sparse_continuous_eml_circuit_forward(
        compiled,
        test_x,
    )

    summary = {
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "train_accuracy": float(accuracy(compiled_train_logits, train_y)),
        "test_accuracy": float(accuracy(compiled_test_logits, test_y)),
        "soft_train_accuracy": float(accuracy(soft_train_logits, train_y)),
        "soft_test_accuracy": float(accuracy(soft_test_logits, test_y)),
        "hard_train_accuracy": float(accuracy(hard_train_logits, train_y)),
        "hard_test_accuracy": float(accuracy(hard_test_logits, test_y)),
        "compiled_train_accuracy": float(accuracy(compiled_train_logits, train_y)),
        "compiled_test_accuracy": float(accuracy(compiled_test_logits, test_y)),
        "hardening_test_accuracy_gap": float(
            accuracy(soft_test_logits, test_y) - accuracy(compiled_test_logits, test_y)
        ),
    }
    approx_specs: list[tuple[str, dict[str, Any]]] = [
        ("approx_lut", {"approximation": "lut"}),
        ("approx_lut_fast_tanh", {"approximation": "lut", "approximate_tanh": True}),
        ("approx_poly", {"approximation": "poly"}),
        ("approx_poly_fast_tanh", {"approximation": "poly", "approximate_tanh": True}),
    ]
    for name, kwargs in approx_specs:
        train_logits = compiled_sparse_continuous_eml_circuit_forward_approx(
            compiled,
            train_x,
            **kwargs,
        )
        test_logits = compiled_sparse_continuous_eml_circuit_forward_approx(
            compiled,
            test_x,
            **kwargs,
        )
        errors = logit_error(test_logits, compiled_test_logits)
        summary[f"{name}_train_accuracy"] = float(accuracy(train_logits, train_y))
        summary[f"{name}_test_accuracy"] = float(accuracy(test_logits, test_y))
        for error_name, value in errors.items():
            summary[f"{name}_{error_name}"] = value
    return summary


def run_baseline(
    train_x: jax.Array,
    train_y: jax.Array,
    test_x: jax.Array,
    test_y: jax.Array,
    key: jax.Array,
    config: DemoConfig,
    *,
    linear: bool,
) -> dict[str, float]:
    params, opt = init_baseline(key, config.input_dim, config.hidden_size)

    @jax.jit
    def train(
        params: BaselineParams,
        opt: AdamState,
    ) -> tuple[BaselineParams, AdamState, jax.Array]:
        def step(
            carry: tuple[BaselineParams, AdamState],
            _: jax.Array,
        ) -> tuple[tuple[BaselineParams, AdamState], jax.Array]:
            current_params, current_opt = carry
            next_params, next_opt, loss = baseline_step(
                current_params,
                current_opt,
                train_x,
                train_y,
                learning_rate=config.learning_rate,
                linear=linear,
            )
            return (next_params, next_opt), loss

        (next_params, next_opt), losses = jax.lax.scan(
            step,
            (params, opt),
            jnp.arange(config.steps),
        )
        return next_params, next_opt, losses

    params, _, losses = train(params, opt)
    train_logits = baseline_forward(params, train_x, linear=linear)
    test_logits = baseline_forward(params, test_x, linear=linear)
    return {
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "train_accuracy": float(accuracy(train_logits, train_y)),
        "test_accuracy": float(accuracy(test_logits, test_y)),
    }


def stderr(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    methods = [
        "linear_adam",
        "mlp_adam",
        "continuous_diffeml_adam",
        "sparse_continuous_eml_circuit_adam",
    ]
    lines = [
        "# Continuous DiffEML Demo",
        "",
        "Synthetic real-valued classification task with standard Adam backpropagation.",
        "",
        "| Method | Test accuracy | Final loss |",
        "|---|---:|---:|",
    ]
    for method in methods:
        rows = [row["summary"] for row in records if row["method"] == method]
        acc = np.asarray([row["test_accuracy"] for row in rows], dtype=np.float64)
        loss = np.asarray([row["final_loss"] for row in rows], dtype=np.float64)
        lines.append(
            f"| `{method}` | {np.mean(acc):.4f} +/- {stderr(acc):.4f} | "
            f"{np.mean(loss):.4f} +/- {stderr(loss):.4f} |"
        )
    sparse_rows = [
        row["summary"]
        for row in records
        if row["method"] == "sparse_continuous_eml_circuit_adam"
    ]
    if sparse_rows and "approx_lut_test_accuracy" in sparse_rows[0]:
        lines.extend(
            [
                "",
                "## Sparse EML Compiled Kernel Accuracy",
                "",
                "| Kernel | Test accuracy | Top-1 agreement | Mean abs logit error |",
                "|---|---:|---:|---:|",
            ]
        )
        kernel_rows = [
            ("compiled_exact", "compiled_test_accuracy", None, None),
            (
                "approx_lut",
                "approx_lut_test_accuracy",
                "approx_lut_top1_agreement",
                "approx_lut_mean_abs_logit_error",
            ),
            (
                "approx_lut_fast_tanh",
                "approx_lut_fast_tanh_test_accuracy",
                "approx_lut_fast_tanh_top1_agreement",
                "approx_lut_fast_tanh_mean_abs_logit_error",
            ),
            (
                "approx_poly",
                "approx_poly_test_accuracy",
                "approx_poly_top1_agreement",
                "approx_poly_mean_abs_logit_error",
            ),
            (
                "approx_poly_fast_tanh",
                "approx_poly_fast_tanh_test_accuracy",
                "approx_poly_fast_tanh_top1_agreement",
                "approx_poly_fast_tanh_mean_abs_logit_error",
            ),
        ]
        for label, acc_key, agreement_key, error_key in kernel_rows:
            acc = np.asarray([row[acc_key] for row in sparse_rows], dtype=np.float64)
            agreement = (
                np.ones_like(acc)
                if agreement_key is None
                else np.asarray([row[agreement_key] for row in sparse_rows], dtype=np.float64)
            )
            error = (
                np.zeros_like(acc)
                if error_key is None
                else np.asarray([row[error_key] for row in sparse_rows], dtype=np.float64)
            )
            lines.append(
                f"| `{label}` | {np.mean(acc):.4f} +/- {stderr(acc):.4f} | "
                f"{np.mean(agreement):.4f} +/- {stderr(agreement):.4f} | "
                f"{np.mean(error):.6f} +/- {stderr(error):.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--sparse-final-temperature", type=float, default=0.15)
    parser.add_argument("--sparse-entropy-weight", type=float, default=0.01)
    parser.add_argument("--sparse-hard-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/step2_continuous_diffeml_demo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_dim < 5:
        raise ValueError("--input-dim must be at least 5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DemoConfig(
        steps=args.steps,
        seeds=args.seeds,
        train_size=args.train_size,
        test_size=args.test_size,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        sparse_final_temperature=args.sparse_final_temperature,
        sparse_entropy_weight=args.sparse_entropy_weight,
        sparse_hard_loss_weight=args.sparse_hard_loss_weight,
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        root, train_key, test_key, diffeml_key, sparse_key, linear_key, mlp_key = jr.split(
            root,
            7,
        )
        train_x, train_y = make_dataset(
            train_key,
            n_samples=args.train_size,
            input_dim=args.input_dim,
        )
        test_x, test_y = make_dataset(
            test_key,
            n_samples=args.test_size,
            input_dim=args.input_dim,
        )
        runners: list[tuple[str, Callable[[], dict[str, float]]]] = [
            (
                "linear_adam",
                lambda: run_baseline(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    linear_key,
                    config,
                    linear=True,
                ),
            ),
            (
                "mlp_adam",
                lambda: run_baseline(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    mlp_key,
                    config,
                    linear=False,
                ),
            ),
            (
                "continuous_diffeml_adam",
                lambda: run_continuous_diffeml(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    diffeml_key,
                    config,
                ),
            ),
            (
                "sparse_continuous_eml_circuit_adam",
                lambda: run_sparse_continuous_eml_circuit(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    sparse_key,
                    config,
                ),
            ),
        ]
        for method, runner in runners:
            summary = runner()
            records.append({"seed": seed_idx, "method": method, "summary": summary})
            print(
                f"seed={seed_idx} {method}: "
                f"test_acc={summary['test_accuracy']:.3f}, "
                f"final_loss={summary['final_loss']:.4f}"
            )

    payload = {
        "config": asdict(config),
        "elapsed_s": time.perf_counter() - start,
        "records": records,
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

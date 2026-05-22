#!/usr/bin/env python3
# mypy: disable-error-code="call-arg,no-any-return,untyped-decorator"
"""Tiny Shakespeare prototype-value meta-reset sweeps.

This worker owns a forked experiment around
``step2_tiny_shakespeare_proto_basis_transformer.py`` without editing the
shared runner.  It tests whether prototype value rows should be reinitialized
by a learned/meta initializer when novelty allocation replaces a center.

Reset policies:

* ``zero``: D18-style neutral reset, replacing novel rows with zeros.
* ``no_reset``: leave recycled value rows untouched.
* ``learned_global``: keep one global initializer vector updated by the
  prototype-value gradient.
* ``success_ema``: update the initializer as a success-weighted EMA of recently
  useful value rows.
* ``success_copy``: copy the most recent sufficiently successful value row.

The prototype-value learning rate is decoupled from the attention/FFN/readout
learning rate, and the initializer update rate is decoupled again.  Each active
component still updates every step; only replacement-row initialization differs.

Smoke:

```bash
source .venv/bin/activate
python -m py_compile "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_meta_reset.py"
python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_proto_meta_reset.py" --smoke
```
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
from step2_tiny_shakespeare_proto_basis_transformer import (
    eval_hybrid_proto_transformer,
    eval_proto_transformer,
    hybrid_transformer_logits,
    init_hybrid_transformer,
    init_proto_transformer,
    make_proto_block,
    proto_transformer_logits,
    select_center_slot,
    summarize_proto_diagnostics,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (
    clip_grads,
    count_array_bytes,
    count_array_elements,
    encode_text,
    ensure_tiny_shakespeare,
    eval_transformer,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    sgd_step,
    stderr,
    summarize_online,
)

from alberta_framework.core.prototype_basis import (
    PrototypeBasisBlock,
    PrototypeBasisParams,
    PrototypeBasisState,
)

RESET_POLICIES = (
    "zero",
    "no_reset",
    "learned_global",
    "success_ema",
    "success_copy",
)
METHOD_KINDS = ("pure", "hybrid")
DEFAULT_OUTPUT_DIR = Path("outputs/step2_new_directions/proto_meta_reset_worker")
DEFAULT_DOC_PATH = Path("docs/research/step2_new_directions/proto_meta_reset_worker.md")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration captured into the result artifact."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    mlp_hidden: int
    proto_count: int
    eval_steps: int
    final_window: int
    train_fraction: float
    baseline_lr: float
    model_lr: float
    proto_value_lrs: tuple[float, ...]
    proto_init_lrs: tuple[float, ...]
    reset_policies: tuple[str, ...]
    method_kinds: tuple[str, ...]
    grad_clip: float
    proto_update_rate: float
    proto_novelty_threshold: float
    proto_bandwidth: float
    proto_adaptive_bandwidth: bool
    proto_bandwidth_update_rate: float
    init_success_temperature: float
    copy_success_threshold: float
    data_path: str
    output_dir: str
    seed: int


@dataclass(frozen=True)
class VariantSpec:
    """One prototype variant in the sweep."""

    kind: str
    reset_policy: str
    proto_value_lr: float
    proto_init_lr: float

    @property
    def method(self) -> str:
        """Stable method id."""
        value = f"{self.proto_value_lr:g}".replace(".", "p").replace("-", "m")
        init = f"{self.proto_init_lr:g}".replace(".", "p").replace("-", "m")
        return (
            f"{self.kind}_proto_{self.reset_policy}_"
            f"value_lr{value}_init_lr{init}"
        )


def parse_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated command-line value."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def add_proto_init(params: dict[str, Any], d_model: int) -> dict[str, Any]:
    """Attach a meta initializer vector to prototype params."""
    return {**params, "proto_init": jnp.zeros((d_model,), dtype=jnp.float32)}


def split_sgd_step(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    model_lr: float,
    proto_value_lr: float,
) -> dict[str, Any]:
    """Apply separate learning rates to model params and prototype values."""
    updated: dict[str, Any] = {}
    for key, value in params.items():
        if key == "proto":
            proto_grads = grads["proto"]
            updated[key] = PrototypeBasisParams(
                values=value.values - proto_value_lr * proto_grads.values,
                bias=value.bias - proto_value_lr * proto_grads.bias,
            )
        elif key == "proto_init":
            updated[key] = value
        else:
            updated[key] = sgd_step(value, grads[key], model_lr)
    return updated


def update_proto_initializer(
    params: dict[str, Any],
    grads: dict[str, Any],
    *,
    slot: jax.Array,
    loss: jax.Array,
    reset_policy: str,
    proto_init_lr: float,
    success_temperature: float,
    copy_success_threshold: float,
) -> dict[str, Any]:
    """Update the meta initializer after observing the current loss/gradient."""
    if reset_policy not in {"learned_global", "success_ema", "success_copy"}:
        return params

    init = params["proto_init"]
    success = jnp.exp(
        -loss / jnp.maximum(jnp.asarray(success_temperature, dtype=jnp.float32), 1e-6)
    )
    success = jnp.clip(success, 0.0, 1.0)
    rate = jnp.clip(proto_init_lr * success, 0.0, 1.0)

    if reset_policy == "learned_global":
        grad_signal = jnp.mean(grads["proto"].values, axis=0)
        new_init = init - proto_init_lr * success * grad_signal
    elif reset_policy == "success_copy":
        candidate = params["proto"].values[slot]
        new_init = jnp.where(success >= copy_success_threshold, candidate, init)
    else:
        candidate = params["proto"].values[slot]
        new_init = (1.0 - rate) * init + rate * candidate
    return {**params, "proto_init": new_init}


def reset_value_row(
    params: dict[str, Any],
    *,
    slot: jax.Array,
    novel: jax.Array,
    reset_policy: str,
) -> dict[str, Any]:
    """Initialize a prototype value row for future predictions."""
    if reset_policy == "no_reset":
        return params
    proto = params["proto"]
    old_row = proto.values[slot]
    if reset_policy == "zero":
        init_row = jnp.zeros_like(old_row)
    else:
        init_row = params["proto_init"]
    new_row = jnp.where(novel, init_row, old_row)
    new_proto = PrototypeBasisParams(
        values=proto.values.at[slot].set(new_row),
        bias=proto.bias,
    )
    return {**params, "proto": new_proto}


def run_meta_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    kind: str,
    model_lr: float,
    proto_value_lr: float,
    proto_init_lr: float,
    grad_clip: float,
    reset_policy: str,
    init_success_temperature: float,
    copy_success_threshold: float,
) -> tuple[dict[str, Any], PrototypeBasisState, np.ndarray]:
    """Train one pure/hybrid prototype transformer with meta reset."""
    if kind == "pure":
        logits_fn = proto_transformer_logits
    elif kind == "hybrid":
        logits_fn = hybrid_transformer_logits
    else:
        raise ValueError(f"unknown method kind {kind!r}")

    @jax.jit
    def scan(
        params: dict[str, Any],
        state: PrototypeBasisState,
    ) -> tuple[tuple[dict[str, Any], PrototypeBasisState], jax.Array]:
        def step(
            carry: tuple[dict[str, Any], PrototypeBasisState],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[dict[str, Any], PrototypeBasisState], jax.Array]:
            params, state = carry
            context, label = inputs

            def loss_fn(
                candidate: dict[str, Any],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
                logits, hidden, activations = logits_fn(block, candidate, state, context)
                return (
                    jax.nn.logsumexp(logits) - logits[label],
                    (logits, hidden, activations),
                )

            (loss, (logits, hidden, activations)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(params)
            grads = clip_grads(grads, grad_clip)
            new_params = split_sgd_step(
                params,
                grads,
                model_lr=model_lr,
                proto_value_lr=proto_value_lr,
            )
            slot, novel = select_center_slot(block, state, hidden)
            new_state, center_metrics = block.update_centers(state, hidden)
            new_params = update_proto_initializer(
                new_params,
                grads,
                slot=slot,
                loss=loss,
                reset_policy=reset_policy,
                proto_init_lr=proto_init_lr,
                success_temperature=init_success_temperature,
                copy_success_threshold=copy_success_threshold,
            )
            new_params = reset_value_row(
                new_params,
                slot=slot,
                novel=novel,
                reset_policy=reset_policy,
            )
            acc = (jnp.argmax(logits) == label).astype(jnp.float32)
            value_norm = jnp.mean(jnp.linalg.norm(new_params["proto"].values, axis=1))
            metrics = jnp.stack(
                [
                    loss,
                    acc,
                    jnp.sum(activations > 1e-6).astype(jnp.float32),
                    center_metrics[0],
                    center_metrics[1],
                    center_metrics[2],
                    novel.astype(jnp.float32),
                    jnp.linalg.norm(new_params["proto_init"]),
                    value_norm,
                ]
            )
            return (new_params, new_state), metrics

        return jax.lax.scan(step, (params, state), (contexts, labels))

    (final_params, final_state), metrics = scan(params, state)
    metrics.block_until_ready()
    return final_params, final_state, np.asarray(metrics)


def eval_meta_proto_transformer(
    block: PrototypeBasisBlock,
    params: dict[str, Any],
    state: PrototypeBasisState,
    contexts: jax.Array,
    labels: jax.Array,
    *,
    kind: str,
) -> dict[str, float]:
    """Evaluate a pure/hybrid prototype transformer."""
    stripped = {key: value for key, value in params.items() if key != "proto_init"}
    if kind == "pure":
        return eval_proto_transformer(block, stripped, state, contexts, labels)
    if kind == "hybrid":
        return eval_hybrid_proto_transformer(block, stripped, state, contexts, labels)
    raise ValueError(f"unknown method kind {kind!r}")


def summarize_meta_diagnostics(metrics: np.ndarray, final_window: int) -> dict[str, float]:
    """Summarize meta-reset diagnostics."""
    window = metrics[-min(final_window, metrics.shape[0]) :]
    return {
        "reset_rate": float(np.mean(metrics[:, 6])),
        "final_window_reset_rate": float(np.mean(window[:, 6])),
        "final_init_norm": float(metrics[-1, 7]),
        "mean_init_norm": float(np.mean(metrics[:, 7])),
        "final_value_row_norm": float(metrics[-1, 8]),
        "mean_value_row_norm": float(np.mean(metrics[:, 8])),
    }


def aggregate_metric(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
) -> np.ndarray:
    """Collect one metric across seeds."""
    return np.asarray(
        [row["summary"][metric] for row in records if row["method"] == method],
        dtype=np.float64,
    )


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return simple per-method means/stderr for all recorded summary metrics."""
    methods = sorted({row["method"] for row in records})
    aggregate: dict[str, Any] = {}
    for method in methods:
        rows = [row["summary"] for row in records if row["method"] == method]
        metrics = sorted({metric for row in rows for metric in row})
        aggregate[method] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "stderr": stderr(values),
                "values": [float(value) for value in values],
            }
    return aggregate


def best_method(
    aggregate: dict[str, Any],
    metric: str,
    *,
    lower_is_better: bool,
    exclude_baseline: bool = False,
) -> tuple[str, float]:
    """Return best method and aggregate mean for one metric."""
    candidates = []
    for method, row in aggregate.items():
        if exclude_baseline and method == "baseline_ffn_tuned":
            continue
        if metric in row:
            candidates.append((method, row[metric]["mean"]))
    if not candidates:
        raise ValueError(f"no candidates for metric {metric!r}")
    key = min if lower_is_better else max
    return key(candidates, key=lambda item: item[1])


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a Markdown summary for the worker."""
    aggregate = payload["aggregate"]
    baseline = aggregate["baseline_ffn_tuned"]
    best_proto_nll = best_method(
        aggregate,
        "final_window_nll",
        lower_is_better=True,
        exclude_baseline=True,
    )
    best_proto_ppl = best_method(
        aggregate,
        "eval_perplexity",
        lower_is_better=True,
        exclude_baseline=True,
    )
    metrics = [
        "final_window_nll",
        "final_window_accuracy",
        "eval_nll",
        "eval_accuracy",
        "eval_perplexity",
        "train_s",
    ]
    lines = [
        "# Prototype Meta-Reset Worker",
        "",
        (
            f"Steps: `{payload['config']['steps']}`. Seeds: "
            f"`{payload['config']['seeds']}`. Final window: "
            f"`{payload['config']['final_window']}`."
        ),
        "",
        "## Baseline Comparison",
        "",
        "| Metric | Tuned FFN baseline | Best prototype method | Diff favoring proto |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("final_window_nll", "eval_perplexity"):
        best_name, best_value = (
            best_proto_nll if metric == "final_window_nll" else best_proto_ppl
        )
        baseline_value = baseline[metric]["mean"]
        diff = baseline_value - best_value
        lines.append(
            f"| `{metric}` | {baseline_value:.6f} | "
            f"`{best_name}` {best_value:.6f} | {diff:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Method Metrics",
            "",
            "| Method | " + " | ".join(f"`{metric}`" for metric in metrics) + " |",
            "|---|" + "---:|" * len(metrics),
        ]
    )
    for method in sorted(aggregate):
        row = aggregate[method]
        cells = []
        for metric in metrics:
            if metric not in row:
                cells.append("")
            else:
                cells.append(f"{row[metric]['mean']:.6f} +/- {row[metric]['stderr']:.6f}")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Reset Diagnostics",
            "",
            "| Method | Reset rate | Init norm | Value-row norm | Active prototypes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(aggregate):
        row = aggregate[method]
        if "reset_rate" not in row:
            continue
        lines.append(
            f"| `{method}` | {row['reset_rate']['mean']:.6f} | "
            f"{row['final_init_norm']['mean']:.6f} | "
            f"{row['final_value_row_norm']['mean']:.6f} | "
            f"{row['final_window_active_prototypes']['mean']:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_variants(args: argparse.Namespace) -> list[VariantSpec]:
    """Expand CLI sweeps into variant specs."""
    policies = parse_csv(args.reset_policies)
    kinds = parse_csv(args.method_kinds)
    unknown_policies = sorted(set(policies) - set(RESET_POLICIES))
    unknown_kinds = sorted(set(kinds) - set(METHOD_KINDS))
    if unknown_policies:
        raise ValueError(f"unknown reset policies: {unknown_policies}")
    if unknown_kinds:
        raise ValueError(f"unknown method kinds: {unknown_kinds}")
    variants: list[VariantSpec] = []
    for kind in kinds:
        for policy in policies:
            for value_lr in args.proto_value_lrs:
                init_lrs = (
                    args.proto_init_lrs
                    if policy in {"learned_global", "success_ema", "success_copy"}
                    else (0.0,)
                )
                for init_lr in init_lrs:
                    variants.append(
                        VariantSpec(
                            kind=kind,
                            reset_policy=policy,
                            proto_value_lr=float(value_lr),
                            proto_init_lr=float(init_lr),
                        )
                    )
    return variants


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--proto-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--final-window", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--model-lr", type=float, default=0.03)
    parser.add_argument("--proto-value-lrs", type=float, nargs="+", default=(0.03,))
    parser.add_argument("--proto-init-lrs", type=float, nargs="+", default=(0.05,))
    parser.add_argument(
        "--reset-policies",
        default="zero,no_reset,learned_global,success_ema",
    )
    parser.add_argument("--method-kinds", default="pure,hybrid")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--proto-update-rate", type=float, default=0.3)
    parser.add_argument("--proto-novelty-threshold", type=float, default=0.08)
    parser.add_argument("--proto-bandwidth", type=float, default=0.01)
    parser.add_argument("--proto-adaptive-bandwidth", action="store_true")
    parser.add_argument("--proto-bandwidth-update-rate", type=float, default=0.1)
    parser.add_argument("--init-success-temperature", type=float, default=3.0)
    parser.add_argument("--copy-success-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("output/subagents/transformer_ffn/data/tinyshakespeare.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.steps = 96
        args.seeds = 1
        args.eval_steps = 64
        args.final_window = 48
        args.proto_count = min(args.proto_count, 32)
        args.d_model = min(args.d_model, 24)
        args.mlp_hidden = min(args.mlp_hidden, 48)
        args.reset_policies = "zero,no_reset,learned_global,success_ema,success_copy"
        args.proto_value_lrs = (args.proto_value_lrs[0],)
        args.proto_init_lrs = (args.proto_init_lrs[0],)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.d_model < 1 or args.mlp_hidden < 1 or args.proto_count < 1:
        raise ValueError("--d-model, --mlp-hidden, and --proto-count must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if args.baseline_lr < 0.0 or args.model_lr < 0.0:
        raise ValueError("model learning rates must be non-negative")
    if any(value < 0.0 for value in args.proto_value_lrs):
        raise ValueError("--proto-value-lrs must be non-negative")
    if any(value < 0.0 for value in args.proto_init_lrs):
        raise ValueError("--proto-init-lrs must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 < args.proto_update_rate <= 1.0:
        raise ValueError("--proto-update-rate must be in (0, 1]")
    if args.proto_novelty_threshold < 0.0:
        raise ValueError("--proto-novelty-threshold must be non-negative")
    if args.proto_bandwidth <= 0.0:
        raise ValueError("--proto-bandwidth must be positive")
    if not 0.0 <= args.proto_bandwidth_update_rate <= 1.0:
        raise ValueError("--proto-bandwidth-update-rate must be in [0, 1]")
    if args.init_success_temperature <= 0.0:
        raise ValueError("--init-success-temperature must be positive")
    if not 0.0 <= args.copy_success_threshold <= 1.0:
        raise ValueError("--copy-success-threshold must be in [0, 1]")
    make_variants(args)


def profile_methods(
    args: argparse.Namespace,
    proto_block: PrototypeBasisBlock,
    vocab_size: int,
    key: jax.Array,
) -> dict[str, dict[str, int]]:
    """Return parameter/state profiles."""
    baseline_params = init_transformer_params(
        key,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    pure_params, pure_state = init_proto_transformer(
        key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
    )
    pure_params = add_proto_init(pure_params, args.d_model)
    hybrid_params, hybrid_state = init_hybrid_transformer(
        key,
        block=proto_block,
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    hybrid_params = add_proto_init(hybrid_params, args.d_model)
    return {
        "baseline_ffn_tuned": {
            "trainable_params": count_array_elements(baseline_params),
            "trainable_bytes": count_array_bytes(baseline_params),
            "state_elements": 0,
            "state_bytes": 0,
        },
        "pure_proto": {
            "trainable_params": count_array_elements(pure_params),
            "trainable_bytes": count_array_bytes(pure_params),
            "state_elements": count_array_elements(pure_state, include_int=True),
            "state_bytes": count_array_bytes(pure_state, include_int=True),
        },
        "hybrid_proto": {
            "trainable_params": count_array_elements(hybrid_params),
            "trainable_bytes": count_array_bytes(hybrid_params),
            "state_elements": count_array_elements(hybrid_state, include_int=True),
            "state_bytes": count_array_bytes(hybrid_state, include_int=True),
        },
    }


def main() -> None:
    """Run the meta-reset sweep."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = int(metadata["vocab_size"])
    final_window = args.final_window if args.final_window > 0 else args.eval_steps
    proto_block = make_proto_block(args)
    variants = make_variants(args)

    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        proto_count=args.proto_count,
        eval_steps=args.eval_steps,
        final_window=final_window,
        train_fraction=args.train_fraction,
        baseline_lr=args.baseline_lr,
        model_lr=args.model_lr,
        proto_value_lrs=tuple(float(value) for value in args.proto_value_lrs),
        proto_init_lrs=tuple(float(value) for value in args.proto_init_lrs),
        reset_policies=parse_csv(args.reset_policies),
        method_kinds=parse_csv(args.method_kinds),
        grad_clip=args.grad_clip,
        proto_update_rate=args.proto_update_rate,
        proto_novelty_threshold=args.proto_novelty_threshold,
        proto_bandwidth=args.proto_bandwidth,
        proto_adaptive_bandwidth=args.proto_adaptive_bandwidth,
        proto_bandwidth_update_rate=args.proto_bandwidth_update_rate,
        init_success_temperature=args.init_success_temperature,
        copy_success_threshold=args.copy_success_threshold,
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )

    root = jr.key(args.seed)
    profiles = profile_methods(args, proto_block, vocab_size, jr.fold_in(root, 999))
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for seed_idx in range(args.seeds):
        run_key = jr.fold_in(root, seed_idx)
        param_key, offset_key = jr.split(run_key, 2)
        max_offset = max(1, int(train_tokens.shape[0]) - args.block_size - args.steps - 1)
        train_offset = int(jr.randint(offset_key, (), 0, max_offset))
        contexts, labels = make_examples(
            train_tokens,
            steps=args.steps,
            block_size=args.block_size,
            offset=train_offset,
        )
        eval_contexts, eval_labels = make_examples(
            eval_tokens,
            steps=args.eval_steps,
            block_size=args.block_size,
            offset=seed_idx * args.eval_steps,
        )

        baseline_params = init_transformer_params(
            param_key,
            vocab_size=vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            ffn_hidden=args.mlp_hidden,
        )
        method_start = time.perf_counter()
        final_baseline, baseline_metrics = run_baseline_transformer(
            baseline_params,
            contexts,
            labels,
            step_size=args.baseline_lr,
            grad_clip=args.grad_clip,
        )
        baseline_train_s = time.perf_counter() - method_start
        baseline_summary = {
            **summarize_online(baseline_metrics, final_window),
            **eval_transformer(final_baseline, eval_contexts, eval_labels),
            "train_s": baseline_train_s,
            "train_steps_per_s": args.steps / baseline_train_s,
        }
        records.append(
            {
                "seed": seed_idx,
                "method": "baseline_ffn_tuned",
                "kind": "baseline",
                "reset_policy": "none",
                "proto_value_lr": 0.0,
                "proto_init_lr": 0.0,
                "summary": baseline_summary,
            }
        )
        print(
            f"seed={seed_idx} baseline_ffn_tuned: "
            f"fw_nll={baseline_summary['final_window_nll']:.4f}, "
            f"eval_ppl={baseline_summary['eval_perplexity']:.2f}, "
            f"train_s={baseline_train_s:.2f}"
        )

        for variant_idx, variant in enumerate(variants):
            variant_key = jr.fold_in(param_key, 10_000 + variant_idx)
            if variant.kind == "pure":
                params, state = init_proto_transformer(
                    variant_key,
                    block=proto_block,
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    d_model=args.d_model,
                )
            else:
                params, state = init_hybrid_transformer(
                    variant_key,
                    block=proto_block,
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    d_model=args.d_model,
                    ffn_hidden=args.mlp_hidden,
                )
            params = add_proto_init(params, args.d_model)
            method_start = time.perf_counter()
            final_params, final_state, metrics = run_meta_proto_transformer(
                proto_block,
                params,
                state,
                contexts,
                labels,
                kind=variant.kind,
                model_lr=args.model_lr,
                proto_value_lr=variant.proto_value_lr,
                proto_init_lr=variant.proto_init_lr,
                grad_clip=args.grad_clip,
                reset_policy=variant.reset_policy,
                init_success_temperature=args.init_success_temperature,
                copy_success_threshold=args.copy_success_threshold,
            )
            final_state.step_count.block_until_ready()
            train_s = time.perf_counter() - method_start
            summary = {
                **summarize_online(metrics, final_window),
                **summarize_proto_diagnostics(metrics, final_window),
                **summarize_meta_diagnostics(metrics, final_window),
                **eval_meta_proto_transformer(
                    proto_block,
                    final_params,
                    final_state,
                    eval_contexts,
                    eval_labels,
                    kind=variant.kind,
                ),
                "train_s": train_s,
                "train_steps_per_s": args.steps / train_s,
            }
            records.append(
                {
                    "seed": seed_idx,
                    "method": variant.method,
                    "kind": variant.kind,
                    "reset_policy": variant.reset_policy,
                    "proto_value_lr": variant.proto_value_lr,
                    "proto_init_lr": variant.proto_init_lr,
                    "summary": summary,
                }
            )
            print(
                f"seed={seed_idx} {variant.method}: "
                f"fw_nll={summary['final_window_nll']:.4f}, "
                f"eval_ppl={summary['eval_perplexity']:.2f}, "
                f"reset={summary['final_window_reset_rate']:.3f}, "
                f"init_norm={summary['final_init_norm']:.4f}, "
                f"train_s={train_s:.2f}"
            )

    aggregate = aggregate_records(records)
    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "prototype_block": proto_block.to_config(),
        "profiles": profiles,
        "variants": [asdict(variant) | {"method": variant.method} for variant in variants],
        "records": records,
        "aggregate": aggregate,
        "elapsed_s": time.perf_counter() - start,
        "note": (
            "Prototype value rows use decoupled learning rates. Meta-reset "
            "initializers are updated causally after observing the current "
            "online loss and before any novel row is used on future examples."
        ),
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    write_summary(args.doc_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.doc_path}")


if __name__ == "__main__":
    main()

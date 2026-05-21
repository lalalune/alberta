#!/usr/bin/env python3
# mypy: disable-error-code="untyped-decorator"
"""Validation-only stronger FFN baselines for Step 2 transformer memory.

This runner does not run memory methods and never accesses lockbox. It uses the
materialized validation corpus from the confirmatory protocol and evaluates
parameter-matched / wider FFN baselines under a frozen LR grid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

STEP2_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "The Alberta Plan" / "Step2"
)
sys.path.insert(0, str(STEP2_EXAMPLE_DIR))

from step2_tiny_shakespeare_upgd_ffn_transformer import (  # type: ignore[import-not-found] # noqa: E402
    count_array_bytes,
    count_array_elements,
    cross_entropy_from_logits,
    encode_text,
    ensure_tiny_shakespeare,
    init_transformer_params,
    make_examples,
    run_baseline_transformer,
    summarize_online,
    transformer_logits,
)

DEFAULT_DATA_PATH = Path(
    "outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/"
    "data/tinyshakespeare_confirmatory_validation.txt"
)
DEFAULT_MEMORY_RESULT = Path(
    "outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/"
    "validation_10000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/"
    "results.json"
)


@dataclass(frozen=True)
class Config:
    """Serializable run config."""

    steps: int
    seeds: int
    block_size: int
    d_model: int
    hidden_sizes: list[int]
    learning_rates: list[float]
    eval_steps: int
    eval_batch_size: int
    final_window: int
    train_fraction: float
    grad_clip: float
    seed: int
    data_path: str
    memory_result_path: str | None
    output_dir: str


def eval_transformer_batched(
    params: dict[str, Any],
    contexts: jax.Array,
    labels: jax.Array,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate heldout contexts in fixed-size batches."""
    total = int(labels.shape[0])
    if total <= 0:
        raise ValueError("eval requires at least one context")
    effective_batch = total if batch_size <= 0 else min(total, batch_size)

    @jax.jit
    def run_batch(
        batch_contexts: jax.Array,
        batch_labels: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = jax.vmap(lambda ctx: transformer_logits(params, ctx))(batch_contexts)
        losses = jax.vmap(cross_entropy_from_logits)(logits, batch_labels)
        acc = jnp.argmax(logits, axis=1) == batch_labels
        return jnp.sum(losses), jnp.sum(acc.astype(jnp.float32))

    loss_sum = 0.0
    acc_sum = 0.0
    for start in range(0, total, effective_batch):
        stop = min(start + effective_batch, total)
        batch_loss, batch_acc = run_batch(contexts[start:stop], labels[start:stop])
        batch_loss.block_until_ready()
        loss_sum += float(batch_loss)
        acc_sum += float(batch_acc)
    mean_loss = loss_sum / total
    return {
        "eval_nll": mean_loss,
        "eval_accuracy": acc_sum / total,
        "eval_perplexity": float(jnp.exp(jnp.minimum(jnp.asarray(mean_loss), 20.0))),
    }


def load_memory_reference(path: Path | None) -> dict[str, Any] | None:
    """Load memory result artifact if present."""
    if path is None:
        return None
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {path}")
    return parsed


def memory_reference_means(payload: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """Return per-method mean reference metrics."""
    if payload is None:
        return {}
    grouped: dict[str, list[dict[str, float]]] = {}
    for record in payload["records"]:
        method = str(record["method"])
        grouped.setdefault(method, []).append(
            {key: float(value) for key, value in record["summary"].items()}
        )
    return {
        method: {
            "final_window_nll": float(np.mean([row["final_window_nll"] for row in rows])),
            "eval_nll": float(np.mean([row["eval_nll"] for row in rows])),
            "eval_perplexity": float(np.mean([row["eval_perplexity"] for row in rows])),
        }
        for method, rows in grouped.items()
    }


def mean_stderr(values: list[float]) -> tuple[float, float]:
    """Return mean and standard error."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0
    return mean, float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate records by method."""
    methods = sorted({str(record["method"]) for record in records})
    rows: list[dict[str, Any]] = []
    for method in methods:
        subset = [record for record in records if record["method"] == method]
        row: dict[str, Any] = {"method": method, "n": len(subset)}
        for metric in (
            "final_window_nll",
            "final_window_accuracy",
            "eval_nll",
            "eval_accuracy",
            "eval_perplexity",
            "train_s",
            "train_steps_per_s",
        ):
            mean, err = mean_stderr([float(record["summary"][metric]) for record in subset])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_stderr"] = err
        rows.append(row)
    return rows


def write_summary(
    path: Path,
    *,
    config: Config,
    aggregate_rows: list[dict[str, Any]],
    memory_means: dict[str, dict[str, float]],
) -> None:
    """Write Markdown summary."""
    lines = [
        "# Stronger FFN Validation Baselines",
        "",
        "Validation split only. No lockbox access.",
        "",
        f"Steps: `{config.steps}`. Seeds: `{config.seeds}`. "
        f"Eval contexts: `{config.eval_steps}`.",
        "",
        "| Method | Final NLL | Eval NLL | Eval PPL | Train steps/s | "
        "Eval diff vs static memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    static_eval = memory_means.get("advantage_post_ffn_memory", {}).get("eval_nll")
    for row in aggregate_rows:
        eval_diff = ""
        if static_eval is not None:
            eval_diff = f"{static_eval - float(row['eval_nll_mean']):+.6f}"
        lines.append(
            f"| `{row['method']}` | "
            f"{row['final_window_nll_mean']:.4f} +/- {row['final_window_nll_stderr']:.4f} | "
            f"{row['eval_nll_mean']:.4f} +/- {row['eval_nll_stderr']:.4f} | "
            f"{row['eval_perplexity_mean']:.2f} +/- {row['eval_perplexity_stderr']:.2f} | "
            f"{row['train_steps_per_s_mean']:.1f} +/- "
            f"{row['train_steps_per_s_stderr']:.1f} | {eval_diff} |"
        )
    if memory_means:
        lines.extend(["", "## Memory Reference Means", ""])
        lines.append("| Method | Final NLL | Eval NLL | Eval PPL |")
        lines.append("|---|---:|---:|---:|")
        for method, metrics in sorted(memory_means.items()):
            lines.append(
                f"| `{method}` | {metrics['final_window_nll']:.4f} | "
                f"{metrics['eval_nll']:.4f} | {metrics['eval_perplexity']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=(96, 128))
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20, 0.30),
    )
    parser.add_argument("--eval-steps", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--final-window", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.84210526315789469)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--memory-result-path", type=Path, default=DEFAULT_MEMORY_RESULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/stronger_ffn_validation_baselines"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI args."""
    if min(args.steps, args.seeds, args.eval_steps, args.final_window) <= 0:
        raise ValueError("--steps, --seeds, --eval-steps, and --final-window must be positive")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be non-negative")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")
    if any(hidden <= 0 for hidden in args.hidden_sizes):
        raise ValueError("--hidden-sizes must be positive")
    if any(lr <= 0.0 for lr in args.learning_rates):
        raise ValueError("--learning-rates must be positive")


def main() -> None:
    """Run validation-only stronger FFN baselines."""
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_tiny_shakespeare(args.data_path)
    tokens, metadata = encode_text(text)
    split = int(tokens.shape[0] * args.train_fraction)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    vocab_size = int(metadata["vocab_size"])
    root = jr.key(args.seed)
    records: list[dict[str, Any]] = []

    for hidden in args.hidden_sizes:
        for lr in args.learning_rates:
            method = f"ffn_h{hidden}_lr{str(lr).replace('.', 'p')}"
            for seed_idx in range(args.seeds):
                run_key = jr.fold_in(root, seed_idx)
                param_key, offset_key = jr.split(run_key, 2)
                max_offset = max(
                    1,
                    int(train_tokens.shape[0]) - args.block_size - args.steps - 1,
                )
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
                params = init_transformer_params(
                    param_key,
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    d_model=args.d_model,
                    ffn_hidden=hidden,
                )
                start = time.perf_counter()
                final_params, metrics = run_baseline_transformer(
                    params,
                    contexts,
                    labels,
                    step_size=lr,
                    grad_clip=args.grad_clip,
                )
                train_s = time.perf_counter() - start
                summary = {
                    **summarize_online(metrics, args.final_window),
                    **eval_transformer_batched(
                        final_params,
                        eval_contexts,
                        eval_labels,
                        batch_size=args.eval_batch_size,
                    ),
                    "train_s": train_s,
                    "train_steps_per_s": args.steps / train_s,
                }
                records.append(
                    {
                        "seed": seed_idx,
                        "method": method,
                        "hidden_size": hidden,
                        "learning_rate": lr,
                        "train_offset": train_offset,
                        "eval_offset": seed_idx * args.eval_steps,
                        "summary": summary,
                    }
                )
                print(
                    f"{method} seed={seed_idx}: fw_nll={summary['final_window_nll']:.4f} "
                    f"eval_nll={summary['eval_nll']:.4f} train_s={train_s:.2f}"
                )

    memory_payload = load_memory_reference(args.memory_result_path)
    memory_means = memory_reference_means(memory_payload)
    aggregate_rows = aggregate(records)
    config = Config(
        steps=args.steps,
        seeds=args.seeds,
        block_size=args.block_size,
        d_model=args.d_model,
        hidden_sizes=list(args.hidden_sizes),
        learning_rates=list(args.learning_rates),
        eval_steps=args.eval_steps,
        eval_batch_size=args.eval_batch_size,
        final_window=args.final_window,
        train_fraction=args.train_fraction,
        grad_clip=args.grad_clip,
        seed=args.seed,
        data_path=str(args.data_path),
        memory_result_path=str(args.memory_result_path)
        if args.memory_result_path is not None
        else None,
        output_dir=str(args.output_dir),
    )
    profiles = {
        f"ffn_h{hidden}": {
            "trainable_params": count_array_elements(
                init_transformer_params(
                    jr.fold_in(root, hidden),
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    d_model=args.d_model,
                    ffn_hidden=hidden,
                )
            ),
            "trainable_bytes": count_array_bytes(
                init_transformer_params(
                    jr.fold_in(root, hidden),
                    vocab_size=vocab_size,
                    block_size=args.block_size,
                    d_model=args.d_model,
                    ffn_hidden=hidden,
                )
            ),
            "state_bytes": 0,
        }
        for hidden in args.hidden_sizes
    }
    payload = {
        "config": asdict(config),
        "vocab_size": vocab_size,
        "profiles": profiles,
        "records": records,
        "aggregate": aggregate_rows,
        "memory_reference_means": memory_means,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_summary(
        args.output_dir / "SUMMARY.md",
        config=config,
        aggregate_rows=aggregate_rows,
        memory_means=memory_means,
    )
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()

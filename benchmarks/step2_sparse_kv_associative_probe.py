#!/usr/bin/env python3
# mypy: disable-error-code="import-not-found"
"""Focused sparse key/value recall probe for Step 2 memory mechanisms.

The external transformer-memory rescue runs show that the residual prototype
block can close its prediction gate, but it still does not behave like an
addressable key/value memory on ``sparse_kv_recall``.  This script tests that
diagnosis directly with a small online associative learner:

* every prediction is made before the current example is written;
* feature weights are learned online from feature-level advantage;
* bindings are overwritten by recency-biased exponential updates;
* token-only, suffix-pair, and full-position-pair feature families are ablated.

This is not a promoted Step 2 learner.  It is a mechanism probe to determine
whether the unsolved external failure is due to the benchmark itself or to the
current prototype-residual parameterization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP2_EXAMPLE_DIR = REPO_ROOT / "examples" / "The Alberta Plan" / "Step2"
sys.path.insert(0, str(STEP2_EXAMPLE_DIR))

from step2_sequence_external_memory_benchmarks import (  # noqa: E402
    ALL_BENCHMARKS,
    make_benchmark,
)
from step2_tiny_shakespeare_upgd_ffn_transformer import (  # noqa: E402
    eval_transformer,
    init_transformer_params,
    run_baseline_transformer,
    stderr,
    summarize_online,
)

FeatureFamily = Literal[
    "position_token",
    "suffix_pair",
    "token_suffix_pair",
    "position_pair",
]


@dataclass(frozen=True)
class VariantSpec:
    """One associative-memory ablation."""

    name: str
    feature_family: FeatureFamily
    suffix_length: int
    write_lr: float
    retention: float
    utility_lr: float
    utility_decay: float
    min_weight: float
    max_weight: float
    logit_scale: float
    normalize_by_weight: bool
    feature_budget: int
    evict_interval: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration written into result artifacts."""

    steps: int
    seeds: int
    benchmarks: list[str]
    eval_steps: int
    block_size: int
    final_window: int
    d_model: int
    mlp_hidden: int
    baseline_lr: float
    grad_clip: float
    run_ffn: bool
    seed: int
    output_dir: str
    variants: list[dict[str, Any]]


class OnlineAssociativeMemory:
    """Sparse online feature-to-label memory with learned feature utility."""

    def __init__(self, *, vocab_size: int, variant: VariantSpec) -> None:
        self.vocab_size = vocab_size
        self.variant = variant
        self.scores: dict[tuple[int, ...], dict[int, float]] = {}
        self.utility: dict[tuple[int, ...], float] = {}
        self.prior: np.ndarray = np.zeros(vocab_size, dtype=np.float64)
        self.step_count = 0
        self.evictions = 0
        self.feature_touches = 0

    def features(self, context: np.ndarray) -> list[tuple[int, ...]]:
        """Return active sparse features for a token context."""
        tokens = [int(token) for token in context.tolist()]
        token_features: list[tuple[int, ...]] = [
            (0, pos, token) for pos, token in enumerate(tokens)
        ]
        if self.variant.feature_family == "position_token":
            return token_features

        if self.variant.feature_family in {"suffix_pair", "token_suffix_pair"}:
            start = max(0, len(tokens) - self.variant.suffix_length)
            suffix = tokens[start:]
            features: list[tuple[int, ...]] = []
            for left in range(len(suffix)):
                for right in range(left + 1, len(suffix)):
                    features.append((1, left, right, suffix[left], suffix[right]))
            if self.variant.feature_family == "token_suffix_pair":
                return [*token_features, *features]
            return features

        features = []
        for left in range(len(tokens)):
            left_token = tokens[left]
            for right in range(left + 1, len(tokens)):
                features.append((2, left, right, left_token, tokens[right]))
        return features

    @staticmethod
    def _cross_entropy(logits: np.ndarray, label: int) -> float:
        shifted = logits - float(np.max(logits))
        log_z = float(np.log(np.sum(np.exp(shifted))) + np.max(logits))
        return log_z - float(logits[label])

    def _weight(self, feature: tuple[int, ...]) -> float:
        raw = self.utility.get(feature, 0.0)
        clipped = max(-8.0, min(8.0, raw))
        return max(
            self.variant.min_weight,
            min(self.variant.max_weight, math.exp(clipped)),
        )

    def _feature_loss(self, scores: dict[int, float], label: int) -> float:
        if not scores:
            return math.log(self.vocab_size)
        max_score = max(0.0, max(scores.values()))
        active_sum = sum(math.exp(value - max_score) for value in scores.values())
        inactive = self.vocab_size - len(scores)
        z = active_sum + inactive * math.exp(-max_score)
        label_score = scores.get(label, 0.0)
        return math.log(z) + max_score - label_score

    def predict_logits(
        self,
        context: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, ...]], int]:
        """Predict logits before observing the current label."""
        features = self.features(context)
        logits = 0.05 * self.prior.copy()
        active_feature_count = 0
        total_weight = 0.0
        for feature in features:
            scores = self.scores.get(feature)
            if not scores:
                continue
            weight = self._weight(feature)
            active_feature_count += 1
            total_weight += weight
            for label, score in scores.items():
                logits[label] += self.variant.logit_scale * weight * score
        if self.variant.normalize_by_weight and total_weight > 0.0:
            logits /= total_weight
        return logits, features, active_feature_count

    def update(self, context: np.ndarray, label: int) -> dict[str, float]:
        """Predict, then update all active feature rows."""
        logits, features, active_feature_count = self.predict_logits(context)
        loss = self._cross_entropy(logits, label)
        accuracy = float(int(np.argmax(logits) == label))
        self.prior *= self.variant.retention
        self.prior[label] += self.variant.write_lr
        for feature in features:
            row = self.scores.setdefault(feature, {})
            feature_loss = self._feature_loss(row, label)
            utility = self.utility.get(feature, 0.0)
            utility = (
                self.variant.utility_decay * utility
                + self.variant.utility_lr * (loss - feature_loss)
            )
            self.utility[feature] = max(-8.0, min(8.0, utility))
            for key in list(row):
                row[key] *= self.variant.retention
                if abs(row[key]) < 1e-7:
                    del row[key]
            row[label] = row.get(label, 0.0) + self.variant.write_lr
        self.step_count += 1
        self.feature_touches += len(features)
        self._evict_if_needed()
        return {
            "loss": loss,
            "accuracy": accuracy,
            "active_feature_count": float(active_feature_count),
            "feature_count": float(len(self.scores)),
            "mean_weight": self._mean_weight(features),
            "evictions": float(self.evictions),
        }

    def _mean_weight(self, features: list[tuple[int, ...]]) -> float:
        if not features:
            return 0.0
        return float(np.mean([self._weight(feature) for feature in features]))

    def _evict_if_needed(self) -> None:
        budget = self.variant.feature_budget
        if budget <= 0 or len(self.scores) <= budget:
            return
        if self.step_count % max(1, self.variant.evict_interval) != 0:
            return
        overflow = len(self.scores) - budget
        ranked = sorted(
            self.scores,
            key=lambda feature: self.utility.get(feature, 0.0),
        )
        for feature in ranked[:overflow]:
            self.scores.pop(feature, None)
            self.utility.pop(feature, None)
        self.evictions += overflow

    def evaluate(
        self,
        contexts: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate without updating memory."""
        losses = []
        accuracies = []
        active_counts = []
        for context, label in zip(contexts, labels, strict=True):
            logits, _, active = self.predict_logits(context)
            label_int = int(label)
            losses.append(self._cross_entropy(logits, label_int))
            accuracies.append(float(int(np.argmax(logits) == label_int)))
            active_counts.append(float(active))
        mean_loss = float(np.mean(losses))
        return {
            "eval_nll": mean_loss,
            "eval_accuracy": float(np.mean(accuracies)),
            "eval_perplexity": float(math.exp(min(mean_loss, 50.0))),
            "eval_active_feature_count": float(np.mean(active_counts)),
        }


def default_variants(args: argparse.Namespace) -> list[VariantSpec]:
    """Return the default ablation set."""
    variants = [
        VariantSpec(
            name="token_utility",
            feature_family="position_token",
            suffix_length=args.suffix_length,
            write_lr=0.8,
            retention=0.90,
            utility_lr=0.08,
            utility_decay=0.995,
            min_weight=0.05,
            max_weight=4.0,
            logit_scale=4.0,
            normalize_by_weight=True,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="suffix_pair_utility_sum",
            feature_family="suffix_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.08,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=6.0,
            logit_scale=1.0,
            normalize_by_weight=False,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="suffix_pair_utility_norm4",
            feature_family="suffix_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.08,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=6.0,
            logit_scale=4.0,
            normalize_by_weight=True,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="suffix_pair_utility_norm8",
            feature_family="suffix_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.08,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=6.0,
            logit_scale=8.0,
            normalize_by_weight=True,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="hybrid_token_suffix_norm4",
            feature_family="token_suffix_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.10,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=8.0,
            logit_scale=4.0,
            normalize_by_weight=True,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="hybrid_token_suffix_norm8",
            feature_family="token_suffix_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.10,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=8.0,
            logit_scale=8.0,
            normalize_by_weight=True,
            feature_budget=0,
            evict_interval=100,
        ),
        VariantSpec(
            name="full_pair_utility_norm4",
            feature_family="position_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.08,
            utility_decay=0.995,
            min_weight=0.02,
            max_weight=6.0,
            logit_scale=4.0,
            normalize_by_weight=True,
            feature_budget=args.full_pair_budget,
            evict_interval=100,
        ),
        VariantSpec(
            name="full_pair_selective_norm16",
            feature_family="position_pair",
            suffix_length=args.suffix_length,
            write_lr=1.0,
            retention=0.80,
            utility_lr=0.20,
            utility_decay=0.99,
            min_weight=0.001,
            max_weight=64.0,
            logit_scale=16.0,
            normalize_by_weight=True,
            feature_budget=args.full_pair_budget,
            evict_interval=100,
        ),
    ]
    if "all" in args.variants:
        return variants
    requested = set(args.variants)
    known = {variant.name for variant in variants}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {sorted(known)}")
    return [variant for variant in variants if variant.name in requested]


def summarize_train(metrics: list[dict[str, float]], final_window: int) -> dict[str, float]:
    """Summarize prequential online metrics."""
    window = metrics[-min(final_window, len(metrics)) :]
    return {
        "final_window_nll": float(np.mean([row["loss"] for row in window])),
        "final_window_accuracy": float(np.mean([row["accuracy"] for row in window])),
        "final_window_active_feature_count": float(
            np.mean([row["active_feature_count"] for row in window]),
        ),
        "final_window_feature_count": float(np.mean([row["feature_count"] for row in window])),
        "final_window_mean_weight": float(np.mean([row["mean_weight"] for row in window])),
        "evictions": float(metrics[-1]["evictions"]),
    }


def run_associative_seed(
    *,
    benchmark: Any,
    variant: VariantSpec,
    final_window: int,
) -> dict[str, Any]:
    """Run one associative-memory variant on one benchmark seed."""
    train_contexts = np.asarray(benchmark.train_contexts)
    train_labels = np.asarray(benchmark.train_labels)
    eval_contexts = np.asarray(benchmark.eval_contexts)
    eval_labels = np.asarray(benchmark.eval_labels)
    learner = OnlineAssociativeMemory(vocab_size=benchmark.vocab_size, variant=variant)
    start = time.perf_counter()
    metrics = [
        learner.update(context, int(label))
        for context, label in zip(train_contexts, train_labels, strict=True)
    ]
    train_s = time.perf_counter() - start
    summary = {
        **summarize_train(metrics, final_window),
        **learner.evaluate(eval_contexts, eval_labels),
        "train_s": train_s,
        "train_steps_per_s": float(len(train_labels) / train_s),
        "feature_touches_per_step": float(learner.feature_touches / len(train_labels)),
        "final_feature_rows": float(len(learner.scores)),
    }
    return {
        "method": variant.name,
        "summary": summary,
    }


def run_ffn_seed(
    *,
    benchmark: Any,
    args: argparse.Namespace,
    seed_idx: int,
) -> dict[str, Any]:
    """Run the FFN transformer baseline on one benchmark seed."""
    key = jr.fold_in(jr.key(args.seed + 50_003), seed_idx)
    params = init_transformer_params(
        key,
        vocab_size=benchmark.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        ffn_hidden=args.mlp_hidden,
    )
    start = time.perf_counter()
    final_params, metrics = run_baseline_transformer(
        params,
        benchmark.train_contexts,
        benchmark.train_labels,
        step_size=args.baseline_lr,
        grad_clip=args.grad_clip,
    )
    if hasattr(metrics, "block_until_ready"):
        metrics.block_until_ready()
    train_s = time.perf_counter() - start
    summary = {
        **summarize_online(metrics, args.final_window),
        **eval_transformer(final_params, benchmark.eval_contexts, benchmark.eval_labels),
        "train_s": train_s,
        "train_steps_per_s": float(args.steps / train_s),
    }
    return {
        "method": "baseline_ffn_transformer",
        "summary": summary,
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate records by benchmark and method."""
    groups = sorted({(row["benchmark"], row["method"]) for row in records})
    metrics = sorted({key for row in records for key in row["summary"]})
    rows = []
    for benchmark, method in groups:
        subset = [
            row
            for row in records
            if row["benchmark"] == benchmark and row["method"] == method
        ]
        out: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "n": len(subset),
        }
        for metric in metrics:
            values = [
                float(row["summary"][metric])
                for row in subset
                if metric in row["summary"]
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(array))
            out[f"{metric}_stderr"] = float(stderr(array))
        rows.append(out)
    return rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown summary."""
    aggregate_rows = payload["aggregate"]
    lines = [
        "# Sparse KV Associative Memory Probe",
        "",
        "This is a mechanism probe for the unresolved `sparse_kv_recall` failure. "
        "It tests whether online associative binding with learned feature utility "
        "can solve the same generated stream that defeated the residual prototype "
        "memory rescue variants.",
        "",
        f"Benchmarks: {', '.join(f'`{name}`' for name in payload['config']['benchmarks'])}.",
        f"Steps: `{payload['config']['steps']}`. Seeds: `{payload['config']['seeds']}`. "
        f"Eval steps: `{payload['config']['eval_steps']}`.",
        "",
        "## Aggregate",
        "",
        "| Benchmark | Method | Eval NLL | Eval accuracy | Final-window NLL | "
        "Final-window accuracy | Steps/s | Feature touches/step | Final rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| `{row['benchmark']}` | `{row['method']}` | "
            f"{row.get('eval_nll_mean', float('nan')):.4f} +/- "
            f"{row.get('eval_nll_stderr', 0.0):.4f} | "
            f"{row.get('eval_accuracy_mean', float('nan')):.4f} +/- "
            f"{row.get('eval_accuracy_stderr', 0.0):.4f} | "
            f"{row.get('final_window_nll_mean', float('nan')):.4f} +/- "
            f"{row.get('final_window_nll_stderr', 0.0):.4f} | "
            f"{row.get('final_window_accuracy_mean', float('nan')):.4f} +/- "
            f"{row.get('final_window_accuracy_stderr', 0.0):.4f} | "
            f"{row.get('train_steps_per_s_mean', float('nan')):.1f} | "
            f"{row.get('feature_touches_per_step_mean', float('nan')):.1f} | "
            f"{row.get('final_feature_rows_mean', float('nan')):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The important comparison is not whether this probe is production-ready. "
            "It is whether a recency-biased key/value table with learned feature "
            "utility beats the FFN and residual prototype memories on the failure "
            "case. If suffix-pair succeeds while token-only fails, the next Step 2 "
            "candidate should expose an explicit associative binding/read path "
            "instead of asking a dense residual prototype value row to discover it.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--benchmarks", nargs="+", default=["sparse_kv_recall"])
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--final-window", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--baseline-lr", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--suffix-length", type=int, default=8)
    parser.add_argument("--full-pair-budget", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help="Variant names to run, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ffn", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/step2_new_directions/sparse_kv_associative_probe"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.seeds <= 0 or args.eval_steps <= 0:
        raise ValueError("--steps, --seeds, and --eval-steps must be positive")
    if args.block_size < 8:
        raise ValueError("--block-size must be at least 8")
    if args.final_window <= 0:
        args.final_window = args.eval_steps
    if "all" in args.benchmarks:
        args.benchmarks = list(ALL_BENCHMARKS)
    unknown_benchmarks = sorted(set(args.benchmarks) - set(ALL_BENCHMARKS))
    if unknown_benchmarks:
        raise ValueError(
            f"unknown benchmarks {unknown_benchmarks}; choose from {list(ALL_BENCHMARKS)}",
        )
    return args


def main() -> None:
    """Run the probe."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = default_variants(args)
    records: list[dict[str, Any]] = []
    benchmark_metadata: list[dict[str, Any]] = []
    start = time.perf_counter()
    for benchmark_name in args.benchmarks:
        first_benchmark = make_benchmark(
            benchmark_name,
            seed=args.seed,
            steps=args.steps,
            eval_steps=args.eval_steps,
            block_size=args.block_size,
        )
        benchmark_metadata.append(
            {
                "name": first_benchmark.name,
                "description": first_benchmark.description,
                "vocab_size": first_benchmark.vocab_size,
                "metadata": first_benchmark.metadata,
            },
        )
        for seed_idx in range(args.seeds):
            benchmark = make_benchmark(
                benchmark_name,
                seed=args.seed + 1009 * seed_idx,
                steps=args.steps,
                eval_steps=args.eval_steps,
                block_size=args.block_size,
            )
            if not args.no_ffn:
                ffn_record = run_ffn_seed(
                    benchmark=benchmark,
                    args=args,
                    seed_idx=seed_idx,
                )
                ffn_record["benchmark"] = benchmark.name
                ffn_record["seed"] = seed_idx
                records.append(ffn_record)
                print(
                    f"{benchmark.name} seed={seed_idx} baseline_ffn_transformer "
                    f"eval_nll={ffn_record['summary']['eval_nll']:.4f} "
                    f"acc={ffn_record['summary']['eval_accuracy']:.4f}"
                )
            for variant in variants:
                record = run_associative_seed(
                    benchmark=benchmark,
                    variant=variant,
                    final_window=args.final_window,
                )
                record["benchmark"] = benchmark.name
                record["seed"] = seed_idx
                records.append(record)
                print(
                    f"{benchmark.name} seed={seed_idx} {variant.name} "
                    f"eval_nll={record['summary']['eval_nll']:.4f} "
                    f"acc={record['summary']['eval_accuracy']:.4f} "
                    f"steps_s={record['summary']['train_steps_per_s']:.1f}"
                )
    config = ExperimentConfig(
        steps=args.steps,
        seeds=args.seeds,
        benchmarks=list(args.benchmarks),
        eval_steps=args.eval_steps,
        block_size=args.block_size,
        final_window=args.final_window,
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        baseline_lr=args.baseline_lr,
        grad_clip=args.grad_clip,
        run_ffn=not args.no_ffn,
        seed=args.seed,
        output_dir=str(args.output_dir),
        variants=[asdict(variant) for variant in variants],
    )
    payload = {
        "config": asdict(config),
        "benchmark": benchmark_metadata,
        "records": records,
        "aggregate": aggregate(records),
        "elapsed_s": time.perf_counter() - start,
    }
    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "SUMMARY.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
